# =============================================================================
# cc_conditional_decode.py  -- M1 on-GPU conditional decode loop for SGLang.
#
# Captures ONE decode sub-step body inside a CUDA-graph WHILE conditional node so
# the autoregressive loop runs entirely on the GPU: the host issues ONE graph
# launch per chunk (not per token), and the loop exits on-GPU at EOS / chunk cap.
# This amortizes the per-token host<->device crossing cost that NVIDIA GPU
# Confidential Computing taxes with software AES-GCM.
#
# This is the proven machinery from condgraph-poc (while_loop_poc.py +
# torch_interop_alloc_poc.py + torch_predicate_poc.py + decode_predicate.cu),
# inlined here so SGLang has no import dependency on the PoC tree:
#   - cu / run_nvrtc / KernelArgs / with_conditional_node   (cuda-python WHILE node)
#   - decode_finish_update / decode_while_cond              (on-GPU loop control)
#
# CRITICAL LIFETIME GOTCHA (cost a long debug in the PoC): KernelArgs.build()
# returns a host address into numpy storage the KernelArgs object owns. The
# object MUST stay alive past graph capture AND every launch, or the captured
# graph gets dangling argument pointers (use-after-free -> nondeterministic).
# CondGraph below holds every KernelArgs object for the graph's lifetime.
# =============================================================================

from __future__ import annotations

import sys

import numpy as np

try:
    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
    from cuda.bindings import nvrtc

    _BINDINGS = "cuda.bindings"
except ImportError:  # pragma: no cover
    from cuda import cuda, cudart, nvrtc  # type: ignore

    _BINDINGS = "cuda (legacy)"

import importlib.metadata as _md


def _detect_version() -> str:
    for pkg in ("cuda-python", "cuda-bindings", "cuda_bindings"):
        try:
            return _md.version(pkg)
        except _md.PackageNotFoundError:
            continue
    return getattr(cuda, "__version__", "12.6.0")


def _ver_ge(ver: str, target: str) -> bool:
    def norm(v):
        out = []
        for part in v.split("."):
            num = "".join(ch for ch in part if ch.isdigit())
            out.append(int(num) if num else 0)
        return out

    a, b = norm(ver), norm(target)
    n = max(len(a), len(b))
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    return a >= b


_CUDA_PY_VERSION = _detect_version()
_IS_CUDA13 = _ver_ge(_CUDA_PY_VERSION, "13.0.0")


def _check(err) -> None:
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            _, msg = cudart.cudaGetErrorString(err)
            raise RuntimeError(f"CUDA runtime error: {err} ({msg})")
    elif isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            _, msg = cuda.cuGetErrorString(err)
            raise RuntimeError(f"CUDA driver error: {err} ({msg})")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err}")
    else:
        raise RuntimeError(f"Unknown CUDA return-code type: {type(err)} = {err!r}")


def cu(ret):
    if not isinstance(ret, (tuple, list)):
        ret = (ret,)
    _check(ret[0])
    rest = ret[1:]
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


class KernelArgs:
    """void** kernel-argument packing (cuLaunchKernel idiom). Keep ALIVE past
    capture+launch (see module header)."""

    def __init__(self):
        self._holders = []
        self._ptrs = []
        self._arr = None

    def add_ptr(self, device_ptr) -> "KernelArgs":
        slot = np.array([int(device_ptr)], dtype=np.uint64)
        self._holders.append(slot)
        self._ptrs.append(slot.ctypes.data)
        return self

    def add_raw(self, value_ptr) -> "KernelArgs":
        self._ptrs.append(int(value_ptr))
        return self

    def add_int(self, value, dtype=np.int32) -> "KernelArgs":
        slot = np.array([value], dtype=dtype)
        self._holders.append(slot)
        self._ptrs.append(slot.ctypes.data)
        return self

    def build(self):
        self._arr = np.array(self._ptrs, dtype=np.uint64)
        return self._arr.ctypes.data


def run_nvrtc(kernel_string: str, kernel_name: bytes, program_name: bytes):
    prog = cu(
        nvrtc.nvrtcCreateProgram(str.encode(kernel_string), program_name, 0, [], [])
    )
    opts = []
    try:
        _check(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0])
    except RuntimeError:
        log_size = cu(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * log_size
        cu(nvrtc.nvrtcGetProgramLog(prog, log))
        sys.stderr.write("NVRTC compile log:\n" + log.decode(errors="replace") + "\n")
        raise
    ptx_size = cu(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    cu(nvrtc.nvrtcGetPTX(prog, ptx))
    ptx = np.char.array(ptx)
    module = cu(cuda.cuModuleLoadData(ptx.ctypes.data))
    kernel = cu(cuda.cuModuleGetFunction(module, kernel_name))
    return kernel


_ALLOC_FREE_SCAN_DONE: set = set()


def _scan_child_graph_for_alloc_free(child_graph) -> None:
    """CCD_DEBUG: before embedding child_graph, check whether it (or any
    graph node it itself contains, recursively) has MemAlloc/MemFree/
    Conditional node types -- CUDA docs: cloning is not supported for
    graphs containing these. Best-effort, never raises, logs once per
    distinct graph handle."""
    import sys as _sys

    key = int(child_graph)
    if key in _ALLOC_FREE_SCAN_DONE:
        return
    _ALLOC_FREE_SCAN_DONE.add(key)

    def _walk(g, depth):
        try:
            err, _n0, n = cudart.cudaGraphGetNodes(g, 0)
            _check(err)
            err, nodes, n2 = cudart.cudaGraphGetNodes(g, n)
            _check(err)
        except Exception as e:  # noqa: BLE001
            _sys.stderr.write(
                f"CCD_DEBUG: alloc/free scan: cudaGraphGetNodes failed at depth {depth}: {e}\n"
            )
            return
        counts = {}
        for node in nodes[:n2] if n2 else []:
            try:
                ntype = cu(cudart.cudaGraphNodeGetType(node))
            except Exception as e:  # noqa: BLE001
                _sys.stderr.write(f"CCD_DEBUG: alloc/free scan: type lookup failed: {e}\n")
                continue
            counts[ntype] = counts.get(ntype, 0) + 1
            if ntype in (
                cudart.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc,
                cudart.cudaGraphNodeType.cudaGraphNodeTypeMemFree,
                cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional,
            ):
                _sys.stderr.write(
                    f"CCD_DEBUG: alloc/free scan: FOUND {ntype} at depth {depth} "
                    f"(graph={hex(int(g))})\n"
                )
            if ntype == cudart.cudaGraphNodeType.cudaGraphNodeTypeGraph:
                try:
                    err, sub = cudart.cudaGraphChildGraphNodeGetGraph(node)
                    _check(err)
                    _walk(sub, depth + 1)
                except Exception as e:  # noqa: BLE001
                    _sys.stderr.write(
                        f"CCD_DEBUG: alloc/free scan: child-graph descend failed: {e}\n"
                    )
        _sys.stderr.write(
            f"CCD_DEBUG: alloc/free scan depth={depth} graph={hex(int(g))} node_counts={counts}\n"
        )

    _sys.stderr.write(f"CCD_DEBUG: alloc/free scan starting for graph={hex(key)}\n")
    _walk(child_graph, 0)


def add_child_graph_node(capture_stream: int, child_graph: int):
    """Embed an already-captured (pre-instantiation) cudaGraph_t as a child
    graph node in whatever graph `capture_stream` is currently recording
    into. `child_graph` must come from `torch.cuda.CUDAGraph(keep_graph=True)`
    + `.raw_cuda_graph()` -- the pre-instantiation structure, NOT the exec
    handle from `.raw_cuda_graph_exec()` (that one is for `cudaGraphLaunch`,
    which the CUDA runtime itself rejects during an active capture).

    Mirrors with_conditional_node.__enter__'s node-adding pattern exactly
    (cudaStreamGetCaptureInfo -> cuGraphAddNode -> cudaStreamUpdateCaptureDependencies),
    just with CU_GRAPH_NODE_TYPE_GRAPH instead of ..._CONDITIONAL. CLONE
    ownership: copies the child graph's structure into the parent rather
    than consuming/invalidating the original `torch.cuda.CUDAGraph` object,
    which must stay independently replayable for requests that don't
    qualify for CC conditional decode.

    Validated in isolation (two dummy CUDAGraphs, repeated launches of the
    resulting conditional graph, exact expected accumulation both times) --
    see the Phase 0 checkpoint for this project. Not yet validated against
    graphs sharing a live memory pool with other captures (the sglang
    FullCudaGraphBackend case) -- that's the next real risk.
    """
    info = cu(cudart.cudaStreamGetCaptureInfo(capture_stream))
    capture_status, _id, graph, dependencies, *_rest = info
    assert (
        capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    ), "add_child_graph_node called outside an active stream capture"
    _scan_child_graph_for_alloc_free(child_graph)
    driver_params = cuda.CUgraphNodeParams()
    driver_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH
    driver_params.graph.graph = child_graph
    driver_params.graph.ownership = (
        cuda.CUgraphChildGraphNodeOwnership.CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE
    )
    if _IS_CUDA13:
        node = cu(
            cuda.cuGraphAddNode(
                graph, dependencies, None, len(dependencies), driver_params
            )
        )
        cu(
            cudart.cudaStreamUpdateCaptureDependencies(
                capture_stream, [node], None, 1,
                cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
            )
        )
    else:
        node = cu(
            cuda.cuGraphAddNode(graph, dependencies, len(dependencies), driver_params)
        )
        cu(
            cudart.cudaStreamUpdateCaptureDependencies(
                capture_stream, [node], 1,
                cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
            )
        )
    return node


class with_conditional_node:
    """Faithful copy of the NeMo/cuda-samples conditional-node context manager
    (see condgraph-poc/while_loop_poc.py). The capture stream must already be
    capturing. Adds a WHILE conditional node, captures the body into its body
    graph, and re-arms the predicate at the end of the body."""

    def __init__(self, while_kernel, while_args_ptr, handle, capture_stream, body_stream):
        self.while_kernel = while_kernel
        self.while_args_ptr = while_args_ptr
        self.handle = handle
        self.capture_stream = int(capture_stream)
        self.body_stream = int(body_stream)
        self.body_graph = None

    def __enter__(self):
        info = cu(cudart.cudaStreamGetCaptureInfo(self.capture_stream))
        capture_status, _id, graph, *_rest = info
        assert (
            capture_status
            == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
        )
        cu(
            cuda.cuLaunchKernel(
                self.while_kernel, 1, 1, 1, 1, 1, 1, 0,
                self.capture_stream, self.while_args_ptr, 0,
            )
        )
        info = cu(cudart.cudaStreamGetCaptureInfo(self.capture_stream))
        capture_status, _id, graph, dependencies, *_rest = info
        driver_params = cuda.CUgraphNodeParams()
        driver_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        driver_params.conditional.handle = self.handle
        driver_params.conditional.type = (
            cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
        )
        driver_params.conditional.size = 1
        ctx = cu(cuda.cuCtxGetCurrent())
        driver_params.conditional.ctx = ctx
        if _IS_CUDA13:
            node = cu(
                cuda.cuGraphAddNode(
                    graph, dependencies, None, len(dependencies), driver_params
                )
            )
        else:
            node = cu(
                cuda.cuGraphAddNode(
                    graph, dependencies, len(dependencies), driver_params
                )
            )
        self.body_graph = driver_params.conditional.phGraph_out[0]
        if _IS_CUDA13:
            cu(
                cudart.cudaStreamUpdateCaptureDependencies(
                    self.capture_stream, [node], None, 1,
                    cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
                )
            )
        else:
            cu(
                cudart.cudaStreamUpdateCaptureDependencies(
                    self.capture_stream, [node], 1,
                    cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
                )
            )
        cu(
            cudart.cudaStreamBeginCaptureToGraph(
                self.body_stream, self.body_graph, None, None, 0,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
            )
        )
        return self.body_stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            cudart.cudaStreamEndCapture(self.body_stream)
            return False
        cu(
            cuda.cuLaunchKernel(
                self.while_kernel, 1, 1, 1, 1, 1, 1, 0,
                self.body_stream, self.while_args_ptr, 0,
            )
        )
        cu(cudart.cudaStreamEndCapture(self.body_stream))
        return False


# decode_predicate.cu (verified in torch_predicate_poc.py), inlined.
PREDICATE_SRC = r"""
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__
void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

// DFlash-aware: a decode step commits 0..block_size tokens per row (variable
// accept length from speculative verify), not exactly one. out_tokens is
// [bs, block_size] row-major; commit_lens[r] says how many of row r's slots
// are real this step. Walk them in order, same accounting as the original
// single-token kernel (out_len incremented per emitted token, EOS/max_new
// checked after each increment) -- degenerates to the original semantics
// exactly when block_size == 1 and commit_lens[r] == 1 always.
//
// This is a GPU-side, best-effort CONTINUATION predicate for the WHILE
// loop only -- it decides whether to keep spending on-GPU steps on a row
// within this chunk. It is not the authoritative finish/truncation source:
// that remains the host-side batch result processor (grammar/stop-string
// matching and other host-only conditions can't run here and are handled
// by the host between chunks).
extern "C" __global__
void decode_finish_update(const long long* out_tokens,   // [bs, block_size]
                          const int* commit_lens,        // [bs]
                          int block_size,
                          const long long* eos_ids, int n_eos,
                          int* out_len,
                          const int* max_new,
                          unsigned char* finished,
                          int* step,
                          int* n_running,
                          int bs)
{
    int r = threadIdx.x;
    if (r < bs && finished[r] == 0) {
        int n_commit = commit_lens[r];
        for (int j = 0; j < n_commit; ++j) {
            long long t = out_tokens[r * block_size + j];
            bool hit = false;
            for (int i = 0; i < n_eos; ++i) hit |= (t == eos_ids[i]);
            out_len[r] += 1;
            if (hit || out_len[r] >= max_new[r]) {
                finished[r] = 1;
                break;
            }
        }
    }
    __syncthreads();
    if (r == 0) {
        int run = 0;
        for (int i = 0; i < bs; ++i) run += (finished[i] == 0);
        *n_running = run;
        *step += 1;
    }
}

extern "C" __global__
void decode_while_cond(cudaGraphConditionalHandle handle,
                       const int* n_running,
                       const int* step,
                       const int* chunk_cap)
{
    cudaGraphSetConditional(handle, (*n_running > 0) && (*step < *chunk_cap));
}
"""


class CondGraph:
    """An instantiated conditional decode graph plus the live kernel-argument
    objects (numpy holders) that MUST outlive it. Launch on the caller's stream
    so subsequent torch reads of the output buffer order correctly."""

    def __init__(self, gexec, finish_ka, cond_ka, pool, streams):
        self.gexec = gexec
        self.finish_ka = finish_ka  # keep alive (dangling-arg gotcha)
        self.cond_ka = cond_ka
        self.pool = pool            # keep the body mem pool alive
        self.streams = streams      # keep capture/body streams alive

    def launch(self, stream_ptr: int):
        cu(cudart.cudaGraphLaunch(self.gexec, int(stream_ptr)))

    def destroy(self) -> None:
        """Release the instantiated graph executable's native CUDA resource.

        `gexec` is a raw `cudart` handle, not an RAII-wrapped Python object --
        Python garbage collecting this `CondGraph` (or its owning
        `DflashCCDChunk`) does not free it. Left undestroyed, the driver
        keeps treating `self.pool`'s memory as referenced forever, so it's
        never returned to the allocator even once every Python reference is
        gone -- confirmed live (task #136): `gc.collect()` correctly drops
        Python-side chunk/graph object counts to zero between requests, yet
        `torch.cuda.memory_allocated()` climbed every chunk regardless,
        eventually OOMing. Caller must call this exactly once, after the
        single `launch()` + result readback this class is built for (CCD
        never reuses a captured chunk -- see `run_ccd_chunk`'s own
        docstring), and only when not itself inside a capture.
        """
        cu(cudart.cudaGraphExecDestroy(self.gexec))


class CCConditionalDecode:
    """Compiles the loop-control kernels once and captures conditional decode
    graphs. The decode-step body (model forward + in-graph greedy argmax +
    autoregressive carry) is supplied by the caller as `body_fn`; this class
    owns the WHILE node, the predicate kernels, and instantiation."""

    def __init__(self):
        self.finish_k = run_nvrtc(
            PREDICATE_SRC, b"decode_finish_update", b"decode_predicate.cu"
        )
        self.cond_k = run_nvrtc(
            PREDICATE_SRC, b"decode_while_cond", b"decode_predicate.cu"
        )

    def make_finish_args(
        self,
        *,
        out_tokens,
        commit_lens,
        block_size,
        eos_ids,
        n_eos,
        out_len,
        max_new,
        finished,
        d_step,
        n_run,
        bs,
    ) -> KernelArgs:
        """Build (and return) the KernelArgs for decode_finish_update. Caller
        keeps the returned object alive and launches it inside body_fn.

        out_tokens: device ptr, [bs, block_size] int64, row-major.
        commit_lens: device ptr, [bs] int32 -- valid token count per row this step.
        block_size: block_size == 1 / commit_lens[r] == 1 always reproduces the
            original single-token-per-step semantics exactly.
        """
        return (
            KernelArgs()
            .add_ptr(out_tokens)
            .add_ptr(commit_lens)
            .add_int(block_size)
            .add_ptr(eos_ids)
            .add_int(n_eos)
            .add_ptr(out_len)
            .add_ptr(max_new)
            .add_ptr(finished)
            .add_ptr(d_step)
            .add_ptr(n_run)
            .add_int(bs)
        )

    def launch_finish(self, finish_args_ptr, bs, body_stream_ptr):
        """Launch decode_finish_update (BS threads) inside the body."""
        cu(
            cuda.cuLaunchKernel(
                self.finish_k, 1, 1, 1, bs, 1, 1, 0,
                int(body_stream_ptr), finish_args_ptr, 0,
            )
        )

    def capture(
        self, *, body_fn, finish_ka, n_run, d_step, d_cap, torch_mod, warmup_fn=None
    ):
        """Capture a conditional WHILE decode graph.

        body_fn(body_stream_ptr): records ONE decode sub-step + the finish kernel
            into the WHILE body. Runs on the body stream (caller wraps its torch
            ops in `with torch.cuda.stream(s_body): with use_mem_pool(pool):`).
        finish_ka: the KernelArgs for decode_finish_update (kept alive here).
        n_run/d_step/d_cap: device int32[1] tensors read by decode_while_cond.
        warmup_fn(body_stream_ptr), optional: run once, eagerly, on s_body/pool
            BEFORE capture begins -- for persistent buffers body_fn reads that
            were last written on a different stream (e.g. built during the
            priming tick, before s_body existed). Touching them here lets
            PyTorch's caching allocator record s_body as a consumer stream
            outside any capture, so body_fn's later read doesn't need a fresh
            cross-stream wait -- which would be an illegal WaitEvent node
            inside this WHILE's captured body graph. See tdx-checkpoints/109.
        """
        torch = torch_mod
        s_cap = torch.cuda.Stream()
        s_body = torch.cuda.Stream()
        pool = torch.cuda.MemPool()

        if warmup_fn is not None:
            with torch.cuda.stream(s_body):
                with torch.cuda.use_mem_pool(pool):
                    warmup_fn(s_body.cuda_stream)
            torch.cuda.current_stream().wait_stream(s_body)

        import os as _os

        _copy_debug = _os.environ.get("SGLANG_CCD_COPY_DEBUG") == "1"
        _orig_copy_ = None
        if _copy_debug:
            _orig_copy_ = torch.Tensor.copy_
            _seen_sites = set()

            def _patched_copy_(self_t, src, non_blocking=False):
                try:
                    if (
                        torch.cuda.is_current_stream_capturing()
                        and getattr(src, "device", None) is not None
                        and src.device.type == "cpu"
                        and getattr(self_t, "device", None) is not None
                        and self_t.device.type == "cuda"
                    ):
                        import traceback

                        frame = "".join(traceback.format_stack(limit=12)[:-1])
                        if frame not in _seen_sites:
                            _seen_sites.add(frame)
                            is_pinned = "?"
                            try:
                                is_pinned = src.is_pinned()
                            except Exception:
                                pass
                            sys.stderr.write(
                                "CCD_COPY_DEBUG: H2D copy_ during capture: "
                                f"dst.device={self_t.device} src.device={src.device} "
                                f"src.is_pinned={is_pinned} src.shape={tuple(src.shape)} "
                                f"src.dtype={src.dtype} non_blocking={non_blocking}\n{frame}\n"
                            )
                            sys.stderr.flush()
                except Exception as _e:
                    sys.stderr.write(f"CCD_COPY_DEBUG: hook error: {_e!r}\n")
                return _orig_copy_(self_t, src, non_blocking=non_blocking)

            torch.Tensor.copy_ = _patched_copy_

        _comm_debug = _os.environ.get("SGLANG_CCD_COMM_DEBUG") == "1"
        _orig_comm_fns = {}
        if _comm_debug:
            import torch.distributed as _dist

            _comm_names = (
                "all_gather_into_tensor",
                "all_gather",
                "all_to_all_single",
                "all_to_all",
                "reduce_scatter_tensor",
                "reduce_scatter",
                "all_reduce",
                "broadcast",
            )

            def _make_comm_hook(_name, _orig_fn):
                def _patched_comm(*args, **kwargs):
                    try:
                        if torch.cuda.is_current_stream_capturing():
                            import traceback

                            frame = "".join(traceback.format_stack(limit=15)[:-1])
                            sys.stderr.write(
                                f"CCD_COMM_DEBUG: {_name} CALLED during capture\n{frame}\n"
                            )
                            sys.stderr.flush()
                    except Exception as _e:
                        sys.stderr.write(f"CCD_COMM_DEBUG: hook error: {_e!r}\n")
                    return _orig_fn(*args, **kwargs)

                return _patched_comm

            for _name in _comm_names:
                _orig_fn = getattr(_dist, _name, None)
                if _orig_fn is not None:
                    _orig_comm_fns[_name] = _orig_fn
                    setattr(_dist, _name, _make_comm_hook(_name, _orig_fn))

        _graph_debug = _os.environ.get("SGLANG_CCD_GRAPH_DEBUG") == "1"
        _orig_graph_fns = {}
        if _graph_debug:

            def _make_hook(_name, _owner, _orig_fn, _event_ptr=False):
                def _patched(self_o, *args, **kwargs):
                    try:
                        if torch.cuda.is_current_stream_capturing():
                            import traceback

                            frame = "".join(traceback.format_stack(limit=20)[:-1])
                            ptr_info = ""
                            if _event_ptr:
                                try:
                                    ptr_info = f" cuda_event={hex(int(self_o.cuda_event))}"
                                except Exception:  # noqa: BLE001
                                    pass
                            sys.stderr.write(
                                f"CCD_GRAPH_DEBUG: {_owner}.{_name} CALLED during capture "
                                f"id={id(self_o)}{ptr_info}\n{frame}\n"
                            )
                            sys.stderr.flush()
                    except Exception as _e:
                        sys.stderr.write(f"CCD_GRAPH_DEBUG: hook error: {_e!r}\n")
                    return _orig_fn(self_o, *args, **kwargs)

                return _patched

            _graph_targets = (
                (torch.cuda.CUDAGraph, "replay", False),
                (torch.cuda.CUDAGraph, "capture_begin", False),
                (torch.cuda.Stream, "wait_stream", False),
                (torch.cuda.Stream, "wait_event", False),
                (torch.cuda.Event, "record", True),
                (torch.cuda.Event, "wait", True),
            )
            for _cls, _name, _event_ptr in _graph_targets:
                _orig_fn = getattr(_cls, _name)
                _orig_graph_fns[(_cls, _name)] = _orig_fn
                setattr(
                    _cls,
                    _name,
                    _make_hook(_name, _cls.__name__, _orig_fn, _event_ptr=_event_ptr),
                )

        try:
            with torch.cuda.stream(s_cap):
                cu(
                    cudart.cudaStreamBeginCapture(
                        s_cap.cuda_stream,
                        cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
                    )
                )
                _st, _id, graph, *_ = cu(
                    cudart.cudaStreamGetCaptureInfo(s_cap.cuda_stream)
                )
                handle = cu(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                cond_ka = (
                    KernelArgs()
                    .add_raw(handle.getPtr())
                    .add_ptr(n_run.data_ptr())
                    .add_ptr(d_step.data_ptr())
                    .add_ptr(d_cap.data_ptr())
                )
                cond_args = cond_ka.build()
                with with_conditional_node(
                    self.cond_k, cond_args, handle, s_cap.cuda_stream, s_body.cuda_stream
                ):
                    with torch.cuda.stream(s_body):
                        with torch.cuda.use_mem_pool(pool):
                            # tvm_ffi ops (FlashInfer's trtllm_mha kernels,
                            # CustomAllReduceV2) resolve "the current stream"
                            # via TVMFFIEnvGetStream -- a bookkeeping variable
                            # entirely separate from torch's current-stream
                            # state, NOT auto-updated by `with torch.cuda.
                            # stream(s_body):` above (documented gotcha in
                            # flashinfer's own moe_utils.py: "does NOT track
                            # PyTorch's torch.cuda.stream(...) Python
                            # context"). Without this, a tvm_ffi op launched
                            # here runs on whatever stream was last pushed
                            # into that separate registry (stale, not
                            # s_body), and reconciling its output back onto
                            # s_body needs a genuine cross-stream join --
                            # exactly the WaitEvent node illegal inside this
                            # WHILE body.
                            import tvm_ffi as _tvm_ffi

                            _ffidbg = _os.environ.get("SGLANG_CCD_TVMFFI_DEBUG") == "1"
                            if _ffidbg:
                                _before = _tvm_ffi.get_raw_stream(
                                    _tvm_ffi.device(str(s_body.device))
                                )
                                sys.stderr.write(
                                    f"CCD_TVMFFI_DEBUG: before enter: ffi_env_stream={hex(_before)} "
                                    f"s_body.cuda_stream={hex(s_body.cuda_stream)}\n"
                                )
                                sys.stderr.flush()
                            with _tvm_ffi.use_torch_stream():
                                if _ffidbg:
                                    _after = _tvm_ffi.get_raw_stream(
                                        _tvm_ffi.device(str(s_body.device))
                                    )
                                    sys.stderr.write(
                                        f"CCD_TVMFFI_DEBUG: inside context: ffi_env_stream={hex(_after)} "
                                        f"s_body.cuda_stream={hex(s_body.cuda_stream)} "
                                        f"match={_after == s_body.cuda_stream}\n"
                                    )
                                    sys.stderr.flush()
                                body_fn(s_body.cuda_stream)
                top = cu(cudart.cudaStreamEndCapture(s_cap.cuda_stream))
        finally:
            if _copy_debug:
                torch.Tensor.copy_ = _orig_copy_
            if _comm_debug:
                import torch.distributed as _dist

                for _name, _orig_fn in _orig_comm_fns.items():
                    setattr(_dist, _name, _orig_fn)
            if _graph_debug:
                for (_cls, _name), _orig_fn in _orig_graph_fns.items():
                    setattr(_cls, _name, _orig_fn)

        try:
            gexec = cu(cudart.cudaGraphInstantiate(top, 0))
        except (TypeError, RuntimeError):
            try:
                gexec = cu(cudart.cudaGraphInstantiateWithFlags(top, 0))
            except RuntimeError:
                _dump_graph_diagnostics(top)
                _dump_instantiate_with_params_diagnostics(top)
                raise

        # `top` (the graph template) is a separate native resource from
        # `gexec` (the instantiated executable) -- cudaGraphInstantiate does
        # not take ownership of it, and nothing needs it after this point
        # (replay only ever uses gexec). Same leak class as CondGraph.destroy
        # if left undestroyed (task #136).
        cu(cudart.cudaGraphDestroy(top))

        return CondGraph(gexec, finish_ka, cond_ka, pool, (s_cap, s_body))


def _resolve_addr_to_lib(addr: int) -> str:
    """CCD_DEBUG: map a host function pointer (e.g. a kernel node's `func`,
    which is the host-side launch stub cudaLaunchKernel was given) to the
    shared library it falls inside, by scanning this process's own
    /proc/self/maps. Identifies which compiled extension (torch, flashinfer,
    trtllm, sgl_kernel, ...) owns a kernel without needing cuFuncGetName,
    which can fail before the graph is fully instantiated. Best-effort."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                lo_hi = parts[0].split("-")
                lo, hi = int(lo_hi[0], 16), int(lo_hi[1], 16)
                if lo <= addr < hi:
                    return parts[5]
        return "<no mapping found>"
    except Exception as e:  # noqa: BLE001
        return f"<lookup failed: {e!r}>"


def _dump_graph_diagnostics(graph) -> None:
    """CCD_DEBUG: on cudaGraphInstantiate failure, dump node count/types and
    a DOT file so the invalid node can be identified without re-deriving
    this from scratch each time. Best-effort -- never raises past this."""
    import sys as _sys

    try:
        _err, _nodes0, n = cudart.cudaGraphGetNodes(graph, 0)
        _check(_err)
        _sys.stderr.write(f"CCD_DEBUG: graph has {n} top-level nodes\n")
        _err, nodes, n2 = cudart.cudaGraphGetNodes(graph, n)
        _check(_err)
        for i, node in enumerate(nodes[:n2] if n2 else []):
            try:
                ntype = cu(cudart.cudaGraphNodeGetType(node))
                _sys.stderr.write(f"CCD_DEBUG: node[{i}] type={ntype}\n")
            except Exception as e:  # noqa: BLE001
                _sys.stderr.write(f"CCD_DEBUG: node[{i}] type lookup failed: {e}\n")
    except Exception as e:  # noqa: BLE001
        _sys.stderr.write(f"CCD_DEBUG: node enumeration failed: {e}\n")

    try:
        import os as _os

        dot_dir = "/home/ubuntu/sglang-cc/cover"
        _os.makedirs(dot_dir, exist_ok=True)
        dot_path = f"{dot_dir}/ccd_failed_graph_pid{_os.getpid()}.dot".encode()
        cu(
            cudart.cudaGraphDebugDotPrint(
                graph,
                dot_path,
                cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose,
            )
        )
        _sys.stderr.write(f"CCD_DEBUG: wrote {dot_path.decode()}\n")
    except Exception as e:  # noqa: BLE001
        _sys.stderr.write(f"CCD_DEBUG: dot print failed: {e}\n")


def _dump_instantiate_with_params_diagnostics(graph) -> None:
    """CCD_DEBUG: cudaGraphInstantiateWithParams reports the SPECIFIC
    offending node and a typed reason (cudaGraphInstantiateResult) instead
    of a bare cudaErrorInvalidValue. Best-effort, never raises."""
    import sys as _sys

    try:
        params = cudart.cudaGraphInstantiateParams()
        params.flags = 0
        err, gexec = cudart.cudaGraphInstantiateWithParams(graph, params)
        _sys.stderr.write(
            f"CCD_DEBUG: cudaGraphInstantiateWithParams err={err} "
            f"result_out={params.result_out} errNode_out={hex(int(params.errNode_out))}\n"
        )
        if int(params.errNode_out) != 0:
            try:
                ntype = cu(cudart.cudaGraphNodeGetType(params.errNode_out))
                _sys.stderr.write(f"CCD_DEBUG: errNode type={ntype}\n")
            except Exception as e:  # noqa: BLE001
                _sys.stderr.write(f"CCD_DEBUG: errNode type lookup failed: {e}\n")

            # CCD_DEBUG: for a WaitEvent node, pull the raw cudaEvent_t handle
            # it waits on -- cross-reference this pointer against
            # CCD_GRAPH_DEBUG's Event.record creation-site stack traces
            # (SGLANG_CCD_GRAPH_DEBUG=1) to identify who created the event,
            # instead of guessing from call-site elimination.
            try:
                ev_err, ev = cuda.cuGraphEventWaitNodeGetEvent(params.errNode_out)
                _check(ev_err)
                _sys.stderr.write(
                    f"CCD_DEBUG: WaitEvent node event handle={hex(int(ev))}\n"
                )
            except Exception as e:  # noqa: BLE001
                _sys.stderr.write(f"CCD_DEBUG: WaitEvent node event lookup failed: {e}\n")

            # CCD_DEBUG: neighboring kernel names. Python-level monkeypatches
            # on torch.cuda.Event/Stream/CUDAGraph (SGLANG_CCD_GRAPH_DEBUG=1)
            # can only see calls that go through torch's own Python classes --
            # a WaitEvent node generated by a compiled C++/CUDA extension
            # calling the driver API directly (e.g. FlashInfer, TensorRT-LLM)
            # is invisible to those hooks. The kernel immediately before/after
            # the errNode in the graph's dependency chain identifies the
            # responsible code region without needing gdb.
            for _label, _lookup in (
                ("dependency (runs BEFORE errNode)", cudart.cudaGraphNodeGetDependencies),
                ("dependent (runs AFTER errNode)", cudart.cudaGraphNodeGetDependentNodes),
            ):
                try:
                    _err, _nbrs0, _edata0, _n = _lookup(params.errNode_out, 0)
                    _check(_err)
                    _err, _nbrs, _edata, _n2 = _lookup(params.errNode_out, _n)
                    _check(_err)
                    for _nb in (_nbrs[:_n2] if _n2 else []):
                        try:
                            _nb_type = cu(cudart.cudaGraphNodeGetType(_nb))
                            _sys.stderr.write(
                                f"CCD_DEBUG: {_label}: type={_nb_type}\n"
                            )
                            if (
                                _nb_type
                                == cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
                            ):
                                _k_err, _k_params = cudart.cudaGraphKernelNodeGetParams(
                                    _nb
                                )
                                _check(_k_err)
                                _func_addr = int(_k_params.func)
                                try:
                                    _f_err, _fname = cuda.cuFuncGetName(
                                        cuda.CUfunction(_func_addr)
                                    )
                                    _check(_f_err)
                                    _sys.stderr.write(
                                        f"CCD_DEBUG: {_label}: kernel name="
                                        f"{_fname.decode() if isinstance(_fname, bytes) else _fname}\n"
                                    )
                                except Exception as e:  # noqa: BLE001
                                    _sys.stderr.write(
                                        f"CCD_DEBUG: {_label}: kernel name lookup failed: {e}\n"
                                    )
                                    _sys.stderr.write(
                                        f"CCD_DEBUG: {_label}: kernel func addr={hex(_func_addr)} "
                                        f"owning lib={_resolve_addr_to_lib(_func_addr)}\n"
                                    )
                        except Exception as e:  # noqa: BLE001
                            _sys.stderr.write(
                                f"CCD_DEBUG: {_label}: neighbor inspect failed: {e}\n"
                            )
                except Exception as e:  # noqa: BLE001
                    _sys.stderr.write(f"CCD_DEBUG: {_label} lookup failed: {e}\n")

            # CCD_DEBUG: for a Memcpy node, pull the actual src/dst pointers
            # and copy kind, then resolve each pointer's owning device --
            # this identifies exactly which two devices the offending copy
            # spans, instead of guessing from Python-level stack traces.
            try:
                mc_err, mc_params = cudart.cudaGraphMemcpyNodeGetParams(
                    params.errNode_out
                )
                _check(mc_err)
                src_ptr = int(mc_params.srcPtr.ptr)
                dst_ptr = int(mc_params.dstPtr.ptr)
                kind = mc_params.kind
                _sys.stderr.write(
                    f"CCD_DEBUG: memcpy kind={kind} "
                    f"src_ptr={hex(src_ptr)} dst_ptr={hex(dst_ptr)}\n"
                )
                for label, ptr in (("src", src_ptr), ("dst", dst_ptr)):
                    try:
                        pa_err, attrs = cudart.cudaPointerGetAttributes(ptr)
                        _check(pa_err)
                        _sys.stderr.write(
                            f"CCD_DEBUG: {label} ptr device={attrs.device} "
                            f"type={attrs.type}\n"
                        )
                    except Exception as e:  # noqa: BLE001
                        _sys.stderr.write(
                            f"CCD_DEBUG: {label} ptr attribute lookup failed: {e}\n"
                        )
            except Exception as e:  # noqa: BLE001
                _sys.stderr.write(f"CCD_DEBUG: memcpy node param dump failed: {e}\n")
    except Exception as e:  # noqa: BLE001
        _sys.stderr.write(f"CCD_DEBUG: cudaGraphInstantiateWithParams itself failed: {e}\n")
