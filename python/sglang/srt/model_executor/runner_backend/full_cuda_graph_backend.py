# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

import contextvars
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    GraphPoolPrecarve,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

# CC conditional decode (cc_conditional_decode.py / dflash_cc_conditional_decode.py):
# needs the pre-instantiation cudaGraph_t of each captured shape so it can be
# embedded as a child graph node inside a WHILE-loop capture. torch.cuda.CUDAGraph
# normally discards that structure after instantiate() to save memory; keep_graph=True
# retains it. Strictly opt-in (retaining it for every shape has a real, if small,
# memory/bookkeeping cost) and confirmed not to change capture/replay behavior for
# the ordinary (non-CCD) path when unset. Read once at import.
SGLANG_CC_CONDITIONAL_DECODE = get_bool_env_var("SGLANG_CC_CONDITIONAL_DECODE")

# Runtime (not import-time) toggle: True only while a CC conditional decode
# chunk's WHILE-loop body is actively being captured. When True, replay()
# embeds the already-captured graph as a child node (cc_conditional_decode.py's
# add_child_graph_node) instead of calling torch.cuda.CUDAGraph.replay() --
# which the CUDA runtime itself rejects during an active outer capture
# (confirmed: cudaErrorStreamCaptureUnsupported). A ContextVar rather than a
# plain attribute so it's correct under whatever thread/task the capture runs
# on; default False leaves every ordinary (non-CCD) replay() call untouched.
_ccd_capturing: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cc_conditional_decode_capturing", default=False
)


@contextmanager
def ccd_capturing_context():
    """Enter while capturing a CC conditional decode chunk body. Every
    FullCudaGraphBackend.replay() call made inside this context embeds a
    child graph node instead of replaying directly."""
    token = _ccd_capturing.set(True)
    try:
        yield
    finally:
        _ccd_capturing.reset(token)


def is_ccd_capturing() -> bool:
    """True inside an active ccd_capturing_context(). Public getter so
    other modules (e.g. attention backends deciding whether it's safe to
    re-invoke a plan() call with its own host-memory traffic) don't reach
    into the private ContextVar directly."""
    return _ccd_capturing.get()


def _allocate_output_buffer(output: Any) -> Optional[Any]:
    """Same-structure buffer for a tensor or a (nested) list/tuple of
    row-major tensors; None when any leaf is something else."""
    if torch.is_tensor(output):
        return None if output.ndim == 0 else torch.empty_like(output)
    if isinstance(output, (list, tuple)) and output:
        buffers = [_allocate_output_buffer(item) for item in output]
        if any(buffer is None for buffer in buffers):
            return None
        return tuple(buffers) if isinstance(output, tuple) else buffers
    return None


def _output_fits_buffer(output: torch.Tensor, output_buffer: Any) -> bool:
    return (
        torch.is_tensor(output_buffer)
        and output.ndim == output_buffer.ndim
        and output.shape[1:] == output_buffer.shape[1:]
        and output.shape[0] <= output_buffer.shape[0]
        and output.dtype == output_buffer.dtype
        and output.device == output_buffer.device
    )


def _copy_output_to_buffer(output: Any, output_buffer: Any) -> Optional[Any]:
    """Copy ``output`` leaf-wise into the leading rows of ``output_buffer`` and
    return the same structure of buffer views; None when it does not fit."""
    if torch.is_tensor(output):
        if not _output_fits_buffer(output, output_buffer):
            return None
        shared_output = output_buffer[: output.shape[0]]
        shared_output.copy_(output)
        return shared_output
    if isinstance(output, (list, tuple)):
        if not isinstance(output_buffer, (list, tuple)) or len(output) != len(
            output_buffer
        ):
            return None
        shared = [
            _copy_output_to_buffer(item, buffer)
            for item, buffer in zip(output, output_buffer)
        ]
        if any(item is None for item in shared):
            return None
        return tuple(shared) if isinstance(output, tuple) else shared
    return None


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        reuse_output_buffer: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = get_parallel().tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._precarve = GraphPoolPrecarve()
        self._reuse_output_buffer = reuse_output_buffer
        self._output_buffer: Optional[Any] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # When per-bs capture traces are enabled (--enable-profile-cuda-graph +
        # SGLANG_GRAPH_BATCH_CAPTURE), the runner created a scheduled
        # torch profiler (wait=2, active=1) and exposed it as _profiler. We step()
        # past the two warmup runs so only the capture run is recorded, and each
        # captured shape produces its own trace via the profiler's on_trace_ready.
        # With --enable-profile-cuda-graph alone the runner leaves _profiler None
        # (its unscheduled profiler records the whole capture in one pass), so no
        # stepping happens here.
        runner = self._cuda_graph_runner
        profiler = (
            getattr(runner, "_profiler", None)
            if getattr(runner, "enable_profile_cuda_graph", False)
            else None
        )

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        warmup_output = None
        for warmup_step in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            with self._precarve.measure():
                output = forward_fn()
            if self._reuse_output_buffer and warmup_step == 1:
                warmup_output = output
            del output
            if profiler is not None:
                profiler.step()
            if post_warmup_hook is not None:
                post_warmup_hook()

        if self._reuse_output_buffer and self._output_buffer is None:
            # Prefill captures the largest shape first and replays one shape at
            # a time, so all graphs can share this eager-tail input buffer.
            self._output_buffer = _allocate_output_buffer(warmup_output)
            self._reuse_output_buffer = self._output_buffer is not None
        del warmup_output

        graph = torch.cuda.CUDAGraph(keep_graph=SGLANG_CC_CONDITIONAL_DECODE)

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
                # replays read capture-time state from the pool, so a pause must back it up
                enable_cpu_backup=True,
            )
        else:
            graph_ctx = self._device_module.graph

        with (
            graph_pool_capture_scope(),
            graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream),
        ):
            self._precarve.mint()
            out = forward_fn()
            if self._reuse_output_buffer:
                output_buffer = self._output_buffer
                assert output_buffer is not None
                shared_output = _copy_output_to_buffer(out, output_buffer)
                self._reuse_output_buffer = shared_output is not None
                if shared_output is not None:
                    out = shared_output

        if profiler is not None:
            profiler.step()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        if _ccd_capturing.get():
            from sglang.srt.model_executor.cc_conditional_decode import (
                add_child_graph_node,
            )

            raw_graph = self.get_raw_graph(shape_key)
            if raw_graph is None:
                raise RuntimeError(
                    f"CC conditional decode: no captured graph (or "
                    f"keep_graph=False at capture time) for shape {shape_key}; "
                    "the CCD eligibility gate should have excluded this batch."
                )
            add_child_graph_node(
                torch.cuda.current_stream().cuda_stream, raw_graph
            )
        else:
            with graph_pool_replay_scope():
                self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        for graph in self._graphs.values():
            graph.reset()
        self._graphs.clear()
        self._outputs.clear()
        self._output_buffer = None
        self._pool = None

    def get_raw_graph(self, shape_key: ShapeKey) -> Optional[int]:
        """Pre-instantiation cudaGraph_t handle for an already-captured shape,
        for embedding as a child graph node elsewhere (CC conditional decode).
        Only populated when SGLANG_CC_CONDITIONAL_DECODE was set at capture
        time (keep_graph=True); returns None otherwise, and None if the shape
        was never captured."""
        graph = self._graphs.get(shape_key)
        if graph is None or not SGLANG_CC_CONDITIONAL_DECODE:
            return None
        return graph.raw_cuda_graph()
