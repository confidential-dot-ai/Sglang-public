# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/cuda_piecewise_backend.py

import dataclasses
import logging
import os
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors
from sglang.srt.model_executor.runner_utils.pool import (
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.utils.common import print_warning_once

logger = logging.getLogger(__name__)


# Fault-localization aid (SGLANG_PCG_SYNC_DEBUG=1): synchronize around every
# captured piecewise-segment replay so an async illegal access is attributed
# to either the segment itself ("post") or the eager region before it
# ("pre"), instead of surfacing at an arbitrary later CUDA call. Debug only.
_PCG_SYNC_DEBUG = os.environ.get("SGLANG_PCG_SYNC_DEBUG", "0") == "1"


def _pcg_sync_checkpoint(
    phase: str, index: int, total: int, runtime_shape
) -> None:
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        raise RuntimeError(
            f"PCG_SYNC_DEBUG[{phase}] segment {index}/{total} "
            f"shape={runtime_shape}: {e}"
        ) from e


def _pcg_general_call(fn, args, index, total, tag):
    if not _PCG_SYNC_DEBUG or torch.cuda.is_current_stream_capturing():
        return fn(*args)
    _pcg_sync_checkpoint(f"general-pre:{tag}", index, total, -1)
    out = fn(*args)
    _pcg_sync_checkpoint(f"general-post:{tag}", index, total, -1)
    return out


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    need_to_compile: bool  # the size is in compile_sizes
    use_cudagraph: bool  # the size is in cudagraph_capture_sizes

    compiled: bool = False
    runnable: Callable = None  # type: ignore
    num_finished_warmup: int = 0
    cudagraph: Optional[torch.cuda.CUDAGraph] = None
    output: Optional[Any] = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class CUDAPiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule,
        compile_config: CompilationConfig,
        inductor_config: dict[str, Any],
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.inductor_config = inductor_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.sglang_backend = sglang_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.compile_sizes: set[int] = set([])
        self.compile_config = compile_config
        self.cudagraph_capture_sizes: set[int] = set(compile_config.get_capture_sizes())

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        # the entries for different shapes that we need to either
        # compile or capture cudagraph
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = self.compile_sizes.copy()
        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            # no specific sizes to compile
            # save the hash of the inductor graph for the next run
            self.sglang_backend.compiler_manager.save_to_file()

    def __call__(self, *args) -> Any:
        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return _pcg_general_call(
                self.compiled_graph_for_general_shape, args,
                self.piecewise_compile_index, self.total_piecewise_compiles,
                "first_run",
            )

        if len(self.sym_shape_indices) == 0:
            return _pcg_general_call(
                self.compiled_graph_for_general_shape, args,
                self.piecewise_compile_index, self.total_piecewise_compiles,
                "no_sym",
            )

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return _pcg_general_call(
                self.compiled_graph_for_general_shape, args,
                self.piecewise_compile_index, self.total_piecewise_compiles,
                f"uncaptured:{runtime_shape}",
            )

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.sglang_backend.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape,
            )

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        if is_in_torch_compile_warmup():
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  # noqa
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            # During normal capture (PiecewiseCudaGraphRunner.capture()),
            # set_pcg_capture_stream() guarantees a valid stream. However,
            # Dynamo may silently recompile serving batches when a dynamic
            # multimodal input introduces a previously unseen guard. The
            # replacement backend has no capture stream, so it cannot safely
            # create a CUDA graph. Execute that subgraph normally instead of
            # crashing the scheduler; subsequent matching shapes still use
            # their captured graphs.
            stream = get_pcg_capture_stream()
            if stream is None:
                # Also covers ring-attention CP: its zigzag split derives an
                # unbacked SymInt shard-length symbol per capture
                # (cp_utils.py's cp_split_and_rebuild_data_pcg / communicator.py's
                # _moe_cp_pad_and_allgather), and a runtime-triggered dynamo
                # recompile on a shape outside the captured set can land here
                # the same way the general case below does. Falling back to
                # eager for this subgraph instead of asserting is slower for
                # that one subgraph, but never a crash, and (unlike
                # re-entering graph_capture() to force a fresh CUDA-graph
                # recapture here) this can't corrupt the shared graph pool on
                # a later replay.
                print_warning_once(
                    "PCG capture stream is not set. This can be a Dynamo runtime "
                    "recompilation or an optional VLM branch pre-warmed outside "
                    "CUDA graph capture; falling back to eager execution for this "
                    "subgraph."
                )
                return entry.runnable(*args)
            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))
                # mind-exploding: carefully manage the reference and memory.
                with (
                    graph_pool_capture_scope(),
                    torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=stream),
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.compile_config.get_enable_debug_mode():
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )
        if _PCG_SYNC_DEBUG:
            _pcg_sync_checkpoint(
                "pre", self.piecewise_compile_index, self.total_piecewise_compiles,
                runtime_shape,
            )
        with graph_pool_replay_scope():
            entry.cudagraph.replay()
        if _PCG_SYNC_DEBUG:
            _pcg_sync_checkpoint(
                "post", self.piecewise_compile_index, self.total_piecewise_compiles,
                runtime_shape,
            )
        return entry.output
