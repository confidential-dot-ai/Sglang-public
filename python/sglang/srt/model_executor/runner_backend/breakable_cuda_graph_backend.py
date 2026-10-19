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
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
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


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._cuda_graph_runner = cuda_graph_runner
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._capture_inputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = get_parallel().tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        self._precarve = GraphPoolPrecarve()
        self._shared_output_rows: int = 0
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self._shared_output_rows = 0
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        warmup_out = None
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            with self._precarve.measure():
                warmup_out = forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = BreakableCUDAGraph(self.deduped_cuda_graph)
        captured_fn = eager_on_graph(forward_fn) if self._debug_eager else forward_fn
        buffer_rows = self._ensure_shared_output_buffer(warmup_out)
        with (
            graph_pool_capture_scope(),
            BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=self._pool,
                stream=self._capture_stream,
                barrier_fn=self._tp_group.barrier,
            ),
        ):
            self._precarve.mint()
            out = captured_fn()
            out_rows = self._output_rows(out, buffer_rows)
            self._copy_output_to_buffer(out, self._shared_output_buffer, out_rows)

        stored = self._slice_output(self._shared_output_buffer, out_rows)
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored
        # CUDA graphs retain tensor addresses, not Python tensor lifetimes.
        self._capture_inputs[shape_key] = capture_inputs

    def _ensure_shared_output_buffer(self, warmup_out: Any) -> int:
        """Allocate (once per capture session) the shared output buffer and
        return its row capacity.

        The buffer respects the runner's capacity override and the FIRST
        captured shape's actual output tensors. Capture order is largest-first, so
        the first shape's rows bound every later shape's. Sized off the
        output itself, never off shape_key.size, whose unit differs by
        runner (prefill: tokens; decode: requests) while output rows are
        always tokens.
        """
        rows = self._max_output_rows(warmup_out)
        capacity_rows = self._cuda_graph_runner.cuda_graph_output_capacity_rows(warmup_out)
        if capacity_rows is not None:
            rows = max(rows, capacity_rows)
        if self._shared_output_buffer is None:
            self._shared_output_buffer = self._alloc_full_buffer(warmup_out, rows)
            self._shared_output_rows = rows
        elif rows > self._shared_output_rows:
            raise RuntimeError(
                "BCG capture-order contract violated: a later captured shape "
                f"produced {rows} output rows but the shared output buffer "
                f"was sized for {self._shared_output_rows} by the first "
                "(largest) shape. Capture shapes largest-first."
            )
        return self._shared_output_rows

    @classmethod
    def _max_output_rows(cls, output: Any) -> int:
        """Largest leading-dim across every tensor in ``output`` (0 if none)."""
        if output is None:
            return 0
        if torch.is_tensor(output):
            return output.shape[0]
        if isinstance(output, PPProxyTensors):
            return max((t.shape[0] for t in output.tensors.values()), default=0)
        if isinstance(output, LogitsProcessorOutput):
            return max(
                (
                    getattr(output, name).shape[0]
                    for name in cls._logits_output_tensor_field_names(output)
                ),
                default=0,
            )
        if isinstance(output, (list, tuple)):
            return max(
                (cls._max_output_rows(o) for o in output if o is not None),
                default=0,
            )
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    @staticmethod
    def _logits_output_tensor_field_names(output: LogitsProcessorOutput) -> list[str]:
        """Field names holding a token-row tensor on this instance.

        Only these fields are graph-buffered (allocated once, refreshed by a
        captured ``.copy_()`` on every replay); every other field (the
        per-token logprob lists, ``customized_info``, ...) is carried through
        unchanged from whichever shape captured first. That mirrors the
        existing post-replay contract: decode/verify capture always runs with
        ``return_logprob=False`` (decode_cuda_graph_runner), and prefill
        routes ``return_logprob`` through a required eager tail instead of
        the graph (prefill_cuda_graph_runner's "requires an eager logits
        tail" assertion) -- so those fields are never live tensor data here.
        """
        return [
            field.name
            for field in dataclasses.fields(output)
            if torch.is_tensor(getattr(output, field.name))
        ]

    def _output_rows(self, output: Any, cap: int) -> int:
        """Leading-dim row count actually produced by the body, clamped to ``cap``.

        ``cap`` is the shared output buffer's row capacity (the first,
        largest captured shape's rows). A smaller shape, or a body that
        shards or prunes its output along dim 0, returns fewer rows.
        """
        runner_rows = self._cuda_graph_runner.cuda_graph_output_rows(output)
        if runner_rows is not None:
            return runner_rows
        if torch.is_tensor(output):
            return min(cap, output.shape[0])
        if isinstance(output, PPProxyTensors):
            rows = [t.shape[0] for t in output.tensors.values()]
            return min([cap, *rows])
        if isinstance(output, LogitsProcessorOutput):
            rows = [
                getattr(output, name).shape[0]
                for name in self._logits_output_tensor_field_names(output)
            ]
            return min([cap, *rows])
        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        """A same-structure buffer as ``output`` but with ``size`` leading rows."""
        if output is None:
            return None
        if torch.is_tensor(output):
            return output.new_empty((size, *output.shape[1:]))
        if isinstance(output, PPProxyTensors):
            return PPProxyTensors(
                {
                    key: t.new_empty((size, *t.shape[1:]))
                    for key, t in output.tensors.items()
                }
            )
        if isinstance(output, LogitsProcessorOutput):
            return dataclasses.replace(
                output,
                **{
                    name: getattr(output, name).new_empty(
                        (size, *getattr(output, name).shape[1:])
                    )
                    for name in self._logits_output_tensor_field_names(output)
                },
            )
        if isinstance(output, tuple):
            return tuple(self._alloc_full_buffer(o, size) for o in output)
        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, LogitsProcessorOutput):
            return dataclasses.replace(
                output,
                **{
                    name: getattr(output, name)[:num_tokens]
                    for name in self._logits_output_tensor_field_names(output)
                },
            )
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, LogitsProcessorOutput) and isinstance(
            output_buffer, LogitsProcessorOutput
        ):
            names = self._logits_output_tensor_field_names(output)
            buffer_names = self._logits_output_tensor_field_names(output_buffer)
            if set(names) != set(buffer_names):
                raise ValueError(
                    "BCG LogitsProcessorOutput tensor fields changed between "
                    f"capture sizes: {sorted(names)} != {sorted(buffer_names)}"
                )
            for name in names:
                self._copy_output_to_buffer(
                    getattr(output, name), getattr(output_buffer, name), num_tokens
                )
            return
        if (isinstance(output, list) and isinstance(output_buffer, list)) or (
            isinstance(output, tuple) and isinstance(output_buffer, tuple)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        with graph_pool_replay_scope():
            self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._capture_inputs.clear()
        self._pool = None
        self._shared_output_buffer = None
        self._shared_output_rows = 0
