"""Lazy breakable graphs for CED encoder-tail replay.

Dense token work uses owned padded storage. Attention and compression consume
live request metadata at the existing prefill eager breaks. Hashing is done
once by the caller, so warmup/capture/check never recommit Engram history.
The captured layers are the model's own span runner over the encoder layers,
ending with the read the tail-start layer makes.
"""

import copy
import logging
from contextlib import contextmanager

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.runtime_context import get_flags

logger = logging.getLogger(__name__)
_LOADED_REVISION = envs.SGLANG_DSV41_CED_TAIL_GRAPH_REVISION.get()


def encoder_bucket_rows(num_requests, live_rows):
    """At most three packed-row buckets per normal CED request count."""
    quantum = num_requests * 128
    return ((live_rows + quantum - 1) // quantum) * quantum


class CedEncoderGraph:
    def __init__(self, model, backend, batch, inputs, capture_dspark, rows):
        self.model = model
        self.rows = rows
        self.inputs = [self._buffer(x) for x in inputs]
        self.batch = copy.copy(batch)
        self.batch.out_cache_loc = batch.out_cache_loc.new_zeros(rows)
        self.swa_loc = backend.get_swa_out_cache_loc(batch).new_zeros(rows)
        self.batch.num_token_non_padded = torch.zeros(
            (), dtype=torch.int32, device=inputs[0].device
        )
        self.batch.global_num_token_non_padded = self.batch.num_token_non_padded
        self.attention_layers = [layer.self_attn.attn_mqa for layer in model.layers]
        self.capture_dspark = capture_dspark
        self.checked_rows = set()
        self.failed_rows = set()
        self.live_rows = inputs[0].shape[0]
        self.invalid = False
        self.graph = BreakableCUDAGraph()
        self.q_pad_buffer = None
        self.pool = getattr(model, "_ced_encoder_pool", None)
        if self.pool is None:
            self.pool = model._ced_encoder_pool = torch.cuda.graph_pool_handle()
        self.refresh(backend, batch, inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with self.context(backend, batch), torch.cuda.stream(stream):
            self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        from sglang.srt.distributed.parallel_state import graph_capture

        with (
            graph_capture(stream=stream),
            self.context(backend, batch),
            BreakableCUDAGraphCapture(
                self.graph,
                pool=self.pool,
                stream=stream,
                capture_error_mode="thread_local",
            ),
        ):
            self.outputs = self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        logger.info(
            "Captured CED encoder graph: reqs=%d rows=%d segments=%d",
            batch.batch_size,
            rows,
            len(self.graph._segments),
        )

    def _buffer(self, x):
        if x is None:
            return None
        return x.new_zeros((self.rows, *x.shape[1:]))

    @property
    def checked(self):
        # A padded bucket can encounter multiple eager GEMM row counts.
        return (
            not envs.SGLANG_DSV41_CED_TAIL_GRAPH_CHECK_EVERY.get()
            and self.live_rows in self.checked_rows
        )

    @checked.setter
    def checked(self, value):
        if value:
            from sglang.srt.model_executor.runner.ced_graph_check import (
                agree_check_failure,
            )

            for layer, actual in getattr(self, "kv_check", []):
                reference = self.check_pool.get_swa_raw_buffer(layer)[self.check_pages]
                if not torch.equal(actual, reference):
                    logger.error(
                        "CED encoder KV check FAILED: layer=%d bytes=%d",
                        layer,
                        (actual != reference).sum().item(),
                    )
                    self.invalid = True
            self.kv_check = []
            self.invalid = agree_check_failure(getattr(self, "invalid", False))
            if self.invalid:
                self.failed_rows.add(self.live_rows)
            else:
                self.checked_rows.add(self.live_rows)
        else:
            self.checked_rows.discard(self.live_rows)

    @property
    def needs_check(self):
        # Padding changes routed-expert GEMM shapes in an input-dependent way.
        # A check on another prompt is insufficient: padded execution remains
        # diagnostic and always compares against the live eager result.
        return self.rows != self.live_rows or (
            envs.SGLANG_DSV41_CED_TAIL_GRAPH_CHECK.get() and not self.checked
        )

    def refresh(self, backend, batch, inputs):
        n = inputs[0].shape[0]
        self.live_rows = n
        for dst, src in zip(self.inputs, inputs):
            if dst is not None:
                dst[:n].copy_(src)
                dst[n:].zero_()
        self.batch.out_cache_loc[:n].copy_(batch.out_cache_loc)
        self.batch.out_cache_loc[n:].zero_()
        self.swa_loc[:n].copy_(backend.get_swa_out_cache_loc(batch))
        self.swa_loc[n:].zero_()  # reserved dummy KV slot, never a live request
        self.batch.num_token_non_padded.fill_(n)
        self.batch.num_token_non_padded_cpu = n
        self.batch.global_num_token_non_padded_cpu = n

    @contextmanager
    def context(self, backend, live_batch):
        old_dispose = get_flags().capture.disable_dispose_tensor
        old_graph = getattr(backend, "ced_encoder_graph", False)
        core = backend.forward_metadata.core_attn_metadata
        old_loc = core.swa_out_cache_loc
        metadata = backend.forward_metadata
        old_q_pad = metadata.q_pad_buffer
        try:
            get_flags().capture.disable_dispose_tensor = True
            backend.ced_encoder_graph = True
            core.swa_out_cache_loc = self.swa_loc
            metadata.q_pad_buffer = self.q_pad_buffer
            with (
                enable_breakable_cuda_graph(),
                set_tc_piecewise_forward_context(
                    live_batch,
                    self.attention_layers,
                    None,
                    None,
                    None,
                    num_tokens=self.rows,
                    raw_num_tokens=live_batch.extend_num_tokens,
                ),
            ):
                yield
        finally:
            # Warmup allocates this outside the graph pool. Keep a strong
            # reference: graph nodes retain addresses, not Tensor lifetimes.
            self.q_pad_buffer = metadata.q_pad_buffer
            metadata.q_pad_buffer = old_q_pad
            core.swa_out_cache_loc = old_loc
            backend.ced_encoder_graph = old_graph
            get_flags().capture.disable_dispose_tensor = old_dispose

    def forward(self):
        positions, hidden, ids, ids_global, prev, hashes = self.inputs
        aux = []
        model = self.model
        hidden, prev = model._run_hc_layer_span(
            model.start_layer,
            model.late_layer_start,
            positions,
            hidden,
            prev,
            ids,
            ids_global,
            hashes,
            self.batch,
            self.capture_dspark,
            aux,
            next_layer=model.layers[model.late_layer_start],
        )
        return (hidden, prev, *aux)

    def replay(self, backend, batch, inputs):
        self.refresh(backend, batch, inputs)
        with self.context(backend, batch):
            self.graph.replay()
        if self.needs_check:
            self.check_pool = backend.token_to_kv_pool
            self.check_pages = torch.unique(
                self.swa_loc[: self.live_rows].to(torch.int64)
                // self.check_pool.swa_page_size
            )
            self.kv_check = [
                (i, self.check_pool.get_swa_raw_buffer(i)[self.check_pages])
                for i in range(self.model.start_layer, self.model.late_layer_start)
            ]
        n = inputs[0].shape[0]
        return tuple(x[:n] if x is not None else None for x in self.outputs)


def run_ced_encoder_graph(model, backend, batch, inputs, capture_dspark):
    from sglang.srt.model_executor.runner.ced_graph_revision import reload_if_requested

    module = reload_if_requested(
        __name__, _LOADED_REVISION, model, ("_ced_encoder_graphs", "_ced_encoder_pool")
    )
    if module is not None:
        return module.run_ced_encoder_graph(
            model, backend, batch, inputs, capture_dspark
        )
    from sglang.srt.model_executor.runner.ced_graph_cache import (
        capture_has_room,
        release_retired,
    )

    release_retired(model)
    # Pack live rows first, then pad to 128/256/384 * request count. This
    # bounds shape proliferation while eager breaks slice to the live count.
    rows = encoder_bucket_rows(batch.batch_size, inputs[0].shape[0])
    if envs.SGLANG_DSV41_CED_TAIL_ENCODER_EXACT_ROWS.get():
        rows = inputs[0].shape[0]
    tail_rows = backend.tail_forward_metadata.late_layer_tail.token_indices.numel()
    if tail_rows != batch.batch_size * 128:
        return None
    key = (batch.batch_size, rows, tail_rows, capture_dspark, inputs[4] is not None)
    cache = getattr(model, "_ced_encoder_graphs", None)
    if cache is None:
        cache = model._ced_encoder_graphs = {}

    def get_graph(key, rows):
        if key not in cache:
            limit = 256 if envs.SGLANG_DSV41_CED_TAIL_ENCODER_EXACT_ROWS.get() else 24
            if len(cache) >= limit:
                return None
            if not capture_has_room():
                return None
            cache[key] = CedEncoderGraph(
                model, backend, batch, inputs, capture_dspark, rows
            )
        return cache[key]

    graph = get_graph(key, rows)
    if graph is None:
        return None
    n = inputs[0].shape[0]
    failed = getattr(graph, "failed_rows", set())
    if graph.invalid and not failed:
        return None
    if n in failed:
        if rows == n:
            return None
        # Padding can select a different GEMM algorithm. Quarantine that live
        # row count and use its exact graph, retaining the bucket for others.
        key = (batch.batch_size, n, tail_rows, capture_dspark, inputs[4] is not None)
        graph = get_graph(key, n)
        if graph is None or graph.invalid:
            return None
    graph.invalid = False
    actual = graph.replay(backend, batch, inputs)
    if graph.needs_check:
        # The model's ordinary layer loop supplies the independent eager oracle.
        return actual, graph
    return actual, None
