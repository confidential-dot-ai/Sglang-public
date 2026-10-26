"""Opt-in lazy graphs for the bounded CED decoder layers.

Only source-free spans of the 128-row/request late segment are captured.
Late index-source layers, the tail-start source layer, encoder compression and
Engram staging remain eager. Spans run through the model's span runner.
With encoder graphs enabled, all late dense work is captured with four live
indexer-operation breaks. Every request-dependent graph input has owned
storage, refreshed before replay; no allocator/KV ownership changes happen here.
"""

import copy
import logging
import time
from contextlib import contextmanager

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.runtime_context import get_flags

logger = logging.getLogger(__name__)
_LOADED_REVISION = envs.SGLANG_DSV41_CED_TAIL_GRAPH_REVISION.get()
# Synthetic tails sit this deep in the prompt so every compressed level's top-k
# window is full, as in long-prompt serving: the captured FlashMLA schedule
# follows those lengths. Arbitrary beyond that.
_PRECAPTURE_PREFIX_TOKENS = 8192

# These are the paged-attention inputs consumed by source-free late layers.
_CORE_INPUTS = (
    "swa_page_indices",
    "swa_topk_lengths",
    "c1_sparse_page_indices",
    "c1_sparse_topk_lengths",
    "c2_sparse_page_indices",
    "c2_sparse_topk_lengths",
    "c1_sparse_raw_indices",
    "c2_sparse_raw_indices",
)


class CedLateGraph:
    def __init__(
        self,
        model,
        backend,
        batch,
        inputs,
        capture_dspark,
        layer_range=None,
        break_sources=False,
        pool=None,
    ):
        start, end = layer_range or (model.late_layer_start, model.end_layer)
        self.break_sources = break_sources
        self.backend = backend
        self.live_batch = batch
        self.live_metadata = backend.forward_metadata
        self.inputs = [x.clone() if x is not None else None for x in inputs]
        self.metadata = copy.copy(backend.forward_metadata)
        self.metadata.core_attn_metadata = copy.copy(self.metadata.core_attn_metadata)
        for name in _CORE_INPUTS:
            value = getattr(self.metadata.core_attn_metadata, name)
            if value is not None:
                setattr(self.metadata.core_attn_metadata, name, value.clone())
        self.metadata.late_layer_tail = msgspec.structs.replace(
            self.metadata.late_layer_tail,
            swa_out_cache_loc=self.metadata.late_layer_tail.swa_out_cache_loc.clone(),
        )
        self.batch = copy.copy(batch)
        # All late rows are real; the full extend's count can be larger.
        n = inputs[1].shape[0]
        if batch.num_token_non_padded is not None:
            self.batch.num_token_non_padded = torch.full_like(
                batch.num_token_non_padded, n
            )
        self.batch.num_token_non_padded_cpu = n
        self.graph = BreakableCUDAGraph() if break_sources else torch.cuda.CUDAGraph()
        self.checked = False
        self.invalid = False
        live_metadata = backend.forward_metadata

        def forward(inputs_override=None, batch_override=None):
            positions, hidden, ids, ids_global, prev = (
                self.inputs if inputs_override is None else inputs_override
            )
            current_batch = self.batch if batch_override is None else batch_override
            aux = []
            for i in range(start, end):
                layer = model.layers[i]
                assert layer.engram is None
                assert layer.self_attn.compressor is None
                assert break_sources or layer.self_attn.indexer is None
            if start < end:
                hidden, prev = model._run_hc_layer_span(
                    start,
                    end,
                    positions,
                    hidden,
                    prev,
                    ids,
                    ids_global,
                    None,
                    current_batch,
                    capture_dspark,
                    aux,
                )
            return (hidden, prev, *aux)

        self.forward = forward

        # Cold captures are synchronized on both TP ranks by identical shape
        # dispatch. Repeated late-layer KV writes are idempotent: there are no
        # compressor states in this segment and attention has a fixed floor.
        old_dispose = get_flags().capture.disable_dispose_tensor
        try:
            get_flags().capture.disable_dispose_tensor = True
            backend.forward_metadata = self.metadata
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with self.source_breaks(), torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            capture = (
                BreakableCUDAGraphCapture(
                    self.graph,
                    pool=pool if pool is not None else torch.cuda.graph_pool_handle(),
                    stream=stream,
                    capture_error_mode="thread_local",
                )
                if break_sources
                else torch.cuda.graph(
                    self.graph, stream=stream, capture_error_mode="thread_local"
                )
            )
            from sglang.srt.distributed.parallel_state import graph_capture

            with graph_capture(stream=stream), self.source_breaks(), capture:
                self.outputs = forward()
            torch.cuda.current_stream().wait_stream(stream)
        finally:
            get_flags().capture.disable_dispose_tensor = old_dispose
            backend.forward_metadata = live_metadata
        self.layer_range = (start, end)
        logger.info(
            "Captured CED late-layer graph: rows=%d layers=[%d,%d) segments=%d",
            n,
            start,
            end,
            len(self.graph._segments) if break_sources else 1,
        )

    @contextmanager
    def source_breaks(self):
        if not self.break_sources:
            yield
            return
        backend = self.backend
        original = backend.forward_low_ratio_sources
        self.prepare_source_metadata()
        for name in _CORE_INPUTS:
            dst = getattr(self.metadata.core_attn_metadata, name)
            if dst is not None:
                dst.copy_(getattr(self.live_metadata.core_attn_metadata, name))

        @eager_on_graph(True)
        def source(**kwargs):
            # Only this operation needs live context lengths, candidate masks,
            # and request slots. Paged late attention uses owned static buffers.
            saved = backend.forward_metadata
            try:
                backend.forward_metadata = self.source_metadata
                kwargs["forward_batch"] = self.live_batch
                original(**kwargs)
            finally:
                backend.forward_metadata = saved

        # Do not enable the global breakable-prefill routing flag: late paged
        # attention is capturable. The decorator uses the capture ContextVar.
        backend.forward_low_ratio_sources = source
        try:
            yield
        finally:
            backend.forward_low_ratio_sources = original

    def prepare_source_metadata(self):
        # The caller's top-k is the layer-20 seed, not scratch for layers 24+.
        # Preserve it across warmup/capture/replay and the independent check.
        self.source_metadata = copy.copy(self.live_metadata)
        self.source_metadata.core_attn_metadata = self.metadata.core_attn_metadata
        # Fresh per replay: this replay's index-source breaks share one layout.
        self.source_metadata.prefill_layout_cache = {}

    def replay(self, backend, inputs, batch=None):
        if batch is not None:
            self.live_batch = batch
        self.live_metadata = backend.forward_metadata
        for dst, src in zip(self.inputs, inputs):
            if dst is not None:
                dst.copy_(src)
        live = backend.forward_metadata
        for name in _CORE_INPUTS:
            dst = getattr(self.metadata.core_attn_metadata, name)
            src = getattr(live.core_attn_metadata, name)
            if dst is not None:
                assert dst.shape == src.shape, name
                dst.copy_(src)
        self.metadata.late_layer_tail.swa_out_cache_loc.copy_(
            live.late_layer_tail.swa_out_cache_loc
        )
        if getattr(self, "break_sources", False):
            self.prepare_source_metadata()
        self.graph.replay()
        if envs.SGLANG_DSV41_CED_TAIL_GRAPH_CHECK.get() and (
            not self.checked or envs.SGLANG_DSV41_CED_TAIL_GRAPH_CHECK_EVERY.get()
        ):
            saved = backend.forward_metadata
            old_dispose = get_flags().capture.disable_dispose_tensor
            try:
                get_flags().capture.disable_dispose_tensor = True
                pool = backend.token_to_kv_pool
                pages = torch.unique(
                    self.metadata.late_layer_tail.swa_out_cache_loc.to(torch.int64)
                    // pool.swa_page_size
                )
                written_kv = [
                    pool.get_swa_raw_buffer(i)[pages] for i in range(*self.layer_range)
                ]
                if getattr(self, "break_sources", False):
                    expected = self._check_against_eager_noise(
                        backend, pool, pages, written_kv, inputs, batch
                    )
                else:
                    # Check the ordinary eager path, including live metadata and
                    # batch fields. Checking the captured copies can hide stale
                    # request state even when graph and copy-based eager agree.
                    expected = self.forward(inputs, batch)
                    self._check_exact(pool, pages, written_kv, expected, inputs)
                from sglang.srt.model_executor.runner.ced_graph_check import (
                    agree_check_failure,
                )

                self.invalid = agree_check_failure(self.invalid)
                if self.invalid:
                    return expected
                logger.info(
                    "CED late graph check passed: rows=%d layers=%s outputs=%d",
                    inputs[1].shape[0],
                    self.layer_range,
                    len(self.outputs),
                )
                self.checked = True
            finally:
                get_flags().capture.disable_dispose_tensor = old_dispose
                backend.forward_metadata = saved
        return self.outputs


    def _check_exact(self, pool, pages, written_kv, expected, inputs):
        from sglang.srt.model_executor.runner.ced_graph_check import bitwise_equal

        for layer_id, actual in zip(range(*self.layer_range), written_kv):
            reference = pool.get_swa_raw_buffer(layer_id)[pages]
            if not torch.equal(actual, reference):
                logger.error(
                    "CED late graph KV check FAILED: layer=%d bytes=%d; disabling shape",
                    layer_id,
                    (actual != reference).sum().item(),
                )
                self.invalid = True
        for index, (actual, reference) in enumerate(zip(self.outputs, expected)):
            if not bitwise_equal(actual, reference):
                delta = (actual.float() - reference.float()).abs().max().item()
                logger.error(
                    "CED late graph exact check FAILED: output=%d rows=%d max_abs=%s; disabling shape",
                    index,
                    inputs[1].shape[0],
                    delta,
                )
                self.invalid = True

    def _check_against_eager_noise(
        self, backend, pool, pages, written_kv, inputs, batch
    ):
        """Index-source layers pick a bf16 top-k whose ties break differently run
        to run, so eager does not reproduce itself bitwise. Run eager twice from
        the same layer-20 seed and accept the graph while its distance to the
        first eager run stays within a margin of the second eager run's."""
        live = backend.forward_metadata.core_attn_metadata
        seed = {
            name: getattr(live, name).clone()
            for name in _CORE_INPUTS
            if getattr(live, name) is not None
        }

        def eager():
            for name, value in seed.items():
                getattr(live, name).copy_(value)
            outputs = self.forward(inputs, batch)
            kv = [pool.get_swa_raw_buffer(i)[pages] for i in range(*self.layer_range)]
            return outputs, kv

        first, first_kv = eager()
        second, second_kv = eager()
        graph_kv = sum((a != b).sum().item() for a, b in zip(written_kv, first_kv))
        eager_kv = sum((a != b).sum().item() for a, b in zip(second_kv, first_kv))
        # Margins from 86 replays under c24 (2026-10-20): graph/eager ratios had
        # median ~1.0, KV max 1.02x and outputs max 8.4x (a one-sample spread is
        # heavy-tailed); stale graph state measured 100-1000x.
        if graph_kv > 1.5 * eager_kv:
            logger.error(
                "CED late graph KV noise check FAILED: rows=%d graph-vs-eager bytes=%d "
                "eager-vs-eager bytes=%d; disabling shape",
                inputs[1].shape[0],
                graph_kv,
                eager_kv,
            )
            self.invalid = True
        ratios = []
        for index, (graph_out, a, b) in enumerate(zip(self.outputs, first, second)):
            graph_err = (graph_out.float() - a.float()).abs().mean().item()
            eager_err = (b.float() - a.float()).abs().mean().item()
            ratios.append(graph_err / eager_err if eager_err else float(graph_err > 0))
            if graph_err > 10 * eager_err:
                logger.error(
                    "CED late graph noise check FAILED: output=%d rows=%d mean_abs "
                    "graph-vs-eager=%.4g eager-vs-eager=%.4g; disabling shape",
                    index,
                    inputs[1].shape[0],
                    graph_err,
                    eager_err,
                )
                self.invalid = True
        if not self.invalid:
            logger.info(
                "CED late graph noise check passed: rows=%d KV bytes graph=%d eager=%d",
                inputs[1].shape[0],
                graph_kv,
                eager_kv,
            )
        logger.info(
            "CED late graph noise ratios: rows=%d kv=%.3f outputs=%s",
            inputs[1].shape[0],
            graph_kv / eager_kv if eager_kv else float(graph_kv > 0),
            " ".join("%.3f" % r for r in ratios),
        )
        return second

def _late_spans(model):
    """Index-source layers refresh top-k using live variable-length metadata.

    The tail starts at the last kv_source layer (its compressed KV is written
    for every row before the tail), so that layer is an eager span too.
    """
    start = model.late_layer_start
    for i in range(start, model.end_layer):
        layer = model.layers[i]
        assert layer.engram is None
        if (
            layer.self_attn.indexer is not None
            or layer.self_attn.compressor is not None
        ):
            if start < i:
                yield start, i, True
            yield i, i + 1, False
            start = i + 1
    if start < model.end_layer:
        yield start, model.end_layer, True


def run_ced_late_graph(model, backend, batch, inputs, capture_dspark):
    from sglang.srt.model_executor.runner.ced_graph_revision import reload_if_requested

    module = reload_if_requested(
        __name__, _LOADED_REVISION, model, ("_ced_late_graphs", "_ced_full_late_graphs")
    )
    if module is not None:
        return module.run_ced_late_graph(model, backend, batch, inputs, capture_dspark)
    from sglang.srt.model_executor.runner.ced_graph_cache import (
        capture_has_room,
        release_retired,
    )

    release_retired(model)
    positions, hidden, ids, ids_global, prev = inputs
    aux = []
    graph_start = _full_graph_start(model)
    if envs.SGLANG_DSV41_CED_TAIL_ENCODER_GRAPH.get() and graph_start < model.end_layer:
        # Layers owning a compressor run eagerly before the full graph.
        if graph_start > model.late_layer_start:
            hidden, prev = model._run_hc_layer_span(
                model.late_layer_start,
                graph_start,
                positions,
                hidden,
                prev,
                ids,
                ids_global,
                None,
                batch,
                capture_dspark,
                aux,
            )
            inputs = (positions, hidden, ids, ids_global, prev)
        cache = getattr(model, "_ced_full_late_graphs", None)
        if cache is None:
            cache = model._ced_full_late_graphs = {}
        key = (
            tuple(None if x is None else (tuple(x.shape), x.dtype) for x in inputs),
            capture_dspark,
        )
        if key not in cache:
            if len(cache) >= 16 or not capture_has_room():
                if graph_start > model.late_layer_start:
                    # The eager prefix already ran: finish eagerly.
                    return _finish_eagerly(model, batch, inputs, capture_dspark, aux)
                return None
            # Full late graphs replay one at a time on one stream and keep
            # their outputs allocated, so every shape shares one pool.
            pool = getattr(model, "_ced_full_late_pool", None)
            if pool is None:
                pool = model._ced_full_late_pool = torch.cuda.graph_pool_handle()
            cache[key] = CedLateGraph(
                model,
                backend,
                batch,
                inputs,
                capture_dspark,
                (graph_start, model.end_layer),
                break_sources=True,
                pool=pool,
            )
        if cache[key].invalid:
            if graph_start > model.late_layer_start:
                return _finish_eagerly(model, batch, inputs, capture_dspark, aux)
            return None
        outputs = cache[key].replay(backend, inputs, batch)
        return (*outputs[:2], *aux, *outputs[2:])
    cache = getattr(model, "_ced_late_graphs", None)
    if cache is None:
        cache = model._ced_late_graphs = {}
    key = (
        tuple(None if x is None else (tuple(x.shape), x.dtype) for x in inputs),
        capture_dspark,
    )
    # Bound whole shape families, not individual spans. No in-flight eviction.
    if key not in cache:
        if len(cache) >= 16 or not capture_has_room():
            return None
        cache[key] = {}
    graphs = cache[key]
    for start, end, capturable in _late_spans(model):
        if capturable:
            span_inputs = (positions, hidden, ids, ids_global, prev)
            if start not in graphs:
                graphs[start] = CedLateGraph(
                    model, backend, batch, span_inputs, capture_dspark, (start, end)
                )
            outputs = graphs[start].replay(backend, span_inputs, batch)
            hidden, prev = outputs[:2]
            aux.extend(outputs[2:])
        else:
            hidden, prev = model._run_hc_layer_span(
                start,
                end,
                positions,
                hidden,
                prev,
                ids,
                ids_global,
                None,
                batch,
                capture_dspark,
                aux,
            )
    return hidden, prev, *aux


def _full_graph_start(model):
    """First late layer after the last one that owns a compressor."""
    start = model.late_layer_start
    for i in range(model.late_layer_start, model.end_layer):
        if model.layers[i].self_attn.compressor is not None:
            start = i + 1
    return start


def _finish_eagerly(model, batch, inputs, capture_dspark, aux):
    positions, hidden, ids, ids_global, prev = inputs
    hidden, prev = model._run_hc_layer_span(
        _full_graph_start(model),
        model.end_layer,
        positions,
        hidden,
        prev,
        ids,
        ids_global,
        None,
        batch,
        capture_dspark,
        aux,
    )
    return hidden, prev, *aux


def precapture_ced_late_graphs(*, model_runner, eager_runner) -> None:
    """Capture the late-layer graphs for the configured decode tail sizes at
    startup, then refuse every capture while serving."""
    from sglang.srt.disaggregation.ced_boundary import (
        CedBoundaryInjection,
        ced_boundary_enabled,
    )
    from sglang.srt.disaggregation.utils import ced_tail_start
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.ced_graph_cache import freeze_captures
    from sglang.srt.runtime_context import get_disagg

    sizes = sorted(
        {int(n) for n in envs.SGLANG_DSV41_CED_TAIL_GRAPH_PRECAPTURE_REQS.get()}
    )
    if (
        not sizes
        or not envs.SGLANG_DSV41_CED_TAIL_LATE_GRAPH.get()
        or get_disagg().disaggregation_mode != "decode"
        or not ced_boundary_enabled()
    ):
        return
    config = model_runner.model_config.hf_text_config
    window = model_runner.sliding_window_size
    seq_len = _PRECAPTURE_PREFIX_TOKENS + window
    prefix_len = ced_tail_start(seq_len, window, model_runner.page_size)
    extend_len = seq_len - prefix_len
    buffers = eager_runner._alloc_dummy_decode_buffers(
        max(sizes), num_tokens_per_req=extend_len, allocate_logits_buffer=False
    )
    device = model_runner.device
    start = time.perf_counter()
    for num_reqs in sizes:
        # Boundary state as prefill hands it over: the tail goes straight to the
        # late layers. Values only feed the capture; writes land in slot 0.
        rows = num_reqs * window

        def attach_boundary(forward_batch, rows=rows):
            forward_batch.ced_boundary = CedBoundaryInjection(
                residual=0.02
                * torch.randn(
                    (rows, config.hc_mult, config.hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                pre=torch.full(
                    (rows, config.hc_mult),
                    1.0 / config.hc_mult,
                    dtype=torch.float32,
                    device=device,
                ),
            )

        try:
            eager_runner._dummy_run(
                num_reqs,
                forward_mode_override=ForwardMode.EXTEND,
                buffers=buffers,
                extend_num_tokens_per_req=extend_len,
                extend_prefix_len=prefix_len,
                prepare_forward_batch=attach_boundary,
            )
        except Exception:
            # Deterministic on every TP rank. Serving stays correct without the
            # remaining graphs: those tails run eagerly.
            logger.exception("CED pre-capture failed at %d requests", num_reqs)
            break
    torch.cuda.synchronize()
    freeze_captures()
    model = model_runner.model.model
    span_graphs = getattr(model, "_ced_late_graphs", None) or {}
    num_spans = sum(capturable for _, _, capturable in _late_spans(model))
    for key in [k for k, graphs in span_graphs.items() if len(graphs) < num_spans]:
        # Span graphs are built lazily on replay: a size that failed part-way
        # would capture its missing spans while serving.
        del span_graphs[key]
    captured = sorted(
        key[0][1][0][0]
        for cache in (
            getattr(model, "_ced_late_graphs", None) or {},
            getattr(model, "_ced_full_late_graphs", None) or {},
        )
        for key in cache
    )
    logger.info(
        "Pre-captured CED late-layer graphs for tail rows %s in %.1fs; "
        "no CED capture after startup",
        captured,
        time.perf_counter() - start,
    )
    if len(captured) < len(sizes):
        logger.warning(
            "CED pre-capture covered %d of %d tail sizes %s; the rest run eagerly",
            len(captured),
            len(sizes),
            sizes,
        )
