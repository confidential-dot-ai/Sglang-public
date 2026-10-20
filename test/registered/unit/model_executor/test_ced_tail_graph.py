"""Captured CED attention must refresh every request-dependent input."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.runner.ced_tail_graph import (
    _CORE_INPUTS,
    CedLateGraph,
    _late_spans,
)


def test_replay_refreshes_owned_inputs_without_rebinding():
    def metadata(value):
        return SimpleNamespace(
            core_attn_metadata=SimpleNamespace(
                **{
                    name: torch.full((2, 3), value, dtype=torch.int32)
                    for name in _CORE_INPUTS
                }
            ),
            late_layer_tail=SimpleNamespace(swa_out_cache_loc=torch.full((2,), value)),
        )

    graph = CedLateGraph.__new__(CedLateGraph)
    graph.metadata = metadata(-1)
    graph.inputs = [torch.zeros(2, 3), None]
    graph.outputs = object()
    graph.graph = Mock()
    pointers = [
        getattr(graph.metadata.core_attn_metadata, n).data_ptr() for n in _CORE_INPUTS
    ]
    for value in (7, 13):
        source = metadata(value)
        assert (
            graph.replay(
                SimpleNamespace(forward_metadata=source),
                [torch.full((2, 3), float(value)), None],
            )
            is graph.outputs
        )
        assert graph.inputs[0].eq(value).all()
        assert graph.metadata.late_layer_tail.swa_out_cache_loc.eq(value).all()
        for name, pointer in zip(_CORE_INPUTS, pointers):
            dst = getattr(graph.metadata.core_attn_metadata, name)
            assert dst.data_ptr() == pointer
            assert dst.eq(value).all()
            assert dst.data_ptr() != getattr(source.core_attn_metadata, name).data_ptr()
    assert graph.graph.replay.call_count == 2


def test_constructor_owns_frozen_tail_write_locations(monkeypatch):
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.model_executor.runner import ced_tail_graph

    loc = torch.tensor([17, 18], dtype=torch.int32)
    tail = LateLayerTail(
        token_indices=torch.arange(2),
        positions=torch.arange(2),
        extend_seq_lens=torch.tensor([2]),
        extend_seq_lens_cpu=[2],
        swa_out_cache_loc=loc,
    )
    metadata = SimpleNamespace(
        core_attn_metadata=SimpleNamespace(**{name: None for name in _CORE_INPUTS}),
        late_layer_tail=tail,
    )
    backend = SimpleNamespace(forward_metadata=metadata)
    capture = SimpleNamespace(disable_dispose_tensor=False)
    monkeypatch.setattr(
        ced_tail_graph, "get_flags", lambda: SimpleNamespace(capture=capture)
    )
    for name in ("CUDAGraph", "Stream", "current_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, Mock())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **kw: nullcontext())
    from sglang.srt.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "graph_capture", lambda *a, **kw: nullcontext())
    graph = CedLateGraph(
        SimpleNamespace(late_layer_start=21, end_layer=21),
        backend,
        SimpleNamespace(num_token_non_padded=None),
        [torch.arange(2), torch.ones(2, 3), None, None, None],
        False,
    )
    assert backend.forward_metadata is metadata
    assert metadata.late_layer_tail is tail
    assert graph.metadata.late_layer_tail.swa_out_cache_loc.data_ptr() != loc.data_ptr()
    graph.metadata.late_layer_tail.swa_out_cache_loc.fill_(99)
    assert loc.tolist() == [17, 18]
    assert not capture.disable_dispose_tensor


def test_v41_late_index_sources_are_eager_boundaries():
    model = SimpleNamespace(
        late_layer_start=21,
        end_layer=40,
        layers=[
            SimpleNamespace(
                engram=None,
                self_attn=SimpleNamespace(
                    compressor=None, indexer=object() if i in (24, 28, 32, 36) else None
                ),
            )
            for i in range(40)
        ],
    )
    spans = list(_late_spans(model))
    assert [(a, b) for a, b, capture in spans if capture] == [
        (21, 24),
        (25, 28),
        (29, 32),
        (33, 36),
        (37, 40),
    ]
    assert [a for a, _, capture in spans if not capture] == [24, 28, 32, 36]
    assert [i for a, b, _ in spans for i in range(a, b)] == list(range(21, 40))


def test_late_source_break_uses_current_batch_and_publishes_owned_indices():
    graph = CedLateGraph.__new__(CedLateGraph)
    graph.break_sources = True
    graph.metadata = SimpleNamespace(
        core_attn_metadata=SimpleNamespace(
            **{name: torch.zeros(2, 3, dtype=torch.int32) for name in _CORE_INPUTS}
        )
    )
    captured_batch = object()
    seen = []
    backend = SimpleNamespace(forward_metadata=graph.metadata)

    def source(*, forward_batch):
        seen.append(forward_batch)
        assert backend.forward_metadata is graph.source_metadata
        for name in _CORE_INPUTS:
            getattr(backend.forward_metadata.core_attn_metadata, name).add_(1)

    backend.forward_low_ratio_sources = source
    graph.backend = backend
    for value in (7, 19):
        graph.live_batch = object()
        graph.live_metadata = SimpleNamespace(
            core_attn_metadata=SimpleNamespace(
                **{
                    name: torch.full((2, 3), value, dtype=torch.int32)
                    for name in _CORE_INPUTS
                }
            )
        )
        with graph.source_breaks():
            backend.forward_low_ratio_sources(forward_batch=captured_batch)
        assert seen[-1] is graph.live_batch
        assert backend.forward_metadata is graph.metadata
        assert backend.forward_low_ratio_sources is source
        for name in _CORE_INPUTS:
            assert getattr(graph.metadata.core_attn_metadata, name).eq(value + 1).all()
            assert getattr(graph.live_metadata.core_attn_metadata, name).eq(value).all()


def test_tail_start_source_layer_is_eager_and_precedes_the_full_graph():
    # Decoder SWA bounded replay starts the tail at the last kv_source layer,
    # which still owns its compressor (skipped inside the tail) and indexer.
    from sglang.srt.model_executor.runner.ced_tail_graph import _full_graph_start

    model = SimpleNamespace(
        late_layer_start=20,
        end_layer=40,
        layers=[
            SimpleNamespace(
                engram=None,
                self_attn=SimpleNamespace(
                    compressor=object() if i == 20 else None,
                    indexer=object() if i in (20, 24) else None,
                ),
            )
            for i in range(40)
        ],
    )
    spans = list(_late_spans(model))
    assert [a for a, _, capture in spans if not capture] == [20, 24]
    assert [(a, b) for a, b, capture in spans if capture] == [(21, 24), (25, 40)]
    assert [i for a, b, _ in spans for i in range(a, b)] == list(range(20, 40))
    assert _full_graph_start(model) == 21
