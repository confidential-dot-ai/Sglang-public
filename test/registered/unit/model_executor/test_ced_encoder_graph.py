"""CED graph padding must not retain the preceding request's inputs or KV slots."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.model_executor.runner.ced_encoder_graph import CedEncoderGraph


def test_smaller_replay_clears_padding_and_refreshes_live_row_mask():
    graph = CedEncoderGraph.__new__(CedEncoderGraph)
    graph.inputs = [torch.zeros(8, 2), None]
    graph.batch = SimpleNamespace(
        out_cache_loc=torch.zeros(8, dtype=torch.int64),
        num_token_non_padded=torch.zeros((), dtype=torch.int32),
    )
    graph.swa_loc = torch.zeros(8, dtype=torch.int32)
    pointers = [
        graph.inputs[0].data_ptr(),
        graph.batch.out_cache_loc.data_ptr(),
        graph.swa_loc.data_ptr(),
        graph.batch.num_token_non_padded.data_ptr(),
    ]
    for n, value in ((7, 13), (3, 29)):
        batch = SimpleNamespace(out_cache_loc=torch.full((n,), value))
        backend = SimpleNamespace(
            get_swa_out_cache_loc=Mock(return_value=torch.full((n,), value + 1)),
            tail_forward_metadata=SimpleNamespace(
                late_layer_tail=SimpleNamespace(
                    token_indices=torch.tensor([n - 2, n - 1])
                )
            ),
        )
        graph.refresh(backend, batch, [torch.full((n, 2), float(value)), None])
        assert graph.inputs[0][:n].eq(value).all()
        assert graph.inputs[0][n:].eq(0).all()
        assert graph.batch.out_cache_loc[:n].eq(value).all()
        assert graph.batch.out_cache_loc[n:].eq(0).all()
        assert graph.swa_loc[:n].eq(value + 1).all()
        assert graph.swa_loc[n:].eq(0).all()
        assert graph.batch.num_token_non_padded.item() == n
        assert graph.batch.num_token_non_padded_cpu == n
        assert graph.batch.global_num_token_non_padded_cpu == n
        assert pointers == [
            graph.inputs[0].data_ptr(),
            graph.batch.out_cache_loc.data_ptr(),
            graph.swa_loc.data_ptr(),
            graph.batch.num_token_non_padded.data_ptr(),
        ]
        assert graph.inputs[1] is None


def test_context_restores_backend_and_capture_flags_after_failure(monkeypatch):
    import pytest

    from sglang.srt.model_executor.runner import ced_encoder_graph as module
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    graph = CedEncoderGraph.__new__(CedEncoderGraph)
    graph.swa_loc = torch.zeros(8, dtype=torch.int32)
    graph.rows = 8
    graph.attention_layers = []
    graph.q_pad_buffer = torch.zeros(8, 4)
    original = torch.arange(3)
    core = SimpleNamespace(swa_out_cache_loc=original)
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(core_attn_metadata=core, q_pad_buffer=None)
    )
    flags = SimpleNamespace(capture=SimpleNamespace(disable_dispose_tensor=False))
    monkeypatch.setattr(module, "get_flags", lambda: flags)
    live = SimpleNamespace(extend_num_tokens=3)
    with pytest.raises(RuntimeError, match="test failure"):
        with graph.context(backend, live):
            assert core.swa_out_cache_loc is graph.swa_loc
            assert backend.forward_metadata.q_pad_buffer is graph.q_pad_buffer
            assert backend.ced_encoder_graph
            assert flags.capture.disable_dispose_tensor
            assert get_tc_piecewise_forward_context().forward_batch is live
            raise RuntimeError("test failure")
    assert core.swa_out_cache_loc is original
    assert backend.forward_metadata.q_pad_buffer is None
    assert not backend.ced_encoder_graph
    assert not flags.capture.disable_dispose_tensor
    assert get_tc_piecewise_forward_context() is None


def test_normal_tail_shapes_use_at_most_three_buckets_per_request_count():
    from sglang.srt.model_executor.runner.ced_encoder_graph import encoder_bucket_rows

    for requests in range(1, 9):
        buckets = set()
        for rows in range(requests * 128, requests * 383 + 1):
            padded = encoder_bucket_rows(requests, rows)
            assert rows <= padded <= requests * 384
            buckets.add(padded)
        assert len(buckets) == 3


def test_failed_shape_is_not_replayed_and_short_tails_remain_eager():
    from sglang.srt.model_executor.runner.ced_encoder_graph import run_ced_encoder_graph

    graph = SimpleNamespace(invalid=True, replay=Mock())
    model = SimpleNamespace(_ced_encoder_graphs={(1, 128, 128, False, False): graph})
    backend = SimpleNamespace(
        tail_forward_metadata=SimpleNamespace(
            late_layer_tail=SimpleNamespace(token_indices=torch.arange(128))
        )
    )
    batch = SimpleNamespace(batch_size=1)
    inputs = (torch.arange(128), torch.zeros(128, 2), None, None, None, None)
    assert run_ced_encoder_graph(model, backend, batch, inputs, False) is None
    graph.replay.assert_not_called()
    backend.tail_forward_metadata.late_layer_tail.token_indices = torch.arange(7)
    assert run_ced_encoder_graph(model, backend, batch, inputs, False) is None
    assert len(model._ced_encoder_graphs) == 1


def test_each_live_row_count_in_a_bucket_requires_its_own_check():
    graph = CedEncoderGraph.__new__(CedEncoderGraph)
    graph.checked_rows = set()
    graph.live_rows = 320
    assert not graph.checked
    graph.checked = True
    graph.live_rows = 352
    assert not graph.checked
    graph.checked = True
    graph.live_rows = 320
    assert graph.checked


def test_padding_requires_live_count_validation_even_without_diagnostics():
    from sglang.srt.environ import envs

    graph = CedEncoderGraph.__new__(CedEncoderGraph)
    graph.rows = 384
    graph.live_rows = 320
    graph.checked_rows = set()
    with envs.SGLANG_DSV41_CED_TAIL_GRAPH_CHECK.override(False):
        assert graph.needs_check
        graph.checked_rows.add(320)
        assert graph.needs_check
        graph.live_rows = 368
        assert graph.needs_check
        graph.live_rows = 384
        assert not graph.needs_check


def test_full_exact_cache_keeps_existing_graphs_and_declines_new_shapes(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.runner import ced_graph_cache
    from sglang.srt.model_executor.runner.ced_encoder_graph import run_ced_encoder_graph

    cache = {i: object() for i in range(256)}
    original = dict(cache)
    model = SimpleNamespace(_ced_encoder_graphs=cache)
    backend = SimpleNamespace(
        tail_forward_metadata=SimpleNamespace(
            late_layer_tail=SimpleNamespace(token_indices=torch.arange(128))
        )
    )
    guard = Mock()
    monkeypatch.setattr(ced_graph_cache, "capture_has_room", guard)
    inputs = (torch.arange(320), torch.zeros(320, 2), None, None, None, None)
    with envs.SGLANG_DSV41_CED_TAIL_ENCODER_EXACT_ROWS.override(True):
        assert (
            run_ced_encoder_graph(
                model, backend, SimpleNamespace(batch_size=1), inputs, False
            )
            is None
        )
    assert cache == original
    guard.assert_not_called()
