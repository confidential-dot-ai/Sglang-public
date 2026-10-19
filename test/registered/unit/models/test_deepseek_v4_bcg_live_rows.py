"""BCG attention breaks consume the live token count, not padded bucket rows."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _low_ratio_source_projections,
)
from sglang.srt.model_executor.runner_backend_utils import tc_piecewise_cuda_graph
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("live", [0, 2, 4])
def test_projection_break_zeroes_padding(live):
    x = torch.arange(12).reshape(4, 3).float()
    project = Mock(side_effect=lambda values: (values + 1, values + 2))
    layer = SimpleNamespace(compressor=SimpleNamespace(project=project), indexer=None)
    bufs = {key: torch.full_like(x, -99) for key in ("kv", "score")}
    context = SimpleNamespace(
        forward_batch=SimpleNamespace(global_num_token_non_padded_cpu=live)
    )
    with patch.object(
        tc_piecewise_cuda_graph,
        "get_tc_piecewise_forward_context",
        return_value=context,
    ):
        _low_ratio_source_projections(
            layer, x, x, torch.arange(4), bufs, run_compressor=True, run_indexer=True
        )
    for name, offset in (("kv", 1), ("score", 2)):
        torch.testing.assert_close(bufs[name][:live], x[:live] + offset)
        assert not bufs[name][live:].count_nonzero()
    assert project.call_count == bool(live)


@pytest.mark.parametrize("live", [0, 2, 4])
def test_source_break_slices_to_live_rows(live):
    batch = SimpleNamespace(global_num_token_non_padded_cpu=live)
    backend = Mock()
    x = torch.ones(4, 3)
    with (
        patch.object(
            deepseek_v4,
            "get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(forward_batch=batch),
        ),
        patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
    ):
        deepseek_v4.deepseek_v4_low_ratio_sources(object(), x, x, torch.arange(4))
    assert backend.forward_low_ratio_sources.call_count == bool(live)
    if live:
        args = backend.forward_low_ratio_sources.call_args.kwargs
        assert args["x"].shape[0] == live
        assert args["q_lora"].shape[0] == live
        assert args["positions"].shape[0] == live


@pytest.mark.parametrize("eager_indexer", [False, True])
def test_prefill_graph_index_layers_follow_the_eager_indexer_switch(eager_indexer):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    req, pos = torch.arange(4), torch.arange(4)
    bufs = {"kv": torch.zeros(4, 3), "q": torch.zeros(4, 2, 3), "w": torch.zeros(4, 2)}
    backend = SimpleNamespace(
        prefill_graph_eager_indexer=eager_indexer,
        tail_source_kv_layer_id=None,
        forward_metadata=SimpleNamespace(
            low_ratio_req_indices=req, low_ratio_pos_i64=pos
        ),
        _low_ratio_in_prefill_graph=Mock(return_value=True),
        _source_projection_buffers=Mock(return_value=bufs),
        _low_ratio_compress_torch=Mock(),
        _low_ratio_index_topk_captured=Mock(),
    )
    layer = SimpleNamespace(
        layer_id=3, compress_ratio=1, compressor=object(), indexer=object()
    )
    x = q_lora = torch.ones(4, 3)
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, encoder_swa_replay=False)
    with (
        patch.object(deepseek_v4_backend, "dsa_use_prefill_cp", return_value=False),
        patch.object(deepseek_v4_backend, "_bcg_low_ratio_source_projections"),
        patch.object(deepseek_v4_backend, "_bcg_low_ratio_index_topk") as eager_break,
    ):
        DeepseekV4AttnBackend.forward_low_ratio_sources(
            backend,
            layer=layer,
            x=x,
            q_lora=q_lora,
            positions=pos,
            forward_batch=batch,
        )
    captured = backend._low_ratio_index_topk_captured
    assert eager_break.call_count == int(eager_indexer)
    assert captured.call_count == int(not eager_indexer)
    if eager_indexer:
        args = eager_break.call_args.args
        assert args[:2] == (backend, layer)
        assert [a is b for a, b in zip(args[2:], (x, q_lora, req, pos))] == [True] * 4
        assert args[6] is bufs["q"] and args[7] is bufs["w"]
    else:
        captured.assert_called_once_with(layer, bufs["q"], bufs["w"])


@pytest.mark.parametrize("live", [0, 2, 4])
def test_index_break_slices_to_live_rows(live):
    batch = SimpleNamespace(global_num_token_non_padded_cpu=live)
    backend = Mock()
    x = torch.ones(4, 3)
    q, w = torch.ones(4, 2, 3), torch.ones(4, 2)
    with patch.object(
        tc_piecewise_cuda_graph,
        "get_tc_piecewise_forward_context",
        return_value=SimpleNamespace(forward_batch=batch),
    ):
        deepseek_v4_backend._low_ratio_index_topk_live(
            backend, object(), x, x, torch.arange(4), torch.arange(4), q, w
        )
    assert backend._low_ratio_index_topk.call_count == bool(live)
    if live:
        call = backend._low_ratio_index_topk.call_args
        assert all(t.shape[0] == live for t in call.args[1:5])
        assert call.kwargs["q"].shape[0] == call.kwargs["weights"].shape[0] == live


def test_precomputed_projection_reaches_the_prefill_indexer():
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    rows = 4
    core = SimpleNamespace(
        sparse_page_indices=lambda ratio: torch.zeros(rows, 8, dtype=torch.int32),
        sparse_raw_indices=lambda ratio: torch.zeros(rows, 8, dtype=torch.int32),
        page_table=torch.zeros(rows, 2, dtype=torch.int32),
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            candidate_metadata=None, core_metadata=core, late_layer_tail=None
        ),
        prefill_candidates=Mock(),
        _publish_candidate_metadata=Mock(),
    )
    backend._low_ratio_prefill_reads_page_indices = lambda _batch: True
    backend._make_low_ratio_prefill_indexer_inputs = partial(
        DeepseekV4AttnBackend._make_low_ratio_prefill_indexer_inputs, backend
    )
    layer = SimpleNamespace(
        layer_id=3,
        compress_ratio=1,
        freqs_cis=torch.zeros(rows, 1),
        indexer=SimpleNamespace(is_candidate_source=True, uses_candidates=False),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens_cpu=torch.tensor([rows]),
        extend_seq_lens=torch.tensor([rows]),
        extend_seq_lens_cpu=[rows],
    )
    x = torch.ones(rows, 3)
    q, w = torch.ones(rows, 2, 3), torch.ones(rows, 2)
    DeepseekV4AttnBackend._low_ratio_index_topk(
        backend,
        layer,
        x,
        x,
        torch.zeros(rows, dtype=torch.int64),
        torch.arange(rows),
        batch,
        q=q,
        weights=w,
    )
    inputs = backend.prefill_candidates.publish_prefill.call_args.args[0]
    assert inputs.q is q and inputs.weights is w


@pytest.mark.parametrize("verify", [False, True])
def test_verify_hashing_does_not_require_prefill_context(verify):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = ForwardMode.TARGET_VERIFY if verify else ForwardMode.EXTEND
    batch = SimpleNamespace(forward_mode=mode)
    hasher = Mock(return_value=torch.zeros(6, 2, dtype=torch.int64))
    model = SimpleNamespace(
        engram_hasher=hasher,
        engram_prefetcher=None,
        late_layer_start=None,
        start_layer=0,
        end_layer=0,
    )
    ids = torch.arange(6)

    class _StopAfterHashing(Exception):
        pass

    with (
        patch.object(deepseek_v4, "is_cp_active", return_value=False),
        patch.object(deepseek_v4, "is_in_breakable_cuda_graph", return_value=True),
        patch.object(deepseek_v4, "bcg_deepseek_v4_engram_hash_ids") as prefill_hash,
        patch.object(
            deepseek_v4, "get_forward", return_value=SimpleNamespace(sp_active=False)
        ),
        patch.object(deepseek_v4.mhc, "HcState", side_effect=_StopAfterHashing),
        pytest.raises(_StopAfterHashing),
    ):
        deepseek_v4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
            model, ids, torch.ones(6, 2), batch, ids, ids, False, []
        )
    assert hasher.call_count == int(verify)
    assert prefill_hash.call_count == int(not verify)
    if verify:
        hasher.assert_called_once_with(ids, batch)


@pytest.mark.parametrize("verify", [False, True])
def test_verify_is_not_a_breakable_prefill(verify):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY if verify else ForwardMode.EXTEND
    )
    with patch.object(deepseek_v4, "is_in_breakable_cuda_graph", return_value=True):
        assert deepseek_v4._is_breakable_prefill(batch) is (not verify)
