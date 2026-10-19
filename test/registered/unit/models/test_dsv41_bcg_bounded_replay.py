"""Tail replay preserves request row order and refreshes capture bridge buffers."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.arg_groups import deepseek_v4_hook as hook
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_utils import dsv41_bounded_replay as replay
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_tail(indices, lengths, bucket, contiguous_start=None):
    n = len(indices)
    return LateLayerTail(
        token_indices=torch.tensor(indices),
        positions=replay.pad_rows(torch.tensor(indices), bucket),
        extend_seq_lens=torch.tensor(lengths),
        extend_seq_lens_cpu=lengths,
        swa_out_cache_loc=replay.pad_rows(torch.arange(1, n + 1), bucket),
        contiguous_start=contiguous_start,
        pad_rows=bucket - n,
    )


@pytest.mark.parametrize("chunk,requests", [(64, 8), (256, 1), (4096, 7), (1000, 32)])
def test_tail_bucket_covers_live_rows_without_exceeding_chunk(chunk, requests):
    buckets = replay.tail_buckets(chunk, requests)
    assert buckets[-1] == min(chunk, requests * 128)
    for live in range(1, chunk + 1):
        lengths = [live // requests + (i < live % requests) for i in range(requests)]
        selected = replay.select_tail_bucket(
            chunk_bucket=chunk, max_requests=requests, extend_lens=lengths
        )
        actual = sum(min(n, 128) for n in lengths)
        assert actual <= selected <= chunk
        assert all(b < actual for b in buckets if b < selected)


def test_contiguous_tail_does_not_include_full_bucket_padding():
    tail = make_tail([3, 4], [2], 4, contiguous_start=3)
    full = torch.arange(8)
    assert tail.real_rows(full).tolist() == [3, 4]
    assert tail.rows(full).tolist() == [3, 4, 0, 0]


def test_replayed_transition_refreshes_rows_and_restores_live_count():
    first = make_tail([2, 3, 6], [2, 1], 4)
    second = make_tail([0, 5], [1, 1], 4)
    metadata = SimpleNamespace(late_layer_tail=first)
    state = SimpleNamespace(metadata=metadata, saved=None, full_live_rows=None)
    backend = Mock(tail_forward_metadata=metadata, bcg_tail_state=state)
    batch = SimpleNamespace(global_num_token_non_padded_cpu=7)
    capture = Mock(cuda_graph=SimpleNamespace(_break_fns=[]), _barrier_fn=None)
    x = torch.arange(8 * 4 * 3).reshape(8, 4, 3).float()
    prev = torch.arange(8 * 4).reshape(8, 4).float()
    with (
        patch.object(replay, "get_attn_backend", return_value=backend),
        patch.object(
            replay,
            "get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(forward_batch=batch),
        ),
        patch.object(bcg, "_weak_ref_if_tensor", side_effect=lambda x: x),
    ):
        token = bcg._current_capture_var.set(capture)
        try:
            # DSpark sources before the transition must use exactly the same map.
            aux = replay.gather_tail(x)[0]
            positions, hidden, coefficients, absent = replay.enter_tail(x, prev, None)
            assert batch.global_num_token_non_padded_cpu == 3
            replay.exit_tail()
            assert batch.global_num_token_non_padded_cpu == 7
        finally:
            bcg._current_capture_var.reset(token)
        addresses = hidden.data_ptr(), coefficients.data_ptr(), aux.data_ptr()
        metadata.late_layer_tail = second
        batch.global_num_token_non_padded_cpu = 6
        capture.cuda_graph._break_fns[0]()
        capture.cuda_graph._break_fns[1]()
        assert batch.global_num_token_non_padded_cpu == 2
        torch.testing.assert_close(hidden, second.rows(x))
        torch.testing.assert_close(coefficients, second.rows(prev))
        torch.testing.assert_close(aux.mean(1), second.rows(x).mean(1))
        assert positions.tolist() == [0, 5, 0, 0]
        assert absent is None
        assert addresses == (hidden.data_ptr(), coefficients.data_ptr(), aux.data_ptr())
        scattered = replay.scatter_tail(hidden.mean(1), 8)
        torch.testing.assert_close(
            scattered[second.token_indices], x[second.token_indices].mean(1)
        )
        capture.cuda_graph._break_fns[2]()
        assert batch.global_num_token_non_padded_cpu == 6
        assert state.saved is None


def test_tail_metadata_refresh_preserves_captured_cache_write_address():
    first = make_tail([2, 3, 6], [2, 1], 4)
    second = make_tail([5], [1], 4)
    original = Mock(
        late_layer_tail=first,
        core_attn_metadata=SimpleNamespace(swa_out_cache_loc=first.swa_out_cache_loc),
    )
    fresh = Mock(late_layer_tail=second)
    backend = Mock()
    backend._build_late_layer_tail_metadata.side_effect = [original, fresh]
    state = replay.BoundedReplayGraphState(backend=backend, batch="capture", bucket=4)
    address = first.swa_out_cache_loc.data_ptr()
    state.prepare(backend=backend, batch="live")
    assert state.metadata.late_layer_tail.swa_out_cache_loc.data_ptr() == address
    assert state.metadata.late_layer_tail.swa_out_cache_loc.tolist() == [1, 0, 0, 0]
    assert state.metadata.late_layer_tail.token_indices.tolist() == [5]
    assert backend.tail_forward_metadata is original
    original.refresh_for_breakable_cuda_graph_replay_.assert_called_once_with(fresh)


@pytest.mark.parametrize(
    "graph,cp,dp,allowed",
    [
        ("disabled", 1, False, True),
        ("breakable", 1, False, True),
        ("full", 1, False, False),
        ("tc_piecewise", 1, False, False),
        ("breakable", 2, False, False),
        ("breakable", 1, True, False),
    ],
)
def test_hook_only_allows_supported_graph_tail_configuration(graph, cp, dp, allowed):
    cfg = SimpleNamespace(
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=True,
        dsv4_attn_backend="auto",
        speculative_algorithm=None,
        enable_hisparse=False,
        enable_two_batch_overlap=False,
        pp_size=1,
        disaggregation_mode="null",
        enable_dp_attention=dp,
        attn_cp_size=cp,
        dcp_size=1,
        enable_prefill_cp=False,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=graph, max_seq_len=16384)
        ),
    )
    with (
        patch.object(hook, "resolving_view", return_value=cfg),
        patch.object(
            hook,
            "model_config_of",
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
        ),
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=False,
        ),
    ):
        if allowed:
            hook.validate_deepseek_v41_features(cfg)
        else:
            with pytest.raises(ValueError, match="bounded-replay"):
                hook.validate_deepseek_v41_features(cfg)


@pytest.mark.parametrize(
    "max_seq_len,eager_indexer,cp,declared",
    [
        (None, False, 1, 16384),
        (None, True, 1, None),
        (None, True, 2, 16384),
        (8192, False, 1, None),
        (32768, True, 1, None),
        (32768, False, 1, ValueError),
    ],
)
def test_hook_lifts_the_prefill_graph_cap_only_for_the_eager_indexer(
    max_seq_len, eager_indexer, cp, declared
):
    cfg = SimpleNamespace(
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=False,
        dsv4_attn_backend="auto",
        speculative_algorithm=None,
        enable_hisparse=False,
        enable_two_batch_overlap=False,
        pp_size=1,
        disaggregation_mode="null",
        enable_dp_attention=False,
        attn_cp_size=cp,
        dcp_size=1,
        enable_prefill_cp=False,
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend="breakable", max_seq_len=max_seq_len)
        ),
    )
    declare = Mock()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41"),
        hf_text_config=SimpleNamespace(
            candidate_topk_blocks=2048, candidate_block_size=8
        ),
    )
    with (
        envs.SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER.override(eager_indexer),
        patch.object(hook, "resolving_view", return_value=cfg),
        patch.object(hook, "model_config_of", return_value=model_config),
        patch.object(hook, "declare_resolution", declare),
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=False,
        ),
    ):
        if declared is ValueError:
            with pytest.raises(ValueError, match="candidate window"):
                hook.validate_deepseek_v41_features(cfg)
            return
        hook.validate_deepseek_v41_features(cfg)
    if declared is None:
        declare.assert_not_called()
    else:
        resolved = declare.call_args.kwargs["cuda_graph_config"]
        assert resolved.prefill.max_seq_len == declared


def _validate_dspark_pd(backend, speculative_algorithm="DSPARK"):
    cfg = SimpleNamespace(
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=False,
        dsv4_attn_backend="auto",
        speculative_algorithm=speculative_algorithm,
        enable_hisparse=False,
        enable_two_batch_overlap=False,
        pp_size=1,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend=backend,
        dp_size=1,
        enable_dp_attention=False,
        attn_cp_size=1,
        dcp_size=1,
        enable_prefill_cp=False,
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend="breakable", max_seq_len=8192)
        ),
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41"),
        hf_text_config=SimpleNamespace(
            candidate_topk_blocks=2048, candidate_block_size=8
        ),
    )
    with (
        envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
        patch.object(hook, "resolving_view", return_value=cfg),
        patch.object(hook, "model_config_of", return_value=model_config),
        patch.object(hook, "declare_resolution", Mock()),
        patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=False,
        ),
    ):
        hook.validate_deepseek_v41_features(cfg)


def test_dspark_pd_accepts_mooncake_and_rejects_other_backends():
    _validate_dspark_pd("mooncake")
    with pytest.raises(ValueError, match="static verify, Mooncake"):
        _validate_dspark_pd("nixl")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
