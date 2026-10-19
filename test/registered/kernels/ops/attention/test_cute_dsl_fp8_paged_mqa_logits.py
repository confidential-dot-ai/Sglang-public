# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa import cutedsl_paged_mqa_logits, pick_dsl_expand
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.paged_mqa import (
    BLOCK_KV,
    HEAD_DIM,
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)

register_cuda_ci(est_time=180, stage="nightly", runner_config="4-gpu-b200")


def _run_cutedsl_paged_mqa_logits(
    data,
    batch_size,
    next_n,
    num_heads,
    max_model_len,
    is_target_verify,
    emit_topk_histogram=False,
    topk_histogram_workspace=None,
):
    """Mirrors the CUTEDSL dispatch in
    sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged."""
    import deep_gemm

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if is_target_verify and next_n >= 2:
        dsl_expand_factor, dsl_atom = pick_dsl_expand(
            next_n,
            batch_size=batch_size,
            max_ctx=max_model_len,
            num_sms=num_sms,
            kernel_atoms=(1, 2, 3, 4),
            num_heads=num_heads,
        )
    else:
        dsl_expand_factor, dsl_atom = 1, 1

    context_lens = data["context_lens"]
    expanded_ctx = (
        context_lens.unsqueeze(-1)
        - next_n
        + torch.arange(1, next_n + 1, device=context_lens.device, dtype=torch.int32)
    ).flatten()
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        expanded_ctx.unsqueeze(-1), BLOCK_KV, num_sms
    )
    block_tables_expanded = data["block_table"].repeat_interleave(next_n, dim=0)

    return cutedsl_paged_mqa_logits(
        data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM),
        data["kv_fused"],
        data["weights"],
        context_lens,
        block_tables_expanded,
        schedule_metadata,
        max_model_len,
        q_offset=batch_size * next_n,
        B=batch_size,
        next_n=next_n,
        is_target_verify=is_target_verify,
        dsl_expand_factor=dsl_expand_factor,
        dsl_atom=dsl_atom,
        blocksize=BLOCK_KV,
        sm_count=num_sms,
        get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
        topk_lengths=expanded_ctx if emit_topk_histogram else None,
        topk_histogram_workspace=topk_histogram_workspace,
        num_init_tokens=4 if emit_topk_histogram else 0,
        num_local_tokens=8 if emit_topk_histogram else 0,
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("avg_ctx", [128, 1024, 4096, 16384])
def test_cutedsl_paged_mqa_logits(batch_size, next_n, num_heads, avg_ctx):
    max_model_len = max(avg_ctx * 2, 2048)
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )

    logits = _run_cutedsl_paged_mqa_logits(
        data,
        batch_size,
        next_n,
        num_heads,
        max_model_len,
        is_target_verify=next_n >= 2,
    )

    ref_logits = ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        BLOCK_KV,
    )
    assert_paged_mqa_matches_ref(
        logits, ref_logits, data["context_lens"], batch_size, next_n, max_model_len
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
def test_cutedsl_paged_mqa_emits_exact_topk_histogram():
    batch_size, next_n, num_heads, avg_ctx = 2, 2, 32, 16384
    max_model_len = avg_ctx * 2
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )
    workspace = torch.zeros(
        (batch_size * next_n, 1024),
        device="cuda",
        dtype=torch.int32,
    )
    logits, histogram = _run_cutedsl_paged_mqa_logits(
        data,
        batch_size,
        next_n,
        num_heads,
        max_model_len,
        is_target_verify=True,
        emit_topk_histogram=True,
        topk_histogram_workspace=workspace,
    )
    assert histogram.data_ptr() == workspace.data_ptr()
    lengths = (
        data["context_lens"].unsqueeze(-1)
        - next_n
        + torch.arange(1, next_n + 1, device="cuda", dtype=torch.int32)
    ).flatten()
    bits = logits.half().view(torch.int16).to(torch.int32) & 0xFFFF
    ordered = torch.where(
        (bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000
    )
    bins = ordered >> 6
    valid = torch.arange(max_model_len, device="cuda")[None, :] < lengths[:, None]
    expected = torch.zeros_like(histogram)
    expected.scatter_add_(1, bins, valid.to(torch.int32))
    torch.testing.assert_close(histogram, expected, rtol=0, atol=0)
    assert torch.isinf(logits[:, :4]).all()
    for row, length in enumerate(lengths.cpu().tolist()):
        assert torch.isinf(logits[row, length - 8 : length]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
