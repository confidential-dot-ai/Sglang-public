"""DeepSeek-V4.1 index kernels at 1M-token contexts.

V4.1 serves up to 1,048,576 positions (65,536 x YaRN 16). Its ratio-1 layers
score every position, its ratio-2 layers half of them, so the indexer logits
and top-k run on rows up to 1M wide; the regular kernel tests stop at 256K.
These pin the selections and logits at 512K and 1M against torch, and check
that a key's logit does not depend on where in a long row it sits.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
pytestmark = pytest.mark.skipif(not _SM100, reason="the V4.1 index kernels need SM100")

ONE_M = 1 << 20
PAGE_SIZE = 64
HEAD_DIM = 128


def _plan(seq_lens):
    # The transform reads its plan before the PDL wait; keep them apart as in serving.
    metadata = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    return metadata


def _assert_topk_values(window, indices, k):
    indices = indices.cpu().long()
    window = window.cpu()
    assert ((indices >= 0) & (indices < window.numel())).all(), indices
    assert indices.unique().numel() == k
    expected = window.topk(k).values.sort().values
    actual = window[indices].sort().values
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [ONE_M // 2],
        [ONE_M],
        [ONE_M, ONE_M - 1],
        [ONE_M, 4096, ONE_M // 2 + 3, 17],
        [ONE_M // 2] * 40,  # more long rows than the persistent cluster pool
    ],
    ids=["512k", "1m", "1m-ragged-tail", "mixed", "512k-x40"],
)
@torch.inference_mode()
def test_topk_v2_paged_selects_exact_top_k_at_1m(seq_lens, k):
    g = torch.Generator(device="cuda").manual_seed(len(seq_lens) * 31 + k)
    width = (max(seq_lens) + 3) & ~3
    scores = torch.randn((len(seq_lens), width), generator=g, device="cuda")
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    out = torch.full((len(seq_lens), k), -2, dtype=torch.int32, device="cuda")

    topk_transform_paged_v2(scores, lens, None, out, PAGE_SIZE, _plan(lens))

    for row, length in enumerate(seq_lens):
        if length <= k:
            got = sorted(v for v in out[row].tolist() if v != -1)
            assert got == list(range(length))
        else:
            _assert_topk_values(scores[row, :length], out[row], k)


@torch.inference_mode()
def test_topk_v2_paged_page_transform_at_1m():
    """The page transform maps positions past 2^20 through the last table pages."""
    k = 2048
    seq = ONE_M
    g = torch.Generator(device="cuda").manual_seed(5)
    scores = torch.randn((2, seq), generator=g, device="cuda")
    # Favour the far tail so the selection lands in the highest pages.
    scores[:, -k:] += 100
    pages = seq // PAGE_SIZE
    perm = torch.randperm(2 * pages, generator=g, device="cuda").to(torch.int32)
    page_table = perm.view(2, pages).contiguous()
    lens = torch.full((2,), seq, dtype=torch.int32, device="cuda")
    out = torch.full((2, k), -2, dtype=torch.int32, device="cuda")

    topk_transform_paged_v2(scores, lens, page_table, out, PAGE_SIZE, _plan(lens))

    positions = torch.arange(seq - k, seq, device="cuda")
    for row in range(2):
        want = page_table[row, positions // PAGE_SIZE] * PAGE_SIZE + positions % PAGE_SIZE
        assert sorted(out[row].tolist()) == sorted(want.tolist())


@pytest.mark.parametrize("k", [512, 2048])
@torch.inference_mode()
def test_topk_v2_ragged_windows_far_into_a_flat_kv(k):
    """Prefill rows whose windows start past 1M columns, rebased by large offsets."""
    rows = [(0, ONE_M), (ONE_M - 4, ONE_M // 2), (3 * ONE_M // 2, 4099), (12, 1)]
    width = (max(s + n for s, n in rows) + 3) & ~3
    g = torch.Generator(device="cuda").manual_seed(k)
    scores = torch.full((len(rows), width), 1e3, device="cuda")
    for i, (start, length) in enumerate(rows):
        scores[i, start : start + length] = torch.randn(
            length, generator=g, device="cuda"
        )
    starts = torch.tensor([s for s, _ in rows], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([n for _, n in rows], dtype=torch.int32, device="cuda")
    # Flattened-KV rebase well past 2^24 but inside int32.
    offsets = starts + (ONE_M * 600)
    out = torch.full((len(rows), k), -2, dtype=torch.int32, device="cuda")

    topk_transform_ragged_v2(
        scores, lengths, out_offsets=offsets, out_indices=out, row_starts=starts
    )

    for i, (start, length) in enumerate(rows):
        picked = torch.tensor(
            [v - int(offsets[i]) for v in out[i].tolist() if v != -1]
        )
        window = scores[i, start : start + length]
        if length <= k:
            assert sorted(picked.tolist()) == list(range(length))
        else:
            _assert_topk_values(window, picked, k)


def _deep_gemm_case(seq: int, heads: int = 32, seed: int = 0):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks = seq // PAGE_SIZE
    q = torch.randn((1, 1, heads, HEAD_DIM), generator=g, device="cuda").bfloat16()
    k = torch.randn((seq, HEAD_DIM), generator=g, device="cuda").bfloat16()
    weights = torch.randn((1, heads), generator=g, device="cuda")
    q_fp4, q_sf = per_token_cast_to_fp4(
        q.view(-1, HEAD_DIM), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    k_fp4, k_sf = per_token_cast_to_fp4(
        k, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    cache = torch.empty((blocks, PAGE_SIZE * 68), dtype=torch.uint8, device="cuda")
    cache[:, : PAGE_SIZE * 64].view(blocks, PAGE_SIZE, 64).copy_(
        k_fp4.view(torch.uint8).view(blocks, PAGE_SIZE, 64)
    )
    cache[:, PAGE_SIZE * 64 :].view(blocks, PAGE_SIZE, 4).copy_(
        k_sf.contiguous().view(torch.uint8).view(blocks, PAGE_SIZE, 4)
    )
    q_pair = (q_fp4.view(1, 1, heads, HEAD_DIM // 2), q_sf.view(1, 1, heads))
    return deep_gemm, q_pair, cache.view(blocks, PAGE_SIZE, 1, 68), weights


def _paged_logits(deep_gemm, q, cache, weights, page_table):
    from sglang.kernels.ops.attention.dsv4.index_logits import (
        deep_gemm_fp4_paged_mqa_logits,
    )

    seq = page_table.shape[1] * PAGE_SIZE
    lens = torch.full((1, 1), seq, dtype=torch.int32, device="cuda")
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        lens, PAGE_SIZE, deep_gemm.get_num_sms()
    )
    return deep_gemm_fp4_paged_mqa_logits(
        q, cache, weights, lens, page_table, schedule, seq
    )[0, :seq]


@torch.inference_mode()
def test_deep_gemm_fp4_paged_logits_do_not_depend_on_row_position_at_1m():
    """A key's logit at column 1M - 1 equals its logit at column 0 of a short row."""
    deep_gemm, q, cache, weights = _deep_gemm_case(ONE_M)
    pages = ONE_M // PAGE_SIZE
    table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    full = _paged_logits(deep_gemm, q, cache, weights, table)
    assert torch.isfinite(full).all()

    short_pages = 1024  # 64K columns
    head = _paged_logits(deep_gemm, q, cache, weights, table[:, :short_pages])
    tail = _paged_logits(
        deep_gemm, q, cache, weights, table[:, -short_pages:].contiguous()
    )
    short = short_pages * PAGE_SIZE
    assert torch.equal(full[:short], head)
    assert torch.equal(full[-short:], tail)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
