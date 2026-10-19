"""Is the V4.1 indexer's run-to-run selection drift at long context only tie-breaking?

The indexer picks the top 512 (2048 for the candidate source) of fp4 logits per
query row. At 256K-1M widths, repeated identical runs were seen to select
different positions. That is benign if (a) the logits are bitwise reproducible
and (b) every position that changes scores exactly the k-th value, i.e. only
equal scores at the selection floor are broken differently. It is a real bug if
the logits differ between runs (a nondeterministic reduction) or if a position
scoring strictly above the k-th value is ever dropped.

These checks run the production kernels (DeepGEMM fp4 logits, flat for prefill
and paged for decode; top-k v2 paged and ragged) on fp4-quantized random data,
which has the coarse score granularity the real index K has.
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

PAGE_SIZE, HEAD_DIM, HEADS = 64, 128, 32
WIDTHS = [1 << 18, 1 << 19, 1 << 20]
REPEATS = 4


def _fp4(x):
    from deep_gemm.utils import per_token_cast_to_fp4

    return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)


def _paged_case(width, rows, seed):
    """A paged fp4 index-K cache of ``width`` keys and ``rows`` decode queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks = width // PAGE_SIZE
    k_fp4, k_sf = _fp4(torch.randn((width, HEAD_DIM), generator=g, device="cuda").bfloat16())
    cache = torch.empty((blocks, PAGE_SIZE * 68), dtype=torch.uint8, device="cuda")
    cache[:, : PAGE_SIZE * 64].view(blocks, PAGE_SIZE, 64).copy_(
        k_fp4.view(torch.uint8).view(blocks, PAGE_SIZE, 64)
    )
    cache[:, PAGE_SIZE * 64 :].view(blocks, PAGE_SIZE, 4).copy_(
        k_sf.contiguous().view(torch.uint8).view(blocks, PAGE_SIZE, 4)
    )
    q = torch.randn((rows, HEADS, HEAD_DIM), generator=g, device="cuda").bfloat16()
    q_fp4, q_sf = _fp4(q.view(-1, HEAD_DIM))
    weights = torch.randn((rows, HEADS), generator=g, device="cuda")
    table = torch.arange(blocks, dtype=torch.int32, device="cuda")[None].expand(rows, -1)
    return (
        (q_fp4.view(rows, 1, HEADS, HEAD_DIM // 2), q_sf.view(rows, 1, HEADS)),
        cache.view(blocks, PAGE_SIZE, 1, 68),
        weights,
        table.contiguous(),
    )


def _paged_logits(case, width):
    import deep_gemm

    from sglang.kernels.ops.attention.dsv4.index_logits import (
        deep_gemm_fp4_paged_mqa_logits,
    )

    q, cache, weights, table = case
    lens = torch.full((table.shape[0], 1), width, dtype=torch.int32, device="cuda")
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        lens, PAGE_SIZE, deep_gemm.get_num_sms()
    )
    return deep_gemm_fp4_paged_mqa_logits(
        q, cache, weights, lens, table, schedule, width
    )[:, :width].clone()


def _flat_logits(width, rows, seed):
    """Prefill logits via DeepGEMM's flat fp4 path (the eager indexer's)."""
    from sglang.kernels.ops.attention.dsv4.index_logits import flat_index_logits_tiles

    g = torch.Generator(device="cuda").manual_seed(seed)
    k_fp4, k_sf = _fp4(torch.randn((width, HEAD_DIM), generator=g, device="cuda").bfloat16())
    # The pool's gathered layout: payload int8 [n, 64] and packed ue8m0 int32 [n].
    k = (k_fp4.view(torch.int8).view(width, HEAD_DIM // 2), k_sf.contiguous().view(torch.int32).view(width))
    q_fp4, q_sf = _fp4(
        torch.randn((rows * HEADS, HEAD_DIM), generator=g, device="cuda").bfloat16()
    )
    weights = torch.randn((rows, HEADS), generator=g, device="cuda")
    starts = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    tiles = flat_index_logits_tiles(
        q=(q_fp4.view(rows, HEADS, HEAD_DIM // 2), q_sf.view(rows, HEADS)),
        kv=k,
        weights=weights,
        starts=starts,
        lengths=lengths,
        context_lengths=[width],
        budget_bytes=1 << 30,
    )
    return torch.cat([t[:, :width] for _, t in tiles])


def _classify(logits, picks, k):
    """(above-floor positions missing, picks below the floor, ties at the floor)."""
    floor = logits.topk(k).values[-1]
    picked = torch.zeros_like(logits, dtype=torch.bool)
    picked[picks.long()] = True
    must = logits > floor
    return (
        int((must & ~picked).sum()),
        int((picked & (logits < floor)).sum()),
        int((logits == floor).sum()),
    )


@pytest.mark.parametrize("width", WIDTHS, ids=["256k", "512k", "1m"])
@torch.inference_mode()
def test_paged_logits_are_bitwise_reproducible(width):
    case = _paged_case(width, rows=6, seed=width)
    first = _paged_logits(case, width)
    for _ in range(REPEATS - 1):
        assert torch.equal(_paged_logits(case, width), first)


@pytest.mark.parametrize("width", WIDTHS, ids=["256k", "512k", "1m"])
@torch.inference_mode()
def test_flat_prefill_logits_are_bitwise_reproducible(width):
    first = _flat_logits(width, rows=4, seed=width + 1)
    for _ in range(REPEATS - 1):
        assert torch.equal(_flat_logits(width, rows=4, seed=width + 1), first)


@pytest.mark.parametrize("tied", [False, True], ids=["fp4", "tie-heavy"])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("width", WIDTHS, ids=["256k", "512k", "1m"])
@torch.inference_mode()
def test_topk_v2_drift_is_only_ties_at_the_floor(width, k, tied, record_property):
    """Repeated top-k on fixed logits: any position that changes between runs
    must score exactly the k-th value; nothing above it may ever be dropped.
    ``tie-heavy`` coarsens the fp4 logits to 1/8 steps so that hundreds of
    positions share the floor, as coarse real index scores can."""
    rows = 6
    logits = _paged_logits(_paged_case(width, rows, seed=width + k), width)
    if tied:
        logits = (logits * 8).round() / 8
    lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    metadata = plan_topk_v2(lens)
    torch.cuda.synchronize()
    runs = []
    for _ in range(REPEATS):
        out = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
        topk_transform_paged_v2(logits, lens, None, out, PAGE_SIZE, metadata)
        runs.append(out.clone())
    ragged = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
    topk_transform_ragged_v2(
        logits.contiguous(), lens, out_offsets=torch.zeros_like(lens), out_indices=ragged
    )
    runs.append(ragged)

    ties, changed = [], 0
    for row in range(rows):
        sets = [set(r[row].tolist()) for r in runs]
        changed += sum(s != sets[0] for s in sets[1:])
        for r in runs:
            missing, below, tie = _classify(logits[row], r[row], k)
            assert missing == 0, f"row {row}: dropped {missing} above-floor positions"
            assert below == 0, f"row {row}: picked {below} below-floor positions"
        ties.append(tie)
    record_property("floor_ties_per_row", ties)
    record_property("runs_with_a_different_set", changed)
    print(f"width={width} k={k} tied={tied} floor ties/row={ties} differing runs={changed}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
