import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_bf16_small
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _expected(scores, lens, table, page_size, topk):
    """Best ``topk`` of each row, ties to the lowest indices, ascending, mapped
    through ``table``; ``-1`` past the row."""
    columns = torch.arange(scores.shape[1], device=scores.device)
    masked = scores.float().masked_fill(columns >= lens[:, None], -torch.inf)
    order = masked.sort(dim=-1, descending=True, stable=True).indices[:, :topk]
    positions = order.sort(dim=-1).values
    positions = torch.where(
        torch.arange(topk, device=scores.device) < lens[:, None], positions, -1
    )
    clamped = positions.clamp_min(0)
    slots = table.gather(-1, clamped // page_size) * page_size + clamped % page_size
    return torch.where(positions >= 0, slots, -1).to(torch.int32)


@pytest.mark.skipif(is_xpu(), reason="topk_bf16_small is a CUDA JIT kernel")
@pytest.mark.parametrize("topk", [512, 2048])
def test_selection_is_ascending_with_lowest_index_ties(topk):
    """A row past topk scores selects the same elements in ascending order on every
    run, ties at the k-th value going to the lowest indices: the sparse attention
    rounds in selection order, and bf16 scores tie often."""
    torch.manual_seed(0)
    page_size, width, reps = 8, 16384, 20
    lens = torch.tensor(
        [1, topk - 1, topk, topk + 1, 1155, 4000, 8191, 16384] * 4,
        dtype=torch.int32,
        device="cuda",
    )
    rows = lens.shape[0]
    scores = torch.randn(rows, width, device="cuda") * 3
    # coarse values: most rows tie at the k-th value, one ties everywhere
    scores[rows // 2 :] = scores[rows // 2 :].round()
    scores[-1] = 1.0
    scores = scores.to(torch.bfloat16)
    table = torch.stack(
        [torch.randperm(1 << 16, device="cuda")[: width // page_size] for _ in range(rows)]
    ).to(torch.int32)

    expected = _expected(scores, lens, table, page_size, topk)
    for _ in range(reps):
        out = torch.full((rows, topk), -7, dtype=torch.int32, device="cuda")
        topk_transform_bf16_small(scores, lens, table, out, page_size)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
