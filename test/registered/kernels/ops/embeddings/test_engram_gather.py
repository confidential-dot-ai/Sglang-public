"""The Triton Engram gather against the torch lookup it replaces, for the fp8
tables and the packed mxfp4 tables of nvidia/DeepSeek-V4.1-Flash-NVFP4."""

import pytest
import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.srt.layers.engram import dequantize_engram_rows
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)

DIM, BLOCK = 256, 32


def _table(rows, fp4, *, host):
    gen = torch.Generator().manual_seed(0)
    if fp4:
        weight = torch.randint(0, 256, (rows, DIM // 2), generator=gen).to(torch.uint8)
    else:
        weight = torch.randn(rows, DIM, generator=gen).to(torch.float8_e4m3fn)
    # Include byte 0 (2**-127, the e8m0 no-zero case) and extreme exponents.
    exps = torch.randint(0, 255, (rows, DIM // BLOCK), generator=gen)
    exps[0, 0] = 0
    scale = exps.to(torch.uint8).view(torch.float8_e8m0fnu)
    if host:
        return weight.pin_memory(), scale.pin_memory()
    return weight.cuda(), scale.cuda()


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("host", [False, True])
def test_gather_matches_torch_dequant_and_zeros_unowned_rows(fp4, host):
    rows, row_lo = 97, 1000
    weight, scale = _table(rows, fp4, host=host)
    ids = torch.tensor([1000, 1096, 1048, 999, 1097, 1000, 0], device="cuda")
    out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
    engram_gather(
        weight.data_ptr(),
        scale.data_ptr(),
        ids,
        out,
        DIM,
        BLOCK,
        row_lo=row_lo,
        row_hi=row_lo + rows,
        fp4=fp4,
    )
    local = (ids - row_lo).cpu()
    owned = (local >= 0) & (local < rows)
    want = dequantize_engram_rows(
        weight.cpu()[local.clamp(0, rows - 1)],
        scale.cpu()[local.clamp(0, rows - 1)],
        fp4=fp4,
    ).to(torch.bfloat16)
    want[~owned] = 0
    torch.testing.assert_close(out.cpu(), want, rtol=0, atol=0)
