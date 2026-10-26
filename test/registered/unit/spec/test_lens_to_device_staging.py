from __future__ import annotations

import time

import pytest
import torch

from sglang.srt.speculative.spec_utils import lens_to_device
from sglang.srt.utils.common import is_host_pinning_effective
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lens_to_device_does_not_block_on_a_deep_queue() -> None:
    """Staging a length list must not stall the host behind queued GPU work.

    Where the platform reports pinned host memory but backs it with managed
    memory (GPU confidential compute), a host-to-device copy from a "pinned"
    source blocks until the target stream drains -- hundreds of milliseconds
    behind a deep prefill batch. The pageable path taken there stages the
    source before returning, so enqueue stays independent of the backlog.
    """
    device = torch.device("cuda")
    lens = list(range(1, 65))
    expected = torch.tensor(lens, dtype=torch.int32)

    # Warm up allocators so the timed section measures only the copy.
    lens_to_device(lens=lens, device=device)
    torch.cuda.synchronize()

    torch.cuda._sleep(600_000_000)  # ~0.3 s of queued work at typical clocks
    start = time.perf_counter()
    out = lens_to_device(lens=lens, device=device)
    enqueue = time.perf_counter() - start
    torch.cuda.synchronize()
    drained = time.perf_counter() - start

    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    assert out.dtype == torch.int32 and out.is_cuda
    if is_host_pinning_effective():
        pytest.skip("host pinning is real here; the blocking path is not exercised")
    # Generous margin: the point is orders of magnitude, not a tight bound.
    assert enqueue < drained / 4, (enqueue, drained)
