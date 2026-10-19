from __future__ import annotations

import time

import pytest
import torch

from sglang.srt.utils.common import is_host_pinning_effective
from sglang.srt.utils.host_staging import copy_values_to_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")

_VALUES = list(range(1, 65))


def _blocked_ms(backend: str, out: torch.Tensor) -> float:
    copy_values_to_device(
        _VALUES,
        dtype=torch.int64,
        device=out.device,
        out=out,
        backend=backend,
        depth=8,
        growth_tokens=256,
    )
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        torch.cuda.synchronize()
        torch.cuda._sleep(600_000_000)  # ~0.3 s of queued work
        start = time.perf_counter()
        copy_values_to_device(
            _VALUES,
            dtype=torch.int64,
            device=out.device,
            out=out,
            backend=backend,
            depth=8,
            growth_tokens=256,
        )
        best = min(best, time.perf_counter() - start)
        torch.cuda.synchronize()
    return best * 1000.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pageable_staging_backend_does_not_block_and_stays_correct() -> None:
    """The pageable backend must stage without waiting on the queued backlog.

    Where the platform cannot supply real page-locked memory, a "pinned"
    staging buffer is managed-backed and its host-to-device copy blocks the
    caller until the target stream drains. The pageable backend exists for
    that case, so it must both copy correctly and enqueue independently of
    the backlog.
    """
    device = torch.device("cuda")
    expected = torch.tensor(_VALUES, dtype=torch.int64)

    out = torch.zeros(len(_VALUES), dtype=torch.int64, device=device)
    pageable_ms = _blocked_ms("pageable", out)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    if is_host_pinning_effective():
        pytest.skip("host pinning is real here; the blocking path is not exercised")
    out_default = torch.zeros(len(_VALUES), dtype=torch.int64, device=device)
    default_ms = _blocked_ms("default", out_default)
    torch.testing.assert_close(out_default.cpu(), expected, atol=0, rtol=0)
    # Orders of magnitude apart, not a tight bound.
    assert pageable_ms < default_ms / 10, (pageable_ms, default_ms)


def test_default_backend_resolves_to_pageable_without_real_pinning(monkeypatch) -> None:
    """'default' must not select pinned staging on a host that fakes pinning.

    'hardwire' is explicit and must be left alone; only the implicit 'default'
    is redirected, so a deployment that asked for doorbell tensors still gets
    them.
    """
    import types

    from sglang.srt.utils import common, host_staging

    class _Graph:
        cuda_graph_host_memory_backend = "default"
        cuda_graph_host_staging_pool_depth = 64
        cuda_graph_host_staging_growth_tokens = 256

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: types.SimpleNamespace(graph=_Graph()),
    )

    monkeypatch.setattr(common, "is_host_pinning_effective", lambda: False)
    assert host_staging._resolved_config()[0] == "pageable"

    monkeypatch.setattr(common, "is_host_pinning_effective", lambda: True)
    assert host_staging._resolved_config()[0] == "default"

    _Graph.cuda_graph_host_memory_backend = "hardwire"
    monkeypatch.setattr(common, "is_host_pinning_effective", lambda: False)
    assert host_staging._resolved_config()[0] == "hardwire"
