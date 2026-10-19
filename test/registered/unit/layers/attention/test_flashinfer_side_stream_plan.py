from __future__ import annotations

import time

import pytest
import torch

from sglang.srt.layers.attention import flashinfer_backend as fib
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _FakeCompiledModule:
    """Stands in for the FlashInfer compiled module: CPU-writes the pinned
    buffer, then issues the pinned-to-device copy on the tvm-ffi env stream
    (which the C++ plan resolves via TVMFFIEnvGetStream)."""

    def __init__(self):
        self.seen_torch_streams = []
        self.seen_ffi_streams = []

    def plan(self, *args):
        import tvm_ffi

        int_workspace, pin_workspace = args[1], args[2]
        self.seen_torch_streams.append(torch.cuda.current_stream().cuda_stream)
        self.seen_ffi_streams.append(
            tvm_ffi.get_raw_stream(tvm_ffi.device(str(int_workspace.device)))
        )
        pin_workspace.fill_(len(self.seen_torch_streams))
        int_workspace.copy_(pin_workspace, non_blocking=True)
        return [len(self.seen_torch_streams)]


class _FakeWrapper:
    def __init__(self, device: torch.device):
        self._int_workspace_buffer = torch.zeros(
            1 << 16, dtype=torch.uint8, device=device
        )
        self._pin_memory_int_workspace_buffer = torch.zeros(
            1 << 16, dtype=torch.uint8, pin_memory=True
        )
        self._cached_module = _FakeCompiledModule()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_side_stream_plan_routes_ffi_stream_and_hands_off() -> None:
    """The plan must run with BOTH the torch current stream and the tvm-ffi
    env stream set to the side stream (torch.cuda.stream alone does not reach
    the ffi env stream -- the documented CCD gotcha), and the schedule must
    land in the graph-referenced real workspace after the stream-ordered
    handoff."""
    pytest.importorskip("tvm_ffi")
    device = torch.device("cuda")
    wrapper = _FakeWrapper(device)

    plan_info = fib._side_stream_plan(wrapper, [None, None, None])
    assert plan_info == [1]
    side = fib._plan_side_stream(wrapper._int_workspace_buffer.device)
    module = wrapper._cached_module
    assert module.seen_torch_streams == [side.cuda_stream]
    assert module.seen_ffi_streams == [side.cuda_stream]
    torch.cuda.synchronize()
    assert bool((wrapper._int_workspace_buffer == 1).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_side_stream_plan_does_not_block_host_on_busy_main_stream() -> None:
    """The point of the side stream: with the main stream holding a long
    kernel (standing in for peer-gated DP allgathers), a busy tick's plans
    must return without draining it, and each tick's schedule must still
    reach the real workspace in order once the queue drains.

    The contract covers one plan per wrapper per busy tick (double-buffered
    slots; a slot's D2D handoff is two ticks old and drained by the time the
    slot is reused -- resolve_seq_lens enforces that on the server), so the
    busy window here spans two ticks' worth of plans for one wrapper."""
    pytest.importorskip("tvm_ffi")
    device = torch.device("cuda")
    wrapper = _FakeWrapper(device)
    # Prime both slots and drain, as a completed prior tick would have.
    fib._side_stream_plan(wrapper, [None, None, None])
    fib._side_stream_plan(wrapper, [None, None, None])
    torch.cuda.synchronize()

    # ~0.5s of main-stream occupancy at typical clocks.
    torch.cuda._sleep(1_000_000_000)
    start = time.perf_counter()
    fib._side_stream_plan(wrapper, [None, None, None])
    fib._side_stream_plan(wrapper, [None, None, None])
    host_elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    drained = time.perf_counter() - start

    assert bool((wrapper._int_workspace_buffer == 4).all())
    # Host returned well before the main stream drained; allow generous
    # margin for slow CI clocks.
    assert host_elapsed < drained / 2, (host_elapsed, drained)
