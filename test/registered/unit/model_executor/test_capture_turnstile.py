"""Capture-turnstile helpers: serialize startup CUDA-graph capture across
co-located workers.

The recorded GSP wedges (Xid 120 -> Xid 154, NV_ERR_INVALID_DATA from
MEMORY_TRANSFER_WITH_GSP) all struck while both workers ran startup graph
capture concurrently; the turnstile lets weight loading overlap while only
capture is serialized.
"""

import time

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner import (
    signal_capture_turnstile,
    wait_for_capture_turnstile,
)


class TestCaptureTurnstile:
    def test_wait_noop_when_unset(self):
        wait_for_capture_turnstile()

    def test_signal_noop_when_unset(self):
        signal_capture_turnstile()

    def test_wait_returns_immediately_when_file_exists(self, tmp_path):
        turnstile = tmp_path / "turnstile"
        turnstile.touch()
        start = time.monotonic()
        with envs.SGLANG_CUDA_GRAPH_CAPTURE_WAIT_FILE.override(str(turnstile)):
            wait_for_capture_turnstile()
        assert time.monotonic() - start < 1.0

    def test_wait_times_out_and_proceeds(self, tmp_path):
        absent = tmp_path / "never-created"
        start = time.monotonic()
        with (
            envs.SGLANG_CUDA_GRAPH_CAPTURE_WAIT_FILE.override(str(absent)),
            envs.SGLANG_CUDA_GRAPH_CAPTURE_WAIT_TIMEOUT_S.override(0.0),
        ):
            wait_for_capture_turnstile()
        assert time.monotonic() - start < 1.0
        assert not absent.exists()

    def test_signal_creates_file_and_is_idempotent(self, tmp_path):
        turnstile = tmp_path / "turnstile"
        with envs.SGLANG_CUDA_GRAPH_CAPTURE_SIGNAL_FILE.override(str(turnstile)):
            signal_capture_turnstile()
            assert turnstile.exists()
            # Every rank of the signaling worker touches the file.
            signal_capture_turnstile()
            assert turnstile.exists()

    def test_signal_then_wait_round_trip(self, tmp_path):
        turnstile = tmp_path / "turnstile"
        with envs.SGLANG_CUDA_GRAPH_CAPTURE_SIGNAL_FILE.override(str(turnstile)):
            signal_capture_turnstile()
        start = time.monotonic()
        with envs.SGLANG_CUDA_GRAPH_CAPTURE_WAIT_FILE.override(str(turnstile)):
            wait_for_capture_turnstile()
        assert time.monotonic() - start < 1.0
