"""Delayed sampling must finish on its producer stream before CPU readback."""

import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from sglang.srt.managers.overlap_copy_worker import AsyncD2HCopyWorker
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stream:
    """Queue device work until a dependency or host wait drives it."""

    def __init__(self):
        self.work = []
        self.completed = 0

    def drain(self, end):
        while self.completed < end:
            fn = self.work[self.completed]
            self.completed += 1
            fn()

    def wait_stream(self, producer):
        end = len(producer.work)
        self.work.append(lambda: producer.drain(end))

    def synchronize(self):
        self.drain(len(self.work))


class _Event:
    def __init__(self, device):
        self.device = device

    def record(self, stream=None):
        self.stream = stream if stream is not None else self.device.current
        self.end = len(self.stream.work)

    def synchronize(self):
        self.stream.drain(self.end)


class _Device:
    def __init__(self, default):
        self.default = default
        self.local = threading.local()
        self.local.stream = default

    @property
    def current(self):
        return self.local.stream

    @contextmanager
    def stream(self, stream):
        # The real copy worker starts with its own default current stream.
        previous = self.local.__dict__.get("stream", self.default)
        self.local.stream = stream
        try:
            yield
        finally:
            self.local.stream = previous

    def Event(self):
        return _Event(self)


class TestDelayedSampleCopyOrder(CustomTestCase):
    def _assert_readback(self, *, confidential):
        """Reading before the producer event returns the old token sentinel."""
        scheduler = object.__new__(Scheduler)
        scheduler.schedule_stream = _Stream()
        scheduler.forward_stream = _Stream()
        scheduler.copy_stream = _Stream()
        device = _Device(scheduler.schedule_stream)
        scheduler.device_module = device
        scheduler.forward_stream_ctx = device.stream(scheduler.forward_stream)
        scheduler.copy_stream_ctx = device.stream(scheduler.copy_stream)
        scheduler.enable_cc_async_copy = confidential
        scheduler._relay_forward_payload = lambda *args, **kwargs: None
        values = {"device": -1, "cpu": None}
        result = SimpleNamespace(future_indices=None, logits_output=None)

        def sample():
            device.current.work.append(lambda: values.update(device=42))
            return result

        def copy_to_cpu(**kwargs):
            device.current.work.append(lambda: values.update(cpu=values["device"]))

        result.delay_sample_func = sample
        result.copy_to_cpu = copy_to_cpu
        batch = SimpleNamespace(
            req_pool_indices_cpu=None, return_logprob=False, return_hidden_states=False
        )
        worker = (
            AsyncD2HCopyWorker(device, scheduler.copy_stream) if confidential else None
        )
        scheduler.d2h_worker = worker
        try:
            scheduler.launch_batch_sample_if_needed(result, batch)
            if confidential:
                self.assertTrue(result.copy_ready_cpu.wait(timeout=5))
            else:
                scheduler.copy_stream.synchronize()
            self.assertEqual(values["cpu"], 42)
        finally:
            if worker is not None:
                worker.shutdown()

    def test_cc_primary_producer(self):
        self._assert_readback(confidential=True)

    def test_non_cc_primary_producer(self):
        self._assert_readback(confidential=False)


if __name__ == "__main__":
    unittest.main()
