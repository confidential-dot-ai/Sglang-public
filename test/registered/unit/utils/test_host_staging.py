import unittest
from unittest.mock import patch

import torch
from sglang.srt.utils.host_staging import EventGatedHostStagingPool


class _FakeEvent:
    def __init__(self):
        self.records = []
        self.synchronize_count = 0

    def record(self, stream=None):
        self.records.append(stream)

    def synchronize(self):
        self.synchronize_count += 1


class _FakeDeviceModule:
    def __init__(self):
        self.events = []
        self.stream = object()

    def Event(self):
        event = _FakeEvent()
        self.events.append(event)
        return event

    def current_stream(self, device=None):
        return self.stream


class _FakeCudaOut:
    def __init__(self, size, dtype):
        self.storage = torch.empty(size, dtype=dtype)
        self.device = torch.device("cuda")

    def numel(self):
        return self.storage.numel()

    def reshape(self, *shape):
        return self

    def __getitem__(self, index):
        return self.storage[index]


class _AllocTracker:
    """Stands in for _allocate_host_buffer and records every allocation."""

    def __init__(self):
        self.allocations = []

    def __call__(self, capacity, dtype, backend):
        self.allocations.append((capacity, dtype, backend))
        return torch.empty(capacity, dtype=dtype)


class TestEventGatedHostStagingPool(unittest.TestCase):
    def test_slot_reuse_waits_for_previous_copy(self):
        allocator = _AllocTracker()
        device_module = _FakeDeviceModule()
        pool = EventGatedHostStagingPool(depth=1, backend="pageable")
        with (
            patch("sglang.srt.utils.host_staging._allocate_host_buffer", allocator),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch("torch.get_device_module", return_value=device_module),
        ):
            first_out = _FakeCudaOut(2, torch.int64)
            result = pool.copy_to_device(
                [1, 2],
                dtype=torch.int64,
                device="cuda",
                out=first_out,
            )
            self.assertIs(result, first_out)
            torch.testing.assert_close(first_out.storage, torch.tensor([1, 2]))
            first_event = device_module.events[0]
            self.assertEqual(first_event.records, [device_module.stream])

            second_out = _FakeCudaOut(2, torch.int64)
            pool.copy_to_device(
                [3, 4],
                dtype=torch.int64,
                device="cuda",
                out=second_out,
            )
            self.assertEqual(first_event.synchronize_count, 1)
            torch.testing.assert_close(second_out.storage, torch.tensor([3, 4]))

            self.assertEqual(
                allocator.allocations,
                [(256, torch.int64, "pageable")],
            )
            second_event = device_module.events[1]
            pool.close()
            self.assertEqual(second_event.synchronize_count, 1)

    def test_buffer_grows_only_after_prior_copy_finishes(self):
        allocator = _AllocTracker()
        device_module = _FakeDeviceModule()
        pool = EventGatedHostStagingPool(depth=1, backend="pageable")
        with (
            patch("sglang.srt.utils.host_staging._allocate_host_buffer", allocator),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch("torch.get_device_module", return_value=device_module),
        ):
            pool.copy_to_device(
                [1],
                dtype=torch.int32,
                device="cuda",
                out=_FakeCudaOut(1, torch.int32),
            )
            first_event = device_module.events[0]
            values = list(range(300))
            pool.copy_to_device(
                values,
                dtype=torch.int32,
                device="cuda",
                out=_FakeCudaOut(300, torch.int32),
            )

        self.assertEqual(first_event.synchronize_count, 1)
        self.assertEqual(
            allocator.allocations,
            [
                (256, torch.int32, "pageable"),
                (512, torch.int32, "pageable"),
            ],
        )

    def test_growth_quantum_is_configurable(self):
        allocator = _AllocTracker()
        pool = EventGatedHostStagingPool(
            depth=1,
            backend="pageable",
            growth_tokens=64,
        )
        with (
            patch("sglang.srt.utils.host_staging._allocate_host_buffer", allocator),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
        ):
            lease = pool.acquire(list(range(65)), dtype=torch.int64)
            lease.release()

        self.assertEqual(
            allocator.allocations,
            [(128, torch.int64, "pageable")],
        )

    def test_host_transfer_is_rejected_during_cuda_graph_capture(self):
        pool = EventGatedHostStagingPool(depth=1, backend="pageable")
        allocator = _AllocTracker()
        with (
            patch("sglang.srt.utils.host_staging._allocate_host_buffer", allocator),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
        ):
            lease = pool.acquire([1], dtype=torch.int32)
            with (
                patch("torch.cuda.is_current_stream_capturing", return_value=True),
                self.assertRaisesRegex(RuntimeError, "during active capture"),
            ):
                lease.copy_to_device(device="cuda")
            # Capture rejection releases the slot rather than leaking a lease.
            replacement = pool.acquire([2], dtype=torch.int32)
            replacement.release()

    def test_deferred_lease_prevents_overwrite_until_consumed(self):
        allocator = _AllocTracker()
        device_module = _FakeDeviceModule()
        pool = EventGatedHostStagingPool(depth=1, backend="pageable")
        with (
            patch("sglang.srt.utils.host_staging._allocate_host_buffer", allocator),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch("torch.get_device_module", return_value=device_module),
        ):
            lease = pool.acquire([7, 8], dtype=torch.int64)
            torch.testing.assert_close(lease.tensor, torch.tensor([7, 8]))
            with self.assertRaisesRegex(RuntimeError, "All pinned host staging"):
                pool.acquire([9], dtype=torch.int64)

            output = _FakeCudaOut(2, torch.int64)
            lease.copy_to_device(device="cuda", out=output)
            torch.testing.assert_close(output.storage, torch.tensor([7, 8]))

            replacement = pool.acquire([9], dtype=torch.int64)
            self.assertEqual(device_module.events[0].synchronize_count, 1)
            replacement.release()


if __name__ == "__main__":
    unittest.main()
