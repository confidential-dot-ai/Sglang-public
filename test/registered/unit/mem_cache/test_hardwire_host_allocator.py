import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.common import (
    HardwireHostTensorAllocator,
    alloc_with_host_register,
    get_allocator_type,
)
from sglang.srt.utils.hardwire_memory_registry import (
    clear_hardwire_memory_registry,
    get_hardwire_memory_snapshot,
)


class TestHardwireHostTensorAllocator(unittest.TestCase):
    def setUp(self):
        clear_hardwire_memory_registry()

    def tearDown(self):
        clear_hardwire_memory_registry()

    def _fake_hardwire(self, *, kind="pinned"):
        module = types.ModuleType("hardwire_torch")
        module.calls = []

        def create(*, shape, dtype, mode):
            module.calls.append((shape, dtype, mode))
            return torch.empty(shape, dtype=dtype)

        module.createHardWireHostPinnedTensor = create
        module.hardwireTensorInfo = lambda tensor: {"kind": kind}
        return module

    def test_allocate_uses_doorbell_tensor_without_host_register(self):
        fake_hardwire = self._fake_hardwire()
        allocator = HardwireHostTensorAllocator()
        with (
            patch.dict(sys.modules, {"hardwire_torch": fake_hardwire}),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch(
                "sglang.srt.mem_cache.pool_host.common._cuda_host_register"
            ) as register,
        ):
            tensor = alloc_with_host_register(
                (4, 8),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
                allocator=allocator,
            )

        self.assertEqual(tensor.shape, (4, 8))
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(fake_hardwire.calls, [((4, 8), torch.bfloat16, "doorbell")])
        register.assert_not_called()
        snapshot = get_hardwire_memory_snapshot(requested_components=["hicache"])
        self.assertEqual(snapshot["total_allocated_bytes"], 64)
        allocator.close()
        self.assertEqual(get_hardwire_memory_snapshot()["total_allocated_bytes"], 0)

    def test_rejects_non_pinned_hardwire_storage(self):
        with (
            patch.dict(
                sys.modules, {"hardwire_torch": self._fake_hardwire(kind="wire")}
            ),
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "did not return pinned"),
        ):
            HardwireHostTensorAllocator().allocate((16,), torch.uint8, device="cpu")

    def test_rejects_cpu_only_hardwire_stub(self):
        with (
            patch.dict(sys.modules, {"hardwire_torch": self._fake_hardwire()}),
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "requires CUDA"),
        ):
            HardwireHostTensorAllocator().allocate((16,), torch.uint8, device="cpu")

    def test_explicit_hardwire_backend_overrides_storage_inference(self):
        args = SimpleNamespace(
            hicache_host_memory_backend="hardwire",
            hicache_storage_backend="file",
        )
        with patch("sglang.srt.mem_cache.pool_host.common.get_memory", return_value=args):
            self.assertEqual(get_allocator_type(), "hardwire")

    def test_host_pool_destroy_leaves_hardwire_deallocation_to_tensor_owner(self):
        pool = SimpleNamespace(
            _destroyed=False,
            kv_buffer=torch.empty(8),
            pin_memory=True,
            allocator=HardwireHostTensorAllocator(),
        )
        with (
            patch("sglang.srt.mem_cache.pool_host.base._is_cuda", True),
            patch(
                "sglang.srt.mem_cache.pool_host.base._cuda_host_unregister"
            ) as unregister,
        ):
            HostKVCache.destroy(pool)

        unregister.assert_not_called()
        self.assertIsNone(pool.kv_buffer)


if __name__ == "__main__":
    unittest.main()
