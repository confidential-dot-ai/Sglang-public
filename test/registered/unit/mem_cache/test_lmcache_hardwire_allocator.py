import sys
import types
import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.storage.lmcache import validate_connector_options
from sglang.srt.mem_cache.storage.lmcache.hardwire_allocator import (
    _resolve_alignment,
    _validate_config,
    install_hardwire_lmcache_allocator,
)
from sglang.srt.utils.hardwire_memory_registry import (
    clear_hardwire_memory_registry,
    get_hardwire_memory_snapshot,
)


class _FakeMixedMemoryAllocator:
    pass


class _FakeTensorMemoryAllocator:
    def __init__(self, tensor, align_bytes):
        self.tensor = tensor
        self.align_bytes = align_bytes


class _FakeBufferAllocator:
    def __init__(self, device):
        self.device = device


class _FakeMemoryObj:
    def __init__(self):
        self.valid = True
        self.raw_data = torch.empty(8)

    def is_valid(self):
        return self.valid

    def invalidate(self):
        self.valid = False


class _FakeLocalCPUBackend:
    def __init__(self):
        self.batched_msg_sender = None
        self.events = []

    def initialize_allocator(self, config, metadata=None):
        return "original"

    def clear(self):
        self.events.append("clear")

    def close(self):
        self.events.append("original_close")


class _FakeTorchDevice:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize():
        raise AssertionError("synchronize should not run without a device")


class _Config:
    max_local_cpu_size = 1 / 1024**3
    enable_p2p = False
    local_cpu_use_hugepages = False
    nixl_buffer_device = "cuda"
    extra_config: ClassVar[dict] = {}

    @staticmethod
    def get_extra_config_value(name, default):
        return default


def _fake_modules():
    hardwire = types.ModuleType("hardwire_torch")
    hardwire.createHardWireHostPinnedTensor = lambda *, shape, dtype, mode: torch.empty(
        shape, dtype=dtype
    )
    hardwire.hardwireTensorInfo = lambda tensor: {"kind": "pinned"}

    lmcache = types.ModuleType("lmcache")
    lmcache.__path__ = []
    lmcache.torch_dev = _FakeTorchDevice

    modules = {
        "hardwire_torch": hardwire,
        "lmcache": lmcache,
        "lmcache.v1": types.ModuleType("lmcache.v1"),
        "lmcache.v1.memory_allocators": types.ModuleType(
            "lmcache.v1.memory_allocators"
        ),
        "lmcache.v1.memory_allocators.buffer_allocator": types.ModuleType(
            "lmcache.v1.memory_allocators.buffer_allocator"
        ),
        "lmcache.v1.memory_allocators.mixed_memory_allocator": types.ModuleType(
            "lmcache.v1.memory_allocators.mixed_memory_allocator"
        ),
        "lmcache.v1.memory_allocators.tensor_memory_allocator": types.ModuleType(
            "lmcache.v1.memory_allocators.tensor_memory_allocator"
        ),
        "lmcache.v1.storage_backend": types.ModuleType("lmcache.v1.storage_backend"),
        "lmcache.v1.storage_backend.local_cpu_backend": types.ModuleType(
            "lmcache.v1.storage_backend.local_cpu_backend"
        ),
    }
    modules["lmcache.v1"].__path__ = []
    modules["lmcache.v1.memory_allocators"].__path__ = []
    modules["lmcache.v1.storage_backend"].__path__ = []
    modules[
        "lmcache.v1.memory_allocators.buffer_allocator"
    ].BufferAllocator = _FakeBufferAllocator
    modules[
        "lmcache.v1.memory_allocators.mixed_memory_allocator"
    ].MixedMemoryAllocator = _FakeMixedMemoryAllocator
    modules[
        "lmcache.v1.memory_allocators.tensor_memory_allocator"
    ].TensorMemoryAllocator = _FakeTensorMemoryAllocator
    modules[
        "lmcache.v1.storage_backend.local_cpu_backend"
    ].LocalCPUBackend = _FakeLocalCPUBackend
    return modules


class TestLMCacheHardwireAllocator(unittest.TestCase):
    def setUp(self):
        clear_hardwire_memory_registry()
        original = getattr(
            _FakeLocalCPUBackend, "_sglang_original_initialize_allocator", None
        )
        if original is not None:
            _FakeLocalCPUBackend.initialize_allocator = original
        original_close = getattr(_FakeLocalCPUBackend, "_sglang_original_close", None)
        if original_close is not None:
            _FakeLocalCPUBackend.close = original_close
        for name in (
            "_sglang_hardwire_allocator_installed",
            "_sglang_original_initialize_allocator",
            "_sglang_original_close",
        ):
            if hasattr(_FakeLocalCPUBackend, name):
                delattr(_FakeLocalCPUBackend, name)

    def tearDown(self):
        clear_hardwire_memory_registry()

    def test_installs_real_hardwire_arena_into_local_cpu_backend(self):
        with (
            patch.dict(sys.modules, _fake_modules()),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
        ):
            install_hardwire_lmcache_allocator()
            backend = _FakeLocalCPUBackend()
            allocator = backend.initialize_allocator(_Config())

        self.assertIsInstance(allocator, _FakeMixedMemoryAllocator)
        self.assertEqual(allocator.buffer.numel(), 1)
        self.assertEqual(allocator.hardwire_tensor_info, {"kind": "pinned"})
        self.assertEqual(allocator.pin_allocator.align_bytes, 4096)
        self.assertEqual(
            get_hardwire_memory_snapshot()["by_component"]["lmcache"][
                "allocated_bytes"
            ],
            1,
        )
        backend.memory_allocator = allocator
        backend.close()
        self.assertEqual(backend.events, ["clear"])
        self.assertEqual(allocator.buffer.numel(), 0)
        self.assertEqual(get_hardwire_memory_snapshot()["total_allocated_bytes"], 0)
        late_object = _FakeMemoryObj()
        allocator.free(late_object)
        self.assertFalse(late_object.is_valid())
        self.assertEqual(late_object.raw_data.numel(), 0)

    def test_rejects_lmcache_modes_with_incompatible_memory_contracts(self):
        for config, expected in (
            (SimpleNamespace(enable_p2p=True), "P2P"),
            (
                SimpleNamespace(
                    enable_p2p=False,
                    local_cpu_use_hugepages=True,
                    extra_config={},
                ),
                "hugepage",
            ),
        ):
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(ValueError, expected),
            ):
                _validate_config(config)

    def test_alignment_must_be_power_of_two(self):
        config = SimpleNamespace(extra_config={"local_cpu.pinned_align_bytes": 6000})
        with self.assertRaisesRegex(ValueError, "power of two"):
            _resolve_alignment(config)

    def test_connector_mode_and_allocator_compatibility(self):
        from sglang.srt.mem_cache.storage.lmcache import resolve_connector_mode

        self.assertEqual(resolve_connector_mode("auto", "default"), "mp")
        self.assertEqual(resolve_connector_mode("auto", "hardwire"), "ip")
        for backend in ("default", "hardwire"):
            self.assertEqual(resolve_connector_mode("ip", backend), "ip")
        self.assertEqual(resolve_connector_mode("mp", "default"), "mp")
        with self.assertRaisesRegex(ValueError, "multiprocess connector"):
            validate_connector_options("mp", "hardwire")


if __name__ == "__main__":
    unittest.main()
