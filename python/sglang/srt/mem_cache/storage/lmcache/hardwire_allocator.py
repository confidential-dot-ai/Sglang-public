"""Hardwire-backed local CPU arena for in-process LMCache.

LMCache normally owns a large pinned CPU arena.  This adapter replaces that
arena with a TD-private Hardwire ``doorbell`` tensor while preserving
LMCache's suballocator and ``MemoryObj`` APIs.  It is intentionally restricted
to in-process LMCache: the multiprocess connector still exports GPU tensors
through CUDA IPC and defeats the purpose of this path.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.utils.hardwire_memory_registry import (
    register_hardwire_allocation,
    unregister_hardwire_allocation,
)

if TYPE_CHECKING:
    from lmcache.v1.metadata import LMCacheMetadata

logger = logging.getLogger(__name__)

_INSTALL_LOCK = threading.Lock()


def _validate_config(config: Any) -> None:
    """Reject LMCache modes that need a different CPU-memory contract."""
    if bool(getattr(config, "enable_p2p", False)):
        raise ValueError("Hardwire LMCache host memory does not support P2P mode.")
    if bool(getattr(config, "local_cpu_use_hugepages", False)):
        raise ValueError(
            "Hardwire LMCache host memory is already CUDA-pinned and cannot "
            "also use LMCache hugepage allocation."
        )

    extra = getattr(config, "extra_config", None) or {}
    if extra.get("shm_name"):
        raise ValueError(
            "Hardwire LMCache host memory cannot be combined with LMCache SHM."
        )
    if (
        extra.get("enable_nixl_storage")
        and getattr(config, "nixl_buffer_device", None) == "cpu"
    ):
        raise ValueError(
            "Hardwire LMCache host memory does not support NIXL's shared CPU pool."
        )

    io_engine = str(extra.get("rust_raw_block.io_engine", "") or "").lower()
    if io_engine == "io_uring" or bool(
        extra.get("rust_raw_block.use_iouring", False)
        or extra.get("rust_raw_block.use_uring", False)
    ):
        raise ValueError(
            "Hardwire LMCache host memory does not expose LMCache's paged "
            "io_uring fixed-buffer allocator."
        )


def _resolve_size_bytes(
    config: Any,
    metadata: LMCacheMetadata | None,
    backend: Any = None,
) -> int:
    cpu_size_gb = float(config.max_local_cpu_size)
    save_only_first_rank = bool(
        metadata is not None
        and config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
        and metadata.use_mla
    )
    if save_only_first_rank and metadata is not None and metadata.is_first_rank():
        cpu_size_gb = float(
            config.get_extra_config_value("first_rank_max_local_cpu_size", cpu_size_gb)
        )
    calculate_effective_size = getattr(backend, "_calculate_effective_cpu_size", None)
    if calculate_effective_size is not None:
        cpu_size_gb = float(calculate_effective_size(cpu_size_gb, config, metadata))
    size_bytes = int(cpu_size_gb * 1024**3)
    if size_bytes <= 0:
        raise ValueError("Hardwire LMCache host memory requires a positive CPU size.")
    return size_bytes


def _resolve_alignment(config: Any) -> int:
    extra = getattr(config, "extra_config", None) or {}
    alignment = int(extra.get("local_cpu.pinned_align_bytes", 4096))
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(
            "extra_config['local_cpu.pinned_align_bytes'] must be a positive "
            "power of two"
        )
    return alignment


def _build_allocator_class():
    try:
        import hardwire_torch
        from lmcache import torch_dev
        from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
        from lmcache.v1.memory_allocators.mixed_memory_allocator import (
            MixedMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.tensor_memory_allocator import (
            TensorMemoryAllocator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Hardwire LMCache integration requires current LMCache and the "
            "hardwire_torch extension from confidential-dot-ai/hardwire."
        ) from exc

    class HardwireLMCacheMemoryAllocator(MixedMemoryAllocator):
        """LMCache mixed allocator whose pinned arena is a Hardwire tensor."""

        def __init__(self, size: int, align_bytes: int) -> None:
            self.numa_mapping = None
            self.use_hugepages = False
            self.align_bytes = align_bytes
            self.shm_name = None
            self.size = size
            self.buffer = hardwire_torch.createHardWireHostPinnedTensor(
                shape=(size,), dtype=torch.uint8, mode="doorbell"
            )
            self.hardwire_tensor_info = hardwire_torch.hardwireTensorInfo(self.buffer)
            if self.hardwire_tensor_info.get("kind") != "pinned":
                raise RuntimeError(
                    "Hardwire LMCache allocation did not return pinned storage: "
                    f"{self.hardwire_tensor_info!r}"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("Hardwire LMCache host memory requires CUDA.")
            if not self.buffer.is_pinned():
                raise RuntimeError(
                    "Hardwire LMCache tensor is not recognized as pinned by CUDA."
                )

            self._unregistered = False
            self.pin_allocator = TensorMemoryAllocator(
                self.buffer, align_bytes=self.align_bytes
            )
            self.host_mem_lock = threading.Lock()
            self.buffer_allocator = BufferAllocator("cpu")
            self._hardwire_registry_key = f"lmcache:{id(self)}"
            register_hardwire_allocation(
                self._hardwire_registry_key,
                component="lmcache",
                tensor=self.buffer,
                kind=self.hardwire_tensor_info["kind"],
                metadata={"arena_bytes": int(size)},
            )
            logger.info(
                "LMCache is using %.2f GB of TD-private Hardwire pinned memory at %#x",
                size / 1024**3,
                self.buffer.data_ptr(),
            )

        def close(self) -> None:
            """Release the Hardwire-owned arena without calling cudaFreeHost twice."""
            if self._unregistered:
                return
            if torch_dev.is_available():
                torch_dev.synchronize()
            self._unregistered = True
            unregister_hardwire_allocation(self._hardwire_registry_key)
            self.pin_allocator = None
            self.buffer = torch.empty(0, dtype=torch.uint8)

        def free(self, memory_obj: Any, allocator_type: str | None = None) -> None:
            """Release one object, including a reference that outlived close()."""
            if not self._unregistered:
                super().free(memory_obj, allocator_type)
                return
            if memory_obj.is_valid():
                memory_obj.invalidate()
                memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)

        def batched_free(
            self,
            memory_objs: list[Any],
            allocator_type: str | None = None,
            update_stats: bool = True,
        ) -> None:
            """Release objects that outlived close() without touching the arena."""
            if not self._unregistered:
                super().batched_free(
                    memory_objs,
                    allocator_type,
                    update_stats,
                )
                return
            for memory_obj in memory_objs:
                self.free(memory_obj, allocator_type)

        def __str__(self) -> str:
            return "HardwireLMCacheMemoryAllocator"

    return HardwireLMCacheMemoryAllocator


def install_hardwire_lmcache_allocator() -> None:
    """Install an idempotent LocalCPUBackend allocator override.

    The installed hook is process-local. SGLang calls it only for the explicit
    Hardwire in-process mode and before LMCache builds its storage manager.
    """
    try:
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    except ImportError as exc:
        raise RuntimeError(
            "Hardwire LMCache integration requires LMCache to be installed."
        ) from exc

    with _INSTALL_LOCK:
        if getattr(LocalCPUBackend, "_sglang_hardwire_allocator_installed", False):
            return

        allocator_class = _build_allocator_class()
        original_initialize_allocator = LocalCPUBackend.initialize_allocator
        original_close = LocalCPUBackend.close

        def initialize_allocator(
            backend: Any,
            config: Any,
            metadata: LMCacheMetadata | None = None,
        ):
            _validate_config(config)
            return allocator_class(
                _resolve_size_bytes(config, metadata, backend),
                _resolve_alignment(config),
            )

        def close(backend: Any) -> None:
            allocator = backend.memory_allocator
            if not isinstance(allocator, allocator_class):
                original_close(backend)
                return

            if backend.batched_msg_sender is not None:
                backend.batched_msg_sender.close()
            # LMCache's default close order releases the arena before clear().
            # Hardwire tensors own their storage, so drain every MemoryObj view
            # before dropping the owning base tensor.
            backend.clear()
            allocator.close()

        LocalCPUBackend._sglang_original_initialize_allocator = (
            original_initialize_allocator
        )
        LocalCPUBackend._sglang_original_close = original_close
        LocalCPUBackend.initialize_allocator = initialize_allocator
        LocalCPUBackend.close = close
        LocalCPUBackend._sglang_hardwire_allocator_installed = True
