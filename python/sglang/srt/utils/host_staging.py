"""Event-gated pinned host staging for small CPU-to-device inputs.

CUDA copies from pinned memory are asynchronous: their source storage must not
be overwritten until the copy stream has consumed it.  This module owns a
bounded ring per ``(device, dtype, backend)`` and gates reuse with a CUDA event.

Host staging always runs outside CUDA graph capture.  Captured graphs continue
to read stable device buffers; callers can pass ``out=`` to stage directly into
one of those buffers without an intermediate GPU tensor.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class _HostStagingSlot:
    buffer: torch.Tensor | None = None
    ready_event: Any = None
    leased: bool = False


def _as_cpu_tensor(
    values: Sequence[int] | torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        values.detach().to(device="cpu", dtype=dtype).reshape(-1)
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=dtype, device="cpu")
    )


def _allocate_host_buffer(
    capacity: int,
    dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    if backend == "pageable":
        return torch.empty(capacity, dtype=dtype, device="cpu")
    if backend == "default":
        return torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
    raise ValueError(
        "cuda graph host-memory backend must be 'default' or 'pageable', "
        f"got {backend!r}"
    )


class EventGatedHostStagingPool:
    """A fixed-depth ring whose slots cannot be reused before GPU consumption."""

    def __init__(
        self,
        *,
        depth: int,
        backend: str,
        growth_tokens: int = 256,
    ) -> None:
        if depth <= 0:
            raise ValueError(f"host staging pool depth must be positive, got {depth}")
        if growth_tokens <= 0:
            raise ValueError(
                f"host staging growth tokens must be positive, got {growth_tokens}"
            )
        self.depth = depth
        self.backend = backend
        self.growth_tokens = growth_tokens
        self._slots = [_HostStagingSlot() for _ in range(depth)]
        self._next = 0
        self._lock = threading.Lock()

    def copy_to_device(
        self,
        values: Sequence[int] | torch.Tensor,
        *,
        dtype: torch.dtype,
        device: str | torch.device,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage ``values`` and enqueue an asynchronous host-to-device copy.

        When ``out`` is supplied, the copy lands directly in its leading
        ``numel(values)`` elements and the same tensor is returned. Otherwise a
        device tensor is created by ``Tensor.to``. The slot's completion event
        is recorded immediately after that copy on the caller's current stream.
        """
        lease = self.acquire(values, dtype=dtype)
        return lease.copy_to_device(device=device, out=out)

    def acquire(
        self,
        values: Sequence[int] | torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> HostStagingLease:
        """Copy values into a pinned slot and return a deferred-copy lease."""
        source = _as_cpu_tensor(values, dtype)
        n = source.numel()
        with self._lock:
            slot_index = self._select_slot_for_write()
            slot = self._slots[slot_index]
            capacity = (
                (max(n, 1) + self.growth_tokens - 1)
                // self.growth_tokens
                * self.growth_tokens
            )
            if (
                slot.buffer is None
                or slot.buffer.numel() < n
                or slot.buffer.dtype != dtype
            ):
                slot.buffer = _allocate_host_buffer(capacity, dtype, self.backend)
            slot.buffer[:n].copy_(source)
            slot.leased = True
            return HostStagingLease(self, slot_index, n)

    def _select_slot_for_write(self) -> int:
        candidate = None
        for offset in range(self.depth):
            index = (self._next + offset) % self.depth
            slot = self._slots[index]
            if slot.leased:
                continue
            if slot.ready_event is None:
                candidate = index
                break
            query = getattr(slot.ready_event, "query", None)
            if query is not None and query():
                candidate = index
                break
            if candidate is None:
                candidate = index
        if candidate is None:
            raise RuntimeError(
                "All pinned host staging slots are leased before their H2D "
                "copies were issued; increase --cuda-graph-host-staging-pool-depth."
            )
        slot = self._slots[candidate]
        if slot.ready_event is not None:
            query = getattr(slot.ready_event, "query", None)
            if query is None or not query():
                slot.ready_event.synchronize()
        self._next = (candidate + 1) % self.depth
        return candidate

    def _consume(
        self,
        slot_index: int,
        n: int,
        *,
        device: str | torch.device,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        if target_device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Pinned host staging must run before CUDA graph capture/replay; "
                "a host transfer was requested during active capture."
            )
        with self._lock:
            slot = self._slots[slot_index]
            if not slot.leased or slot.buffer is None:
                raise RuntimeError("Pinned host staging lease was already consumed")
            staged = slot.buffer[:n]
            try:
                if target_device.type == "cpu":
                    if out is None:
                        result = staged.clone()
                    else:
                        if out.numel() < n:
                            raise ValueError(
                                f"out has {out.numel()} elements but staging needs {n}"
                            )
                        out.reshape(-1)[:n].copy_(staged)
                        result = out
                    slot.ready_event = None
                    return result

                if out is not None:
                    if out.device.type != target_device.type or (
                        target_device.index is not None
                        and out.device.index != target_device.index
                    ):
                        raise ValueError(
                            f"out device {out.device} does not match target "
                            f"{target_device}"
                        )
                    if out.numel() < n:
                        raise ValueError(
                            f"out has {out.numel()} elements but staging needs {n}"
                        )
                    out.reshape(-1)[:n].copy_(staged, non_blocking=True)
                    result = out
                else:
                    result = staged.to(target_device, non_blocking=True)

                device_module = torch.get_device_module(target_device)
                ready_event = device_module.Event()
                ready_event.record(device_module.current_stream(target_device))
                slot.ready_event = ready_event
                return result
            finally:
                slot.leased = False

    def _release(self, slot_index: int) -> None:
        with self._lock:
            self._slots[slot_index].leased = False

    def close(self) -> None:
        """Wait for in-flight reads before releasing host storage."""
        with self._lock:
            for slot in self._slots:
                if slot.ready_event is not None:
                    slot.ready_event.synchronize()
                slot.ready_event = None
                slot.leased = False
                slot.buffer = None


class HostStagingLease:
    """Ownership token for a pinned slot awaiting its H2D copy."""

    def __init__(
        self,
        pool: EventGatedHostStagingPool,
        slot_index: int,
        numel: int,
    ) -> None:
        self._pool = pool
        self._slot_index = slot_index
        self._numel = numel
        self._active = True

    @property
    def tensor(self) -> torch.Tensor:
        slot = self._pool._slots[self._slot_index]
        if not self._active or slot.buffer is None:
            raise RuntimeError("Pinned host staging lease is no longer active")
        return slot.buffer[: self._numel]

    def copy_to_device(
        self,
        *,
        device: str | torch.device,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("Pinned host staging lease was already consumed")
        try:
            return self._pool._consume(
                self._slot_index,
                self._numel,
                device=device,
                out=out,
            )
        except Exception:
            self._pool._release(self._slot_index)
            raise
        finally:
            self._active = False

    def release(self) -> None:
        if not self._active:
            return
        self._pool._release(self._slot_index)
        self._active = False

    def __del__(self) -> None:
        try:
            self.release()
        except (AttributeError, RuntimeError):
            pass


_pools: dict[tuple[str, torch.dtype, str, int, int], EventGatedHostStagingPool] = {}
_pools_lock = threading.Lock()


def _resolved_config() -> tuple[str, int, int]:
    from sglang.srt.runtime_context import get_exec
    from sglang.srt.utils.common import is_host_pinning_effective

    graph_config = get_exec().graph
    backend = getattr(graph_config, "cuda_graph_host_memory_backend", "default")
    if backend == "default" and not is_host_pinning_effective():
        backend = "pageable"
    return (
        backend,
        getattr(graph_config, "cuda_graph_host_staging_pool_depth", 64),
        getattr(graph_config, "cuda_graph_host_staging_growth_tokens", 256),
    )


def copy_values_to_device(
    values: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    out: torch.Tensor | None = None,
    backend: str | None = None,
    depth: int | None = None,
    growth_tokens: int | None = None,
) -> torch.Tensor:
    """Copy small host values through the process-wide event-gated pool."""
    target_device = torch.device(device)
    if target_device.type == "cpu":
        source = _as_cpu_tensor(values, dtype)
        if out is None:
            return source
        if out.numel() < source.numel():
            raise ValueError(
                f"out has {out.numel()} elements but staging needs {source.numel()}"
            )
        out.reshape(-1)[: source.numel()].copy_(source)
        return out
    if backend is None or depth is None or growth_tokens is None:
        resolved_backend, resolved_depth, resolved_growth = _resolved_config()
        backend = resolved_backend if backend is None else backend
        depth = resolved_depth if depth is None else depth
        growth_tokens = resolved_growth if growth_tokens is None else growth_tokens
    assert backend is not None and depth is not None and growth_tokens is not None
    key = (str(target_device), dtype, backend, depth, growth_tokens)
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            pool = EventGatedHostStagingPool(
                depth=depth,
                backend=backend,
                growth_tokens=growth_tokens,
            )
            _pools[key] = pool
    return pool.copy_to_device(values, dtype=dtype, device=device, out=out)


def acquire_host_staging_lease(
    values: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    backend: str | None = None,
    depth: int | None = None,
    growth_tokens: int | None = None,
) -> HostStagingLease | None:
    """Copy values into a persistent pinned slot for a deferred H2D transfer.

    CPU targets need no pinned staging and return ``None``. Callers retain the
    returned lease until they invoke ``HostStagingLease.copy_to_device`` or
    ``release`` it without issuing a transfer.
    """
    target_device = torch.device(device)
    if target_device.type == "cpu":
        return None
    if backend is None or depth is None or growth_tokens is None:
        resolved_backend, resolved_depth, resolved_growth = _resolved_config()
        backend = resolved_backend if backend is None else backend
        depth = resolved_depth if depth is None else depth
        growth_tokens = resolved_growth if growth_tokens is None else growth_tokens
    assert backend is not None and depth is not None and growth_tokens is not None
    key = (str(target_device), dtype, backend, depth, growth_tokens)
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            pool = EventGatedHostStagingPool(
                depth=depth,
                backend=backend,
                growth_tokens=growth_tokens,
            )
            _pools[key] = pool
    return pool.acquire(values, dtype=dtype)


def clear_host_staging_pools() -> None:
    """Test/shutdown hook that safely releases every staging pool."""
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.close()
