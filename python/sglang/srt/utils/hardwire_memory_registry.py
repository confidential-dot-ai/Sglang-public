"""Process-local accounting for verified Hardwire host allocations."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class _HardwireAllocation:
    component: str
    kind: str
    allocated_bytes: int
    storage_identity: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


class HardwireMemoryRegistry:
    """Thread-safe registry containing no tensor references or raw pointers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._allocations: dict[str, _HardwireAllocation] = {}

    def register(
        self,
        key: str,
        *,
        component: str,
        tensor: torch.Tensor,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register or replace one verified allocation owned by ``key``."""
        storage = tensor.untyped_storage()
        allocated_bytes = int(storage.nbytes())
        if allocated_bytes < 0:
            raise ValueError("Hardwire allocation size cannot be negative")
        record = _HardwireAllocation(
            component=str(component),
            kind=str(kind),
            allocated_bytes=allocated_bytes,
            storage_identity=(int(storage.data_ptr()), allocated_bytes),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._allocations[str(key)] = record

    def unregister(self, key: str) -> None:
        """Remove an allocation; missing keys are already unregistered."""
        with self._lock:
            self._allocations.pop(str(key), None)

    def snapshot(
        self,
        *,
        requested_components: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Return a JSON/msgpack-safe summary with unique-storage accounting."""
        requested = sorted({str(x) for x in requested_components})
        with self._lock:
            records = list(self._allocations.values())

        # A base tensor and one of its views may be registered by different
        # owners during migration. Count physical storage once globally and
        # once within each component rather than inflating the pool size.
        global_seen: set[tuple[int, int]] = set()
        by_component: dict[str, dict[str, Any]] = {}
        for record in records:
            component = by_component.setdefault(
                record.component,
                {
                    "requested": record.component in requested,
                    "enabled": True,
                    "active": True,
                    "verified": True,
                    "allocated_bytes": 0,
                    "allocation_count": 0,
                    "kinds": [],
                },
            )
            seen = component.setdefault("_seen", set())
            if record.storage_identity not in seen:
                seen.add(record.storage_identity)
                component["allocated_bytes"] += record.allocated_bytes
                component["allocation_count"] += 1
            if record.kind not in component["kinds"]:
                component["kinds"].append(record.kind)
            global_seen.add(record.storage_identity)

        for name in requested:
            by_component.setdefault(
                name,
                {
                    "requested": True,
                    "enabled": False,
                    "active": False,
                    "verified": False,
                    "allocated_bytes": 0,
                    "allocation_count": 0,
                    "kinds": [],
                },
            )

        for component in by_component.values():
            component.pop("_seen", None)
            component["kinds"].sort()

        total_bytes = sum(size for _, size in global_seen)
        active_components = sorted(
            name for name, value in by_component.items() if value["active"]
        )
        requested_any = bool(requested)
        if active_components:
            status = "active"
        elif requested_any:
            status = "requested_unallocated"
        else:
            status = "disabled"
        return {
            "requested": requested_any,
            "enabled": bool(active_components),
            "active": bool(active_components),
            "verified": bool(active_components),
            "status": status,
            "requested_components": requested,
            "active_components": active_components,
            "total_allocated_bytes": total_bytes,
            "allocation_count": len(global_seen),
            "by_component": {name: by_component[name] for name in sorted(by_component)},
        }

    def clear(self) -> None:
        """Clear process-local accounting; intended for tests and shutdown."""
        with self._lock:
            self._allocations.clear()


_REGISTRY = HardwireMemoryRegistry()


def register_hardwire_allocation(
    key: str,
    *,
    component: str,
    tensor: torch.Tensor,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    _REGISTRY.register(
        key,
        component=component,
        tensor=tensor,
        kind=kind,
        metadata=metadata,
    )


def unregister_hardwire_allocation(key: str) -> None:
    _REGISTRY.unregister(key)


def get_hardwire_memory_snapshot(
    *, requested_components: Iterable[str] = ()
) -> dict[str, Any]:
    return _REGISTRY.snapshot(requested_components=requested_components)


def clear_hardwire_memory_registry() -> None:
    _REGISTRY.clear()


def aggregate_hardwire_memory_snapshots(
    snapshots: Iterable[dict[str, Any]],
    *,
    requested_components: Iterable[str] = (),
) -> dict[str, Any]:
    """Aggregate process-local snapshots into one server-wide summary."""
    requested = {str(name) for name in requested_components}
    active_components = set()
    by_component: dict[str, dict[str, Any]] = {}
    total_allocated_bytes = 0
    allocation_count = 0
    valid_snapshots = [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]
    for snapshot in valid_snapshots:
        requested.update(snapshot.get("requested_components", []))
        active_components.update(snapshot.get("active_components", []))
        total_allocated_bytes += int(snapshot.get("total_allocated_bytes", 0))
        allocation_count += int(snapshot.get("allocation_count", 0))
        for name, component in snapshot.get("by_component", {}).items():
            was_active = bool(by_component.get(name, {}).get("active", False))
            aggregate = by_component.setdefault(
                name,
                {
                    "requested": False,
                    "enabled": False,
                    "active": False,
                    "verified": False,
                    "allocated_bytes": 0,
                    "allocation_count": 0,
                    "kinds": [],
                },
            )
            aggregate["requested"] |= bool(component.get("requested", False))
            component_active = bool(component.get("active", False))
            aggregate["active"] |= component_active
            aggregate["enabled"] = aggregate["active"]
            if component_active:
                component_verified = bool(component.get("verified", False))
                aggregate["verified"] = (
                    aggregate["verified"] and component_verified
                    if was_active
                    else component_verified
                )
            aggregate["allocated_bytes"] += int(component.get("allocated_bytes", 0))
            aggregate["allocation_count"] += int(component.get("allocation_count", 0))
            aggregate["kinds"] = sorted(
                set(aggregate["kinds"]) | set(component.get("kinds", []))
            )

    for name in requested:
        by_component.setdefault(
            name,
            {
                "requested": True,
                "enabled": False,
                "active": False,
                "verified": False,
                "allocated_bytes": 0,
                "allocation_count": 0,
                "kinds": [],
            },
        )["requested"] = True

    requested_any = bool(requested)
    active = bool(active_components)
    return {
        "requested": requested_any,
        "enabled": active,
        "active": active,
        "verified": active
        and all(
            bool(component["verified"])
            for component in by_component.values()
            if component["active"]
        ),
        "status": (
            "active"
            if active
            else "requested_unallocated"
            if requested_any
            else "disabled"
        ),
        "requested_components": sorted(requested),
        "active_components": sorted(active_components),
        "total_allocated_bytes": total_allocated_bytes,
        "allocation_count": allocation_count,
        "rank_count": len(valid_snapshots),
        "by_component": {name: by_component[name] for name in sorted(by_component)},
    }
