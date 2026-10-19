import unittest

import torch
from sglang.srt.utils.hardwire_memory_registry import (
    HardwireMemoryRegistry,
    aggregate_hardwire_memory_snapshots,
)


class TestHardwireMemoryRegistry(unittest.TestCase):
    def test_requested_but_unallocated_is_distinct_from_active(self):
        snapshot = HardwireMemoryRegistry().snapshot(
            requested_components=["graph_staging"]
        )

        self.assertTrue(snapshot["requested"])
        self.assertFalse(snapshot["enabled"])
        self.assertFalse(snapshot["active"])
        self.assertFalse(snapshot["verified"])
        self.assertEqual(snapshot["status"], "requested_unallocated")
        self.assertEqual(snapshot["total_allocated_bytes"], 0)
        self.assertTrue(snapshot["by_component"]["graph_staging"]["requested"])
        self.assertFalse(snapshot["by_component"]["graph_staging"]["enabled"])

    def test_actual_storage_bytes_are_deduplicated(self):
        registry = HardwireMemoryRegistry()
        base = torch.empty(128, dtype=torch.uint8)
        registry.register(
            "base",
            component="hicache",
            tensor=base,
            kind="pinned",
        )
        registry.register(
            "view",
            component="hicache",
            tensor=base[32:],
            kind="pinned",
        )

        snapshot = registry.snapshot(requested_components=["hicache"])

        self.assertEqual(snapshot["total_allocated_bytes"], 128)
        self.assertEqual(snapshot["allocation_count"], 1)
        self.assertEqual(snapshot["by_component"]["hicache"]["allocated_bytes"], 128)
        self.assertEqual(snapshot["by_component"]["hicache"]["allocation_count"], 1)

    def test_update_and_unregister_track_lifecycle(self):
        registry = HardwireMemoryRegistry()
        registry.register(
            "slot",
            component="graph_staging",
            tensor=torch.empty(64, dtype=torch.int32),
            kind="pinned",
        )
        self.assertEqual(registry.snapshot()["total_allocated_bytes"], 256)

        registry.register(
            "slot",
            component="graph_staging",
            tensor=torch.empty(128, dtype=torch.int32),
            kind="pinned",
        )
        self.assertEqual(registry.snapshot()["total_allocated_bytes"], 512)

        registry.unregister("slot")
        snapshot = registry.snapshot()
        self.assertEqual(snapshot["status"], "disabled")
        self.assertEqual(snapshot["total_allocated_bytes"], 0)

    def test_multi_rank_aggregation_sums_physical_allocations(self):
        snapshots = [
            {
                "requested_components": ["graph_staging"],
                "active_components": ["graph_staging"],
                "total_allocated_bytes": 1024,
                "allocation_count": 2,
                "by_component": {
                    "graph_staging": {
                        "requested": True,
                        "active": True,
                        "verified": True,
                        "allocated_bytes": 1024,
                        "allocation_count": 2,
                        "kinds": ["pinned"],
                    }
                },
            },
            {
                "requested_components": ["graph_staging", "hicache"],
                "active_components": ["graph_staging", "hicache"],
                "total_allocated_bytes": 6144,
                "allocation_count": 3,
                "by_component": {
                    "graph_staging": {
                        "requested": True,
                        "active": True,
                        "verified": True,
                        "allocated_bytes": 2048,
                        "allocation_count": 2,
                        "kinds": ["pinned"],
                    },
                    "hicache": {
                        "requested": True,
                        "active": True,
                        "verified": True,
                        "allocated_bytes": 4096,
                        "allocation_count": 1,
                        "kinds": ["pinned"],
                    },
                },
            },
        ]

        aggregate = aggregate_hardwire_memory_snapshots(snapshots)

        self.assertEqual(aggregate["status"], "active")
        self.assertTrue(aggregate["enabled"])
        self.assertTrue(aggregate["verified"])
        self.assertEqual(aggregate["rank_count"], 2)
        self.assertEqual(aggregate["total_allocated_bytes"], 7168)
        self.assertEqual(aggregate["allocation_count"], 5)
        self.assertEqual(
            aggregate["by_component"]["graph_staging"]["allocated_bytes"],
            3072,
        )


if __name__ == "__main__":
    unittest.main()
