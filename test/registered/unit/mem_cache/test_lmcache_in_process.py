"""Unified tree ownership and public LMCache's asymmetric slot-map contract.

The in-memory engine substitutes only the external dependency; allocation,
radix insertion, eviction, cancellation and connector transfers use real code.
"""

import sys
import types
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache import (
    LMCacheUnifiedRadixCache,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _MemoryEngine:
    """Emulate the public engine's chunk masks and absolute store slot indices."""

    def __init__(self):
        self.config = SimpleNamespace(chunk_size=4)
        self.chunks = {}
        self.pins = set()
        self.fail_load = False
        self.drop_store = False

    @staticmethod
    def _key(tokens, end, configs):
        return tuple(tokens[:end]), tuple(sorted(configs.items()))

    def lookup(self, tokens, *, pin=False, lookup_id=None, request_configs):
        end = 0
        for boundary in range(4, len(tokens) + 1, 4):
            if self._key(tokens, boundary, request_configs) not in self.chunks:
                break
            end = boundary
        if pin:
            self.pins.add(lookup_id)
        return end

    def lookup_unpin(self, rid):
        self.pins.discard(rid)

    def store(self, tokens, *, mask, kvcaches, slot_mapping, request_configs, **kwargs):
        if self.drop_store:
            return
        for start in range(0, len(tokens), 4):
            if not mask[start]:
                continue
            slots = slot_mapping[start : start + 4]
            assert len(slots) == 4 and (slots >= 0).all()
            self.chunks[self._key(tokens, start + 4, request_configs)] = [
                buf[slots].clone() for buf in kvcaches
            ]

    def retrieve(
        self, tokens, *, mask, kvcaches, slot_mapping, offset, request_configs, **kwargs
    ):
        result = torch.zeros(len(tokens), dtype=torch.bool)
        for start in range(offset, len(tokens), 4):
            values = self.chunks.get(self._key(tokens, start + 4, request_configs))
            if values is None:
                break
            slots = slot_mapping[start - offset : start + 4 - offset]
            for buf, value in zip(kvcaches, values):
                buf[slots] = value
            if self.fail_load:
                # Failure after a partial write must not publish any private KV.
                raise RuntimeError("injected partial transfer failure")
            result[start : start + 4] = True
        return result

    def clear(self):
        self.chunks.clear()


class TestInProcessLMCache(CustomTestCase):
    def setUp(self):
        self.engine = _MemoryEngine()
        self.buffers = [torch.zeros((65, 1, 2)) for _ in range(2)]
        modules = {}
        for name in ("lmcache", "lmcache.integration", "lmcache.integration.sglang"):
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
        adapter = types.ModuleType("lmcache.integration.sglang.sglang_adapter")
        adapter.LMCacheConnector = lambda *args: SimpleNamespace(
            lmcache_engine=self.engine, kvcaches=self.buffers, close=lambda: None
        )
        config = types.ModuleType("lmcache.integration.sglang.utils")
        config.lmcache_get_config = lambda _: SimpleNamespace(
            chunk_size=4, get_extra_config_value=lambda name, default: default
        )
        modules[adapter.__name__] = adapter
        modules[config.__name__] = config
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(
            patch(
                "sglang.srt.mem_cache.storage.lmcache.in_process_connector._cache_buffers",
                return_value=([self.buffers[0]], [self.buffers[1]]),
            )
        )
        self.sync = self.enterContext(patch("torch.cuda.synchronize"))
        self.enterContext(
            patch(
                "sglang.srt.runtime_context.get_memory",
                return_value=SimpleNamespace(
                    lmcache_mode="ip", lmcache_host_memory_backend="default"
                ),
            )
        )
        req_pool = ReqToTokenPool(4, 64, "cpu", False)
        self.allocator = TokenToKVPoolAllocator(64, torch.float32, "cpu", None, False)
        self.cache = LMCacheUnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            ),
            model_config=SimpleNamespace(),
            tp_size=1,
            tp_rank=0,
            lmcache_config_file=None,
            forward_stream=None,
        )
        self.addCleanup(self.cache.shutdown)
        self.connector = self.cache.lmcache_connector

    def _req(self, rid="request", salt=None):
        return Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", range(8)),
            sampling_params=SamplingParams(),
            cache_salt=salt,
        )

    def _seed(self, salt=None):
        slots = self.allocator.alloc(8)
        for layer, buf in enumerate(self.buffers):
            buf[slots] = (
                torch.arange(16, dtype=torch.float32).reshape(8, 1, 2) + 100 * layer
            )
        operation = self.connector.submit_store(
            "seed",
            list(range(8)),
            slots,
            device_indices_start=0,
            cache_salt=self.connector.build_cache_salt(salt, None),
        )
        self.assertTrue(operation.result)
        self.allocator.free(slots)
        for buf in self.buffers:
            buf.zero_()

    def _prefetch(self, req):
        key = RadixKey(req.origin_input_ids, cache_salt=req.cache_salt)
        match = self.cache.match_prefix(MatchPrefixParams(key=key))
        local = match.device_prefix_len
        self.cache.prefetch_from_storage(
            req.cache_request_handle,
            match.last_device_node,
            list(req.origin_input_ids[local:]),
            matched_prefix_tokens=list(req.origin_input_ids[:local]),
            cache_salt=req.cache_salt,
        )
        self.assertTrue(self.cache.check_prefetch_progress(req.cache_request_handle))
        match = self.cache.match_prefix(MatchPrefixParams(key=key, req=req))
        return key, match

    def _matched_indices(self, req, match):
        # The scheduler adopts the match onto the request before reading slots.
        req.last_node = match.last_device_node
        req.prefix_len = match.device_prefix_len
        return self.cache.prefix_device_indices(req)

    def _load(self, req, match):
        """Start the load and return the request-private slots it fills."""
        loaded, _ = self.cache.init_load_back(
            InitLoadBackParams(match.best_match_node, match.host_hit_length, req=req)
        )
        if not loaded:
            return torch.empty(0, dtype=torch.int64)
        return self.cache._external_flows[req.rid].load.device_indices

    def test_round_trip_abort_releases_private_slots_and_lookup_pins(self):
        """Even a completed transfer remains private until the tree takes ownership."""
        self._seed()
        req = self._req()
        key, match = self._prefetch(req)
        self.assertEqual(match.host_hit_length, 8)
        before = self.allocator.available_size()
        slots = self._load(req, match)
        for layer, buf in enumerate(self.buffers):
            torch.testing.assert_close(
                buf[slots], (torch.arange(16).reshape(8, 1, 2) + 100 * layer).float()
            )
        self.assertEqual(
            self.cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len, 0
        )
        self.cache.release_aborted_request(req.cache_request_handle)
        self.assertEqual(self.allocator.available_size(), before - 8)
        self.cache.check_hicache_events()
        self.assertEqual(self.allocator.available_size(), before)
        self.assertFalse(self.engine.pins)
        self.assertFalse(self.cache.has_pending_cache_operations())

    def test_partial_write_failure_falls_back_without_publishing_or_leaking(self):
        self._seed()
        req = self._req()
        key, match = self._prefetch(req)
        before = self.allocator.available_size()
        self.engine.fail_load = True
        self.assertEqual(len(self._load(req, match)), 0)
        self.assertEqual(self.allocator.available_size(), before)
        self.assertEqual(
            self.cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len, 0
        )
        self.assertFalse(self.engine.pins)
        self.assertFalse(self.cache.has_pending_cache_operations())

    def test_partial_local_chunk_reloads_and_retires_shadow_slots(self):
        """A 3-token local prefix must not produce a negative LMCache slot slice."""
        self._seed()
        local = self.allocator.alloc(3)
        self.cache.insert(InsertParams(RadixKey(array("q", range(3))), local))
        req = self._req()
        key, match = self._prefetch(req)
        self.assertEqual(match.device_prefix_len, 0)
        before = self.allocator.available_size()
        self._load(req, match)
        match = self.cache.match_prefix(MatchPrefixParams(key=key, req=req))
        self.assertEqual(match.device_prefix_len, 8)
        self.assertEqual(self._matched_indices(req, match)[:3].tolist(), local.tolist())
        self.cache.check_hicache_events()
        self.assertEqual(self.allocator.available_size(), before - 5)
        self.cache.release_aborted_request(req.cache_request_handle)
        self.assertEqual(self.allocator.available_size(), before)

    def test_store_suffix_absolute_mapping_and_tree_lock_lifetime(self):
        req = self._req()
        slots = self.allocator.alloc(8)
        for buf in self.buffers:
            buf[slots] = torch.arange(16).reshape(8, 1, 2).float()
        self.cache.insert(InsertParams(RadixKey(req.origin_input_ids), slots))
        self.cache._submit_store(req, req.origin_input_ids[:4])
        self.cache._submit_store(req, req.origin_input_ids)
        self.cache._request_session_finish(req.rid)
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.allocator.available_size(), 56)
        self.cache.check_hicache_events()
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.allocator.available_size(), 64)
        self.assertFalse(self.cache.has_pending_cache_operations())
        for buf in self.buffers:
            buf.zero_()
        key, match = self._prefetch(self._req("next"))
        slots = self._load(self._req("next"), match)
        for buf in self.buffers:
            torch.testing.assert_close(
                buf[slots], torch.arange(16).reshape(8, 1, 2).float()
            )

    def test_salt_isolation_and_declined_store_can_retry(self):
        self._seed(salt="tenant@one")
        _, match = self._prefetch(self._req(salt="tenant@two"))
        self.assertEqual(match.host_hit_length, 0)
        self.assertNotEqual(
            self.connector.build_cache_salt("a", "b:c"),
            self.connector.build_cache_salt("a:b", "c"),
        )
        self.engine.drop_store = True
        operation = self.connector.submit_store(
            "retry",
            list(range(8, 16)),
            torch.arange(1, 9),
            device_indices_start=0,
            cache_salt=None,
        )
        self.assertFalse(operation.result)
        self.assertEqual(self.connector.get_store_start("retry", 8), 0)
        self.engine.drop_store = False
        operation = self.connector.submit_store(
            "retry",
            list(range(8, 16)),
            torch.arange(1, 9),
            device_indices_start=0,
            cache_salt=None,
        )
        self.assertTrue(operation.result)

    def test_remote_rank_miss_prevents_local_admission(self):
        self._seed()
        with patch.object(
            self.connector,
            "parallel_all_reduce",
            side_effect=lambda tensor, op: tensor.zero_(),
        ):
            _, match = self._prefetch(self._req())
        self.assertEqual(match.host_hit_length, 0)
        self.assertFalse(self.engine.pins)


if __name__ == "__main__":
    unittest.main()
