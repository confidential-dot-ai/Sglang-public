"""In-process LMCache transport for the unified radix tree.

This adapter deliberately completes transfers before returning to admission.
The tree still owns allocation, cancellation, publication and store locks. The
public LMCache SGLang connector supports ordinary MHA/MLA buffers, not hybrid
state or packed KV with auxiliary scales/indexers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import Future
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class _Lookup:
    request_id: str
    token_ids: list[int]
    cache_salt: str | None
    total_hit: int
    lock_start: int = 0


@dataclass
class _Transfer:
    request_id: str
    future: Future
    result: bool
    device_indices: torch.Tensor | None = None
    local_hit_tokens: int = 0


def _cache_buffers(pool, model_config, tp_size):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

    # Exact types prevent accidentally dropping a subclass's quantization or
    # sparse-index sidecars. New layouts need an explicit serialization contract.
    if type(pool) is MHATokenToKVPool:
        if pool.use_hnd or pool.kv_cache_layout != "nhd":
            raise ValueError("In-process LMCache requires the NHD MHA layout.")
        k_pool, v_pool = pool.k_buffer, pool.v_buffer
        expected_shape = (model_config.get_num_kv_heads(tp_size), model_config.head_dim)
    elif type(pool) is MLATokenToKVPool and not pool.use_dsa:
        k_pool, v_pool = pool.kv_buffer, []
        expected_shape = (1, pool.kv_cache_dim)
    else:
        raise ValueError(
            "In-process LMCache supports plain MHA/MLA pools only; "
            "SWA, Mamba, sparse indexers and packed/quantized sidecars are unsupported."
        )
    if len(k_pool) != model_config.num_hidden_layers or (
        v_pool and len(v_pool) != len(k_pool)
    ):
        raise ValueError("In-process LMCache requires all model layers on this rank.")
    for tensor in k_pool + v_pool:
        if (
            tensor.device.type != "cuda"
            or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or tensor.ndim != 3
            or tuple(tensor.shape[1:]) != expected_shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "In-process LMCache requires contiguous CUDA NHD FP16/BF16/FP32 "
                "buffers with equal K/V head dimensions."
            )
    return k_pool, v_pool


class UnifiedLMCacheIPConnector:
    """Synchronous in-process implementation of the unified cache transport."""

    operation_timeout = None

    def __init__(
        self,
        *,
        config_file,
        model_config,
        tp_size,
        tp_rank,
        tp_group,
        pp_size,
        pp_rank,
        pp_group,
        page_size,
        token_to_kv_pool_allocator,
        req_to_token_pool,
        tree_components,
        mamba_component,
        sliding_window_size,
        host_memory_backend="default",
    ):
        del pp_rank, pp_group, req_to_token_pool
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        if (
            pp_size != 1
            or mamba_component is not None
            or sliding_window_size is not None
            or set(tree_components) != {ComponentType.FULL}
        ):
            raise ValueError(
                "In-process LMCache requires Full-only KV and PP=1; "
                "use the multiprocess connector for hybrid state."
            )
        pool = token_to_kv_pool_allocator.get_kvcache()
        k_pool, v_pool = _cache_buffers(pool, model_config, tp_size)
        try:
            from lmcache.integration.sglang.sglang_adapter import LMCacheConnector
            from lmcache.integration.sglang.utils import lmcache_get_config
        except ImportError as exc:
            raise RuntimeError(
                "In-process LMCache requires LMCache's SGLang LMCacheConnector "
                "and lmcache_get_config interfaces; see storage/lmcache/README.md."
            ) from exc

        config = lmcache_get_config(config_file or "")
        unsupported = [
            name
            for name in (
                "use_layerwise",
                "async_loading",
                "enable_blending",
                "enable_p2p",
            )
            if getattr(config, name, False)
        ]
        if unsupported:
            raise ValueError(
                "In-process unified LMCache requires these options disabled: "
                + ", ".join(unsupported)
            )
        use_mla = not v_pool
        if tp_size > 1 and config.get_extra_config_value(
            "save_only_first_rank", use_mla
        ):
            raise ValueError(
                "In-process LMCache TP requires extra_config.save_only_first_rank=false; "
                "each rank stores its own KV."
            )
        self.chunk_size = int(config.chunk_size)
        if self.chunk_size <= 0 or self.chunk_size % page_size:
            raise ValueError(
                "LMCache chunk_size must be a positive multiple of page_size."
            )
        if host_memory_backend == "hardwire":
            from sglang.srt.mem_cache.storage.lmcache.hardwire_allocator import (
                install_hardwire_lmcache_allocator,
            )

            install_hardwire_lmcache_allocator()
        self._connector = LMCacheConnector(
            model_config, tp_size, tp_rank, k_pool, v_pool, config_file or ""
        )
        self._engine = self._connector.lmcache_engine
        self._kvcaches = self._connector.kvcaches
        self._device = k_pool[0].device
        self._tp_size = tp_size
        self._tp_group = tp_group
        self._lookups: dict[str, _Lookup] = {}
        self._store_ends: dict[str, int] = {}
        self._closed = False

    @staticmethod
    def build_cache_salt(cache_salt, extra_key):
        if cache_salt is None and extra_key is None:
            return None
        # Length-delimited JSON plus a digest avoids separator collisions in
        # LMCache's serialized tags (and distinguishes None from empty strings).
        payload = json.dumps([cache_salt, extra_key], ensure_ascii=True).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _request_configs(cache_salt):
        return {} if cache_salt is None else {"lmcache.tag.sglang_salt": cache_salt}

    def parallel_all_reduce(self, tensor, op):
        if self._tp_size > 1:
            if self._tp_group is None:
                raise RuntimeError("In-process LMCache TP requires a CPU cache group.")
            torch.distributed.all_reduce(tensor, op=op, group=self._tp_group)

    def _global_min(self, value):
        tensor = torch.tensor([value], dtype=torch.int64, device="cpu")
        self.parallel_all_reduce(tensor, torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def submit_lookup(self, request_id, token_ids, *, local_hit_tokens, cache_salt):
        del local_hit_tokens
        hit = 0
        try:
            hit = self._engine.lookup(
                token_ids,
                pin=True,
                lookup_id=request_id,
                request_configs=self._request_configs(cache_salt),
            )
        except Exception:
            logger.exception("LMCache lookup failed for %s; using prefill", request_id)
        hit = self._global_min(
            min(int(hit), len(token_ids)) // self.chunk_size * self.chunk_size
        )
        lookup = _Lookup(request_id, list(token_ids), cache_salt, hit)
        self._lookups[request_id] = lookup
        if not hit:
            self.free_lookup_locks(request_id, 0, 0)
        return lookup

    @staticmethod
    def poll_lookup(lookup):
        return lookup.total_hit

    def free_lookup_locks(self, request_id, start, end):
        lookup = self._lookups.get(request_id)
        if lookup is not None:
            lookup.lock_start = max(lookup.lock_start, end)
            # Public LMCache can unpin only an entire lookup. Retain all pins
            # until the suffix transfer finishes or the entire hit is local.
            if end >= lookup.total_hit:
                self._engine.lookup_unpin(request_id)
                self._lookups.pop(request_id, None)

    @staticmethod
    def device_indices_by_group(device_indices, *, mamba_value=None, **kwargs):
        if mamba_value is not None:
            raise ValueError("In-process LMCache cannot transfer Mamba state.")
        return device_indices

    def _transfer(self, request_id, fn, *, device_indices=None, local_hit_tokens=0):
        success = False
        # Public LMCache may use internal streams. Drain before a source lock
        # or destination allocation can be released, including exception paths.
        torch.cuda.synchronize(self._device)
        try:
            success = bool(fn())
        except Exception:
            logger.exception("LMCache transfer failed for %s", request_id)
        finally:
            torch.cuda.synchronize(self._device)
        success = bool(self._global_min(int(success)))
        future = Future()
        future.set_result(success)
        return _Transfer(request_id, future, success, device_indices, local_hit_tokens)

    def submit_load(
        self,
        lookup,
        mapping,
        *,
        local_hit_tokens,
        owned_device_indices,
        producer_stream,
    ):
        del producer_stream
        end = lookup.total_hit
        if local_hit_tokens % self.chunk_size or len(mapping) != end - local_hit_tokens:
            raise ValueError("LMCache load requires a chunk-aligned local prefix.")

        def retrieve():
            mask = torch.arange(end, device="cpu") >= local_hit_tokens
            retrieved = self._engine.retrieve(
                lookup.token_ids[:end],
                mask=mask,
                kvcaches=self._kvcaches,
                slot_mapping=mapping.to(device=self._device, dtype=torch.int64),
                offset=local_hit_tokens,
                request_configs=self._request_configs(lookup.cache_salt),
                req_id=lookup.request_id,
            )
            return torch.equal(retrieved.cpu(), mask)

        operation = self._transfer(
            lookup.request_id,
            retrieve,
            device_indices=owned_device_indices,
            local_hit_tokens=local_hit_tokens,
        )
        self.free_lookup_locks(lookup.request_id, 0, end)
        return operation

    @staticmethod
    def prepare_load_on_stream(load, forward_stream):
        return load.result

    @staticmethod
    def complete_load(load, synchronize=True):
        return load.result

    @staticmethod
    def ready_prefix_count(operations):
        return len(operations)

    def get_store_start(self, request_id, token_count):
        return min(self._store_ends.get(request_id, 0), token_count)

    def submit_store(
        self, request_id, token_ids, mapping, *, device_indices_start, cache_salt
    ):
        start, end = device_indices_start, len(token_ids)
        if (
            start % self.chunk_size
            or end % self.chunk_size
            or len(mapping) != end - start
        ):
            raise ValueError(
                "LMCache store requires chunk-aligned token and slot ranges."
            )

        def store():
            # SGLangGPUConnector.from_gpu indexes absolute token offsets, whereas
            # to_gpu accepts a suffix offset. Supply the full mapping for stores.
            slots = torch.full((end,), -1, dtype=torch.int64, device=self._device)
            slots[start:] = mapping
            configs = self._request_configs(cache_salt)
            self._engine.store(
                token_ids,
                mask=torch.arange(end, device="cpu") >= start,
                kvcaches=self._kvcaches,
                slot_mapping=slots,
                request_configs=configs,
                req_id=request_id,
            )
            # store() may decline chunks under CPU pressure without raising.
            return self._engine.lookup(token_ids, request_configs=configs) >= end

        operation = self._transfer(request_id, store)
        if operation.result:
            self._store_ends[request_id] = end
        return operation

    @staticmethod
    def complete_store(operation, synchronize=True):
        return operation.result

    def finish_request(self, request_id):
        self._engine.lookup_unpin(request_id)
        self._lookups.pop(request_id, None)
        self._store_ends.pop(request_id, None)

    def end_all_sessions(self):
        for request_id in list(self._lookups):
            self.finish_request(request_id)
        self._store_ends.clear()

    def clear(self):
        self.end_all_sessions()
        self._engine.clear()
        return True

    def close(self):
        if not self._closed:
            self.end_all_sessions()
            self._connector.close()
            self._closed = True
