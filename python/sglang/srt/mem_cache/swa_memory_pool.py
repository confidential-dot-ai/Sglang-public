import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class SWAKVPool(BaseSWAKVPool):
    """Hybrid full/SWA cache composed from independently configurable pools.

    The default remains two MHA pools. Supplying ``full_kv_pool_class`` and
    ``swa_kv_pool_class`` enables other KV cache families, including MLA/DSA,
    without adding model-specific behavior to the pool selector.
    """

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        device: str,
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        full_kv_pool_class: Optional[type] = None,
        swa_kv_pool_class: Optional[type] = None,
        full_kv_pool_kwargs: Optional[dict] = None,
        swa_kv_pool_kwargs: Optional[dict] = None,
        enable_memory_saver: bool = False,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.layer_num = self.full_layer_nums + self.swa_layer_nums
        self.start_layer = 0
        self.page_size = page_size
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        full_kv_pool_class = full_kv_pool_class or token_to_kv_pool_class
        swa_kv_pool_class = swa_kv_pool_class or token_to_kv_pool_class
        # Both sub-pools must allocate inside the torch_memory_saver region, otherwise
        # release_memory_occupation(kv_cache) cannot unmap the KV cache of SWA models
        # and a colocated trainer never gets that memory back.
        common_kwargs = {
            "page_size": page_size,
            "enable_memory_saver": enable_memory_saver,
            "device": device,
        }
        if full_kv_pool_kwargs is None:
            full_kv_pool_kwargs = {
                **common_kwargs,
                "head_num": head_num,
                "head_dim": head_dim,
                "allocation_label": "Full",
                **kwargs,
            }
            full_kv_pool_kwargs.pop("swa_head_num", None)
            full_kv_pool_kwargs.pop("swa_head_dim", None)
            full_kv_pool_kwargs.pop("swa_v_head_dim", None)
        if swa_kv_pool_kwargs is None:
            swa_kv_pool_kwargs = {
                **common_kwargs,
                "head_num": head_num,
                "head_dim": head_dim,
                "allocation_label": "SWA",
                **kwargs,
            }

        self.full_kv_pool = full_kv_pool_class(
            size=size,
            dtype=dtype,
            layer_num=self.full_layer_nums,
            **full_kv_pool_kwargs,
        )
        self.swa_kv_pool = swa_kv_pool_class(
            size=size_swa,
            dtype=dtype,
            layer_num=self.swa_layer_nums,
            **swa_kv_pool_kwargs,
        )
        for pool, layer_ids in (
            (self.full_kv_pool, full_attention_layer_ids),
            (self.swa_kv_pool, swa_attention_layer_ids),
        ):
            if isinstance(pool, MHATokenToKVPool):
                pool.model_layer_ids = tuple(layer_ids)
        self.dsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = None
        self.index_head_dim = None
        if isinstance(self.full_kv_pool, MLATokenToKVPool):
            self.dsa_kv_cache_store_fp8 = self.full_kv_pool.dsa_kv_cache_store_fp8
            self.kv_cache_dim = self.full_kv_pool.kv_cache_dim
        if isinstance(self.full_kv_pool, DSATokenToKVPool):
            self.index_head_dim = self.full_kv_pool.index_head_dim
        # {layer_id: (index, is_swa_layer)}
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            f"SWAKVPool {'VA upper bound' if self.post_capture_active else 'mem usage'}: {self.mem_usage:.2f} GB, swa size: {self.size_swa}, full size: {self.size}"
        )

    @property
    def post_capture_active(self) -> bool:
        """True iff the sub-pools took the post-capture VA-backed path (both share the flag)."""
        return self.full_kv_pool.post_capture_active

    @property
    def post_capture_backed_bytes(self) -> int:
        """Physically-backed KV bytes across both sub-pools (post-capture only)."""
        return (
            self.full_kv_pool.post_capture_backed_bytes
            + self.swa_kv_pool.post_capture_backed_bytes
        )

    def finalize_backing(self, config) -> None:
        """Back both sub-pools to their post-capture final sizes and record them."""
        self.full_kv_pool._finalize_backing_tokens(config.full_max_total_num_tokens)
        self.swa_kv_pool._finalize_backing_tokens(config.swa_max_total_num_tokens)
        self.size = int(config.full_max_total_num_tokens)
        self.size_swa = int(config.swa_max_total_num_tokens)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor):
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def register_layer_transfer_counter(self, layer_transfer_counter):
        # Wait happens at this wrapper. Inner pools must not wait again.
        self.layer_transfer_counter = layer_transfer_counter
        self.full_kv_pool.register_layer_transfer_counter(None)
        self.swa_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_kv_size_bytes(self):
        def split_size(pool):
            size = pool.get_kv_size_bytes()
            return size if isinstance(size, tuple) else (size, 0)

        k_size, v_size = split_size(self.full_kv_pool)
        k_size_swa, v_size_swa = split_size(self.swa_kv_pool)
        return k_size + k_size_swa, v_size + v_size_swa

    def is_mla(self) -> bool:
        return isinstance(self.full_kv_pool, MLATokenToKVPool) and isinstance(
            self.swa_kv_pool, MLATokenToKVPool
        )

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        return (
            full_kv_data_ptrs,
            full_kv_data_lens,
            full_kv_item_lens,
        )

    def get_kv_scale_buf_infos(self):
        return self.full_kv_pool.get_kv_scale_buf_infos()

    def get_swa_kv_scale_buf_infos(self):
        return self.swa_kv_pool.get_kv_scale_buf_infos()

    def get_state_buf_infos(self):
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        return swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_v_head_dim(self):
        # The FULL side's dim, as HybridLinearKVPool.get_v_head_dim(): a caller
        # asking a pool for "the" v_head_dim wants the full-attention geometry.
        # A window-only pool (a sliding-window MTP depth) has no full side;
        # its swa side is the only geometry there is.
        # `start_layer`, not 0, so pipeline parallelism (start_layer > 0) works,
        # and because layer 0 need not be a full-attention layer.
        lead = self.full_kv_pool if self.full_layer_nums > 0 else self.swa_kv_pool
        return lead.get_value_buffer(lead.start_layer).shape[-1]

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def get_kv_scale_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        if is_swa_layer:
            return self.swa_kv_pool.get_kv_scale_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_scale_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        # -1 in kv_indices maps to -1 via the sentinel appended to the mapping.
        return self.full_to_swa_index_mapping[kv_indices]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        # loc_info bundles the full loc and the pre-translated SWA loc.
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        layer_id = layer.layer_id
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            # swa_loc is the full->SWA translation, computed once per forward by
            # the attention backend; set_kv_buffer never translates internally.
            assert swa_loc is not None
            loc = swa_loc
        if is_swa_layer and (
            not loc.is_cuda or not torch.cuda.is_current_stream_capturing()
        ):
            # -1 is the established skip sentinel on this pool's write paths
            # (the commit gate writes it for rejected verify columns, the ring
            # publishers write it for tokens older than the window), but this
            # kernel ASSERTS on a negative index rather than skipping. Drop
            # the skip rows host-side. Eager only: during capture a sync here
            # is illegal (cudaErrorStreamCaptureUnsupported) and the capture
            # wrapper's fill(0) guarantees no -1 exists then; replay re-runs
            # kernels, not this Python.
            if loc.numel() and bool((loc < 0).any()):
                _keep = torch.nonzero(loc >= 0, as_tuple=False).flatten()
                if _keep.numel() == 0:
                    return
                loc = loc.index_select(0, _keep)
                cache_k = cache_k.index_select(0, _keep)
                cache_v = cache_v.index_select(0, _keep)
        if isinstance(pool, MLATokenToKVPool):
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                layer_id_override=layer_id_pool,
            )
        else:
            pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Commit-gated 2D write, translated into the SWA sub-pool.

        The speculative verify path writes a [bs, verify_len] block of locs and
        commits only the first ``commit_lens[i]`` columns of each row -- the
        rest are rejected draft tokens whose KV must not land. On a hybrid SWA
        pool those locs are in the FULL index space, so a sliding layer needs
        them translated before they mean anything, exactly as ``set_kv_buffer``
        requires a pre-resolved ``swa_loc``.

        Two gates, not one, and the second is what makes this different from
        the dense pool's version:

          * ``col < commit_lens[i]`` -- the caller's commit gate.
          * ``swa_loc >= 0`` -- the ring's OWN gate. A ring-pooled draft
            publishes -1 for any token that has fallen out of its sliding
            window (older tokens share a ring slot with a live token and would
            race). Writing a -1 would index the buffer's LAST row and silently
            corrupt whichever token owns it, so the sentinel has to be dropped
            here rather than trusted to the callee.

        Both gates are resolved on the host side into a flat valid-row gather
        and one ``set_kv_buffer`` call. That gives up the dense pool's fused
        tiled kernel, which cannot express the second gate; the cost is a
        gather over at most ``max_running_requests * verify_len`` rows (~1.2k
        for M3 at 128 x 9), which is negligible next to the forward itself.
        """
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )
        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} "
                f"{tuple(loc_2d.shape)=}."
            )

        flat = loc_2d.reshape(-1).to(torch.int64)
        if is_swa_layer:
            flat = self.translate_loc_from_full_to_swa(flat).to(torch.int64)

        cols = torch.arange(loc_2d.shape[1], device=loc_2d.device)
        valid = cols[None, :] < commit_lens.to(torch.int64)[:, None]
        valid = valid.reshape(-1)
        if is_swa_layer:
            valid &= flat >= 0
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return

        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        loc = flat.index_select(0, valid_idx)
        k = cache_k.index_select(0, valid_idx)
        v = cache_v.index_select(0, valid_idx)
        if isinstance(pool, MLATokenToKVPool):
            pool.set_kv_buffer(None, loc, k, v, layer_id_override=layer_id_pool)
        else:
            pool.set_kv_buffer(
                None, loc, k, v, k_scale, v_scale, layer_id_override=layer_id_pool
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc, swa_loc, _ = unwrap_write_loc(loc_info)
        layer_id_pool, is_swa_layer = self.layers_mapping[layer.layer_id]
        pool = self.swa_kv_pool if is_swa_layer else self.full_kv_pool
        if is_swa_layer:
            assert swa_loc is not None
            loc = swa_loc
        if not isinstance(pool, MLATokenToKVPool):
            raise TypeError(f"Layer {layer.layer_id} is not backed by an MLA KV pool")
        pool.set_mla_kv_buffer(
            None,
            loc,
            cache_k_nope,
            cache_k_rope,
            layer_id_override=layer_id_pool,
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_with_scale_buffer(layer_id_pool)

    def get_index_k_scale_buffer(self, layer_id: int, *args, **kwargs):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.get_index_k_scale_buffer(
            layer_id_pool, *args, **kwargs
        )

    def set_index_k_scale_buffer(self, layer_id: int, *args, **kwargs):
        layer_id_pool, is_swa_layer = self.layers_mapping[layer_id]
        assert not is_swa_layer
        return self.full_kv_pool.set_index_k_scale_buffer(
            layer_id_pool, *args, **kwargs
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)
        tgt_loc_swa = self.translate_loc_from_full_to_swa(tgt_loc)
        src_loc_swa = self.translate_loc_from_full_to_swa(src_loc)
        self.swa_kv_pool.move_kv_cache(tgt_loc_swa, src_loc_swa)

    def _filter_swa_cpu_copy(self, *, swa_kv_cpu, row_mask: torch.Tensor):
        if swa_kv_cpu is None:
            return None
        if row_mask is None or bool(torch.all(row_mask).item()):
            return swa_kv_cpu

        chunk_size = getattr(
            self.swa_kv_pool, "cpu_offloading_chunk_size", len(row_mask)
        )
        filtered = []
        for layer_chunks in swa_kv_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue

            # A chunk is whatever the sub-pool produced: k/v, plus the block
            # scales for a quantized pool. Filter every tensor it carries.
            num_tensors = len(layer_chunks[0])
            tensors = [
                torch.cat([chunk[t] for chunk in layer_chunks], dim=0)[row_mask]
                for t in range(num_tensors)
            ]

            filtered_layer = []
            for i in range(0, len(tensors[0]), chunk_size):
                filtered_layer.append([t[i : i + chunk_size] for t in tensors])
            filtered.append(filtered_layer)
        return filtered

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # For SWA, we need to copy KV cache from both full and SWA pools
        # The indices are for the full pool, and we use mapping to get SWA indices
        full_kv_cpu = self.full_kv_pool.get_cpu_copy(
            indices, req_pool_index=req_pool_index
        )

        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            # Slot 0 is reserved as a dummy slot. Tail-only SWA allocations leave
            # the out-of-window full KV indices unmapped, so only copy mapped SWA
            # tokens and keep their positions for load_cpu_copy().
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                swa_kv_cpu = self.swa_kv_pool.get_cpu_copy(swa_indices[swa_mask])
                swa_mask = swa_mask.cpu()
            else:
                swa_kv_cpu = None
        else:
            swa_kv_cpu = None

        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        # Load KV cache back from CPU to both full and SWA pools
        # Note: indices here are NEW indices (newly allocated), different from get_cpu_copy indices
        full_kv_cpu = kv_cache_cpu["full"]
        swa_kv_cpu = kv_cache_cpu["swa"]

        # Load full KV cache to the new indices
        self.full_kv_pool.load_cpu_copy(
            full_kv_cpu, indices, req_pool_index=req_pool_index
        )

        # Load SWA KV cache if it exists
        if swa_kv_cpu is not None and self.full_to_swa_index_mapping is not None:
            swa_indices = self.full_to_swa_index_mapping[indices]
            new_swa_mask = swa_indices > 0
            old_swa_mask = kv_cache_cpu.get("swa_mask")
            if old_swa_mask is not None:
                old_swa_mask = old_swa_mask.to(indices.device)
                row_mask = new_swa_mask[old_swa_mask].cpu()
                swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
            else:
                row_mask = new_swa_mask.cpu()
                swa_indices = swa_indices[new_swa_mask]

            if swa_indices.numel() == 0:
                return

            swa_kv_cpu = self._filter_swa_cpu_copy(
                swa_kv_cpu=swa_kv_cpu, row_mask=row_mask
            )
            self.swa_kv_pool.load_cpu_copy(swa_kv_cpu, swa_indices)
