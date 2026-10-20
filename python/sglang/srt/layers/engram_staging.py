"""Eager CPU Engram lookup for GPUs without mapped host-memory access.

The immutable safetensors mappings share the OS page cache across TP groups.
Only selected rows (FP8, or packed MXFP4) and their scales cross PCIe. CPU
gathering runs eagerly between CUDA graph segments; dequantization stays inside
the next segment.

With SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH, a step (prefill, decode or verify)
copies the ids of all staged layers to the host once, right after hashing. Small
steps (decode, verify) then gather and copy every layer's rows to the GPU at once,
so the later layer breaks neither wait nor drain the GPU. Large prefill chunks
instead gather on a worker thread in layer order while the GPU runs the layers
before each lookup, and the break at a layer waits for its rows and copies them.
The bridge buffers and break order are unchanged, so graph capture and replay
see the same segments plus the one extra break that starts the gathers.

The copy to the GPU stays on the compute stream at the break. Issuing it from the
worker on a side stream, overlapping graph replay, hung TP2 decode under CC
(2026-10-17: both ranks spun at 100% GPU with the scheduler blocked on a sync).
"""

import ctypes
import json
import logging
import mmap
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)

logger = logging.getLogger(__name__)


_MADV_POPULATE_READ = 22  # Linux >= 5.14


def _map_into_page_tables(tensor):
    # The holder keeps the file in the page cache, but this process still takes
    # a minor fault on each first touch: measured 7.2 ms vs 1.1 ms per 2.5K-row
    # gather (2026-10-17). Populating maps the cached pages (huge pages, ~0.1 s).
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    start = tensor.data_ptr() // mmap.PAGESIZE * mmap.PAGESIZE
    end = tensor.data_ptr() + tensor.untyped_storage().nbytes()
    if libc.madvise(start, end - start, _MADV_POPULATE_READ) != 0:
        logger.warning(
            "Could not pre-map Engram rows (errno %d); gathers will page-fault",
            ctypes.get_errno(),
        )


# A graph replays fixed buffers, so address and layout identify an ids tensor.
def _ids_key(indices):
    return (indices.data_ptr(), tuple(indices.shape), tuple(indices.stride()))


class StagedEngramTable:
    def __init__(self, model_path, layer_id, rows, dim, *, fp4=False):
        root = Path(model_path)
        index = json.loads((root / "model.safetensors.index.json").read_text())
        tensors = []
        # MXFP4 rows pack two e2m1 values per byte, low nibble first.
        weight_layout = (
            ((rows, dim // 2), torch.uint8)
            if fp4
            else ((rows, dim), torch.float8_e4m3fn)
        )
        for suffix, shape, dtype in (
            ("weight", *weight_layout),
            ("scale", (rows, dim // 32), torch.float8_e8m0fnu),
        ):
            key = f"layers.{layer_id}.engram.embed.{suffix}"
            path = root / index["weight_map"][key]
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                tensor = checkpoint.get_tensor(key)
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(
                    f"Unexpected Engram tensor {key}: {tensor.shape}, {tensor.dtype}"
                )
            # Tensor storage keeps the file mapping alive after safe_open closes.
            # Never write to it: writes would create private COW pages.
            _map_into_page_tables(tensor)
            tensors.append(tensor)
        self.weight, self.scale = tensors
        self.dim = dim
        self.fp4 = fp4
        # (_ids_key of the ids, Future of their rows) set by StagedEngramPrefetcher.
        self.prefetched: Optional[tuple[tuple, Future]] = None

    def gather_bytes(self, indices):
        return self.gather_rows(indices.reshape(-1).to(device="cpu", dtype=torch.int64))

    def gather_rows(self, ids):
        # Index uint8 views: CPU index_select need not support FP8 dtypes.
        return (
            self.weight.view(torch.uint8).index_select(0, ids),
            self.scale.view(torch.uint8).index_select(0, ids),
        )

    def _take_prefetched(self, indices):
        prefetched, self.prefetched = self.prefetched, None
        if prefetched is None or prefetched[0] != _ids_key(indices):
            return None
        return prefetched[1].result()

    @eager_on_graph(True)
    def _stage_bytes(self, indices):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CPU Engram staging requires breakable CUDA graphs")
        rows = self._take_prefetched(indices)
        weight, scale = self.gather_bytes(indices) if rows is None else rows
        return weight.to(indices.device), scale.to(indices.device)

    def lookup(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        out = torch.empty(
            (*indices.shape, self.dim), dtype=torch.bfloat16, device=indices.device
        )
        if not indices.numel():
            return out
        # The breakable runner retains these bridge buffers and refreshes them
        # on every replay. Dequantization and subsequent GPU work stay captured.
        weight, scale = self._stage_bytes(indices)
        local_ids = torch.arange(indices.numel(), device=indices.device)
        engram_gather(
            weight.data_ptr(),
            scale.data_ptr(),
            local_ids,
            out.view(-1, self.dim),
            self.dim,
            32,
            fp4=self.fp4,
        )
        return out


def _set_gather_threads(num_threads):
    if num_threads > 0:
        torch.set_num_threads(num_threads)


def _done(result):
    future = Future()
    future.set_result(result)
    return future


def _skip_at_capture(prefetcher, hash_ids):
    # Capture only records the break; its hash ids are not computed yet.
    return None


class StagedEngramPrefetcher:
    def __init__(self, tables, gather_threads, inline_max_tokens=1024):
        # Hash column (Engram.layer_hash_index) -> that layer's table.
        self.tables = tables
        self.gather_threads = gather_threads
        # Steps up to this many tokens (decode, verify) stage every layer here;
        # larger prefill chunks gather on the worker while earlier layers run.
        self.inline_max_tokens = inline_max_tokens
        # Created on first use: a model holding this object stays copyable.
        self.executor: Optional[ThreadPoolExecutor] = None

    # hash_ids: [num_tokens, n_engram_layers, n_hash_cols] int64 from EngramHasher.
    @eager_on_graph(True, capture_stub=_skip_at_capture)
    def start(self, hash_ids):
        if hash_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CPU Engram staging requires breakable CUDA graphs")
        for table in self.tables.values():
            table.prefetched = None
        if not hash_ids.numel():
            return
        ids = hash_ids.to(device="cpu", dtype=torch.int64)
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="engram-gather",
                initializer=_set_gather_threads,
                initargs=(self.gather_threads,),
            )
        inline = hash_ids.shape[0] <= self.inline_max_tokens
        # Inline, this thread gathers the first layer while the worker does the rest.
        columns = list(self.tables.items())[1:] if inline else self.tables.items()
        futures = {
            column: self.executor.submit(table.gather_rows, ids[:, column].reshape(-1))
            for column, table in columns
        }
        if inline:
            # The copy to ids above already waited for the GPU, so stage every
            # layer now; the layer breaks then find rows on the GPU and do not sync.
            first, first_table = next(iter(self.tables.items()))
            futures[first] = _done(first_table.gather_rows(ids[:, first].reshape(-1)))
            futures = {
                column: _done(
                    tuple(rows.to(hash_ids.device) for rows in future.result())
                )
                for column, future in futures.items()
            }
        for column, table in self.tables.items():
            table.prefetched = (_ids_key(hash_ids[:, column]), futures[column])
