"""Debug checksums for chasing run-to-run divergence across PD.

With SGLANG_DEBUG_PD_DIVERGENCE set, prefill logs one line per sent KV chunk
(per PP stage) with a checksum of every transfer buffer's pages, and the aux
metadata slot it sends with the last chunk; decode logs the aux slot it received
and its first sampled token. Lines carry the bootstrap room, so two runs of one
prompt, or prefill against decode, can be diffed line by line.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

LAYER_PROBES_ENABLED = envs.SGLANG_DEBUG_PD_DIVERGENCE_LAYERS.get()
_LAYER_PROBE_POINTS = (
    "in",
    "index_q",
    "index_w",
    "scores",
    "scores_again",
    "selected",
    "attn",
    "out",
)
_LAYER_PROBE_ROW_STRIDE = 16
# Arbitrary bounds above any prefill graph bucket and layer count; rows past
# them are not sampled.
_LAYER_PROBE_MAX_ROWS = 32768
_LAYER_PROBE_MAX_LAYERS = 64
_LAYER_PROBE_DUMP_DIR = "/tmp/pd-divergence-layers"
_layer_probe_buffer: Optional[torch.Tensor] = None
_probed_layers: set = set()


def ensure_layer_probe_buffer(device: torch.device) -> None:
    """Allocate before graph capture: a probe that runs first under capture is a no-op."""
    global _layer_probe_buffer
    if LAYER_PROBES_ENABLED and _layer_probe_buffer is None:
        _layer_probe_buffer = torch.zeros(
            (
                _LAYER_PROBE_MAX_LAYERS,
                len(_LAYER_PROBE_POINTS),
                _LAYER_PROBE_MAX_ROWS // _LAYER_PROBE_ROW_STRIDE,
            ),
            dtype=torch.int64,
            device=device,
        )


def layer_probe(
    *,
    layer_id: int,
    point: str,
    tensor,
    row_offset: int = 0,
    valid_cols: Optional[torch.Tensor] = None,
) -> None:
    """Hash every 16th row of ``tensor`` into the probe buffer; graph-capturable.

    Rows are hashed separately because padded graph rows hold garbage: the host
    combines only the live ones. ``valid_cols`` zeroes each row past its count.
    """
    buf = _layer_probe_buffer
    if buf is None or not isinstance(tensor, torch.Tensor) or tensor.shape[0] == 0:
        return
    _probed_layers.add(layer_id)
    stride = _LAYER_PROBE_ROW_STRIDE
    sample = tensor.reshape(tensor.shape[0], -1)[::stride]
    if valid_cols is not None:
        cols = torch.arange(sample.shape[1], device=tensor.device)
        lens = valid_cols[::stride].reshape(-1, 1)
        sample = sample.masked_fill(cols[None, :] >= lens, 0)
    if sample.shape[1] * sample.element_size() % 4 == 0:
        # Rows keep a unit last stride, so this reinterprets without a copy.
        words = sample.view(torch.int32)
    else:
        words = sample.contiguous().view(torch.uint8)
    weights = torch.arange(
        1, words.shape[1] + 1, dtype=torch.int32, device=tensor.device
    )
    # int32 products wrap; the int64 row sum keeps every bit of each row.
    row_hashes = (words.to(torch.int32) * weights).sum(dim=1, dtype=torch.int64)
    start = row_offset // stride
    count = min(row_hashes.shape[0], buf.shape[2] - start)
    if count > 0:
        buf[layer_id, _LAYER_PROBE_POINTS.index(point), start : start + count].copy_(
            row_hashes[:count]
        )


class _DeviceBytes:
    """Exposes raw device memory to ``torch.as_tensor`` without a copy."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PageChecksummer:
    """Order-sensitive checksums of selected items of registered PD buffers.

    Buffer ``i`` is ``data_lens[i]`` bytes at ``ptrs[i]``, in items of
    ``item_lens[i]`` bytes, addressed by the same indices the transfer uses.
    """

    def __init__(
        self,
        *,
        ptrs: Sequence[int],
        data_lens: Sequence[int],
        item_lens: Sequence[int],
        device: torch.device,
    ):
        self._device = torch.device(device)
        self._views: List[Optional[torch.Tensor]] = []
        with torch.cuda.device(self._device):
            for ptr, data_len, item_len in zip(ptrs, data_lens, item_lens):
                num_items = data_len // item_len if item_len else 0
                if num_items == 0 or item_len % 4 or ptr % 4:
                    self._views.append(None)
                    continue
                raw = torch.as_tensor(
                    _DeviceBytes(int(ptr), num_items * item_len), device=self._device
                )
                self._views.append(raw.view(torch.int32).view(num_items, item_len // 4))
        self._streams = threading.local()

    def sums(self, indices: Sequence[int], first_position: int = 0) -> List[int]:
        """One checksum per buffer; item ``k`` is weighted by ``first_position + k``,
        its logical position in the request, so reordering changes the sum."""
        host_idx = np.asarray(indices, dtype=np.int64)
        if host_idx.size == 0:
            return [0] * len(self._views)
        lo, hi = int(host_idx.min()), int(host_idx.max())
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream()):
            idx = torch.as_tensor(host_idx, device=self._device)
            positions = torch.arange(
                first_position + 1,
                first_position + 1 + idx.numel(),
                dtype=torch.int64,
                device=self._device,
            )
            out = []
            for view in self._views:
                # Checked on the host: an out-of-range gather is a device assert.
                if view is None or lo < 0 or hi >= view.shape[0]:
                    out.append(torch.zeros((), dtype=torch.int64, device=self._device))
                    continue
                rows = view.index_select(0, idx).to(torch.int64)
                word_weights = torch.arange(
                    1, rows.shape[1] + 1, dtype=torch.int64, device=self._device
                )
                out.append(((rows * word_weights).sum(1) * positions).sum())
            return torch.stack(out).tolist()

    def _stream(self) -> torch.cuda.Stream:
        # A side stream per thread, so a transfer worker never orders itself
        # against the scheduler's streams.
        stream = getattr(self._streams, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self._device)
            self._streams.stream = stream
        return stream


class ProxyChecksums:
    """Checksums of the hidden states one PP stage sends and the next receives.

    Hashing runs on a side stream after the producing event and is logged once
    it has finished, so the scheduler never blocks on it: a sync here would
    change the stage timing being debugged.
    """

    # Rows hashed per pass: ~15 MB of int64 temporaries for a DSV4 mHC row,
    # since prefill runs near its memory limit.
    _ROWS_PER_PASS = 128

    def __init__(self, device: torch.device):
        self._device = torch.device(device)
        self._stream = torch.cuda.Stream(device=self._device)
        self._pending = []

    def submit(self, *, side: str, tensors: dict, ready_event, batch):
        """Returns the event marking the end of the hashing."""
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            if ready_event is not None:
                self._stream.wait_event(ready_event)
            sums = {}
            for name, tensor in tensors.items():
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    continue
                tensor.record_stream(self._stream)
                sums[name] = _tensor_sum(tensor, self._ROWS_PER_PASS)
            done = torch.cuda.Event()
            done.record(self._stream)
        reqs = _describe_reqs(batch)

        def emit():
            values = " ".join(
                f"{name}={int(value.item()) & 0xFFFFFFFFFFFFFFFF:016x}"
                for name, value in sums.items()
            )
            logger.info(f"PD divergence pp-proxy {side} reqs=[{reqs}] {values}")

        self._pending.append((done, emit))
        self.drain()
        return done

    def begin_layer_probes(self, stream: torch.cuda.Stream) -> None:
        """Clear the probe buffer ahead of the next forward on ``stream``."""
        if _layer_probe_buffer is not None:
            with torch.cuda.stream(stream):
                _layer_probe_buffer.zero_()

    def submit_layer_probes(self, *, ready_event, batch) -> Optional[torch.cuda.Event]:
        """Copy this stage's probe rows to the host after ``ready_event``; returns
        the event the next forward must wait on before it overwrites them."""
        buf = _layer_probe_buffer
        if buf is None or not _probed_layers:
            return None
        lo, hi = min(_probed_layers), max(_probed_layers) + 1
        with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
            self._stream.wait_event(ready_event)
            host = torch.empty(
                (hi - lo, *buf.shape[1:]), dtype=buf.dtype, pin_memory=True
            )
            host.copy_(buf[lo:hi], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self._stream)
        reqs = _describe_reqs(batch)
        live_rows = sum(req.extend_len for req in batch.reqs)
        dump_name = (
            f"{batch.reqs[0].bootstrap_room}_{batch.reqs[0].prefix_len}"
            if len(batch.reqs) == 1 and live_rows > 1
            else None
        )
        self._pending.append(
            (
                done,
                lambda: _emit_layer_probes(
                    host=host.numpy(),
                    first_layer=lo,
                    live_rows=live_rows,
                    reqs=reqs,
                    dump_name=dump_name,
                ),
            )
        )
        self.drain()
        return done

    def drain(self) -> None:
        while self._pending and self._pending[0][0].query():
            _, emit = self._pending.pop(0)
            emit()


def _describe_reqs(batch) -> str:
    return ",".join(
        f"{req.bootstrap_room}:{req.prefix_len}+{req.extend_len}" for req in batch.reqs
    )


def _emit_layer_probes(
    *, host: np.ndarray, first_layer: int, live_rows: int, reqs: str, dump_name
) -> None:
    rows = host[:, :, : -(-live_rows // _LAYER_PROBE_ROW_STRIDE)].astype(np.uint64)
    weights = np.arange(1, rows.shape[2] + 1, dtype=np.uint64)
    for offset, layer_rows in enumerate(rows):
        values = " ".join(
            f"{point}={int((point_rows * weights).sum()):016x}"
            for point, point_rows in zip(_LAYER_PROBE_POINTS, layer_rows)
            if point_rows.any()
        )
        if values:
            logger.info(
                f"PD divergence layer={first_layer + offset} reqs=[{reqs}] "
                f"live_rows={live_rows} {values}"
            )
    if dump_name is not None:
        os.makedirs(_LAYER_PROBE_DUMP_DIR, exist_ok=True)
        # Indexed by global layer id: [layer, point, sampled row].
        np.save(
            f"{_LAYER_PROBE_DUMP_DIR}/{dump_name}_{os.getpid()}_{time.time():.3f}.npy",
            np.concatenate([np.zeros((first_layer, *rows.shape[1:]), np.uint64), rows]),
        )


def _tensor_sum(tensor: torch.Tensor, rows_per_pass: int) -> torch.Tensor:
    flat = tensor.reshape(tensor.shape[0], -1) if tensor.dim() > 1 else tensor.view(-1, 1)
    words = flat.contiguous().view(torch.uint8)
    if words.shape[1] % 4 == 0:
        words = words.view(torch.int32)
    total = torch.zeros((), dtype=torch.int64, device=tensor.device)
    weights = torch.arange(1, words.shape[1] + 1, dtype=torch.int64, device=tensor.device)
    for start in range(0, words.shape[0], rows_per_pass):
        rows = words[start : start + rows_per_pass].to(torch.int64)
        positions = torch.arange(
            start + 1, start + 1 + rows.shape[0], dtype=torch.int64, device=tensor.device
        )
        total += ((rows * weights).sum(1) * positions).sum()
    return total


def kv_checksummer(kv_args) -> PageChecksummer:
    """Checksummer over the KV pages PD transfers, which always live on the GPU."""
    return PageChecksummer(
        ptrs=kv_args.kv_data_ptrs,
        data_lens=kv_args.kv_data_lens,
        item_lens=kv_args.kv_item_lens,
        device=torch.device(f"cuda:{kv_args.gpu_id}"),
    )


def metadata_slot_sums(buffers: Sequence[torch.Tensor], index: int) -> List[int]:
    """One checksum per metadata buffer for slot ``index``.

    Reads the tensors rather than raw pointers: the buffers can be host memory,
    which a GPU kernel must not touch.
    """
    sums = []
    for buf in buffers:
        words = buf[index].contiguous().view(torch.uint8).to(torch.int64)
        weights = torch.arange(1, words.numel() + 1, dtype=torch.int64, device=buf.device)
        sums.append(int((words * weights).sum().item()))
    return sums


def format_sums(sums: Sequence[int]) -> str:
    total = 0
    for value in sums:
        total = (total * 1000003 + value) & 0xFFFFFFFFFFFFFFFF
    return f"total={total:016x} per_buffer=[{','.join(f'{v & 0xFFFFFFFF:08x}' for v in sums)}]"


def log_prefill_chunk(
    *,
    kv: PageChecksummer,
    room: int,
    pp_rank: int,
    page_indices: Sequence[int],
    first_page_position: int,
    is_last_chunk: bool,
) -> None:
    kv_sums = kv.sums(page_indices, first_page_position)
    logger.info(
        f"PD divergence prefill room={room} pp={pp_rank} t={time.time():.3f} "
        f"pages=[{first_page_position},{first_page_position + len(page_indices)}) "
        f"last={int(is_last_chunk)} kv {format_sums(kv_sums)}"
    )


def log_metadata_slot(
    *, side: str, room: int, buffers: Sequence[torch.Tensor], index: int
) -> None:
    logger.info(
        f"PD divergence {side} room={room} t={time.time():.3f} "
        f"aux {format_sums(metadata_slot_sums(buffers, index))}"
    )


def log_first_tokens(*, reqs, next_token_ids: Sequence[int], logits) -> None:
    top = None
    if logits is not None and logits.dim() == 2 and logits.shape[0] >= len(reqs):
        values, ids = logits[: len(reqs)].float().topk(5, dim=-1)
        top = (values.tolist(), ids.tolist())
    for i, (req, token_id) in enumerate(zip(reqs, next_token_ids)):
        detail = ""
        if top is not None:
            detail = " top5=" + ",".join(
                f"{tid}:{val:.4f}" for val, tid in zip(top[0][i], top[1][i])
            )
        logger.info(
            f"PD divergence decode-first room={req.bootstrap_room} "
            f"t={time.time():.3f} token={token_id}{detail}"
        )
