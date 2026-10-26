"""CED boundary handoff: the encoder's layer-boundary state for each prompt's last
SWA window, computed on prefill and replayed from on decode.

Under the CED split a decode tail replay reran the encoder (layers before the
last kv_source layer) over every tail row only to rebuild the hidden state the
late layers start from; prefill had computed that state already, and its encoder
KV reaches decode through the ordinary transfer. With the handoff, the prefill
stage that runs the last kv_source layer captures that layer's input state (the
mHC residual and its pre-mix) for the rows the late layers run over -- the
prompt's last SWA_WINDOW positions -- into the request's PD metadata slot, and
decode starts its tail replay at that layer.

Slot layout (one uint8 row per metadata slot):
    header  int64[8]: magic, bootstrap_room, window_start, window_rows, has_pre,
                      rows_written
    residual  bf16[window, hc_mult, hidden]
    pre       fp32[window, hc_mult]
A slot whose header does not carry this request's room, or whose window was not
fully computed on prefill (a prefix-cache hit inside it), is rejected and the
request replays its encoder as before.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Tail batches that re-ran the encoder because a request had no handoff state.
_fallback_batches = 0
_fallback_reqs = 0
_fallback_logged_at = 0.0

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_MAGIC = 0x4345_4442_4E44  # "CEDBND"
_HEADER_INT64 = 8
_HEADER_BYTES = _HEADER_INT64 * 8
# Header fields.
_H_MAGIC, _H_ROOM, _H_START, _H_ROWS, _H_HAS_PRE, _H_WRITTEN = range(6)


def ced_boundary_enabled() -> bool:
    return (
        envs.SGLANG_OPT_DSV41_CED_SPLIT.get()
        and envs.SGLANG_OPT_DSV41_CED_BOUNDARY_HANDOFF.get()
    )


class CedBoundaryLayout(msgspec.Struct, frozen=True):
    window: int
    hc_mult: int
    hidden_size: int

    @property
    def residual_bytes(self) -> int:
        return self.window * self.hc_mult * self.hidden_size * 2

    @property
    def pre_bytes(self) -> int:
        return self.window * self.hc_mult * 4

    @property
    def nbytes(self) -> int:
        # 64 B aligned so every slot row starts RDMA- and dtype-aligned.
        n = _HEADER_BYTES + self.residual_bytes + self.pre_bytes
        return (n + 63) // 64 * 64

    def views(
        self, slot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(header int64[8], residual bf16[w, hc, h], pre fp32[w, hc]) of a slot row."""
        header = slot[:_HEADER_BYTES].view(torch.int64)
        start = _HEADER_BYTES
        residual = (
            slot[start : start + self.residual_bytes]
            .view(torch.bfloat16)
            .view(self.window, self.hc_mult, self.hidden_size)
        )
        start += self.residual_bytes
        pre = (
            slot[start : start + self.pre_bytes]
            .view(torch.float32)
            .view(self.window, self.hc_mult)
        )
        return header, residual, pre


def maybe_ced_boundary_layout(
    hf_text_config, sliding_window_size: Optional[int], *, holds_boundary: bool
) -> Optional[CedBoundaryLayout]:
    """The slot layout when this worker sends or receives boundary state: decode,
    and the prefill pipeline stage that runs the last kv_source layer."""
    if not ced_boundary_enabled() or not holds_boundary:
        return None
    assert sliding_window_size, "the CED boundary handoff needs the SWA window"
    return CedBoundaryLayout(
        window=int(sliding_window_size),
        hc_mult=int(hf_text_config.hc_mult),
        hidden_size=int(hf_text_config.hidden_size),
    )


def window_start(seq_len: int, window: int) -> int:
    return max(0, seq_len - window)


# --- prefill: capture --------------------------------------------------------


class CedBoundaryCapture(msgspec.Struct, kw_only=True):
    """The rows of one prefill forward that fall in their prompt's last window.

    Built on the host from the ScheduleBatch; the model fills ``residual`` and
    ``pre`` (GPU, row order of ``rows``) at the boundary layer, and the result
    processor scatters them into the requests' metadata slots.
    """

    rows: torch.Tensor  # int64 device indices into the forward's token rows
    # Per captured request: (batch request index, first absolute position, count).
    spans: List[Tuple[int, int, int]]
    residual: Optional[torch.Tensor] = None
    pre: Optional[torch.Tensor] = None


def build_ced_boundary_capture(
    *,
    reqs: List["Req"],
    extend_prefix_lens: List[int],
    extend_lens: List[int],
    window: int,
    device: torch.device,
) -> Optional[CedBoundaryCapture]:
    rows: List[int] = []
    spans: List[Tuple[int, int, int]] = []
    offset = 0
    for i, (req, prefix, n) in enumerate(zip(reqs, extend_prefix_lens, extend_lens)):
        start = max(prefix, window_start(len(req.origin_input_ids), window))
        end = prefix + n
        if start < end:
            rows.extend(range(offset + start - prefix, offset + n))
            spans.append((i, start, end - start))
        offset += n
    if not spans:
        return None
    return CedBoundaryCapture(
        rows=torch.tensor(rows, dtype=torch.int64).to(device, non_blocking=True),
        spans=spans,
    )


def ced_boundary_for_forward(
    *, batch, forward_batch, window: Optional[int], device: torch.device
):
    """The forward's capture spec (prefill) or injection (decode), if any."""
    from sglang.srt.runtime_context import get_disagg, get_parallel

    if batch.ced_boundary_injection is not None:
        return (
            batch.ced_boundary_injection
            if forward_batch.forward_mode.is_extend()
            else None
        )
    if (
        not window
        or get_disagg().disaggregation_mode != "prefill"
        or not get_parallel().pp_group.is_last_rank
        or not forward_batch.forward_mode.is_extend()
        or forward_batch.extend_seq_lens_cpu is None
    ):
        return None
    return build_ced_boundary_capture(
        reqs=batch.reqs,
        extend_prefix_lens=forward_batch.extend_prefix_lens_cpu,
        extend_lens=forward_batch.extend_seq_lens_cpu,
        window=window,
        device=device,
    )


def fill_ced_boundary_for_forward(
    forward_batch, residual: torch.Tensor, pre: Optional[torch.Tensor]
) -> None:
    capture = forward_batch.ced_boundary
    if isinstance(capture, CedBoundaryCapture):
        capture.residual = residual.index_select(0, capture.rows)
        capture.pre = None if pre is None else pre.index_select(0, capture.rows)


def write_ced_boundary_slots(
    *,
    capture: Optional[CedBoundaryCapture],
    reqs: List["Req"],
    slots: torch.Tensor,
    layout: CedBoundaryLayout,
) -> None:
    """Scatter one forward's captured rows into the requests' metadata slots.

    Runs after the forward's results are ready. The slot header counts the window
    rows written for its room; ``finalize_ced_boundary_slot`` validates it.
    """
    if not isinstance(capture, CedBoundaryCapture) or capture.residual is None:
        return
    residual = capture.residual.to("cpu", non_blocking=False)
    pre = None if capture.pre is None else capture.pre.to("cpu", non_blocking=False)
    row = 0
    for batch_index, start, count in capture.spans:
        req = reqs[batch_index]
        if req.metadata_buffer_index >= 0:
            header, slot_residual, slot_pre = layout.views(
                slots[req.metadata_buffer_index]
            )
            room = int(req.bootstrap_room or 0)
            if int(header[_H_ROOM]) != room or int(header[_H_MAGIC]) != 0:
                # A slot freshly holding this request (or a stale finalized one).
                header.zero_()
                header[_H_ROOM] = room
            lo = start - window_start(len(req.origin_input_ids), layout.window)
            slot_residual[lo : lo + count].copy_(residual[row : row + count])
            if pre is not None:
                slot_pre[lo : lo + count].copy_(pre[row : row + count])
            header[_H_WRITTEN] += count
            header[_H_HAS_PRE] = int(pre is not None)
        row += count


def finalize_ced_boundary_slot(
    *, req: "Req", slots: torch.Tensor, layout: CedBoundaryLayout
) -> None:
    """Stamp the header at the last chunk; a window prefill did not fully compute
    (a prefix-cache hit inside it) is marked invalid so decode replays."""
    header, _, _ = layout.views(slots[req.metadata_buffer_index])
    room = int(req.bootstrap_room or 0)
    seq_len = len(req.origin_input_ids)
    rows = min(seq_len, layout.window)
    written = int(header[_H_WRITTEN]) if int(header[_H_ROOM]) == room else 0
    has_pre = int(header[_H_HAS_PRE]) if written else 0
    header.zero_()
    header[_H_MAGIC] = _MAGIC if written == rows else 0
    header[_H_ROOM] = room
    header[_H_START] = window_start(seq_len, layout.window)
    header[_H_ROWS] = rows
    header[_H_HAS_PRE] = has_pre


# --- decode: read and inject --------------------------------------------------


class CedBoundaryState(msgspec.Struct, frozen=True):
    """A request's handed-off window state, on the host until its tail replays."""

    residual: torch.Tensor  # bf16[rows, hc, hidden], CPU
    pre: Optional[torch.Tensor]  # fp32[rows, hc], CPU


def take_ced_boundary_slot(
    *, req: "Req", slots: torch.Tensor, idx: int, layout: CedBoundaryLayout
) -> Optional[CedBoundaryState]:
    """Copy a request's window state out of its metadata slot before the slot is
    freed; None when prefill did not hand one over."""
    header, residual, pre = layout.views(slots[idx])
    seq_len = len(req.origin_input_ids)
    rows = min(seq_len, layout.window)
    valid = (
        int(header[_H_MAGIC]) == _MAGIC
        and int(header[_H_ROOM]) == int(req.bootstrap_room or 0)
        and int(header[_H_START]) == window_start(seq_len, layout.window)
        and int(header[_H_ROWS]) == rows
    )
    has_pre = bool(int(header[_H_HAS_PRE]))
    header.zero_()
    if not valid:
        return None
    return CedBoundaryState(
        residual=residual[:rows].clone(),
        pre=pre[:rows].clone() if has_pre else None,
    )


def _note_fallback(missing: int, total: int) -> None:
    """Warn (at most every 30 s) that tails are re-running the encoder."""
    global _fallback_batches, _fallback_reqs, _fallback_logged_at
    _fallback_batches += 1
    _fallback_reqs += missing
    now = time.monotonic()
    if now - _fallback_logged_at >= 30:
        _fallback_logged_at = now
        logger.warning(
            "CED boundary handoff missing: this batch %d/%d requests lack state; "
            "%d tail batches (%d requests) re-ran the encoder so far",
            missing,
            total,
            _fallback_batches,
            _fallback_reqs,
        )


class CedBoundaryInjection(msgspec.Struct, frozen=True):
    """A tail replay's handed-off state, in the late-layer tail's row order."""

    residual: torch.Tensor  # bf16[tail_rows, hc, hidden], device
    pre: Optional[torch.Tensor]  # fp32[tail_rows, hc], device
    # Debug: also run the encoder and compare its boundary state with this one.
    check: bool = False


def build_ced_boundary_injection(
    *, reqs: List["Req"], device: torch.device
) -> Optional[CedBoundaryInjection]:
    """The whole tail batch replays from the boundary only if every request has a
    handed-off state; otherwise the batch replays its encoder as before."""
    states = [req.ced_boundary_state for req in reqs]
    for req in reqs:
        req.ced_boundary_state = None
    missing = sum(state is None for state in states)
    if not states or missing:
        _note_fallback(missing, len(states))
        return None
    has_pre = [state.pre is not None for state in states]
    if any(has_pre) and not all(has_pre):
        _note_fallback(0, len(states))
        return None
    residual = torch.cat([state.residual for state in states]).to(
        device, non_blocking=True
    )
    pre = (
        torch.cat([state.pre for state in states]).to(device, non_blocking=True)
        if all(has_pre)
        else None
    )
    return CedBoundaryInjection(
        residual=residual,
        pre=pre,
        check=envs.SGLANG_DSV41_CED_BOUNDARY_CHECK.get(),
    )
