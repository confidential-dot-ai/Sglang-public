"""CED boundary handoff: the encoder's layer-boundary state for each prompt's last
SWA window, computed on prefill and replayed from on decode.

Under the CED split a decode tail replay reran the encoder (layers before the
last kv_source layer) over every tail row only to rebuild the hidden state the
late layers start from; prefill had computed that state already, and its encoder
KV reaches decode through the ordinary transfer. With the handoff, the prefill
stage that runs the last kv_source layer captures that layer's input state
(the mHC residual and its pre-mix) for the rows the late layers run over -- the
prompt's last SWA_WINDOW positions -- into the request's PD metadata slot, and
decode starts its tail replay at that layer.

Slot layout (one uint8 row per metadata slot):
    header  int64[8]: magic, bootstrap_room, window_start, window_rows, has_pre
    residual  bf16[window, hc_mult, hidden]
    pre       fp32[window, hc_mult]
A slot whose header does not carry this request's room, or whose window was not
fully computed on prefill (a prefix-cache hit inside the window), is rejected and
the request replays its encoder as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_MAGIC = 0x4345_4442_4E44  # "CEDBND"
_HEADER_INT64 = 8
_HEADER_BYTES = _HEADER_INT64 * 8


def ced_boundary_enabled() -> bool:
    return (
        envs.SGLANG_OPT_DSV41_CED_SPLIT.get()
        and envs.SGLANG_OPT_DSV41_CED_BOUNDARY_HANDOFF.get()
    )


@dataclass(frozen=True)
class CedBoundaryLayout:
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


def ced_boundary_layout(hf_text_config, sliding_window_size: int) -> CedBoundaryLayout:
    return CedBoundaryLayout(
        window=int(sliding_window_size),
        hc_mult=int(hf_text_config.hc_mult),
        hidden_size=int(hf_text_config.hidden_size),
    )


def maybe_ced_boundary_layout(
    hf_text_config, sliding_window_size: Optional[int], *, holds_boundary: bool
) -> Optional[CedBoundaryLayout]:
    """The slot layout when this worker sends or receives boundary state: decode,
    and the prefill pipeline stage that runs the last kv_source layer."""
    if not ced_boundary_enabled() or not holds_boundary:
        return None
    assert sliding_window_size, "the CED boundary handoff needs the SWA window"
    return ced_boundary_layout(hf_text_config, sliding_window_size)


def window_start(seq_len: int, window: int) -> int:
    return max(0, seq_len - window)


# --- prefill: capture --------------------------------------------------------


@dataclass
class CedBoundaryCapture:
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
    filled: bool = False


def build_ced_boundary_capture(
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


def fill_ced_boundary_capture(
    capture: CedBoundaryCapture,
    residual: torch.Tensor,
    pre: Optional[torch.Tensor],
) -> None:
    capture.residual = residual.index_select(0, capture.rows)
    capture.pre = None if pre is None else pre.index_select(0, capture.rows)
    capture.filled = True


def fill_ced_boundary_for_forward(
    forward_batch, residual: torch.Tensor, pre: Optional[torch.Tensor]
) -> None:
    capture = getattr(forward_batch, "ced_boundary", None)
    if isinstance(capture, CedBoundaryCapture):
        fill_ced_boundary_capture(capture, residual, pre)


def write_ced_boundary_slots(
    capture: Optional[CedBoundaryCapture],
    reqs: List["Req"],
    slots: torch.Tensor,
    layout: CedBoundaryLayout,
) -> None:
    """Scatter one forward's captured rows into the requests' metadata slots.

    Runs after the forward's results are ready. A request records how many window
    rows it has written; ``finalize_ced_boundary_slot`` stamps the header.
    """
    if capture is None or not capture.filled:
        return
    residual = capture.residual.to("cpu", non_blocking=False)
    pre = None if capture.pre is None else capture.pre.to("cpu", non_blocking=False)
    row = 0
    for batch_index, start, count in capture.spans:
        req = reqs[batch_index]
        idx = req.metadata_buffer_index
        if idx is None or idx < 0:
            row += count
            continue
        _, slot_residual, slot_pre = layout.views(slots[idx])
        base = window_start(len(req.origin_input_ids), layout.window)
        lo = start - base
        slot_residual[lo : lo + count].copy_(residual[row : row + count])
        if pre is not None:
            slot_pre[lo : lo + count].copy_(pre[row : row + count])
        req.ced_boundary_rows = getattr(req, "ced_boundary_rows", 0) + count
        req.ced_boundary_has_pre = pre is not None
        row += count


def finalize_ced_boundary_slot(
    req: "Req", slots: torch.Tensor, layout: CedBoundaryLayout
) -> None:
    """Stamp the header at the last chunk; a window prefill did not fully compute
    (a prefix-cache hit inside it) is marked invalid so decode replays."""
    header, _, _ = layout.views(slots[req.metadata_buffer_index])
    seq_len = len(req.origin_input_ids)
    rows = min(seq_len, layout.window)
    valid = getattr(req, "ced_boundary_rows", 0) == rows
    header.zero_()
    header[0] = _MAGIC if valid else 0
    header[1] = int(req.bootstrap_room or 0)
    header[2] = window_start(seq_len, layout.window)
    header[3] = rows
    header[4] = int(getattr(req, "ced_boundary_has_pre", False))
    req.ced_boundary_rows = 0


def ced_boundary_for_forward(batch, forward_batch, window: Optional[int], device):
    """The forward's capture spec (prefill) or injection (decode), if any."""
    from sglang.srt.runtime_context import get_disagg, get_parallel

    injection = getattr(batch, "ced_boundary_injection", None)
    if injection is not None:
        # One tail replay: the tail batch can become the running decode batch.
        batch.ced_boundary_injection = None
        return injection if forward_batch.forward_mode.is_extend() else None
    if (
        not window
        or get_disagg().disaggregation_mode != "prefill"
        or not get_parallel().pp_group.is_last_rank
        or not forward_batch.forward_mode.is_extend()
        or forward_batch.extend_seq_lens_cpu is None
    ):
        return None
    return build_ced_boundary_capture(
        batch.reqs,
        forward_batch.extend_prefix_lens_cpu,
        forward_batch.extend_seq_lens_cpu,
        window,
        device,
    )


# --- decode: read and inject --------------------------------------------------


@dataclass
class CedBoundaryState:
    """A request's handed-off window state, on the host until its tail replays."""

    residual: torch.Tensor  # bf16[rows, hc, hidden], CPU
    pre: Optional[torch.Tensor]  # fp32[rows, hc], CPU


def take_ced_boundary_slot(
    req: "Req", slots: torch.Tensor, idx: int, layout: CedBoundaryLayout
) -> Optional[CedBoundaryState]:
    """Copy a request's window state out of its metadata slot before the slot is
    freed; None when prefill did not hand one over."""
    header, residual, pre = layout.views(slots[idx])
    seq_len = len(req.origin_input_ids)
    rows = min(seq_len, layout.window)
    if (
        int(header[0]) != _MAGIC
        or int(header[1]) != int(req.bootstrap_room or 0)
        or int(header[2]) != window_start(seq_len, layout.window)
        or int(header[3]) != rows
    ):
        return None
    has_pre = bool(int(header[4]))
    header.zero_()
    return CedBoundaryState(
        residual=residual[:rows].clone(),
        pre=pre[:rows].clone() if has_pre else None,
    )


@dataclass
class CedBoundaryInjection:
    """A tail replay's handed-off state, in the late-layer tail's row order."""

    residual: torch.Tensor  # bf16[tail_rows, hc, hidden], device
    pre: Optional[torch.Tensor]  # fp32[tail_rows, hc], device
    # Debug: also run the encoder and compare its boundary state with this one.
    check: bool = False
    stats: Dict[str, float] = field(default_factory=dict)


def build_ced_boundary_injection(
    reqs: List["Req"], device: torch.device
) -> Optional[CedBoundaryInjection]:
    """The whole tail batch replays from the boundary only if every request has a
    handed-off state; otherwise the batch replays its encoder as before."""
    states = [getattr(req, "ced_boundary_state", None) for req in reqs]
    if not states or any(state is None for state in states):
        return None
    has_pre = [state.pre is not None for state in states]
    if any(has_pre) and not all(has_pre):
        return None
    residual = torch.cat([state.residual for state in states]).to(
        device, non_blocking=True
    )
    pre = (
        torch.cat([state.pre for state in states]).to(device, non_blocking=True)
        if all(has_pre)
        else None
    )
    for req in reqs:
        req.ced_boundary_state = None
    return CedBoundaryInjection(
        residual=residual,
        pre=pre,
        check=envs.SGLANG_DSV41_CED_BOUNDARY_CHECK.get(),
    )
