"""Memory of the DeepSeek-V4.1 prefill indexer's eager score tiles.

The ratio-1/2 index layers score a prefill chunk's rows against every visible
compressed position in fp32 logits tiles that no pool sized by
mem_fraction_static accounts for, so they come out of what CUDA-graph capture
leaves free. ``plan_index_logits_budget`` sizes the tile budget from that
headroom once, after capture, and refuses to start when even the smallest tile
does not fit, instead of running out of memory mid-traffic.

``eager_indexer_peak_bytes`` models the measured peak of one eager index layer
(DeepGEMM path, one TP2 rank): the flattened index K of every visible
position and its slot map, plus the larger of two phases: scoring, which holds
one logits tile, the candidate source's block keys (one fp32 per 8 columns) and
the per-row operands, including each row's index page table (rows x context);
and building the candidate source's sparse table from those rows. Measured on
B200 at 256K-1M context with 256-16384 rows and 64 MiB-2 GiB budgets (candidate
source, full top-k and consumer layers), it bounds every point.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.utils.common import ceil_align

INDEX_LOGITS_BUDGET_CAP_BYTES = 2 << 30
# Planning halves the budget down to this; below it the tile loop's launches dominate.
INDEX_LOGITS_BUDGET_FLOOR_BYTES = 128 << 20
_LOGITS_ITEM_BYTES = 4
_WIDTH_ALIGN = 64
# Per visible position: its flattened index K row (68 B) and slot (8 B), with slack.
_POSITION_BYTES = 96
# Per row while scoring: q/weights/selection operands (~16 KiB) plus ~0.05 B per
# visible position (mostly its index page table).
_ROW_BYTES = 16 << 10
_ROW_CONTEXT_BYTES_DIV = 20
# Per row while the candidate source sorts and schedules its blocks.
_TABLE_ROW_BYTES = 48 << 10

_planned_budget_bytes: Optional[int] = None


def eager_indexer_peak_bytes(*, budget_bytes: int, rows: int, context_len: int) -> int:
    """Peak bytes of one eager index layer scoring ``rows`` query rows against
    ``context_len`` positions with logits tiles of at most ``budget_bytes``."""
    row_bytes = ceil_align(max(context_len, 1), _WIDTH_ALIGN) * _LOGITS_ITEM_BYTES
    tile = min(rows * row_bytes, max(budget_bytes, row_bytes))
    per_position = context_len // _ROW_CONTEXT_BYTES_DIV
    scoring = tile + tile // 8 + rows * (_ROW_BYTES + per_position)
    table = rows * (_TABLE_ROW_BYTES + per_position)
    return context_len * _POSITION_BYTES + max(scoring, table)


def plan_index_logits_budget(
    *, free_bytes: int, rows: int, context_len: int, reserve_bytes: int
) -> int:
    """The largest tile budget (halving from the cap) whose eager peak plus
    ``reserve_bytes`` (the prefill's other eager transients and allocator slack)
    fits ``free_bytes``; raise a ValueError when not even the floor fits."""
    budget = INDEX_LOGITS_BUDGET_CAP_BYTES
    while True:
        need = reserve_bytes + eager_indexer_peak_bytes(
            budget_bytes=budget, rows=rows, context_len=context_len
        )
        if need <= free_bytes:
            return budget
        if budget // 2 < INDEX_LOGITS_BUDGET_FLOOR_BYTES:
            break
        budget //= 2
    gib = 1 << 30
    raise ValueError(
        f"DeepSeek-V4.1 prefill: {free_bytes / gib:.2f} GiB free after CUDA-graph "
        f"capture cannot hold the eager indexer at context_len={context_len}, "
        f"{rows} rows per forward: it needs {need / gib:.2f} GiB with "
        f"{budget / gib:.3f} GiB logits tiles ({reserve_bytes / gib:.2f} GiB "
        "reserved for other eager transients, SGLANG_DSV41_PREFILL_EAGER_RESERVE_GB). "
        "Lower --max-total-tokens, --chunked-prefill-size or --context-length, or "
        "free capture memory (fewer prefill graph sizes); "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces the "
        "fragmentation the reserve covers."
    )


def set_planned_index_logits_budget(budget_bytes: Optional[int]) -> None:
    """Record a plan; repeated plans (target, then draft capture) keep the smallest."""
    global _planned_budget_bytes
    if budget_bytes is None or _planned_budget_bytes is None:
        _planned_budget_bytes = budget_bytes
    else:
        _planned_budget_bytes = min(_planned_budget_bytes, budget_bytes)


def planned_index_logits_budget() -> int:
    """The planned tile budget, or the cap before planning ran."""
    if _planned_budget_bytes is None:
        return INDEX_LOGITS_BUDGET_CAP_BYTES
    return _planned_budget_bytes
