"""Opt-in timing and one-shot profiling of DeepSeek-V4.1 CED decode tail replays.

SGLANG_DSV41_CED_TAIL_TIMING logs each replay's host wall time and, once its CUDA
events complete on a later step, its GPU time, without synchronizing.
SGLANG_DSV41_CED_TAIL_PROFILE_DIR arms a one-shot profiler on TP0 under the CED
split: touch <directory>/trigger to capture the next replay. It re-arms only
after the trace is exported.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


@dataclass
class _TailProbe:
    stream: Optional[torch.cuda.Stream] = None
    fields: str = ""
    wall_start: float = 0.0
    start: Optional[torch.cuda.Event] = None
    end: Optional[torch.cuda.Event] = None
    profiler: Optional[torch.profiler.profile] = None
    profile_dir: str = ""


@dataclass
class CedTailTimer:
    """Times decode EXTEND batches, which under the CED split are tail replays."""

    tp_rank: int
    _pending: List[Tuple[torch.cuda.Event, torch.cuda.Event, str]] = field(
        default_factory=list
    )

    def begin(
        self,
        batch: ScheduleBatch,
        *,
        forward_ct: int,
        forward_stream: Optional[torch.cuda.Stream],
    ) -> Optional[_TailProbe]:
        timing = envs.SGLANG_DSV41_CED_TAIL_TIMING.get()
        if timing:
            self._log_completed()
        if not batch.forward_mode.is_extend():
            return None

        probe = _TailProbe()
        if timing:
            probe.stream = forward_stream or torch.cuda.current_stream()
            probe.fields = "iter=%d reqs=%d rows=%d" % (
                forward_ct,
                batch.batch_size(),
                batch.extend_num_tokens,
            )
            probe.wall_start = time.perf_counter()
            probe.start = torch.cuda.Event(enable_timing=True)
            probe.end = torch.cuda.Event(enable_timing=True)
            probe.start.record(probe.stream)

        profile_dir = envs.SGLANG_DSV41_CED_TAIL_PROFILE_DIR.get()
        if profile_dir and self.tp_rank == 0 and envs.SGLANG_OPT_DSV41_CED_SPLIT.get():
            trigger = Path(profile_dir) / "trigger"
            if trigger.exists():
                trigger.unlink()
                probe.profile_dir = profile_dir
                probe.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                )
                probe.profiler.start()

        if probe.start is None and probe.profiler is None:
            return None
        return probe

    def end(
        self, probe: Optional[_TailProbe], batch: ScheduleBatch, *, forward_ct: int
    ) -> None:
        if probe is None:
            return
        if probe.start is not None:
            probe.end.record(probe.stream)
            logger.info(
                "CED_TAIL %s prepare_ms=%.3f wall_ms=%.3f",
                probe.fields,
                getattr(batch, "ced_tail_prepare_ms", 0.0),
                (time.perf_counter() - probe.wall_start) * 1000,
            )
            self._pending.append((probe.start, probe.end, probe.fields))
        if probe.profiler is not None:
            probe.profiler.stop()
            destination = Path(probe.profile_dir) / f"tail-{forward_ct}"
            probe.profiler.export_chrome_trace(str(destination) + ".json")
            Path(str(destination) + ".txt").write_text(
                probe.profiler.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total", row_limit=100
                )
            )
            logger.info("CED tail profile exported to %s", destination)

    def _log_completed(self) -> None:
        remaining = []
        for start, end, fields in self._pending:
            if end.query():
                logger.info(
                    "CED_TAIL_GPU %s gpu_ms=%.3f", fields, start.elapsed_time(end)
                )
            else:
                remaining.append((start, end, fields))
        self._pending = remaining
