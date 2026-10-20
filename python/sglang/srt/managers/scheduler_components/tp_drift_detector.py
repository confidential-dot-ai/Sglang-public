"""Detects TP ranks disagreeing on a counter they are required to share.

Speculative decoding samples on every rank. A rank that accepts a different
number of draft tokens than its peers diverges, and the group deadlocks at the
next collective whose shape depends on that difference -- with no error, no
Xid, and no NCCL timeout, because the collective simply never completes.

The divergence is observable well before the deadlock: a wedged worker was
seen drifting by one token nine minutes before its ranks stopped. Checking the
counter turns that silent hang into a reported fault with warning to spare.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# A peer quieter than this is not reporting, so the check abstains. Generous
# because the publish cadence follows decode_log_interval, which stretches out
# when the batch is small.
_PEER_STALENESS_LIMIT_SECONDS = 120.0


def _default_on_drift() -> None:
    """Abort collectives, then exit so the rank leaves rotation."""
    try:
        from sglang.srt.distributed.device_communicators.pynccl import (
            abort_all_communicators,
        )

        abort_all_communicators()
    except Exception as e:
        logger.error("Could not abort pynccl communicators on drift: %s", e)

    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)


class TpDriftDetector:
    """Compares a per-rank counter across the TP group over gloo.

    The check runs on the CPU group on purpose. A device collective would
    queue behind whatever is already stuck on the stream, which is exactly the
    condition being detected.
    """

    def __init__(
        self,
        group_key: str,
        tp_rank: int,
        tp_size: int,
        on_drift: Optional[Callable[[], None]] = None,
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.on_drift = on_drift or _default_on_drift

        self.interval = envs.SGLANG_TP_DRIFT_CHECK_INTERVAL.get()
        self.exit_after = envs.SGLANG_TP_DRIFT_EXIT_AFTER.get()
        self.enabled = self.interval > 0 and tp_size > 1

        digest = hashlib.sha1(group_key.encode()).hexdigest()[:16]
        self._dir = Path(tempfile.gettempdir()) / f"sglang-tp-drift-{digest}"
        if self.enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

        self._calls = 0
        self._consecutive_divergences = 0

    def observe(self, counter: int) -> None:
        """Feed the shared counter. Never raises; a broken check must not
        take down a worker that is otherwise serving."""
        if not self.enabled:
            return

        self._calls += 1
        if self._calls % self.interval != 0:
            return

        try:
            values = self._gather(counter)
        except Exception as e:
            # Includes the gather timing out, which on its own is not proof of
            # drift -- the watchdog owns the stuck-forever case.
            logger.warning("TP drift check did not complete: %s", e)
            return

        if values is None or len(set(values)) == 1:
            self._consecutive_divergences = 0
            return

        self._consecutive_divergences += 1
        logger.error(
            "TP rank divergence: decoded-token counters disagree across the "
            "group (%s), spread=%d, consecutive=%d/%d. Ranks have sampled "
            "differently and the group will deadlock at a later collective.",
            ", ".join(f"tp{i}={v}" for i, v in enumerate(values)),
            max(values) - min(values),
            self._consecutive_divergences,
            self.exit_after,
        )

        if self._consecutive_divergences >= self.exit_after:
            logger.error(
                "TP rank divergence persisted for %d checks; taking this rank "
                "down so the worker leaves rotation instead of hanging.",
                self._consecutive_divergences,
            )
            self.on_drift()

    def _gather(self, counter: int) -> Optional[list[int]]:
        """Publish this rank's counter and read the peers'.

        Deliberately not a collective. A detector that can block is a detector
        that can cause the fault it watches for; this exchange has no barrier,
        so a stalled peer makes the check abstain rather than hang. Returns
        None when the group has not all reported recently, or not yet for this
        same check, which is a "no verdict", not agreement.
        """
        own = self._dir / f"rank{self.tp_rank}"
        staging = self._dir / f"rank{self.tp_rank}.tmp"
        # Ranks reach each check at the same decode step, but not at the same
        # instant: tagging the counter with the check number keeps a peer's
        # previous reading from being compared with this one.
        staging.write_text(f"{self._calls} {counter}")
        # Atomic, so a peer never reads a half-written counter.
        staging.replace(own)

        now = time.time()
        values = []
        for rank in range(self.tp_size):
            path = self._dir / f"rank{rank}"
            try:
                stat = path.stat()
            except FileNotFoundError:
                return None
            if now - stat.st_mtime > _PEER_STALENESS_LIMIT_SECONDS:
                # That rank is behind or stopped. Being stuck is the
                # watchdog's business; this check only rules on disagreement
                # between ranks that are all currently reporting.
                return None
            check, value = path.read_text().split()
            if int(check) != self._calls:
                return None
            values.append(int(value))
        return values
