from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import Callable, List, Optional

import psutil

from sglang.srt.utils.cudacore_pyspy_dump_utils import pyspy_dump_schedulers

logger = logging.getLogger(__name__)

# Grace between asking the parent to tear the group down and exiting anyway.
# Long enough for an orderly shutdown to win, short enough that a wedged rank
# does not sit in rotation.
_SELF_TERMINATE_GRACE_SECONDS = 30.0

# Budget for startup, which is active but advances no forward counter.
# Measured on the B300 rig: ~17 min weight load plus ~3.5 min graph capture.
_STARTUP_WATCHDOG_TIMEOUT_SECONDS = 1800.0


def _abort_pynccl_communicators() -> None:
    """Best effort; never let teardown fail on the way to exiting."""
    try:
        from sglang.srt.distributed.device_communicators.pynccl import (
            abort_all_communicators,
        )

        aborted = abort_all_communicators()
        if aborted:
            logger.error("Aborted %d pynccl communicator(s) before teardown", aborted)
    except Exception as e:
        logger.error("Could not abort pynccl communicators: %s", e)


class Watchdog:
    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert test_stuck_time == 0, (
                f"stuck tester can be enabled only if soft watchdog is enabled."
            )
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        pass

    @contextmanager
    def disable(self):
        yield


class _WatchdogReal(Watchdog):
    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0
        self._active = True
        self._test_stuck_time = test_stuck_time
        self._test_stuck_triggered = False
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        # Only trigger the test stuck behavior once to avoid blocking server
        # startup health checks while still testing watchdog timeout detection
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    pass


class WatchdogRaw:
    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
        is_starting_up: Optional[Callable[[], bool]] = None,
    ):
        self.debug_name = debug_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info
        self.is_starting_up = is_starting_up
        # Startup counts as active but never advances the forward counter, so
        # weight load and graph capture run against this budget instead. Held
        # at or above the serving timeout so lowering that cannot kill a boot.
        self.startup_timeout = max(watchdog_timeout, _STARTUP_WATCHDOG_TIMEOUT_SECONDS)

        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

    def _current_timeout(self) -> float:
        if self.is_starting_up is not None and self.is_starting_up():
            return self.startup_timeout
        return self.watchdog_timeout

    def _watchdog_once(self):
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    if current > watchdog_last_time + self._current_timeout():
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            # Paced off the serving timeout so a short one still gets
            # sub-timeout resolution during startup.
            time.sleep(self.watchdog_timeout / 2)

        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        pyspy_dump_schedulers()
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            # A collective spinning on the GPU keeps the process alive through
            # SIGTERM and holds its HBM, so abort the communicators first: the
            # kernels poll the abort flag and exit, and teardown can proceed.
            _abort_pynccl_communicators()

            # Wait for some time so that the parent process can print the error.
            time.sleep(5)
            self.parent_process.send_signal(signal.SIGQUIT)
            # SIGQUIT to the parent is a request, not a guarantee: a parent that
            # is wedged on the same collective never reaps us, and the caller
            # re-arms this thread, so the rank stays up logging a timeout every
            # interval instead of going down. Exit on our own after a grace.
            time.sleep(_SELF_TERMINATE_GRACE_SECONDS)
            logger.error(
                f"{self.debug_name} still alive {_SELF_TERMINATE_GRACE_SECONDS}s "
                f"after SIGQUIT to parent pid={self.parent_process.pid}; "
                f"self-terminating so the worker is removed from rotation"
            )
            sys.stderr.flush()
            sys.stdout.flush()
            # _exit, not sys.exit: this is a daemon thread, and a SystemExit
            # here would only unwind this thread and leave the process hung.
            os._exit(1)


class SubprocessWatchdog:
    """Monitors subprocess liveness and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog polls subprocess liveness in a daemon thread and
    sends SIGQUIT to trigger proper cleanup.

    See: https://github.com/sgl-project/sglang/issues/18421
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        interval: float = 1.0,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _monitor_loop(self) -> None:
        try:
            while not self._stop_event.wait(self._interval):
                if self._check_processes():
                    return
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)

    def _check_processes(self) -> bool:
        for proc, name in zip(self._processes, self._names):
            if proc.is_alive() or proc.exitcode == 0:
                continue

            logger.error(
                f"Subprocess {name} (pid={proc.pid}) crashed "
                f"with exit code {proc.exitcode}. "
                f"Triggering SIGQUIT for cleanup..."
            )
            os.kill(os.getpid(), signal.SIGQUIT)
            return True
        return False
