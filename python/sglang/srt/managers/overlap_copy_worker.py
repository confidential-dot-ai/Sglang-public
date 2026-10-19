from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class D2HCopyCompletion(threading.Event):
    """Completion signal that propagates a deferred D2H copy failure."""

    def __init__(self):
        super().__init__()
        self._exception: BaseException | None = None

    def set_exception(self, exception: BaseException) -> None:
        self._exception = exception
        self.set()

    def wait(self, timeout: float | None = None) -> bool:
        completed = super().wait(timeout)
        if completed and self._exception is not None:
            raise RuntimeError("asynchronous D2H readback failed") from self._exception
        return completed


class AsyncD2HCopyWorker:
    """Runs the per-step result Device->Host readback on a dedicated daemon thread.

    Under NVIDIA Confidential Computing (bounce-buffer CC), a device-to-host
    ``cudaMemcpyAsync`` is forced synchronous and blocks AT ISSUE (the host
    destination is staged through an encrypted bounce buffer). Issuing it inline
    on the scheduler thread stalls the overlap pipeline: the scheduler cannot
    launch the next step's CUDA graph until the copy returns (~one decode step).

    This worker performs the (still-blocking) copy off the scheduler thread so
    the scheduler keeps issuing work. It mirrors TensorRT-LLM PR #8463:
      - the scheduler thread records a CUDA event after the producing work and
        hands (event, copy_fn, done) here, then returns immediately;
      - this worker ``cudaEventSynchronize``-s on that event (event-sync, NOT
        stream-wait, so the blocking copy does not stall the scheduler's CUDA
        API calls), runs the copy on the dedicated copy stream, blocks until it
        completes, and signals the returned completion;
      - the scheduler later waits on that completion (worker-done => copy-done),
        which re-raises any deferred readback failure.

    The copy itself stays synchronous under CC; it is merely non-blocking *to the
    scheduler thread*, restoring overlap.
    """

    def __init__(self, device_module, copy_stream):
        self.device_module = device_module
        self.copy_stream = copy_stream
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name="sglang-d2h-copy-worker", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        src_ready: "torch.cuda.Event",  # noqa: F821 - annotation only
        copy_fn: Callable[[], None],
    ) -> D2HCopyCompletion:
        """Enqueue a readback. ``src_ready`` must already be recorded on the
        stream that produces the copy sources; ``copy_fn`` performs the actual
        ``.to("cpu", ...)`` copies. The returned completion raises the worker
        exception from ``wait`` if the readback fails."""
        done = D2HCopyCompletion()
        self._queue.put((src_ready, copy_fn, done))
        return done

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            src_ready, copy_fn, done = item
            try:
                # Wait until the producing forward+sample has materialized the
                # source tensors. Event-sync (cudaEventSynchronize), not a
                # stream wait — see class docstring / TRT-LLM #8463.
                src_ready.synchronize()
                # Issue the copies on a dedicated stream owned by this thread.
                # PyTorch's current stream is thread-local, so this does not
                # affect the scheduler thread's stream.
                with self.device_module.stream(self.copy_stream):
                    copy_fn()
                self.copy_stream.synchronize()
            except Exception as exception:
                logger.exception("AsyncD2HCopyWorker readback failed")
                done.set_exception(exception)
            else:
                done.set()

    def shutdown(self, timeout: float = 2.0):
        """Signal the worker to stop and require it to join within ``timeout``."""
        if not self._thread.is_alive():
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("AsyncD2HCopyWorker did not stop before timeout")
