# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profiler span helpers for hot SGLang code paths.

A span has two independent emitters:

* ``record_function`` -- emitted whenever a torch profiler is active, so spans
  show up in torch/Perfetto traces for free (no env, no extra package).
* ``nvtx`` range -- emitted only when the caller opts in via ``nvtx_enabled``
  (wired to a per-subsystem ``SGLANG_ENABLE_NVTX_*`` gate) and the ``nvtx``
  package is importable, for Nsight Systems timelines.

Decoupling the two lets every annotation site -- scheduler stages, batch-overlap
ops, and the speculative-decoding / forward spans -- share one primitive.
"""

import logging
import os
import time
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial, wraps
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# TEMPORARY debug instrumentation for the profiler-pathology investigation
# (tdx-checkpoints/074-077): isolated microbenchmarks showed record_function
# itself (and record_function + CUDA-graph-replay) are both fast on this host
# with an active profiler, which rules out the primitive itself -- something
# specific to the real scheduler process is slow. Times each of the 5
# per-tick spans end-to-end (span entry + wrapped function body + span exit)
# and logs a warning above a threshold, so a real reproduction pinpoints
# which stage balloons. Off by default; remove once root-caused.
_PROFILE_RANGE_DEBUG = os.environ.get("SGLANG_PROFILE_RANGE_DEBUG", "0") == "1"
_PROFILE_RANGE_DEBUG_THRESHOLD_S = float(
    os.environ.get("SGLANG_PROFILE_RANGE_DEBUG_THRESHOLD_S", "0.1")
)

_SCHEDULER_NVTX = envs.SGLANG_ENABLE_NVTX_SCHEDULER.get()
_OPERATIONS_NVTX = envs.SGLANG_ENABLE_NVTX_OPERATIONS.get()

_nvtx_module = None
if _SCHEDULER_NVTX or _OPERATIONS_NVTX:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "An SGLANG_ENABLE_NVTX_* flag is set, but the `nvtx` package is "
            "missing. NVTX markers are disabled; torch profiler spans still emit."
        )

NVTX_AVAILABLE = _nvtx_module is not None
# Per-subsystem nvtx gates: emit nvtx ranges only when the flag is set AND the
# package is importable. The record_function path is independent of both.
NVTX_SCHEDULER_ENABLED = _SCHEDULER_NVTX and NVTX_AVAILABLE
NVTX_OPERATIONS_ENABLED = _OPERATIONS_NVTX and NVTX_AVAILABLE

# Default nvtx colors for statically-named spans (only used on the nvtx path).
_NVTX_COLOR_MAP = {
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
}

_NULL_CONTEXT = nullcontext()


@contextmanager
def _profile_range_impl(
    debug_name: str, color: Optional[str], record: bool, nvtx_enabled: bool
):
    with ExitStack() as stack:
        if record:
            if _PROFILE_RANGE_DEBUG:
                t_enter0 = time.monotonic()
                stack.enter_context(torch.profiler.record_function(debug_name))
                t_enter1 = time.monotonic()
                if t_enter1 - t_enter0 > _PROFILE_RANGE_DEBUG_THRESHOLD_S:
                    logger.warning(
                        "[PROFILE_RANGE_DEBUG] record_function ENTER for %s took %.3fs",
                        debug_name,
                        t_enter1 - t_enter0,
                    )
            else:
                stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx_enabled:
            if color is None:
                color = _NVTX_COLOR_MAP.get(debug_name)
            stack.enter_context(_nvtx_module.annotate(debug_name, color=color))
        if _PROFILE_RANGE_DEBUG:
            t_body0 = time.monotonic()
            yield
            t_body1 = time.monotonic()
            if t_body1 - t_body0 > _PROFILE_RANGE_DEBUG_THRESHOLD_S:
                logger.warning(
                    "[PROFILE_RANGE_DEBUG] body for %s took %.3fs",
                    debug_name,
                    t_body1 - t_body0,
                )
            t_exit0 = time.monotonic()
        else:
            yield
    if _PROFILE_RANGE_DEBUG and record:
        t_exit1 = time.monotonic()
        if t_exit1 - t_exit0 > _PROFILE_RANGE_DEBUG_THRESHOLD_S:
            logger.warning(
                "[PROFILE_RANGE_DEBUG] EXIT (record_function/nvtx teardown) for %s took %.3fs",
                debug_name,
                t_exit1 - t_exit0,
            )


def profile_range(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Context manager emitting a profiler span for ``debug_name``.

    A torch ``record_function`` is emitted whenever a torch profiler is active;
    an nvtx range is emitted additionally when ``nvtx_enabled`` is true. Returns a
    shared no-op when neither applies, so off-profile hot paths pay only one
    ``_profiler_enabled()`` check.
    """
    record = torch.autograd._profiler_enabled()
    if not record and not nvtx_enabled:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, color, record, nvtx_enabled)


def profile_method(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Decorator form of ``profile_range``."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_range(debug_name, color=color, nvtx_enabled=nvtx_enabled):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-bound per-subsystem helpers: torch spans always (under a profiler), nvtx
# ranges only when that subsystem's gate is on.
scheduler_nvtx_method = partial(profile_method, nvtx_enabled=NVTX_SCHEDULER_ENABLED)
operations_nvtx_range = partial(profile_range, nvtx_enabled=NVTX_OPERATIONS_ENABLED)
