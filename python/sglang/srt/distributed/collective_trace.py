"""Opt-in NVTX tracing for cross-rank collective arrival analysis.

The decorator in this module is resolved when the decorated method is defined.
When ``SGLANG_COLLECTIVE_TRACE`` is disabled, it returns the original method
unchanged so normal serving does not pay a per-collective branch.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import Callable, Optional, TypeVar

import torch

from sglang.srt.environ import envs


_F = TypeVar("_F", bound=Callable)
_CURRENT_LAYER_STACK: ContextVar[tuple[int, ...]] = ContextVar(
    "sglang_collective_trace_layer_stack", default=()
)
_COLLECTIVE_TRACE_DEPTH: ContextVar[int] = ContextVar(
    "sglang_collective_trace_depth", default=0
)
_TRACE_PREFIX = "sglang.collective"


def current_layer_id() -> Optional[int]:
    """Return the innermost traced decoder layer, if one is active."""
    stack = _CURRENT_LAYER_STACK.get()
    return stack[-1] if stack else None


def format_collective_trace_label(group, op: str, input_: torch.Tensor) -> str:
    """Build a compact, machine-readable NVTX label for one collective."""
    sequence = getattr(group, "_sglang_collective_trace_sequence", 0)
    group._sglang_collective_trace_sequence = sequence + 1
    layer_id = current_layer_id()
    shape = "x".join(str(dim) for dim in input_.shape) or "scalar"
    dtype = str(input_.dtype).removeprefix("torch.")
    nbytes = input_.numel() * input_.element_size()
    return (
        f"{_TRACE_PREFIX}|op={op}|group={group.unique_name}|seq={sequence}"
        f"|rank={group.rank_in_group}|global_rank={group.rank}"
        f"|world={group.world_size}|layer={layer_id if layer_id is not None else 'none'}"
        f"|bytes={nbytes}|dtype={dtype}|shape={shape}"
    )


def trace_collective(op: str) -> Callable[[_F], _F]:
    """Decorate a ``GroupCoordinator`` collective with an NVTX range.

    The environment is intentionally checked at decoration/import time. The
    server launch sets the diagnostic flag before importing SGLang, allowing
    the disabled path to remain exactly the undecorated function.
    """

    def decorator(func: _F) -> _F:
        if not envs.SGLANG_COLLECTIVE_TRACE.get():
            return func

        @wraps(func)
        def wrapped(group, input_: torch.Tensor, *args, **kwargs):
            if torch.compiler.is_compiling() or (
                input_.is_cuda and torch.cuda.is_current_stream_capturing()
            ):
                return func(group, input_, *args, **kwargs)

            depth = _COLLECTIVE_TRACE_DEPTH.get()
            if depth:
                return func(group, input_, *args, **kwargs)

            depth_token = _COLLECTIVE_TRACE_DEPTH.set(depth + 1)
            label = format_collective_trace_label(group, op, input_)
            torch.cuda.nvtx.range_push(label)
            try:
                return func(group, input_, *args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
                _COLLECTIVE_TRACE_DEPTH.reset(depth_token)

        return wrapped  # type: ignore[return-value]

    return decorator


def trace_decoder_layer(func: _F) -> _F:
    """Add eager-only layer context while remaining invisible to graph builds."""
    if not envs.SGLANG_COLLECTIVE_TRACE.get():
        return func

    @wraps(func)
    def wrapped(layer, *args, **kwargs):
        if torch.compiler.is_compiling():
            return func(layer, *args, **kwargs)

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = next(
                (value for value in args if isinstance(value, torch.Tensor)),
                None,
            )
        if (
            isinstance(hidden_states, torch.Tensor)
            and hidden_states.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            return func(layer, *args, **kwargs)

        stack = _CURRENT_LAYER_STACK.get()
        layer_token = _CURRENT_LAYER_STACK.set((*stack, int(layer.layer_id)))
        try:
            return func(layer, *args, **kwargs)
        finally:
            _CURRENT_LAYER_STACK.reset(layer_token)

    return wrapped  # type: ignore[return-value]
