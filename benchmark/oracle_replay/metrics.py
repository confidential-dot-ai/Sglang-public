"""Metric primitives for sustained concurrent oracle-trace replay.

The functions in this module are deliberately independent of HTTP and CUDA so
the throughput contract can be regression-tested on CPU.  In particular, the
physical GPU count is always explicit; benchmark topology is never inferred
from tensor-parallel size or baked into a metric name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class TokenCounters:
    """Monotonic completed-request counters captured at one instant."""

    completed_requests: int = 0
    failed_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    def delta_since(self, earlier: "TokenCounters") -> "TokenCounters":
        values = {
            field: getattr(self, field) - getattr(earlier, field)
            for field in self.__dataclass_fields__
        }
        negative = {field: value for field, value in values.items() if value < 0}
        if negative:
            raise ValueError(f"Counters must be monotonic; negative deltas: {negative}")
        return TokenCounters(**values)


@dataclass(frozen=True)
class MixBands:
    """Accepted runtime traffic mix for the 3M TPM scorecard."""

    cache_hit_min: float = 0.94
    cache_hit_max: float = 0.96
    output_share_min: float = 0.0065
    output_share_max: float = 0.0085

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.cache_hit_min > self.cache_hit_max:
            raise ValueError("cache_hit_min cannot exceed cache_hit_max")
        if self.output_share_min > self.output_share_max:
            raise ValueError("output_share_min cannot exceed output_share_max")


class ScorecardError(ValueError):
    """Raised when a measured window violates the declared scorecard."""


def summarize_window(
    start: TokenCounters,
    end: TokenCounters,
    *,
    elapsed_s: float,
    physical_gpus: int,
) -> dict[str, Any]:
    """Calculate rates and shares from counter deltas over one exact window."""

    if elapsed_s <= 0:
        raise ValueError(f"elapsed_s must be positive, got {elapsed_s}")
    if (
        isinstance(physical_gpus, bool)
        or not isinstance(physical_gpus, int)
        or physical_gpus <= 0
    ):
        raise ValueError(
            f"physical_gpus must be a positive integer, got {physical_gpus}"
        )

    delta = end.delta_since(start)
    if delta.cached_prompt_tokens > delta.prompt_tokens:
        raise ValueError(
            "cached_prompt_tokens cannot exceed prompt_tokens: "
            f"{delta.cached_prompt_tokens} > {delta.prompt_tokens}"
        )

    uncached_prompt_tokens = delta.prompt_tokens - delta.cached_prompt_tokens
    served_tokens = delta.prompt_tokens + delta.completion_tokens
    cache_hit_rate = (
        delta.cached_prompt_tokens / delta.prompt_tokens if delta.prompt_tokens else 0.0
    )
    output_share = delta.completion_tokens / served_tokens if served_tokens else 0.0
    cached_share = delta.cached_prompt_tokens / served_tokens if served_tokens else 0.0
    uncached_input_share = (
        uncached_prompt_tokens / served_tokens if served_tokens else 0.0
    )
    served_tok_s = served_tokens / elapsed_s

    return {
        "elapsed_s": elapsed_s,
        "physical_gpus": physical_gpus,
        "normalization_denominator_gpu_seconds": physical_gpus * elapsed_s,
        "completed_requests": delta.completed_requests,
        "failed_requests": delta.failed_requests,
        "prompt_tokens": delta.prompt_tokens,
        "cached_prompt_tokens": delta.cached_prompt_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "completion_tokens": delta.completion_tokens,
        "served_tokens": served_tokens,
        "request_rate": delta.completed_requests / elapsed_s,
        "prompt_tok_s": delta.prompt_tokens / elapsed_s,
        "cached_prompt_tok_s": delta.cached_prompt_tokens / elapsed_s,
        "uncached_prompt_tok_s": uncached_prompt_tokens / elapsed_s,
        "completion_tok_s": delta.completion_tokens / elapsed_s,
        "served_tok_s": served_tok_s,
        "cache_hit_rate": cache_hit_rate,
        "cached_served_share": cached_share,
        "uncached_input_share": uncached_input_share,
        "output_share": output_share,
        "served_tpm_total": served_tok_s * 60,
        "served_tpm_per_gpu": served_tok_s * 60 / physical_gpus,
    }


def scorecard_failures(
    summary: Mapping[str, Any], bands: Optional[MixBands] = None
) -> list[str]:
    """Return human-readable reasons a window is not scorecard-valid."""

    bands = bands or MixBands()
    failures = []
    if summary["failed_requests"]:
        failures.append(f"failed_requests={summary['failed_requests']} (required 0)")

    cache_hit = summary["cache_hit_rate"]
    if not bands.cache_hit_min <= cache_hit <= bands.cache_hit_max:
        failures.append(
            f"cache_hit_rate={cache_hit:.6f} outside "
            f"[{bands.cache_hit_min:.6f}, {bands.cache_hit_max:.6f}]"
        )

    output_share = summary["output_share"]
    if not bands.output_share_min <= output_share <= bands.output_share_max:
        failures.append(
            f"output_share={output_share:.6f} outside "
            f"[{bands.output_share_min:.6f}, {bands.output_share_max:.6f}]"
        )
    return failures


def assert_scorecard(
    summary: Mapping[str, Any], bands: Optional[MixBands] = None
) -> None:
    failures = scorecard_failures(summary, bands)
    if failures:
        raise ScorecardError("Scorecard failed: " + "; ".join(failures))


def visible_cuda_device_count(value: Optional[str]) -> Optional[int]:
    """Count an explicit CUDA mask, or return None when it is not informative."""

    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"all", "void"}:
        return None
    if not value or value == "-1":
        return 0
    return len([device for device in value.split(",") if device.strip()])


def validate_physical_gpu_count(
    physical_gpus: int,
    *,
    cuda_visible_devices: Optional[str],
    multi_node: bool = False,
) -> None:
    """Reject a local topology/denominator mismatch before a run starts."""

    if (
        isinstance(physical_gpus, bool)
        or not isinstance(physical_gpus, int)
        or physical_gpus <= 0
    ):
        raise ValueError(
            f"physical_gpus must be a positive integer, got {physical_gpus}"
        )
    visible = visible_cuda_device_count(cuda_visible_devices)
    if visible is not None and visible != physical_gpus and not multi_node:
        raise ValueError(
            f"--physical-gpus={physical_gpus} but CUDA_VISIBLE_DEVICES exposes "
            f"{visible}; set the correct denominator or pass --multi-node with an "
            "explicit topology description"
        )
