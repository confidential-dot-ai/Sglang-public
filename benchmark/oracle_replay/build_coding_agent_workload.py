#!/usr/bin/env python3
"""Build bounded Oracle replay corpora from TraceLab token geometry.

The public TraceLab release contains sanitized lengths and timing rather than
prompt text.  This builder samples real, contiguous session windows, supplies
deterministic synthetic input text with the sampled token lengths, and uses
real forced completions from an Oracle source corpus.  Outputs are explicitly
labelled modeled workloads; they are not verbatim TraceLab sessions.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int | float], quantile: float) -> int | float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def summary(values: list[int | float]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def parse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def compact_round(row: dict[str, Any]) -> dict[str, Any]:
    events = row.get("timing_events") or []
    event_times = [
        timestamp
        for event in events
        if (timestamp := parse_timestamp(event.get("timestamp"))) is not None
    ]
    tool_times = [
        timestamp
        for tool in row.get("tools") or []
        for key in ("emitted_at", "result_at")
        if (timestamp := parse_timestamp(tool.get(key))) is not None
    ]
    times = event_times + tool_times
    return {
        "round_index": int(row.get("round_index") or 0),
        "prefix_tokens": max(0, int(row.get("prefix_tokens") or 0)),
        "append_tokens": max(1, int(row.get("newly_append_tokens") or 0)),
        "output_tokens": max(1, int(row.get("output_tokens") or 0)),
        "step_type": str(row.get("first_input_event_type") or "unknown"),
        "tool_calls": len(row.get("tools") or []),
        "started_at": min(times).isoformat() if times else None,
        "finished_at": max(times).isoformat() if times else None,
    }


def load_sessions(trace_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    sessions: dict[str, dict[str, list[dict[str, Any]]]] = {
        "codex": {},
        "claude": {},
    }
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt") as file:
        for line in file:
            row = json.loads(line)
            provider = row.get("provider")
            if provider not in sessions:
                continue
            session_id = str(row["session_id"])
            sessions[provider].setdefault(session_id, []).append(compact_round(row))
    for provider_sessions in sessions.values():
        for rows in provider_sessions.values():
            rows.sort(key=lambda row: row["round_index"])
    return sessions


def select_windows(
    sessions: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    provider: str,
    count: int,
    max_steps: int,
    seed: int,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    candidates = []
    providers = ("codex", "claude") if provider == "mixed" else (provider,)
    for item_provider in providers:
        for session_id, rows in sessions[item_provider].items():
            candidates.append((item_provider, session_id, rows))
    candidates.sort(key=lambda item: stable_rank(seed, f"{item[0]}:{item[1]}"))
    selected = []
    for item_provider, session_id, rows in candidates[:count]:
        if len(rows) > max_steps:
            window_rng = random.Random(
                int.from_bytes(stable_rank(seed + 1, session_id)[:8], "big")
            )
            start = window_rng.randrange(len(rows) - max_steps + 1)
            rows = rows[start : start + max_steps]
        selected.append((item_provider, session_id, rows))
    if len(selected) != count:
        raise ValueError(f"requested {count} sessions but found {len(selected)}")
    return selected


def interpolate_quantiles(
    quantile: float, anchors: tuple[tuple[float, int], ...]
) -> int:
    for (left_q, left_value), (right_q, right_value) in zip(anchors, anchors[1:]):
        if quantile <= right_q:
            fraction = (quantile - left_q) / (right_q - left_q)
            return round(left_value + fraction * (right_value - left_value))
    return anchors[-1][1]


def together_shape(
    windows: list[tuple[str, str, list[dict[str, Any]]]],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Map sampled Codex steps onto Together's published benchmark anchors."""
    shaped = copy.deepcopy(windows)
    rows = [row for _, _, session_rows in shaped for row in session_rows]
    input_order = sorted(
        range(len(rows)),
        key=lambda index: rows[index]["prefix_tokens"] + rows[index]["append_tokens"],
    )
    output_order = sorted(range(len(rows)), key=lambda index: rows[index]["output_tokens"])
    input_anchors = (
        (0.00, 45_000),
        (0.25, 70_000),
        (0.50, 90_000),
        (0.75, 120_000),
        (0.90, 160_000),
        (1.00, 200_000),
    )
    # Together publishes mean~450, p50=293, and p99=2,230. The remaining
    # anchors are documented modeling choices that yield a mean near 450.
    output_anchors = (
        (0.00, 32),
        (0.25, 140),
        (0.50, 293),
        (0.75, 535),
        (0.90, 760),
        (0.99, 2_230),
        (1.00, 4_096),
    )
    denominator = max(1, len(rows) - 1)
    for rank, index in enumerate(input_order):
        target_input = interpolate_quantiles(rank / denominator, input_anchors)
        append = min(rows[index]["append_tokens"], target_input)
        rows[index]["append_tokens"] = max(1, append)
        rows[index]["prefix_tokens"] = max(0, target_input - append)
    for rank, index in enumerate(output_order):
        rows[index]["output_tokens"] = interpolate_quantiles(
            rank / denominator, output_anchors
        )
    return shaped


class ExactTextFactory:
    """Create deterministic text with an exact tokenizer content-token count."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        candidates = (" code", " token", " repository", " data")
        self.filler = next(
            text
            for text in candidates
            if len(tokenizer.encode(text, add_special_tokens=False)) == 1
        )
        # Repetition must stay one token per copy; this lets large modeled
        # prompts be materialized without re-tokenizing hundreds of MB twice.
        probe = self.filler * 1024
        if len(tokenizer.encode(probe, add_special_tokens=False)) != 1024:
            raise ValueError("selected filler is not repetition-stable")

    def make(self, target: int, marker: str) -> str:
        marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
        if len(marker_ids) >= target:
            text = self.tokenizer.decode(marker_ids[:target], skip_special_tokens=True)
        else:
            text = marker + self.filler * (target - len(marker_ids))
        # Validate the marker/filler boundary on a small equivalent string.
        probe_count = max(0, min(32, target - len(marker_ids)))
        probe = marker + self.filler * probe_count
        expected = len(marker_ids) + probe_count
        if len(self.tokenizer.encode(probe, add_special_tokens=False)) != expected:
            raise ValueError("marker/filler boundary is not token-stable")
        return text


def forced_pool(source_path: Path) -> list[list[int]]:
    with source_path.open() as file:
        source = json.load(file)
    pool = [
        list(turn.get("forced_token_ids") or [])
        for conversation in source["conversations"]
        for turn in conversation
        if turn.get("forced_token_ids")
    ]
    if not pool:
        raise ValueError("Oracle source contains no forced completions")
    return pool


def resize_forced(
    original: list[int], target: int, filler_ids: list[int]
) -> list[int]:
    if len(original) >= target:
        return original[:target]
    repeats = (target - len(original) + len(filler_ids) - 1) // len(filler_ids)
    return (original + filler_ids * repeats)[:target]


def gap_after(rows: list[dict[str, Any]], index: int) -> float:
    if index + 1 >= len(rows):
        return 0.0
    finished = parse_timestamp(rows[index].get("finished_at"))
    started = parse_timestamp(rows[index + 1].get("started_at"))
    if finished is None or started is None:
        return 0.0
    return max(0.0, (started - finished).total_seconds())


def build_dataset(
    *,
    windows: list[tuple[str, str, list[dict[str, Any]]]],
    tokenizer: Any,
    forced: list[list[int]],
    timing_mode: str,
    max_context_tokens: int,
    seed: int,
    provenance: dict[str, Any],
    workload_profile: str,
) -> dict[str, Any]:
    text_factory = ExactTextFactory(tokenizer)
    filler_ids = tokenizer.encode(
        " I will inspect the repository, use the tools, and report the validated result.",
        add_special_tokens=False,
    )
    conversations = []
    realized_rows = []
    target_prefixes = []
    prefix_errors = []
    session_steps = []
    for conversation_index, (provider, session_id, rows) in enumerate(windows):
        turns = []
        materialized_prefix = 0
        session_steps.append(len(rows))
        for turn_index, row in enumerate(rows):
            prefix = min(row["prefix_tokens"], max_context_tokens - 2)
            append = min(row["append_tokens"], max_context_tokens - prefix - 1)
            append = max(1, append)
            output = min(row["output_tokens"], max_context_tokens - prefix - append)
            output = max(1, output)
            relative_error = abs(prefix - materialized_prefix) / max(prefix, 1)
            discontinuity = (
                turn_index == 0
                or prefix < materialized_prefix - 4096
                or relative_error > 0.20
                or materialized_prefix + append + output > max_context_tokens
            )
            messages = []
            if discontinuity:
                materialized_prefix = prefix
            if discontinuity and prefix:
                messages.append(
                    {
                        "role": "system",
                        "content": text_factory.make(
                            prefix,
                            f"MODELED TRACELAB PREFIX {conversation_index:04d}. ",
                        ),
                    }
                )
            messages.append(
                {
                    "role": "user" if row["step_type"] == "user_message" else "tool",
                    "content": text_factory.make(
                        append,
                        f"MODELED {provider.upper()} APPEND {conversation_index:04d}-{turn_index:03d}. ",
                    ),
                }
            )
            donor = forced[(conversation_index * 131 + turn_index) % len(forced)]
            forced_ids = resize_forced(donor, output, filler_ids)
            delay = 0.0 if timing_mode == "saturated" else gap_after(rows, turn_index)
            turns.append(
                {
                    "messages": messages,
                    "forced_token_ids": forced_ids,
                    "max_new_tokens": len(forced_ids) + 8,
                    "reset_history": discontinuity,
                    "workload": {
                        "provider": provider,
                        "source_session_id_hash": hashlib.sha256(
                            session_id.encode()
                        ).hexdigest()[:16],
                        "source_round_index": row["round_index"],
                        "step_type": row["step_type"],
                        "target_prefix_tokens": prefix,
                        "materialized_prefix_tokens": materialized_prefix,
                        "target_append_tokens": append,
                        "target_output_tokens": output,
                        "tool_calls": row["tool_calls"],
                        "gap_after_seconds": delay,
                        "history_discontinuity": discontinuity,
                    },
                }
            )
            target_prefixes.append(prefix)
            prefix_errors.append(abs(prefix - materialized_prefix))
            realized_rows.append((materialized_prefix, append, output, delay))
            materialized_prefix += append + output
        conversations.append(turns)

    prefixes = [row[0] for row in realized_rows]
    appends = [row[1] for row in realized_rows]
    outputs = [row[2] for row in realized_rows]
    gaps = [row[3] for row in realized_rows]
    prompts = [row[0] + row[1] for row in realized_rows]
    target_prompts = [prefix + append for prefix, append in zip(target_prefixes, appends)]
    served = [sum(row[:3]) for row in realized_rows]
    metadata = {
        "num_conversations": len(conversations),
        "source": "UW SyFI TraceLab v0.0.1 plus Oracle forced completions",
        "tokenizer": provenance["tokenizer"],
        "transform": {
            "name": "tracelab-modeled-v1",
            "synthetic": True,
            "reporting_label": "trace-geometry-modeled; not verbatim TraceLab content",
            "timing_mode": timing_mode,
            "workload_profile": workload_profile,
            "seed": seed,
            "max_steps_per_session": max(session_steps),
            "max_context_tokens": max_context_tokens,
            "method": [
                "uniform deterministic sampling of TraceLab sessions",
                "contiguous windows bounded for Oracle replay practicality",
                "synthetic exact-length input text with real Oracle forced completions",
                "history reset on sampled prefix discontinuity",
            ],
            "metrics": {
                "sessions": len(conversations),
                "steps": len(realized_rows),
                "steps_per_session": summary(session_steps),
                "prefix_tokens_per_step": summary(prefixes),
                "target_prefix_tokens_per_step": summary(target_prefixes),
                "absolute_prefix_error_tokens": summary(prefix_errors),
                "append_tokens_per_step": summary(appends),
                "prompt_tokens_per_step": summary(prompts),
                "target_prompt_tokens_per_step": summary(target_prompts),
                "output_tokens_per_step": summary(outputs),
                "served_tokens_per_step": summary(served),
                "gap_after_seconds": summary(gaps),
                "output_share": sum(outputs) / sum(served),
                "planned_prefix_hit_rate": sum(prefixes)
                / (sum(prefixes) + sum(appends)),
            },
            "provenance": provenance,
            "caveat": (
                "Content counts omit chat-template overhead. Server usage counters are "
                "authoritative. reset_history requires the maintained concurrent replay driver."
            ),
        },
    }
    return {"metadata": metadata, "conversations": conversations}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--oracle-source", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--max-context-tokens", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=260630560)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Generate only this output filename; repeat as needed",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    trace_path = Path(args.trace).resolve()
    oracle_path = Path(args.oracle_source).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    sessions = load_sessions(trace_path)
    forced = forced_pool(oracle_path)
    provenance = {
        "tracelab_path": str(trace_path),
        "tracelab_sha256": sha256_file(trace_path),
        "oracle_source_path": str(oracle_path),
        "oracle_source_sha256": sha256_file(oracle_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path / "tokenizer.json"),
    }
    specs = (
        ("oracle-coding-codex-256-empirical-v1.json", "codex", 256, "empirical", "tracelab-codex"),
        ("oracle-coding-claude-256-empirical-v1.json", "claude", 256, "empirical", "tracelab-claude"),
        ("oracle-coding-mixed-800-empirical-v1.json", "mixed", 800, "empirical", "tracelab-mixed"),
        ("oracle-coding-codex-256-saturated-v1.json", "codex", 256, "saturated", "tracelab-codex"),
        ("oracle-coding-together-256-saturated-v1.json", "codex", 256, "saturated", "together-shaped"),
    )
    window_cache = {}
    for filename, provider, count, timing_mode, workload_profile in specs:
        if args.only and filename not in args.only:
            continue
        output_path = output_dir / filename
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output_path}")
        key = (provider, count)
        windows = window_cache.setdefault(
            key,
            select_windows(
                sessions,
                provider=provider,
                count=count,
                max_steps=args.max_steps,
                seed=args.seed,
            ),
        )
        if workload_profile == "together-shaped":
            windows = together_shape(windows)
        dataset = build_dataset(
            windows=windows,
            tokenizer=tokenizer,
            forced=forced,
            timing_mode=timing_mode,
            max_context_tokens=args.max_context_tokens,
            seed=args.seed,
            provenance=provenance,
            workload_profile=workload_profile,
        )
        with output_path.open("w") as file:
            json.dump(dataset, file, separators=(",", ":"))
        print(f"wrote {output_path} ({output_path.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
