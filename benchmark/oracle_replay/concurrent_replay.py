#!/usr/bin/env python3
"""Sustained concurrent oracle-trace replay with auditable per-GPU metrics.

Conversations run concurrently, while turns within each conversation remain
strictly sequential.  The scored result is a fixed post-warmup window computed
from counter deltas.  Cumulative-to-cutoff and drain-inclusive metrics are
reported separately.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import random
import string
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from .metrics import (
        MixBands,
        TokenCounters,
        scorecard_failures,
        summarize_window,
        validate_physical_gpu_count,
    )
except ImportError:
    # Support direct execution: python benchmark/oracle_replay/concurrent_replay.py
    from metrics import (  # type: ignore[no-redef]
        MixBands,
        TokenCounters,
        scorecard_failures,
        summarize_window,
        validate_physical_gpu_count,
    )


_ALPHABET = string.ascii_lowercase + " "
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REDACT_SERVER_INFO_KEYS = {
    "admin_api_key",
    "api_key",
    "decrypted_config_file",
    "decrypted_draft_config_file",
    "hf_token",
    "ssl_keyfile_password",
}


@dataclass(frozen=True)
class TimedSnapshot:
    at_s: float
    counters: TokenCounters


@dataclass(frozen=True)
class LatencySample:
    completed_at_s: float
    ttft_s: Optional[float]
    admission_wait_s: float
    event_itl_s: Optional[float]
    token_itl_s: Optional[float]
    conv_idx: int = -1
    turn_idx: int = -1
    prompt_tokens: int = 0
    cached_tokens: int = 0


class AdmissionController:
    """Strict-FIFO client admission with a bounded number of in-flight calls."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Admission capacity must be positive")
        self.capacity = capacity
        self.active = 0
        self.next_ticket = 0
        self.serving_ticket = 0
        self.condition = threading.Condition()

    def acquire(self, stop: threading.Event) -> Optional[float]:
        started_at = time.monotonic()
        with self.condition:
            ticket = self.next_ticket
            self.next_ticket += 1
            while True:
                if ticket == self.serving_ticket and (
                    stop.is_set() or self.active < self.capacity
                ):
                    self.serving_ticket += 1
                    self.condition.notify_all()
                    if stop.is_set():
                        return None
                    self.active += 1
                    return time.monotonic() - started_at
                self.condition.wait(timeout=0.1)

    def release(self) -> None:
        with self.condition:
            if self.active <= 0:
                raise RuntimeError("Admission slot released without an active request")
            self.active -= 1
            self.condition.notify_all()


class ReplayState:
    def __init__(self, fail_log: str = "") -> None:
        self.lock = threading.Lock()
        self.counters = {
            "completed_requests": 0,
            "failed_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
        }
        self.latencies: list[LatencySample] = []
        self.fail_log = fail_log

    def snapshot(self) -> TimedSnapshot:
        with self.lock:
            return TimedSnapshot(
                at_s=time.monotonic(),
                counters=TokenCounters(**self.counters),
            )

    def record_success(
        self,
        usage: dict[str, Any],
        *,
        ttft_s: Optional[float],
        admission_wait_s: float,
        event_itl_s: Optional[float],
        token_itl_s: Optional[float],
        conv_idx: int = -1,
        turn_idx: int = -1,
    ) -> None:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(
            details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
        )
        with self.lock:
            self.counters["completed_requests"] += 1
            self.counters["prompt_tokens"] += prompt_tokens
            self.counters["completion_tokens"] += completion_tokens
            self.counters["cached_prompt_tokens"] += cached_tokens
            self.latencies.append(
                LatencySample(
                    completed_at_s=time.monotonic(),
                    ttft_s=ttft_s,
                    admission_wait_s=admission_wait_s,
                    event_itl_s=event_itl_s,
                    token_itl_s=token_itl_s,
                    conv_idx=conv_idx,
                    turn_idx=turn_idx,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached_tokens,
                )
            )

    def record_failure(self, conv_idx: int, turn_idx: int, detail: str) -> None:
        with self.lock:
            self.counters["failed_requests"] += 1
            if self.fail_log:
                with open(self.fail_log, "a") as file:
                    file.write(f"conv={conv_idx} turn={turn_idx} {detail[:1000]}\n")

    def latency_summary(self, start_s: float, end_s: float) -> dict[str, Any]:
        with self.lock:
            samples = [
                sample
                for sample in self.latencies
                if start_s <= sample.completed_at_s <= end_s
            ]
        return summarize_latencies(samples)


_thread_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def _percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def summarize_latencies(samples: list[LatencySample]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(samples)}
    fields = (
        ("ttft_s", "ttft", 1.0),
        ("admission_wait_s", "admission_wait", 1.0),
        ("event_itl_s", "event_itl_ms", 1000.0),
        ("token_itl_s", "token_itl_ms", 1000.0),
    )
    for attribute, label, scale in fields:
        values = [
            value
            for sample in samples
            if (value := getattr(sample, attribute)) is not None
        ]
        if not values:
            continue
        result.update(
            {
                f"{label}_sample_count": len(values),
                f"{label}_mean": sum(values) / len(values) * scale,
                f"{label}_p50": _percentile(values, 0.50) * scale,
                f"{label}_p95": _percentile(values, 0.95) * scale,
                f"{label}_p99": _percentile(values, 0.99) * scale,
            }
        )
    return result


def _random_blob(rng: random.Random, n_chars: int) -> str:
    return "".join(rng.choices(_ALPHABET, k=n_chars))


def _throwaway_placeholder(rng: random.Random) -> str:
    return "[discarded: " + _random_blob(rng, 24) + "]"


def _stream_request(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[str, dict[str, Any], Optional[float], Optional[float], Optional[float]]:
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    request_start = time.monotonic()
    first_event_at = None
    last_event_at = None
    event_count = 0
    text_parts = []
    usage: dict[str, Any] = {}

    with _session().post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_s,
        stream=True,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        # Keep lines as bytes.  str.splitlines() treats several valid Unicode
        # characters inside JSON strings as line breaks.
        for line in response.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip().decode("utf-8")
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    now = time.monotonic()
                    first_event_at = first_event_at or now
                    last_event_at = now
                    event_count += 1
                    text_parts.append(content)
            if chunk.get("usage"):
                usage = chunk["usage"]

    ttft_s = first_event_at - request_start if first_event_at is not None else None
    event_itl_s = None
    token_itl_s = None
    if first_event_at is not None and last_event_at is not None:
        if event_count > 1:
            event_itl_s = (last_event_at - first_event_at) / (event_count - 1)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if completion_tokens > 1:
            token_itl_s = (last_event_at - first_event_at) / (completion_tokens - 1)
    return "".join(text_parts), usage, ttft_s, event_itl_s, token_itl_s


def _nonstream_request(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[str, dict[str, Any], None, None, None]:
    payload["stream"] = False
    response = _session().post(url, json=payload, headers=headers, timeout=timeout_s)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    message = body["choices"][0]["message"]
    return message.get("content") or "", body.get("usage") or {}, None, None, None


def run_conversation(
    conv_idx: int,
    conversation: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    url: str,
    state: ReplayState,
    stop: threading.Event,
    admission: Optional[AdmissionController],
) -> None:
    rng = random.Random(args.seed + conv_idx)
    try:
        if args.ramp_in > 0 and stop.wait(rng.uniform(0, args.ramp_in)):
            return

        headers = {"X-SMG-Routing-Key": f"conv-{conv_idx}"}
        history: list[dict[str, Any]] = []
        turn_spans: list[tuple[int, int]] = []
        thrown: set[int] = set()

        for turn_idx, turn in enumerate(conversation[: args.max_turns]):
            if stop.is_set():
                return

            if turn.get("reset_history"):
                history = []
                turn_spans = []
                thrown = set()

            forced = list(turn["forced_token_ids"])
            max_new_tokens = int(turn.get("max_new_tokens", len(forced) + 8))
            if args.out_cap:
                forced = forced[: args.out_cap]
                max_new_tokens = min(max_new_tokens, args.out_cap)

            start = len(history)
            messages = turn["messages"]
            if args.drop_tool_role_messages:
                messages = [
                    message for message in messages if message.get("role") != "tool"
                ]
            if args.noise_chars and messages and turn_idx % args.noise_every == 0:
                messages = [dict(message) for message in messages]
                for message in messages:
                    content = str(message.get("content") or "")
                    message["content"] = (
                        content + " " + _random_blob(rng, args.noise_chars)
                    )
            history.extend(messages)

            if (
                args.refresh_prob
                and len(turn_spans) > 1
                and rng.random() < args.refresh_prob
            ):
                # Bias invalidations toward old non-system turns and preserve
                # content size so token geometry does not collapse.
                victim = 1 + rng.randrange(max(1, (len(turn_spans) - 1) // 3))
                victim_start, victim_end = turn_spans[victim]
                for index in range(victim_start, victim_end):
                    old = str(history[index].get("content") or "")
                    history[index] = {
                        **history[index],
                        "content": _random_blob(rng, max(len(old), 16)),
                    }

            if args.throwaway_after:
                target = turn_idx - args.throwaway_after
                if target >= 0 and target not in thrown:
                    thrown.add(target)
                    victim_start, victim_end = turn_spans[target]
                    for index in range(victim_start, victim_end):
                        history[index] = {
                            **history[index],
                            "content": _throwaway_placeholder(rng),
                        }

            if (
                args.sysrefresh_every
                and turn_spans
                and turn_idx > 0
                and turn_idx % args.sysrefresh_every == 0
            ):
                system_start, _ = turn_spans[0]
                content = str(history[system_start].get("content") or "")
                history[system_start] = {
                    **history[system_start],
                    "content": (
                        _random_blob(rng, args.noise_chars or 200) + " " + content
                    ),
                }

            sent = history
            if args.window_keep and len(turn_spans) - 1 > args.window_keep:
                first_turn_end = turn_spans[0][1]
                window_start = turn_spans[-args.window_keep][0]
                sent = history[:first_turn_end] + history[window_start:]

            payload = {
                "model": args.model,
                "messages": sent,
                "max_completion_tokens": max_new_tokens,
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }
            if not args.no_forced_tokens:
                payload["forced_token_ids"] = forced
            if args.ignore_eos:
                payload["ignore_eos"] = True

            admission_started_at = time.monotonic()
            acquired_slot = False
            if admission is not None:
                admission_wait = admission.acquire(stop)
                if admission_wait is None:
                    return
                admission_wait_s = admission_wait
                acquired_slot = True
            else:
                admission_wait_s = time.monotonic() - admission_started_at

            try:
                if args.stream:
                    text, usage, ttft_s, event_itl_s, token_itl_s = _stream_request(
                        url,
                        payload,
                        headers,
                        args.turn_timeout,
                    )
                else:
                    text, usage, ttft_s, event_itl_s, token_itl_s = _nonstream_request(
                        url,
                        payload,
                        headers,
                        args.turn_timeout,
                    )
            except Exception as error:
                state.record_failure(
                    conv_idx,
                    turn_idx,
                    f"{type(error).__name__}: {error!r}",
                )
                return
            finally:
                if acquired_slot:
                    admission.release()

            history.append({"role": "assistant", "content": text})
            turn_spans.append((start, len(history)))
            if ttft_s is not None:
                # TTFT is user-observed latency, so include client-side
                # admission waiting when --max-in-flight is active.
                ttft_s += admission_wait_s
            state.record_success(
                usage,
                ttft_s=ttft_s,
                admission_wait_s=admission_wait_s,
                event_itl_s=event_itl_s,
                token_itl_s=token_itl_s,
                conv_idx=conv_idx,
                turn_idx=turn_idx,
            )

            if args.turn_gap > 0 and stop.wait(args.turn_gap * rng.uniform(0.5, 1.5)):
                return
            if args.use_dataset_gaps:
                workload = turn.get("workload") or {}
                dataset_gap = float(workload.get("gap_after_seconds") or 0)
                if dataset_gap > 0 and stop.wait(dataset_gap):
                    return

            completed_turns = turn_idx + 1
            if args.prune_every and completed_turns % args.prune_every == 0:
                keep_from_turn = max(0, len(turn_spans) - args.prune_keep)
                cut = turn_spans[keep_from_turn][0]
                history = history[cut:]
                turn_spans = [
                    (span_start - cut, span_end - cut)
                    for span_start, span_end in turn_spans[keep_from_turn:]
                ]
                thrown = {
                    old_turn - keep_from_turn
                    for old_turn in thrown
                    if old_turn >= keep_from_turn
                }
    except Exception as error:
        state.record_failure(
            conv_idx,
            -1,
            f"outer {type(error).__name__}: {error!r}",
        )


def _tile_conversations(
    conversations: list[list[dict[str, Any]]], tenants: int
) -> list[list[dict[str, Any]]]:
    if tenants == 1:
        return conversations
    tiled = []
    for tenant in range(tenants):
        for conversation in conversations:
            clone = copy.deepcopy(conversation)
            if tenant:
                salt = f"[tenant-{tenant:04d} deployment context] "
                first_message = clone[0]["messages"][0]
                first_message["content"] = salt + str(
                    first_message.get("content") or ""
                )
            tiled.append(clone)
    return tiled


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(args: list[str], *, binary: bool = False) -> Any:
    return subprocess.check_output(
        ["git", *args],
        cwd=_REPO_ROOT,
        stderr=subprocess.DEVNULL,
        text=not binary,
    )


def _git_provenance() -> dict[str, Any]:
    try:
        sha = _git_output(["rev-parse", "HEAD"]).strip()
        branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        status = _git_output(
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            binary=True,
        )
        dirty = bool(status)
        fingerprint = None
        if dirty:
            digest = hashlib.sha256()
            digest.update(_git_output(["diff", "--binary", "HEAD", "--"], binary=True))
            untracked = _git_output(
                ["ls-files", "--others", "--exclude-standard", "-z"],
                binary=True,
            )
            for raw_path in sorted(path for path in untracked.split(b"\0") if path):
                digest.update(b"\0untracked\0" + raw_path + b"\0")
                path = _REPO_ROOT / os.fsdecode(raw_path)
                if path.is_file():
                    with path.open("rb") as file:
                        while chunk := file.read(1024 * 1024):
                            digest.update(chunk)
            fingerprint = digest.hexdigest()
        return {
            "sha": sha,
            "branch": branch,
            "dirty": dirty,
            "dirty_tree_sha256": fingerprint,
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _redact_server_info(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if key.lower() in _REDACT_SERVER_INFO_KEYS and item is not None
                else _redact_server_info(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_server_info(item) for item in value]
    return value


def _fetch_server_info(urls: list[str]) -> list[dict[str, Any]]:
    results = []
    for raw_url in urls:
        url = raw_url.rstrip("/")
        if not url.endswith("/server_info"):
            url += "/server_info"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = _redact_server_info(response.json())
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            )
            results.append(
                {
                    "url": url,
                    "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                    "payload": payload,
                }
            )
        except Exception as error:
            results.append(
                {
                    "url": url,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return results


def _select_request_url(request_urls: list[str], conversation_index: int) -> str:
    """Route one conversation deterministically to a direct worker endpoint."""

    if not request_urls:
        raise ValueError("request_urls must not be empty")
    return request_urls[conversation_index % len(request_urls)]


def _window(
    start: TimedSnapshot,
    end: TimedSnapshot,
    physical_gpus: int,
) -> dict[str, Any]:
    result = summarize_window(
        start.counters,
        end.counters,
        elapsed_s=end.at_s - start.at_s,
        physical_gpus=physical_gpus,
    )
    result["start_monotonic_s"] = start.at_s
    result["end_monotonic_s"] = end.at_s
    return result


def _print_interval(
    label: str,
    start: TimedSnapshot,
    end: TimedSnapshot,
    physical_gpus: int,
    benchmark_start_s: float,
) -> None:
    result = _window(start, end, physical_gpus)
    print(
        f"  t={end.at_s - benchmark_start_s:.0f}s {label} "
        f"turns={result['completed_requests']} fail={result['failed_requests']} "
        f"cached={result['cached_prompt_tok_s']:.0f} tok/s "
        f"uncached={result['uncached_prompt_tok_s']:.0f} tok/s "
        f"output={result['completion_tok_s']:.0f} tok/s "
        f"hit={result['cache_hit_rate']:.3f} "
        f"output_share={result['output_share']:.4f} "
        f"tpm/gpu={result['served_tpm_per_gpu']:,.0f}",
        flush=True,
    )


def _validate_args(args: argparse.Namespace) -> None:
    validate_physical_gpu_count(
        args.physical_gpus,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        multi_node=args.multi_node,
    )
    positive = (
        "n_convs",
        "concurrency",
        "max_turns",
        "tenants",
        "noise_every",
        "prune_keep",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "warmup_seconds",
        "measurement_seconds",
        "report_interval",
        "turn_timeout",
        "throwaway_after",
        "noise_chars",
        "refresh_prob",
        "window_keep",
        "sysrefresh_every",
        "prune_every",
        "turn_gap",
        "ramp_in",
        "out_cap",
        "max_in_flight",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.measurement_seconds <= 0:
        raise ValueError("--measurement-seconds must be positive")
    if args.report_interval <= 0:
        raise ValueError("--report-interval must be positive")
    if args.turn_timeout <= 0:
        raise ValueError("--turn-timeout must be positive")
    if not 0 <= args.refresh_prob <= 1:
        raise ValueError("--refresh-prob must be between 0 and 1")
    if args.multi_node and not args.topology.strip():
        raise ValueError("--multi-node requires an explicit --topology")
    if args.max_in_flight > args.concurrency:
        raise ValueError("--max-in-flight cannot exceed --concurrency")
    if args.base_url and args.worker_base_url:
        raise ValueError("--base-url and --worker-base-url are mutually exclusive")
    MixBands(
        cache_hit_min=args.cache_hit_min,
        cache_hit_max=args.cache_hit_max,
        output_share_min=args.output_share_min,
        output_share_max=args.output_share_max,
    )


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    _validate_args(args)
    dataset_path = Path(args.dataset).expanduser().resolve()
    with dataset_path.open() as file:
        dataset = json.load(file)
    conversations = dataset["conversations"][: args.n_convs]
    if not conversations:
        raise ValueError("The selected dataset contains no conversations")
    conversations = _tile_conversations(conversations, args.tenants)

    base_urls = args.worker_base_url or [
        args.base_url or f"http://127.0.0.1:{args.port}"
    ]
    request_urls = [
        base_url.rstrip("/") + "/v1/chat/completions" for base_url in base_urls
    ]
    server_info_urls = args.server_info_url or base_urls
    server_info = _fetch_server_info(server_info_urls)
    state = ReplayState(args.fail_log)
    stop = threading.Event()
    benchmark_start = state.snapshot()
    timeline: dict[str, TimedSnapshot] = {}
    timeline_lock = threading.Lock()

    def controller() -> None:
        if stop.wait(args.warmup_seconds):
            return
        with timeline_lock:
            timeline["fixed_start"] = state.snapshot()
        if stop.wait(args.measurement_seconds):
            return
        with timeline_lock:
            timeline["fixed_end"] = state.snapshot()
        stop.set()

    def reporter() -> None:
        previous = benchmark_start
        while not stop.wait(args.report_interval):
            current = state.snapshot()
            _print_interval(
                "interval",
                previous,
                current,
                args.physical_gpus,
                benchmark_start.at_s,
            )
            _print_interval(
                "cumulative",
                benchmark_start,
                current,
                args.physical_gpus,
                benchmark_start.at_s,
            )
            previous = current

    controller_thread = threading.Thread(
        target=controller, name="oracle-window-controller", daemon=True
    )
    reporter_thread = threading.Thread(
        target=reporter, name="oracle-progress-reporter", daemon=True
    )
    controller_thread.start()
    reporter_thread.start()

    admission = (
        AdmissionController(args.max_in_flight)
        if args.max_in_flight and args.max_in_flight < args.concurrency
        else None
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        list(
            executor.map(
                lambda item: run_conversation(
                    item[0],
                    item[1],
                    args=args,
                    url=_select_request_url(request_urls, item[0]),
                    state=state,
                    stop=stop,
                    admission=admission,
                ),
                enumerate(conversations),
            )
        )
    stop.set()
    controller_thread.join()
    reporter_thread.join()
    final = state.snapshot()

    with timeline_lock:
        fixed_start = timeline.get("fixed_start")
        fixed_end = timeline.get("fixed_end")

    cutoff = fixed_end or final
    cumulative = _window(benchmark_start, cutoff, args.physical_gpus)
    drain_inclusive = _window(benchmark_start, final, args.physical_gpus)
    fixed = (
        _window(fixed_start, fixed_end, args.physical_gpus)
        if fixed_start is not None and fixed_end is not None
        else None
    )
    fixed_complete = fixed is not None

    bands = MixBands(
        cache_hit_min=args.cache_hit_min,
        cache_hit_max=args.cache_hit_max,
        output_share_min=args.output_share_min,
        output_share_max=args.output_share_max,
    )
    failures = (
        scorecard_failures(fixed, bands)
        if fixed is not None
        else ["fixed measurement window did not complete before traffic ended"]
    )
    if drain_inclusive["failed_requests"] and (
        fixed is None or drain_inclusive["failed_requests"] != fixed["failed_requests"]
    ):
        failures.append(
            "run-wide failed_requests="
            f"{drain_inclusive['failed_requests']} (required 0)"
        )
    if (
        fixed is not None
        and args.min_fixed_tpm_per_gpu
        and fixed["served_tpm_per_gpu"] < args.min_fixed_tpm_per_gpu
    ):
        failures.append(
            f"fixed served_tpm_per_gpu={fixed['served_tpm_per_gpu']:.3f} below "
            f"{args.min_fixed_tpm_per_gpu:.3f}"
        )
    if (
        args.min_drain_tpm_per_gpu
        and drain_inclusive["served_tpm_per_gpu"] < args.min_drain_tpm_per_gpu
    ):
        failures.append(
            "drain-inclusive "
            f"served_tpm_per_gpu={drain_inclusive['served_tpm_per_gpu']:.3f} "
            f"below {args.min_drain_tpm_per_gpu:.3f}"
        )

    fixed_latency = (
        state.latency_summary(fixed_start.at_s, fixed_end.at_s)
        if fixed_start is not None and fixed_end is not None
        else None
    )
    interactive_scorecard = {
        "ttft_p50_target_s": args.ttft_p50_target,
        "ttft_p50_observed_s": (
            fixed_latency.get("ttft_p50") if fixed_latency is not None else None
        ),
        "pass": (
            fixed_latency.get("ttft_p50") <= args.ttft_p50_target
            if fixed_latency is not None and "ttft_p50" in fixed_latency
            else None
        ),
    }
    scorecard_pass = not failures
    result = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metric_contract": (
            "(prompt_tokens + completion_tokens) / elapsed_s * 60 / physical_gpus"
        ),
        "provenance": {
            "gpu_model": args.gpu_model,
            "physical_gpus": args.physical_gpus,
            "topology": args.topology,
            "multi_node": args.multi_node,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git": _git_provenance(),
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256_file(dataset_path),
                "selected_conversations": min(
                    args.n_convs, len(dataset["conversations"])
                ),
                "tenants": args.tenants,
                "scheduled_conversations": len(conversations),
            },
            "argv": sys.argv,
            "arguments": vars(args),
            "server_info": server_info,
            "worker_routing": {
                "policy": "conversation_index_modulo",
                "base_urls": base_urls,
            },
        },
        "timing": {
            "warmup_requested_s": args.warmup_seconds,
            "measurement_requested_s": args.measurement_seconds,
            "fixed_window_complete": fixed_complete,
            "fixed_start_offset_s": (
                fixed_start.at_s - benchmark_start.at_s
                if fixed_start is not None
                else None
            ),
            "fixed_end_offset_s": (
                fixed_end.at_s - benchmark_start.at_s if fixed_end is not None else None
            ),
            "drain_end_offset_s": final.at_s - benchmark_start.at_s,
        },
        "metrics": {
            "cumulative_to_cutoff": cumulative,
            "fixed_window": fixed,
            "drain_inclusive": drain_inclusive,
        },
        "latency": {
            "fixed_window": fixed_latency,
            "drain_inclusive": state.latency_summary(benchmark_start.at_s, final.at_s),
        },
        "scorecard": {
            "bands": asdict(bands),
            "min_fixed_tpm_per_gpu": args.min_fixed_tpm_per_gpu or None,
            "min_drain_tpm_per_gpu": args.min_drain_tpm_per_gpu or None,
            "pass": scorecard_pass,
            "failures": failures,
            "enforced": not args.allow_scorecard_failure,
            "interactive": interactive_scorecard,
        },
    }

    output_path = Path(args.out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    if args.latency_samples_out:
        samples_path = Path(args.latency_samples_out).expanduser()
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w") as file:
            for sample in state.latencies:
                record = asdict(sample)
                record["completed_at_s"] -= benchmark_start.at_s
                file.write(json.dumps(record) + "\n")

    terminal_summary = {
        "result_path": str(output_path.resolve()),
        "fixed_window": fixed,
        "drain_inclusive": drain_inclusive,
        "scorecard": result["scorecard"],
    }
    print(json.dumps(terminal_summary, indent=2, sort_keys=True), flush=True)
    return result, scorecard_pass or args.allow_scorecard_failure


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    endpoint = parser.add_argument_group("endpoint and provenance")
    endpoint.add_argument("--base-url", default="")
    endpoint.add_argument(
        "--worker-base-url",
        action="append",
        default=[],
        help=(
            "Direct worker base URL; repeat for deterministic conversation-sticky "
            "round-robin routing. Mutually exclusive with --base-url."
        ),
    )
    endpoint.add_argument("--port", type=int, default=8000)
    endpoint.add_argument("--model", default="mm27")
    endpoint.add_argument("--dataset", required=True)
    endpoint.add_argument("--out", required=True, help="Result JSON path")
    endpoint.add_argument(
        "--latency-samples-out",
        help="Optional JSONL of per-request latency samples (diagnostic)",
    )
    endpoint.add_argument("--physical-gpus", type=int, required=True)
    endpoint.add_argument("--gpu-model", required=True)
    endpoint.add_argument(
        "--topology",
        required=True,
        help="Explicit topology, for example '1 node; TP4+EP4; GPUs 0,1,2,3'",
    )
    endpoint.add_argument(
        "--multi-node",
        action="store_true",
        help="Allow CUDA_VISIBLE_DEVICES to differ from the physical denominator",
    )
    endpoint.add_argument(
        "--server-info-url",
        action="append",
        default=[],
        help="Server base URL or /server_info URL; repeat for each worker",
    )

    timing = parser.add_argument_group("timing and load")
    timing.add_argument("--warmup-seconds", type=float, default=60)
    timing.add_argument("--measurement-seconds", type=float, default=600)
    timing.add_argument("--report-interval", type=float, default=30)
    timing.add_argument("--n-convs", type=int, default=60)
    timing.add_argument("--concurrency", type=int, default=16)
    timing.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help=(
            "Client-side admission cap independent of conversation concurrency "
            "(0=use --concurrency). Admission wait is included in TTFT."
        ),
    )
    timing.add_argument("--max-turns", type=int, default=25)
    timing.add_argument(
        "--turn-timeout",
        type=float,
        default=600,
        help=(
            "Silence tolerated on a turn, in seconds -- requests applies a scalar "
            "timeout as (connect, read), and under stream=True the read timeout "
            "measures the gap between chunks, not the turn's total duration. Keep "
            "it that way: a total deadline here would fail long generations the "
            "server completed correctly. Matches the gateway deadline ladder "
            "(sgl-model-gateway core::deadlines)."
        ),
    )
    timing.add_argument("--turn-gap", type=float, default=0)
    timing.add_argument(
        "--use-dataset-gaps",
        action="store_true",
        help="Sleep for each turn's workload.gap_after_seconds when present",
    )
    timing.add_argument("--ramp-in", type=float, default=0)
    timing.add_argument("--seed", type=int, default=42)
    timing.add_argument("--fail-log", default=os.environ.get("CCD_FAIL_LOG", ""))

    traffic = parser.add_argument_group("traffic-shape controls")
    traffic.add_argument("--tenants", type=int, default=1)
    traffic.add_argument("--throwaway-after", type=int, default=0)
    traffic.add_argument(
        "--drop-tool-role-messages",
        action="store_true",
        help=(
            "Filter role=='tool' messages out of each turn before replay. "
            "Oracle-forced assistant turns replay text only, with no "
            "synthetic tool_calls metadata, so a later tool-role message "
            "in the trace fails chat-completions validation ('no previous "
            "assistant message with a tool call'). This is an approximate "
            "workaround (drops real tool-result content, shortening the "
            "traffic shape), not a fix for the underlying dataset gap."
        ),
    )
    traffic.add_argument("--noise-chars", type=int, default=0)
    traffic.add_argument("--noise-every", type=int, default=1)
    traffic.add_argument("--refresh-prob", type=float, default=0)
    traffic.add_argument("--window-keep", type=int, default=0)
    traffic.add_argument("--sysrefresh-every", type=int, default=0)
    traffic.add_argument("--prune-every", type=int, default=0)
    traffic.add_argument("--prune-keep", type=int, default=3)
    traffic.add_argument("--out-cap", type=int, default=0)
    traffic.add_argument("--ignore-eos", action="store_true")
    traffic.add_argument(
        "--no-forced-tokens",
        action="store_true",
        help="Sample freely instead of replaying the oracle tokens, for servers "
        "started without oracle commit support. Turn lengths still come from the "
        "dataset; pair with --ignore-eos to keep the output share.",
    )
    traffic.add_argument("--stream", action="store_true")

    scorecard = parser.add_argument_group("scorecard")
    scorecard.add_argument("--cache-hit-min", type=float, default=0.94)
    scorecard.add_argument("--cache-hit-max", type=float, default=0.96)
    scorecard.add_argument("--output-share-min", type=float, default=0.0065)
    scorecard.add_argument("--output-share-max", type=float, default=0.0085)
    scorecard.add_argument("--min-fixed-tpm-per-gpu", type=float, default=0)
    scorecard.add_argument("--min-drain-tpm-per-gpu", type=float, default=0)
    scorecard.add_argument("--ttft-p50-target", type=float, default=3.0)
    scorecard.add_argument(
        "--allow-scorecard-failure",
        action="store_true",
        help="Write exploratory results but exit zero when any scorecard gate misses",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        _, passed = run(args)
    except Exception as error:
        print(f"benchmark configuration/run failed: {error}", file=sys.stderr)
        return 2
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
