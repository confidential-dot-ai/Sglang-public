#!/usr/bin/env python3
"""Correlate TP-rank PyTorch traces into a DSpark critical-path timeline.

The analyzer relies on two invariants exposed by PyTorch's Perfetto export:

* every ``cudaGraphLaunch`` has a launch correlation id;
* every kernel launched by that graph carries the same correlation id.

This makes graph-internal attribution exact even though the target verify
forward is captured as one full CUDA graph.  Optional DSpark metadata dumps
add accepted-token and context-length fields, and an optional expert recorder
dump adds routed-expert assignment counts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


DECODE_FN = "dspark_components/dspark_worker_v2.py("
RANK_RE = re.compile(r"TP-(\d+)-EP-(\d+)")
STEP_RE = re.compile(r"step\[TARGET_VERIFY bs=(\d+)\]")
EXTEND_RE = re.compile(r"step\[EXTEND bs=(\d+) toks=(\d+)\]")


def _load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload.get("traceEvents", payload)


def _rank(path: Path) -> int:
    match = RANK_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot extract TP rank from {path}")
    return int(match.group(1))


def _is_python_call(event: dict, suffix: str) -> bool:
    name = str(event.get("name", ""))
    return (
        event.get("ph") == "X"
        and event.get("cat") == "python_function"
        and DECODE_FN in name
        and name.endswith(suffix)
    )


def _kernel_bucket(name: str) -> str:
    lower = name.lower()
    if any(
        token in lower
        for token in (
            "nccl",
            "all_reduce",
            "allreduce",
            "all_gather",
            "allgather",
            "reduce_e4m3x8_bf16",
            "broadcast_ring",
        )
    ):
        return "communication"
    if any(
        token in lower
        for token in (
            "groupproblemshape",
            "doactivationkernel",
            "expandinputrowskernel",
            "_router_triton_kernel",
        )
    ):
        return "moe_compute"
    if any(
        token in lower
        for token in (
            "sparseattentionforward",
            "flash_attn_fwd_with_block_score",
            "convert_exp_scales_to_e4m3",
            "_topk_index_kernel",
            "_build_direct_decode_schedule_kernel",
            "batchprefillwithpagedkvcachekernel",
            "fused_gemma_qknorm_rope",
        )
    ):
        return "attention_indexer"
    if any(
        token in lower
        for token in (
            "mxfp8_block_scaled_matmul",
            "nvjet_sm100",
            "cublas",
        )
    ):
        return "dense_matmul"
    return "other"


def _graph_stats(events: list[dict], launch: dict) -> dict:
    correlation = launch.get("args", {}).get("correlation")
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and event.get("args", {}).get("correlation") == correlation
    ]
    if not kernels:
        return {
            "correlation": correlation,
            "host_launch_ms": launch.get("dur", 0.0) / 1000.0,
            "gpu_start_us": None,
            "gpu_end_us": None,
            "gpu_span_ms": None,
            "kernel_work_ms": {},
            "kernel_count": 0,
        }
    work = defaultdict(float)
    for kernel in kernels:
        work[_kernel_bucket(str(kernel.get("name", "")))] += (
            kernel.get("dur", 0.0) / 1000.0
        )
    start = min(event["ts"] for event in kernels)
    end = max(event["ts"] + event.get("dur", 0.0) for event in kernels)
    return {
        "correlation": correlation,
        "host_launch_ms": round(launch.get("dur", 0.0) / 1000.0, 4),
        "gpu_start_us": start,
        "gpu_end_us": end,
        "gpu_span_ms": round((end - start) / 1000.0, 4),
        "kernel_work_ms": {key: round(value, 4) for key, value in work.items()},
        "kernel_count": len(kernels),
    }


def _nested(events: list[dict], parent: dict, *, name: str | None = None) -> list[dict]:
    start = parent["ts"]
    end = start + parent.get("dur", 0.0)
    pid = parent.get("pid")
    tid = parent.get("tid")
    return [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") == pid
        and event.get("tid") == tid
        and event.get("ts", -math.inf) >= start
        and event.get("ts", 0.0) + event.get("dur", 0.0) <= end
        and (name is None or event.get("name") == name)
    ]


def _parse_rank_trace(path: Path) -> dict:
    rank = _rank(path)
    events = _load_trace(path)
    decodes = sorted(
        (event for event in events if _is_python_call(event, "_forward_decode")),
        key=lambda event: event["ts"],
    )
    prefills = sorted(
        (event for event in events if _is_python_call(event, "_forward_prefill")),
        key=lambda event: event["ts"],
    )
    model_calls = sorted(decodes + prefills, key=lambda event: event["ts"])
    previous_end = {
        id(call): (
            None
            if index == 0
            else model_calls[index - 1]["ts"]
            + model_calls[index - 1].get("dur", 0.0)
        )
        for index, call in enumerate(model_calls)
    }

    decoded = []
    for ordinal, call in enumerate(decodes):
        steps = sorted(
            (
                event
                for event in _nested(events, call)
                if event.get("cat") == "user_annotation"
                and STEP_RE.fullmatch(str(event.get("name", "")))
            ),
            key=lambda event: event["ts"],
        )
        if len(steps) != 2:
            raise RuntimeError(
                f"rank {rank} decode {ordinal}: expected draft+target steps, got {len(steps)}"
            )
        graph_stats = []
        for step in steps:
            launches = _nested(events, step, name="cudaGraphLaunch")
            if not launches:
                raise RuntimeError(
                    f"rank {rank} decode {ordinal}: no cudaGraphLaunch in {step['name']}"
                )
            graph_stats.append(_graph_stats(events, launches[-1]))

        match = STEP_RE.fullmatch(str(steps[-1]["name"]))
        prior_end = previous_end[id(call)]
        decoded.append(
            {
                "ordinal": ordinal,
                "rank": rank,
                "batch_size": int(match.group(1)),
                "cpu_start_us": call["ts"],
                "cpu_call_ms": round(call.get("dur", 0.0) / 1000.0, 4),
                "cpu_gap_ms": (
                    None
                    if prior_end is None
                    else round((call["ts"] - prior_end) / 1000.0, 4)
                ),
                "draft": graph_stats[0],
                "target_verify": graph_stats[1],
            }
        )

    parsed_prefills = []
    for call in prefills:
        spans = [
            event
            for event in _nested(events, call)
            if event.get("cat") == "user_annotation"
            and EXTEND_RE.fullmatch(str(event.get("name", "")))
        ]
        if not spans:
            continue
        match = EXTEND_RE.fullmatch(str(sorted(spans, key=lambda e: e["ts"])[0]["name"]))
        parsed_prefills.append(
            {
                "rank": rank,
                "cpu_start_us": call["ts"],
                "cpu_call_ms": round(call.get("dur", 0.0) / 1000.0, 4),
                "batch_size": int(match.group(1)),
                "tokens": int(match.group(2)),
            }
        )
    return {"rank": rank, "decodes": decoded, "prefills": parsed_prefills}


def _load_metadata(directory: Path | None) -> dict[int, list[dict]]:
    if directory is None:
        return {}
    result = {}
    for path in directory.glob("dspark-critical-path-tp*.json"):
        payload = json.loads(path.read_text())
        result[int(payload["tp_rank"])] = payload["records"]
    return result


def _attach_metadata(rank_data: dict, records: list[dict]) -> None:
    candidates = [record for record in records if record.get("host_begin_ns")]
    decodes = rank_data["decodes"]
    if not candidates or not decodes:
        return

    # Kineto exports a different monotonic clock domain on some CUDA/PyTorch
    # builds, so an absolute perf_counter_ns -> trace timestamp comparison is
    # not portable.  First try the timestamp join.  If the clock domains do
    # not agree, locate the recorded window using its batch-size sequence and
    # use the event-timed target duration to disambiguate repeated sequences.
    timestamp_matches = []
    for decode in decodes:
        record = min(
            candidates,
            key=lambda item: abs(
                item["host_begin_ns"] / 1000.0 - decode["cpu_start_us"]
            ),
        )
        delta_ms = (
            abs(record["host_begin_ns"] / 1000.0 - decode["cpu_start_us"])
            / 1000.0
        )
        timestamp_matches.append((record, delta_ms))

    if all(delta_ms <= 20.0 for _, delta_ms in timestamp_matches):
        matches = [record for record, _ in timestamp_matches]
    elif len(candidates) >= len(decodes):
        def window_score(offset: int) -> tuple[int, float]:
            window = candidates[offset : offset + len(decodes)]
            batch_mismatches = sum(
                record.get("bs") != decode["batch_size"]
                for record, decode in zip(window, decodes, strict=True)
            )
            duration_error = sum(
                abs(
                    float(record.get("target_verify_gpu_ms") or 0.0)
                    - float(decode["target_verify"]["gpu_span_ms"] or 0.0)
                )
                for record, decode in zip(window, decodes, strict=True)
            )
            return batch_mismatches, duration_error

        offset = min(
            range(len(candidates) - len(decodes) + 1), key=window_score
        )
        matches = candidates[offset : offset + len(decodes)]
    else:
        return

    for decode, record in zip(decodes, matches, strict=True):
        if record.get("bs") != decode["batch_size"]:
            continue
        decode["forward_ct"] = record.get("forward_ct")
        decode["accepted_tokens"] = record.get("accepted_tokens")
        decode["context_len_min"] = record.get("context_len_min")
        decode["context_len_p50"] = record.get("context_len_p50")
        decode["context_len_max"] = record.get("context_len_max")
        decode["event_cpu_gap_ms"] = record.get("cpu_gap_ms")
        decode["event_host_step_ms"] = record.get("host_step_ms")
        decode["event_step_cpu_ms"] = record.get("step_cpu_ms")
        decode["event_step_gpu_ms"] = record.get("step_gpu_ms")
        decode["event_draft_gpu_ms"] = record.get("draft_gpu_ms")
        decode["event_target_verify_gpu_ms"] = record.get(
            "target_verify_gpu_ms"
        )


def _load_expert_steps(path: Path | None):
    if path is None:
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    counts = payload["logical_count"]
    counts = counts.tolist() if hasattr(counts, "tolist") else counts
    # The stat recorder uses a fixed circular buffer and dumps unused slots as
    # zeros.  Preserve execution order while excluding those empty slots.
    return [step for step in counts if sum(sum(layer) for layer in step) > 0]


def _timeline(rank_zero: dict) -> list[dict]:
    entries = [
        {
            "phase": "decode",
            "start_us": row["cpu_start_us"],
            "batch_size": row["batch_size"],
            "expected_tokens": row["batch_size"] * 9,
            "decode_ordinal": row["ordinal"],
        }
        for row in rank_zero["decodes"]
    ]
    entries += [
        {
            "phase": "prefill",
            "start_us": row["cpu_start_us"],
            "batch_size": row["batch_size"],
            "expected_tokens": row["tokens"],
            "prefill_tokens": row["tokens"],
        }
        for row in rank_zero["prefills"]
    ]
    return sorted(entries, key=lambda entry: entry["start_us"])


def _attach_expert_counts(
    timeline: list[dict], expert_steps, *, num_experts: int, ep_size: int, top_k: int
) -> None:
    if not expert_steps or len(expert_steps) < len(timeline):
        return

    totals = [sum(sum(layer) for layer in step) for step in expert_steps]
    expected = [entry["expected_tokens"] for entry in timeline]

    def window_score(offset: int) -> float:
        ratios = []
        for index, token_count in enumerate(expected):
            total = totals[offset + index]
            if total > 0 and token_count > 0:
                ratios.append(math.log(total / token_count))
        if not ratios:
            return math.inf
        center = statistics.median(ratios)
        return sum(abs(value - center) for value in ratios)

    max_offset = len(expert_steps) - len(timeline)
    offset = min(range(max_offset + 1), key=window_score)
    selected = expert_steps[offset : offset + len(timeline)]

    # TP ranks observe the same routing decision and the recorder's dump
    # all-reduces them. Infer that replication factor from the known top-k
    # assignment count rather than hard-coding TP size.
    factors = []
    factor_rows = list(zip(timeline, selected, strict=True))
    # Prefill has no decode CUDA-graph padding, so its token count is exact.
    # Prefer it when inferring the TP replication introduced by all_reduce.
    prefill_rows = [row for row in factor_rows if row[0]["phase"] == "prefill"]
    for entry, step in prefill_rows or factor_rows:
        total = sum(sum(layer) for layer in step)
        num_moe_layers = len(step)
        denominator = entry["expected_tokens"] * top_k * num_moe_layers
        if denominator:
            factors.append(total / denominator)
    replication = max(1.0, round(statistics.median(factors))) if factors else 1.0
    experts_per_rank = num_experts // ep_size

    for entry, step in zip(timeline, selected, strict=True):
        per_rank = []
        for rank in range(ep_size):
            begin = rank * experts_per_rank
            end = begin + experts_per_rank
            value = sum(sum(layer[begin:end]) for layer in step) / replication
            per_rank.append(round(value))
        entry["expert_assignments_per_rank"] = per_rank
        mean = statistics.fmean(per_rank) if per_rank else 0.0
        entry["expert_load_max_over_mean"] = (
            round(max(per_rank) / mean, 4) if mean else None
        )
        entry["expert_recorder_replication_factor"] = replication


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def analyze(args) -> dict:
    traces = sorted(Path(args.trace_dir).glob("*.trace.json*"))
    if len(traces) != args.tp_size:
        raise RuntimeError(
            f"expected {args.tp_size} rank traces in {args.trace_dir}, found {len(traces)}"
        )
    ranks = {_rank(path): _parse_rank_trace(path) for path in traces}
    metadata = _load_metadata(Path(args.metadata_dir) if args.metadata_dir else None)
    for rank, records in metadata.items():
        if rank in ranks:
            _attach_metadata(ranks[rank], records)

    decode_count = {len(rank["decodes"]) for rank in ranks.values()}
    if len(decode_count) != 1:
        raise RuntimeError(f"rank decode counts differ: {decode_count}")

    iterations = []
    for ordinal in range(next(iter(decode_count))):
        rows = [ranks[rank]["decodes"][ordinal] for rank in sorted(ranks)]
        gpu_ends = [row["target_verify"]["gpu_end_us"] for row in rows]
        target_spans = [row["target_verify"]["gpu_span_ms"] for row in rows]
        slowest_end = max(gpu_ends)
        earliest_start = min(row["draft"]["gpu_start_us"] for row in rows)
        latest_end_rank = rows[gpu_ends.index(slowest_end)]["rank"]
        slowest_rank = rows[target_spans.index(max(target_spans))]["rank"]
        shortest_target_span = min(target_spans)
        for row in rows:
            row["rank_barrier_slack_ms"] = round(
                (slowest_end - row["target_verify"]["gpu_end_us"]) / 1000.0, 4
            )
            # Full-graph target replays finish at nearly the same timestamp
            # because their collectives synchronize them. Excess span over
            # the fastest rank is therefore the useful rank-wait proxy; the
            # post-graph endpoint slack alone is normally close to zero.
            row["target_span_excess_ms"] = round(
                row["target_verify"]["gpu_span_ms"] - shortest_target_span, 4
            )
        iterations.append(
            {
                "ordinal": ordinal,
                "batch_size": rows[0]["batch_size"],
                "slowest_rank": slowest_rank,
                "latest_end_rank": latest_end_rank,
                "four_rank_critical_path_ms": round(
                    (slowest_end - earliest_start) / 1000.0, 4
                ),
                "ranks": rows,
            }
        )

    timeline = _timeline(ranks[0])
    _attach_expert_counts(
        timeline,
        _load_expert_steps(Path(args.expert_record) if args.expert_record else None),
        num_experts=args.num_experts,
        ep_size=args.tp_size,
        top_k=args.expert_top_k,
    )
    expert_by_decode = {
        entry["decode_ordinal"]: entry
        for entry in timeline
        if entry["phase"] == "decode"
    }
    for iteration in iterations:
        expert = expert_by_decode.get(iteration["ordinal"], {})
        for key in (
            "expert_assignments_per_rank",
            "expert_load_max_over_mean",
            "expert_recorder_replication_factor",
        ):
            if key in expert:
                iteration[key] = expert[key]

    critical = [item["four_rank_critical_path_ms"] for item in iterations]
    target_spans = [
        row["target_verify"]["gpu_span_ms"]
        for item in iterations
        for row in item["ranks"]
    ]
    categories = defaultdict(list)
    for item in iterations:
        for row in item["ranks"]:
            for category, value in row["target_verify"]["kernel_work_ms"].items():
                categories[category].append(value)

    summary = {
        "decode_iterations": len(iterations),
        "prefill_iterations": sum(
            1 for entry in timeline if entry["phase"] == "prefill"
        ),
        "critical_path_ms_mean": round(statistics.fmean(critical), 4),
        "critical_path_ms_p95": round(_percentile(critical, 0.95), 4),
        "target_verify_span_ms_mean_all_ranks": round(
            statistics.fmean(target_spans), 4
        ),
        "target_kernel_work_ms_mean_all_ranks": {
            key: round(statistics.fmean(values), 4)
            for key, values in sorted(categories.items())
        },
        "longest_target_span_rank_counts": {
            str(rank): sum(item["slowest_rank"] == rank for item in iterations)
            for rank in sorted(ranks)
        },
    }
    return {
        "trace_dir": str(args.trace_dir),
        "summary": summary,
        "timeline": timeline,
        "decode_iterations": iterations,
    }


def _print_summary(result: dict) -> None:
    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    print()
    print(
        "iter bs critical_ms slowest "
        "target_ms[r0,r1,r2,r3] comm_ms[r0,r1,r2,r3] slack_ms[r0,r1,r2,r3]"
    )
    for item in result["decode_iterations"]:
        target = [row["target_verify"]["gpu_span_ms"] for row in item["ranks"]]
        comm = [
            row["target_verify"]["kernel_work_ms"].get("communication", 0.0)
            for row in item["ranks"]
        ]
        slack = [row["rank_barrier_slack_ms"] for row in item["ranks"]]
        print(
            f"{item['ordinal']:>4} {item['batch_size']:>2} "
            f"{item['four_rank_critical_path_ms']:>11.3f} r{item['slowest_rank']} "
            f"{','.join(f'{x:.2f}' for x in target)} "
            f"{','.join(f'{x:.2f}' for x in comm)} "
            f"{','.join(f'{x:.2f}' for x in slack)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--metadata-dir")
    parser.add_argument("--expert-record")
    parser.add_argument("--output")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--expert-top-k", type=int, default=4)
    args = parser.parse_args()
    result = analyze(args)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    _print_summary(result)


if __name__ == "__main__":
    main()
