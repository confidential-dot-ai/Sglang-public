"""Summarize inherited fixed-window metrics and non-synchronizing tail timings."""

import argparse
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
root = a.output
# Rank zero only: each TP rank logs the same batch.
pattern = re.compile(
    r"CED_TAIL iter=(\d+) reqs=(\d+) rows=(\d+) prepare_ms=([\d.]+) wall_ms=([\d.]+)"
)
gpu_pattern = re.compile(
    r"CED_TAIL_GPU iter=(\d+) reqs=(\d+) rows=(\d+) gpu_ms=([\d.]+)"
)
tails, gpu = [], {}
capture_pending = False
check_pending = False
profiled = set()
for line in (root / "server/decode.log").read_text().splitlines():
    if "TP0" not in line:
        continue
    if (
        "Captured CED late-layer graph:" in line
        or "Captured CED encoder graph:" in line
    ):
        capture_pending = True
    if "CED" in line and ("check passed" in line or "check FAILED" in line):
        check_pending = True
    if "CED tail profile exported to" in line:
        profiled.add(int(line.rsplit("tail-", 1)[1]))
    m = pattern.search(line)
    if m:
        i, n, rows, prep, wall = map(float, m.groups())
        tails.append(
            dict(
                iter=int(i),
                reqs=int(n),
                rows=int(rows),
                prepare_ms=prep,
                wall_ms=wall,
                cold_capture=capture_pending,
                eager_check=check_pending,
            )
        )
        capture_pending = False
        check_pending = False
    m = gpu_pattern.search(line)
    if m:
        gpu[int(m[1])] = float(m[4])
by_size = {}
for n in sorted({x["reqs"] for x in tails}):
    selected = [x for x in tails if x["reqs"] == n]
    warm = [
        x
        for x in selected
        if not x["cold_capture"] and not x["eager_check"] and x["iter"] not in profiled
    ]
    gpu_samples = [gpu[x["iter"]] / n for x in selected if x["iter"] in gpu]
    by_size[n] = dict(
        samples=len(selected),
        captures=sum(x["cold_capture"] for x in selected),
        eager_checks=sum(x["eager_check"] for x in selected),
        host_ms_per_request_p50=statistics.median(
            (x["wall_ms"] + x["prepare_ms"]) / n for x in selected
        ),
        gpu_ms_per_request_p50=statistics.median(gpu_samples) if gpu_samples else None,
        warm_samples=len(warm),
        warm_host_ms_per_batch_p50=statistics.median(
            x["wall_ms"] + x["prepare_ms"] for x in warm
        )
        if warm
        else None,
    )
results = dict(tail_by_batch_size=by_size, sweeps=[])
for c in (56, 72):
    f = root / f"c{c}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    m, l = d["metrics"]["fixed_window"], d["latency"]["fixed_window"]
    results["sweeps"].append(
        dict(
            concurrency=c,
            served_tpm_per_gpu=m["served_tpm_per_gpu"],
            cache_hit=m["cache_hit_rate"],
            ttft_p95=l["ttft_p95"],
            user_tps_p50=1000 / l["token_itl_ms_p50"],
            user_tps_slowest5=1000 / l["token_itl_ms_p95"],
            event_itl_p50_ms=l["event_itl_ms_p50"],
            completed=m["completed_requests"],
            failed=m["failed_requests"],
        )
    )
(root / "results-summary.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
