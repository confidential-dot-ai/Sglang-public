"""Summarize actual CUDA kernel durations, distinct from event elapsed time."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("trace", type=Path)
a = p.parse_args()
events = json.loads(a.trace.read_text())["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
by_name = defaultdict(lambda: [0, 0.0])
for event in kernels:
    row = by_name[event["name"]]
    row[0] += 1
    row[1] += event["dur"] / 1000
rows = [
    dict(kernel=name, count=count, total_ms=ms, mean_ms=ms / count)
    for name, (count, ms) in by_name.items()
]
rows.sort(key=lambda x: x["total_ms"], reverse=True)
intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
merged = []
for start, end in intervals:
    if merged and start <= merged[-1][1]:
        merged[-1][1] = max(end, merged[-1][1])
    else:
        merged.append([start, end])
result = dict(
    kernel_count=len(kernels),
    summed_kernel_ms=sum(x["total_ms"] for x in rows),
    kernel_busy_union_ms=sum(y - x for x, y in merged) / 1000,
    first_to_last_kernel_ms=(max(y for _, y in intervals) - intervals[0][0]) / 1000
    if intervals
    else 0,
    kernels=rows,
)
a.trace.with_suffix(".kernels.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, indent=2))
for row in rows[:30]:
    print(
        f"{row['total_ms']:9.3f} ms {row['count']:5d} calls {row['mean_ms']:8.3f} ms/call  {row['kernel']}"
    )
