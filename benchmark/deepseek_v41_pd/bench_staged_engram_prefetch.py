"""Per-step cost of the staged Engram breaks: blocking gathers vs worker-thread prefetch.

A synthetic checkpoint stands in for the real tables. GPU "layers" are spin kernels
of a calibrated length, so the timeline matches a decode/verify step: hash ->
layer 0 -> Engram 1 -> layers 2-13 -> Engram 14 -> the remaining layers. Run on one
GPU: CUDA_VISIBLE_DEVICES=0 python bench_staged_engram_prefetch.py
"""

import argparse
import json
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.layers.engram_staging import (  # noqa: E402
    StagedEngramPrefetcher,
    StagedEngramTable,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (  # noqa: E402
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)

LAYER_IDS = (1, 14)
SCALE_BLOCK = 32
CHUNK = 1 << 26


def write_checkpoint(root, rows, dim):
    """Minimal safetensors file with random finite FP8 payloads, written in chunks."""
    header, offset, plan = {}, 0, []
    for layer in LAYER_IDS:
        for suffix, dtype, cols in (
            ("weight", "F8_E4M3", dim),
            ("scale", "F8_E8M0", dim // SCALE_BLOCK),
        ):
            size = rows * cols
            name = f"layers.{layer}.engram.embed.{suffix}"
            header[name] = {
                "dtype": dtype,
                "shape": [rows, cols],
                "data_offsets": [offset, offset + size],
            }
            plan.append((size, suffix == "weight"))
            offset += size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    generator = torch.Generator().manual_seed(0)
    with open(root / "table.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for size, is_weight in plan:
            low, high = (0, 120) if is_weight else (120, 134)
            for start in range(0, size, CHUNK):
                n = min(CHUNK, size - start)
                chunk = torch.randint(
                    low, high, (n,), dtype=torch.uint8, generator=generator
                )
                f.write(chunk.numpy().tobytes())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in header}})
    )


def capture(forward):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = BreakableCUDAGraph()
    with BreakableCUDAGraphCapture(graph, stream=stream):
        output = forward()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def cycles_per_ms():
    cycles = 50_000_000
    torch.cuda._sleep(cycles)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    end.synchronize()
    return cycles / start.elapsed_time(end)


def build_step(tables, prefetcher, static_ids, spins):
    """One decode-shaped step. tables=None gives the GPU-only floor."""
    pre, mid, post = spins

    def forward():
        hash_ids = static_ids + 0
        if prefetcher is not None:
            prefetcher.start(hash_ids)
        torch.cuda._sleep(pre)
        first = tables[0].lookup(hash_ids[:, 0]) if tables else hash_ids[:, 0]
        torch.cuda._sleep(mid)
        second = tables[1].lookup(hash_ids[:, 1]) if tables else hash_ids[:, 1]
        torch.cuda._sleep(post)
        return (first.float() * 2 + second.float()).sum(dim=1)

    return capture(forward)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(values):
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "p50_ms": round(percentile(values, 0.5), 3),
        "p90_ms": round(percentile(values, 0.9), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
    }


def time_gathers(table, rows, num_ids, count, executor=None):
    samples = []
    for _ in range(count):
        ids = torch.randint(0, rows, (num_ids,))
        start = time.perf_counter()
        if executor is None:
            table.gather_rows(ids)
        else:
            executor.submit(table.gather_rows, ids).result()
        samples.append((time.perf_counter() - start) * 1e3)
    return summarize(samples[10:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4_000_000, help="rows per table")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--cols", type=int, default=24)
    parser.add_argument("--layer-ms", type=float, default=1.4)
    parser.add_argument("--mid-layers", type=int, default=12)
    parser.add_argument("--post-layers", type=int, default=29)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads", type=int, default=0, help="torch threads, 0=default"
    )
    parser.add_argument("--gather-threads", type=int, default=0)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    main_threads = torch.get_num_threads()
    spin = cycles_per_ms() * args.layer_ms
    spins = (int(spin), int(spin * args.mid_layers), int(spin * args.post_layers))
    gpu_ms = args.layer_ms * (1 + args.mid_layers + args.post_layers)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_checkpoint(root, args.rows, args.dim)
        tables = [
            StagedEngramTable(root, layer, args.rows, args.dim) for layer in LAYER_IDS
        ]
        prefetcher = StagedEngramPrefetcher(
            tables=dict(enumerate(tables)), gather_threads=args.gather_threads
        )
        num_ids = args.tokens * args.cols
        pool = torch.randint(
            0, args.rows, (64, args.tokens, len(LAYER_IDS), args.cols), device="cuda"
        )
        static_ids = pool[0].clone()
        variants = {
            "floor": build_step(None, None, static_ids, spins),
            "blocking": build_step(tables, None, static_ids, spins),
            "prefetch": build_step(tables, prefetcher, static_ids, spins),
        }
        worker_threads = prefetcher.executor
        assert worker_threads is not None

        outputs = {name: [] for name in variants}
        for i in range(8):
            static_ids.copy_(pool[i])
            for name, (graph, output) in variants.items():
                graph.replay()
                outputs[name].append(output.clone())
        torch.cuda.synchronize()
        identical = all(
            torch.equal(a, b) for a, b in zip(outputs["blocking"], outputs["prefetch"])
        )

        samples = {name: [] for name in variants}
        for _ in range(args.repeats):
            for name, (graph, _) in variants.items():
                for i in range(args.iters + 10):
                    static_ids.copy_(pool[i % len(pool)])
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graph.replay()
                    torch.cuda.current_stream().synchronize()
                    if i >= 10:
                        samples[name].append((time.perf_counter() - start) * 1e3)

        worker_count = worker_threads.submit(torch.get_num_threads).result()
        report = {
            "rows_per_table": args.rows,
            "tokens": args.tokens,
            "ids_per_layer": num_ids,
            "gpu_ms_per_step": round(gpu_ms, 2),
            "main_threads": main_threads,
            "worker_threads": worker_count,
            "main_threads_after": torch.get_num_threads(),
            "bit_identical": identical,
            "gather_main": time_gathers(tables[0], args.rows, num_ids, 60),
            "gather_worker": time_gathers(
                tables[0], args.rows, num_ids, 60, executor=worker_threads
            ),
            "step": {name: summarize(values) for name, values in samples.items()},
        }
        step = report["step"]
        report["saved_ms_mean"] = round(
            step["blocking"]["mean_ms"] - step["prefetch"]["mean_ms"], 3
        )
        report["saved_ms_p50"] = round(
            step["blocking"]["p50_ms"] - step["prefetch"]["p50_ms"], 3
        )
        report["staging_overhead_ms_blocking"] = round(
            step["blocking"]["mean_ms"] - step["floor"]["mean_ms"], 3
        )
        report["staging_overhead_ms_prefetch"] = round(
            step["prefetch"]["mean_ms"] - step["floor"]["mean_ms"], 3
        )
        print(json.dumps(report, indent=2))
        assert identical, "prefetch output differs from the blocking path"


if __name__ == "__main__":
    main()
