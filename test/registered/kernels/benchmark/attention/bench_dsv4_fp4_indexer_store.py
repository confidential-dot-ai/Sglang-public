from __future__ import annotations

import sys

import torch
import triton

from sglang.benchmark.bench_utils import run_bench
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _store_fp4_index_k_cache_unfused,
    store_fp4_index_k_cache,
)
from sglang.srt.utils import is_sm100_supported, is_sm120_supported


HEAD_DIM = 128
PAGE_SIZE = 64
ROW_BYTES = 68


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[1, 8, 32, 128, 512, 4096],
        x_log=True,
        line_arg="provider",
        line_vals=["unfused", "fused"],
        line_names=["Quantize + scatter", "Fused quantize/scatter"],
        styles=[("blue", "--"), ("green", "-")],
        ylabel="us",
        plot_name="dsv4-fp4-indexer-store",
        args={},
    )
)
def benchmark(num_tokens: int, provider: str):
    num_pages = triton.cdiv(num_tokens, PAGE_SIZE)
    x = torch.randn(num_tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    loc = torch.randperm(num_pages * PAGE_SIZE, device="cuda")[:num_tokens].to(
        torch.int64
    )
    cache = torch.empty(
        num_pages, PAGE_SIZE * ROW_BYTES, device="cuda", dtype=torch.uint8
    )
    if provider == "fused":
        fn = lambda: store_fp4_index_k_cache(x, cache, loc, page_size=PAGE_SIZE)
    elif provider == "unfused":
        fn = lambda: _store_fp4_index_k_cache_unfused(
            x, cache, loc, page_size=PAGE_SIZE
        )
    else:
        raise ValueError(provider)
    return tuple(value * 1000 for value in run_bench(fn, use_cuda_graph=False))


if __name__ == "__main__":
    if not (is_sm100_supported() or is_sm120_supported()):
        print("[skip] DSV4 FP4 indexer-store benchmark requires SM100/SM120 CUDA.")
        sys.exit(0)
    benchmark.run(print_data=True)
