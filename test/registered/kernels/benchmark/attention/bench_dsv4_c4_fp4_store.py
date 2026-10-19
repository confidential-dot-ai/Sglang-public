from __future__ import annotations

import sys

import torch
import triton

from sglang.benchmark.bench_utils import run_bench
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    compress_c4_norm_rope_store_fp4,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.utils import is_sm100_supported, is_sm120_supported


HEAD_DIM = 128
PAGE_SIZE = 64
ROW_BYTES = 68


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 8, 32, 128, 512],
        x_log=True,
        line_arg="provider",
        line_vals=["control", "fused"],
        line_names=["C4 + norm/store", "Fused C4/norm/store"],
        styles=[("blue", "--"), ("green", "-")],
        ylabel="us",
        plot_name="dsv4-c4-fp4-store",
        args={},
    )
)
def benchmark(batch_size: int, provider: str):
    req_pool_indices = torch.arange(batch_size, device="cuda", dtype=torch.int64)
    seq_lens = torch.full((batch_size,), 8, device="cuda", dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(4, req_pool_indices, seq_lens)
    kv_score_input = torch.randn(
        batch_size, 4 * HEAD_DIM, device="cuda", dtype=torch.float32
    )
    ape = torch.randn(8, HEAD_DIM, device="cuda", dtype=torch.float32)
    norm_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    freqs_cis = precompute_freqs_cis(64, 9, 0, 10000, 1, 32, 1).cuda()
    out_loc = torch.arange(batch_size, device="cuda", dtype=torch.int64)
    num_cache_pages = triton.cdiv(batch_size, PAGE_SIZE)
    state = torch.randn(
        batch_size * 2,
        4,
        4 * HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    cache = torch.empty(
        num_cache_pages,
        PAGE_SIZE * ROW_BYTES,
        device="cuda",
        dtype=torch.uint8,
    )

    def control() -> None:
        compressed = compress_forward(
            state,
            kv_score_input,
            ape,
            plan,
            head_dim=HEAD_DIM,
            compress_ratio=4,
        )
        compress_norm_rope_store(
            compressed,
            plan,
            norm_weight=norm_weight,
            norm_eps=1.0e-6,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            kvcache=cache,
            page_size=PAGE_SIZE,
            use_fp4=True,
        )

    def fused() -> None:
        compress_c4_norm_rope_store_fp4(
            state,
            kv_score_input,
            ape,
            plan,
            norm_weight=norm_weight,
            norm_eps=1.0e-6,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            kvcache=cache,
            page_size=PAGE_SIZE,
        )

    if provider == "control":
        fn = control
    elif provider == "fused":
        fn = fused
    else:
        raise ValueError(provider)
    return tuple(value * 1000 for value in run_bench(fn, use_cuda_graph=False))


if __name__ == "__main__":
    if not (is_sm100_supported() or is_sm120_supported()):
        print("[skip] DSV4 fused C4 FP4 benchmark requires SM100/SM120 CUDA.")
        sys.exit(0)
    benchmark.run(print_data=True)
