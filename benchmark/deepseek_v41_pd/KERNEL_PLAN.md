# DeepSeek V4.1 Flash custom-kernel plan

Status: proposed implementation plan, 2026-09-28. No serving configuration changes are part of this document.

Upstream audit update: evaluate DeepGEMM Mega-mHC, Mega Gate, existing MegaMoE,
and V4.1 mega-attention before writing replacements. The installed
`sgl-deep-gemm==0.1.7` exports MegaMoE and `tf32_hc_prenorm_gemm`, but no Mega-mHC
or Mega Gate API. Current V4.1 mHC uses `_hc_mix_stats_partial_kernel` and
`_hc_mix_reduce_sinkhorn_kernel`; sparse FlashMLA attention is integrated through
`sgl_kernel`, without a mega-attention call found in the inspected backend.
MegaMoE integration exists in this checkout but is not selected by the recorded
serving configuration (`moe_runner_backend=flashinfer_mxfp4`,
`moe_a2a_backend=none`). Test library/API upgrades in an isolated environment.
See [DeepGEMM's September release](https://github.com/deepseek-ai/DeepGEMM/pull/432)
and [vLLM's V4.1 mega-attention integration](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn/).

## Scope and measured baseline

Keep two GPUs for TP2/EP2 prefill and two for TP2/EP2 decode, CPU-resident Engram tables, CUDA IPC transfer, split CUDA graphs, and DSpark with five draft positions plus the target position. Optimize decode first because that is the workload measured here; validate prefill separately before enabling any shared replacement there.

Evidence: `/data/dsv41-bringup/profile-dspark-b64-20260928-143420/kernel-breakdown.{md,json,csv}` and its two rank traces. Forty speculative iterations at concurrency 64 average 63.5 ms between iterations and 41.257 ms of summed GPU kernel durations per GPU. An iteration can emit multiple tokens. Ranks execute concurrently; kernel durations can overlap and are not wall-clock fractions.

The unprofiled baseline in `/data/dsv41-bringup/tpm-dspark-b64-20260928-0428.summary.json` completed 256 requests with 1,024 input and 256 output tokens at concurrency 64: 2,687 output tokens/s, or 161,237 output TPM. Repeat a baseline under the exact same GPU/DevTools mode as each candidate; historical runs in different modes are not controlled comparisons.

| Kernel or family | ms / iteration / GPU | Notes |
| --- | ---: | --- |
| Target expert gate/up GEMM with SwiGLU | 11.157 | 40 calls |
| Expert down GEMM | 5.823 | 43 calls; target and draft |
| Draft alternate gate/up GEMM | 0.247 | 3 calls |
| NCCL BF16 all-reduce | 5.068 | 88 calls |
| mHC partial statistics/mixing | 3.105 | 86 calls |
| mHC reduction + Sinkhorn | 1.039 | 86 calls |
| mHC post-split and combine | 0.774 | 172 calls |
| Sparse MLA + combine | 3.217 | 43 calls each |
| Main block-scaled dense GEMM symbol | 2.582 | 230 calls |
| BF16 direct copies | 0.991 | 99 calls |
| 5,120-wide RMSNorm / MXFP8 quantization | 0.617 / 0.364 | 89 / 90 calls |

CPU Engram index-select separately consumes approximately 5.7 ms per iteration. GPU-recorded H2D transfers take approximately 0.86 ms. These overlap other activity; do not add them to GPU totals as an end-to-end budget.

## Why EP2 still has expensive expert multiplication

The target has 384 routed experts per layer, selects six per token, and has 40 layers. EP2 assigns 192 experts to each GPU. It does not reduce the model to two experts. With TP2/EP2, expert ownership and expert tensor partitioning must be distinguished; do not divide the expert intermediate dimension by two again.

At batch 64 with a six-position verification window, the target processes 384 token positions. This produces 2,304 token-expert assignments per layer, approximately 1,152 per GPU with balanced routing. Uniform routing would average only six rows per expert:

* gate/up: `(M_expert, 5120) @ (5120, 4608)`, then fused SwiGLU;
* down: `(M_expert, 2304) @ (2304, 5120)`.

One grouped launch runs many expert matrices. The 40 and 43 launch counts are consistent with 40 target layers plus three draft layers. Actual routing may be skewed, especially with similar benchmark prompts; six rows is an arithmetic average, not a measured histogram or proof that every expert is active.

If every local expert is active and its packed four-bit weights are read once, gate/up reads `192 * 5120 * 4608 / 2 * 40 = 90.60 GB` per GPU per iteration. Down adds 45.30 GB, excluding scales, padding, shared experts, and draft work. Gate/up's 90.60 GB divided by 11.157 ms is an equivalent 8.12 TB/s. NVIDIA advertises approximately 8 TB/s per B300 GPU. This is a useful bandwidth-scale sanity check, **not measured bandwidth**: actual active experts, caching, packing and repeated reads remain unknown. It argues against promising a large GEMM speedup before measuring memory traffic.

## Milestone 0: establish where a rewrite can win

1. Capture actual per-layer/rank expert row counts, active-expert counts, skew, activation shapes, strides, packed weight/scaling layouts, and MoE TP/EP settings. Capture target and draft separately at batches 1, 8, 16, 32, and 64. Include varied prompts as well as the original benchmark distribution.
2. Build a standalone replay harness with representative tensors and routing. Use spare GPUs after checking availability, without loading another full model. Match the model's weight working set; repeatedly timing one cache-resident expert is not representative.
3. Profile replay kernels with hardware counters where available: actual DRAM bytes/bandwidth, L2 hit rate, tensor-core utilization, occupancy and scheduling stalls. Inspect tile mapping before inferring wasted token rows from the kernel symbol.
4. Compare existing FlashInfer tactics and activation-precision options with identical numerical requirements. Determine whether the large GEMM cost is unavoidable weight traffic, redundant traffic, tile waste, load imbalance, or an implementation limitation.
5. Attribute GPU idle intervals and collectives to the critical path. A kernel's share of summed GPU time is not its share of request latency.

Deliverable: measured shapes, replay inputs, counter report and a go/no-go decision for the grouped GEMM prototype.

## Custom kernels, in proposed implementation order

### 1. mHC mixing, statistics and normalization

First prototype: specialize the partial mixing/statistics and reduction/Sinkhorn stages for the observed decode shapes. The immediate budget is 4.144 ms; the wider family is 4.918 ms.

Use CuTe DSL initially; ThunderKittens is an alternative for a compact CUDA implementation. Compare a row-owned implementation against fixed split-K with a legal reduction stage. Reduce temporary writes and repeated input reads; only fuse post/combine stages where ownership and dependencies permit it. A single large CTA is an experiment, not an assumed win.

Integration: `python/sglang/kernels/ops/layernorm/mhc.py`, particularly `_hc_mix_stats_partial_kernel`, `_hc_mix_reduce_sinkhorn_kernel`, and `hc_mix_stats_sinkhorn`; also `mhc_post_split_h.py`.

The current implementation deliberately maintains FP32 arithmetic and fixed reductions for batch invariance. Preserve this contract and Sinkhorn behavior. Do not silently substitute BF16 tensor-core arithmetic. Run `test/registered/kernels/ops/layernorm/test_hc_mix_stats.py` and add candidate-specific edge cases only where necessary. Aim for at least a 20% family improvement in replay before integrating; this is a selection threshold, not a forecast.

### 2. Two-rank reduction fused with its consumer

The measured all-reduce costs 5.068 ms across 88 calls. First qualify the existing optimized collective paths: the current deployment disables custom all-reduce and FlashInfer all-reduce fusion. Test compatibility under the actual confidential-computing and CUDA IPC configuration before writing another collective.

If existing paths cannot deliver the needed fusion, prototype a two-GPU IPC/NVLink reduction fused with a genuinely adjacent residual or normalization operation. ThunderKittens is a reasonable implementation candidate; CuTe DSL is also possible. Measure actual payload sizes and rank arrival skew first.

CUDA IPC shares memory handles, not completion or synchronization. Require correct stream ordering, cross-device synchronization, buffer lifetime and graph replay. Keep prefill and decode TP groups separate. Do not combine reductions across dependent layers or change attestation/CC modes for this optimization.

### 3. Small-row grouped MXFP4 expert GEMM, conditional on milestone 0

This has the largest measured budget: approximately 17.227 ms across target and draft. Promote it ahead of other prototypes if the counter report shows substantial recoverable headroom. Defer a rewrite if useful weight reads already saturate HBM.

Prefer CuTe DSL's block-scaled/grouped GEMM building blocks. Prototype persistent scheduling over nonempty experts, routing-aware row buckets, efficient tails and weight/scale loads. Preserve existing fused activation and quantization semantics; gate/up already fuses SwiGLU. Test target and draft routing independently. A different tile or scheduler cannot eliminate required expert weights.

Actual integration point: `python/sglang/srt/layers/quantization/mxfp4.py`, its FlashInfer MXFP4 preparation and SM100-family execution path. Confirm the runtime call and representation before selecting a replacement. The repository's `moe_runner/flashinfer_cutedsl.py` contains an NVFP4 path with 16-element scaling blocks; it is not automatically a drop-in replacement for MXFP4 with 32-element scaling blocks. Preserve weight swizzles, scale encodings, accumulation, clipping and routing weights.

### 4. Normalization, quantization and avoidable materialization

Inspect RMSNorm (0.617 ms), MXFP8 quantization (0.364 ms), BF16 copies (0.991 ms), and the many small dense projections. Fuse only adjacent operations sharing compatible tensor ownership and numerical semantics. CuTe DSL epilogues or a ThunderKittens fused kernel are candidates.

Attribute each copy before removing it: some buffers are required to keep graph bridge addresses stable. These totals are an upper scope of investigation, not an additive savings prediction. Prioritize launch and memory-traffic reductions rather than rewriting every dense GEMM.

## Companion host work and deferred work

CPU Engram gathers remain outside the CUDA graphs and CPU-resident as requested. Investigate preallocated gather outputs, staging allocation overhead, combined payloads, and pinned/asynchronous double buffering if supported by the current CC setup. Prefetch only once the correct n-gram IDs exist; speculative future IDs cannot be assumed (`SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH` does this from the hash output; the remaining exposed cost is the two host-to-device copies). Preserve stable GPU bridge pointers and avoid duplicating the roughly 189 GiB tables. This work is not solved by a GPU kernel rewrite.

Keep existing sparse MLA kernels initially. Their approximately 3.217 ms combined cost is smaller than MoE and the proposed mHC/collective work; revisit only after the above changes shift the profile.

## Validation and promotion

Each candidate gets an independent opt-in flag and the existing implementation as fallback. Validate empty experts, one-row experts, skewed routing, tails, padding, actual quantization layouts, and representative intermediate/output comparisons. Preserve mHC batch invariance. Check greedy output behavior and DSpark acceptance, not just synthetic tensor error.

Test CUDA graph capture/replay with changing input values and stable buffers, two-rank correctness, IPC synchronization stress, and memory usage. Exercise both verification and draft paths. Shared code must pass prefill tests before enabling it on prefill.

Measure unprofiled end-to-end throughput and latency with the same GPU mode, serving settings and workload. Repeat the original 256-request 1,024/256-token benchmark at concurrency 16, 32 and 64, at least three runs per candidate, then a longer varied-prompt run. Record output TPM, TTFT, TPOT, p95 latency, acceptance rate and failures. Promote only for a reproducible improvement outside noise without correctness or relevant latency regressions; a 5% end-to-end gain is a useful initial target, not a guaranteed result.

For scale, a 20% improvement in the 11.157 ms gate/up kernel saves only 2.23 ms of kernel duration per iteration. Even if entirely on the critical path, that is about 3.5% of the measured 63.5 ms interval. Re-profile after each accepted change rather than adding independent savings estimates.

## Upstream references

* [NVIDIA HGX AI factory components and B300 aggregate bandwidth](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
* [CUTLASS and CuTe DSL, including block-scaled Blackwell support](https://github.com/NVIDIA/cutlass)
* [CuTe DSL Blackwell grouped GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py)
* [ThunderKittens: Blackwell and multi-GPU primitives](https://github.com/HazyResearch/ThunderKittens)

Check the installed toolkit/package revision before adopting upstream examples. ThunderKittens NVFP4 support does not by itself establish compatibility with this checkpoint's MXFP4 representation.
