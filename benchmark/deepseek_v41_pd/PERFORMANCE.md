# DeepSeek V4.1 Flash performance report

Evidence snapshot: 2026-09-28. Branch: `dsv4.1`, based on `a5b2e2b6a1` with local deployment fixes. The original sections consolidate saved synthetic measurements. A subsequent oracle real-mix replay uses a separately recorded longer-context deployment; see [the replay report](REALMIX_BENCHMARK.md). Configuration in the original deployment table applies to the synthetic runs.

## What the measurements establish

The latest unprofiled synthetic run completed 256/256 requests at concurrency 64 and delivered **161,237 output tokens/minute** (2,687 output tokens/s). Mean time to first token was **2.042 s**, mean time per output token **12.811 ms**, and mean request latency **5.309 s**.

In the separate CPU+GPU profile, expert matrix multiplication is the largest GPU category: **17.23 ms per speculative iteration per GPU**, about 42% of summed kernel durations. Decode collectives, mHC and dense projections are also material. CPU n-gram row gathers take approximately **5.7 ms per iteration per rank**, while GPU-recorded host-to-device copies take approximately **0.86 ms**.

**Prefill-to-decode handoff overhead has not been isolated.** Earlier conversation suggesting that a large amount of time was measured in this handoff was an overstatement. The measured NCCL collectives are within the decode pair. Neither those collectives, CPU n-gram staging, nor unattributed GPU gaps establishes a PD handoff bottleneck.

These results demonstrate working inference and a useful optimization baseline. They do not establish peak capacity, broad model quality, a controlled DSpark speedup, or a fully optimized deployment.

## Oracle real-mix replay: completed four-level baseline

The user-supplied modeled multi-turn dataset was replayed with the recovered M3
harness, 60s warmup and 300s scored windows, on four GPUs with DSpark and six-token
oracle commits. Full methodology, raw artifacts and comparison caveats are in
[the real-mix report](REALMIX_BENCHMARK.md).

| Concurrency | Served TPM/GPU | Cache hit | Median TTFT | Median TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 261,356 | 56.4% | 54.45s | 4.13ms |
| 54 | 276,835 | 58.6% | 68.27s | 4.18ms |
| 67 | 284,863 | 59.6% | 82.13s | 4.61ms |
| 80 | 291,069 | 61.2% | 91.32s | 4.66ms |

All 887 scored requests succeeded, but all levels failed the harness's cache-hit
and interactive-latency targets. Aggregate uncached prefill rate remained roughly
7.5K tokens/sec; decode generally had only one or two active requests. This workload
exposes a prefill/cache bottleneck that the short synthetic decode profile did not.

Confirmed configuration findings: prefill's default SWA reservation covers 32
prefix tails for 40–80 active conversations, and its graph context bound remains
16K despite the benchmark's 128K serving limit. Long contexts therefore use eager
prefill with late-layer trimming disabled. Prefix-tail eviction as a root cause
needs a controlled A/B, and expanding static graph bounds is not automatically a
speedup. Prioritize these configuration experiments before assuming new decode
kernels will close the real-mix gap. PD copy overhead remains unquantified.

## Deployment behind the results

| Component | Measured configuration |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, local `/data/DeepSeek-V4.1-Flash` |
| Checkpoint revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Hardware | Eight B300 SXM6 GPUs; four used by this PD pair |
| Prefill | GPUs 0–1, TP2 + EP2, maximum eight running requests |
| Decode | GPUs 2–3, TP2 + EP2, maximum 64 running requests |
| KV capacity | Configured cap 131,072 tokens per worker; context limit 16,384 |
| Quantization | MXFP8 dense path, MXFP4 expert weights, FP8 KV cache |
| Speculation | Bundled DSpark draft model enabled; five proposals and six verification positions |
| Graphs | Split/breakable CUDA graphs with eager CPU Engram staging |
| Prefill graph buckets | 256, 512, 1,024, 2,048, 4,096 tokens |
| Decode graph buckets | 1, 2, 4, 8, 16, 32, 64 requests |
| Engram | Two CPU-resident tables, immutable checkpoint mappings sharing locked page-cache pages |
| PD transport | Mooncake intra-node CUDA IPC/NVLink path |
| Collectives | Custom all-reduce and FlashInfer all-reduce fusion disabled in this deployment |
| Security/profiling mode | CC ON, NVLE; DevTools ON for current DSpark results and GPU profile |
| Driver | 595.91.07 |
| Serving endpoint | `http://127.0.0.1:8030/v1`, model `dsv41` |
| Deployment logs | `/data/dsv41-bringup/pd-dspark-03/` |

The checkpoint's Engram layers are **1 and 14, zero-based**. The target has **40 layers**. Layer 20 is the last shared-KV source, not the end of the model. The early conversational description of layers 0/14/20 should not be treated as the actual model layout.

Prefill graph capture currently requires disabling late-layer SWA token trimming on prefill. Decode retains trimming. Short prefill shapes can use the eager fallback when the smallest graph bucket introduces excessive padding. Consequently, graph enablement alone does not establish a speedup for every prompt length.

## Throughput and latency history

All rows in the main table use 256 requests, 1,024 input tokens and exactly 256 output tokens per request. Each run emits 65,536 output tokens. Prefix caches were flushed before a one-request warmup. Concurrency is the client limit, not proof that every server iteration had that batch size.

| Run | Concurrency | DSpark | Output tokens/s | Output TPM | Mean TTFT | Mean TPOT | Mean request latency | Run duration |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Target-only sweep | 16 | Off | 737.09 | 44,226 | 453.6 ms | 19.664 ms | 5.468 s | 88.912 s |
| Target-only sweep | 32 | Off | 1,206.72 | 72,403 | 655.8 ms | 23.550 ms | 6.661 s | 54.309 s |
| Target-only sweep | 64 | Off | 1,734.07 | 104,044 | 1,188.6 ms | 30.465 ms | 8.957 s | 37.793 s |
| Initial DSpark qualification | 64 | On | 2,338.17 | 140,290 | 2,244.6 ms | 14.545 ms | 5.953 s | 28.029 s |
| Latest DSpark rerun | 64 | On | 2,687.28 | 161,237 | 2,042.3 ms | 12.811 ms | 5.309 s | 24.388 s |

Target-only p95 request latencies were 6.056 s, 8.005 s and 12.794 s at concurrency 16, 32 and 64 respectively. The two DSpark summary files do not contain p95 latency, so this report does not invent it.

Within the target-only sweep, concurrency 64 achieved 2.35 times concurrency-16 throughput, at the cost of higher TTFT and per-request latency. This is a measured throughput/latency tradeoff.

The DSpark results are numerically faster than the target-only concurrency-64 result, but the earlier run used DevTools OFF and the DSpark deployment used DevTools ON. They are **not a controlled speculation-only comparison**. The two DSpark runs also differ enough to require repeated measurements before treating either as a stable expected production rate. No change-specific speedup is inferred from their difference.

TPM here means **output** tokens per minute, calculated as total output tokens divided by elapsed run time times 60. It is not input-plus-output TPM. TTFT includes work and waiting before the first token. TPOT is a request-level output pacing metric, not one GPU iteration's duration.

### Earlier graph and eager observations

An initial target-only batch-16 run with 128 requests recorded 43,281 output TPM. It differs in request count from the table above.

Split-graph smoke samples at 439 and 919 prompt tokens took 0.54 s and 0.21 s, compared with earlier eager samples of 3.32 s and 1.42 s. These were single-request sanity timings affected by cache, warmup and the prefill trimming change. They are not a controlled CUDA-graph throughput comparison. Responses were checked at 439, 919, 1,731 and 3,441 prompt tokens, including concurrency and cached prefixes; arithmetic formatting differed between some eager and graph outputs, so bitwise equivalence was not established.

## Comparison with the supplied MiniMax M3 report

The user supplied an M3 NVFP4 qualification report for eight B300s, two TP4/EP4
replicas, replayed multi-turn agent traffic, approximately 38K input tokens per
request, and roughly 95% prefix-cache hits. Measurements used 300-second windows
after 60 seconds of warmup. This is a different workload and measurement protocol
from the DeepSeek random 1,024-input/256-output-token synthetic test.

### Token accounting explains the headline discrepancy

M3's reported 2.18–4.00 million TPM/GPU must not be compared directly with
DeepSeek's 161,237 aggregate output TPM. The recovered M3 replay driver defines served TPM as **prompt + completion
tokens, including cached prompt tokens**. The exact invocation behind the supplied
M3 report is unavailable, but its arithmetic agrees with this input-dominated
accounting:

```text
M3 at concurrency 40:
461 requests/min × approximately 38,000 input tokens/request ÷ 8 GPUs
  = 2,189,750 input TPM/GPU
Reported: 2,178,260 TPM/GPU

Implied counted tokens/request:
2,178,260 × 8 ÷ 461 = 37,801
```

At the other three primary levels, reported TPM and RPM imply approximately
38,513, 39,814 and 38,453 counted tokens per request, again close to the reported
prompt length. The recovered driver confirms the input-plus-output formula and credits cached
input equally. Raw M3 counters are still unavailable for an exact reconstruction
of that report.

| Metric | M3 supplied report | DeepSeek latest synthetic run |
| --- | --- | --- |
| Allocated GPUs | 8 | 4, including prefill and decode |
| Topology | Two TP4/EP4 colocated replicas | One TP2/EP2 prefill + TP2/EP2 decode pair |
| Prompt length | Approximately 38K tokens, multi-turn replay | 1,024 tokens, random workload |
| Output length | Not specified in supplied throughput table | 256 tokens/request |
| Prefix reuse | Reported hit ratio 0.946–0.952 | Flushed before warmup; subsequent hit ratio not recorded |
| Headline TPM/GPU | 2.18–4.00M served, consistent with the recovered harness | 40,309 output TPM per allocated GPU |
| Input + output TPM/GPU | Prompt + completion in the recovered M3 harness | 201,546, counting all requested input tokens |
| Output TPM/GPU | Cannot recover from the supplied summary | 40,309 |
| Request-level decode speed | P50 66.6 TPS at concurrency 40; 44.6 at 67 | Mean TPOT 12.811 ms at concurrency 64 |
| TTFT | P50 0.82 s at concurrency 40; 0.87 s at 67 | Mean 2.042 s at concurrency 64 |

The reciprocal of DeepSeek's mean TPOT is approximately 78.1 tokens/s. This is
neither median request TPS nor aggregate throughput, so it is not a matched win
over M3's median TPS. It does show that the available output-pacing measurements
do not establish an orders-of-magnitude decoding gap.

If M3's hit ratio is token-weighted, a 38K prompt at 95% hits requires only about
1,900 uncached prompt tokens, while an input-inclusive throughput metric can
still credit all 38K. Reusing cached input is valuable service throughput, but
does not mean generating millions of new tokens per minute per GPU. The supplied
report describes real traffic replay and does not establish oracle acceptance.

### What remains slow or inefficient in DeepSeek

The measured DeepSeek costs remain real: expert GEMMs, decode collectives, mHC,
CPU Engram gathering and unattributed timeline gaps. New upstream megakernels
remain unqualified for this deployment. These establish optimization targets;
they do not quantify an architectural disadvantage relative to M3 without an M3
profile under matched conditions. PD handoff is still unquantified.

As an illustrative oracle calculation only, hold batch 64 and the profiled
63.5 ms iteration cadence constant and commit all six positions every iteration:
`64 * 6 / 0.0635 * 60 = 362,835` output TPM for the pair, or 90,709 per allocated
GPU. This is not a benchmark, a production ceiling, or a predicted improvement:
profiling overhead, sequence growth, batching and oracle-mode execution change
the cadence. It illustrates why controlling acceptance alone cannot make this
output-only number comparable to M3's input-dominated millions of TPM.

The [oracle real-mix replay](REALMIX_BENCHMARK.md) now reports cached input,
uncached input and output separately, with GPU-normalized rates and latency
statistics. It uses the user-supplied modeled dataset and six-token oracle commits;
it is not the 800-conversation trace described by the M3 report. A controlled
real-acceptance comparison and matched eight-GPU topology remain outstanding.
Do not infer aggregate output throughput by multiplying median TPS by concurrency.

## CPU+GPU profiling methodology

Profile directory: `/data/dsv41-bringup/profile-dspark-b64-20260928-143420/`.

The profile captured both decode TP/EP ranks for **40 speculative iterations**, under 64 concurrent requests with 1,024 generated tokens per request to sustain the capture window. All 64 requests completed; router readiness reported two healthy workers afterward. This workload is separate from the 256-output-token throughput test.

| Measurement | TP0 / GPU2 | TP1 / GPU3 |
| --- | ---: | ---: |
| Mean speculative iteration interval | 63.498 ms | 63.492 ms |
| Median iteration interval | 63.508 ms | 63.550 ms |
| GPU trace span | 2,542.975 ms | 2,543.971 ms |
| Union of GPU activity intervals | 1,586.479 ms | 1,573.595 ms |
| Sum of kernel durations / iteration | 41.420 ms | 41.093 ms |

GPU activity occupies approximately **62% of the observed span**, counting the union of kernels, memcpy and memset intervals. This is timeline coverage, not SM occupancy or tensor-core utilization. The remaining gaps require attribution; they cannot all be assigned to CPU staging, scheduling or PD handoff without further evidence.

One speculative iteration contains target verification and draft work and can emit multiple accepted tokens. **63.5 ms is not per-output-token latency.** Instrumentation adds overhead; use unprofiled runs for TPM comparisons.

Kernel sums can overlap across streams. CPU ranges overlap GPU work. The two ranks run concurrently. Do not add rank durations together or add CPU and GPU categories to construct request latency. Likewise, subtracting the kernel sum from the iteration interval does not yield a clean CPU-overhead measurement.

## GPU costs by family

Values below are averages across the two ranks, per speculative iteration per GPU. Categories follow the saved trace summarizer.

| Family | Summed kernel duration | Approximate share of all kernel durations |
| --- | ---: | ---: |
| Expert matrix multiplication | 17.227 ms | 41.76% |
| NCCL collectives | 5.595 ms | 13.56% |
| mHC | 4.919 ms | 11.92% |
| Dense matrix multiplication | 4.082 ms | 9.89% |
| Sparse attention and combine | 3.217 ms | 7.80% |
| Other kernels | 6.217 ms | 15.07% |
| Total | 41.257 ms | 100% |

The expert family ranges from 16.973 ms on TP0 to 17.482 ms on TP1; collectives range from 6.006 ms to 5.184 ms respectively. These differences warrant checking work balance and rank arrival times, but they do not by themselves establish routing imbalance or its cause.

## Individual kernel breakdown

The following table preserves distinct kernel symbols while abbreviating long template names. Exact symbols, per-rank durations and all 152 entries are available in the companion [complete kernel breakdown](PERFORMANCE_KERNELS.md).

| # | Kernel (short name) | Calls/iteration/GPU | Mean µs/call | ms/iteration/GPU | Kernel time % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `FP4 expert GEMM + SwiGLU: bmm_MxE4m3…t128x16x256u2` | 40 | 278.92 | 11.157 | 27.04% |
| 2 | `FP4 expert GEMM: bmm_Bfloat16…t128x16x256` | 43 | 135.42 | 5.823 | 14.11% |
| 3 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 88 | 57.59 | 5.068 | 12.28% |
| 4 | `_hc_mix_stats_partial_kernel` | 86 | 36.11 | 3.105 | 7.53% |
| 5 | `flash_fwd_splitkv_mla_fp8_sparse_kernel` | 43 | 66.14 | 2.844 | 6.89% |
| 6 | `Sm100BlockScaledPersistentDenseGemmKernel` | 230 | 11.23 | 2.582 | 6.26% |
| 7 | `_hc_mix_reduce_sinkhorn_kernel` | 86 | 12.09 | 1.039 | 2.52% |
| 8 | `at::native::direct_copy_kernel_cuda (BF16, vectorized)` | 99 | 10.01 | 0.991 | 2.40% |
| 9 | `nvjet_sm103_tst_128x96_64x9_2x2_2cta_h_bz_TNT` | 40 | 18.06 | 0.722 | 1.75% |
| 10 | `RMSNormKernel (5120-wide specialization)` | 89 | 6.93 | 0.617 | 1.50% |
| 11 | `_mhc_post_split_h_kernel` | 86 | 6.29 | 0.541 | 1.31% |
| 12 | `moe::dev::finalize::finalizeKernelVecLoad` | 40 | 12.15 | 0.486 | 1.18% |
| 13 | `ncclDevKernel_AllGather_RING_LL` | 6 | 77.61 | 0.466 | 1.13% |
| 14 | `flash_fwd_mla_combine_kernel` | 43 | 8.67 | 0.373 | 0.90% |
| 15 | `MXFP8QuantizeSwizzledKernel (5120-wide specialization)` | 90 | 4.04 | 0.364 | 0.88% |
| 16 | `nvjet_sm103_tss_64x32_64x16_2x1_2cta_v_bz_splitK_TNT` | 40 | 6.32 | 0.253 | 0.61% |
| 17 | `FP4 expert GEMM + SwiGLU: bmm_MxE4m3…t128x16x256 (no u2)` | 3 | 82.45 | 0.247 | 0.60% |
| 18 | `_router_triton_kernel` | 43 | 5.46 | 0.235 | 0.57% |
| 19 | `_hc_combine_kernel` | 86 | 2.71 | 0.233 | 0.56% |
| 20 | `get_mla_metadata_kernel` | 4 | 56.21 | 0.225 | 0.54% |

The top 20 account for 90.58% of summed kernel durations. The remaining 132 symbols account for approximately 3.884 ms per iteration per GPU. These percentages are kernel-time shares, not end-to-end latency shares.

## Why expert multiplication is expensive despite EP2

The target has 384 routed experts per layer, selects six experts per token, and has 40 layers. EP2 assigns 192 experts to each GPU. It does not reduce the model to two experts. Expert parallelism splits expert ownership; expert tensor parallelism is a separate partitioning dimension.

At batch 64, six verification positions produce 384 target token positions. Top-6 routing creates 2,304 token-expert assignments per layer, or about 1,152 per GPU under balanced routing. Uniform routing would average just **six token rows per expert**. Actual active-expert counts and skew remain unmeasured; similar prompts can produce a different distribution.

Each expert computes gate/up with logical shape `(M_expert, 5120) @ (5120, 4608)`, followed by fused SwiGLU, and down with `(M_expert, 2304) @ (2304, 5120)`. This combines small row counts with large weight matrices. One grouped launch handles many experts. The observed 40 gate/up calls and 43 down calls are consistent with 40 target layers and three draft layers; symbol counts alone are not a substitute for explicit layer correlation.

### Conditional bandwidth sanity check

If all 192 local experts are active and each packed four-bit weight is read once, the target's gate/up weights require:

```text
192 experts × 5120 × 4608 × 0.5 bytes × 40 layers = 90.60 GB/GPU/iteration
```

Down weights add 45.30 GB, for 135.90 GB total, excluding scales, padding, shared experts and draft work. Dividing 90.60 GB by the measured gate/up duration of 11.157 ms gives an equivalent 8.12 TB/s. That is on the scale of the approximately 8 TB/s per GPU implied by NVIDIA's 64 TB/s aggregate HGX B300 specification. [NVIDIA hardware reference](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)

**This is not measured DRAM bandwidth or proof of saturation.** Active experts, real packed bytes, cache reuse, repeated reads and actual tile efficiency must be measured. It explains why a large GEMM duration can be reasonable even at EP2, and why a rewrite should be gated on demonstrated headroom. Tile dimensions in a kernel name alone do not establish token-row padding waste.

## CPU n-gram staging and copy costs

The CPU tables occupy 202,758,032,400 bytes of embedding payload, approximately 188.833 GiB. Immutable mappings share page-cache pages across workers. The residency helper locks 203,073,077,576 bytes including shard metadata and other small tensors. Two private full-table copies would consume roughly 377.666 GiB, exceeding the machine's approximately 376.709 GiB host RAM before other overhead.

At each eager Engram break, indices reach CPU, rows/scales are gathered, compact buffers are copied to GPU, and GPU dequantization/model computation resumes. The actual GPU activity timings distinguish transfers from CPU waits:

| Measurement | TP0 per iteration | TP1 per iteration | Interpretation |
| --- | ---: | ---: | --- |
| CPU row gathers | 5.768 ms | 5.688 ms | Four recorded `index_select` calls per iteration |
| GPU-recorded H2D copies | 0.876 ms | 0.850 ms | Four copies; 4,866,048 bytes per iteration per rank |
| GPU-recorded D2H copies | 0.0142 ms | 0.0145 ms | Two copies; 147,456 bytes per iteration per rank |
| CPU output-copy range | 21.562 ms | 21.558 ms | Includes waiting for preceding GPU work; not pure transfer time |

The CPU gathers are material. Measured H2D copy duration is much smaller. Calling the entire 21.6 ms CPU output-copy range a memory-copy bottleneck would be incorrect. These staging operations are also distinct from the prefill-to-decode KV transfer.

Potential improvements include preallocated gather outputs, allocation/threading tuning, combining payloads and supported asynchronous staging. Any pipeline must respect when n-gram IDs become available and preserve stable CUDA-graph bridge buffers. Earlier capability probes found host registration/pageable access unsupported in this CC configuration; pinned-memory techniques need capability validation rather than assumption. Keep both tables on CPU as requested.

### Historical CPU-only profiles

Before DevTools-enabled GPU profiling, CUPTI returned `CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED`. Those profiles could not attribute GPU kernels or communication.

| Target-only profile | Iteration interval | CPU gather | Blocking index-copy range | Selected-row copy range | Output-copy range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 16, 40 steps | 20.65 ms | 0.39–0.41 ms | 3.45–3.47 ms | 0.41–0.53 ms | 5.50–5.64 ms |
| Batch 64, ten selected full-batch steps | 30.3 ms | 1.13–1.24 ms | 5.48–5.84 ms | 0.95–0.97 ms | 9.71–10.09 ms |

Values are per step/rank. CPU copy ranges include synchronization waits. These non-speculative steps differ from six-position speculative iterations, and their profiling mode differs too; comparing their raw durations does not isolate DSpark cost.

## PD disaggregation: costs and unresolved overhead

The expected PD-specific costs are routing/bootstrap, destination allocation/admission, transferring prompt KV and auxiliary state, completion detection and scheduling the first decode work. Chunk transfers may overlap prefill, so exposed handoff delay can be smaller than total transfer duration. Ordinary generation hands the request from prefill to decode at the boundary; it does not send the request back to prefill for every generated token. Retraction/rebootstrap is a separate case to measure if it occurs.

The deployed transport uses CUDA IPC to access GPU memory and the intra-node NVLink path to move it. An `rdma` environment label by itself does not establish that the active transfer path is RDMA.

The saved configuration has `enable_metrics=false` and `enable_request_time_stats_logging=false`. The decode-only GPU trace does not isolate the complete handoff. No numeric PD-overhead percentage or latency is supported by the current evidence.

| Observation | What it does not prove |
| --- | --- |
| 5.068 ms NCCL all-reduce per iteration | PD transfer cost; this communication is within decode TP2 |
| 5.7 ms CPU Engram gathers | PD handoff overhead |
| 2.042 s mean TTFT | Pure transfer latency; it includes other work and waiting |
| Gaps in the decode GPU timeline | That prefill-to-decode handoff is responsible |
| Exact cross-process IPC payload probe passes | Sustained model-scale transfer bandwidth or end-to-end PD overhead |

To quantify PD overhead, correlate request IDs across router and both workers; timestamp prefill admission/execution, every transfer submission/completion, destination resource waits and first decode execution. Record actual KV/auxiliary bytes and GPU-copy durations. The current generic transfer-metric fallback measures the last transfer queue interval when backend-specific latency is absent; it must not be treated as a measurement of every chunk's transfer time.

Sweep 1K/4K/8K prompt lengths and concurrency 1/16/32/64 with cold/warm prefixes and short/long outputs. Report p50/p95 exposed handoff, total transfer time, TTFT, TPOT and throughput. Compare colocated and PD serving with matched kernels, speculation and GPU mode, including an equal-total-GPU throughput comparison. Inspect capacity imbalance between the prefill cap of eight and decode cap of 64 separately from intrinsic transfer overhead.

## Startup, graph memory and correctness evidence

Startup costs are separate from warmed throughput. The initial split-graph run recorded approximately 23.2 s prefill capture with 3.25 GiB additional HBM per GPU, and 19.6 s decode capture with 0.35 GiB. In the DSpark deployment, target verification capture took 13.83–14.02 s and 1.72 GiB per rank; draft verification capture took 112.41–112.53 s including first-time tuning and 0.25 GiB per rank. These come from different capture stages/runs and should not be summed into a universal startup estimate.

The initial TP2 target fit with 65,536-token KV pools and 121.69 GiB free per GPU. That is historical capacity evidence, not current headroom for the larger DSpark deployment. Peak loader memory and complete target/draft/vision capacity remain qualification tasks.

Existing validation includes exact CPU/GPU gather probes, graph replay with changed IDs, structured logits/hidden-state replay, graph-dispatch regression tests, exact cross-process IPC payload comparison, routed arithmetic/chat smoke checks, and completed synthetic workloads. Ten model graph regression tests and nineteen launcher tests passed at DSpark bring-up. These do not substitute for broad model-quality evaluation, target-reference comparisons or a long serving soak. The initial DSpark validation logs showed accepted draft tokens and no retractions; acceptance-rate distributions are not available in the cited summaries.

## Optimization priorities and decision gates

| Work | Measured motivation | Proposed action |
| --- | --- | --- |
| Observability and replay | PD cost and GEMM headroom unknown | Capture routing/shapes, GPU memory counters, rank timing and request handoff stages |
| mHC | 4.144 ms mixing/statistics/Sinkhorn; 4.919 ms whole family | First custom prototype in CuTe DSL; preserve FP32, fixed reductions and batch invariance |
| Decode collectives | 5.068 ms all-reduce across 88 calls | Qualify existing fast paths first, then consider two-rank reduction fused with its consumer |
| Grouped expert GEMM | 17.227 ms, largest GPU family | Promote a CuTe DSL small-row MXFP4 prototype if counters demonstrate recoverable headroom |
| Materialization and fusion | 0.991 ms BF16 copies, 0.617 ms RMSNorm, 0.364 ms MXFP8 quantization | Attribute callers and fuse eligible neighbors without breaking graph buffers |
| CPU Engram | Approximately 5.7 ms gather | Tune gathering/staging while preserving shared CPU residency |
| Prefill | Graphs currently sacrifice late-layer trimming | Measure graph/trimming tradeoff and phase capacity balance |
| Sparse MLA | Approximately 3.217 ms including combine | Revisit after higher-priority work changes the profile |

CuTe DSL is the proposed starting point for block-scaled grouped GEMM. ThunderKittens is an alternative for compact fused kernels and two-GPU communication. Neither guarantees a speedup. The repository's existing NVFP4 CuTe path is not automatically compatible with MXFP4 scale encoding and block layout. Gate/up already fuses SwiGLU; that fusion is not a new optimization opportunity. See [CUTLASS](https://github.com/NVIDIA/cutlass) and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens).

Every candidate needs an independent switch/fallback, representative numerical validation, CUDA-graph replay, target/draft coverage, two-rank correctness and unprofiled end-to-end measurement. Repeat benchmark conditions at least three times and add varied prompts and a longer soak. Track acceptance, failures, memory and latency as well as TPM.

For scale, reducing the 11.157 ms gate/up kernel by 20% removes 2.23 ms of kernel duration. Even if entirely on the critical path, that is only about 3.5% of the observed 63.5 ms iteration interval. Kernel-time percentages should not be advertised as potential end-to-end speedups.

Detailed design: [KERNEL_PLAN.md](KERNEL_PLAN.md).

## Evidence index and remaining limitations

All paths below are local to this machine. The companion kernel report preserves the full symbol breakdown in the repository; raw traces remain under `/data`.

| Artifact under `/data/dsv41-bringup/` | Evidence |
| --- | --- |
| `tpm-dspark-b64-20260928-0428.summary.json` | Latest short synthetic unprofiled throughput/latency |
| `realmix/c{40,54,67,80}.json` | Oracle real-mix fixed-window and drain results, server configuration and provenance |
| `dspark-benchmark-summary.json` | Initial DSpark qualification |
| `batch-sweep-summary.json` | Target-only concurrency sweep, including p95 request latency |
| `profile-dspark-b64-20260928-143420/summary.json` | Per-rank profile groups, CPU ranges, GPU copies and activity coverage |
| `profile-dspark-b64-20260928-143420/kernel-breakdown.{md,csv,json}` | All 152 exact kernel symbols and timings |
| `profile-dspark-b64-20260928-143420/*.trace.json.gz` | Raw decode CPU+GPU traces |
| `profile-dspark-b64-20260928-143420/{profile_decode.py,summarize_trace.py,kernel_breakdown.py}` | Capture and analysis scripts |
| `pd-dspark-03/` | Launch configuration, server-info, worker logs and smoke results |
| `profile-b16-summary.json`, `profile-b64-summary.json` | Historical target-only CPU profiles |
| `split-baseline.json`, `split-results.json` | Earlier eager/graph smoke timings |
| `probe-transfer.log`, `engram-cache.log` | IPC correctness and CPU residency evidence |

Outstanding measurements include controlled DSpark on/off and eager/graph comparisons, acceptance distributions, isolated PD overhead, sustained link bandwidth, actual expert histograms and DRAM counters, full prefill profiling, stable repeated-run statistics, representative workload latency tails, quality qualification and long-duration reliability. No conclusions about these are implied by the short synthetic runs.
