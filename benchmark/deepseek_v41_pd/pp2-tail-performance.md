# CED PP2 tail-replay performance experiments

All runs use GPUs 4–7 and ports 8040–8044 / 29001. Replica A is untouched.
Artifacts are in `/data/dsv41-bringup/ced/pp2-<variant>/`.

The instrumented control uses `SGLANG_DSV41_CED_TAIL_TIMING=1` on decode.
Timing emits CPU preparation time, forward-call wall time, and deferred CUDA
start/end event time without synchronizing. CUDA elapsed time includes GPU idle
periods between host launches; it is not summed kernel time and cannot be
subtracted from wall time to obtain exclusive CPU time. Rank-zero samples only
are summarized. The event interval reported below comes from the replay
client's streaming event ITL, consistent with the inherited scorecard.

| Variant | Single-request tail median incl. preparation | c56 event interval p50 | c56 TPM/GPU | TTFT p95 | User tok/s p50 / slowest 5% | Correctness |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Instrumented control | 482 ms | 203 ms | 1.543M | 2.46 s | 29.4 / 25.8 | All five probes exact against PP2; oracle before/after load passes |
| Pageable staging only | 485 ms | 213 ms | 1.479M | 2.57 s | 28.1 / 24.1 | Four exact probes; known 20K continuation drift; oracle before/after load passes |

Control c56: 699 completed, zero failed; cache hit 95.59%. Natural tail batch
sizes 1/2/3/4 have median wall costs 482/245/173/129 ms per request. These
include startup, probes, warmup, measurement, and drain; they are diagnostic
samples, not a controlled batch-size experiment.

## Audit and experiment order

1. Measure the existing pageable-host flag alone on decode. The inherited
   launcher lacked the documented per-role environment arguments; they were
   added with `SGLANG_`-only keys to preserve GPU assignment.
2. Audit found explicit pinned H2D in `DeepseekV4AttnBackend._move_to_device`
   and `dsv4/prefill_replay.py::_chunk_geometry`, plus a GPU-count
   `repeat_interleave` without known `output_size` in late-tail metadata.
   These are unchanged in the flag-only experiment.
3. CPU Engram staging copies GPU hashes to CPU synchronously before gathering
   immutable rows. This is a real dependency, not a redundant `.item()`; it
   cannot simply be removed. Most other `.item()` occurrences in metadata
   operate on CPU tensors or are capture-only.
4. Batching must make identical dispatch decisions on both TP ranks and keep
   running decode steps progressing during any coalescing delay.
5. Tail graph capture needs distinct stable full-tail and last-128-row layouts;
   the current hook deliberately rejects bounded replay with prefill graphs.
   Simply relaxing that gate would not establish correct captured buffers.

## Reproduction

`validate_ced_variant.py OUTPUT --sweep` waits for readiness, runs the inherited
`compare.py`, compares all output IDs to `/data/dsv41-bringup/ced/pp2/probe1.json`,
runs the copied exact oracle qualification, flushes prefill cache, executes the
copied sweep, and checks the oracle after load. `summarize_ced_variant.py OUTPUT`
extracts fixed-window scorecard metrics and rank-zero tail timings.

CPU checks: 23 tests and 14 subtests passed (CED PP transfer and PD launcher).
See `pp2-timing/cpu-tests.log`.

Pageable c56: 676 completed, zero failed, cache hit 95.61%. Repeated 20K
probe differs at tokens 3–4 (`_hints` versus `_paths`) and matches the other
22 tokens; all five first-token logprobs are identical. The strict validation
runner stopped at that difference; a repeat probe and explicit oracle checks
were completed before the manual sweep. Flag-only performance does not improve
on the control. Metadata cleanup CPU checks: 127 passed with flag off,
114 passed with flag on. Runtime validation of that separate flag is pending.

## Kernel profile (2026-09-29)

Artifacts: `/data/dsv41-bringup/ced/pp2-profile/traces/tail-{20,32,53}.json`
with operator tables and raw-kernel summaries. Captures are decode TP0 only,
respectively warm 8K, warm 40K, and 40K while another request is decoding.
The one-shot profiler adds overhead; do not use profiled elapsed times as
unprofiled latency measurements.

| Capture | Kernel count | Sum/union kernel ms | First-to-last kernel elapsed ms |
| --- | ---: | ---: | ---: |
| 8K | 2722 | 49.06 | 532.30 |
| 40K | 2740 | 40.73 | 541.16 |
| Light load, 40K | 2740 | 35.43 | 642.18 |

8K top kernels: NCCL bf16 allreduce 22.99 ms / 81 calls (two per model
layer plus one outside the stack); FP4 expert gate/up 7.74 ms / 40 calls and down 4.29 ms /
40 calls (one each per layer); dense block-scaled GEMM 2.25 ms / 212 calls;
mHC partial statistics 1.93 ms / 80 calls (two per layer); sparse prefill
attention 0.89 ms / 21 encoder calls; bounded paged attention 0.47 ms /
19 late-layer calls; KV dequantization 0.30 ms / 40 calls. At 40K, allreduce
is 12.85 ms and the expert GEMMs total 12.41 ms. Under light load allreduce
is 7.64 ms and expert GEMMs total 12.37 ms. NCCL durations include rank skew.

The dominant cost is eager host dispatch, **not 450–500 ms of GPU kernel
execution**. CUDA event elapsed time includes launch gaps. CPU operator
inclusive times at 8K include 72.3 ms / 212 FlashInfer dense GEMM calls,
41.1 ms / 252 quantization calls, 38.4 ms / 81 allreduce wrappers, and
2526 tensor allocations. These overlap/nest and must not be summed as exclusive
CPU time. Prefix dequantization and attention kernels are small.

Both profiled long-probe outputs exactly match the existing PP2 reference;
all three exact oracle checks pass. The first profiling launch failed because
this scheduler exposes TP rank as `self.ps.tp_rank`; this was corrected in
`c970820863`. Failed startup artifacts are retained under `server-failed-rank-gate`.

Next isolated experiment: `pp2-batch8`, max eight tails, max wait 1000 ms while
requests run, with a decode step between consecutive replays. Idle requests
still dispatch immediately. Profiling is unarmed during the benchmark.
