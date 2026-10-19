# Node A: CED replay graph results

The encoder's main layer path (0–20), the last-window row gather, and the
late-layer path (21–39) now support CUDA graph replay. **The performance
target is not met.** The stable configuration is running on GPUs 0–3 with
batch limit 8, maximum wait 100 ms, exact-row encoder graphs, and a fixed
256-shape cache. Uncached shapes run the encoder eagerly and still use the
full late graph. No captured graph is evicted during serving.

Router/prefill/decode/oracle: 8030/8031/8032/8034. Bootstrap: 8998. NCCL base:
29500. Metrics: 29000. Sessions: `nodea`, `nodea-proxy`. Kernel cache:
`/data/dsv41-bringup/ced/nodeA-kcache`. The shared Engram holder PID 10436 was
never signalled. No changes were made to the other worktree or its GPUs.

## Implementation and remaining eager work

The encoder uses 28 breakable graph segments, owning its input tensors,
hashes, positions, live-row counts, KV write locations, query-padding scratch,
and tail-gather indices/output. Hash history is computed once before capture
and replay. The late stack uses five segments with private sparse-index seeds
and live metadata at index-source breaks. Both captures use the distributed
capture context to register TP collective buffers.

Remaining breaks are two CPU Engram gathers, 21 encoder attention calls,
four encoder source calls, and four late index-source calls. The encoder
attention breaks include sparse-prefill work; source breaks include
projections and data-dependent index selection. Per-batch metadata hoisting,
attention plans, and asynchronous tail metadata staging are enabled.

Padding can change GEMM results. The measured configuration therefore uses
exact `(request count, packed rows)` shapes. Experimental padded execution
always checks against live eager output; failed live counts are quarantined.
A capture requires 20 GiB free on both ranks. Loader scratch is released once
before the first CED capture. Per-layer diagnostic activation references were
removed; final hidden/mHC/DSpark and SWA KV checks remain.

## Correctness

- 29 focused CPU tests pass: ownership/refresh, padding, immutable late
  metadata, control validation, byte equality, cache limits, and memory guards.
- Final post-load GPU validation records **102 passing rank/shape checks and
  zero mismatches**. Hidden states, mHC coefficients, and DSpark hidden states
  are compared byte-for-byte, including signed zero. Written SWA pages are
  compared bytewise against the ordinary eager path with live inputs/metadata.
- Encoder shapes include 160, 274, 288, 320, 368, and 640–2560 rows. Late
  shapes include 128–1024 rows, covering batches 1–8, plus short requests.
- Every exact-oracle case passed before and after the completed sweeps and
  after final validation.
- All five saved 8K–100K continuations matched exactly after the 50 ms load
  test. The final check matched four; 20K varied again. All final first-token
  IDs/log probabilities matched. Earlier eager/eager/graph/graph controls
  showed 20K variability even with both replay graphs disabled. This remains
  an end-to-end reproducibility gap, not an unconditional five-probe pass.

## Profile

These are TP0 one-shot traces; profiler overhead is included.

| Probe | Original kernel sum / span | Final kernel sum / span |
| --- | ---: | ---: |
| 8K | 49.06 / 532.30 ms | 28.77 / 137.50 ms |
| 40K | 40.73 / 541.16 ms | 30.14 / 120.11 ms |
| 40K with decode active | 35.43 / 642.18 ms | 30.17 / 131.66 ms |

Each final replay launches 33 graphs containing 1,659 kernels. Those kernels
sum to 24.7–25.1 ms and occupy 25.9–26.4 ms of graph-segment spans. The
remaining 1,012–1,030 eager kernels sum to only 4.1–5.0 ms but dominate the
remaining host-dispatch gaps. The complete replay is still host-bound;
its span is not close to its kernel sum.

## Fixed-window results

Protocol: 300 s warm-up, with a 240 s admission ramp inside it, then 300 s
measurement and drain; 256 conversations, up to 25 turns, physical GPU count
4, seed 42, oracle commit length 6. Reported iteration is the median streaming
event-interval proxy, not a scheduler-instrumented per-step timer.

Tail columns are warm complete replay medians for batch sizes **1 / 2 / 4**,
including eager fallbacks and excluding captures/checks. The 100 ms tail
medians combine its c56 and c72 log interval.

| Variant | Tail ms, b1 / b2 / b4 | Iteration proxy p50 | TPM/GPU | TTFT p95 | Users p50 / slowest 5% | Correctness / errors |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Retired replica A, c56 | — | 52 ms | 3.010M | 2.48 s | 114 / 106 | Reference |
| Retired replica A, c72 | — | — | 3.460M | 3.74 s | 100 / 88 | Reference |
| Node A, 50 ms, c56 | 122 / 282 / 319 | 143.8 ms | 2.109M | 2.284 s | 41.58 / 24.78 | Oracle pass; 1 timeout |
| Node A, 100 ms, c56 | 122 / 299 / 327 | 114.5 ms | 1.904M | 1.775 s | 52.11 / 42.09 | Oracle pass; 0 failures |
| Node A, 100 ms, c72 | 122 / 299 / 327 | 143.0 ms | 2.263M | 3.429 s | 41.74 / 34.45 | Oracle pass; 0 failures |

The 50 ms timeout was conversation 2, turn 6, a proxy read timeout. The
100 ms scorecards pass their generic cache-hit/output-share/error bands,
but both loads fail the requested 60 tok/s slowest-user floor and TPM target.
100 ms is left active for its better c56 user rates, lower TTFT, and zero
failures in the completed c56/c72 pair. It is not a baseline-beating result.

## Cache coverage and warmed shapes

Across the final 100 ms sweeps, only 751 of 2,250 non-short tail batches
(33.4%) matched an admitted encoder graph. Warm cached medians for batches
1–4 were 100.28 / 102.85 / 110.02 / 116.79 ms; fallback medians were
298.26 / 307.87 / 313.74 / 328.01 ms. The batch-4 cached value has one sample.

A separate post-load 8K cached-shape probe measured batches 1–8 at
**114 / 116 / 108 / 119 / 127 / 142 / 141 / 146 ms**. Batch 1 has ten
samples; other sizes have one each. These are observational timings, not
statistically precise batch-size comparisons.

Remaining work is to reduce the eager indexer/attention scaffolding and
support more exact shapes without large capture cost or retained storage.
Padded buckets are not numerically interchangeable with eager execution.
The whole-stack dense capture helps cache hits, but cache misses and eager
breaks prevent the throughput and user-rate targets from being reached.

## Failed experiments

- Unbounded exact-shape retention: stopped during ramp as decode memory grew
  from about 207 GiB to 260 GiB; no measurement scorecard.
- Padded/exact LRU candidate: OOM during ramp. Per-layer diagnostic references
  retained intermediate activations and were removed.
- Bounded LRU candidate: illegal memory access after the cache reached its
  limit. Shared-pool/weak-intermediate lifetime during eviction is suspected,
  not proven. Static retention has survived all three completed sweeps and
  subsequent repeated byte checks without that failure.

## Artifacts and commits

All artifacts are under `/data/dsv41-bringup/ced/nodeA/`:

- `static/`: running worker logs, launch script, manifest, probes, and traces.
- `static-50ms-c56/`: first completed scorecard, summary, coverage, post-oracle.
- `static-100ms-final/`: c56/c72 scorecards, summary, coverage, warmed-shape
  timings, 102-check validation, final probes/oracle, and `final-options.json`.
- `full-livecheck/control/`: eager/eager/graph/graph control evidence.

Key local commits: `61da51334c` (encoder graph), `531c17d1e8` (full late
stack), `1ed27f638b` (private index seeds and graph ownership), `485ffceeb0`
(live eager oracle), `16e228468a` (byte checks, capture context, memory),
`71df7f9665` (exact-row default), and `01d66b2649` (static retention).
The distributed capture-context change follows `d2ee554b1e` from the parallel
branch. All commits are local, authored with the requested identity; nothing
was pushed and Git configuration was not changed.
