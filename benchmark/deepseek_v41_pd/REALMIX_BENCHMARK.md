# DeepSeek V4.1 oracle real-mix replay

Run date: 2026-09-28. Sweep complete (40/54/67/80). Raw artifacts: `/data/dsv41-bringup/realmix/`.

## Workload and provenance

User-supplied dataset:
https://f004.backblazeb2.com/file/fileblob/oracle-datasets/oracle-realmix-256-r67-m3-modeled-long25-38k-v3.json

Downloaded size: 75,128,452 bytes. SHA-1 matches the storage response:
`c5bcc40ac292fe229431adf7b51deb8ae39a4806`.
SHA-256: `baca830c1fd0cd8d4900c52faf52d7b3893df5a1b89e44b6927567ff8bd2ea60`.

The file contains 256 conversations and 6,400 turns. Its metadata explicitly
labels it a synthetic modeled mix derived from real agent traces, with stitched
continuations and synthetic project archives. It is not an unmodified production
trace, and it is not the 800-conversation file mentioned in the supplied M3 report.

The replay driver and token metrics were recovered from this repository's
`m3-fixes` branch at revision `6ab772dbb86dd5678eef73032cafef6042ca5181`
into `benchmark/oracle_replay/`. Its served TPM numerator is
confirmed in code: **prompt tokens + completion tokens**, including cached prompt
tokens. Output, uncached input and cached input are also reported separately.

## Token conversion

Source forced IDs belong to MiniMax's vocabulary. Downloaded tokenizer-only files
from `MiniMaxAI/MiniMax-M3` revision
`f0e1c1e04d40177e4673a22097036854f536e9c0`; no M3 weights were downloaded.
Each forced completion is decoded to text and encoded with the local DeepSeek
checkpoint tokenizer. Messages and conversation ordering remain unchanged.

Mean forced completion length changes from 555.82 M3 tokens to 551.31 DeepSeek
tokens. Output limits are set to the exact converted length, with EOS ignored,
so organic tail generation cannot alter the intended replay. The supplied file
originally adds eight tokens of output slack. This difference is recorded rather
than treated as an exact reproduction of an unknown M3 invocation.

Final accumulated history plus output estimates average 61,188 DeepSeek tokens,
with maximum 81,305. The benchmark raises context from 16K to 128K rather than
truncating the data. Actual request usage counters remain authoritative.

Conversion script, source tokenizer and summary are retained under the run
directory. The converter uses the same accumulated-history behavior as the
recovered replay driver, including supplied assistant messages in turn deltas.

## Oracle execution

This branch previously supported simulated acceptance but lacked M3's
`forced_token_ids` API. A benchmark-only port adds request-specific forcing,
initial prefill-token forcing, and static DSpark oracle commit handling.

DSpark still runs its draft forward and the target verification forward. The
configured commit length is six (five draft positions plus a bonus), capped by
remaining forced tokens. This is a controlled acceptance assumption, not a
measured real acceptance rate or an inference-quality evaluation. The exact
commit setting used for the supplied M3 report was not provided.

Unlike the older output-only M3 override, the DeepSeek path also replaces target
verification inputs with the oracle sequence after draft computation. This keeps
committed target KV and CPU Engram history consistent with replayed tokens. The
draft computation remains real, but the verification token distribution is
teacher-forced; document this difference in any comparison. Fused in-graph
acceptance is disabled only for oracle requests; target/draft model graph replay
remains eligible. Static verification is required.

Forcing is rejected unless the server explicitly enables
`--oracle-commit-length`. Ordinary requests retain real sampling and acceptance.
The endpoint remains local; this work does not send traffic to the supplied M3
provider endpoint or use its credentials.

## Local PD adapter and qualification

The scored run uses `oracle_pd_proxy.py` on loopback port 8034. The existing Rust
router drops the added `forced_token_ids` field, while the diagnostic Python
MiniLB preserves forcing but does not combine prefill cache usage with decode
usage. Neither diagnostic route is used for the scored sweep.

The adapter renders messages with the official V4.1 encoder in thinking mode,
sends identical native token IDs and bootstrap coordinates to the prefill and
decode workers, and streams decode responses. Prompt/completion counts come from
decode; cached prompt counts come from prefill's native response metadata. It
adds no synthetic cache hits. This is a benchmark adapter, not a production router.
Its rendering differs slightly from the worker's default Chat API formatting.

Qualification checked exact output IDs and decoded text for two concurrent
153-token forced completions. Repeating the 786-token prompt returned 768 cached
tokens. Four oracle unit tests passed. An earlier run without cache details is
retained as `diagnostic-c40-no-cache-details.log` and excluded from scored results.

## Measurement plan

* One four-GPU TP2/EP2 prefill + TP2/EP2 decode pair; B300, CC/NVLE, DevTools ON.
* Both CPU Engram tables stay in shared locked RAM; CUDA IPC transfer and split graphs.
* Requested context 131,072 and KV cap 2,097,152 tokens; record actual allocated
  capacities from server-info rather than assuming the requested cap fits.
* Client concurrency 40, 54, 67 and 80; decode running-request cap remains 64.
* 60-second warmup, 300-second scored window, streaming, up to 25 turns per conversation.
* Four allocated GPUs in the denominator, including both prefill GPUs.
* Fixed-window counter deltas are primary; drain-inclusive results are separate.
* Retain cache state between levels; report actual hit rate rather than imposing 95%.
* Record failures, served/output TPM per GPU, RPM, cache hit, output share,
  p50/p95 TTFT and token pacing. Scorecard failures remain visible.

The matched dataset and driver improve comparability, but hardware topology,
tokenization, acceptance setting and the supplied M3 report's 800-conversation
description remain differences. No speedup or quality equivalence is assumed.

## Scored results

Each row uses the 300-second window after the 60-second warmup. The denominator
is four physical GPUs, including prefill. Served TPM counts cached input, uncached
input and output. Output TPM is shown separately. TPOT is per-request token pacing
after first output, summarized by its median; this is an oracle-acceptance run.

| Concurrency | Served TPM/GPU | Output TPM/GPU | RPM | Cache hit | TTFT p50 / p95 | TPOT p50 | Completed / failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 261,356 | 3,725 | 39.8 | 56.4% | 54.45s / 69.96s | 4.13ms | 199 / 0 |
| 54 | 276,835 | 3,929 | 43.8 | 58.6% | 68.27s / 84.50s | 4.18ms | 219 / 0 |
| 67 | 284,863 | 4,025 | 45.8 | 59.6% | 82.13s / 98.27s | 4.61ms | 229 / 0 |
| 80 | 291,069 | 4,108 | 48.0 | 61.2% | 91.32s / 110.72s | 4.66ms | 240 / 0 |

All four rows fail the harness cache-hit band (94–96%) and its interactive
TTFT target (3 seconds). Successful HTTP requests do not mean those performance
targets passed. Warmup and drain counters are retained separately in the JSONs.

All four levels fail the scorecard; drain-inclusive runs completed 300/338/365/384
requests with zero failures. Doubling client concurrency from 40 to 80 raised served
TPM/GPU by only 11%, while TTFT p50 rose from 54s to 91s. Uncached prompt throughput
stays nearly flat at 7.4–7.6K tokens/s across all levels, so the prefill pair is
saturated on recomputed prompt tokens. Additional concurrency mostly adds queueing
before prefill. Decode pacing is not the limiter: median TPOT stays at 4.1–4.7ms.
The small hit-rate increase, 56% to 61%, is too small to change this. Raising the
cache hit rate (prefix-tail retention) or prefill capacity is the lever, not
decode concurrency.

At concurrency 40, 43.6% of prompt tokens were recomputed versus 5.4% in the supplied
M3 report. That is about an eightfold difference in uncached fraction. Holding
DeepSeek's measured uncached-input rate and output share constant while assuming
94.6% cache hits would arithmetically yield about 2.11M served TPM/GPU. This is a
conditional accounting illustration, **not** a measured result or a prediction:
cache changes also change scheduling, batching, prompt mix and kernel costs.

The concurrency-40 scored window averaged 25,893 prompt tokens and 374 output
tokens per completed request. Slow progression through the conversations means
the fixed window does not sample the full dataset's ~38K-token mean geometry.

### Comparison with the supplied M3 report

| Concurrency | DeepSeek served TPM/GPU | Reported M3 served TPM/GPU | Reported M3 / DeepSeek |
| ---: | ---: | ---: | ---: |
| 40 | 261,356 | 2,178,260 | 8.33x |
| 54 | 276,835 | 2,392,620 | 8.64x |
| 67 | 284,863 | 2,632,693 | 9.24x |
| 80 | 291,069 | 2,662,877 | 9.15x |

These are comparisons to the supplied report, not a fresh M3 run on this exact
file. DeepSeek uses four GPUs in PD, M3 used eight GPUs in two colocated replicas;
this file is a 256-conversation modeled trace whereas the report describes 800
conversations. Tokenization and oracle assumptions also differ. The observed gap
is not an isolated model-architecture or kernel comparison.

DeepSeek's aggregate uncached input rate stays at 7.43–7.56K tokens/sec across the
four scored windows. Decode logs generally show only one or two active requests,
with most requests waiting upstream. Median token pacing is 4.13–4.66ms after the
first output under the six-token oracle assumption. This supports a prefill/cache
bottleneck for this workload, rather than a conclusion that decode is intrinsically
8–9 times slower. IPC transfer latency remains unisolated.

There were 887 successful scored requests and zero scored failures. Including
warmup and drain, 1,387 requests completed and zero failed. Scored mean prompt
lengths are 25,893 / 24,923 / 24,527 / 23,913 tokens at concurrency 40 / 54 / 67 / 80;
mean output lengths are 374 / 359 / 351 / 342 tokens. Output share stays at
1.41–1.43%. Cache state was retained between levels, so this is not an independent
cold-cache concurrency A/B.

## Fix 1: prefill `--swa-prefix-tails 256`

Same replica, dataset, driver and oracle settings; only the prefill worker adds
`--swa-prefix-tails 256` (new launcher flag `--prefill-swa-prefix-tails`). Prefill's
SWA pool grows from 30,208 to 116,224 tokens (384 per tail). Oracle qualification
matched the baseline exactly. The cache was cold at c40 and retained between levels.
c80 was skipped by request to free the replica for profiling. Raw results:
`/data/dsv41-bringup/realmix/tails256/`.

| Concurrency | Served TPM/GPU | Output TPM/GPU | RPM | Cache hit | TTFT p50 / p95 | TPOT p50 | Completed / failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 1,864,135 | 26,065 | 186.6 | 95.0% | 7.67s / 10.97s | 7.19ms | 933 / 0 |
| 54 | 1,847,178 | 26,186 | 199.8 | 95.4% | 11.85s / 14.91s | 7.27ms | 999 / 2 |
| 67 | 1,634,959 | 23,213 | 200.2 | 95.0% | 16.13s / 30.56s | 7.45ms | 1001 / 0 |

At c40 served TPM/GPU rises 7.1× over the baseline and the hit rate enters the
94–96% band, confirming SWA tail eviction as the dominant cause of the low hit
rate. Scored requests now average 39.4K prompt tokens, close to the dataset mean.
Throughput saturates at ~200 RPM above c54 while TTFT grows. The two c54 failures
were 180-second client read timeouts on the first request of conversations 0 and 1
at level start; servers logged no errors and c67 did not repeat it. Treat this as an
unexplained PD hand-off stall until reproduced.

### Profile at saturation

Profile under an unscored c80 load: `/data/dsv41-bringup/realmix/tails256/profile/`.
Prefill runs eagerly for every chunk (zero graph launches; graph `max_seq_len` is
16,384 against ~39K-token prompts). Each ~3.7K-token chunk takes ~710 ms of wall time
for ~130 ms of non-collective GPU compute; TP0's GPU is busy 22% of the trace. The
prefill all-reduce dominates summed kernel time (59%) only as waiting: both ranks are
host-bound, TP0 marginally later, and TP1 spins in NCCL until TP0 arrives (matching
all-reduces end within 3 µs). TP0's all-reduce median is 109 µs for ~40 MB; its kernels
start a median 13 µs after launch. Profiler overhead is ~10–15% of prefill throughput. Decode replays graphs,
runs TARGET_VERIFY at batch ~10 of 64 and is starved by prefill.

The next lever is prefill host overhead on long contexts: graph coverage beyond 16K
(context-specific buckets, measuring indexer logits memory) or reduced eager launch
overhead. Kernel work on decode's expert GEMMs cannot raise throughput while decode
is starved.

## Configuration findings during the baseline

Actual full-token capacity is 2,097,152 on each role. Prefill SWA capacity is
30,208 tokens; decode SWA capacity is 74,496. These are separate pools, so raising
the full-token cap did not proportionally increase sliding-window capacity.
Prefill defaults to 32 cached prefix tails (four times its eight running-request
slots), below the 40–80 client conversations in this sweep. This is a cache-retention
tuning candidate; eviction causality needs an A/B measurement.

The normalized prefill graph configuration has `max_seq_len=16384`, despite the
131,072-token context limit. The attention backend explicitly rejects graph replay
above that limit. Long-context prefill therefore runs eagerly; decode graph replay
remains active. Prefill late-layer SWA trimming is also disabled to permit the
configured split graphs, including when a request takes this eager fallback.

An offline check of eight turns each from conversations 0, 1 and 40 finds that
successive rendered prompts share all but the last token of the previous prompt.
Potential prompt reuse on those follow-up turns is about 86–97%, before page
alignment and cache eviction. The chat template is not intrinsically destroying
most of these sampled prefixes. Raw results are in `prefix-potential.json`; this
small sample is not a measured server cache-hit rate or a full-dataset estimate.

Next controlled experiments should increase prefill prefix-tail headroom (for
example 256, then 512 tails) without changing kernels, then compare long-context
prefill execution modes separately. Simply raising graph `max_seq_len` to 128K is
not a demonstrated optimization: the low-ratio indexer uses this bound for a
static logits width, so graph memory/work can grow. Context-specific graph buckets
or an eager path with late-layer trimming need measurement.

Low logged pool utilization does not disprove cache pressure: the scheduler's
usage calculation subtracts evictable cached tokens as well as free tokens. The
32-tail limit is a reservation policy, not a direct eviction-counter measurement.

No serving configuration is changed during the scored sweep. Queue size alone
does not measure PD transfer cost: decode's transfer queue includes requests whose
prefill computation has not yet finished.

## Steady-state protocol

The driver starts conversations 0..N-1 together at every level. Once throughput is
high, the first wave of 25-turn conversations finishes within seconds of each
other and N never-seen conversations start cold together: prefill logs
`#cached-token: 0`, the interval hit rate falls to ~26-37%, and prefill saturates for
1.5-3 minutes. Windows that end before the first wave (the baseline and Fix 1) omit
conversation-start cost entirely and are optimistic; windows that contain one wave
are distorted the other way. From here on levels use `--ramp-in 240`,
`--warmup-seconds 300` and a 300 s window, so starts are staggered and the window
is steady state. Earlier numbers are not comparable with the tables below.

A larger decode KV pool (`--decode-max-total-tokens 8388608`, new launcher flag) was
tested as a fix for the wave and did not remove it; it is kept because it removes
decode admission stalls (decode reached 97-99% KV use with 84-95% pre-allocated during
a wave). V4.1 KV costs ~1.6 KB/token: only the four kv-source layers own compressed
latents (584 B/entry) and FP4 index keys (68 B/entry); 8.4M tokens use ~14 GB/GPU.

## Long-context prefill graphs

Prefill graphs previously replayed only up to `max_seq_len=16384`, so every real-mix
chunk (~39K-token contexts) ran eagerly and host-bound (TP0 GPU busy 22%; ~710 ms per
~3.7K-token chunk for ~130 ms of GPU compute). The 16K cap protected one captured
op, the ratio-1/2 low-ratio indexer. `SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER`
(launcher `--prefill-graph-eager-indexer`) runs that indexer as an eager break on
live rows with the eager dense indexer (including two-level candidate selection)
and lifts the cap; everything else stays captured.

Single-request probes (8K-100K, prefill cache flushed): first tokens identical to
eager; first-token logprobs and top-5 bitwise identical except 70K, where tail
alternatives (logprob -9 to -15) shift 0.3-0.9 nats with identical greedy output.
Unexplained; a GPU test separating bucket padding from index-key differences is
proposed. A static Codex review found no correctness bugs. Warm prefill latency:
40K 4.3 -> 2.7 s, 70K 9.8 -> 5.6 s, 100K 11.9 -> 8.1 s.

## Per-batch hoisting

A trace attribution of the attention path found ~115 ms of host time per prefill batch,
including ~23 device-to-host syncs from per-request indexing in the low-ratio indexer
and per-layer rework of batch-invariant data.
`SGLANG_OPT_DSV41_PREFILL_HOIST_PER_BATCH` (launcher `--prefill-hoist-per-batch`)
builds the indexer's slot layout once per batch per ratio on the device, gathers
index keys once per latent source, recombines top-k indices only at index-source
layers (8 instead of 38) and dequantizes each source's compressed region once (4
instead of 38). Probes are bitwise identical to the graph path without hoisting;
warm prefill latency drops a further 7-19% (100K 8.1 -> 7.2 s).

## Steady-state results, one pair (4 GPUs)

All rows: 256 prefix tails, long-context prefill graphs, decode KV 8.4M, staggered
protocol, oracle commit 6. Per-user speed is 1000 / per-request token pacing.

| Variant | Concurrency | Served TPM/GPU | RPM | Cache hit | TTFT p50 / p95 | Per-user tok/s p50 / p95-slowest | Iteration p50 | Done / failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk 4096 | 40 | 2,238,212 | 225.8 | 95.0% | 0.98s / 1.91s | 126 / 116 | 47 ms | 1129 / 0 |
| chunk 4096 | 56 | 2,771,803 | 278.8 | 95.0% | 1.91s / 4.21s | 111 / 104 | 54 ms | 1394 / 0 |
| chunk 4096 | 72 | 2,885,052 | 289.2 | 94.9% | 5.04s / 6.94s | 111 / 104 | 54 ms | 1446 / 0 |
| chunk 4096 | 96 | 989,153 | 99.4 | 79.8% | 11.13s / 145.92s | 104 / 98 | 57 ms | 497 / 1 |
| chunk 8192 | 56 | 2,805,573 | 282.8 | 94.9% | 1.92s / 3.63s | 111 / 102 | 54 ms | 1414 / 1 |
| chunk 8192 | 72 | 2,891,930 | 291.6 | 94.9% | 4.71s / 6.92s | 109 / 101 | 55 ms | 1458 / 0 |
| chunk 8192 | 96 | 1,083,845 | 108.4 | 80.7% | 11.40s / 112.09s | 101 / 86 | 59 ms | 542 / 0 |
| + hoist | 56 | 2,963,193 | 297.2 | 95.0% | 1.15s / 2.54s | 107 / 92 | 56 ms | 1486 / 0 |
| + hoist | 72 | **3,325,279** | 337.2 | 94.9% | 2.28s / 3.56s | 97 / 86 | 62 ms | 1686 / 0 |

- Chunk size is not a lever for this workload: prefill batches rarely fill a chunk
  (median 3,328 new tokens, p90 7,936), because a 95%-hit turn brings ~2K uncached
  tokens and prefill batches 1-2 requests. Host cost scales with batches, not chunk.
- c96 collapses with either chunk size: 96 live ~40K-token conversations need
  ~3.8M tokens of prefix cache, prefill's pool is 2.1M, and the hit rate falls to ~80%.
  A larger prefill pool (KV is ~1.6 KB/token) is the next fix to test.
- At every level the slowest 5% of users decode at >= 86 tok/s under the oracle
  (commit 6) assumption; decode runs a batch of ~23 at the one-pair ceiling, so
  prefill limits throughput. Real acceptance is unmeasured (one unforced sample
  accepted ~4.0 tokens per iteration).

## Artifacts

`/data/dsv41-bringup/realmix/` contains source/converted datasets, checksums,
conversion metadata, launch script/logs, oracle unit-test results, server-info,
and the replay sweep script. The server and benchmark run in named tmux sessions.
Per-level result files are `c40.json`, `c54.json`, `c67.json`, `c80.json`, with
corresponding logs and failure records.

## End state and reproducibility

The sweep completed at approximately 17:14 UTC on 2026-09-28. `SWEEP_DONE` records
successful orchestration. `results-summary.json` and `summarize.py` provide compact
fixed-window summaries alongside the complete raw results.

The model remains running in tmux session `dsv41-oracle`; the ordinary serving
router is `http://127.0.0.1:8030` and reported two healthy workers after the sweep.
The benchmark adapter remains in `dsv41-oracle-proxy` on loopback port 8034. Oracle
forcing is request-specific; ordinary requests retain real sampling/acceptance.
The earlier `dsv41-pd.service` is not supervising this benchmark deployment. CPU
Engram residency remains managed by `dsv41-engram-cache.service`.

Reproduce with the saved `launch-oracle.sh`, then the local adapter and
`run-sweep.sh` under `/data/dsv41-bringup/realmix/`. Re-run `qualify-and-run.py` only
when intentionally replacing the level result files; archive existing results
first. The four oracle unit tests and exact-token/cache qualification precede this
sweep; no new kernel speedup is claimed.
