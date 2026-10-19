# DeepSeek V4.1: TP2 prefill + TP2 decode with CPU Engram

For the consolidated measurement snapshot, comparison caveats and unresolved PD
overhead, see [the performance report](PERFORMANCE.md) and its
[complete kernel breakdown](PERFORMANCE_KERNELS.md). The sections below include
historical bring-up configurations and results. The newer
[M3-style oracle real-mix replay](REALMIX_BENCHMARK.md) records the 128K-context
benchmark configuration, four-level results, cache accounting and tuning findings.

This is a reproducible bring-up and optimization candidate, **not a qualified
optimized deployment**. CPU staging works around the mapped-host-memory
incompatibility found on this confidential-computing machine. TP2 weights and
KV pools fit, and routed inference with split CUDA graphs is serving. The
initial fixed-length batch-16 benchmark delivered 43,281 output tokens/minute
(128 requests, 1,024 input and 256 output tokens each; DSpark off).

## Current serving configuration: split CUDA graphs

`--cpu-engram-staging --split-cuda-graphs` uses the existing
breakable graph backend. `StagedEngramTable._stage_bytes` is an eager break:
copy current indices to CPU, gather FP8 rows/scales, then copy those compact
buffers to GPU. The runner retains stable bridge buffers and refreshes their
contents on every replay. Dequantization, projection, and subsequent model
computation can be captured. Existing model-specific graph breaks still apply.

DSpark is enabled by default; add `--no-dspark` for the target-only control.
The runtime automatically loads the draft head bundled in the target checkpoint.
Block size 5 means five proposed tokens and a six-token verification window.
Verification also satisfies `ForwardMode.is_extend()`, but uses the decode graph
runner. The model's hash/attention helpers and low-ratio projection dispatch
therefore exclude verification from prefill-only graph breaks. Ten CPU regression
tests cover the live-row handling and this dispatch boundary.

The checkpoint's Engram layers are 1 and 14 (zero-based), with 40 target layers.
Layer 20 is the last shared-KV source, not the end of the target model.
The prefill worker captures buckets 256/512/1024/2048/4096 tokens, capped by
chunk size; the decode worker captures powers of two up to the request limit,
plus the exact limit when it is not a power of two. Thus `--decode-requests 64`
captures batches 1/2/4/8/16/32/64. Unsupported shapes retain the runner's eager
fallback.

The model currently rejects prefill graphs with decoder SWA bounded replay,
which trims token rows after layer 20. This candidate disables that trimming
on the prefill worker while retaining it on the decode worker. It is a
measurable tradeoff, not a claim that every prefill workload improves.

Validation passed: the tiny two-lookup GPU test creates three actual
CUDA graph segments and verifies exact output after repeated changes to
GPU-computed indices, including duplicate rows. It also captures structured
decode outputs at two batch sizes and checks logits/hidden-state replay after
shape switches. The generic breakable backend now retains `LogitsProcessorOutput`
per shape, matching the full-graph backend; previously it rejected this output.
The prefill low-ratio source helpers also now read the canonical
`global_num_token_non_padded_cpu` field. Six CPU tests verify that empty and
padded buckets only project their live rows and zero padding.
Run the GPU probe with:

```bash
CUDA_VISIBLE_DEVICES=4 /data/venvs/dsv41/bin/python \
  benchmark/deepseek_v41_pd/probe_staged_engram.py --split-graphs
```

Baseline response/timing evidence: `/data/dsv41-bringup/split-baseline.json`.
Initial split-graph logs: `/data/dsv41-bringup/pd-split-04`. Both workers completed capture;
prefill took 23.2 s and added 3.25 GiB per GPU, decode took 19.6 s and added
0.35 GiB. Router readiness reports two healthy workers and `smoke.json` passed.
The endpoint remains `http://127.0.0.1:8030/v1`, model `dsv41`, managed by
`dsv41-pd.service`; Engram residency remains in `dsv41-engram-cache.service`.

Actual request logs show graph replay for prefill and decode. Correct chat
responses were verified at 439, 919, 1731 and 3441 prompt tokens, including
concurrent requests and cached prefixes. Evidence is `split-results.json` in
the bring-up directory and `large-bucket-smoke.json` in the live log directory.
Short prompts can still run eager prefill because the smallest captured bucket
is 256 and the runner rejects excessive padding; their decode uses graphs.

Against the earlier eager samples, the 439-token request took 0.54 s versus
3.32 s, and the 919-token request 0.21 s versus 1.42 s. These are single-request
sanity timings, affected by cache/warmup and the prefill trimming change, not a
controlled throughput benchmark. The arithmetic-list response was correct but
changed formatting (`²` versus `^2`), so bitwise eager equivalence is not claimed.
DSpark was off for these initial measurements. Broader quality/performance qualification and graph-compatible
late-layer trimming remain future work.

## DSpark deployment — 2026-09-28

Live logs: `/data/dsv41-bringup/pd-dspark-03`. The serving pair enables the bundled
DSpark model, block size 5, CPU Engram staging, and split CUDA graphs. Decode
capacity is 64 requests and the per-worker KV cap is 131,072 tokens. The topology
remains TP2+EP2 prefill on GPUs 0–1 and TP2+EP2 decode on GPUs 2–3. Mooncake's
intra-node transport uses CUDA IPC.

The initial DSpark capture exposed verification entering prefill-only hash and
attention helpers, because `is_extend()` includes `TARGET_VERIFY`. The model and
low-ratio source dispatch now exclude that mode from prefill-specific breaks.
Ten CPU regression tests and nineteen launcher tests pass. Target verification
captured all seven buckets through 64 in 13.83–14.02 s, using 1.72 GiB per rank.
Draft verification captured in 112.41–112.53 s including first-time tuning,
using 0.25 GiB per rank; greedy and sampling proposals are folded into its graph.

Both workers report `speculative_algorithm=DSPARK`, the local target checkpoint
as the draft path, and six verification tokens. The routed smoke and chat checks
passed (squares 1–5, Paris, and 42); see `chat-smoke.json` in the run directory.
DevTools was already ON at this startup, so the launcher uses `--devtools-mode
on`; CC remains ON/NVLE and Ready. The launch did not change GPU mode. Previous
DevTools-OFF timings are not a controlled DSpark-only comparison.

The fixed-length synthetic validation completed 256/256 requests at concurrency
64 (1,024 input / 256 output tokens each), in 28.03 s: 2,338 output tokens/s,
or 140,290 output tokens/min. Mean TTFT was 2.245 s, mean time/output token
14.545 ms, and mean request latency 5.953 s. Both workers' prefix caches were
cleared before the benchmark's one-request warmup. Decode logs show accepted
draft tokens, graph replay, and no retractions. Evidence:
`/data/dsv41-bringup/dspark-benchmark-summary.json`, `tpm-dspark-b64.jsonl`, and
`dspark-validation.log`. This is a short synthetic run, not peak-capacity or
broad quality qualification.

## Larger-batch throughput — 2026-09-27

Target-only baseline logs: `/data/dsv41-bringup/pd-batch64-01`. That pair used
`--decode-requests 64 --max-total-tokens 131072`, with split graphs and DSpark
off. Decode captured all seven buckets through 64 in 32.38 s, adding 0.64 GiB
per GPU. The launcher now generates graph buckets through the configured limit;
previously the explicit list stopped at 16 even when request capacity increased.

Each run below completed 256/256 requests, each with exactly 1,024 input and 256
output tokens. Both workers' prefix caches were cleared before each run; the
benchmark then issued one warmup request. These are unprofiled synthetic runs,
not a maximum-capacity or production workload qualification.

| Client concurrency | Output tokens/min | Mean TTFT | Mean time/output token | Mean request latency |
| --- | ---: | ---: | ---: | ---: |
| 16 | 44,226 | 454 ms | 19.7 ms | 5.47 s |
| 32 | 72,403 | 656 ms | 23.5 ms | 6.66 s |
| 64 | 104,044 | 1,189 ms | 30.5 ms | 8.96 s |

Batch 64 delivered 2.35 times batch-16 throughput, with increased request
latency. Evidence: `/data/dsv41-bringup/batch-sweep-summary.json` and
`tpm-batch{16,32,64}-256.jsonl` in the same directory. GPU topology remains
TP2 prefill on 0–1 and TP2 decode on 2–3, with both Engram tables in CPU RAM.

## Decode profile — 2026-09-27

A 40-step CPU trace at actual decode batch 16 measured a mean step interval of
20.65 ms on both ranks. Per rank, the four CPU Engram `index_select` calls total
0.39–0.41 ms/step. Blocking index copies to CPU take 3.45–3.47 ms/step; selected
rows/scales copied back to GPU take 0.41–0.53 ms/step. The output-copy range takes
5.50–5.64 ms/step. Copy timings include waiting for preceding asynchronous GPU
work, so they do not measure pure transfer cost. Nested events and concurrent
TP ranks must not be summed together.

CUPTI returns `CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED`, so GPU kernel
and communication attribution is unavailable in this trace. No CC settings
were changed. Evidence: `/data/dsv41-bringup/profile-b16-steady/` and
`/data/dsv41-bringup/profile-b16-summary.json`. These measurements do not support
calling the CPU table gather itself the dominant bottleneck.

A second 40-step trace while filling a 64-request batch includes ten steps at
actual batch 64. Restricting events to those ten scheduler iterations gives a
30.3 ms mean step interval, 1.13–1.24 ms CPU gather, 5.48–5.84 ms blocking index
copies, 0.95–0.97 ms selected-row copies, and 9.71–10.09 ms output-copy range
per step/rank. The same GPU-wait caveat applies. Evidence:
`/data/dsv41-bringup/profile-b64-summary.json` and `profile-b64-steady/`.

## Historical eager baseline — 2026-09-27 19:10 UTC

The target-only PD pair is serving at `http://127.0.0.1:8030/v1`, model `dsv41`.
Router `/readiness` reports two healthy workers. The full routed completion
returned `Paris.` and a chat request for 17 + 25 returned `42`. Evidence:
`/data/dsv41-bringup/pd-target-06/smoke.json` and `chat-smoke.json`.
A further 35-token chat response correctly listed squares 1 through 5
(`chat-multitoken.json`), exercising multiple decode steps.
`dsv41-pd.service` and `dsv41-engram-cache.service` remain active. These are
transient systemd units, not an automatic reboot-persistent deployment.

GPUs 0–1 serve prefill; 2–3 serve decode, about 146.4 GiB HBM per GPU after
warmup. Engram uses one shared CPU mapping. DSpark and CUDA graphs are off.
These smoke checks establish live inference, not throughput or quality
qualification. Stop the pair with `sudo systemctl stop dsv41-pd`; stop the
residency helper separately to release the locked CPU pages.

## Earlier bring-up details

The driver now consistently uses 595.91.07. All eight GPUs passed the NVIDIA
local verifier's certificate, signed RIM and runtime measurement checks; the
verifier set Ready, with CC ON and NVLE. Evidence is in
`/data/dsv41-bringup/attestation-persistent.log`. No VM restart was needed.
A runtime-only persistence service override enables persistence and prevents
the service stopping between clients:
`/run/systemd/system/nvidia-persistenced.service.d/dsv41-recovery.conf`.
The standard udev rules have been restored after controlled driver recovery.

`--cpu-engram-staging` now selects an experimental eager lookup. It maps the
original safetensors tables, sharing immutable page-cache pages across P/D,
gathers selected FP8 rows on CPU, and copies those compact rows to CUDA for
the existing dequantization kernel. It allocates no full private table copies.
Both prefill and decode CUDA graphs are disabled. Tiny CPU and exact GPU
tests pass, including duplicate indices, changed indices and empty batches.
Both actual checkpoint tables were mapped and sampled. This is a correctness
bring-up path, not an optimized/graph-compatible implementation. Checkpoint
files must remain immutable while serving.

`dsv41-engram-cache.service` runs `hold_engram_cache.py` with unlimited memlock.
It holds read-only mappings of the two Engram shards and locks their shared
page-cache pages in RAM: 203073077576 bytes including shard metadata and small
non-Engram tensors. This prevents disk faults during serving without allocating
per-rank copies. Never mlock the writable private safetensors mappings: faulting
those for write could break sharing. Stop the residency helper with
`sudo systemctl stop dsv41-engram-cache` when the deployment no longer needs it.

The separate environment is `/data/venvs/dsv41`; recorded dependency versions,
attestation and preflight results are under `/data/dsv41-bringup`. Mooncake's
wheel links CUDA 12, so the launcher service sets
`LD_LIBRARY_PATH=/data/venvs/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib`.
`dsv41-pd.service` supervises the target-only PD bring-up using
`--cpu-engram-staging --no-dspark`. Worker logs are under
`/data/dsv41-bringup/pd-target-06`. The standalone cross-process Mooncake NVLink transfer probe also passed with
exact payload verification under CC (`probe-transfer.log`). The launcher waits on worker `/health` and router `/readiness` (which requires
both PD roles registered), then requires a successful `smoke.json`. Router
`/health` and systemd active state alone do not indicate serving readiness.
This run uses `--parallel-load` and a shared `--kernel-cache-dir` pointing to
the previous prefill cache, reusing compiled artifacts. The matching
`flashinfer-jit-cache==0.6.18+cu130` package supplies compiled extension modules;
FlashInfer still downloads individual cubins during first-run tuning. Mooncake's TCP override must be absent for
NVLink: even `MC_FORCE_TCP=0` activates TCP in the installed engine. The launcher
also explicitly disables auto-enabled FlashInfer all-reduce fusion for the
initial NCCL control configuration.

The first full routed request exposed a thread-local CUDA device bug in
Mooncake's intra-node path: TP1 executor threads defaulted to device 0, and
`cudaMemcpyBatchAsync` failed with `operation not supported`, followed by a
native crash. Main-thread rank-1 copies succeeded; a threaded reproducer failed.
Selecting the rank's CUDA device in the transfer wrapper fixes the reproducer,
with exact payload verification (`probe-transfer-wrapper.log`). The wrapper
restores the caller's previous device on exit. Six registration/transfer unit
tests pass. The current run passed routed completion and chat checks with the fix.

Autotune cache keys now include the PD role to prevent prefill and decode from
writing conflicting rank records when compiled-kernel caches are shared. Four
autotune tests pass, including the role-isolation regression.

## Requested topology

| Resource | Responsibility |
| --- | --- |
| GPUs 0,1 | One model instance, TP2+EP2, prefill only |
| GPUs 2,3 | A second instance, TP2+EP2, decode only |
| CPU RAM | Both instances' immutable Engram embedding tables |
| Router | Sends a request through prefill, state transfer, then decode |

Both workers use DP=1, CP=1 and PP=1. The Engram tables are part of the target
model; they are distinct from the optional DSpark speculation head and from
SGLang's unrelated n-gram speculative-decoding algorithm.

An eight-GPU expansion repeats this four-GPU pair on GPUs 4–7. The original
direct-host path creates a host-table copy per TP group. The new eager staging
path instead shares the original checkpoint pages across all instances.

## Initial audit and blockers (2026-09-27; see live update above)

Audited base revision: `a5b2e2b6a1` on `dsv4.1`. Hardware inventory showed eight
B300 SXM6 GPUs, 275040 MiB HBM each, connected by NV18. The host has
395007940 KiB RAM (about 377 GiB). GPU mode queries returned CC ON, DevTools
OFF and multi-GPU NVLE. These mode queries are **not cryptographic attestation**.

1. **CPU Engram cannot use its current direct-access implementation here.**
   All four selected GPUs report `host_register_supported=0` and
   `pageable_memory_access=0`. A separate 8 KiB `cudaHostRegister` experiment
   returned CUDA error 801 (`cudaErrorNotSupported`, as defined in the installed
   CUDA headers). The default host-table path registers an mmap and has the GPU
   dereference its address in a Triton kernel. Registration is unavailable in
   this configuration. Setting `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=0` would leave
   a pointer the GPU cannot access; it is not a solution on this machine.
2. **Checkpoint download and integrity verification are complete.** The user selected the official
   `deepseek-ai/DeepSeek-V4.1-Flash` repository. Revision
   `dba1be0a40aa45a94ad051997016db3960a90277` is public and contains 510313353565
   bytes (about 510.3 GB), including 48 safetensors shards. `/data` had only
   59.6 GB free; the root disk had about 146 GiB free. Downloaded all 40 non-weight
   files to `/data/DeepSeek-V4.1-Flash`, including config, tokenizer and index.
   The user subsequently expanded the data disk to 1 TiB; `/data` has 1007
   GiB usable and had 834 GiB free before resuming the full download. All 88
   repository files are now downloaded and match the pinned Hugging Face
   checksums. All 96085 tensors also match the index and valid file offsets.
   `/data` has 359 GiB free afterward. Existing models were preserved. The
   config specifies MXFP8 dense weights and FP4 experts.
3. **Two Engram copies do not fit in host RAM.** The actual config has
   384006168 and 384016682 rows, each 256 FP8 values plus eight e8m0 scale bytes.
   One copy is 202758032400 bytes (188.833 GiB); two are **377.666 GiB**, exceeding
   the machine's **376.709 GiB total RAM** before any OS or loader overhead.
   Sharing the immutable table across P/D is required on this VM, or host RAM
   must increase. The current per-TP-group sharing is insufficient.
4. **The installed serving environment is older than this branch.**

| Package | Branch pin | `/home/ubuntu/sglang-env` |
| --- | --- | --- |
| PyTorch | 2.13.0 | 2.11.0 |
| Transformers | 5.12.1 | 5.6.0 |
| FlashInfer | 0.6.18 | 0.6.16rc4 |
| sglang-kernel | 0.4.6.post1 | 0.4.2.post2 |

Mooncake and `sglang_router` are also absent from that environment. The system
Python has PyTorch 2.13.0+cu130 but lacks the serving dependencies. Do not upgrade
the existing M2.7 environment in place; prepare a separate V4.1 environment
using this branch's `python/pyproject.toml` and matching gateway/Mooncake builds.

After the data-disk restart, `nvidia-smi` fails with a driver/library mismatch:
the loaded NVIDIA module is 595.71.05, while installed NVML and the on-disk
module are 595.91.07. Resolve this before GPU testing; the earlier hardware
observations above precede that restart. Downloading weights does not use CUDA.

## Source audit

Paths below are relative to the repository root. Findings describe inspected
source, not a completed live-model qualification.

| Area and source | Finding / deployment consequence |
| --- | --- |
| `python/sglang/srt/configs/deepseek_v41.py` | Normalizes nested text/vision config and Engram dimensions. Read the actual config before budgeting memory. |
| `python/sglang/srt/models/deepseek_v4.py` | Attention divides heads and output groups by attention TP. Several projections are replicated. Weight/KV memory is not uniformly divided by two. |
| `python/sglang/srt/layers/engram.py`, `python/sglang/kernels/ops/embeddings/engram_gather.py` | Host layout is shared per TP group or private per-rank shards. FP8 payload plus one e8m0 byte per 32 values. Lookup reads CPU memory directly from GPU; no CPU-gather/H2D fallback exists. |
| `python/sglang/srt/model_executor/model_runner_components/ngram_embedding_manager.py` | PD PREBUILT batches seed predecessor history from prompt and previous output, excluding the sampled boundary token. Added a regression covering short prompts, stale slots and rebootstrap. |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | DSpark PD requires Mooncake, static verification, DP=1 and CP=1. PP and two-batch overlap are rejected. Encoder SWA bounded replay is rejected with PD. |
| `python/sglang/srt/disaggregation/utils.py`, `common/conn.py` | DSpark peers exchange and compare draft width, compression ratios, KV/state layout and attention TP. Both sides must enable DSpark with matching settings. |
| `python/sglang/srt/disaggregation/decode.py` | DSpark retraction reboots the request through prefill instead of pretending the transferred state can be locally reconstructed. Existing unit tests cover this. |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | Separate packed compressed KV/index pools for ratios 1/2, source-layer sharing, and target/draft SWA state. Account for allocated pools and request slots, not just serialized weights. |
| `python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py` | Blackwell fused low-ratio compression is selected by capability. Retain these existing kernels; no reason to replace them without profiling. |
| `python/sglang/srt/layers/attention/dsv4/candidate_indexer.py` | DeepGEMM candidate indexing is opt-in and defaults to the Torch implementation. The optimized path requires new paged sparse MQA functions and changes index-pool configuration. Qualify it as a separate boot. |
| `python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py` | Dense MXFP8 can auto-select FlashInfer CuTe DSL on Blackwell. MoE auto-selection depends on the checkpoint, including MXFP4 versus NVFP4. Do not force a provider before reading the checkpoint. |
| `python/sglang/srt/models/deepseek_v4.py` Engram prefetch | Optional layer-14 overlap only applies to BS=1 ordinary decode, a shared host table, no vision tower and the CuTe DSL projection provider. It is not a general DSpark/batched-decode optimization. Left off. |
| `python/sglang/srt/disaggregation/mooncake/utils.py`, `utils.py` | Intra-node NVLink mode also changes auxiliary-buffer placement. NV18 topology alone does not prove which transfer route executes or its CC compatibility. |
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | Batch registration previously returned errors that callers ignored. Changed it to raise for failed registration, covering both KV/state and staging callers without changing the separate Ascend engine. |
| `test/registered/models_e2e/test_deepseek_v41.py` | Existing V4.1 sanity is TP4+EP4 at 16K, not TP2 PD qualification. V4 Flash/NIXL tests cannot substitute for V4.1 DSpark/Mooncake validation. |

## Candidate launcher

Download the pinned checkpoint after provisioning sufficient disk space:

```bash
/home/ubuntu/sglang-env/bin/python benchmark/deepseek_v41_pd/download_model.py
```

The downloader targets `/data/DeepSeek-V4.1-Flash`, checks free space before
fetching weights, and uses Hugging Face's resumable local download state.
Use `--local-dir /larger/volume/DeepSeek-V4.1-Flash` for a new mount, or
`--metadata-only` for config/tokenizer/reference files without weights.
`download-manifest.json` records revision and completion state.

No CUDA initialization, checkpoint reads or server launch in dry-run mode:

```bash
cd /home/ubuntu/sglang-private
python3 benchmark/deepseek_v41_pd/launch.py \
  --model /path/to/DeepSeek-V4.1 --dry-run
```

After resolving host-access support, obtaining the checkpoint and installing a
matching environment, inspect preflight first:

```bash
/path/to/v41-venv/bin/python benchmark/deepseek_v41_pd/launch.py \
  --model /path/to/DeepSeek-V4.1 \
  --preflight-only --output-dir /tmp/dsv41-preflight-01
```

Omit `--preflight-only` with a **new output directory** to launch. The script
refuses to overwrite a previous run. It checks B300/GB300 inventory, selected
GPU memory usage, CC mode, ports, dependency pins/imports, configured host-table
memory and actual eager/graph host gathers. It starts prefill, then decode, then
the router, saves both `/server_info` responses and runs a routed generation
smoke. It supervises only its own process groups and never stops other jobs.

The memory check is a necessary lower bound, not proof of fit. Configured host
table bytes per worker are `sum(engram_num_embeddings) * (head_dim + head_dim/32)`.
It budgets two copies plus a 32 GiB host reserve. Checkpoint loader temporaries,
vision weights, replicated projections, DSpark, KV, graphs and kernels still
need measured startup peak and steady-state budgets.

Defaults deliberately start at 16K context, 4K prefill chunks, 64K KV tokens per
worker, 8 prefill requests and 16 decode requests. These are bring-up limits,
not proposed production capacity. Decoder SWA bounded replay is enabled, so
prefill CUDA graphs are explicitly disabled as required by the branch. Decode
graphs remain enabled on the decode worker. Custom all-reduce is disabled for
the NCCL control (`--custom-all-reduce off`, the default). `--custom-all-reduce
on` only drops `--disable-custom-all-reduce` on both workers, as the M3 launchers
do (no env var is involved), and preflight then runs
`probe_custom_all_reduce.py` on every TP2 GPU pair. FlashInfer all-reduce fusion
stays off either way. The probe builds the TP group with the server's own
constructor, so a failed IPC setup shows as the same silent NCCL fallback; it
compares CustomAllReduceV2 with NCCL on a 384 x 5120 BF16 payload (64 requests x
6 DSpark verify positions) and times both. After each TP worker is healthy the
launcher also fails if its log shows that fallback. Run the probe alone:

```bash
CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=python timeout 120 \
  python benchmark/deepseek_v41_pd/probe_custom_all_reduce.py --precompile
```

The first run on a GPU set builds the JIT kernels and the cross-process cudaIpc
peer-access cache (about 45 s under CC); `--precompile` does both under its own
timeout. On 2x B200 under CC (2026-10-15, three runs) it initialized, matched
NCCL bit for bit, and took 10.3-10.4 us per all-reduce inside a CUDA graph
against 49-52 us for NCCL. B300, TP4, the fused TP4 MoE-finalize and mHC paths,
and a served model are unverified.

Worker ports are 8031/8032, router 8030 and bootstrap 8998. All HTTP services bind
to loopback. Compile caches are per run/revision/worker. The launcher clears
inherited SGLANG/SGL/MC/MOONCAKE/NCCL tuning variables and sets the intended
contract explicitly; use its arguments rather than a previous shell's tuning
environment. It does not change THP settings or evict checkpoint page caches.

Relevant A/B arguments:

- `--no-dspark`: disable speculation on **both** workers for the control.
- `--custom-all-reduce off|on`: NCCL control (default) or CustomAllReduceV2;
  see the probe description under the launch defaults.
- `--dspark-block-size N`: gamma; the runtime resolves verify width to N+1.
- `--candidate-indexer deepgemm`: explicit unqualified optimization arm; checks
  that the two required sparse MQA functions are present before launching.
- `--host-layout shared|private|auto`: huge-page-aware host layout; direct access
  must still pass the probe. Shared avoids the lookup all-reduce but does not
  share the table between the two model instances.
- `--transport intra-node-nvlink|rdma|tcp`: explicit transport candidate.
  Intra-node NVLink uses CUDA IPC memory handles and GPU copies. It is the
  default candidate for this topology; verify the
  actual Mooncake route and transfer correctness under CC. TCP is a diagnostic
  control, not a demonstrated performance-equivalent fallback.
- `--devtools-mode off|on`: require the specified existing DevTools mode during
  preflight (default off). This only checks the mode; it never changes GPU mode.
- `--prefill-graph-max-seq-len N`: the longest context the split prefill graphs
  replay (needs `--split-cuda-graphs`). The server caps it at 16,384 by default;
  above that the prefill ratio-1/2 index layers run as eager breaks
  (`SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER`, set on prefill workers by
  this flag) and longer contexts prefill eagerly. Unmeasured on this branch and
  rejected together with `--prefill-bounded-replay`. `--prefill-graph-eager-indexer`
  and `--prefill-hoist-per-batch` from the `dsv4.1` runs fail on purpose.
- `--chunk-size`, `--prefill-requests`, `--decode-requests`,
  `--max-total-tokens`, `--context-length`: sweep one at a time after fit and
  correctness are established.

For a second independent pair use `--gpus 4,5,6,7 --port 8130
--bootstrap-port 9098 --nccl-port 29600` and a separate output directory.
Its CPU memory budget must include the first pair's resident allocation.

### N prefill : 1 decode

`--num-prefill N` launches N TP2 prefill workers feeding one TP2 decode worker;
`--gpus` lists 2N+2 GPUs (prefill pairs first, the last two decode). Prefill i
uses HTTP `--port`+1+i, NCCL `--nccl-port`+i and bootstrap `--bootstrap-port`+i;
decode uses `--port`+1+N and `--nccl-port`+N. Per-role options
(`--prefill-swa-prefix-tails`, `--prefill-graph-max-seq-len`,
`--prefill-requests`) apply to every prefill worker, and prefill workers share one
compile cache. N=1 is the unchanged single pair. For 3P:1D on one node:

```bash
python benchmark/deepseek_v41_pd/launch.py --model /data/DeepSeek-V4.1-Flash \
  --num-prefill 3 --gpus 0,1,2,3,4,5,6,7 --output-dir /tmp/dsv41-3p1d
# prefill 8031-8033 (bootstrap 8998-9000), decode 8034, router 8030
python benchmark/deepseek_v41_pd/oracle_pd_proxy.py --port 8040 \
  --prefill-url http://127.0.0.1:8031@8998 --prefill-url http://127.0.0.1:8032@8999 \
  --prefill-url http://127.0.0.1:8033@9000 --decode-url http://127.0.0.1:8034
```

The router gets every prefill URL with its bootstrap port and `--prefill-policy
manual`, which pins each `X-SMG-Routing-Key` to one prefill worker. The oracle
proxy pins by a SHA-1 of the same header (or of the first user/system message),
sends the chosen worker's bootstrap endpoint to both prefill and decode, and
reports per-worker counts at `GET /stats`. Its default port 8034 collides with the
3P decode worker, so pass `--port`.

## Required engineering and qualification

1. **Implement a CC-compatible Engram staging path.** Keep immutable FP8/e8m0
   tables in host RAM, gather only needed rows on CPU, and copy bounded batches
   through supported CUDA transfers to device buffers. Since hashes depend on
   token IDs rather than hidden states, prefill lookups can be prepared ahead
   of the layer computation. Verify IDs arrive after drafting, so speculation
   needs explicit D2H/H2D completion and buffer-lifetime handling. CUDA graphs
   must consume stable buffers populated before replay; do not insert an
   unsynchronized CPU operation into a captured graph. This path is **not
   implemented by this change** and its throughput is unknown.
2. Share immutable tables across both instances (required by the actual RAM
   budget unless RAM is expanded), with checkpoint
   identity/shape/dtype validation and explicit ownership/lifetime. Current
   `/proc/<pid>/fd` sharing only operates within a TP group. Do not point two
   independent writers at the same file and assume they are synchronized.
3. Establish a target-only TP2 integrated reference, then target-only PD, then
   symmetric DSpark PD. Compare greedy outputs and logits where available.
   Exercise prefix hits, chunk boundaries, odd/even ratio-2 positions, slot
   reuse, retraction, streaming, tool calls, and vision if enabled. Test eager
   and captured decode separately and genuine stochastic sampling.
4. Profile CPU gather/H2D, prefill and decode MoE, indexer, attention, collectives,
   KV/state transfer, queue delay and scheduler gaps. Measure transfer bytes
   and actual transport; require zero registration failures or fallback.
5. Qualify the DeepGEMM candidate indexer against its reference tests and the
   model output. Sweep DSpark gamma, graph batch cap and prefill chunk size.
   Compare bounded replay versus full work using a separately constructed
   control before attributing its speedup.
6. Increase resident context 16K -> 64K -> 128K -> 256K only after earlier tiers
   fit. Record peak HBM/CPU RAM, KV capacity/eviction/retraction, accepted
   tokens/step, prompt/output tokens/s, TTFT/ITL and queue time at each tier.
7. Repeat winning configurations and soak before duplicating the pair on eight
   GPUs. Measure host-memory and interconnect contention after duplication.

Use the actual DeepSeek workload and an agreed latency/throughput target.
The M3 revenue mix and 4.5M TPM/GPU goal are not measurements or acceptance
criteria for this V4.1 deployment. If reporting a whole-eight-GPU node score,
include all eight GPUs in the denominator even while testing one pair.

## Verification and artifacts

```bash
python3 -m unittest discover -s benchmark/deepseek_v41_pd -v
PYTHONPATH="$PWD/python" /home/ubuntu/sglang-env/bin/python -m unittest discover \
  -s test/registered/unit/disaggregation -p test_mooncake_registration.py
PYTHONPATH="$PWD/python" /home/ubuntu/sglang-env/bin/python -m unittest discover \
  -s test/registered/unit/disaggregation -p test_dsv41_dspark_pd.py
PYTHONPATH="$PWD/python" /home/ubuntu/sglang-env/bin/python -m unittest discover \
  -s test/registered/unit/model_executor/model_runner_components \
  -p test_ngram_embedding_manager.py
CUDA_VISIBLE_DEVICES=0 python3 benchmark/deepseek_v41_pd/probe_engram.py
```

Audit artifacts on this host are under `/tmp/dsv41-pd-audit/`: `host-probe.json`,
`preflight/preflight.json`, `preflight/launch-plan.json`, `preflight.log`,
`launcher-tests.log`, `runtime-tests.log`, `download-full.log` and
`download-metadata.log`. These temporary artifacts did not survive the VM reboot
used for the disk expansion. The test/probe results above describe that earlier
run. New download artifacts are on the persistent data disk:
`/data/DeepSeek-V4.1-Flash/download.log`, `download-manifest.json` and
`expected-files.json`. Full checksum results are in `verification.json`;
`header-audit.json` records the 48-shard / 96085-tensor layout check and
`memory-budget.json` records the exact configured Engram payload. The 11 launcher
tests were rerun after adding incomplete/truncated checkpoint rejection; their
new log is `launcher-tests.log` in the model directory.
No full model was loaded, no GPU mode was changed and no end-to-end performance
result is claimed.

The current run uses `--parallel-load --resident-engram-pid 134016` after
validating TP2 fit. Preflight credits only measured read-only locked pages
matching the checkpoint files in that PID's kernel memory maps. Both roles use
`--kernel-cache-dir` pointing at the previous run's prefill cache.
`flashinfer-jit-cache==0.6.18+cu130` is installed from
`https://flashinfer.ai/whl/cu130`; its MoE library was independently loaded
successfully. Install this matching prebuilt cache before first boot to avoid
the lengthy MoE routing source build. The current smoke request timeout is
600 seconds to accommodate any remaining first-request specialization.
