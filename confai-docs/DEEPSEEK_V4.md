# DeepSeek-V4.1 on B300 confidential VMs

This branch serves **DeepSeek-V4.1-Flash** as a one-node prefill/decode (PD) pair: TP2 prefill
and TP2 decode over Mooncake, with the model's Engram tables held in CPU RAM. It runs under
confidential computing (CC ON, multi-GPU NVLE).

Read this page for the DeepSeek-specific facts: how the model shapes serving, how the pair is
deployed, and the optimizations and knobs behind the shipped configuration.

**Status:** the code was ported from the `dsv4.1` and `pp-encoder` bring-up branches. The PD pair
in `recipes/deepseek-v41/container/` is the production configuration. The CED and PP2 prefill
layouts are experimental and have not run on a GPU on this branch. The merge with upstream
rewrote the V4.1 decoder layer (`deepseek_v4_boundary.py`, `deepseek_v4_mhc.py`) and brought
upstream's decoder SWA bounded replay; the CED graphs, the PP pre-mix hand-off and decode/verify
TBO were re-ported onto it.

## At a glance

| | DeepSeek-V4.1-Flash |
| --- | --- |
| Topology | 1 PD pair = TP2+EP2 prefill (2 GPUs) + TP2+EP2 decode (2 GPUs) |
| Speculation | DSpark from the checkpoint, block size 5, 6-token verify |
| Engram | Two FP8 tables in host RAM, one shared locked copy, gathered on the CPU |
| Deployment | [`recipes/deepseek-v41/container/`](../recipes/deepseek-v41/container/README.md): image, compose file and systemd units |
| Launcher | [`launch.py`](../recipes/deepseek-v41/launcher/launch.py), wrapped by the [recipes](../recipes/deepseek-v41/README.md) |
| Operator guide | [`Agent_Readme.md`](../recipes/deepseek-v41/Agent_Readme.md) |

## 1. The model as it matters for serving

These facts come from the [V4.1 cookbook page](../docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx)
and the bring-up work. The config class is `DeepseekV41Config` in
`python/sglang/srt/configs/deepseek_v41.py`. The runtime maps V4.1 onto the V4 model,
`DeepseekV4ForCausalLM` in `python/sglang/srt/models/deepseek_v4.py`.

- **Layers and experts.** 40 target layers. 384 routed experts at top-6, plus one shared
  expert. Dense weights are MXFP8 (FP8 with a 32-wide UE8M0 block scale). Routed experts are
  MXFP4. The pinned checkpoint is `dba1be0a40aa45a94ad051997016db3960a90277`: 510.3 GB in 48
  shards.
- **Attention.** Each layer keeps one 512-wide latent per position. That latent serves as both
  key and value for all 64 query heads. A layer reads two stores:
  - compressed latents, produced only at **KV-source layers** and shared forward. They
    compress 2:1 in layers 2–19 and 1:1 from layer 20 on.
  - a **128-position sliding window (SWA)** that each layer computes from its own
    activations.

  There are four KV-source layers and eight index-source layers. Each retrieving layer picks a
  top-512 candidate set. **Layer 20 is the last shared-KV source, not the end of the model.**
- **KV bytes.** About **1.6 KB/token**: only the four KV-source layers own compressed latents
  (584 B/entry) and FP4 index keys (68 B/entry), so 8.4M tokens use about 14 GB/GPU. The
  launcher passes `--kv-cache-dtype fp8_e4m3` and `--page-size 256`.
- **mHC (manifold-constrained hyper-connections).** The residual stream is four parallel copies
  (`hc_mult=4`) mixed by a per-token doubly stochastic matrix. Each sublayer's `pre`
  coefficients are consumed by the next sublayer. Under pipeline parallelism, `prev_pre`
  therefore has to cross the stage boundary along with the hidden states. The kernels are in
  `python/sglang/kernels/ops/layernorm/mhc.py` (`_hc_mix_stats_partial_kernel`,
  `_hc_mix_reduce_sinkhorn_kernel`, `hc_mix_stats_sinkhorn`) and `mhc_post_split_h.py`.
- **Engram.** An additive n-gram hash memory sits at **layers 1 and 14** (zero-based). Its two
  FP8 tables have 384,006,168 and 384,016,682 rows. Each row holds 256 FP8 values plus 8 UE8M0
  scale bytes. One copy is **188.833 GiB**, and two copies (377.666 GiB) exceed the VM's
  376.709 GiB of RAM, so every worker must share one copy. The table is part of the target
  model; it is unrelated to DSpark and to SGLang's n-gram speculative decoding.
- **DSpark.** The draft head is bundled in the target checkpoint
  (`python/sglang/srt/models/deepseek_v4_dspark.py`). Block size 5 means five proposals and a
  six-position verification window.

## 2. How V4.1 is served

### The PD pair

| Resource | Role |
| --- | --- |
| GPUs 0–1 | Prefill, TP2+EP2, DSpark |
| GPUs 2–3 | Decode, TP2+EP2, DSpark |
| CPU RAM | Both Engram tables, one shared locked copy |
| Router (`smg`, PD mode) on 8030 | Prefill, then Mooncake KV transfer, then decode |

These are the launcher defaults. The container deployment runs its pair on GPUs 4–7 with router
8040, prefill 8041, decode 8042, bootstrap 8999, NCCL 29600 and metrics 29001
(`recipes/deepseek-v41/container/dsv41.env`).

The router, prefill, decode and prefill bootstrap bind `127.0.0.1`. The router's Prometheus
listener does not: `launch.py` passes no `--prometheus-host`, and `sglang_router` defaults it to
`0.0.0.0`.

**Losing either worker takes down the pair.** The launcher tears down every PID it owns when
any one of them exits.

`python/sglang/srt/arg_groups/deepseek_v4_hook.py` enforces the V4.1 constraints:

- **DSpark with PD** requires static verify, the Mooncake backend, DP=1 and CP=1. Any other
  `--disaggregation-transfer-backend` is rejected on a DSpark PD worker.
- **Rejected for V4.1:** two-batch overlap, HiSparse, the unified KV layout, the trtllm DSv4
  attention backend, and speculation other than DSpark.
- **Pipeline parallelism** is allowed only on a prefill worker with
  `SGLANG_OPT_DSV41_CED_SPLIT` or `SGLANG_OPT_DSV41_PREFILL_TAIL` set. DSpark itself
  (`speculative_hook.py`) also allows PP only on a PD prefill worker.
- **`--enable-decoder-swa-bounded-replay`** cannot combine with DP attention, with any prefill
  graph backend other than `breakable`, or with a breakable prefill graph under CP.
- **Prefill graph length.** If the prefill graph has no `max_seq_len`, the hook sets 16,384 and
  logs that longer contexts run eager prefill. With `SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER=1`
  it leaves the cap unset. An explicit `max_seq_len` above the candidate window (16,384
  positions) is rejected unless that variable is set; under context parallelism the variable is
  ignored and the 16,384 default stays.

Retraction on a DSpark decode worker reboots the request through prefill
(`python/sglang/srt/disaggregation/decode.py`). For DSpark PD, the peers compare the DSpark
layout (block size, target/draft KV layout) and attention TP at bootstrap
(`python/sglang/srt/disaggregation/common/conn.py`). The attention-TP check is skipped for the
CED split and the prefill tail, so a TP1 x PP2 prefill can feed a TP2 decode.

**Transfer backend.** `launch.py` always passes `--disaggregation-transfer-backend mooncake`.
`--transport` (default `intra-node-nvlink`) only selects Mooncake's path through environment
variables: `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK`, `MC_INTRANODE_NVLINK=true` and
`MOONCAKE_PROTOCOL=rdma`; `rdma` sets only the protocol; `tcp` sets `MC_FORCE_TCP=1`.

Mooncake's intra-node path uses CUDA IPC handles over NVLink internally. Three bring-up fixes
matter:

- **Mooncake transfer threads.** TP1 executor threads defaulted to device 0, and
  `cudaMemcpyBatchAsync` failed. The wrapper in
  `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` now selects
  the rank's device.
- **Registration errors.** Batch registration failures now raise instead of being ignored.
- **Autotune cache keys.** Keys include the PD role, so prefill and decode cannot overwrite each
  other's records when they share a compile cache
  (`python/sglang/srt/model_executor/runner/flashinfer_autotune.py`).

**Any `MC_FORCE_TCP` value forces TCP, including `MC_FORCE_TCP=0`.** The launcher clears
inherited `SGLANG_*`, `SGL_*`, `MC_*`, `MOONCAKE_*` and `NCCL_*` tuning variables.

### Launcher and recipes

`recipes/deepseek-v41/launcher/launch.py` supports `--dry-run` and `--preflight-only`. Preflight
checks:

- the checkpoint (V4.1 config, complete shards) and the Engram byte budget;
- that every assigned GPU is an idle B300/GB300 (at most 256 MiB used);
- `nvidia-smi conf-compute`: CC ON, multi-GPU mode NVLE, and DevTools mode equal to
  `--devtools-mode` (default `off`). These are mode checks only, not attestation;
- host RAM for the Engram pages, crediting a `--resident-engram-pid` holder;
- free ports, dependency pins from `python/pyproject.toml`, and imports (including Mooncake and
  `sglang_router`);
- a per-GPU Engram access probe (`probe_staged_engram.py` under `--cpu-engram-staging`).

The launcher then starts prefill, decode and router. It saves both `/server_info` responses and
requires a routed smoke completion. It refuses to reuse an output directory. It always passes
`--enforce-disable-flashinfer-allreduce-fusion`, and passes `--disable-custom-all-reduce` unless
`--custom-all-reduce on`; preflight then probes CustomAllReduceV2 on each TP2 GPU pair with
`probe_custom_all_reduce.py`. The production configuration uses `--custom-all-reduce on`.

Every worker gets `--attention-backend dsv4`, `--kv-cache-dtype fp8_e4m3`, `--page-size 256`,
`--moe-a2a-backend none`, `--random-seed 0`, and the environment
`SGLANG_RAGGED_VERIFY_MODE=static` and `SGLANG_RUST_SERVER=0`. DSpark (`--speculative-algorithm
DSPARK`, block size 5) is on unless `--no-dspark`.

The [recipes](../recipes/deepseek-v41/README.md) are thin wrappers.
[`_common.sh`](../recipes/deepseek-v41/_common.sh) is sourced by each one and sets the shared
defaults: GPUs 4–7, router 8040 (prefill 8041, decode 8042), bootstrap 8999, NCCL 29600 (decode
29601), router metrics 29001. It also sets a timestamped output directory under
`/data/dsv41-bringup/recipes/`. All three recipes pass `--cpu-engram-staging --split-cuda-graphs`
and use the Mooncake backend with the default intra-node NVLink transport.

| Recipe | Prefill (GPUs 4–5) | Decode (GPUs 6–7) | Feature env, both workers | Status |
| --- | --- | --- | --- | --- |
| [`tp2-pd.sh`](../recipes/deepseek-v41/tp2-pd.sh) | TP2+EP2, all 40 layers, DSpark | TP2+EP2, DSpark | None | Production layout; the container runs it |
| [`ced-pp2.sh`](../recipes/deepseek-v41/ced-pp2.sh) | TP1 x PP2 (EP1), encoder layers 0–20, partition `8,13`, no DSpark | TP2+EP2, DSpark, prompt-tail replay | `SGLANG_OPT_DSV41_CED_SPLIT=1` | Experimental; ran on `pp-encoder`, not on this branch |
| [`prefill-tail-pp2.sh`](../recipes/deepseek-v41/prefill-tail-pp2.sh) | TP1 x PP2 (EP1), all 40 layers, partition `14,26`, DSpark on the last stage, upstream bounded replay of layers 20–39 | TP2+EP2, DSpark | `SGLANG_OPT_DSV41_PREFILL_TAIL=1` | Experimental. No GPU run recorded on any branch |

The PP recipes get `SGLANG_PP_LAYER_PARTITION` on prefill only, from `--prefill-pp-partition` or
the defaults above. `--ced-split` and `--prefill-bounded-replay` are mutually exclusive, and
`--prefill-bounded-replay` requires `--split-cuda-graphs`.

**The recipes do not encode the serving settings.** `tp2-pd.sh` passes only
`--cpu-engram-staging --split-cuda-graphs`, and the launcher defaults are bring-up limits
(16,384 context, 65,536 KV tokens, 8 prefill and 16 decode requests, `--mem-fraction 0.80`). The
production values are `DSV41_LAUNCH_ARGS` in `recipes/deepseek-v41/container/dsv41.env`, with
the reasoning for each in that file's comments: 131,072 context, 8,192-token chunks, 16 prefill
requests under HRRN, 64 decode requests, 256 prefill SWA tails, a 1.5M-token prefill pool with a
4x HiCache host tier, a 4M-token decode pool, custom all-reduce, prefill graphs to 65,536 tokens,
Engram prefetch with 8 gather threads, NCCL LL128 on prefill and four tokenizer workers.

`--num-prefill N` launches N TP2 prefill workers feeding one TP2 decode worker, for example
3P:1D on eight GPUs. The router then uses `--prefill-policy manual`.

### Host-side requirements

- The deployment runs in containers: `recipes/deepseek-v41/container/` builds the `dsv41-pd`
  image (the frozen `/data/venvs/dsv41` environment) and runs it as two enabled systemd units.
- `dsv41-engram-cache.service` runs the holder (`hold_engram_cache.py`), which must outlive every
  pair; `dsv41-pd.service` runs the pair and passes the holder's PID as `--resident-engram-pid`.
- Host-venv launches still need `LD_LIBRARY_PATH=.../site-packages/nvidia/cuda_runtime/lib`,
  because Mooncake's wheel links CUDA 12; the image sets it.
- The `nvidia-persistenced` override, where present, is transient. A reboot loses it.

## 3. DeepSeek-specific optimizations

### Prefix-tail retention

Each layer's SWA is per request and cannot be shared, so prefix caching needs a cached SWA
"tail" for every retained prefix. Prefill reserves **32 tails by default** (four times its eight
running-request slots), fewer than the live conversations of a multi-turn agent workload. Tails
are then evicted and the cache hit rate collapses.

`--prefill-swa-prefix-tails 256` passes `--swa-prefix-tails` to prefill. It grows the prefill SWA
pool from 30,208 to 116,224 tokens (384 per tail). In the bring-up this lifted the prefix-cache
hit rate from about 56% to 95% and was the largest single throughput lever. Raising the
full-token cap alone does not help: full-token and SWA pools are separate.

The next limit is the prefill full-token pool. About 96 live 40K-token conversations need about
3.8M tokens of prefix cache; a 2.1M pool dropped the hit rate to about 80%. The production pool is
1.5M tokens on the GPU plus a HiCache host tier four times that size, so cache-thrash past the
GPU pool reloads KV from host instead of recomputing it.

### CPU Engram staging

The default host-table path registers an mmap with `cudaHostRegister` and has a kernel
dereference it. On this CC VM, that fails:

- every GPU reports `host_register_supported=0` and `pageable_memory_access=0`;
- `cudaHostRegister` returns error 801.

The host-table path always calls `cudaHostRegister` (`python/sglang/srt/layers/engram.py`).

`--cpu-engram-staging` (`SGLANG_DSV41_ENGRAM_CPU_STAGING`) works around this
(`StagedEngramTable` in `python/sglang/srt/layers/engram_staging.py`). The CPU gathers the
selected FP8 rows from the shared read-only checkpoint mapping, then copies only those compact
rows to the GPU. Every worker shares one set of page-cache pages: 203,073,077,576 bytes, locked
by the holder.

`StagedEngramTable._stage_bytes` is an eager break inside otherwise captured graphs. It cannot
be removed: the n-gram hashes depend on token IDs, which must reach the CPU first. In a decode
profile (DSpark, c64) the CPU row gathers cost about 5.7 ms per iteration per rank.

`SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH=1` (default off; on in production) moves the gathers
off the critical path. A new break right after the hash copies the ids of both Engram layers to
the host in one transfer. Small steps (decode, verify) then gather and copy every layer's rows
at once; large prefill chunks gather on a worker thread in layer order while the GPU runs the
layers in between, so each layer's break only waits for its finished rows. Rows are matched to
the ids tensor they were gathered for and anything else falls back to the blocking gather. The
bridge buffers stay static, and rows are bit-identical to the blocking path
(`probe_staged_engram.py`). `SGLANG_DSV41_ENGRAM_GATHER_THREADS` sets the OpenMP threads of that
worker (0 keeps the process-wide value).

### Split CUDA graphs and long-context prefill

`--split-cuda-graphs` uses the breakable graph backend through an explicit
`--cuda-graph-config`. Prefill captures powers of two from 256 tokens up to the chunk size, plus
the chunk size itself. Decode captures powers of two up to the request limit, plus the limit
itself and any `--decode-graph-extra-bs`. Because the prefill backend is set explicitly, PP2
prefill keeps its graphs; `cuda_graph_hook.py` otherwise disables breakable prefill graphs under
PP.

- **Short prompts.** Prompts under the 256-token bucket run eager prefill.
- **Late-layer trimming.** Decoder SWA bounded replay is upstream's implementation: the tail
  starts at the last kv_source layer (20), whose compressed KV is first written for every row,
  and prefill graphs run it as a fixed-capacity tail block. Upstream defaults it on, so the
  launcher passes it explicitly: on for decode, off for prefill
  (`--no-enable-decoder-swa-bounded-replay`), as in production. `--prefill-bounded-replay`, or
  turning split graphs off (which disables both prefill and decode graphs), enables it on
  prefill too. A CED-split prefill keeps it off.
- **Verification.** `TARGET_VERIFY` satisfies `is_extend()`, so the model excludes it from
  prefill-only graph breaks.

**Prefill graphs stop at 16,384 tokens by default.** The captured low-ratio indexer scores a
static context width, so the hook defaults prefill `max_seq_len` to 16,384. With long
agent-workload prompts every chunk then runs eager and host-bound.
`SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER` lifts the cap: the ratio-1/2 index layers of the
breakable prefill graph run as an eager break on the live rows. The projection break still
computes the queries and head weights, and the break hands them to the same candidate indexer
that serves eager prefill, so contexts past the window use candidate blocks as they do eagerly.
Everything else stays captured. Enable it with the launcher's `--prefill-graph-max-seq-len N`
(with `--split-cuda-graphs`); production uses 65,536. The launcher's
`--prefill-graph-eager-indexer` and `--prefill-hoist-per-batch` fail on purpose and point to
that flag. Not validated with `--prefill-bounded-replay` or context parallelism.

### CED split and tail graphs (experimental)

In the CED split, prefill builds and runs only the **encoder layers 0–20**, through the last
KV-source layer. It hands their KV and a wider SWA window to decode. Decode then replays each
prompt's tail through every layer (decoder SWA bounded replay). That replay seeds:

- the late-layer windows;
- the first token;
- the DSpark draft ring.

Late layers 21–39 own no compressed KV, so prefill can skip them.

Set `SGLANG_OPT_DSV41_CED_SPLIT=1` on both workers, or use the launcher's `--ced-split`. The
prefill side runs TP1 x PP2 with partition `8,13` and no DSpark.

**The cost moves to decode.** The eager tail replay is host-dispatch bound, not GPU bound. Tail
graphs reduce the launch cost:

| Feature | Knob or file |
| --- | --- |
| Encoder tail graphs | `SGLANG_DSV41_CED_TAIL_ENCODER_GRAPH`, `..._ENCODER_EXACT_ROWS` (`ced_encoder_graph.py`) |
| Late-layer tail graphs | `SGLANG_DSV41_CED_TAIL_LATE_GRAPH` (`ced_tail_graph.py`) |
| Tail coalescing | `SGLANG_DSV41_CED_TAIL_BATCH_SIZE`, `..._MAX_WAIT_MS` |
| Static graph retention | `ced_graph_cache.py` |

Padded buckets change GEMM results, so only exact-row encoder graphs are used. In the bring-up
CED stayed behind the plain TP2 pair.

**How to check CED:**

- `SGLANG_DSV41_CED_TAIL_GRAPH_CHECK` cross-checks the first replay per shape against eager.
  `..._CHECK_EVERY` checks every replay.
- `SGLANG_DSV41_CED_TAIL_TIMING` emits non-synchronizing tail timing.
- CED options can be changed live through the decode worker's `/set_internal_state`.
  `python/sglang/srt/disaggregation/ced_options.py` validates them on all TP ranks.

### Pipeline-parallel prefill (experimental)

PP2 exists only as a prefill-side layout for the two split modes. It puts TP1 on each of two
GPUs, with `SGLANG_PP_LAYER_PARTITION` set by `--prefill-pp-partition`. Stages must split at
KV-source layers.

Bring-up exposed two bugs, both fixed:

- **Missing layer IDs.** Ratio-2 KV/index entries did not publish layer IDs, which crashed
  Mooncake's transfer thread with an `IndexError`.
- **Dropped `prev_pre`.** The prefill graph buffers omitted the mHC `prev_pre` coefficients, so
  graph replay could silently discard them. They are now carried as float32
  `[tokens, hc_mult]`; see `test/registered/unit/models/test_deepseek_v4_pp_prev_pre_handoff.py`.

`SGLANG_OPT_DSV41_PREFILL_TAIL` is the alternative layout: PP2 prefill over all 40 layers
(partition `14,26`), the late layers over only the bounded prompt tail, and DSpark on the last
stage. The model side is upstream's bounded replay (including its prefill-graph tail block); the
flag selects the PD transfer layout and the last-stage draft worker. No GPU run is recorded on
any branch.

### DSpark on DeepSeek

DSpark runs on both PD roles with identical settings. `--no-dspark` is the matched control.
Target and draft graph replay are both captured.

### Decode kernels

In a decode profile (40 speculative iterations at c64, DSpark), routed-expert MXFP4 GEMMs were
about 42% of kernel time, NCCL all-reduce about 14%, mHC about 12%, dense GEMM about 10% and
sparse MLA about 8%. At EP2 each GPU still holds 192 experts, and batch 64 x 6 verify positions
x top-6 averages only about 6 rows per expert, so expert GEMMs are bandwidth bound.

`SGLANG_DSV41_FUSED_WO_A` (default on) collapses the BF16 `wo_a` verify chain into one
megakernel on SM100/SM103. DeepGEMM HC prenorm (`SGLANG_OPT_DEEPGEMM_HC_PRENORM`, default on)
computes the mHC prenorm. DeepGEMM MegaMoE (`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`) is off and
unqualified here.

### KV and index formats

- **V4.1 deployment.** FP8 KV (`fp8_e4m3`) with FP4 index keys.
- **Layout switch.** `SGLANG_DSV4_KV_LAYOUT` defaults to `v4` (584 B/token per layer). `v41`
  selects the SM100 FlashMLA V4.1 formats: a 528 B FP8 SWA cache plus FP8 or FP4 compressed
  caches, controlled by `SGLANG_DSV4_COMPRESSED_KV_LAYOUT`. The launcher does not set the
  switch.

## 4. Open problems

1. **Prefill capacity.** Prefill limits throughput, and cache misses queue behind each other.
   Faster prefill kernels or a split with more prefill than decode GPUs (`--num-prefill N`) are
   the remaining levers.
2. **Real DSpark acceptance under DevTools OFF** is the number to qualify against.
3. **Isolate PD handoff cost.** Correlate request IDs across router and workers, and record
   transfer bytes and durations.
4. **CED.** Reduce eager indexer and attention scaffolding in the tail replay, and cover more
   exact shapes without large capture memory. Until then, CED stays behind the plain TP2 pair.
5. **Operations before any exposure:** auth, TLS, worker metrics and an admission cap. See the
   [operator guide](../recipes/deepseek-v41/Agent_Readme.md#before-go-live).

## 5. Knob reference

All of these are defined in `python/sglang/srt/environ.py` unless marked as a launcher flag.

| Knob | Default | Purpose |
| --- | --- | --- |
| `SGLANG_DSV41_ENGRAM_CPU_STAGING` | off | Eager CPU Engram gather (`--cpu-engram-staging`). Takes precedence over the host table |
| `SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH`, `SGLANG_DSV41_ENGRAM_GATHER_THREADS` | off, 0 | Gather both staged Engram layers from one id copy (all at once for decode/verify, on a worker thread for prefill); OpenMP threads of that worker (0 = process-wide torch value) |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | off, `shared` | Host-table path; needs host registration, which this CC VM lacks. The launcher always sets the first to `1` and the layout to `--host-layout` (default `auto`). `engram.py` accepts only `shared` or `per_rank`, so a launch without `--cpu-engram-staging` fails at model load unless `--host-layout shared` is passed |
| Set by the launcher, read by nothing | | `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN`, `SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE`, `SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH`, `SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER`. `--candidate-indexer deepgemm` therefore only adds DeepGEMM import checks to preflight |
| `SGLANG_RAGGED_VERIFY_MODE` | `static` | The launcher sets `static`; the DSpark PD hook requires it |
| `SGLANG_OPT_DSV41_CED_SPLIT` | off | CED split, set on both workers |
| `SGLANG_OPT_DSV41_PREFILL_TAIL` | off | PP2 prefill-side tail; exclusive with CED |
| `SGLANG_DSV41_CED_TAIL_{ENCODER_GRAPH,LATE_GRAPH,BATCH_SIZE,MAX_WAIT_MS}` | off, off, 1, 1000 | CED tail graphs and coalescing |
| `SGLANG_DSV41_CED_TAIL_GRAPH_CHECK[_EVERY]`, `SGLANG_DSV41_CED_TAIL_TIMING` | off | CED verification and timing |
| `SGLANG_OPT_DSV41_ASYNC_TAIL_METADATA` | off | A/B for pinned uploads in tail metadata |
| `SGLANG_DSV41_TORCH_PREFILL_INDEXER` | off | Torch ratio-1/2 prefill indexer (fallback) |
| `SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER` | off | Eager break for the prefill graph's ratio-1/2 index layers; lifts the 16,384-token prefill graph cap. Set by `--prefill-graph-max-seq-len N` above 16,384 |
| `SGLANG_DSV41_FUSED_WO_A` | on | `wo_a` verify megakernel (SM100/SM103) |
| `SGLANG_DSV4_KV_LAYOUT`, `SGLANG_DSV4_COMPRESSED_KV_LAYOUT` | `v4`, `auto` | KV pool layout |
| `SGLANG_DSV4_COMPRESS_STATE_DTYPE` | `float32` | V4 compressed-state dtype |
| `SGLANG_OPT_FUSE_C4_FP4_STORE` | on | Fused C4 FP4 index-cache store |
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` | off | MegaMoE; unqualified here |
| `SGLANG_PP_LAYER_PARTITION` | unset | Set by `--prefill-pp-partition` |
| Launcher: `--prefill-swa-prefix-tails`, `--decode-max-total-tokens`, `--num-prefill`, `--candidate-indexer`, `--transport`, `--devtools-mode` | | See the [launcher README](../recipes/deepseek-v41/launcher/README.md#launcher-arguments) and `launch.py --help` |

## 6. Further reading

- [recipes/deepseek-v41/launcher/README.md](../recipes/deepseek-v41/launcher/README.md): launcher usage,
  arguments, preflight, Engram residency and N:1 prefill.
- [recipes/deepseek-v41/README.md](../recipes/deepseek-v41/README.md): recipe usage, port and
  environment overrides.
- [recipes/deepseek-v41/container/README.md](../recipes/deepseek-v41/container/README.md): the
  container deployment.
- [recipes/deepseek-v41/Agent_Readme.md](../recipes/deepseek-v41/Agent_Readme.md): operator and
  agent guide, including hard rules, diagnosis, recovery and go-live gaps.
