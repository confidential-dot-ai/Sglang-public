# DeepSeek V4 on B300 confidential VMs

This fork has two DeepSeek V4 lines:

- **DeepSeek-V4.1-Flash** is the active work. It runs as a one-node prefill/decode (PD) pair:
  TP2 prefill and TP2 decode over Mooncake. The CPU holds the model's Engram tables. It is a
  **bring-up and benchmark deployment, not a qualified one**.
- **DeepSeek-V4-Flash-0731** with its bundled DSpark head is **dormant**. Its B300/CC tooling
  is on `cc-main`, but no B300/CC result was ever recorded for it.

Read this page for the DeepSeek-specific facts.

**Status as of this page:** no DeepSeek number has been measured on `cc-main`. Every V4.1
result below comes from the `dsv4.1` or `pp-encoder` branches, or from local commits. The V4.1
code was ported to `cc-main` in `8e98e4cf81` (2026-10-09). That commit records
"Not yet run on GPU". Some experiments behind the best numbers were not ported (see
[What did not reach cc-main](#what-did-not-reach-cc-main)).

## At a glance

| | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash-0731 |
| --- | --- | --- |
| Status | Active bring-up; unqualified | Dormant since 2026-08-19 |
| Topology | 1 PD pair = TP2+EP2 prefill (2 GPUs) + TP2+EP2 decode (2 GPUs) | Planned TP4, then 2xTP4 integrated |
| Speculation | DSpark from the checkpoint, block size 5, 6-token verify | DSpark from the checkpoint, `gamma=5`, 6-token verify |
| Best measured | 3,325,279 served TPM/GPU (B300, CC ON, DevTools ON; 4-GPU denominator; oracle commit 6; `dsv4.1` branch; not reproducible on `cc-main`) | None |
| Launch | [`recipes/deepseek-v41/`](../recipes/deepseek-v41/README.md) wrapping [`launch.py`](../benchmark/deepseek_v41_pd/launch.py) | [`benchmark/deepseek_v4_b300/`](../benchmark/deepseek_v4_b300/README.md) |
| Operator guide | [`Agent_Readme.md`](../recipes/deepseek-v41/Agent_Readme.md) | None |

## 1. The model as it matters for serving

### DeepSeek-V4.1-Flash

These facts come from the [V4.1 cookbook page](../docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx)
and the bring-up reports. The config class is `DeepseekV41Config` in
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
- **KV bytes.** The real-mix report gives about **1.6 KB/token**: only the four KV-source
  layers own compressed latents (584 B/entry) and FP4 index keys (68 B/entry). It reports that
  8.4M tokens use about 14 GB/GPU. The launcher passes `--kv-cache-dtype fp8_e4m3` and `--page-size 256`.
- **mHC (manifold-constrained hyper-connections).** The residual stream is four parallel copies
  (`hc_mult=4`) mixed by a per-token doubly stochastic matrix. Each sublayer's `pre`
  coefficients are consumed by the next sublayer. Under pipeline parallelism, `prev_pre`
  therefore has to cross the stage boundary along with the hidden states. The kernels are in
  `python/sglang/kernels/ops/layernorm/mhc.py` (`_hc_mix_stats_partial_kernel`,
  `_hc_mix_reduce_sinkhorn_kernel`, `hc_mix_stats_sinkhorn`) and `mhc_post_split_h.py`.
- **Engram.** An additive n-gram hash memory sits at **layers 1 and 14** (zero-based). Its two
  FP8 tables have 384,006,168 and 384,016,682 rows. Each row holds 256 FP8 values plus 8 UE8M0
  scale bytes. One copy is **188.833 GiB**, and two copies (377.666 GiB) exceed the VM's
  376.709 GiB of RAM. The table is part of the target model; it is unrelated to DSpark and to
  SGLang's n-gram speculative decoding.
- **DSpark.** The draft head is bundled in the target checkpoint
  (`python/sglang/srt/models/deepseek_v4_dspark.py`). Block size 5 means five proposals and a
  six-position verification window.

### DeepSeek-V4-Flash-0731

The [V4 cookbook page](../docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) describes the
architecture:

- hybrid CSA + HCA compressed attention, with C4 and C128 compressed state;
- a C4 indexer and mHC;
- FP4 experts with FP8 attention and dense layers;
- a bundled DSpark head (`gamma=5`).

Its FlashMLA KV cache is **584 bytes/token per layer**: 448 FP8 values,
seven UE8M0 scales plus padding, and 64 BF16 RoPE values.

### What differs from MiniMax M3

| Aspect | MiniMax M3 | DeepSeek-V4.1-Flash |
| --- | --- | --- |
| Weights | NVFP4 | MXFP8 dense, MXFP4 experts (32-element scale blocks) |
| Attention and KV | MSA sparse attention; FP4 KV plus FP8 index KV | Shared 512-wide latent (K = V); compressed KV only at 4 source layers; per-layer 128-token SWA |
| Residual | Ordinary residual | mHC, 4 streams with Sinkhorn mixing |
| Extra state | None | 189 GiB Engram tables per copy, held in host RAM |
| Serving unit | Two integrated TP4+EP4 replicas | One TP2 prefill + TP2 decode PD pair |
| Draft | DSpark | Bundled DSpark |

None of the M3 MSA kernels, FP4 KV format, or MiniMax all-reduce tactics transfer. The CC
preflight, the replay scorecard and the tuner framework transfer; the kernels and tactics do
not.

## 2. Goal, workload and current numbers

**No DeepSeek qualification target has been approved.** The M3 revenue mix and its 4.5M
TPM/GPU goal are measurement discipline, not an acceptance criterion for V4.1 or V4-Flash. The
V4-Flash scorecard leaves throughput thresholds at zero for this reason.

The V4.1 real-mix work replays a **different dataset from the M3 primary workload**:

- **Dataset.** `oracle-realmix-256-r67-m3-modeled-long25-38k-v3.json` (r67), not the r132 file
  of the M3 workload. It has 256 conversations and 6,400 turns. It is a synthetic mix modeled on
  agent traces.
- **Tokenization.** Forced completions were decoded from MiniMax tokens and re-encoded with the
  DeepSeek tokenizer.
- **Oracle commits.** Commits used six-token oracle forcing (`--oracle-commit-length 6`). The
  draft forward is real, but verification inputs are teacher-forced. **Real acceptance is
  unmeasured.** One unforced sample accepted about 4.0 tokens per iteration.
- **Denominator.** Four GPUs: one pair, prefill included. An eight-GPU node score would have to
  divide by eight.
- **Harness bands.** Cache hit 94–96% and TTFT 3 s. These are not the M3 95.5–96.5% band.
- **CC mode.** CC ON with multi-GPU NVLE on 8x B300 SXM6, driver 595.91.07. **DevTools was ON
  for every DSpark run and profile.** Qualification requires DevTools OFF.

Current best numbers (all from a single four-GPU pair on the `dsv4.1` branch, B300, CC ON with
NVLE, DevTools ON; none from `cc-main`; details in [section 6](#6-results-and-caveats)):

- **3,325,279 served TPM/GPU** at c72, cache hit 94.9%, TTFT p50 2.28 s, zero failures. It
  needs two experiments: the per-batch indexer hoist, which is not on `cc-main`, and long-context
  prefill graphs, which are on `cc-main` behind `SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER`
  and unmeasured there.
- **2,885,052** at c72 without the hoist. This still used long-context prefill graphs.
- **161,237 output TPM** for the pair in a short synthetic run: 1,024 input / 256 output tokens
  per request, c64, DSpark on (2026-09-28).

## 3. How V4.1 is served

### The PD pair

| Resource | Role |
| --- | --- |
| GPUs 0–1 | Prefill, TP2+EP2, DSpark, at most 8 running requests |
| GPUs 2–3 | Decode, TP2+EP2, DSpark, at most 64 running requests (operator-guide value; launcher default 16) |
| CPU RAM | Both Engram tables, one shared locked copy |
| Router (`smg`, PD mode) on 8030 | Prefill, then Mooncake KV transfer, then decode |

The router (8030), prefill (8031), decode (8032) and prefill bootstrap (8998) bind
`127.0.0.1`. The router's Prometheus listener on 29000 does not: `launch.py` passes no
`--prometheus-host`, and `sglang_router` defaults it to `0.0.0.0`. GPUs 4–7 can hold a second
pair or a PP2 experiment. The recipes default to GPUs 4–7 on ports 8040–8042, bootstrap 8999,
NCCL 29600 and metrics 29001.

**Losing either worker takes down the pair.** The launcher tears down every PID it owns when
any one of them exits.

`python/sglang/srt/arg_groups/deepseek_v4_hook.py` enforces the V4.1 constraints on `cc-main`:

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
`deepseek_v4_hook.py` requires `mooncake` for DSpark PD.

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

The real-mix sweeps did not use the Rust router: it drops `forced_token_ids`. They used
`benchmark/deepseek_v41_pd/oracle_pd_proxy.py`, a benchmark adapter that is not a production
router.

### Launcher and recipes

`benchmark/deepseek_v41_pd/launch.py` supports `--dry-run` and `--preflight-only`. Preflight
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
`--custom-all-reduce on` (opt-in; preflight then probes CustomAllReduceV2 on each TP2 GPU pair with
`probe_custom_all_reduce.py`). Neither collective fast path has been qualified end to end under CC
here.

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
| [`tp2-pd.sh`](../recipes/deepseek-v41/tp2-pd.sh) | TP2+EP2, all 40 layers, DSpark | TP2+EP2, DSpark | None | Control layout. Not run on GPU on `cc-main`; the real-mix rows came from `launch-oracle.sh` on `dsv4.1` |
| [`ced-pp2.sh`](../recipes/deepseek-v41/ced-pp2.sh) | TP1 x PP2 (EP1), encoder layers 0–20, partition `8,13`, no DSpark | TP2+EP2, DSpark, prompt-tail replay | `SGLANG_OPT_DSV41_CED_SPLIT=1` | Experimental. Ran on `pp-encoder` (2026-09-29, 1.457M at c56); not run on `cc-main` |
| [`prefill-tail-pp2.sh`](../recipes/deepseek-v41/prefill-tail-pp2.sh) | TP1 x PP2 (EP1), all 40 layers, partition `14,26`, DSpark on the last stage, bounded replay of layers 21–39 | TP2+EP2, DSpark | `SGLANG_OPT_DSV41_PREFILL_TAIL=1` | Experimental. No GPU run recorded on any branch |

The PP recipes get `SGLANG_PP_LAYER_PARTITION` on prefill only, from `--prefill-pp-partition` or
the defaults above. `--ced-split` and `--prefill-bounded-replay` are mutually exclusive, and
`--prefill-bounded-replay` requires `--split-cuda-graphs`.

**The recipes do not encode the real-mix serving settings.** `tp2-pd.sh` passes only
`--cpu-engram-staging --split-cuda-graphs`. The launcher defaults are bring-up limits:

- 16,384 context;
- 65,536 KV tokens;
- 8 prefill and 16 decode requests;
- `--mem-fraction 0.80`.

The [operator guide](../recipes/deepseek-v41/Agent_Readme.md) passes the real-mix settings
explicitly:

```bash
recipes/deepseek-v41/tp2-pd.sh \
  --resident-engram-pid "$ENGRAM_PID" --parallel-load \
  --context-length 131072 --prefill-swa-prefix-tails 256 \
  --max-total-tokens 2097152 --decode-max-total-tokens 8388608 \
  --decode-requests 64
```

The real-mix runs themselves were launched on `dsv4.1` from the saved `launch-oracle.sh` under
`/data/dsv41-bringup/realmix/`, not from these recipes.

`--num-prefill N` launches N TP2 prefill workers feeding one TP2 decode worker, for example
3P:1D on eight GPUs. The router then uses `--prefill-policy manual`. No N:1 measurement is
recorded.

### Host-side requirements

- The deployment runs in containers: `recipes/deepseek-v41/container/` builds the `dsv41-pd`
  image (the frozen `/data/venvs/dsv41` environment) and runs it as two enabled systemd units.
- `dsv41-engram-cache.service` runs the holder (`hold_engram_cache.py`), which must outlive every
  pair; `dsv41-pd.service` runs the pair and passes the holder's PID as `--resident-engram-pid`.
- Host-venv launches still need `LD_LIBRARY_PATH=.../site-packages/nvidia/cuda_runtime/lib`,
  because Mooncake's wheel links CUDA 12; the image sets it.
- The `nvidia-persistenced` override, where present, is transient. A reboot loses it.

## 4. DeepSeek-V4-Flash-0731 tooling (dormant)

`benchmark/deepseek_v4_b300/` implements the archived plan's fail-closed control. It is
integrated TP4 with DSpark, then 2xTP4 on GPUs 0–3 and 4–7 (ports 30300/30301, router 8030).
The control defaults are:

- `flashinfer_mxfp4` MoE and DSpark on;
- `--mem-fraction-static 0.90`, chunk 4096, `--swa-full-tokens-ratio 0.1`;
- TBO and DP attention off;
- FP32 compressed state.

It contains:

- `preflight.py`: CC, idle-GPU, topology and attestation-hash checks;
- `launch_dsv4_flash_0731_tp4.sh`, `launch_dsv4_flash_0731_tp4_dp2.sh`;
- `workload_manifest.py`, `run_scorecard.sh`, `experiment_matrix.py`;
- `tune.py` with `collectives` and `variants` suites, and `bundle_candidate.py`;
- `compare_probes.py` with `scripts/dsv4_correctness_probe.py`.

The launcher refuses an `auto` capacity. Its switches are `DSV4_ENABLE_DSPARK`,
`DSV4_ENABLE_TBO`, `DSV4_ENABLE_TBO_CUDA_GRAPH`, `DSV4_ENABLE_DP_ATTENTION`,
`DSV4_COMPRESS_STATE_DTYPE` and `DSV4_ENABLE_FP4_INDEXER`.

On `cc-main`:

- the tooling (`1a99fcd06c`, `7dd2a404ba`, `76601d7010`);
- the fused C4 FP4 index-cache store (`a49b932754`, `SGLANG_OPT_FUSE_C4_FP4_STORE`, default
  on);
- the CC copy-worker and graph-TBO fixes (`92422b1c05`);
- the DSpark auxiliary-state merge for DSV4 prefill TBO.

Only on `origin/deepseek-v4-cc-perf`: for example `0d5844e503` (ThunderMLA megakernels) and
`8e99b618f6` (DP replay and cache paths).

**Why dormant.** Work moved to V4.1. Every "remaining live gate" in the
archived V4-Flash plan is still open:

- no DSV4-tokenized corpus;
- no B300 capacity ladder;
- no matched target/DSpark controls;
- no TBO parity gate;
- no PD launcher.

Whether the plan's D3 rejection-sampling fallback reached `cc-main` was not confirmed. TBO and
DP attention for DSV4 remain opt-in experiments here.

## 5. DeepSeek-specific optimizations

### Prefix-tail retention: the largest measured lever

Each layer's SWA is per request and cannot be shared, so prefix caching needs a cached SWA
"tail" for every retained prefix. Prefill reserves **32 tails by default** (four times its eight
running-request slots), which is fewer than the 40–80 live conversations in the sweep. Tails
were evicted, and the cache hit rate stayed at 56–61%.

The fix was `--prefill-swa-prefix-tails 256`, which passes `--swa-prefix-tails` to prefill. It
grows the prefill SWA pool from 30,208 to 116,224 tokens (384 per tail):

- At c40, served TPM/GPU rose 7.1x, from 261,356 to 1,864,135.
- The cache hit rate rose from 56.4% to 95.0%.
- Oracle output matched the baseline exactly.

This is a one-variable A/B on `dsv4.1`, run under the older protocol. Raising the full-token cap
alone did not help: full-token and SWA pools are separate (2,097,152 full tokens on each role;
decode SWA 74,496).

The next limit is the prefill full-token pool. At c96, about 96 live 40K-token conversations
need about 3.8M tokens of prefix cache against prefill's 2.1M. The hit rate fell to about 80%
and throughput collapsed to 989,153. A larger prefill pool has not been tested. A larger
**decode** pool (`--decode-max-total-tokens 8388608`) removed decode admission stalls but did
not fix the conversation-start wave.

### CPU Engram staging

The default host-table path registers an mmap with `cudaHostRegister` and has a kernel
dereference it. On this CC VM, that fails:

- every GPU reports `host_register_supported=0` and `pageable_memory_access=0`;
- `cudaHostRegister` returns error 801.

On `dsv4.1`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=0` was not a fix: it left a pointer the GPU
could not read. `cc-main` reads no such switch; its host-table path always calls
`cudaHostRegister` (`python/sglang/srt/layers/engram.py`).

`--cpu-engram-staging` (`SGLANG_DSV41_ENGRAM_CPU_STAGING`) works around this
(`StagedEngramTable` in `python/sglang/srt/layers/engram_staging.py`). The CPU gathers the
selected FP8 rows from the shared read-only checkpoint mapping, then copies only those compact
rows to the GPU. Every worker shares one set of page-cache pages: 203,073,077,576 bytes, locked
by the holder.

`StagedEngramTable._stage_bytes` is an eager break inside otherwise captured graphs. It cannot
be removed: the n-gram hashes depend on token IDs, which must reach the CPU first.

Measured cost (decode, DSpark, c64, 40-iteration profile):

- CPU row gathers: about **5.7 ms per iteration per rank**;
- GPU-recorded H2D copies: about 0.86 ms;
- the CPU output-copy range is 21.6 ms, but it includes waiting for GPU work, so it is not
  transfer time.

`SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH=1` (default off) moves the gathers off the critical
path of decode and verify steps. A new break right after the hash copies the ids of both Engram
layers to the host in one transfer; one worker thread gathers layer 1, then layer 14, while the
GPU runs the layers in between, so each layer's break only waits for its finished rows and copies
them over. Rows are matched to the ids tensor they were gathered for and anything else falls
back to the blocking gather. The bridge buffers stay static, and rows are bit-identical to the
blocking path (`probe_staged_engram.py`). `SGLANG_DSV41_ENGRAM_GATHER_THREADS` sets the OpenMP
threads of that worker (0 keeps the process-wide value, 1 after model load).
`bench_staged_engram_prefetch.py` reports the saving on a synthetic table: about 2.3-3.0 ms per
step at 1 gather thread and 3.6-4.4 ms at 8 (steps of about 59 ms of GPU work, 384 tokens, one
rank; a two-rank contention run is not measured).

### Split CUDA graphs and long-context prefill

`--split-cuda-graphs` uses the breakable graph backend through an explicit
`--cuda-graph-config`. Prefill captures powers of two from 256 tokens up to the chunk size, plus
the chunk size itself (256/512/1024/2048/4096 at the default chunk 4096). Decode captures powers
of two up to the request limit, plus the limit itself. Because the prefill backend is set
explicitly, PP2 prefill keeps its graphs; `cuda_graph_hook.py` otherwise disables breakable
prefill graphs under PP.

- **Short prompts.** Prompts under the 256-token bucket run eager prefill.
- **Late-layer trimming.** The launcher always enables decoder SWA bounded replay (late-layer
  trimming) on decode. On prefill it enables it with `--prefill-bounded-replay`, or when split
  graphs are off (then both prefill and decode graphs are disabled).
- **Verification.** `TARGET_VERIFY` satisfies `is_extend()`, so the model excludes it from
  prefill-only graph breaks.

Startup cost in the DSpark deployment:

- target verify capture took 13.83–14.02 s and 1.72 GiB per rank;
- draft verify capture took 112.41–112.53 s, including first-time tuning, and 0.25 GiB.

**Prefill graphs stop at 16,384 tokens.** The captured low-ratio indexer scores a static
context width, so the hook defaults prefill `max_seq_len` to 16,384. With about 39K-token
real-mix prompts, every chunk then ran eager and host-bound. Profile at saturation: about 710 ms
of wall time per ~3.7K-token chunk against about 130 ms of non-collective GPU compute, with
TP0's GPU busy 22%.

On `dsv4.1`, `SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER` lifted the cap by running that
indexer as an eager break. Warm prefill latency (single-request probes):

| Prompt | Before | After |
| ---: | ---: | ---: |
| 40K | 4.3 s | 2.7 s |
| 70K | 9.8 s | 5.6 s |
| 100K | 11.9 s | 8.1 s |

**Ported to `cc-main`, off by default and unmeasured here.** That code is commit `61854d8876`
on the `pp-encoder*` branches (not `origin/dsv4.1`), and `cc-main` rewrote the indexer it
targeted. The port keeps the variable name. With it set, the ratio-1/2 index layers of the
breakable prefill graph run as an eager break on the live rows: the projection break still
computes the queries and head weights, and the break hands them to the same candidate indexer
that serves eager prefill, so contexts past the window use candidate blocks as they do eagerly.
Everything else stays captured, and the captured indexer's paged metadata is not built. Enable
it with the launcher's `--prefill-graph-max-seq-len N` (with `--split-cuda-graphs`). The
launcher's `--prefill-graph-eager-indexer` fails and points to that flag. Not validated with
`--prefill-bounded-replay` or context parallelism. On GPU only synthetic indexer checks have
run; no real-model number exists on `cc-main`.

### Prefill indexer hoisting (not on cc-main)

A trace found about 115 ms of host time per prefill batch, including about 23 device-to-host
syncs in the low-ratio indexer. `SGLANG_OPT_DSV41_PREFILL_HOIST_PER_BATCH` built the slot layout
once per batch and recombined top-k only at the 8 index-source layers instead of 38. Effects on
`dsv4.1`:

- warm prefill latency dropped a further 7–19% (100K: 8.1 s to 7.2 s);
- probes were bitwise identical;
- c72 went from 2,885,052 to **3,325,279** served TPM/GPU.

`cc-main` rewrote the low-ratio indexer (`python/sglang/srt/layers/attention/dsv4/v41_indexer/`).
This hoist and three sibling experiments were not ported (the eager-break indexer was, see
above):

- batched candidates;
- fused metadata;
- attention plan.

`--prefill-hoist-per-batch` now fails on purpose. Whether the rewrite recovers this host cost is
unmeasured.

### CED split and tail graphs

In the CED split, prefill builds and runs only the **encoder layers 0–20**, through the last
KV-source layer. It hands their KV and a wider SWA window to decode. Decode then replays each
prompt's tail through every layer (decoder SWA bounded replay). That replay seeds:

- the late-layer windows;
- the first token;
- the DSpark draft ring.

Late layers 21–39 own no compressed KV, so prefill can skip them.

Set `SGLANG_OPT_DSV41_CED_SPLIT=1` on both workers, or use the launcher's `--ced-split`. The
prefill side runs TP1 x PP2 with partition `8,13` and no DSpark.

**The cost moves to decode.** On PP2 (`pp-encoder` branch), the eager tail replay had a
single-request median of 482 ms, including preparation. The kernel sum was 49 ms at 8K, with
2,722 kernels in 532 ms. The replay is **host-dispatch bound, not GPU bound**.

Node A (local commits) added the following, under these knobs:

| Feature | Knob or file |
| --- | --- |
| Encoder tail graphs | `SGLANG_DSV41_CED_TAIL_ENCODER_GRAPH`, `..._ENCODER_EXACT_ROWS` (`ced_encoder_graph.py`) |
| Late-layer tail graphs | `SGLANG_DSV41_CED_TAIL_LATE_GRAPH` (`ced_tail_graph.py`) |
| Tail coalescing | `SGLANG_DSV41_CED_TAIL_BATCH_SIZE`, `..._MAX_WAIT_MS` |
| Static graph retention | `ced_graph_cache.py` |

Results:

- The kernel span at 8K fell from 532 ms to 137.5 ms. Each replay launches 33 graphs with 1,659
  kernels.
- The remaining 1,012–1,030 eager kernels sum to only 4.1–5.0 ms, but they dominate the host
  gaps.
- Only 33.4% of non-short tail batches hit a captured exact-row encoder graph. Padded buckets
  change GEMM results, so they are not used.
- Batch-1 tail median: 100.28 ms on a graph hit, 298.26 ms on fallback.

Best node A results: 1.904M at c56 and 2.263M at c72 with 100 ms wait. That is below the
reported TP2 replica A reference: 3.010M at c56 and 3.460M at c72. Unbounded retention ran
out of memory, and a bounded LRU hit an illegal memory access.

**How to check CED:**

- `SGLANG_DSV41_CED_TAIL_GRAPH_CHECK` cross-checks the first replay per shape against eager.
  `..._CHECK_EVERY` checks every replay. Node A's final validation recorded 102 byte-exact
  rank/shape checks with zero mismatches.
- `SGLANG_DSV41_CED_TAIL_TIMING` emits non-synchronizing tail timing. It is summarized by
  `benchmark/deepseek_v41_pd/summarize_ced_variant.py`.
- `validate_ced_variant.py` runs probes, exact-oracle checks and an optional sweep.
  `check_ced_graph_shapes.py` exercises batch sizes 1–8.
- CED options can be changed live through the decode worker's `/set_internal_state`.
  `python/sglang/srt/disaggregation/ced_options.py` validates them on all TP ranks.

These graph modules reached `cc-main` in `8e98e4cf81` and have **not run on a GPU there**. Node
A also used the unported attention-plan and hoist experiments.

### Pipeline-parallel prefill

PP2 exists only as a prefill-side layout for the two split modes. It puts TP1 on each of two
GPUs, with `SGLANG_PP_LAYER_PARTITION` set by `--prefill-pp-partition`. Stages must split at
KV-source layers.

Bring-up exposed two bugs, fixed on `pp-encoder` and ported:

- **Missing layer IDs.** Ratio-2 KV/index entries did not publish layer IDs, which crashed
  Mooncake's transfer thread with an `IndexError`.
- **Dropped `prev_pre`.** The prefill graph buffers omitted the mHC `prev_pre` coefficients, so
  graph replay could silently discard them. They are now carried as float32
  `[tokens, hc_mult]`; see `test/registered/unit/models/test_deepseek_v4_pp_prev_pre_handoff.py`.

CED PP2 measured **1.457M at c56** (95.6% hit, 2 read timeouts) and 1.294M at c72 (86.3% hit,
outside band). That is 52% and 63% below the replica A reference.

`SGLANG_OPT_DSV41_PREFILL_TAIL` is the alternative layout: PP2 prefill over all 40 layers
(partition `14,26`), the late layers over only the bounded prompt tail, and DSpark on the last
stage. It was ported, but no GPU run is recorded on any branch.

### DSpark on DeepSeek

DSpark runs on both PD roles with identical settings. `--no-dspark` is the matched control.

DeepSeek-specific points:

- **Graph eligibility.** Under oracle forcing, fused in-graph acceptance is disabled, but
  target and draft graph replay stay eligible.
- **No controlled speedup yet.** Synthetic c64 measured 161,237 output TPM with DSpark against
  104,044 target-only. The target-only run had DevTools OFF and the DSpark runs had it ON, so
  this is **not a controlled speedup**.
- **Real acceptance** is unmeasured.

For V4-Flash, DSV4 prefill TBO was originally disabled whenever DSpark captured auxiliary
hidden states. The merge path behind `DSV4_ENABLE_TBO` exists, but its parity gate never ran.

### Decode kernels: where the time goes

The profile covers 40 speculative iterations at c64 on decode TP0/TP1, with DevTools ON and
`dsv4.1`. The iteration interval was 63.5 ms. Summed kernel time was 41.257 ms per GPU, and GPU
activity covered about 62% of the span.

| Family | ms/iter/GPU | Share of kernel time |
| --- | ---: | ---: |
| Expert GEMM (MXFP4, gate/up + SwiGLU, down) | 17.227 | 41.76% |
| NCCL collectives (all-reduce `RING_LL`, 88 calls) | 5.595 | 13.56% |
| mHC | 4.919 | 11.92% |
| Dense GEMM | 4.082 | 9.89% |
| Sparse MLA + combine | 3.217 | 7.80% |
| Other | 6.217 | 15.07% |

Why expert GEMMs cost so much at EP2:

- EP2 still leaves 192 experts per GPU.
- Batch 64 x 6 verify positions x top-6 averages only about 6 rows per expert.
- If every local expert is active, gate/up reads 90.60 GB per GPU per iteration. That is an
  equivalent 8.12 TB/s against B300's roughly 8 TB/s. This is a sanity check, **not measured
  bandwidth**.

A 20% gate/up win would save 2.23 ms, at most 3.5% of the iteration.

[KERNEL_PLAN.md](../benchmark/deepseek_v41_pd/KERNEL_PLAN.md) is a proposal, not a result. Its
order:

1. A measurement milestone: expert histograms, counters and a replay harness.
2. A CuTe DSL mHC stats/Sinkhorn prototype that preserves FP32 and batch invariance.
3. Qualifying the existing collective fast paths under CC before writing a fused two-rank
   reduction.
4. A small-row MXFP4 grouped GEMM, only if counters show headroom. The NVFP4 CuTe path does not
   drop in: it uses 16-element scale blocks, while MXFP4 uses 32.
5. Fusing RMSNorm, quantization and copy.

Evaluate DeepGEMM MegaMoE (`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`, default off) and upstream
mega-kernels before writing replacements. The `dsv4.1` fused mHC stats were not ported;
`cc-main` uses DeepGEMM HC prenorm (`SGLANG_OPT_DEEPGEMM_HC_PRENORM`, default on).

`SGLANG_DSV41_FUSED_WO_A` (default on) collapses the BF16 `wo_a` verify chain into one
megakernel on SM100/SM103. These docs record no measurement of it.

### KV and index formats

- **V4.1 deployment.** FP8 KV (`fp8_e4m3`) with FP4 index keys.
- **Layout switch.** `SGLANG_DSV4_KV_LAYOUT` defaults to `v4` (584 B/token per layer). `v41`
  selects the SM100 FlashMLA V4.1 formats: a 528 B FP8 SWA cache plus FP8 or FP4 compressed
  caches, controlled by `SGLANG_DSV4_COMPRESSED_KV_LAYOUT`. The launcher does not set the
  switch, and no A/B is recorded.
- **V4-Flash knobs.** `--enable-deepseek-v4-fp4-indexer`, `SGLANG_OPT_FUSE_C4_FP4_STORE` and
  `SGLANG_DSV4_COMPRESS_STATE_DTYPE` (FP32 default versus BF16) were tuner variants. None was
  measured.

## 6. Results and caveats

All V4.1 rows share these conditions unless noted:

- B300;
- CC ON with NVLE;
- DevTools ON (except the target-only sweep);
- one PD pair with a 4-GPU denominator;
- oracle commit 6 for real-mix rows.

None of these numbers is a qualification result, and none was measured on `cc-main`.

| Number | Conditions | Kind | Evidence |
| --- | --- | --- | --- |
| 44,226 / 72,403 / 104,044 output TPM (pair) | Target-only, c16/32/64, 1,024 in / 256 out, DevTools OFF, `dsv4.1` | Short synthetic, single runs | [PERFORMANCE.md](../benchmark/deepseek_v41_pd/PERFORMANCE.md); `/data/dsv41-bringup/batch-sweep-summary.json` |
| 161,237 output TPM (pair); TTFT mean 2.042 s; TPOT mean 12.811 ms | DSpark, c64, same synthetic | Short synthetic, single run | `/data/dsv41-bringup/tpm-dspark-b64-20260928-0428.summary.json` |
| 261,356–291,069 served TPM/GPU; hit 56.4–61.2%; TTFT p50 54–91 s | Real-mix r67, c40–80, 32 tails, 60 s warmup + 300 s | Fixed window; fails scorecard | [REALMIX_BENCHMARK.md](../benchmark/deepseek_v41_pd/REALMIX_BENCHMARK.md); `/data/dsv41-bringup/realmix/` |
| 1,864,135 served TPM/GPU; hit 95.0%; TTFT p50 7.67 s | 256 tails, c40, old protocol | Fixed window ends before the first conversation wave, so optimistic | `/data/dsv41-bringup/realmix/tails256/` |
| 2,771,803 / 2,885,052; TTFT p50 1.91 / 5.04 s | 256 tails, long-context prefill graphs, chunk 4096, c56/c72; 240 s ramp, 300 s warmup, 300 s window | Steady-state fixed window, single runs | REALMIX_BENCHMARK.md |
| 989,153; hit 79.8% | Same, c96 | Prefill pool collapse | REALMIX_BENCHMARK.md |
| **3,325,279**; hit 94.9%; TTFT p50 / p95 2.28 / 3.56 s; 0 failed | + per-batch hoist, c72 | Steady-state, single run; **hoist and long-context graphs not on `cc-main`** | REALMIX_BENCHMARK.md |
| 1.457M / 1.294M | CED PP2, c56/c72, `pp-encoder` | Fixed window; c56 two timeouts, c72 out of cache band | [pp2-ced-report.md](../benchmark/deepseek_v41_pd/pp2-ced-report.md); `/data/dsv41-bringup/ced/pp2` |
| 1.904M / 2.263M | Node A CED tail graphs, 100 ms wait, c56/c72, local commits | Fixed window; fails 60 tok/s slowest-user floor | [nodea-full-replay.md](../benchmark/deepseek_v41_pd/nodea-full-replay.md); `/data/dsv41-bringup/ced/nodeA/` |
| 3.010M / 3.460M | "Replica A" TP2 reference, c56/c72 | Provided reference; raw artifact not cited in repo docs | pp2-ced-report.md, nodea-full-replay.md |
| 63.5 ms iteration; 41.257 ms kernel sum | Decode profile, DSpark c64 | Profiled; overhead included | [PERFORMANCE_KERNELS.md](../benchmark/deepseek_v41_pd/PERFORMANCE_KERNELS.md); `/data/dsv41-bringup/profile-dspark-b64-20260928-143420/` |
| None | DeepSeek-V4-Flash-0731 on B300/CC | No measurement exists | — |

Caveats that change how to read the table:

- The real-mix "served TPM" counts cached input. Comparing it with M3 needs matched datasets,
  topology and acceptance. The reports compare against a supplied M3 report on a different
  800-conversation trace, so treat that ratio as context, not a result.
- Earlier windows (the baseline and Fix 1) ended before conversations restarted cold. They are
  not comparable with the steady-state rows.
- Decode ran at a batch of about 23 at the one-pair ceiling. Prefill limited throughput, so
  decode kernel work cannot raise these numbers until prefill improves.
- PD handoff overhead has never been isolated. Router and worker metrics were off in the
  profiled deployment.

### What did not reach cc-main

Commit `8e98e4cf81` ported the CED and prefill-tail code, CPU Engram staging, the Mooncake fixes
and oracle commits. It did not port:

- four of the five prefill-indexer experiments: per-batch hoist, batched candidates, fused
  metadata, attention plan (the eager-break indexer was ported later, off by default);
- the `dsv4.1` fused mHC stats.

Every steady-state real-mix row used long-context prefill graphs from the eager-break indexer.
On `cc-main`, prefill graphs default to 16,384 tokens, and
`SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER` replays longer contexts without a measurement.
**Re-measure on `cc-main` before quoting any V4.1 number.**

## 7. Open problems and next steps

1. **Re-baseline on `cc-main`.** Run `tp2-pd.sh` with the operator-guide flags on GPU, then
   repeat the steady-state real-mix sweep. Check whether the rewritten `v41_indexer` recovers
   the hoist's host savings, and what `--prefill-graph-max-seq-len` buys over the eager
   long-context prefill that is the default.
2. **Prefill capacity.** Prefill limits throughput, and the c96 collapse is prefill-pool bound.
   Test a larger prefill full-token pool. Measure prefill graph coverage beyond 16K
   (`--prefill-graph-max-seq-len`) or lower eager launch overhead. Measure `--num-prefill N` (N:1) as the equal-GPU alternative.
3. **Real acceptance and DevTools OFF.** Every DSpark result is oracle-forced with DevTools ON.
   Measure genuine acceptance, and a controlled DSpark on/off pair in the same GPU mode.
4. **Unexplained streaming read timeouts.** Two 180 s client timeouts appeared at c54 in the
   real-mix sweep and two at c56 on CED PP2. The servers logged no error. A PD handoff stall is
   suspected, not shown.
5. **Isolate PD handoff cost.** Correlate request IDs across router and workers, and record
   transfer bytes and durations.
6. **20K continuation drift.** Four of five saved 8K–100K probes match exactly. The 20K
   continuation varies, even between two eager runs; first-token IDs and log-probabilities
   match. Bitwise equivalence is not claimed.
7. **CED.** Reduce eager indexer and attention scaffolding in the tail replay, and cover more
   exact shapes without large capture memory. Until then, CED stays behind the plain TP2 pair.
8. **Decode kernels**, after prefill: follow KERNEL_PLAN milestone 0 first.
9. **Operations before any exposure:** auth, TLS, worker metrics, boot persistence, an admission
   cap below c96, and a second pair on GPUs 4–7. See the
   [operator guide](../recipes/deepseek-v41/Agent_Readme.md#before-go-live).
10. **V4-Flash 0731** stays dormant unless reprioritized. Its first gates are a DSV4-tokenized
    corpus and a B300 capacity ladder.

## 8. Knob reference

All of these are defined in `python/sglang/srt/environ.py` unless marked as a launcher flag.

| Knob | Default | Purpose |
| --- | --- | --- |
| `SGLANG_DSV41_ENGRAM_CPU_STAGING` | off | Eager CPU Engram gather (`--cpu-engram-staging`). Takes precedence over the host table |
| `SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH`, `SGLANG_DSV41_ENGRAM_GATHER_THREADS` | off, 0 | Worker-thread gather of both staged Engram layers from one id copy in decode/verify steps; OpenMP threads of that worker (0 = process-wide torch value) |
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | off, `shared` | Host-table path; needs host registration, which this CC VM lacks. The launcher always sets the first to `1` and the layout to `--host-layout` (default `auto`). `engram.py` accepts only `shared` or `per_rank`, so a launch without `--cpu-engram-staging` fails at model load unless `--host-layout shared` is passed |
| Set by the launcher, read by nothing on `cc-main` | | `SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN`, `SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE`, `SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH`, `SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER`. `--candidate-indexer deepgemm` therefore only adds DeepGEMM import checks to preflight |
| `SGLANG_RAGGED_VERIFY_MODE` | `static` | The launcher sets `static`; the DSpark PD hook requires it |
| `SGLANG_ORACLE_FIXED_COMMIT_LEN` | 0 | Set by `--oracle-commit-length` (benchmark-only forced-token replay) |
| `SGLANG_OPT_DSV41_CED_SPLIT` | off | CED split, set on both workers |
| `SGLANG_OPT_DSV41_PREFILL_TAIL` | off | PP2 prefill-side tail; exclusive with CED |
| `SGLANG_DSV41_CED_TAIL_{ENCODER_GRAPH,LATE_GRAPH,BATCH_SIZE,MAX_WAIT_MS}` | off, off, 1, 1000 | CED tail graphs and coalescing |
| `SGLANG_DSV41_CED_TAIL_GRAPH_CHECK[_EVERY]`, `SGLANG_DSV41_CED_TAIL_TIMING` | off | CED verification and timing |
| `SGLANG_OPT_DSV41_ASYNC_TAIL_METADATA` | off | A/B for pinned uploads in tail metadata |
| `SGLANG_DSV41_TORCH_PREFILL_INDEXER` | off | Torch ratio-1/2 prefill indexer (oracle/fallback) |
| `SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER` | off | Eager break for the prefill graph's ratio-1/2 index layers; lifts the 16,384-token prefill graph cap. Set by `--prefill-graph-max-seq-len N` above 16,384 |
| `SGLANG_DSV41_FUSED_WO_A` | on | `wo_a` verify megakernel (SM100/SM103) |
| `SGLANG_DSV4_KV_LAYOUT`, `SGLANG_DSV4_COMPRESSED_KV_LAYOUT` | `v4`, `auto` | KV pool layout |
| `SGLANG_DSV4_COMPRESS_STATE_DTYPE` | `float32` | V4 compressed-state dtype |
| `SGLANG_OPT_FUSE_C4_FP4_STORE` | on | Fused C4 FP4 index-cache store |
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` | off | MegaMoE; unqualified here |
| `SGLANG_PP_LAYER_PARTITION` | unset | Set by `--prefill-pp-partition` |
| Launcher: `--prefill-swa-prefix-tails`, `--decode-max-total-tokens`, `--oracle-commit-length`, `--num-prefill`, `--candidate-indexer`, `--transport`, `--devtools-mode` | | See [README](../benchmark/deepseek_v41_pd/README.md#candidate-launcher) and `launch.py --help` |


**On `cc-main-nightly`:** the mHC logic is refactored into
`python/sglang/srt/models/deepseek_v4_mhc.py` with an `HcState` type, and model layers use
`python/sglang/srt/layers/layer_boundary`. Neither exists on `cc-main`.

## 9. Further reading

- [benchmark/deepseek_v41_pd/README.md](../benchmark/deepseek_v41_pd/README.md): bring-up
  history, Engram/CC blockers, source audit, launcher flags, N:1 prefill.
- [PERFORMANCE.md](../benchmark/deepseek_v41_pd/PERFORMANCE.md): synthetic throughput, decode
  profile, CPU Engram costs, the unresolved PD-overhead question.
- [PERFORMANCE_KERNELS.md](../benchmark/deepseek_v41_pd/PERFORMANCE_KERNELS.md): all 152 decode
  kernel symbols with timings.
- [KERNEL_PLAN.md](../benchmark/deepseek_v41_pd/KERNEL_PLAN.md): proposed mHC, collective,
  grouped-GEMM and fusion work, with promotion gates.
- [REALMIX_BENCHMARK.md](../benchmark/deepseek_v41_pd/REALMIX_BENCHMARK.md): real-mix dataset
  conversion, oracle mode, prefix-tail fix, steady-state protocol, prefill experiments.
- [pp2-ced-report.md](../benchmark/deepseek_v41_pd/pp2-ced-report.md): CED PP2 bring-up bugs,
  validation and sweep.
- [pp2-tail-performance.md](../benchmark/deepseek_v41_pd/pp2-tail-performance.md): CED tail
  timing and kernel profile.
- [nodea-full-replay.md](../benchmark/deepseek_v41_pd/nodea-full-replay.md): CED encoder and
  late-layer tail graphs, cache coverage, failed retention experiments.
- [recipes/deepseek-v41/README.md](../recipes/deepseek-v41/README.md): recipe usage, port and
  environment overrides.
- [recipes/deepseek-v41/Agent_Readme.md](../recipes/deepseek-v41/Agent_Readme.md): operator and
  agent guide, including hard rules, diagnosis, recovery and go-live gaps.
- [benchmark/deepseek_v4_b300/README.md](../benchmark/deepseek_v4_b300/README.md): V4-Flash
  0731 launch, tuner, matrix and scorecard commands.
