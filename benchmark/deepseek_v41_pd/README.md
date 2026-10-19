# DeepSeek V4.1: TP2 prefill + TP2 decode with CPU Engram

`launch.py` starts one DeepSeek-V4.1-Flash prefill/decode pair on one node: a TP2+EP2 prefill
worker, a TP2+EP2 decode worker, and the PD router. The model's Engram tables stay in CPU RAM
and are gathered on the CPU, because this confidential-computing VM cannot map host memory
into the GPU. The production deployment runs this launcher inside the container in
[`recipes/deepseek-v41/container/`](../../recipes/deepseek-v41/container/README.md); the design
and knobs are in [`confai-docs/DEEPSEEK_V4.md`](../../confai-docs/DEEPSEEK_V4.md).

| File | Purpose |
| --- | --- |
| `launch.py` | Preflight, launch and supervise one PD pair plus router |
| `hold_engram_cache.py` | Holds and locks the Engram checkpoint pages for every worker |
| `download_model.py` | Resumable download of the pinned checkpoint |
| `probe_staged_engram.py`, `probe_engram.py` | Per-GPU Engram access probes run by preflight |
| `probe_custom_all_reduce.py` | CustomAllReduceV2 probe run by preflight with `--custom-all-reduce on` |
| `test_launch.py` | Launcher unit tests |

## Topology

| Resource | Responsibility |
| --- | --- |
| GPUs 0,1 | One model instance, TP2+EP2, prefill only |
| GPUs 2,3 | A second instance, TP2+EP2, decode only |
| CPU RAM | Both instances' immutable Engram embedding tables, one shared copy |
| Router | Sends a request through prefill, state transfer, then decode |

Both workers use DP=1 and CP=1. The Engram tables are part of the target model; they are
distinct from the DSpark speculation head and from SGLang's unrelated n-gram speculative
decoding. A second pair on GPUs 4–7 needs its own ports and output directory (see
[Launch](#launch)).

## Serving configuration: CPU Engram staging and split CUDA graphs

The checkpoint's Engram layers are 1 and 14 (zero-based), with 40 target layers. Layer 20 is
the last shared-KV source, not the end of the target model.

The default host-table path registers an mmap and has the GPU dereference it. On this CC VM
every GPU reports `host_register_supported=0` and `cudaHostRegister` fails, so the deployment
uses `--cpu-engram-staging`. It maps the original safetensors tables read-only, sharing
immutable page-cache pages across both workers, gathers the selected FP8 rows on the CPU and
copies only those compact rows to the GPU for the existing dequantization kernel. It allocates
no private table copies. Checkpoint files must remain immutable while serving.

`--split-cuda-graphs` uses the breakable graph backend. `StagedEngramTable._stage_bytes` is an
eager break: copy the current indices to CPU, gather FP8 rows/scales, then copy those compact
buffers to the GPU. The runner keeps stable bridge buffers and refreshes their contents on every
replay. Dequantization, projection and the rest of the model stay captured. With
`SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH=1` the ids of both Engram layers are copied once per
step and the gathers run ahead of the layers that need them.

The prefill worker captures powers of two from 256 tokens up to the chunk size; the decode
worker captures powers of two up to the request limit, plus the exact limit and any
`--decode-graph-extra-bs` sizes. Thus `--decode-requests 64 --decode-graph-extra-bs 10,12,14`
captures batches 1/2/4/8/10/12/14/16/32/64. Unsupported shapes keep the runner's eager fallback.
Prompts shorter than the smallest prefill bucket run eager prefill; their decode uses graphs.

DSpark is enabled by default; add `--no-dspark` for the target-only control. The runtime loads
the draft head bundled in the target checkpoint. Block size 5 means five proposed tokens and a
six-token verification window. Verification satisfies `ForwardMode.is_extend()` but uses the
decode graph runner, so the model's hash/attention helpers and low-ratio projection dispatch
exclude verification from prefill-only graph breaks.

Decoder SWA bounded replay (late-layer trimming after layer 20) is always on for decode. On
prefill it is on only with `--prefill-bounded-replay`, because the model rejects prefill graphs
with bounded replay otherwise.

## Engram residency

`hold_engram_cache.py` holds read-only mappings of the two Engram shards and locks their shared
page-cache pages in RAM: 203,073,077,576 bytes including shard metadata and small non-Engram
tensors. This prevents disk faults during serving without allocating per-rank copies. Run it
with unlimited memlock (systemd `LimitMEMLOCK=infinity`) and pass its PID to the launcher as
`--resident-engram-pid`; preflight then credits only measured read-only locked pages matching
the checkpoint files. Never mlock the writable private safetensors mappings: faulting those for
write would break sharing. Stopping the holder releases about 189 GiB.

In the container deployment, `dsv41-engram-cache.service` runs the holder and
`dsv41-pd.service` runs the pair with its PID.

## Launch

Download the pinned checkpoint (revision `dba1be0a40aa45a94ad051997016db3960a90277`):

```bash
python benchmark/deepseek_v41_pd/download_model.py
```

The downloader targets `/data/DeepSeek-V4.1-Flash`, checks free space before fetching weights,
and uses Hugging Face's resumable local download state. Use
`--local-dir /larger/volume/DeepSeek-V4.1-Flash` for another mount, or `--metadata-only` for
config/tokenizer/reference files without weights. `download-manifest.json` records revision and
completion state.

Print the resolved plan without CUDA initialization, checkpoint reads or a server launch:

```bash
python3 benchmark/deepseek_v41_pd/launch.py \
  --model /path/to/DeepSeek-V4.1 --dry-run
```

Inspect preflight first:

```bash
/path/to/v41-venv/bin/python benchmark/deepseek_v41_pd/launch.py \
  --model /path/to/DeepSeek-V4.1 \
  --preflight-only --output-dir /tmp/dsv41-preflight-01
```

Omit `--preflight-only` with a **new output directory** to launch; the script refuses to
overwrite a previous run. Preflight checks B300/GB300 inventory, selected GPU memory usage, CC
mode, ports, dependency pins/imports, the configured host-table memory and actual eager/graph
host gathers. The launcher then starts prefill, then decode, then the router, saves both
`/server_info` responses and runs a routed generation smoke (`smoke.json`). It supervises only
its own process groups and never stops other jobs. Router `/health` and systemd active state
alone do not indicate serving readiness; wait for `smoke.json`.

The memory check is a necessary lower bound, not proof of fit. Configured host table bytes per
worker are `sum(engram_num_embeddings) * (head_dim + head_dim/32)`. It budgets two copies plus a
32 GiB host reserve, crediting a resident holder.

Launcher defaults are bring-up limits: 16K context, 4K prefill chunks, 64K KV tokens per worker,
8 prefill requests and 16 decode requests. The production values are `DSV41_LAUNCH_ARGS` in
[`recipes/deepseek-v41/container/dsv41.env`](../../recipes/deepseek-v41/container/dsv41.env).

Worker ports default to 8031/8032, router 8030 and bootstrap 8998. All HTTP services bind to
loopback. Compile caches are per run/revision/worker unless `--kernel-cache-dir` is given. The
launcher clears inherited SGLANG/SGL/MC/MOONCAKE/NCCL tuning variables and sets the intended
contract explicitly; use its arguments rather than a previous shell's environment. It does not
change THP settings or evict checkpoint page caches.

For a second independent pair use `--gpus 4,5,6,7 --port 8130 --bootstrap-port 9098
--nccl-port 29600` and a separate output directory. Its CPU memory budget must include the first
pair's resident allocation, or both pairs must share one holder.

### Launcher arguments

- `--no-dspark`: disable speculation on **both** workers for the control.
- `--custom-all-reduce off|on`: NCCL (default) or CustomAllReduceV2. `on` drops
  `--disable-custom-all-reduce` on both workers, and preflight runs `probe_custom_all_reduce.py`
  on every TP2 GPU pair. FlashInfer all-reduce fusion stays off either way. The probe builds the
  TP group with the server's own constructor, so a failed IPC setup shows as the same silent
  NCCL fallback; it compares CustomAllReduceV2 with NCCL on a 384 x 5120 BF16 payload and times
  both. After each TP worker is healthy the launcher also fails if its log shows that fallback.
  On 2x B200 under CC it matched NCCL bit for bit at 10.3–10.4 us per all-reduce inside a CUDA
  graph against 49–52 us for NCCL.
- `--dspark-block-size N`: gamma; the runtime resolves verify width to N+1.
- `--candidate-indexer deepgemm`: unqualified optimization arm; only adds DeepGEMM import checks
  to preflight.
- `--host-layout shared|private|auto`: host-table layout for the non-staging path.
- `--transport intra-node-nvlink|rdma|tcp`: Mooncake transport. Intra-node NVLink (default) uses
  CUDA IPC memory handles and GPU copies. TCP is a diagnostic control. Mooncake's TCP override
  must be absent for NVLink: even `MC_FORCE_TCP=0` activates TCP.
- `--devtools-mode off|on`: require the specified existing DevTools mode during preflight
  (default off). This only checks the mode; it never changes GPU mode.
- `--prefill-graph-max-seq-len N`: the longest context the split prefill graphs replay (needs
  `--split-cuda-graphs`). The server caps it at 16,384 by default; above that the prefill
  ratio-1/2 index layers run as eager breaks (`SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER`, set
  on prefill workers by this flag). Rejected together with `--prefill-bounded-replay`.
  `--prefill-graph-eager-indexer` and `--prefill-hoist-per-batch` fail on purpose.
- `--prefill-swa-prefix-tails N`: SWA prefix tails retained on prefill. Each cached prefix needs
  its own tail, so this sets how many conversations keep their prefix cache.
- `--prefill-hicache-ratio R`, `--prefill-hicache-io-backend`, `--prefill-hicache-write-policy`:
  HiCache host tier on prefill. Under CC the defaults (direct IO, write-back) avoid the GPU cost
  of the kernel IO backend with write-through.
- `--prefill-schedule-policy`, `--prefill-requests`, `--decode-requests`,
  `--decode-graph-extra-bs`, `--decode-polling-interval`, `--tokenizer-workers`: scheduling and
  capacity; see the comments in `dsv41.env` for the chosen values.
- `--chunk-size`, `--max-total-tokens`, `--decode-max-total-tokens`, `--context-length`,
  `--mem-fraction`: memory and context.
- `--prefill-env` / `--decode-env SGLANG_KEY=VALUE`: per-role environment.
- `--parallel-load`, `--kernel-cache-dir`, `--compact-host-memory`: startup. Under CC, CUDA
  initialization needs free 2 MiB host pages; `--compact-host-memory` drops the page cache and compacts host memory right
  before each worker starts.
- `--ced-split`, `--prefill-pp`, `--prefill-pp-partition`, `--prefill-bounded-replay`:
  experimental split layouts; see `confai-docs/DEEPSEEK_V4.md`.

### N prefill : 1 decode

`--num-prefill N` launches N TP2 prefill workers feeding one TP2 decode worker; `--gpus` lists
2N+2 GPUs (prefill pairs first, the last two decode). Prefill i uses HTTP `--port`+1+i, NCCL
`--nccl-port`+i and bootstrap `--bootstrap-port`+i; decode uses `--port`+1+N and
`--nccl-port`+N. Per-role options (`--prefill-swa-prefix-tails`, `--prefill-graph-max-seq-len`,
`--prefill-requests`) apply to every prefill worker, and prefill workers share one compile
cache. N=1 is the single pair. For 3P:1D on one node:

```bash
python benchmark/deepseek_v41_pd/launch.py --model /data/DeepSeek-V4.1-Flash \
  --num-prefill 3 --gpus 0,1,2,3,4,5,6,7 --output-dir /tmp/dsv41-3p1d
# prefill 8031-8033 (bootstrap 8998-9000), decode 8034, router 8030
```

The router gets every prefill URL with its bootstrap port and `--prefill-policy manual`, which
pins each `X-SMG-Routing-Key` to one prefill worker.

## Environment notes

- Mooncake's wheel links CUDA 12, so host-venv launches need
  `LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cuda_runtime/lib`; the container
  image sets it.
- Install the matching prebuilt `flashinfer-jit-cache` (from `https://flashinfer.ai/whl/cu130`)
  before first boot to avoid the lengthy MoE routing source build. FlashInfer still downloads
  individual cubins during first-run tuning.
- Autotune cache keys include the PD role, so prefill and decode can share a compile cache.

## Verification

```bash
python3 -m unittest discover -s benchmark/deepseek_v41_pd -v
PYTHONPATH="$PWD/python" python -m pytest -q \
  test/registered/unit/disaggregation/test_mooncake_registration.py \
  test/registered/unit/model_executor/model_runner_components/test_ngram_embedding_manager.py
CUDA_VISIBLE_DEVICES=0 python benchmark/deepseek_v41_pd/probe_staged_engram.py --split-graphs
```

The staged-Engram probe creates real CUDA graph segments and verifies exact output after
repeated changes to GPU-computed indices, including duplicate rows, and checks logits and
hidden-state replay after shape switches.
