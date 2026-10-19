This is the deployment guide for DeepSeek-V4.1-Flash on our 8x B300 confidential-computing node. It covers the design, how to reach and check the deployment, its failure modes, and instructions to hand to your agent when something goes wrong. Unlike M3, **V4.1 is not a public production endpoint yet**: it is a loopback-only bring-up and benchmark deployment. Every statement below comes from the bring-up reports in `benchmark/deepseek_v41_pd/` or from the launcher code. Anything not yet decided is marked **TBD**; do not fill those in from memory.

**Branch status.** The V4.1 code reached `cc-main` in `8e98e4cf81` (2026-10-09), which records "Not yet run on GPU". Every observed behavior and number in this guide was measured on the `dsv4.1` branch, or on `pp-encoder` for CED PP2. The first `cc-main` launch is itself a bring-up. Background, the recipe table and the full results table: [confai-docs/DEEPSEEK_V4.md](../../confai-docs/DEEPSEEK_V4.md).

## Design

### Status and access

| item | state |
| --- | --- |
| public base URL | **TBD**: none exists. Nothing is exposed off the node |
| serving endpoint | `http://127.0.0.1:8030/v1`, model **`dsv41`** (router, loopback only) |
| auth | **none**. Router, workers and bootstrap bind `127.0.0.1`; the router has no API key configured. The router's Prometheus listener is the exception (see [Metrics](#metrics)) |
| TLS / front door / readiness service | **TBD**: no Caddy, `node-front` or `node-readiness.py` equivalent |
| monitoring / paging | **TBD**: no Better Stack monitors; router Prometheus on `:29000` only |
| qualification | **unqualified**. Bring-up, correctness probes and oracle real-mix sweeps only |

Before this becomes a customer endpoint, every TBD row needs a real answer. See [Before go-live](#before-go-live).

**One PD pair, no redundancy.** A V4.1 serving unit is one *pair*: a TP2+EP2 **prefill** worker on GPUs 0–1 and a TP2+EP2 **decode** worker on GPUs 2–3, behind one router. A request needs **both** roles: prefill computes the prompt, Mooncake (`--disaggregation-transfer-backend mooncake`, intra-node NVLink) moves its KV to decode, and decode generates. This differs from M3's two independent replicas, where losing one leaves 50% capacity. **Losing either V4.1 worker takes the pair down**, and the launcher tears down all of its processes when any one exits (see [Hard rules](#hard-rules)).

**Why PD and TP2.** The TP2 weights and KV pools fit, and splitting prefill from decode keeps long prefills from stalling decode. GPUs 4–7 can hold a second pair or one of the PP2 experiment variants (see [Configurations](#configurations)).

**Why the Engram tables live in CPU RAM, shared.** The checkpoint carries two Engram embedding tables (layers 1 and 14). One copy is 188.8 GiB, so two copies need 377.7 GiB, which is **more than the VM's 376.7 GiB of RAM**. On top of that, this CC VM cannot register host memory for GPU access: `cudaHostRegister` returns error 801 and every GPU reports `host_register_supported=0`. So:

* `dsv41-engram-cache.service` (`hold_engram_cache.py`) maps the two Engram shards **read-only** and `mlock`s their page-cache pages once (203,073,077,576 bytes). Every worker shares those pages.
* Workers run `--cpu-engram-staging`: the CPU gathers the selected FP8 rows from the shared mapping and copies only those compact rows to the GPU. This is an eager graph break inside otherwise captured CUDA graphs (`--split-cuda-graphs`).

### Components

| component | where | port | what it does |
| --- | --- | --- | --- |
| **dsv41-engram-cache.service** | node | — | runs the Engram holder container, which keeps the Engram pages locked in RAM for every worker. Must outlive any pair |
| **dsv41-pd.service** | node | — | runs the pair container: `benchmark/deepseek_v41_pd/launch.py`, which supervises prefill, decode and router. Active only while the pair serves; relaunches the whole pair 240 s after a failure |

Both units are enabled at boot and run containers from one image (`dsv41-pd`); see [container/README.md](container/README.md). GPUs, ports and serving flags come from `container/dsv41.env`.
| **router** (`smg`, PD mode) | node | 8030 | prefill → transfer → decode routing; `/readiness` requires both PD roles; Prometheus on `:29000` (all interfaces) |
| **prefill worker** | node | 8031 | TP2+EP2 on GPUs 0–1, Mooncake bootstrap on 8998 |
| **decode worker** | node | 8032 | TP2+EP2 on GPUs 2–3, DSpark speculation |
| **oracle proxy** (benchmarks only) | node | 8034 | `oracle_pd_proxy.py`: forced-token replay adapter for scored sweeps. Not a production router |

Worker NCCL ports start at 29500 (prefill +0, decode +1). Recipes for a second pair use router 8040 / bootstrap 8999 / NCCL 29600 / metrics 29001.

### Why readiness is built this way

`launch.py` only declares the pair up after all three checks pass:

1. each worker's `/health` returns 200;
2. the router's **`/readiness`** returns 200, which requires both a prefill and a decode worker registered;
3. a real routed completion succeeds and is written to `smoke.json`.

**Router `/health` and `systemctl is-active` prove nothing about serving.** Only `/readiness` plus a real completion do.

---

## TL;DR for someone paged at 3am

1. Is it actually broken? `curl` a **real completion** (below). Never trust `/v1/models` or router `/health`.
2. Check the router's `/readiness`. A 503 means a PD role is missing, which means the whole pair is down.
3. Check `systemctl status dsv41-engram-cache dsv41-pd`. If the engram cache is down, workers that start next will fault from disk or fail the memory check.
4. If a worker died, the launcher has already torn down the pair. Recover by relaunching the pair (procedure below), not by restarting one worker.

```bash
# is it alive (on the node; there is no public URL)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":50}'

# pair state
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8030/readiness
```

---

## Credentials

**None exist yet.** The router runs without an API key and every port is loopback-only, which is safe only while nothing is exposed. Before any exposure, an inference key, an admin key separate from it, and metrics auth are **TBD**. Copy the M3 split: the router enforces auth inside the guest, and TLS terminates outside it. **Do not paste credentials into this file**; reference where they are stored.

---

## Layout

| item | value |
| --- | --- |
| hardware | 8x B300 SXM6, 275,040 MiB HBM each, NV18; 376.7 GiB host RAM; confidential VM |
| CC state | CC ON, multi-GPU NVLE; all 8 GPUs passed the local verifier (`/data/dsv41-bringup/attestation-persistent.log`). **DevTools ON** for every current DSpark result and profile |
| driver | 595.91.07 |
| node hostname / SSH | **TBD** |
| repo | `/home/ubuntu/sglang-private` (cc-main) |
| Python env | the `dsv41-pd` image: `/data/venvs/dsv41` (Python 3.12) frozen into `container/requirements.lock`. The host venv stays for tests and probes. **Separate from the M2.7/M3 env; never upgrade either in place** |
| model | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`, at `/data/DeepSeek-V4.1-Flash` (510.3 GB, 48 shards, checksums verified) |
| bring-up artifacts | `/data/dsv41-bringup/` (attestation, preflight, every run's logs, `smoke.json`, sweep results) |

Model facts that people get wrong: 40 target layers. Engram layers are **1 and 14** (zero-based). Layer **20 is the last shared-KV source**, not the end of the model. Weights are MXFP8 dense with MXFP4 experts, and the KV cache is FP8. The DSpark draft head is **bundled in the target checkpoint**; no separate draft path is needed.

## Configurations

Recipes live next to this file and wrap `benchmark/deepseek_v41_pd/launch.py`. [README.md](README.md) covers usage and port overrides; the recipe table (topology, ports, transfer, speculation, status) is in [DEEPSEEK_V4.md](../../confai-docs/DEEPSEEK_V4.md#launcher-and-recipes). Operationally:

* `tp2-pd.sh` is the layout of the serving pair. With the flags in [Procedure: launch the serving pair](#procedure-launch-the-serving-pair) it matches the real-mix configuration below. The real-mix numbers were not produced by this recipe: they came from `launch-oracle.sh` on the `dsv4.1` branch.
* `ced-pp2.sh` and `prefill-tail-pp2.sh` are PP2 experiments for GPUs 4–7. CED PP2 ran on the `pp-encoder` branch; prefill-tail PP2 has no recorded GPU run on any branch.
* No recipe has run on a GPU on `cc-main`.

### Serving config (real-mix steady state)

| setting | value | why |
| --- | --- | --- |
| `--context-length` | 131,072 | real-mix prompts average ~39K tokens |
| `--max-total-tokens` (prefill) | 2,097,152 | prefill full-token pool; the launcher default is 65,536 |
| `--decode-max-total-tokens` | 8,388,608 | removes decode admission stalls (decode reached 97–99% KV during waves). ~14 GB/GPU, since V4.1 KV is ~1.6 KB/token |
| `--prefill-swa-prefix-tails` | 256 | **the cache-hit fix**. The default (32 tails = 4× 8 running requests) evicted prefixes and held hit rate at 56–61%; 256 brought it to 95% |
| `--chunk-size` | 4096 | 8192 measured the same; prefill batches rarely fill a chunk |
| `--decode-requests` | 64 | decode graph buckets 1–64; the launcher default is 16 |
| `--prefill-requests` | 8 | launcher default |
| DSpark | on, block size 5 (six verify positions) | `--no-dspark` is the matched control |
| graphs | `--split-cuda-graphs --cpu-engram-staging` | prefill buckets 256–4096 tokens. On `cc-main` prefill graphs stop at 16,384 tokens and longer prompts prefill eagerly unless `--prefill-graph-max-seq-len N` is set (the real-mix runs' long-context prefill graphs came from an indexer experiment, ported later and unmeasured on `cc-main`) |
| transport | `--disaggregation-transfer-backend mooncake`, intra-node NVLink | the launcher's default `--transport intra-node-nvlink` sets `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` and `MC_INTRANODE_NVLINK=true`. |
| collectives | custom all-reduce and FlashInfer all-reduce fusion **off** | the NCCL control has not been qualified under CC; `--custom-all-reduce on` (any recipe forwards it) opts in to CustomAllReduceV2 |

---

## Metrics

Only the router exports metrics: Prometheus format on port `29000` (`29001` for a second pair). `launch.py` passes `--prometheus-port` but not `--prometheus-host`, and `sglang_router` defaults that host to `0.0.0.0`, so this listener is **not** loopback-only. Firewall it or pass a loopback host before the node is exposed. **Workers expose none** (`/metrics` is off). No M3-style log exporter, Prometheus server, or dashboards exist for V4.1 yet: **TBD**.

```bash
curl -s http://127.0.0.1:29000/metrics | grep -E '^smg_worker_health\{'   # both roles must be 1
```

Per-batch KV usage, queue depth, cache hit and DSpark accept length are only in the worker logs under the run's output directory: `<output-dir>/prefill.log` and `decode.log`. The M3 histogram gotchas apply to the router's `smg_*` histograms too: query `_bucket`, never the bare name.

---

## Performance baseline

**None of these numbers was measured on `cc-main`.** All come from 8x B300 SXM6 with CC ON (multi-GPU NVLE) and DevTools ON, **one pair (4 GPUs in the denominator, prefill included)**, the r67 oracle real-mix dataset and oracle commit length 6. The TP2 rows ran on the `dsv4.1` branch from `/data/dsv41-bringup/realmix/launch-oracle.sh`: the 32-tail row is from the 2026-09-28 sweep (60 s warmup, 300 s window), and the 256-tail rows from the later steady-state protocol (240 s ramp, 300 s warmup, 300 s window; run date not recorded in the report). The CED PP2 row ran on the `pp-encoder` branch on 2026-09-29. They are measurements, not qualification. Full reports: `benchmark/deepseek_v41_pd/REALMIX_BENCHMARK.md`, `PERFORMANCE.md` and `pp2-ced-report.md`.

| config | branch | c | served TPM/GPU | cache hit | TTFT p50 / p95 | failed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| default 32 prefix tails | `dsv4.1` | 80 | 291,069 | 61.2% | 91.3 s / 110.7 s | 0 |
| 256 tails, long-context graphs, chunk 4096 | `dsv4.1` | 56 | 2,771,803 | 95.0% | 1.91 s / 4.21 s | 0 |
| same | `dsv4.1` | 72 | 2,885,052 | 94.9% | 5.04 s / 6.94 s | 0 |
| same | `dsv4.1` | 96 | **989,153 (collapse)** | 79.8% | 11.1 s / 145.9 s | 1 |
| + per-batch indexer hoist | `dsv4.1` | 72 | 3,325,279 | 94.9% | 2.28 s / 3.56 s | 0 |
| CED PP2 (the layout `ced-pp2.sh` now encodes) | `pp-encoder` | 56 | 1,457,000 | 95.6% | p95 2.55 s | 2 |

**Caveats that change how to read this table:**

* **Every 256-tail row used long-context prefill graphs from the eager-break indexer.** `cc-main` has a port behind `--prefill-graph-max-seq-len N` (off by default, unmeasured). Without it, expect eager prefill above 16,384 tokens.
* **The 3.33M "+ hoist" row cannot be reproduced on cc-main.** The per-batch hoist and three other prefill-indexer experiments were *not ported*, because cc-main rewrote that indexer. `--prefill-hoist-per-batch` now fails on purpose. Re-measure on cc-main before quoting any V4.1 number.
* **c96 collapses** because 96 live ~40K-token conversations need ~3.8M tokens of prefix cache and prefill's pool is 2.1M. The hit rate falls to ~80% and TTFT explodes. Cap concurrency below that, or test a larger prefill pool first.
* Decode runs a batch of ~23 at the ceiling, so **prefill limits throughput**. Under the oracle, the slowest 5% of users still decode at ≥86 tok/s. Real (unforced) acceptance is unmeasured; one sample accepted ~4.0 tokens per iteration.
* DevTools was ON for every DSpark run and OFF for older target-only runs, so those are not a controlled DSpark comparison.

---

## Operational rules

* **The pair lives and dies together.** The launcher polls prefill, decode and router every 2 s. If any of them exits, it SIGTERMs and then SIGKILLs every PID it owns, and only those. Never "restart just decode".
* **Never kill the Engram holder while a pair is running.** Workers would start faulting the 203 GB tables from disk. Stop it (`sudo systemctl stop dsv41-engram-cache`) only after every pair is down; the unit dependency stops `dsv41-pd` first. The pair container passes the holder's PID to the launcher as `--resident-engram-pid`, so the launcher credits the locked memory and never signals that process; a hand launch must pass it.
* **Checkpoint files are immutable while serving.** The workers and the holder map them. Never rewrite, re-download or `mlock` a writable mapping of them.
* **One output directory per launch.** `launch.py` refuses to reuse one. Every run keeps its own logs, `server_info`, manifest and `smoke.json`.
* **Never set `MC_FORCE_TCP` unless you mean TCP.** Mooncake checks only that the variable is *present*, so even `MC_FORCE_TCP=0` forces TCP. The launcher clears inherited `SGLANG_*`/`SGL_*`/`MC_*`/`MOONCAKE_*`/`NCCL_*` tuning variables; pass settings through its arguments, not your shell.
* **Another pair or experiment may be on GPUs 4–7.** Check `nvidia-smi` and the tmux sessions before touching anything. The PP2 and node A runs recorded protected PIDs and never signalled them.
* **Never two load sources against a pair.** The M3 rule applies unchanged.

## Known issues

* **Unexplained streaming read timeouts.** There were two 180 s client timeouts at c54 in the real-mix sweep, and two at c56 on CED PP2. The servers logged no error and no scheduler or transfer exception. Possibly a PD hand-off stall; the root cause is open.
* **20K continuation drift.** Four of five saved 8K–100K probe continuations match exactly, while the 20K one varies, even between two eager runs. First-token IDs and log probabilities always match. This is a reproducibility gap; bitwise equivalence is not claimed.
* **Short prompts prefill eagerly.** The smallest prefill graph bucket is 256 tokens, and the runner rejects excessive padding. Decode still uses graphs.
* **Long prompts prefill eagerly on `cc-main` by default.** `deepseek_v4_hook.py` caps the prefill graph `max_seq_len` at 16,384 and logs that longer contexts run eager. Real-mix prompts average ~39K tokens. `--prefill-graph-max-seq-len N` (needs `--split-cuda-graphs`; not validated with `--prefill-bounded-replay`) runs the ratio-1/2 index layers as an eager break so longer contexts replay graphs.
* **CED PP2 is far below the TP2 pair** (1.46M vs ~2.8M TPM/GPU at c56). Its decode prompt-tail replay is bound by host dispatch, not GPU kernels.
* **Boot persistence is configured but not yet exercised by a reboot.** `dsv41-engram-cache` and `dsv41-pd` are enabled container units. The `nvidia-persistenced` override (`/run/systemd/system/nvidia-persistenced.service.d/dsv41-recovery.conf`), where a node has one, still lives in `/run` and is lost on reboot.
* **A native `/generate` request carrying `forced_token_ids` crashes DSpark decode, and the pair with it**, unless the pair runs with `--oracle-commit-length`. The router does not forward that field; direct worker ports do.
* **Driver mismatch after a restart.** After the data-disk restart, the loaded module (595.71.05) disagreed with NVML and the on-disk module (595.91.07), and `nvidia-smi` failed. Check `nvidia-smi` works before any launch.
* **First boot downloads FlashInfer cubins** during tuning. Point `--kernel-cache-dir` at a compatible previous run to reuse compiled kernels.
* **Mooncake links CUDA 12.** The image sets `LD_LIBRARY_PATH` to its `nvidia/cuda_runtime/lib`; a host-venv launch needs `LD_LIBRARY_PATH=/data/venvs/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib`.

## Before go-live

None of these exist for V4.1. Each needs an owner and a decision before customer traffic:

| need | M3 precedent | V4.1 |
| --- | --- | --- |
| public hostname + TLS | Caddy on the host, outside the attestation chain | **TBD** |
| auth | router-enforced inference key + separate admin key | **TBD** |
| readiness | `node-readiness.py` `/_ready` + `/_serving` with capacity guard | **TBD**; router `/readiness` is the only signal |
| worker metrics | `m3-worker-exporter.py` → Prometheus (90 d) | **TBD** |
| paging | Better Stack Playwright completion check | **TBD** |
| boot persistence | `boot-recover.v5.sh` wrapper chain | enabled container units (`container/`); not yet tested across a reboot |
| redundancy | two TP4 replicas | one PD pair; a second pair on GPUs 4–7 is the obvious step |
| admission cap | `--max-concurrent-requests 160`, queue 0 | **TBD**: stay under c96, where the pair collapses |
| qualification | 3× 600 s runs, provider verifier, GPQA | **TBD** |

## Files

* `recipes/deepseek-v41/`: launch recipes and their README.
* `benchmark/deepseek_v41_pd/`:
  * `launch.py`: the supervisor;
  * `hold_engram_cache.py`: the Engram residency holder;
  * `oracle_pd_proxy.py`: the oracle replay adapter;
  * `probe_engram.py` / `probe_staged_engram.py`: Engram access probes;
  * `validate_ced_variant.py` / `summarize_ced_variant.py`: CED variant validation and summaries;
  * the reports: `README.md`, `PERFORMANCE.md`, `PERFORMANCE_KERNELS.md`, `REALMIX_BENCHMARK.md`, `pp2-ced-report.md`, `pp2-tail-performance.md`, `nodea-full-replay.md`.
* `/data/dsv41-bringup/`: every run's artifacts.

---

# Hand this to your agent

Everything below is written for an agent or engineer operating the V4.1 deployment with **no prior context**. Read the rules before the procedures.

## Access

Node hostname and SSH route: **TBD**. Once you are on the node, everything is local:

```bash
cd /home/ubuntu/sglang-private
PY=/data/venvs/dsv41/bin/python
systemctl status dsv41-engram-cache dsv41-pd --no-pager
sudo docker ps -a             # dsv41-engram-cache-1 and dsv41-pd-1, with health
tmux ls                       # host-venv benchmark deployments run in named tmux sessions
```

## Hard rules

1. **Never restart one worker.** A PD pair is prefill + decode + router under one `launch.py`. Any exit tears down the whole pair. Recover by relaunching the pair.
2. **Never stop `dsv41-engram-cache` while any pair runs**, and never pass its PID to anything that signals process groups.
3. **Kill explicit PIDs only.** SIGTERM, wait, then SIGKILL. The launcher already does this for the PIDs it owns. If you must clean up by hand, enumerate the GPU PIDs on *your* pair's GPUs from `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv`. Do not touch PIDs on other GPUs.
4. **Check who owns GPUs 4–7 first.** They may run a second pair or an experiment from another worktree.
5. **Never reuse an output directory or edit a running launcher script.** Write a new script to a temp file and `mv` it; bash reads running scripts by byte offset.
6. **Never set `MC_FORCE_TCP`, and never upgrade `/data/venvs/dsv41` or the M2.7/M3 environment in place.**
7. **Never modify the checkpoint under `/data/DeepSeek-V4.1-Flash` while anything maps it.**
8. **Capture evidence before tearing down a wedged pair:** `dmesg -T | grep -i xid`, the tail of `prefill.log` / `decode.log`, `nvidia-smi`, and `py-spy dump` (needs sudo) of the scheduler PIDs.

## Diagnose before you touch anything

```bash
# 1. is the product broken? (real completion, never /v1/models or router /health)
curl -s -w '\n%{http_code}\n' http://127.0.0.1:8030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"What is 17 + 25?"}],"max_tokens":50}'

# 2. the ports: prefill, decode, router readiness
for u in 8031/health 8032/health 8030/readiness; do
  curl -s -m 4 -o /dev/null -w "$u:%{http_code} " http://127.0.0.1:$u
done; echo

# 3. supporting state
curl -s http://127.0.0.1:29000/metrics | grep -E '^smg_worker_health\{'
systemctl is-active dsv41-engram-cache dsv41-pd
nvidia-smi --query-gpu=index,memory.used --format=csv
nvidia-smi conf-compute -f                 # must still say CC ON
sudo dmesg -T | grep -i xid                # compare against a captured baseline, not zero
```

**Interpreting it:**

| symptom | meaning |
| --- | --- |
| `/readiness` 503, one worker `/health` fails | a role is down, so the pair is down. The launcher has torn it down or is doing so; relaunch the pair |
| all ports refuse, GPUs 0–3 near-empty | the launcher exited. Read the end of `dsv41-pd`'s journal and the newest output dir |
| `/readiness` 200 but completions hang | possible PD hand-off stall (known open issue). Capture evidence, then relaunch |
| completion 200 but slow, TTFT tens of seconds | prefix-cache collapse. Check concurrency (≥96 collapses) and the cache hit in `prefill.log` |
| `nvidia-smi` fails with a driver/library mismatch | loaded module ≠ installed NVML. Fix the driver before any launch |
| `Xid 120` → `154` | GPU reset required; it cannot be cleared inside the guest. Same as M3: guest stop/start |
| Mooncake `operation not supported` / native crash in transfer | thread-device regression in the intra-node path (fixed in `mooncake_transfer_engine.py`). Check you are on cc-main |

## Procedure: start the Engram holder (only if it is not running)

```bash
sudo systemctl start dsv41-engram-cache       # returns once every shard is locked
cat /var/lib/docker/volumes/dsv41_run/_data/engram.ready   # its host PID, for a hand launch
```

It is ready when its journal prints `READY: 203073077576 checkpoint bytes locked read-only in shared CPU page cache`.

## Procedure: launch the serving pair

```bash
# the deployment: edit container/dsv41.env (GPUS, PORT, --gpu-models, pool sizes), then
sudo systemctl start dsv41-pd                 # returns once the smoke completion passed
journalctl -u dsv41-pd -f                     # launcher output; worker logs in the run's output dir
```

Without containers (host venv, e.g. to try an uncommitted change):

```bash
cd /home/ubuntu/sglang-private
export LD_LIBRARY_PATH=/data/venvs/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
ENGRAM_PID=$(systemctl show -p MainPID --value dsv41-engram-cache)

# always dry-run first: prints the full plan, touches no GPU
PYTHON=/data/venvs/dsv41/bin/python GPUS=0,1,2,3 PORT=8030 BOOTSTRAP_PORT=8998 \
  NCCL_PORT=29500 ROUTER_METRICS_PORT=29000 \
  OUTPUT_DIR=/data/dsv41-bringup/pd-$(date -u +%Y%m%dT%H%M%SZ) \
  recipes/deepseek-v41/tp2-pd.sh --dry-run \
  --resident-engram-pid "$ENGRAM_PID" --parallel-load \
  --context-length 131072 --prefill-swa-prefix-tails 256 \
  --max-total-tokens 2097152 --decode-max-total-tokens 8388608 \
  --decode-requests 64
# then: the same command with --preflight-only, then without either flag, inside tmux or systemd-run
```

Preflight fails unless the GPUs' DevTools mode equals `--devtools-mode` (default `off`), CC is ON, multi-GPU mode is NVLE, and every assigned GPU is an idle B300/GB300 (at most 256 MiB used). If the node is still in DevTools ON, add `--devtools-mode on` and record that the run is not CC-qualifiable. Preflight checks modes only; it does not attest.

Green means the launcher prints `PD smoke passed: http://127.0.0.1:8030/v1 (model dsv41)`, `/readiness` is 200, and a real completion returns 200. Save both workers' `/server_info` into the output directory, and record the actual allocated KV capacities from it rather than the requested caps.

## Procedure: stop the pair

```bash
sudo systemctl stop dsv41-pd          # or Ctrl-C a host-venv launcher in its tmux session
```

The launcher signals only its own PIDs. Leave `dsv41-engram-cache` running unless every pair is down and you mean to free 189 GiB of RAM.

## Procedure: recover after a worker death

`dsv41-pd` relaunches the pair by itself 240 s after a failure, at most three times in four hours; after that the unit stays failed until `sudo systemctl reset-failed dsv41-pd && sudo systemctl start dsv41-pd`. By hand:

1. Capture evidence (Hard rule 8).
2. Confirm the launcher has exited and GPUs 0–3 hold no leftover compute PIDs. If any remain, SIGTERM those explicit PIDs, wait 30 s, then SIGKILL.
3. Wait for the GPUs to settle before relaunching. On M3 an early relaunch caused `Xid 120 → 154`. Use the same gate: ≥180 s, then 60 s of zero D-state.
4. Relaunch with a **new** output directory (procedure above).

## Running the test suites

```bash
# CPU unit tests for the V4.1 PD/PP code (no GPU needed)
$PY -m pytest -q test/registered/unit/disaggregation/test_dsv41_ced_pp_transfer.py \
  test/registered/unit/disaggregation/test_dsv41_prefill_tail_layout.py \
  test/registered/unit/disaggregation/test_ced_options.py \
  test/registered/unit/disaggregation/test_ced_tail_coalescing.py \
  test/registered/unit/disaggregation/test_mooncake_registration.py \
  benchmark/deepseek_v41_pd/test_launch.py benchmark/deepseek_v41_pd/test_oracle_pd_proxy.py \
  benchmark/deepseek_v41_pd/test_oracle_replay.py

# exact-output oracle qualification against a live pair
$PY benchmark/deepseek_v41_pd/oracle_pd_proxy.py --port 8034 \
  --prefill-url http://127.0.0.1:8031@8998 --decode-url http://127.0.0.1:8032 &

# probe/oracle/sweep validation for a pair on the 804x ports (PP2 recipes)
$PY benchmark/deepseek_v41_pd/validate_ced_variant.py <new-output-dir> [--sweep]
```

Real-mix sweeps are reproduced from the saved `launch-oracle.sh` and `run-sweep.sh` under `/data/dsv41-bringup/realmix/`. Those scripts were written for the `dsv4.1` branch; on `cc-main` launch through `tp2-pd.sh` with the serving flags plus `--oracle-commit-length 6` instead, and treat the result as a new baseline. Archive existing result files before re-running; never overwrite them.

## Shell and tooling traps

These are the V4.1-specific ones:

* **`MC_FORCE_TCP=0` still forces TCP.** Unset it; don't zero it.
* **`/health` on the router means nothing.** Use `/readiness` and a real completion.
* **Two prefill sessions per experiment.** Benchmark deployments live in tmux (`dsv41-oracle` + `dsv41-oracle-proxy`, `cedpp` + `cedpp-proxy`, `nodea` + `nodea-proxy`). Kill the proxy and the server separately, and only your own.
* **The 20K probe varies run to run.** Compare first-token IDs/log probabilities, and re-run before you call a continuation mismatch a regression.
