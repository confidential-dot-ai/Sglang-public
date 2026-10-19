This is the deployment guide for DeepSeek-V4.1-Flash on our 8x B300 confidential-computing node. It covers the design, how to reach and check the deployment, its failure modes, and instructions to hand to your agent when something goes wrong. **V4.1 is not a public endpoint yet**: it is a loopback-only deployment. Every statement below comes from the bring-up work or from the launcher code. Anything not yet decided is marked **TBD**; do not fill those in from memory.

Background, the recipe table and the knob reference: [confai-docs/DEEPSEEK_V4.md](../../confai-docs/DEEPSEEK_V4.md).

## Design

### Status and access

| item | state |
| --- | --- |
| public base URL | **TBD**: none exists. Nothing is exposed off the node |
| serving endpoint | `http://127.0.0.1:8030/v1`, model **`dsv41`** (router, loopback only) |
| auth | **none**. Router, workers and bootstrap bind `127.0.0.1`; the router has no API key configured. The router's Prometheus listener is the exception (see [Metrics](#metrics)) |
| TLS / front door / readiness service | **TBD**: no Caddy, `node-front` or `node-readiness.py` equivalent |
| monitoring / paging | **TBD**: no Better Stack monitors; router Prometheus on `:29000` only |
| qualification | **unqualified**. Bring-up and correctness probes only |

Before this becomes a customer endpoint, every TBD row needs a real answer. See [Before go-live](#before-go-live).

**One PD pair, no redundancy.** A V4.1 serving unit is one *pair*: a TP2+EP2 **prefill** worker on GPUs 0–1 and a TP2+EP2 **decode** worker on GPUs 2–3, behind one router. A request needs **both** roles: prefill computes the prompt, Mooncake (`--disaggregation-transfer-backend mooncake`, intra-node NVLink) moves its KV to decode, and decode generates. **Losing either V4.1 worker takes the pair down**, and the launcher tears down all of its processes when any one exits (see [Hard rules](#hard-rules)).

**Why PD and TP2.** The TP2 weights and KV pools fit, and splitting prefill from decode keeps long prefills from stalling decode. GPUs 4–7 can hold a second pair (see [Configurations](#configurations)).

**Why the Engram tables live in CPU RAM, shared.** The checkpoint carries two Engram embedding tables (layers 1 and 14). One copy is 188.8 GiB, so two copies need 377.7 GiB, which is **more than the VM's 376.7 GiB of RAM**. On top of that, this CC VM cannot register host memory for GPU access: `cudaHostRegister` returns error 801 and every GPU reports `host_register_supported=0`. So:

* `dsv41-engram-cache.service` (`hold_engram_cache.py`) maps the two Engram shards **read-only** and `mlock`s their page-cache pages once (203,073,077,576 bytes). Every worker shares those pages.
* Workers run `--cpu-engram-staging`: the CPU gathers the selected FP8 rows from the shared mapping and copies only those compact rows to the GPU. This is an eager graph break inside otherwise captured CUDA graphs (`--split-cuda-graphs`).

### Components

| component | where | port | what it does |
| --- | --- | --- | --- |
| **dsv41-engram-cache.service** | node | — | runs the Engram holder container, which keeps the Engram pages locked in RAM for every worker. Must outlive any pair |
| **dsv41-pd.service** | node | — | runs the pair container: `benchmark/deepseek_v41_pd/launch.py`, which supervises prefill, decode and router. Active only while the pair serves; relaunches the whole pair 240 s after a failure |
| **router** (`smg`, PD mode) | node | 8030 | prefill → transfer → decode routing; `/readiness` requires both PD roles; Prometheus on `:29000` (all interfaces) |
| **prefill worker** | node | 8031 | TP2+EP2 on GPUs 0–1, Mooncake bootstrap on 8998 |
| **decode worker** | node | 8032 | TP2+EP2 on GPUs 2–3, DSpark speculation |

Both units are enabled at boot and run containers from one image (`dsv41-pd`); see [container/README.md](container/README.md). GPUs, ports and serving flags come from `container/dsv41.env`, which runs the pair on GPUs 4–7 with router 8040, prefill 8041, decode 8042, bootstrap 8999, NCCL 29600 and metrics 29001. The ports above (8030–8032, 8998, NCCL 29500, metrics 29000) are the launcher defaults for a pair on GPUs 0–3; substitute the container's ports in the commands below when checking the container pair.

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

**None exist yet.** The router runs without an API key and every port is loopback-only, which is safe only while nothing is exposed. Before any exposure, an inference key, an admin key separate from it, and metrics auth are **TBD**. The intended split: the router enforces auth inside the guest, and TLS terminates outside it. **Do not paste credentials into this file**; reference where they are stored.

---

## Layout

| item | value |
| --- | --- |
| hardware | 8x B300 SXM6, 275,040 MiB HBM each, NV18; 376.7 GiB host RAM; confidential VM |
| CC state | CC ON, multi-GPU NVLE; all 8 GPUs passed the local verifier (`/data/dsv41-bringup/attestation-persistent.log`). Qualification requires DevTools OFF |
| driver | 595.91.07 |
| node hostname / SSH | **TBD** |
| repo | this checkout |
| Python env | the `dsv41-pd` image: `/data/venvs/dsv41` (Python 3.12) frozen into `container/requirements.lock`. The host venv stays for tests and probes. **Separate from other serving envs; never upgrade it in place** |
| model | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`, at `/data/DeepSeek-V4.1-Flash` (510.3 GB, 48 shards, checksums verified) |
| run artifacts | `/data/dsv41-bringup/` (attestation, preflight, every run's logs, `smoke.json`) |

Model facts that people get wrong: 40 target layers. Engram layers are **1 and 14** (zero-based). Layer **20 is the last shared-KV source**, not the end of the model. Weights are MXFP8 dense with MXFP4 experts, and the KV cache is FP8. The DSpark draft head is **bundled in the target checkpoint**; no separate draft path is needed.

## Configurations

Recipes live next to this file and wrap `benchmark/deepseek_v41_pd/launch.py`. [README.md](README.md) covers usage and port overrides; the recipe table (topology, ports, transfer, speculation, status) is in [DEEPSEEK_V4.md](../../confai-docs/DEEPSEEK_V4.md#launcher-and-recipes). Operationally:

* `tp2-pd.sh` is the layout of the serving pair; the container runs it with the flags below.
* Decoder SWA bounded replay is upstream's implementation. The launcher passes `--enable-decoder-swa-bounded-replay` to decode and `--no-enable-decoder-swa-bounded-replay` to prefill (upstream defaults it on); `--prefill-bounded-replay` turns it on for prefill too. `--ced-split` is rejected: the CED encoder and tail graphs were not ported onto the upstream model.

### Serving config

These are `DSV41_LAUNCH_ARGS` in `container/dsv41.env`, whose comments record the measurement behind each one.

| setting | value | why |
| --- | --- | --- |
| `--context-length` | 131,072 | agent-workload prompts average ~39K tokens |
| `--chunk-size` | 8192 | fewer prefill batches per cache miss than 4096 |
| `--prefill-requests` / `--prefill-schedule-policy` | 16 / `hrrn` | requests still sending KV keep their slot, so a cap of 8 admitted only 3–4 per batch; HRRN stops cached turns queueing behind bursts of 38K-token misses |
| `--decode-requests` / `--decode-graph-extra-bs` | 64 / 10,12,14 | decode graph buckets 1–64, plus 10/12/14 so 9–15 requests do not pad to 16 |
| `--decode-polling-interval` | 2 | decode polls KV transfers every second step |
| `--tokenizer-workers` | 4 | the router sends chat text, and one process tokenizing ~38K-token prompts capped the pair |
| `--prefill-swa-prefix-tails` | 256 | **the cache-hit fix**. The default (32 tails = 4× 8 running requests) evicted prefixes and held the hit rate near 56–61%; 256 brought it to ~95% |
| `--max-total-tokens` (prefill) | 1,572,864 | at 1M tokens the prefix cache of more than 24 ~39K-token conversations thrashed |
| `--prefill-hicache-ratio` | 4 | HiCache host tier 4× the GPU pool (~6M tokens, ~56 GB host RAM), direct IO + write-back; cache-thrash reloads KV from host instead of recomputing |
| `--decode-max-total-tokens` | 4,194,304 | decode KV pool; V4.1 KV is ~1.6 KB/token |
| `--mem-fraction` | 0.92 | |
| `--custom-all-reduce` | on | CustomAllReduceV2; preflight probes it on each TP2 pair. FlashInfer all-reduce fusion stays off |
| `--prefill-graph-max-seq-len` | 65,536 | prefill graphs replay up to 64K tokens; the ratio-1/2 index layers run as eager breaks |
| prefill/decode env | `SGLANG_OPT_DSV41_ENGRAM_STAGING_PREFETCH=1`, `SGLANG_DSV41_ENGRAM_GATHER_THREADS=8`; prefill `NCCL_PROTO=LL128` | Engram gathers ahead of their layers; LL128 is bit-exact and fastest at prefill's 1.5K–8K-token messages |
| graphs | `--split-cuda-graphs --cpu-engram-staging` | breakable graphs around the eager Engram gather |
| DSpark | on, block size 5 (six verify positions) | `--no-dspark` is the matched control |
| transport | `--disaggregation-transfer-backend mooncake`, intra-node NVLink | the launcher's default `--transport intra-node-nvlink` |
| startup | `--compact-host-memory`, `--kernel-cache-dir` keyed to the image | CC CUDA init needs free 2 MiB host pages; the kernel cache avoids recompiling |

---

## Metrics

Only the router exports metrics: Prometheus format on port `29000` (`29001` for a second pair). `launch.py` passes `--prometheus-port` but not `--prometheus-host`, and `sglang_router` defaults that host to `0.0.0.0`, so this listener is **not** loopback-only. Firewall it or pass a loopback host before the node is exposed. **Workers expose none** (`/metrics` is off). No log exporter, Prometheus server, or dashboards exist for V4.1 yet: **TBD**.

```bash
curl -s http://127.0.0.1:29000/metrics | grep -E '^smg_worker_health\{'   # both roles must be 1
```

Per-batch KV usage, queue depth, cache hit and DSpark accept length are only in the worker logs under the run's output directory: `<output-dir>/prefill.log` and `decode.log`. For the router's `smg_*` histograms, query `_bucket`, never the bare name.

---

## Operational rules

* **The pair lives and dies together.** The launcher polls prefill, decode and router every 2 s. If any of them exits, it SIGTERMs and then SIGKILLs every PID it owns, and only those. Never "restart just decode".
* **Never kill the Engram holder while a pair is running.** Workers would start faulting the 203 GB tables from disk. Stop it (`sudo systemctl stop dsv41-engram-cache`) only after every pair is down; the unit dependency stops `dsv41-pd` first. The pair container passes the holder's PID to the launcher as `--resident-engram-pid`, so the launcher credits the locked memory and never signals that process; a hand launch must pass it.
* **Checkpoint files are immutable while serving.** The workers and the holder map them. Never rewrite, re-download or `mlock` a writable mapping of them.
* **One output directory per launch.** `launch.py` refuses to reuse one. Every run keeps its own logs, `server_info`, manifest and `smoke.json`.
* **Never set `MC_FORCE_TCP` unless you mean TCP.** Mooncake checks only that the variable is *present*, so even `MC_FORCE_TCP=0` forces TCP. The launcher clears inherited `SGLANG_*`/`SGL_*`/`MC_*`/`MOONCAKE_*`/`NCCL_*` tuning variables; pass settings through its arguments, not your shell.
* **Another pair or experiment may be on the other four GPUs.** Check `nvidia-smi` and the tmux sessions before touching anything.
* **Never two load sources against a pair.**

## Known issues

* **Unexplained streaming read timeouts.** A few 180 s client timeouts appeared at c54–c56 in bring-up load tests. The servers logged no error and no scheduler or transfer exception. Possibly a PD hand-off stall; the root cause is open.
* **20K continuation drift.** Four of five saved 8K–100K probe continuations match exactly, while the 20K one varies, even between two eager runs. First-token IDs and log probabilities always match. This is a reproducibility gap; bitwise equivalence is not claimed.
* **Short prompts prefill eagerly.** The smallest prefill graph bucket is 256 tokens, and the runner rejects excessive padding. Decode still uses graphs.
* **Long prompts prefill eagerly by default.** `deepseek_v4_hook.py` caps the prefill graph `max_seq_len` at 16,384 and logs that longer contexts run eager. `--prefill-graph-max-seq-len N` (production: 65,536; needs `--split-cuda-graphs`; not validated with `--prefill-bounded-replay`) runs the ratio-1/2 index layers as an eager break so longer contexts replay graphs.
* **Boot persistence is configured but not yet exercised by a reboot.** `dsv41-engram-cache` and `dsv41-pd` are enabled container units. The `nvidia-persistenced` override (`/run/systemd/system/nvidia-persistenced.service.d/dsv41-recovery.conf`), where a node has one, still lives in `/run` and is lost on reboot.
* **A native `/generate` request carrying `forced_token_ids` crashes DSpark decode, and the pair with it.** The router does not forward that field; direct worker ports do, so keep the worker ports unexposed.
* **Driver mismatch after a restart.** After the data-disk restart, the loaded module (595.71.05) disagreed with NVML and the on-disk module (595.91.07), and `nvidia-smi` failed. Check `nvidia-smi` works before any launch.
* **First boot downloads FlashInfer cubins** during tuning. Point `--kernel-cache-dir` at a compatible previous run to reuse compiled kernels.
* **Mooncake links CUDA 12.** The image sets `LD_LIBRARY_PATH` to its `nvidia/cuda_runtime/lib`; a host-venv launch needs `LD_LIBRARY_PATH=/data/venvs/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib`.

## Before go-live

None of these exist for V4.1. Each needs an owner and a decision before customer traffic:

| need | V4.1 |
| --- | --- |
| public hostname + TLS | **TBD** |
| auth | **TBD**: router-enforced inference key + separate admin key |
| readiness | **TBD**; router `/readiness` is the only signal |
| worker metrics | **TBD** |
| paging | **TBD**: an external real-completion check |
| boot persistence | enabled container units (`container/`); not yet tested across a reboot |
| redundancy | one PD pair; a second pair on the other four GPUs is the obvious step |
| admission cap | **TBD**: stay below the concurrency where the prefill pool thrashes |
| qualification | **TBD** |

## Files

* `recipes/deepseek-v41/`: launch recipes and their README.
* `benchmark/deepseek_v41_pd/`:
  * `launch.py`: the supervisor;
  * `hold_engram_cache.py`: the Engram residency holder;
  * `probe_engram.py` / `probe_staged_engram.py` / `probe_custom_all_reduce.py`: preflight probes;
  * `download_model.py`: the pinned checkpoint download;
  * `README.md`: launcher usage and arguments.
* `/data/dsv41-bringup/`: every run's artifacts.

---

# Hand this to your agent

Everything below is written for an agent or engineer operating the V4.1 deployment with **no prior context**. Read the rules before the procedures.

## Access

Node hostname and SSH route: **TBD**. Once you are on the node, everything is local:

```bash
cd <this checkout>
PY=/data/venvs/dsv41/bin/python
systemctl status dsv41-engram-cache dsv41-pd --no-pager
sudo docker ps -a             # dsv41-engram-cache-1 and dsv41-pd-1, with health
tmux ls                       # host-venv launches run in named tmux sessions
```

## Hard rules

1. **Never restart one worker.** A PD pair is prefill + decode + router under one `launch.py`. Any exit tears down the whole pair. Recover by relaunching the pair.
2. **Never stop `dsv41-engram-cache` while any pair runs**, and never pass its PID to anything that signals process groups.
3. **Kill explicit PIDs only.** SIGTERM, wait, then SIGKILL. The launcher already does this for the PIDs it owns. If you must clean up by hand, enumerate the GPU PIDs on *your* pair's GPUs from `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv`. Do not touch PIDs on other GPUs.
4. **Check who owns the other four GPUs first.** They may run a second pair or an experiment from another worktree.
5. **Never reuse an output directory or edit a running launcher script.** Write a new script to a temp file and `mv` it; bash reads running scripts by byte offset.
6. **Never set `MC_FORCE_TCP`, and never upgrade `/data/venvs/dsv41` in place.**
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
| `Xid 120` → `154` | GPU reset required; it cannot be cleared inside the guest. Guest stop/start |
| Mooncake `operation not supported` / native crash in transfer | thread-device regression in the intra-node path (fixed in `mooncake_transfer_engine.py`). Check the image is built from this branch |

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
cd <this checkout>
export LD_LIBRARY_PATH=/data/venvs/dsv41/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
ENGRAM_PID=$(systemctl show -p MainPID --value dsv41-engram-cache)

# always dry-run first: prints the full plan, touches no GPU
PYTHON=/data/venvs/dsv41/bin/python GPUS=0,1,2,3 PORT=8030 BOOTSTRAP_PORT=8998 \
  NCCL_PORT=29500 ROUTER_METRICS_PORT=29000 \
  OUTPUT_DIR=/data/dsv41-bringup/pd-$(date -u +%Y%m%dT%H%M%SZ) \
  recipes/deepseek-v41/tp2-pd.sh --dry-run \
  --resident-engram-pid "$ENGRAM_PID" --parallel-load \
  $(grep '^DSV41_LAUNCH_ARGS=' recipes/deepseek-v41/container/dsv41.env | cut -d= -f2-)
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
3. Wait for the GPUs to settle before relaunching: an early relaunch has caused `Xid 120 → 154` on this hardware. Wait ≥180 s, then 60 s of zero D-state.
4. Relaunch with a **new** output directory (procedure above).

## Running the test suites

```bash
# CPU unit tests for the V4.1 PD/PP code and the launcher (no GPU needed)
$PY -m pytest -q test/registered/unit/disaggregation/test_dsv41_ced_pp_transfer.py \
  test/registered/unit/disaggregation/test_dsv41_prefill_tail_layout.py \
  test/registered/unit/disaggregation/test_ced_options.py \
  test/registered/unit/disaggregation/test_ced_tail_coalescing.py \
  test/registered/unit/disaggregation/test_mooncake_registration.py \
  benchmark/deepseek_v41_pd/test_launch.py
```

## Shell and tooling traps

These are the V4.1-specific ones:

* **`MC_FORCE_TCP=0` still forces TCP.** Unset it; don't zero it.
* **`/health` on the router means nothing.** Use `/readiness` and a real completion.
* **The 20K probe varies run to run.** Compare first-token IDs/log probabilities, and re-run before you call a continuation mismatch a regression.
