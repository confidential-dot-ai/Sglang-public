# DeepSeek-V4.1 in containers

The V4.1 PD deployment as two containers from one image, each run by a systemd
unit. They replace the host venv (`/data/venvs/dsv41`), the host-venv
`dsv41-engram-cache` unit (saved in `/data/dsv41-bringup/host-units/`) and the
hand-started launcher.

| piece | what it is |
| --- | --- |
| `Dockerfile` | CUDA 13.0.3 devel (the host toolkit), Python 3.12, `requirements.lock`, this checkout's sglang (editable) and the V4.1 launcher |
| `requirements.lock` | the resolved environment for `python/pyproject.toml` plus Mooncake and the router (regeneration command in its header), installed with `--no-deps` |
| `build.sh` | builds `dsv41-pd:<sha>[-dirty-<hash>]` and `dsv41-pd:latest`, recording the source state for `launch.py` |
| `compose.yaml` | `engram-cache` (the holder), `pd` (one pair via `tp2-pd.sh`) |
| `dsv41.env` | GPUs, ports, image, and the serving flags |
| `dsv41-engram-cache.service` | runs the holder container; active once every shard is locked |
| `dsv41-pd.service` | runs the pair container; active once its smoke completion passed, relaunched 240 s after a failure |
| `wait-healthy.sh` | the units' readiness gate on the compose healthchecks |

## Host prerequisites

These stay on the host because a container cannot own them:

* NVIDIA driver, CC mode (ON, multi-GPU NVLE) and attestation, `nvidia-persistenced`.
* Docker Engine and the NVIDIA Container Toolkit, with the `nvidia` runtime
  registered (`sudo nvidia-ctk runtime configure --runtime=docker`).
* `kernel.yama.ptrace_scope = 0` in `/etc/sysctl.d/99-sglang-cuda-ipc.conf`
  (a host-wide setting; kept as it was).
* The checkpoint at `DSV41_MODEL` and the output root `DSV41_DATA`.

## Build, start, stop

```bash
cd /home/ubuntu/sglang-private
recipes/deepseek-v41/container/build.sh

sudo cp recipes/deepseek-v41/container/dsv41-{engram-cache,pd}.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dsv41-pd    # starts the holder first; returns once the pair is serving
sudo systemctl stop dsv41-pd            # the pair only; the holder keeps its pages locked
sudo systemctl stop dsv41-engram-cache  # stops the pair too, then frees ~189 GiB
```

`systemctl start dsv41-pd` blocks until `pd` is healthy: the launcher wrote
`smoke.json` after a real routed completion and the router's `/readiness` is 200.
The unit stays active exactly while the pair runs. When any worker dies the
launcher tears the pair down, the unit fails, and systemd starts a fresh pair
(new output directory) 240 s later, at most three times in four hours. A first
boot of a new kernel cache also compiles kernels into `--kernel-cache-dir`; on
B200 that took about an hour.

Logs: `journalctl -u dsv41-pd` has the launcher; worker logs are in the run's
output directory.

## Why it is shaped this way

* **The holder is its own container.** It must outlive every pair and keep
  ~189 GiB locked, so it gets `memlock: -1` and `IPC_LOCK`. Both containers
  bind-mount the checkpoint at the same path, so the workers read the holder's
  locked page cache; nothing is copied.
  Measured: preflight credited all 203.0 GB across the container boundary.
* **Host PID namespace.** `launch.py` credits the holder's locked pages by
  reading `/proc/<pid>/smaps` (`--resident-engram-pid`), and recognizes its own
  GPU processes by PID. The holder writes its PID to `/run/dsv41/engram.ready`
  (a shared tmpfs volume), which is also its healthcheck.
* **`pd` is privileged** for `--compact-host-memory`, which writes
  `/proc/sys/vm` before each worker's CUDA init (CC's cuInit needs free 2 MiB
  host blocks). It sees all GPUs so `GPUS` uses host indices, and uses host
  networking and IPC like the bare-metal launch.
* **systemd supervises, Docker does not.** Neither service has a Docker restart
  policy, so there is one supervisor. The pair lives and dies together, so the
  unit relaunches the whole pair after the GPUs settle, never one worker.

Outputs land in `DSV41_DATA/recipes/container-pd-<UTC>` with the usual
`launch-plan.json`, worker logs, `smoke.json` and server info. Files the
containers write are owned by root.

## Kernel caches

Two host directories keep a reboot or relaunch from recompiling:

* `--kernel-cache-dir` (`DSV41_DATA/container-cache/dsv41-pd-latest`): FlashInfer
  JIT ops and autotune results, Triton, TorchInductor. A cold one costs about an
  hour on B200, mostly `ptxas` on the TRT-LLM MoE routing kernels.
* `DSV41_DATA/container-cache/root-cache`, mounted at `/root/.cache`: SGLang's
  JIT kernels (`~/.cache/sglang/jit`) and torch extensions, a few minutes cold.

The image sets every file in `/opt/dsv41` to the epoch, because FlashInfer's
ninja cache compares source and object mtimes: without that, each image rebuild
reinstalls the wheels and recompiles every op. A FlashInfer, CUDA or flag change
still recompiles, as it should.
