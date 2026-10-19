# DeepSeek-V4.1 PD recipes

Each recipe runs `benchmark/deepseek_v41_pd/launch.py` for one prefill/decode
pair on one node: prefill on the first two GPUs, decode on the last two.

| Recipe | What it runs |
| --- | --- |
| `tp2-pd.sh` | TP2 prefill + TP2 decode, DSpark on both. The control layout |
| `ced-pp2.sh` | CED split: TP1 x PP2 encoder-only prefill, TP2 decode with prompt-tail replay |
| `prefill-tail-pp2.sh` | Prefill-side tail: TP1 x PP2 prefill over all 40 layers, TP2 decode |
| `_common.sh` | Shared defaults, sourced by every recipe; not a recipe |

None of these recipes has run on a GPU on `cc-main`. Topology, flags, status
and every measured number, with provenance, are in
[`confai-docs/DEEPSEEK_V4.md`](../../confai-docs/DEEPSEEK_V4.md#launcher-and-recipes).
The operator guide is [Agent_Readme.md](Agent_Readme.md).

`ced-pp2.sh` and `prefill-tail-pp2.sh` are mutually exclusive. The launcher
sets their feature variables itself and clears inherited `SGLANG_*`, `SGL_*`,
`MC_*`, `MOONCAKE_*` and `NCCL_*` variables, so do not export them.

## Usage

```bash
# Print the resolved plan without touching CUDA.
recipes/deepseek-v41/ced-pp2.sh --dry-run

# Check hardware, ports and dependencies, then launch.
recipes/deepseek-v41/ced-pp2.sh --preflight-only
recipes/deepseek-v41/ced-pp2.sh
```

Extra arguments are appended after the recipe's flags and passed to
`launch.py`, so a later valued flag overrides the recipe's (for example
`--decode-requests 64` or `--context-length 131072`). Switches such as
`--ced-split` cannot be turned off this way. The recipes do not set the
real-mix serving values; the operator guide passes them explicitly. Run
`launch.py --help` for the full list.

Preflight requires the GPUs' DevTools mode to equal `--devtools-mode`
(default `off`). Pass `--devtools-mode on` on a node left in DevTools ON.

The defaults below can be overridden with environment variables:

| Variable | Default |
| --- | --- |
| `MODEL` | `/data/DeepSeek-V4.1-Flash` |
| `GPUS` | `4,5,6,7` |
| `PORT` | `8040` (router; prefill `+1`, decode `+2`) |
| `BOOTSTRAP_PORT` | `8999` |
| `NCCL_PORT` | `29600` (prefill; decode `+1`) |
| `ROUTER_METRICS_PORT` | `29001` |
| `OUTPUT_DIR` | `/data/dsv41-bringup/recipes/<recipe>-<UTC timestamp>` |
| `PYTHON` | `python3` (also the server interpreter unless `--python` is passed) |

These ports stay clear of a pair on GPUs 0–3 that uses the launcher defaults
(router 8030, bootstrap 8998, NCCL 29500, metrics 29000).

The exact-output oracle proxy for a pair on these ports:

```bash
python benchmark/deepseek_v41_pd/oracle_pd_proxy.py --port 8044 \
  --prefill-url http://127.0.0.1:8041@8999 --decode-url http://127.0.0.1:8042
```
