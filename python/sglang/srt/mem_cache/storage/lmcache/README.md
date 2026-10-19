# LMCache Connector for SGLang

This document describes how to use LMCache as KV Cache Management Backend for SGLang engine.
For more details about LMCache, please refer to: https://lmcache.ai

## Install LMCache

### Method 1: with pip

```bash
pip install lmcache
```

### Method 2: from source

Clone LMCache project:

```bash
git clone https://github.com/LMCache/LMCache
```

Install:

```bash
cd LMCache
pip install -e . --no-build-isolation
```


## Use LMCache

SGLang uses `LMCacheUnifiedRadixCache` with LMCache's multiprocess connector.
The standalone LMCache daemon owns the external cache and can survive SGLang
process restarts. Daemon host and port come from the LMCache YAML config.

Terminal 1 — start the LMCache daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --l1-size-gb 4 \
  --eviction-policy LRU
```

Use the bundled `example_config_mp.yaml` (or any YAML setting `mp_host` / `mp_port`):

Terminal 2 — start SGLang:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_mp.yaml
```

For full LMCache config options see https://docs.lmcache.ai/api_reference/configurations.html.

### In-process and Hardwire mode

`--lmcache-mode ip` selects SGLang's in-process adapter for the unified radix
tree. `auto` keeps the multiprocess default, except that selecting Hardwire
host memory automatically selects in-process mode. Explicit `mp` plus
Hardwire is rejected because MP exports GPU tensors through CUDA IPC.

```bash
python -m sglang.launch_server --model-path MODEL \
  --enable-lmcache --lmcache-mode ip --lmcache-config-file lmcache-ip.yaml \
  --lmcache-host-memory-backend hardwire
```

Example `lmcache-ip.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 4
use_layerwise: false
async_loading: false
extra_config:
  save_only_first_rank: false
```

The adapter supports ordinary contiguous NHD MHA and MLA pools with
FP16/BF16/FP32 storage, all layers on each rank, and Full-only unified trees.
Chunk size must be divisible by SGLang's page size. SWA, Mamba, sparse-index
sidecars, packed FP4/FP8 layouts, pipeline/context parallelism and extra draft
KV pools need additional serialization support and are rejected. In particular,
this is not an LMCache path for MiniMax M3's sparse/InnerQ pool.

In-process transfers finish before admission returns. This preserves the
unified tree's private-slot, cancellation and store-lock lifecycle, but does
not reproduce the old layerwise overlap optimization. Layerwise, blending,
async-loading and P2P modes are rejected. TP uses independent storage on each
rank and a CPU process group for consensus; `save_only_first_rank` must be
false. Cache salt and extra-key isolation are propagated as LMCache request
tags for lookup, load and store.

The public LMCache SGLang API used here was inspected at
[`dc68527c76fd08c5501e19097973938143f06348`](https://github.com/LMCache/LMCache/tree/dc68527c76fd08c5501e19097973938143f06348).
It provides `LMCacheConnector`, `lmcache_get_config`, and the engine's
`lookup`/`lookup_unpin`/`retrieve`/`store` interfaces. MP mode still requires
CC main's LMCache package with `unified_lmcache_mp_connector`; that module is
absent from the inspected public revision. IP imports do not require it.

Hardwire additionally requires `hardwire_torch` and TD-private `doorbell`
tensors recognized as pinned by CUDA. Its allocator is installed before
LMCache creates the local CPU arena. Ciphertext `sealed` tensors are not local
cache storage. Native HiCache can also use Hardwire independently:

```bash
python -m sglang.launch_server --model-path MODEL \
  --enable-hierarchical-cache --hicache-host-memory-backend hardwire
```

Local adapter tests exercise real unified-tree ownership with an in-memory
external engine. They do not qualify the real LMCache/Hardwire extension under
confidential computing; that deployment still requires an integration run.
