# DeepSeek-V4.1 prefill memory at 1M context

Per GPU, TP2 prefill. B200 numbers are measured; B300 numbers are the same
formula with B300's HBM. Startup prints the planned tile size (`DeepSeek-V4.1
index logits tiles: ... GiB from ... GiB free`) and refuses to start when the
eager indexer cannot fit.

## What CUDA-graph capture has to leave free

The eager index layers' fp32 logits tiles and their operands are allocated
outside every pool. `memory_budget.eager_indexer_peak_bytes` bounds one layer;
the largest is the candidate source (layer 20, ratio 1, all positions).
Measured peak on one B200 rank (DeepGEMM path, `--chunked-prefill-size 12288`
or `max_prefill_tokens 16384` rows):

| context | rows | 2 GiB tiles (before) | 2 GiB tiles | 1 GiB tiles | 0.5 GiB tiles |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128K | 12288 | 4.43 | 2.43 | 1.30 | 0.74 |
| 1M | 12288 | 4.82 | 2.82 | 1.70 | 1.13 |
| 1M | 16384 | 4.99 | 2.99 | 1.86 | 1.30 |

Two fixes produced the change: the candidate source now frees each logits tile
before scoring the next (it held two), and the index page-table expansion is
built in place as int32 (an int64 detour peaked at 2 GiB at 1M x 16384 rows).
Full top-k layers (2.0-2.1 GiB) and candidate consumers (<= 0.55 GiB) were
already bounded. Results are unchanged; tile size only changes how rows are
grouped.

Planning (`plan_dsv41_index_logits_budget`, after the last capture) needs

    free_after_capture >= SGLANG_DSV41_PREFILL_EAGER_RESERVE_GB (4)
                          + eager_indexer_peak_bytes(tile, rows, context_len)

and halves the tile from 2 GiB down to 128 MiB until it fits. At 1M context
and 16384 rows the 2 GiB-tile requirement is ~4 + 3.4 = 7.4 GiB. The 4 GiB
reserve is calibrated, not derived: it covers the rest of the eager forward
and allocator fragmentation (the chunk-16384 failure on 2026-10-18 had 4.4 GiB
cached but unusable). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
reduces fragmentation if the reserve proves short.

## Static memory (B200, measured 2026-10-19)

| item | prefill | decode |
| --- | ---: | ---: |
| HBM (torch total) | 179.1 GiB | 179.1 GiB |
| free before weights | 175.3 GiB | 175.3 GiB |
| target + DSpark draft weights | 146.8 GiB | 146.8 GiB |
| KV bytes per token (all pools) | ~3,960 B | ~2,110 B |
| between pool and capture | 0.3 GiB | 0.5 GiB |
| CUDA graphs | 12.7 GiB (prefill, chunk 12288) | 1.8 GiB (verify + draft) |
| free after capture today | 9.7 GiB | 18.1 GiB |

## Pool size from the formula

    pool_tokens <= (free_before_weights - weights - between - graphs
                    - free_after_capture_needed) / kv_bytes_per_token

with `free_after_capture_needed` = 7.4 GiB on prefill (1M, 2 GiB tiles), and
`--mem-fraction-static` high enough that `weights + pool` fits under it.

| GPU | prefill pool, 1M context | decode pool (8 GiB headroom) |
| --- | ---: | ---: |
| B200 (175.3 GiB free) | <= 2.2M tokens (today 1.57M) | <= 9.3M tokens (today 4.19M) |
| B300 (288 GB = 268.2 GiB; ~264.4 free) | <= 26M tokens | <= 54M tokens |

On B300 the KV pool stops being the limit for 1M context; host RAM is. Prefill
HiCache keeps `--prefill-hicache-ratio` (4) times the device pool in host
memory per GPU, so a 26M-token prefill pool would need ~390 GiB of host RAM per
GPU. Size the B300 prefill pool from host RAM (e.g. 8M tokens: ~30 GiB on the
GPU, ~118 GiB of host tier per GPU at ratio 4) and give the rest of the HBM to
CUDA-graph sizes, larger tiles, or decode.

The B300 figures assume B300's nominal 288 GB, the same per-GPU weight and
graph sizes, and the same eager reserve; re-read them from the first B300
startup log (the lines above) before fixing the recipe.
