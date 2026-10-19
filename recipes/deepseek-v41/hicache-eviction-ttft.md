# HiCache write-back on the DSV4.1 PD prefill critical path

Measured 2026-10-19 on the B200 CC VM (DevTools ON), image `release-bebe25661d`,
prefill TP2 with `--enable-hierarchical-cache --hicache-ratio 4
--hicache-io-backend direct --hicache-write-policy write_back
--hicache-mem-layout layer_first`, c24 router replay. Traces:
`/data/dsv41-bringup/ttft-profile-20261019T174152Z/` (`prefill/` torch
profiler, 11.5 s; `pyspy/prefill_tp0.txt`, 7,936 samples at 100 Hz).

## Where prefill's between-step time goes (py-spy, % of scheduler wall)

| Phase | % | Notes |
| --- | ---: | --- |
| `run_batch` (forward, host-blocking) | 57.9 | not HiCache |
| `get_next_disagg_prefill_batch_to_run` | 10.1 | below |
| — `prepare_for_extend` → `alloc_for_extend` → eviction | 4.0 | **HiCache write-back** |
| — `calc_priority` (HRRN) | 1.7 | |
| — `_sync_hicache_ready_counts` (TP CPU all-reduce) | 1.5 | HiCache, every call |
| — `init_next_round_input` | 1.2 | |
| `process_batch_result` → `checkpoint_kv_cache` (radix insert) | 5.1 | radix, not HiCache |
| `process_disagg_prefill_inflight_queue` → `checkpoint_into_tree` | 2.6 | radix |
| `ingest_requests` (`broadcast_pyobj`) | 2.5 | |

Radix insert self time is mostly `_insert_walk_step`; the per-node
`state.key = key[prefix_len:]` slice (`unified_tree_core.py:1357-1358`) is ~0.9%.

## The eviction bursts

Write-back backs a node up when it is evicted, on the scheduler thread:
`_evict_components` → `_execute_and_commit_kv_backup` → `writing_check(write_back=True)`
→ `start_writing` (index move, then `transfer_kv_direct` per pool) → blocking ack wait.

* Only 22 of 167 batch formations in the trace evicted anything, but those 22 took
  1,632 of the 1,917 ms spent in `get_next_batch_to_run` (26-132 ms each, 2-20
  nodes each). These are the long between-step gaps.
* Per node: 4 pools (`deepseek_v4_c1` 1 layer, `c1_indexer` 1, `c2` 3,
  `c2_indexer` 3), ~2 page runs each → ~16 D2H copies, plus ~5 device-index
  `.cpu()` moves in `move_indices`. The blocking ack waits total <1 ms per burst:
  the GPU is idle by then, the cost is host-side issue + Python.
* 872 `transfer_kv_direct` calls = 415 ms per 11.5 s under the profiler
  (inflated by profiler per-op overhead; py-spy puts all of eviction at 4.0%).

## What was tried: staged copy (gather on device, one D2H, host scatter)

Prototype behind `SGLANG_OPT_HICACHE_STAGED_DIRECT_BACKUP` (not committed):
per pool, `index_select` every layer's rows into a device staging buffer, one
D2H into a reused host buffer, numpy row-assign into the host pool; write path
skipped `move_indices` for those pools. Byte-identical to the direct copy in GPU
tests (aligned, chunked, CPU indices, unaligned fallback).

Measured no gain, so it is not shipped. Per evicted node (4 pools, 3 pages, 2
runs, `torch.set_num_threads(1)` as in the scheduler, 300 interleaved
iterations on GPU 7): direct median 2.38 ms, staged 2.73 ms. Primitives (host
time): async D2H 150 KB into the (driver-managed, CC) pinned host pool 74.5 us;
three async D2H + one sync 179.7 us; gather + synchronous D2H 450 KB 302.7 us;
`index_select` launch 18.1 us; idle stream sync 2.4 us. Async copies cost
~60 us each, so ~16 per node is ~1 ms; a synchronous D2H waits for the gather
and the CC bounce and costs more than the async copies it replaces.

Two traps found on the way, relevant to any host-side copy work under CC:
* A torch CPU `index_copy_`/slice `copy_` at the default 96 intra-op threads on
  a loaded host took 8-17 ms for 450 KB (8 threads: 0.05 ms; 1 thread: 0.3 ms;
  numpy row assignment: 0.02 ms). Use numpy or one thread for small host copies.
* D2H into freshly allocated pageable memory pays first-touch faults; reuse
  a touched buffer.

## What would actually move it

1. **A/B the policy before code.** The cost is eviction-time backups of nodes
   whose host copy may never be read. Compare `--hicache-write-policy
   write_through_selective` (backs up only hot nodes, at insert time) and HiCache
   off at matched c24/c32 with `--enable-request-time-stats-logging`, and read
   the host-tier hit rate (`load_back` volume) to see what the tier buys. Old-image
   points: `container-tpm-nohicache-c24-20261018T155824Z` 4.33M, TTFT
   1.14/4.08/5.96 s vs HRRN+HiCache 4.26M, 1.34/3.01/6.59 s (different days).
2. **Issue the backups off the scheduler thread.** Correctness needs every
   backup copy *enqueued* before `run_batch` launches kernels that overwrite the
   freed slots (the allocator's `set_hicache_transfer_done_event` orders the GPU
   side only for copies already enqueued), and a later `load_back` must wait on
   the backup's finish event. A worker thread could run `start_writing` while the
   scheduler finishes `prepare_for_extend`, with a join before `run_batch`. The
   available overlap is the rest of batch formation, a few ms, against ~1-2 ms
   per evicted node; it needs a GIL check on `transfer_kv_direct` (a custom op)
   and a CC side-stream hang review (see the 2026-10-17 engram side-stream hang).
3. **Skip `_sync_hicache_ready_counts` when no op is in flight** (~1 ms per batch
   formation). Only safe if the skip predicate is identical on every TP rank; a
   mismatch deadlocks the collective, so it needs rank-symmetric inputs only
   (queue lengths, never event readiness) and must keep the reclaim digest check.
