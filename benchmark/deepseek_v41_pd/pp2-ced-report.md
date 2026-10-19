# DeepSeek-V4.1 Flash CED PP2 validation

Worktree: `pp-encoder`. Run artifacts: `/data/dsv41-bringup/ced/pp2`.
Topology: GPUs 4–5 run TP1/PP2 encoder prefill (layers 0–7 | 8–20);
GPUs 6–7 run TP2/EP2 decode with DSpark and CED prompt-tail replay.
Router 8040, prefill 8041, decode 8042, oracle proxy 8044, bootstrap 8999.

## Findings

- The inherited run reached prefill graph capture and warmup, then failed PD
  smoke in Mooncake's transfer thread (`IndexError` indexing destination
  pointers). DeepSeek-V4.1's source-owned ratio-2 KV/index entries were not
  publishing layer IDs. Its ratio-2 pending-pair state also used a legacy
  `C128_STATE` PP slicer that counted only ratio-128 layers. The pool now
  publishes IDs for KV/index, SWA/compressor, and request-scoped state entries;
  generic state transfer forwards these IDs into the existing entry-pairing
  planner. Both decode TP ranks receive the appropriate PP-local source entries.
- Single-Pass mHC sends `prev_pre` across the PP boundary, but the prefill
  graph buffers omitted it. Graph replay could therefore silently discard the
  previous stage's coefficients. The prefill graph runner now allocates and
  refreshes float32 `[tokens, hc_mult]` coefficients alongside flattened hidden
  states. V4.1 uses a distinct HF config class, so the allocation gate recognizes
  both V4 and V4.1 configurations.
- CPU tests with `CUDA_VISIBLE_DEVICES=""` failed during test utility import:
  default port selection indexed the empty string. The test harness now uses
  GPU index zero for its port calculation when no GPU is visible.
- An existing graph eligibility test fixture lacked `model_runner.attn_backend`;
  the fixture now supplies it without changing runtime eligibility.

## CPU validation

With `CUDA_VISIBLE_DEVICES=""` and this worktree on `PYTHONPATH`:
109 tests and 26 subtests passed across the CED transfer, hybrid PP transfer,
wire protocol, DSpark PD, prefill graph helpers, prefill graph runner, and graph
buffer registry modules. Output: `cpu-tests.log` in the artifact directory.
The final rerun passed the same 109 tests and 26 subtests in 14.50 seconds;
output is in `cpu-tests-final.log`. The mHC regression test was also run with the
new buffer allocation removed and failed, then passed with the allocation
restored. Removing layer-ID forwarding in a separate CPU process reproduced
the pending-pair transfer `IndexError`; the fixed tests passed.
Artifacts: `transfer-tests.log`, `transfer-negative-test.log`.

## Runtime validation

PD smoke passed at 04:05 UTC on 2026-09-29.

All five uncached probes (8K, 20K, 40K, 70K, 100K) completed using graph replay
on both PP stages. Four of five 24-token continuations match each reference
exactly. At 20K the first differing token is token 2 against CED TP2 and token 3
against full prefill (one-based). All five first-token IDs match both references.
Maximum absolute first-token logprob difference: 0.00193975 against CED TP2 and
0.0268651 against full prefill. This is not bitwise numerical equivalence.
Artifacts: `probe1.json`, `probe1.log`, `probe-comparison.json`.

Exact oracle output passed three times (153 forced tokens each), including two
concurrent requests and a cached-prefix repeat (768 cached tokens). The first
qualification invocation raced proxy startup and was retried after it listened;
that connection-refusal log is preserved as `qualify-proxy-startup-race.log`.
Successful results are in `qualify.log` and `oracle-smoke-{0,1}.json`.

The concurrency 56/72 sweep completed after a successful prefill cache flush.
The baseline protocol was unchanged: 240-second ramp, 300-second warmup,
300-second measurement, 256 conversations, up to 25 turns, streaming oracle
replay, and six-token commits. Both post-sweep readiness checks returned two
healthy workers. A post-load oracle rerun passed all three exact-output checks
(`qualify-post-sweep.log`).

## Fixed-window sweep

| Run | Concurrency | Served TPM/GPU | TTFT p95 | Slowest-5% user tok/s | Cache hit | Completed / failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Replica A baseline | 56 | 3.01M | 2.48s | 106 | — | — |
| CED PP2 | 56 | 1.457M | 2.55s | 25.1 | 95.6% | 673 / 2 |
| Replica A baseline | 72 | 3.46M | 3.74s | 88 | — | — |
| CED PP2 | 72 | 1.294M | 42.12s | 19.3 | 86.3% | 646 / 0 |

Detailed metrics: `c56.json`, `c72.json`, `results-summary.json`, and
`results-summary.md` in the artifact directory. Baseline values above are the
provided replica A results. PP2 served TPM is approximately 52% / 63% lower,
but the c72 cache mix is outside the prescribed comparison band.

## Open issues

- Performance remains unqualified. c56's scorecard fails for two streaming read
  timeouts (conversation 50 turn 5 and conversation 29 turn 16). No scheduler or
  transfer-thread exception accompanied them. The saved replica A c56 run also
  has one read timeout, on a different conversation; this does not establish a
  common cause. The timeout root cause remains unresolved.
- c72 had no failed requests, but cache hit fell below the 94–96% target band to
  86.3%, with a late burst of uncached prefills. Its 5.00s TTFT p50 also fails the
  3s interactive target. Cache residency and the late cache-hit decline need
  further investigation; the sweep alone does not establish a root cause.
- Both runs miss the 60 tok/s slowest-user threshold and trail baseline
  throughput. Decode prompt-tail replay and scheduling need profiling before
  attributing the slowdown to a particular component.
- The 20K continuation is not identical to either reference. Four other probes
  and all first-token IDs match; no claim of bitwise numerical equivalence is
  made. TP2 CED and non-CED protocol paths have CPU regression coverage; they
  were not relaunched for a separate GPU sweep during this task.

## Final running state

The PP2 replica remains running in `cedpp`, with the oracle proxy in
`cedpp-proxy`. Router `http://127.0.0.1:8040/readiness` reports two healthy
workers. GPUs 4–7 are the only GPUs used by this run. Protected replica A PIDs
228154, 228155, 228156, 228157, and 236015 remained alive at the final check;
no protected processes or tmux sessions were signalled.

## Changed files

- `python/sglang/srt/disaggregation/mooncake/conn.py`
- `python/sglang/srt/disaggregation/utils.py`
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- `python/sglang/srt/model_executor/runner_utils/buffers.py`
- `python/sglang/test/test_utils.py`
- `test/registered/unit/disaggregation/test_disaggregation_wire.py`
- `test/registered/unit/disaggregation/test_dsv41_ced_pp_transfer.py`
- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py`
- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py`
- `benchmark/deepseek_v41_pd/pp2-ced-report.md` (this report)

## Local commits

- `8732a23879 Fix CED PP graph mixing coefficients and layer-addressed KV transfer`
- `3a6ea2c5f8 Update prefill graph eligibility fixture for backend hook`
- `1abe9b713a Recognize V4.1 config when allocating PP mixing coefficients`
- `0e96871994 Reproduce ratio-2 PP state slicing failure in regression fixture`
