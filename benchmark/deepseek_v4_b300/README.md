# DeepSeek V4 Flash 0731 on B300/GB300 CC

This directory implements the reproducible launch, measurement, and tuning
parts of the DeepSeek V4 CC performance plan. It deliberately starts from the
official integrated TP4+DSpark configuration. DP Attention, context
parallelism, PD disaggregation, conditional decode, and TBO are not control
defaults.

## Dry-run the tuner

Dry-run does not inspect GPUs or select a performance winner. The built-in
collective shapes are orchestration seeds only, not a DSV4 production profile.

```bash
python3 benchmark/deepseek_v4_b300/tune.py \
  --dry-run --quick --allow-non-b300 --no-require-cc \
  --no-require-attestation \
  --output-dir /tmp/dsv4-b300-tune-plan
```

`profile.json` records the proposed commands. `profile.env` contains only
allowlisted launch variables and retains the fail-closed FP32/default-indexer
control with TBO disabled until measured artifacts select something else.

## Generate the staged experiment matrix

After the workload, refresh probability, capacity, and attestation bundle are
frozen, generate the complete run ledger. It records hashes for all immutable
inputs and expands three artifact paths per candidate. The stages preserve a
one-variable-at-a-time parent relationship; TBO is marked as blocked until its
live auxiliary-state parity gate passes.

```bash
python3 benchmark/deepseek_v4_b300/experiment_matrix.py \
  --run-root /home/ubuntu/logs-dsv4-b300/matrix \
  --dataset /path/to/dsv4-tokenized-workload.json \
  --workload-manifest /path/to/dsv4-workload-manifest.json \
  --attestation-evidence /path/to/verified-attestation-bundle.json \
  --max-total-tokens REPLACE_WITH_MEASURED_CAPACITY \
  --refresh-prob REPLACE_WITH_CALIBRATED_VALUE \
  --out /home/ubuntu/logs-dsv4-b300/matrix.json
```

The matrix emits argv arrays and environments; it intentionally does not
start, stop, or overlap candidates. Inspect the node before each launch and
shut down only the processes owned by the preceding candidate.

Hashing an attestation file is only provenance; platform verification of that
evidence is still required.

## Launch

Choose an explicit KV/token capacity from a measured startup ladder. The
launcher refuses `auto` so scored runs cannot silently change resident
capacity. Qualification also requires a non-empty attestation artifact; its
path and SHA-256 are recorded, while semantic attestation verification remains
the responsibility of the platform verifier.

```bash
MAX_TOTAL_TOKENS=REPLACE_WITH_MEASURED_CAPACITY \
ATTESTATION_EVIDENCE=/path/to/verified-attestation-bundle.json \
LOGDIR=/home/ubuntu/logs-dsv4-b300-tp4 \
  bash benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4.sh
```

After one TP4 worker passes correctness and capacity gates, launch two
replicas on the full eight-GPU node:

```bash
MAX_TOTAL_TOKENS=REPLACE_WITH_MEASURED_CAPACITY \
ATTESTATION_EVIDENCE=/path/to/verified-attestation-bundle.json \
LOGDIR=/home/ubuntu/logs-dsv4-b300-tp4x2 \
  bash benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4_dp2.sh
```

The single-TP4 control uses GPUs 0-3. The full-node launcher uses GPUs 0-3 and
4-7, ports 30300/30301, and a cache-aware router on 8030. Qualification traffic
should use the direct worker URLs so placement is deterministic and
conversation-sticky.

After the ordinary TP-attention controls pass, run the isolated DP-attention
topology candidate on the same two TP4 islands:

```bash
DSV4_ENABLE_DP_ATTENTION=1 \
MAX_TOTAL_TOKENS=REPLACE_WITH_MEASURED_CAPACITY \
ATTESTATION_EVIDENCE=/path/to/verified-attestation-bundle.json \
LOGDIR=/home/ubuntu/logs-dsv4-b300-tp4x2-dpa4 \
  bash benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4_dp2.sh
```

This one switch expands to `--dp 4 --enable-dp-attention`, the required DP
lm-head, local-control broadcast, and `--moe-a2a-backend none`. It deliberately
does not enable TBO or MegaMoE, so each remains an independent experiment. Run
the target-only leg first with `DSV4_ENABLE_DSPARK=0`; DSpark may advance only
after target graph/eager parity passes.

For the matched target-only control, use the same launcher and settings with
`DSV4_ENABLE_DSPARK=0`. Do not change capacity, graph, workload, or admission
settings between the target-only and DSpark legs.

`DSV4_ENABLE_TBO=1` enables both ordinary TBO and DeepSeek V4 TBO CUDA-graph
capture. Set `DSV4_ENABLE_TBO_CUDA_GRAPH=0` only for an explicitly labeled
eager-only TBO diagnostic. The resolved graph setting is recorded in the
launch manifest and is allowlisted in generated tuning profiles.

For an explicitly non-qualification development launch, all three safeguards
must be disabled deliberately:

```bash
REQUIRE_CC=0 REQUIRE_ATTESTATION=0 ALLOW_NON_B300=1 \
MAX_TOTAL_TOKENS=REPLACE_WITH_EXPLICIT_CAPACITY \
  bash benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4.sh
```

## Correctness and scorecard

Run the probe against both workers and keep separate target-only and DSpark
artifacts:

```bash
python3 scripts/dsv4_correctness_probe.py \
  --base-url http://127.0.0.1:30300 \
  --model dsv4-flash-0731 \
  --tag dspark-worker-a \
  --out /home/ubuntu/logs-dsv4-b300/probe-worker-a.json
```

Compare each deterministic DSpark artifact against its matched target-only
artifact. The comparison includes finish reason, text, and structured tool
calls:

```bash
python3 benchmark/deepseek_v4_b300/compare_probes.py \
  --reference /path/to/target-only-worker-a.json \
  --candidate /path/to/dspark-worker-a.json \
  --out /path/to/target-vs-dspark-worker-a.json
```

The scorecard requires a DSV4-tokenized dataset, a refresh probability
calibrated from server counters, and an immutable workload manifest. Create the
manifest after calibration; it records the dataset hash, exact model/tokenizer
revisions, and all traffic-shape inputs:

```bash
python3 benchmark/deepseek_v4_b300/workload_manifest.py create \
  --dataset /path/to/dsv4-tokenized-workload.json \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --model-revision REPLACE_WITH_COMMIT \
  --tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tokenizer-revision REPLACE_WITH_COMMIT \
  --n-convs 256 --concurrency 256 --max-turns 25 \
  --warmup-seconds 60 --measurement-seconds 600 --physical-gpus 8 \
  --cache-hit-min 0.955 --cache-hit-max 0.965 \
  --output-share-min 0.007 --output-share-max 0.008 \
  --refresh-prob REPLACE_WITH_CALIBRATED_VALUE \
  --out /path/to/dsv4-workload-manifest.json
```

Throughput thresholds default to zero until a DSV4 qualification target is
approved; cache/output/failure and fixed-window gates remain enforced.

```bash
DATASET=/path/to/dsv4-tokenized-workload.json \
WORKLOAD_MANIFEST=/path/to/dsv4-workload-manifest.json \
REFRESH_PROB=REPLACE_WITH_CALIBRATED_VALUE \
OUT=/home/ubuntu/logs-dsv4-b300/control-run1.json \
  bash benchmark/deepseek_v4_b300/run_scorecard.sh
```

Use `ALLOW_SCORECARD_FAILURE=1` only for labeled calibration. Never use a
calibration result as a performance claim.

## Compile a measured profile

Collective tuning requires a measured shape profile rather than inheriting M3
message sizes. The file format is:

```json
{
  "collective_weights": {
    "0.125MiB": 0.20,
    "4MiB": 0.30,
    "192MiB": 0.50
  }
}
```

Run on one idle TP4 island in the real confidential VM:

```bash
python3 benchmark/deepseek_v4_b300/tune.py \
  --gpus 0,1,2,3 \
  --shape-profile /home/ubuntu/logs-dsv4-b300/shape-profile.json \
  --attestation-evidence /path/to/verified-attestation-bundle.json \
  --output-dir /home/ubuntu/logs-dsv4-b300-tune
```

FP4-indexer and compressed-state choices are selected only from matched
end-to-end bundles. Build one bundle per candidate:

```bash
python3 benchmark/deepseek_v4_b300/bundle_candidate.py \
  --name fp32-default \
  --replay /path/to/fp32-default-replay.json \
  --correctness /path/to/fp32-default-worker-a-probe.json \
  --correctness /path/to/fp32-default-worker-b-probe.json \
  --server-info /path/to/fp32-default-worker-a-server-info.json \
  --server-info /path/to/fp32-default-worker-b-server-info.json \
  --set DSV4_COMPRESS_STATE_DTYPE=float32 \
  --set DSV4_ENABLE_FP4_INDEXER=0 \
  --out /path/to/fp32-default-bundle.json
```

Then pass every matched candidate to the tuner using repeated
`--variant-result NAME=PATH`. A candidate must complete its fixed window, pass
the traffic scorecard and all correctness probes, improve fixed-window TPM/GPU
by at least 2%, retain at least 98% of drain-inclusive throughput, and regress
neither TTFT p50 nor ITL p50 by more than 5%. Otherwise the FP32/default-indexer
control remains selected.
