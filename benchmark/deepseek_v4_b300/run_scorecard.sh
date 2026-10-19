#!/usr/bin/env bash
# Fixed-window DSV4 scorecard. DATASET and REFRESH_PROB are intentionally required.
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/data/sglang-venv/bin/python3}
DATASET=${DATASET:-}
WORKLOAD_MANIFEST=${WORKLOAD_MANIFEST:-}
OUT=${OUT:-}
REFRESH_PROB=${REFRESH_PROB:-}
MODEL=${MODEL:-dsv4-flash-0731}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-8}
GPU_MODEL=${GPU_MODEL:-B300}
TOPOLOGY=${TOPOLOGY:-1 node; 2xTP4; GPUs 0-7; integrated DSpark; CC on}
PORT_A=${PORT_A:-30300}
PORT_B=${PORT_B:-30301}
N_CONVS=${N_CONVS:-256}
CONCURRENCY=${CONCURRENCY:-256}
MAX_TURNS=${MAX_TURNS:-25}
WARMUP_SECONDS=${WARMUP_SECONDS:-60}
MEASUREMENT_SECONDS=${MEASUREMENT_SECONDS:-600}
CACHE_HIT_MIN=${CACHE_HIT_MIN:-0.955}
CACHE_HIT_MAX=${CACHE_HIT_MAX:-0.965}
OUTPUT_SHARE_MIN=${OUTPUT_SHARE_MIN:-0.0070}
OUTPUT_SHARE_MAX=${OUTPUT_SHARE_MAX:-0.0080}
MIN_FIXED_TPM_PER_GPU=${MIN_FIXED_TPM_PER_GPU:-0}
MIN_DRAIN_TPM_PER_GPU=${MIN_DRAIN_TPM_PER_GPU:-0}
TTFT_P50_TARGET=${TTFT_P50_TARGET:-3}

[[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
[[ -n "${DATASET}" && -f "${DATASET}" ]] || die "DATASET must name a DSV4-tokenized corpus"
[[ -n "${WORKLOAD_MANIFEST}" && -f "${WORKLOAD_MANIFEST}" ]] \
  || die "WORKLOAD_MANIFEST is required"
[[ -n "${OUT}" ]] || die "OUT is required"
[[ -n "${REFRESH_PROB}" ]] || die "REFRESH_PROB is required and must be calibrated"

"${PYTHON}" "${REPO_ROOT}/benchmark/deepseek_v4_b300/workload_manifest.py" verify \
  --manifest "${WORKLOAD_MANIFEST}" \
  --dataset "${DATASET}" \
  --n-convs "${N_CONVS}" \
  --concurrency "${CONCURRENCY}" \
  --max-turns "${MAX_TURNS}" \
  --warmup-seconds "${WARMUP_SECONDS}" \
  --measurement-seconds "${MEASUREMENT_SECONDS}" \
  --physical-gpus "${PHYSICAL_GPUS}" \
  --cache-hit-min "${CACHE_HIT_MIN}" \
  --cache-hit-max "${CACHE_HIT_MAX}" \
  --output-share-min "${OUTPUT_SHARE_MIN}" \
  --output-share-max "${OUTPUT_SHARE_MAX}" \
  --refresh-prob "${REFRESH_PROB}"

ARGS=(
  --worker-base-url "http://127.0.0.1:${PORT_A}"
  --worker-base-url "http://127.0.0.1:${PORT_B}"
  --server-info-url "http://127.0.0.1:${PORT_A}"
  --server-info-url "http://127.0.0.1:${PORT_B}"
  --model "${MODEL}"
  --dataset "${DATASET}"
  --out "${OUT}"
  --physical-gpus "${PHYSICAL_GPUS}"
  --gpu-model "${GPU_MODEL}"
  --topology "${TOPOLOGY}"
  --n-convs "${N_CONVS}"
  --concurrency "${CONCURRENCY}"
  --max-turns "${MAX_TURNS}"
  --stream
  --refresh-prob "${REFRESH_PROB}"
  --warmup-seconds "${WARMUP_SECONDS}"
  --measurement-seconds "${MEASUREMENT_SECONDS}"
  --cache-hit-min "${CACHE_HIT_MIN}"
  --cache-hit-max "${CACHE_HIT_MAX}"
  --output-share-min "${OUTPUT_SHARE_MIN}"
  --output-share-max "${OUTPUT_SHARE_MAX}"
  --min-fixed-tpm-per-gpu "${MIN_FIXED_TPM_PER_GPU}"
  --min-drain-tpm-per-gpu "${MIN_DRAIN_TPM_PER_GPU}"
  --ttft-p50-target "${TTFT_P50_TARGET}"
)
[[ "${ALLOW_SCORECARD_FAILURE:-0}" == "1" ]] && ARGS+=(--allow-scorecard-failure)

exec "${PYTHON}" "${REPO_ROOT}/benchmark/oracle_replay/concurrent_replay.py" \
  "${ARGS[@]}" "$@"
