#!/usr/bin/env bash
# Launch one or two integrated DeepSeek-V4-Flash-0731 TP4+DSpark replicas.
# DSV4_REPLICAS=1 is the verified topology control. The dp2 wrapper below sets 2.
set -euo pipefail

die() {
  echo "[$(date -u +%FT%TZ)] ERROR: $*" >&2
  exit 1
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/data/sglang-venv/bin/python3}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-dsv4-flash-0731}
LOGDIR=${LOGDIR:-/home/ubuntu/logs-dsv4-flash-0731-b300}
DSV4_REPLICAS=${DSV4_REPLICAS:-1}
PORT_A=${PORT_A:-30300}
PORT_B=${PORT_B:-30301}
ROUTER_PORT=${ROUTER_PORT:-8030}
REQUIRE_CC=${REQUIRE_CC:-1}
REQUIRE_IDLE_GPUS=${REQUIRE_IDLE_GPUS:-1}
REQUIRE_ATTESTATION=${REQUIRE_ATTESTATION:-1}
ATTESTATION_EVIDENCE=${ATTESTATION_EVIDENCE:-}
ALLOW_NON_B300=${ALLOW_NON_B300:-0}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-1800}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
CHUNK_SIZE=${CHUNK_SIZE:-4096}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-256}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
MEM_FRACTION=${MEM_FRACTION:-0.90}
SWA_FULL_TOKENS_RATIO=${SWA_FULL_TOKENS_RATIO:-0.1}
MIN_POST_START_FREE_MIB=${MIN_POST_START_FREE_MIB:-16384}
DSV4_MOE_RUNNER_BACKEND=${DSV4_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}
DSV4_ENABLE_DSPARK=${DSV4_ENABLE_DSPARK:-1}
DSV4_ENABLE_FP4_INDEXER=${DSV4_ENABLE_FP4_INDEXER:-0}
DSV4_FUSE_C4_FP4_STORE=${DSV4_FUSE_C4_FP4_STORE:-1}
DSV4_ENABLE_TBO=${DSV4_ENABLE_TBO:-0}
DSV4_ENABLE_TBO_CUDA_GRAPH=${DSV4_ENABLE_TBO_CUDA_GRAPH:-}
DSV4_ENABLE_DP_ATTENTION=${DSV4_ENABLE_DP_ATTENTION:-0}
DSV4_COMPRESS_STATE_DTYPE=${DSV4_COMPRESS_STATE_DTYPE:-float32}
DSV4_DSPARK_BLOCK_SIZE=${DSV4_DSPARK_BLOCK_SIZE:-}
DSV4_TUNING_PROFILE=${DSV4_TUNING_PROFILE:-}
CPU_AFFINITY=${CPU_AFFINITY:-1}
CPU_AFFINITY_MODE=${CPU_AFFINITY_MODE:-strided}
CPU_AFFINITY_ALL_THREADS=${CPU_AFFINITY_ALL_THREADS:-1}

[[ "${DSV4_REPLICAS}" == "1" || "${DSV4_REPLICAS}" == "2" ]] \
  || die "DSV4_REPLICAS must be 1 or 2"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v ss >/dev/null || die "ss is unavailable"
[[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
[[ "${MAX_TOTAL_TOKENS}" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_TOTAL_TOKENS must be an explicit positive integer"

if [[ -n "${DSV4_TUNING_PROFILE}" ]]; then
  [[ -f "${DSV4_TUNING_PROFILE}" ]] \
    || die "missing DSV4 tuning profile: ${DSV4_TUNING_PROFILE}"
  # Generated profiles contain only the allowlisted variables documented by
  # benchmark/deepseek_v4_b300/tune.py.
  # shellcheck disable=SC1090
  source "${DSV4_TUNING_PROFILE}"
fi
if [[ "${DSV4_NCCL_AUTO_CTAS:-0}" == "1" ]]; then
  unset NCCL_MIN_CTAS NCCL_MAX_CTAS
fi
# Unless a profile explicitly separates eager-only TBO from graph TBO, the
# high-level TBO switch controls both paths.
DSV4_ENABLE_TBO_CUDA_GRAPH=${DSV4_ENABLE_TBO_CUDA_GRAPH:-${DSV4_ENABLE_TBO}}
[[ "${DSV4_ENABLE_FP4_INDEXER}" == "0" || "${DSV4_ENABLE_FP4_INDEXER}" == "1" ]] \
  || die "DSV4_ENABLE_FP4_INDEXER must be 0 or 1"
[[ "${DSV4_FUSE_C4_FP4_STORE}" == "0" || "${DSV4_FUSE_C4_FP4_STORE}" == "1" ]] \
  || die "DSV4_FUSE_C4_FP4_STORE must be 0 or 1"
[[ "${DSV4_ENABLE_TBO}" == "0" || "${DSV4_ENABLE_TBO}" == "1" ]] \
  || die "DSV4_ENABLE_TBO must be 0 or 1"
[[ "${DSV4_ENABLE_TBO_CUDA_GRAPH}" == "0" || "${DSV4_ENABLE_TBO_CUDA_GRAPH}" == "1" ]] \
  || die "DSV4_ENABLE_TBO_CUDA_GRAPH must be 0 or 1"
[[ "${DSV4_ENABLE_TBO_CUDA_GRAPH}" == "0" || "${DSV4_ENABLE_TBO}" == "1" ]] \
  || die "DSV4_ENABLE_TBO_CUDA_GRAPH=1 requires DSV4_ENABLE_TBO=1"
[[ "${DSV4_ENABLE_DP_ATTENTION}" == "0" || "${DSV4_ENABLE_DP_ATTENTION}" == "1" ]] \
  || die "DSV4_ENABLE_DP_ATTENTION must be 0 or 1"
[[ "${DSV4_COMPRESS_STATE_DTYPE}" =~ ^(float32|fp32|bfloat16|bf16)$ ]] \
  || die "DSV4_COMPRESS_STATE_DTYPE must be float32/fp32/bfloat16/bf16"
[[ "${CUDA_GRAPH_MAX_BS}" =~ ^[1-9][0-9]*$ ]] \
  || die "CUDA_GRAPH_MAX_BS must be a positive integer"
if [[ -n "${DSV4_DSPARK_BLOCK_SIZE}" ]]; then
  [[ "${DSV4_DSPARK_BLOCK_SIZE}" =~ ^[1-5]$ ]] \
    || die "DSV4_DSPARK_BLOCK_SIZE must be between 1 and 5"
fi

WORKER_PORTS=("${PORT_A}")
WORKER_GPUS=("0,1,2,3")
WORKER_NAMES=("worker-a")
if [[ "${DSV4_REPLICAS}" == "2" ]]; then
  WORKER_PORTS+=("${PORT_B}")
  WORKER_GPUS+=("4,5,6,7")
  WORKER_NAMES+=("worker-b")
fi
SELECTED_GPUS=$(IFS=,; echo "${WORKER_GPUS[*]}")

for port in "${WORKER_PORTS[@]}"; do
  if ss -H -ltn "sport = :${port}" | grep -q .; then
    die "port ${port} is already occupied"
  fi
done
if [[ "${DSV4_REPLICAS}" == "2" ]] \
  && ss -H -ltn "sport = :${ROUTER_PORT}" | grep -q .; then
  die "router port ${ROUTER_PORT} is already occupied"
fi

mkdir -p "${LOGDIR}"
PREFLIGHT_ARGS=(
  --repo-root "${REPO_ROOT}"
  --selected-gpus "${SELECTED_GPUS}"
  --expected-gpu-count 8
  --output "${LOGDIR}/preflight.json"
)
[[ "${REQUIRE_CC}" == "1" ]] || PREFLIGHT_ARGS+=(--no-require-cc)
[[ "${REQUIRE_IDLE_GPUS}" == "1" ]] || PREFLIGHT_ARGS+=(--no-require-idle)
[[ "${ALLOW_NON_B300}" == "1" ]] && PREFLIGHT_ARGS+=(--allow-non-b300)
if [[ "${REQUIRE_ATTESTATION}" == "1" ]]; then
  [[ -n "${ATTESTATION_EVIDENCE}" ]] \
    || die "ATTESTATION_EVIDENCE is required for a qualification launch"
  PREFLIGHT_ARGS+=(--attestation-evidence "${ATTESTATION_EVIDENCE}")
else
  PREFLIGHT_ARGS+=(--no-require-attestation)
  [[ -z "${ATTESTATION_EVIDENCE}" ]] \
    || PREFLIGHT_ARGS+=(--attestation-evidence "${ATTESTATION_EVIDENCE}")
fi
"${PYTHON}" "${REPO_ROOT}/benchmark/deepseek_v4_b300/preflight.py" \
  "${PREFLIGHT_ARGS[@]}"

GIT_SHA=$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD)
CACHE_ROOT=${CACHE_ROOT:-/home/ubuntu/.cache/dsv4-flash-0731-b300-${GIT_SHA}}
mkdir -p "${CACHE_ROOT}"

export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="/usr/local/cuda/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SGLANG_DSV4_COMPRESS_STATE_DTYPE="${DSV4_COMPRESS_STATE_DTYPE}"
export SGLANG_OPT_FUSE_C4_FP4_STORE="${DSV4_FUSE_C4_FP4_STORE}"
export SGLANG_ENABLE_DEEPSEEK_V4_TBO_CUDA_GRAPH="${DSV4_ENABLE_TBO_CUDA_GRAPH}"

# Qualification controls: use actual CC detection and the normal DSpark graph
# path. Experimental conditional decode, CCSpec, PD, DP, and TBO are separate
# experiment legs, never implicit launcher defaults.
unset SGLANG_CONFIDENTIAL_COMPUTE
unset SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK
unset SGLANG_CC_CONDITIONAL_DECODE
unset SGLANG_CCSPEC_DECODE

COMMON_FLAGS=(
  --model-path "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tp 4
  --moe-runner-backend "${DSV4_MOE_RUNNER_BACKEND}"
  --reasoning-parser deepseek-v4
  --tool-call-parser deepseekv4
  --mem-fraction-static "${MEM_FRACTION}"
  --chunked-prefill-size "${CHUNK_SIZE}"
  --context-length "${CONTEXT_LENGTH}"
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}"
  --enable-cache-report
  --enable-request-time-stats-logging
  --watchdog-timeout 1800
  --host 0.0.0.0
)
[[ "${DSV4_ENABLE_DSPARK}" == "0" || "${DSV4_ENABLE_DSPARK}" == "1" ]] \
  || die "DSV4_ENABLE_DSPARK must be 0 or 1"
[[ "${DSV4_ENABLE_DSPARK}" == "0" ]] \
  || COMMON_FLAGS+=(--speculative-algorithm DSPARK)
COMMON_FLAGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
[[ "${DSV4_ENABLE_FP4_INDEXER}" == "1" ]] \
  && COMMON_FLAGS+=(--enable-deepseek-v4-fp4-indexer)
[[ "${DSV4_ENABLE_TBO}" == "1" ]] \
  && COMMON_FLAGS+=(--enable-two-batch-overlap)
if [[ "${DSV4_ENABLE_DP_ATTENTION}" == "1" ]]; then
  # First DSV4 DP-attention candidate: keep MoE on the built-in TP path so
  # MegaMoE/A2A and ragged-verify policy remain separate experiment variables.
  # DSpark requires the DP lm-head when dp_size > 1.
  COMMON_FLAGS+=(
    --dp 4
    --enable-dp-attention
    --enable-dp-attention-local-control-broadcast
    --enable-dp-lm-head
    --moe-a2a-backend none
  )
fi
[[ "${DSV4_ENABLE_DSPARK}" == "0" || -z "${DSV4_DSPARK_BLOCK_SIZE}" ]] \
  || COMMON_FLAGS+=(--speculative-dspark-block-size "${DSV4_DSPARK_BLOCK_SIZE}")
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
  read -r -a EXTRA_FLAGS_ARRAY <<<"${EXTRA_FLAGS}"
  COMMON_FLAGS+=("${EXTRA_FLAGS_ARRAY[@]}")
fi

MANIFEST_GIT_SHA="${GIT_SHA}" \
MANIFEST_MODEL="${MODEL}" \
MANIFEST_SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
MANIFEST_REPLICAS="${DSV4_REPLICAS}" \
MANIFEST_WORKER_GPU_SETS="$(IFS=';'; echo "${WORKER_GPUS[*]}")" \
MANIFEST_WORKER_PORTS="$(IFS=,; echo "${WORKER_PORTS[*]}")" \
MANIFEST_ROUTER_PORT="${ROUTER_PORT}" \
MANIFEST_MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
MANIFEST_CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
MANIFEST_CHUNK_SIZE="${CHUNK_SIZE}" \
MANIFEST_CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS}" \
MANIFEST_MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
MANIFEST_MEM_FRACTION="${MEM_FRACTION}" \
MANIFEST_SWA_RATIO="${SWA_FULL_TOKENS_RATIO}" \
MANIFEST_MOE_BACKEND="${DSV4_MOE_RUNNER_BACKEND}" \
MANIFEST_ENABLE_DSPARK="${DSV4_ENABLE_DSPARK}" \
MANIFEST_COMPRESS_DTYPE="${DSV4_COMPRESS_STATE_DTYPE}" \
MANIFEST_FP4_INDEXER="${DSV4_ENABLE_FP4_INDEXER}" \
MANIFEST_FUSE_C4_FP4_STORE="${DSV4_FUSE_C4_FP4_STORE}" \
MANIFEST_ENABLE_TBO="${DSV4_ENABLE_TBO}" \
MANIFEST_ENABLE_TBO_CUDA_GRAPH="${DSV4_ENABLE_TBO_CUDA_GRAPH}" \
MANIFEST_ENABLE_DP_ATTENTION="${DSV4_ENABLE_DP_ATTENTION}" \
MANIFEST_DSPARK_BLOCK_SIZE="${DSV4_DSPARK_BLOCK_SIZE}" \
MANIFEST_TUNING_PROFILE="${DSV4_TUNING_PROFILE}" \
MANIFEST_EXTRA_FLAGS="${EXTRA_FLAGS:-}" \
"${PYTHON}" - "${LOGDIR}/launch-manifest.json" <<'PY'
import json, os, pathlib
path = pathlib.Path(__import__('sys').argv[1])
mode = "integrated DSpark" if os.environ["MANIFEST_ENABLE_DSPARK"] == "1" else "target-only"
dp_attention = os.environ["MANIFEST_ENABLE_DP_ATTENTION"] == "1"
payload = {
    "schema_version": 1,
    "git_sha": os.environ["MANIFEST_GIT_SHA"],
    "model": os.environ["MANIFEST_MODEL"],
    "served_model_name": os.environ["MANIFEST_SERVED_MODEL_NAME"],
    "topology": (
        f'{os.environ["MANIFEST_REPLICAS"]}xTP4 '
        f'{"DP-attention-4" if dp_attention else "TP-attention"} {mode}'
    ),
    "worker_gpu_sets": os.environ["MANIFEST_WORKER_GPU_SETS"].split(";"),
    "worker_ports": [int(value) for value in os.environ["MANIFEST_WORKER_PORTS"].split(",")],
    "router_port": int(os.environ["MANIFEST_ROUTER_PORT"]),
    "max_total_tokens": int(os.environ["MANIFEST_MAX_TOTAL_TOKENS"]),
    "context_length": int(os.environ["MANIFEST_CONTEXT_LENGTH"]),
    "chunked_prefill_size": int(os.environ["MANIFEST_CHUNK_SIZE"]),
    "cuda_graph_max_bs": int(os.environ["MANIFEST_CUDA_GRAPH_MAX_BS"]),
    "max_running_requests": int(os.environ["MANIFEST_MAX_RUNNING_REQUESTS"]),
    "mem_fraction_static": float(os.environ["MANIFEST_MEM_FRACTION"]),
    "swa_full_tokens_ratio": float(os.environ["MANIFEST_SWA_RATIO"]),
    "moe_runner_backend": os.environ["MANIFEST_MOE_BACKEND"],
    "enable_dspark": os.environ["MANIFEST_ENABLE_DSPARK"] == "1",
    "compress_state_dtype": os.environ["MANIFEST_COMPRESS_DTYPE"],
    "enable_fp4_indexer": os.environ["MANIFEST_FP4_INDEXER"] == "1",
    "fuse_c4_fp4_store": os.environ["MANIFEST_FUSE_C4_FP4_STORE"] == "1",
    "enable_two_batch_overlap": os.environ["MANIFEST_ENABLE_TBO"] == "1",
    "enable_deepseek_v4_tbo_cuda_graph": (
        os.environ["MANIFEST_ENABLE_TBO_CUDA_GRAPH"] == "1"
    ),
    "enable_dp_attention": dp_attention,
    "dp_attention_size": 4 if dp_attention else 1,
    "enable_dp_lm_head": dp_attention,
    "moe_a2a_backend": "none",
    "dspark_block_size": os.environ["MANIFEST_DSPARK_BLOCK_SIZE"] or None,
    "tuning_profile": os.environ["MANIFEST_TUNING_PROFILE"] or None,
    "extra_flags": os.environ["MANIFEST_EXTRA_FLAGS"],
    "preflight": str((path.parent / "preflight.json").resolve()),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cleanup_failed_launch() {
  local status=$?
  if (( status != 0 )); then
    for pid_file in "${LOGDIR}"/*.pid; do
      [[ -f "${pid_file}" ]] || continue
      pid=$(<"${pid_file}")
      kill -TERM "${pid}" 2>/dev/null || true
    done
  fi
  exit "${status}"
}
trap cleanup_failed_launch EXIT

for worker_index in "${!WORKER_NAMES[@]}"; do
  worker_name=${WORKER_NAMES[worker_index]}
  worker_port=${WORKER_PORTS[worker_index]}
  worker_gpus=${WORKER_GPUS[worker_index]}
  worker_cache="${CACHE_ROOT}/${worker_name}"
  mkdir -p \
    "${worker_cache}/torchinductor" \
    "${worker_cache}/triton" \
    "${worker_cache}/flashinfer"
  echo "[$(date -u +%FT%TZ)] launching ${worker_name}: GPUs ${worker_gpus}, port ${worker_port}"
  CUDA_VISIBLE_DEVICES="${worker_gpus}" \
  TORCHINDUCTOR_CACHE_DIR="${worker_cache}/torchinductor" \
  TRITON_CACHE_DIR="${worker_cache}/triton" \
  FLASHINFER_CACHEDIR="${worker_cache}/flashinfer" \
  SGLANG_SET_CPU_AFFINITY="${CPU_AFFINITY}" \
  SGLANG_CPU_AFFINITY_MODE="${CPU_AFFINITY_MODE}" \
  SGLANG_CPU_AFFINITY_ALL_THREADS="${CPU_AFFINITY_ALL_THREADS}" \
  SGLANG_CPU_AFFINITY_WORLD_SIZE=8 \
  SGLANG_CPU_AFFINITY_RANK_OFFSET=$((worker_index * 4)) \
    "${PYTHON}" -m sglang.launch_server \
      "${COMMON_FLAGS[@]}" --port "${worker_port}" \
      >"${LOGDIR}/${worker_name}.log" 2>&1 &
  echo "$!" >"${LOGDIR}/${worker_name}.pid"
done

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
for worker_index in "${!WORKER_NAMES[@]}"; do
  worker_name=${WORKER_NAMES[worker_index]}
  worker_port=${WORKER_PORTS[worker_index]}
  worker_pid=$(<"${LOGDIR}/${worker_name}.pid")
  health_code=""
  while (( SECONDS < deadline )); do
    kill -0 "${worker_pid}" 2>/dev/null || die "${worker_name} exited during startup"
    health_code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
      "http://127.0.0.1:${worker_port}/health" || true)
    [[ "${health_code}" == "200" ]] && break
    sleep 5
  done
  [[ "${health_code}" == "200" ]] || die "${worker_name} did not become healthy"
  curl -fsS "http://127.0.0.1:${worker_port}/server_info" \
    >"${LOGDIR}/${worker_name}-server-info.json"
done

while IFS=, read -r gpu_index used_mib total_mib; do
  gpu_index=${gpu_index// /}
  used_mib=${used_mib// /}
  total_mib=${total_mib// /}
  free_mib=$((total_mib - used_mib))
  (( free_mib >= MIN_POST_START_FREE_MIB )) \
    || die "GPU ${gpu_index} has only ${free_mib} MiB free after startup"
done < <(nvidia-smi \
  --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)

if [[ "${DSV4_REPLICAS}" == "2" ]]; then
  ulimit -n 65535
  "${PYTHON}" -m sglang_router.launch_router \
    --worker-urls \
      "http://127.0.0.1:${PORT_A}" \
      "http://127.0.0.1:${PORT_B}" \
    --policy cache_aware \
    --balance-abs-threshold 4 \
    --host 127.0.0.1 \
    --port "${ROUTER_PORT}" \
    >"${LOGDIR}/router.log" 2>&1 &
  echo "$!" >"${LOGDIR}/router.pid"
fi

trap - EXIT
echo "[$(date -u +%FT%TZ)] DeepSeek V4 launch complete"
for port in "${WORKER_PORTS[@]}"; do
  echo "worker: http://127.0.0.1:${port}"
done
[[ "${DSV4_REPLICAS}" == "1" ]] \
  || echo "router: http://127.0.0.1:${ROUTER_PORT}"
echo "logs: ${LOGDIR}"
