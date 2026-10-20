# Shared defaults for the DeepSeek-V4.1 PD recipes. Sourced, not executed.
# Every value can be overridden from the environment; extra arguments to a
# recipe are passed straight to launch.py (for example --dry-run).

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$RECIPE_DIR/../.." && pwd)"
LAUNCH="$REPO/recipes/deepseek-v41/launcher/launch.py"

PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-/data/DeepSeek-V4.1-Flash}"
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8040}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8999}"
NCCL_PORT="${NCCL_PORT:-29600}"
ROUTER_METRICS_PORT="${ROUTER_METRICS_PORT:-29001}"

# launch.py refuses to reuse an output directory.
default_output_dir() {
    echo "/data/dsv41-bringup/recipes/$1-$(date -u +%Y%m%dT%H%M%SZ)"
}

run_launch() {
    exec "$PYTHON" "$LAUNCH" \
        --model "$MODEL" \
        --gpus "$GPUS" \
        --port "$PORT" \
        --bootstrap-port "$BOOTSTRAP_PORT" \
        --nccl-port "$NCCL_PORT" \
        --router-metrics-port "$ROUTER_METRICS_PORT" \
        "$@"
}
