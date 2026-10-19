#!/usr/bin/env bash
# `engram` holds the Engram pages, `pd` runs one PD pair through tp2-pd.sh with
# the holder's PID; anything else runs as given (e.g. a probe or the oracle proxy).
set -euo pipefail
cd /opt/sglang
export MODEL="${MODEL:-/data/DeepSeek-V4.1-Flash}"
READY_FILE="${ENGRAM_READY_FILE:-/run/dsv41/engram.ready}"

case "${1:-pd}" in
engram)
    shift
    exec python benchmark/deepseek_v41_pd/hold_engram_cache.py \
        --model "$MODEL" --ready-file "$READY_FILE" "$@"
    ;;
pd)
    shift
    pid="$(cat "$READY_FILE")"
    # Both containers share the host PID namespace, so this is the holder's host PID.
    if ! grep -qa hold_engram_cache "/proc/$pid/cmdline" 2>/dev/null; then
        echo "Engram holder PID $pid from $READY_FILE is not running" >&2
        exit 1
    fi
    export PYTHON=python
    export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-/data/dsv41-bringup/recipes}/container-pd-$(date -u +%Y%m%dT%H%M%SZ)}"
    # For the compose healthcheck, which finds smoke.json there.
    echo "$OUTPUT_DIR" > /tmp/output-dir
    exec recipes/deepseek-v41/tp2-pd.sh --resident-engram-pid "$pid" "$@"
    ;;
*)
    exec "$@"
    ;;
esac
