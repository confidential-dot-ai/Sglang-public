#!/usr/bin/env bash
# Run the oracle-mode agent-trace replay through sglang's real bench_serving
# harness (proper concurrency, TTFT/ITL/throughput metrics), instead of the
# one-rollout-at-a-time replay.py.
#
# Requires a server built with oracle-mode support (feat/oracle-decode-mode)
# already running, and --dataset-name oracle-trace registered (same branch).
set -euo pipefail

TOKENIZER="${TOKENIZER:?set TOKENIZER, e.g. MiniMaxAI/MiniMax-M3}"
MODEL="${MODEL:?set MODEL to the name the server was launched with}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
PAYLOADS_GLOB="${PAYLOADS_GLOB:?set PAYLOADS_GLOB, e.g. \"payloads_*.parquet\"}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
OUT_DIR="${OUT_DIR:-./oracle_bench_out}"

mkdir -p "$OUT_DIR"
COMBINED="$OUT_DIR/combined.json"

echo "== building oracle-trace dataset from $PAYLOADS_GLOB =="
python3 build_trace.py \
  --payloads-parquet "$PAYLOADS_GLOB" \
  --tokenizer "$TOKENIZER" \
  --num-rollouts "$NUM_ROLLOUTS" \
  --combined-output "$COMBINED"

echo "== running sglang bench_serving (oracle-trace, sglang-oai-chat) =="
python3 -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --dataset-name oracle-trace \
  --dataset-path "$COMBINED" \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --num-prompts "$NUM_ROLLOUTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --output-file "$OUT_DIR/results.jsonl" \
  "$@"
