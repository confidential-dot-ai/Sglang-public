#!/usr/bin/env bash
# DeepSeek-V4.1 CED split with a two-stage prefill pipeline.
#
#   prefill  GPUs 4,5  TP1 x PP2, encoder layers only (0-7 | 8-20), no DSpark
#   decode   GPUs 6,7  TP2/EP2 with DSpark, replays each prompt tail
#   router 8040, prefill 8041, decode 8042, bootstrap 8999
#
# Environment set by launch.py:
#   both     SGLANG_OPT_DSV41_CED_SPLIT=1
#   prefill  SGLANG_PP_LAYER_PARTITION=8,13 (stages split at kv_source layers)
#
# GPU-validated on the pp-encoder branch (benchmark/deepseek_v41_pd/pp2-ced-report.md).
# Not yet rerun on cc-main.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_launch \
    --ced-split \
    --prefill-pp 2 \
    --cpu-engram-staging \
    --split-cuda-graphs \
    --output-dir "${OUTPUT_DIR:-$(default_output_dir ced-pp2)}" \
    "$@"
