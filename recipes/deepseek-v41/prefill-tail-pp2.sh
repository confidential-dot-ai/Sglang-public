#!/usr/bin/env bash
# DeepSeek-V4.1 prefill-side tail with a two-stage prefill pipeline.
#
#   prefill  GPUs 4,5  TP1 x PP2, all 40 layers (0-13 | 14-39), DSpark on the
#                      last stage, layers 21-39 replayed over padded SWA tail buckets
#   decode   GPUs 6,7  TP2/EP2 with DSpark, normal decode hand-off
#   router 8040, prefill 8041, decode 8042, bootstrap 8999
#
# Environment set by launch.py:
#   both     SGLANG_OPT_DSV41_PREFILL_TAIL=1
#   prefill  SGLANG_PP_LAYER_PARTITION=14,26
#
# Mutually exclusive with ced-pp2.sh. Not yet GPU-validated on cc-main.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_launch \
    --prefill-pp 2 \
    --prefill-bounded-replay \
    --cpu-engram-staging \
    --split-cuda-graphs \
    --output-dir "${OUTPUT_DIR:-$(default_output_dir prefill-tail-pp2)}" \
    "$@"
