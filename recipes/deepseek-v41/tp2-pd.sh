#!/usr/bin/env bash
# DeepSeek-V4.1 baseline: one TP2 prefill worker and one TP2 decode worker.
#
#   prefill  GPUs 4,5  TP2, split CUDA graphs, DSpark
#   decode   GPUs 6,7  TP2/EP2, split CUDA graphs, DSpark
#   router 8040, prefill 8041, decode 8042, bootstrap 8999
#
# The matched control for the PP recipes: split CUDA graphs, TP2 prefill.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_launch \
    --cpu-engram-staging \
    --split-cuda-graphs \
    --output-dir "${OUTPUT_DIR:-$(default_output_dir tp2-pd)}" \
    "$@"
