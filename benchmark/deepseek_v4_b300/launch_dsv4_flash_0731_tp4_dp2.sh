#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DSV4_REPLICAS=2
exec "${SCRIPT_DIR}/launch_dsv4_flash_0731_tp4.sh" "$@"
