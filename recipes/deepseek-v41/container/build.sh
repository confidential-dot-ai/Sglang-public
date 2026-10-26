#!/usr/bin/env bash
# Build the DeepSeek-V4.1 PD image from this checkout. The tag names the source
# state, and the image records it for launch.py, which has no .git to read.
#   recipes/deepseek-v41/container/build.sh [extra docker build args]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
DOCKER="${DOCKER:-docker}"

revision="$(git rev-parse HEAD)"
status="$(git status --porcelain)"
# Tracked edits plus untracked file contents, so a dirty build stays identifiable.
diff_sha256="$(
    {
        git diff HEAD --binary
        git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum
    } | sha256sum | cut -d' ' -f1
)"
tag="${revision:0:12}"
if [[ -n "$status" ]]; then
    tag="$tag-dirty-${diff_sha256:0:8}"
fi
record="$(python3 -c 'import json, sys; print(json.dumps(dict(zip(("revision", "status", "diff_sha256"), sys.argv[1:]))))' \
    "$revision" "$status" "$diff_sha256")"

$DOCKER build \
    -f recipes/deepseek-v41/container/Dockerfile \
    --build-arg SOURCE_RECORD="$record" \
    -t "dsv41-pd:$tag" -t dsv41-pd:latest \
    "$@" .
echo "Built dsv41-pd:$tag (also tagged dsv41-pd:latest)"
