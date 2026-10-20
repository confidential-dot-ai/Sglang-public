#!/usr/bin/env bash
# Block until a compose container's healthcheck passes; fail as soon as it stops.
#   wait-healthy.sh CONTAINER
set -euo pipefail
name="$1"
since="$(date -u +%s)"
# `docker compose up` may not have created or started the container yet, and a
# container left stopped by the previous boot must not count as this start failing.
for _ in $(seq 90); do
    started="$(docker inspect -f '{{.State.StartedAt}}' "$name" 2>/dev/null || true)"
    [[ -n "$started" ]] && (($(date -u -d "$started" +%s) >= since - 2)) && break
    sleep 2
done
while true; do
    state="$(docker inspect -f '{{.State.Status}} {{.State.Health.Status}}' "$name")"
    case "$state" in
    "running healthy") exit 0 ;;
    running*) sleep 10 ;;
    *)
        echo "$name stopped before becoming healthy: $state" >&2
        exit 1
        ;;
    esac
done
