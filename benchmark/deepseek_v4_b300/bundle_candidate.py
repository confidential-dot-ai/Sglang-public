#!/usr/bin/env python3
"""Bundle DSV4 replay and correctness artifacts for fail-closed selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_CONFIGURATION_KEYS = {
    "CUDA_GRAPH_MAX_BS",
    "DSV4_COMPRESS_STATE_DTYPE",
    "DSV4_DSPARK_BLOCK_SIZE",
    "DSV4_ENABLE_FP4_INDEXER",
    "DSV4_ENABLE_DP_ATTENTION",
    "DSV4_FUSE_C4_FP4_STORE",
    "DSV4_ENABLE_TBO",
    "DSV4_ENABLE_TBO_CUDA_GRAPH",
    "DSV4_MOE_RUNNER_BACKEND",
    "DSV4_NCCL_AUTO_CTAS",
    "NCCL_ALGO",
    "NCCL_BUFFSIZE",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_CTAS",
    "NCCL_NVLS_ENABLE",
    "NCCL_PROTO",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def parse_setting(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--set requires KEY=VALUE")
    key, setting = value.split("=", 1)
    if key not in ALLOWED_CONFIGURATION_KEYS:
        raise argparse.ArgumentTypeError(f"configuration key is not allowlisted: {key}")
    return key, setting


def validate_bundle(bundle: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    replay = bundle["replay"]
    fixed = replay.get("metrics", {}).get("fixed_window")
    drain = replay.get("metrics", {}).get("drain_inclusive")
    if replay.get("timing", {}).get("fixed_window_complete") is not True:
        failures.append("fixed measurement window is incomplete")
    if not isinstance(fixed, dict) or not isinstance(drain, dict):
        failures.append("fixed-window or drain-inclusive metrics are missing")
    else:
        if fixed.get("failed_requests") != 0 or drain.get("failed_requests") != 0:
            failures.append("replay contains failed requests")
        if float(fixed.get("served_tpm_per_gpu", 0)) <= 0:
            failures.append("fixed served_tpm_per_gpu is not positive")
        if float(drain.get("served_tpm_per_gpu", 0)) <= 0:
            failures.append("drain served_tpm_per_gpu is not positive")
    if replay.get("scorecard", {}).get("pass") is not True:
        failures.append("replay scorecard did not pass")
    probes = bundle.get("correctness", [])
    if not probes:
        failures.append("no correctness probes were supplied")
    for index, probe in enumerate(probes):
        if probe.get("passed") is not True:
            failures.append(f"correctness probe {index} did not pass")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, action="append", required=True)
    parser.add_argument("--server-info", type=Path, action="append", default=[])
    parser.add_argument("--set", dest="settings", type=parse_setting, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configuration = dict(args.settings)
    bundle = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "name": args.name,
        "configuration": configuration,
        "source_paths": {
            "replay": str(args.replay.resolve()),
            "correctness": [str(path.resolve()) for path in args.correctness],
            "server_info": [str(path.resolve()) for path in args.server_info],
        },
        "replay": load_json(args.replay),
        "correctness": [load_json(path) for path in args.correctness],
        "server_info": [load_json(path) for path in args.server_info],
    }
    failures = validate_bundle(bundle)
    bundle["valid"] = not failures
    bundle["failures"] = failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
