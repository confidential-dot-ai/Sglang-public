#!/usr/bin/env python3
"""Create and verify an immutable DSV4 replay workload contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def validate_ranges(contract: dict[str, Any]) -> None:
    cache_min = float(contract["cache_hit_min"])
    cache_max = float(contract["cache_hit_max"])
    output_min = float(contract["output_share_min"])
    output_max = float(contract["output_share_max"])
    if not 0 <= cache_min <= cache_max <= 1:
        raise ValueError("cache-hit bounds must satisfy 0 <= min <= max <= 1")
    if not 0 <= output_min <= output_max <= 1:
        raise ValueError("output-share bounds must satisfy 0 <= min <= max <= 1")
    for key in (
        "n_convs",
        "concurrency",
        "max_turns",
        "warmup_seconds",
        "measurement_seconds",
        "physical_gpus",
    ):
        if int(contract[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    payload = load_json(dataset)
    contract = {
        "n_convs": args.n_convs,
        "concurrency": args.concurrency,
        "max_turns": args.max_turns,
        "warmup_seconds": args.warmup_seconds,
        "measurement_seconds": args.measurement_seconds,
        "physical_gpus": args.physical_gpus,
        "cache_hit_min": args.cache_hit_min,
        "cache_hit_max": args.cache_hit_max,
        "output_share_min": args.output_share_min,
        "output_share_max": args.output_share_max,
        "refresh_prob": args.refresh_prob,
    }
    validate_ranges(contract)
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < args.n_convs:
        raise ValueError(
            f"dataset has fewer than the required {args.n_convs} conversations"
        )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "conversation_count": len(conversations),
            "metadata": payload.get("metadata"),
            "tokenization_assertion": (
                "forced token IDs and modeled lengths were generated with the "
                "recorded DeepSeek V4 tokenizer revision"
            ),
        },
        "contract": contract,
    }


def verify_manifest(
    manifest: dict[str, Any], *, dataset: Path, expected: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    dataset = dataset.resolve()
    recorded = manifest.get("dataset", {})
    if recorded.get("sha256") != sha256_file(dataset):
        failures.append("dataset SHA-256 differs from the workload manifest")
    contract = manifest.get("contract", {})
    try:
        validate_ranges(contract)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"manifest contract is invalid: {exc}")
        return failures
    for key, value in expected.items():
        recorded_value = contract.get(key)
        if isinstance(value, float):
            matches = recorded_value is not None and math_isclose(
                float(recorded_value), value
            )
        else:
            matches = recorded_value == value
        if not matches:
            failures.append(
                f"{key}={value!r} differs from manifest value {recorded_value!r}"
            )
    return failures


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right))


def add_contract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-convs", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--warmup-seconds", type=int, required=True)
    parser.add_argument("--measurement-seconds", type=int, required=True)
    parser.add_argument("--physical-gpus", type=int, required=True)
    parser.add_argument("--cache-hit-min", type=float, required=True)
    parser.add_argument("--cache-hit-max", type=float, required=True)
    parser.add_argument("--output-share-min", type=float, required=True)
    parser.add_argument("--output-share-max", type=float, required=True)
    parser.add_argument("--refresh-prob", type=float, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--dataset", type=Path, required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--model-revision", required=True)
    create.add_argument("--tokenizer", required=True)
    create.add_argument("--tokenizer-revision", required=True)
    create.add_argument("--out", type=Path, required=True)
    add_contract_args(create)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--dataset", type=Path, required=True)
    add_contract_args(verify)
    return parser.parse_args()


def expected_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key)
        for key in (
            "n_convs",
            "concurrency",
            "max_turns",
            "warmup_seconds",
            "measurement_seconds",
            "physical_gpus",
            "cache_hit_min",
            "cache_hit_max",
            "output_share_min",
            "output_share_max",
            "refresh_prob",
        )
    }


def main() -> int:
    args = parse_args()
    if args.command == "create":
        payload = create_manifest(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    manifest = load_json(args.manifest)
    failures = verify_manifest(
        manifest, dataset=args.dataset, expected=expected_contract(args)
    )
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
