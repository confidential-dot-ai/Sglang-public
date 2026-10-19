#!/usr/bin/env python3
"""Compare deterministic DSV4 target-only and candidate probe artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalized_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "finish_reason": case.get("finish_reason"),
        "content": case.get("content", ""),
        "tool_calls": case.get("tool_calls", []),
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if reference.get("passed") is not True:
        failures.append("reference probe did not pass its own checks")
    if candidate.get("passed") is not True:
        failures.append("candidate probe did not pass its own checks")
    reference_cases = {case.get("prompt"): case for case in reference.get("cases", [])}
    candidate_cases = {case.get("prompt"): case for case in candidate.get("cases", [])}
    if set(reference_cases) != set(candidate_cases):
        failures.append("reference and candidate prompt sets differ")
        return failures
    for prompt in reference_cases:
        if normalized_case(reference_cases[prompt]) != normalized_case(
            candidate_cases[prompt]
        ):
            failures.append(f"deterministic output mismatch for prompt {prompt!r}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    failures = compare(reference, candidate)
    payload = {
        "schema_version": 1,
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "passed": not failures,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
