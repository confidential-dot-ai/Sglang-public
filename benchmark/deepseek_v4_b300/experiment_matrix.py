#!/usr/bin/env python3
"""Generate a hash-bound, one-variable-at-a-time DSV4 B300 run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_TP4 = "benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4.sh"
LAUNCH_TP4X2 = "benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4_dp2.sh"
SCORECARD = "benchmark/deepseek_v4_b300/run_scorecard.sh"
PROBE = "scripts/dsv4_correctness_probe.py"

BASELINE_PROFILE = {
    "CUDA_GRAPH_MAX_BS": "256",
    "DSV4_COMPRESS_STATE_DTYPE": "float32",
    "DSV4_DSPARK_BLOCK_SIZE": "5",
    "DSV4_ENABLE_DP_ATTENTION": "0",
    "DSV4_ENABLE_FP4_INDEXER": "0",
    "DSV4_FUSE_C4_FP4_STORE": "1",
    "DSV4_ENABLE_TBO": "0",
    "DSV4_ENABLE_TBO_CUDA_GRAPH": "0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def candidate_definitions(include_tbo: bool = True) -> list[dict[str, Any]]:
    """Return a staged matrix; every candidate changes one parent setting."""

    baseline = dict(BASELINE_PROFILE)
    rows: list[dict[str, Any]] = [
        {
            "name": "target-tp4",
            "phase": "matched-controls",
            "parent": None,
            "replicas": 1,
            "enable_dspark": False,
            "profile": baseline,
            "changed_keys": ["DSV4_ENABLE_DSPARK", "DSV4_REPLICAS"],
        },
        {
            "name": "dspark-tp4",
            "phase": "matched-controls",
            "parent": "target-tp4",
            "replicas": 1,
            "enable_dspark": True,
            "profile": baseline,
            "changed_keys": ["DSV4_ENABLE_DSPARK"],
        },
        {
            "name": "target-tp4x2",
            "phase": "matched-controls",
            "parent": "target-tp4",
            "replicas": 2,
            "enable_dspark": False,
            "profile": baseline,
            "changed_keys": ["DSV4_REPLICAS"],
        },
        {
            "name": "fp32-default",
            "phase": "matched-controls",
            "parent": "target-tp4x2",
            "replicas": 2,
            "enable_dspark": True,
            "profile": baseline,
            "changed_keys": ["DSV4_ENABLE_DSPARK"],
        },
        {
            "name": "target-dp-attention-4",
            "phase": "dp-attention-topology",
            "parent": "target-tp4x2",
            "replicas": 2,
            "enable_dspark": False,
            "profile": dict(baseline, DSV4_ENABLE_DP_ATTENTION="1"),
            "changed_keys": ["DSV4_ENABLE_DP_ATTENTION"],
            "blocked_until": [
                "target-only TP4 control passes graph/eager correctness on B300"
            ],
        },
        {
            "name": "dspark-dp-attention-4",
            "phase": "dp-attention-topology",
            "parent": "target-dp-attention-4",
            "replicas": 2,
            "enable_dspark": True,
            "profile": dict(baseline, DSV4_ENABLE_DP_ATTENTION="1"),
            "changed_keys": ["DSV4_ENABLE_DSPARK"],
            "blocked_until": [
                "target DP-attention-4 passes and DSpark acceptance parity passes"
            ],
        },
    ]

    for block_size in range(1, 5):
        profile = dict(baseline, DSV4_DSPARK_BLOCK_SIZE=str(block_size))
        rows.append(
            {
                "name": f"dspark-block-{block_size}",
                "phase": "dspark-depth",
                "parent": "fp32-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": profile,
                "changed_keys": ["DSV4_DSPARK_BLOCK_SIZE"],
            }
        )

    rows.extend(
        [
            {
                "name": "bf16-default",
                "phase": "compressed-state",
                "parent": "fp32-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(baseline, DSV4_COMPRESS_STATE_DTYPE="bf16"),
                "changed_keys": ["DSV4_COMPRESS_STATE_DTYPE"],
            },
            {
                "name": "fp32-fp4-indexer",
                "phase": "fp4-indexer",
                "parent": "fp32-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(baseline, DSV4_ENABLE_FP4_INDEXER="1"),
                "changed_keys": ["DSV4_ENABLE_FP4_INDEXER"],
            },
            {
                "name": "bf16-fp4-indexer",
                "phase": "interaction",
                "parent": "bf16-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(
                    baseline,
                    DSV4_COMPRESS_STATE_DTYPE="bf16",
                    DSV4_ENABLE_FP4_INDEXER="1",
                ),
                "changed_keys": ["DSV4_ENABLE_FP4_INDEXER"],
            },
            {
                "name": "fp32-fp4-indexer-unfused",
                "phase": "fp4-indexer-fusion",
                "parent": "fp32-fp4-indexer",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(
                    baseline,
                    DSV4_ENABLE_FP4_INDEXER="1",
                    DSV4_FUSE_C4_FP4_STORE="0",
                ),
                "changed_keys": ["DSV4_FUSE_C4_FP4_STORE"],
            },
            {
                "name": "graph-bs-128",
                "phase": "graph-tiers",
                "parent": "fp32-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(baseline, CUDA_GRAPH_MAX_BS="128"),
                "changed_keys": ["CUDA_GRAPH_MAX_BS"],
            },
            {
                "name": "graph-bs-384",
                "phase": "graph-tiers",
                "parent": "fp32-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(baseline, CUDA_GRAPH_MAX_BS="384"),
                "changed_keys": ["CUDA_GRAPH_MAX_BS"],
            },
        ]
    )
    if include_tbo:
        rows.append(
            {
                "name": "tbo-on",
                "phase": "post-correctness-tbo",
                "parent": "fp32-default",
                "replicas": 2,
                "enable_dspark": True,
                "profile": dict(
                    baseline,
                    DSV4_ENABLE_TBO="1",
                    DSV4_ENABLE_TBO_CUDA_GRAPH="1",
                ),
                "changed_keys": [
                    "DSV4_ENABLE_TBO",
                    "DSV4_ENABLE_TBO_CUDA_GRAPH",
                ],
                "blocked_until": [
                    "DSpark auxiliary-state graph/eager parity passes on B300"
                ],
            }
        )
    return rows


def validate_definitions(rows: list[dict[str, Any]]) -> None:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["name"]
        if name in by_name:
            raise ValueError(f"duplicate candidate name: {name}")
        parent_name = row.get("parent")
        if parent_name is not None:
            if parent_name not in by_name:
                raise ValueError(f"candidate {name} has unknown/later parent {parent_name}")
            parent = by_name[parent_name]
            actual = {
                key
                for key in set(row["profile"]) | set(parent["profile"])
                if row["profile"].get(key) != parent["profile"].get(key)
            }
            if row["replicas"] != parent["replicas"]:
                actual.add("DSV4_REPLICAS")
            if row["enable_dspark"] != parent["enable_dspark"]:
                actual.add("DSV4_ENABLE_DSPARK")
            declared = set(row["changed_keys"])
            if actual != declared:
                raise ValueError(
                    f"candidate {name} changes {sorted(actual)}, declared {sorted(declared)}"
                )
        by_name[name] = row


def _run_record(
    row: dict[str, Any],
    repetition: int,
    *,
    run_root: Path,
    dataset: Path,
    workload_manifest: Path,
    attestation: Path,
    max_total_tokens: int,
    refresh_prob: float,
) -> dict[str, Any]:
    run_dir = run_root / row["name"] / f"run-{repetition}"
    worker_ports = [30300] if row["replicas"] == 1 else [30300, 30301]
    launch_env = {
        "ATTESTATION_EVIDENCE": str(attestation),
        "DSV4_ENABLE_DSPARK": "1" if row["enable_dspark"] else "0",
        "LOGDIR": str(run_dir),
        "MAX_TOTAL_TOKENS": str(max_total_tokens),
        **row["profile"],
    }
    scorecard_out = run_dir / "replay.json"
    scorecard_env = {
        "DATASET": str(dataset),
        "OUT": str(scorecard_out),
        "REFRESH_PROB": str(refresh_prob),
        "WORKLOAD_MANIFEST": str(workload_manifest),
    }
    probes = [
        {
            "argv": [
                "python3",
                PROBE,
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--model",
                "dsv4-flash-0731",
                "--tag",
                f"{row['name']}-run-{repetition}-worker-{index}",
                "--out",
                str(run_dir / f"worker-{index}-probe.json"),
            ],
            "worker_port": port,
        }
        for index, port in enumerate(worker_ports)
    ]
    return {
        "repetition": repetition,
        "run_dir": str(run_dir),
        "launch": {
            "argv": [
                "bash",
                LAUNCH_TP4 if row["replicas"] == 1 else LAUNCH_TP4X2,
            ],
            "environment": launch_env,
        },
        "correctness_probes": probes,
        "scorecard": {
            "argv": ["bash", SCORECARD],
            "environment": scorecard_env,
            "output": str(scorecard_out),
        },
        "required_artifacts": [
            str(run_dir / "launch-manifest.json"),
            str(run_dir / "preflight.json"),
            *[
                str(run_dir / f"worker-{'a' if i == 0 else 'b'}-server-info.json")
                for i in range(len(worker_ports))
            ],
            *[str(run_dir / f"worker-{i}-probe.json") for i in range(len(worker_ports))],
            str(scorecard_out),
        ],
    }


def build_matrix(
    *,
    repo_root: Path,
    run_root: Path,
    dataset: Path,
    workload_manifest: Path,
    attestation: Path,
    max_total_tokens: int,
    refresh_prob: float,
    repetitions: int,
    include_tbo: bool,
) -> dict[str, Any]:
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0 <= refresh_prob <= 1:
        raise ValueError("refresh_prob must be between 0 and 1")
    for path in (dataset, workload_manifest, attestation):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required immutable input is missing or empty: {path}")

    rows = candidate_definitions(include_tbo=include_tbo)
    validate_definitions(rows)
    candidates = []
    for row in rows:
        candidate = dict(row)
        candidate["runs"] = [
            _run_record(
                row,
                repetition,
                run_root=run_root,
                dataset=dataset,
                workload_manifest=workload_manifest,
                attestation=attestation,
                max_total_tokens=max_total_tokens,
                refresh_prob=refresh_prob,
            )
            for repetition in range(1, repetitions + 1)
        ]
        candidates.append(candidate)

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository": {
            "root": str(repo_root.resolve()),
            "git_sha": git_output(repo_root, "rev-parse", "HEAD"),
            "git_status": git_output(repo_root, "status", "--porcelain=v1"),
        },
        "immutable_inputs": {
            "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
            "workload_manifest": {
                "path": str(workload_manifest),
                "sha256": sha256_file(workload_manifest),
            },
            "attestation": {
                "path": str(attestation),
                "sha256": sha256_file(attestation),
                "note": "hash binding does not replace platform semantic verification",
            },
        },
        "contract": {
            "max_total_tokens_per_replica": max_total_tokens,
            "refresh_prob": refresh_prob,
            "repetitions": repetitions,
            "selection_order": [
                "matched-controls",
                "dp-attention-topology",
                "dspark-depth",
                "compressed-state",
                "fp4-indexer",
                "interaction",
                "graph-tiers",
                "post-correctness-tbo",
            ],
            "execution": "manual-gated; never run two candidates concurrently",
        },
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--attestation-evidence", type=Path, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--refresh-prob", type=float, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--no-tbo", dest="include_tbo", action="store_false")
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(include_tbo=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_matrix(
        repo_root=args.repo_root,
        run_root=args.run_root,
        dataset=args.dataset.resolve(),
        workload_manifest=args.workload_manifest.resolve(),
        attestation=args.attestation_evidence.resolve(),
        max_total_tokens=args.max_total_tokens,
        refresh_prob=args.refresh_prob,
        repetitions=args.repetitions,
        include_tbo=args.include_tbo,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
