#!/usr/bin/env python3
"""Fail-closed B300/GB300 and confidential-compute launch preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


CommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_output(command: Sequence[str]) -> tuple[int, str]:
    process = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return process.returncode, process.stdout


def parse_gpu_indices(value: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU indices must be integers") from exc
    if not indices or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("GPU indices must be non-empty and unique")
    return indices


def is_b300_name(name: str) -> bool:
    return bool(re.search(r"\b(?:B300|GB300)\b", name, re.IGNORECASE))


def parse_inventory(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fields in csv.reader(output.splitlines()):
        fields = [field.strip() for field in fields]
        if not fields:
            continue
        if len(fields) != 7:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {fields!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "pci_bus_id": fields[3],
                "memory_total_mib": int(fields[4]),
                "memory_used_mib": int(fields[5]),
                "driver_version": fields[6],
            }
        )
    return rows


def parse_compute_processes(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fields in csv.reader(output.splitlines()):
        fields = [field.strip() for field in fields]
        if not fields or not any(fields):
            continue
        if len(fields) != 4:
            raise RuntimeError(f"unexpected compute-process row: {fields!r}")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": fields[1],
                "process_name": fields[2],
                "used_memory_mib": fields[3],
            }
        )
    return rows


def cc_is_qualified(cc: dict[str, dict[str, Any]]) -> bool:
    feature = cc["feature"]["output"]
    devtools = cc["devtools"]["output"]
    multi_gpu = cc["multi_gpu"]["output"]
    commands_ok = all(item["returncode"] == 0 for item in cc.values())
    return (
        commands_ok
        and bool(re.search(r"CC (?:feature )?status:\s*ON", feature, re.I))
        and bool(re.search(r"DevTools [Mm]ode:\s*OFF", devtools, re.I))
        and bool(
            re.search(
                r"(?:multi.?gpu.*(?:ON|enabled)|status:\s*(?:ON|enabled))",
                multi_gpu,
                re.I,
            )
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(
    *,
    repo_root: Path,
    selected_gpu_indices: list[int],
    expected_gpu_count: int,
    require_cc: bool,
    require_idle: bool,
    allow_non_b300: bool,
    attestation_evidence: Path | None,
    require_attestation: bool,
    run: CommandRunner = command_output,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    inventory_command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    rc, output = run(inventory_command)
    if rc != 0:
        raise RuntimeError(f"nvidia-smi inventory failed:\n{output}")
    inventory = parse_inventory(output)
    if len(inventory) != expected_gpu_count:
        raise RuntimeError(
            f"expected exactly {expected_gpu_count} physical GPUs, found {len(inventory)}"
        )
    selected = [gpu for gpu in inventory if gpu["index"] in selected_gpu_indices]
    if len(selected) != len(selected_gpu_indices):
        raise RuntimeError(
            f"selected GPU indices are unavailable: {selected_gpu_indices}"
        )
    if not allow_non_b300:
        wrong = [gpu["name"] for gpu in inventory if not is_b300_name(gpu["name"])]
        if wrong:
            raise RuntimeError(f"B300/GB300 required; node includes {wrong}")

    cc: dict[str, dict[str, Any]] = {}
    for flag, key in (("-f", "feature"), ("-d", "devtools"), ("-mgm", "multi_gpu")):
        status, text = run(["nvidia-smi", "conf-compute", flag])
        cc[key] = {"returncode": status, "output": text.strip()}
    if require_cc and not cc_is_qualified(cc):
        raise RuntimeError(
            "CC feature ON, DevTools OFF, and multi-GPU CC ON were not all proved"
        )

    process_rc, process_output = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = parse_compute_processes(process_output) if process_rc == 0 else []
    selected_uuids = {gpu["uuid"] for gpu in selected}
    selected_processes = [p for p in processes if p["gpu_uuid"] in selected_uuids]
    if require_idle and process_rc != 0:
        raise RuntimeError("could not prove selected GPUs are idle")
    if require_idle and selected_processes:
        raise RuntimeError(
            f"selected GPUs have active compute processes: {selected_processes}"
        )

    topology_rc, topology = run(["nvidia-smi", "topo", "-m"])
    if topology_rc != 0:
        raise RuntimeError(f"nvidia-smi topology query failed:\n{topology}")

    evidence: dict[str, Any] | None = None
    if attestation_evidence is not None:
        if not attestation_evidence.is_file() or attestation_evidence.stat().st_size == 0:
            raise RuntimeError(
                f"attestation evidence is missing or empty: {attestation_evidence}"
            )
        evidence = {
            "path": str(attestation_evidence.resolve()),
            "size_bytes": attestation_evidence.stat().st_size,
            "sha256": sha256_file(attestation_evidence),
            "semantic_verification": "external",
        }
    elif require_attestation:
        raise RuntimeError(
            "qualification requires --attestation-evidence; presence and hash are "
            "recorded here, while semantic verification remains external"
        )

    git_sha_rc, git_sha = run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    git_status_rc, git_status = run(
        ["git", "-C", str(repo_root), "status", "--porcelain=v1"]
    )
    submodule_rc, submodules = run(
        ["git", "-C", str(repo_root), "submodule", "status", "--recursive"]
    )
    if git_sha_rc != 0 or git_status_rc != 0:
        raise RuntimeError("could not record the repository revision and status")

    version_probe = (
        "import json, sys, torch; "
        "import flashinfer, sglang; "
        "print(json.dumps({"
        "'python': sys.version.split()[0], "
        "'torch': torch.__version__, "
        "'cuda': torch.version.cuda, "
        "'nccl': torch.cuda.nccl.version(), "
        "'flashinfer': getattr(flashinfer, '__version__', 'unknown'), "
        "'sglang_path': sglang.__file__}))"
    )
    versions_rc, versions_output = run([python_executable, "-c", version_probe])
    if versions_rc != 0:
        raise RuntimeError(f"dependency version probe failed:\n{versions_output}")
    try:
        versions = json.loads(versions_output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("dependency version probe did not emit JSON") from exc
    nvcc_rc, nvcc_output = run(["nvcc", "--version"])
    if nvcc_rc != 0:
        raise RuntimeError(f"nvcc version query failed:\n{nvcc_output}")

    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "repository": {
            "root": str(repo_root.resolve()),
            "git_sha": git_sha.strip(),
            "status_porcelain": git_status.splitlines(),
            "submodules_returncode": submodule_rc,
            "submodules": submodules.splitlines(),
        },
        "software": {
            "versions": versions,
            "nvcc_version": nvcc_output.strip(),
        },
        "hardware": {
            "expected_gpu_count": expected_gpu_count,
            "inventory": inventory,
            "selected_gpu_indices": selected_gpu_indices,
            "topology": topology,
            "compute_process_query_returncode": process_rc,
            "selected_gpu_compute_processes": selected_processes,
        },
        "confidential_compute": {
            "required": require_cc,
            "qualified": cc_is_qualified(cc),
            "queries": cc,
            "attestation_required": require_attestation,
            "attestation_evidence": evidence,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--selected-gpus", type=parse_gpu_indices, required=True)
    parser.add_argument("--expected-gpu-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation-evidence", type=Path)
    parser.add_argument("--no-require-cc", dest="require_cc", action="store_false")
    parser.add_argument("--no-require-idle", dest="require_idle", action="store_false")
    parser.add_argument(
        "--no-require-attestation", dest="require_attestation", action="store_false"
    )
    parser.add_argument("--allow-non-b300", action="store_true")
    parser.set_defaults(require_cc=True, require_idle=True, require_attestation=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = inspect(
        repo_root=args.repo_root,
        selected_gpu_indices=args.selected_gpus,
        expected_gpu_count=args.expected_gpu_count,
        require_cc=args.require_cc,
        require_idle=args.require_idle,
        allow_non_b300=args.allow_non_b300,
        attestation_evidence=args.attestation_evidence,
        require_attestation=args.require_attestation,
        python_executable=sys.executable,
    )
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
