"""Assemble a DeepSeek-V4.1 checkpoint from pinned Hub sources; idempotent.

Spec `1017-nvfp4-fp8-engram`: LibertAI's NVFP4 conversion of V4.1-Flash-1017
with DeepSeek's own FP8 Engram tables in place of its MXFP4 ones. That is the
layout of nvidia/DeepSeek-V4.1-Flash-NVFP4 at 3431dde, the revision NVIDIA
validated on SGLang, over the 1017 weights.

Every file is checked against the Hub's hashes at the pinned revisions, and
config.json and the index are derived, not downloaded. Once the manifest says
complete, a rerun only checks file sizes and mtimes. --seed-dir reuses files
that already exist elsewhere after hashing them: hard-linked when they share a
mount with the checkpoint, copied otherwise.

    prepare_checkpoint.py --spec 1017-nvfp4-fp8-engram --local-dir DIR [--seed-dir DIR]
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional


class Source(NamedTuple):
    repo: str
    revision: str


class Spec(NamedTuple):
    # Everything but the Engram shards, config.json and the index.
    base: Source
    # The Engram shards only.
    engram: Source
    # Config fields the derived checkpoint drops.
    drop_quant_fields: tuple


SPECS = {
    "1017-nvfp4-fp8-engram": Spec(
        base=Source(
            "LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4",
            "5be99cd15675589fb6693d0a8651e413bcb9a396",
        ),
        engram=Source(
            "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "1bae0e1088993b9b82de08f59de5be5a34e42170",
        ),
        # They select MXFP4 tables in sglang/srt/layers/engram.py; without them
        # the loader reads FP8, as the 1017 tables are.
        drop_quant_fields=(
            "engram_embed_quant",
            "engram_embed_block_size",
            "engram_embed_scale_dtype",
            "engram_embed_layers",
        ),
    ),
}

INDEX = "model.safetensors.index.json"
CONFIG = "config.json"
MANIFEST = "checkpoint-manifest.json"
ENGRAM_KEY = ".engram."
# Headroom for Hub temp files and logs; arbitrary.
DISK_MARGIN_BYTES = 8 * 1024**3


class RemoteFile(NamedTuple):
    source: Source
    path: str
    size: int
    # LFS files carry a sha256; small files are checked by git blob id.
    sha256: Optional[str]
    blob_id: str


def engram_shards(weight_map: Dict[str, str]) -> List[str]:
    """Shards holding Engram tensors, which must hold nothing else."""
    shards = sorted({f for k, f in weight_map.items() if ENGRAM_KEY in k})
    mixed = sorted(
        {f for k, f in weight_map.items() if f in shards and ENGRAM_KEY not in k}
    )
    if mixed:
        raise ValueError(f"Engram shards also hold other tensors: {mixed}")
    return shards


def plan_files(
    *,
    base_files: List[RemoteFile],
    engram_files: List[RemoteFile],
    base_index: dict,
    engram_index: dict,
) -> Dict[str, RemoteFile]:
    """Every file of the result except the derived config.json and index."""
    shards = engram_shards(base_index["weight_map"])
    if engram_shards(engram_index["weight_map"]) != shards:
        raise ValueError("The two sources keep their Engram tables in different shards")
    base_keys = {k for k in base_index["weight_map"] if ENGRAM_KEY in k}
    engram_keys = {k for k in engram_index["weight_map"] if ENGRAM_KEY in k}
    if base_keys != engram_keys:
        raise ValueError(
            f"Engram tensors differ: {sorted(base_keys ^ engram_keys)[:8]}"
        )
    files = {
        f.path: f
        for f in base_files
        if f.path not in shards and f.path not in (CONFIG, INDEX)
    }
    by_path = {f.path: f for f in engram_files}
    files.update({path: by_path[path] for path in shards})
    return files


def derived_config(config: dict, drop_quant_fields: tuple) -> dict:
    quant = {
        k: v
        for k, v in config["quantization_config"].items()
        if k not in drop_quant_fields
    }
    return {**config, "quantization_config": quant}


def safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def derived_index(index: dict, local_dir: Path) -> dict:
    """The base index with its total size recounted from the local shards."""
    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        for tensor in safetensors_header(local_dir / shard).values():
            start, end = tensor["data_offsets"]
            total += end - start
    return {**index, "metadata": {**index.get("metadata", {}), "total_size": total}}


def file_matches(path: Path, remote: RemoteFile) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    if remote.sha256 is not None:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(f"blob {remote.size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest() == (remote.sha256 or remote.blob_id)


def ensure_file(
    remote: RemoteFile,
    *,
    local_dir: Path,
    seed_dir: Optional[Path],
    download: Callable[[RemoteFile, Path], None],
) -> str:
    target = local_dir / remote.path
    if file_matches(target, remote):
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    seed = seed_dir / remote.path if seed_dir is not None else None
    if seed is not None and file_matches(seed, remote):
        try:
            os.link(seed, target)
        except OSError:
            shutil.copyfile(seed, target)
        return "seeded"
    download(remote, local_dir)
    if not file_matches(target, remote):
        raise RuntimeError(f"{remote.path} does not match {remote.source} after download")
    return "downloaded"


def manifest_is_current(local_dir: Path, spec_name: str, spec: Spec) -> bool:
    try:
        manifest = json.loads((local_dir / MANIFEST).read_text())
    except (OSError, ValueError):
        return False
    if manifest.get("status") != "complete" or manifest.get("spec") != spec_name:
        return False
    if manifest.get("sources") != {k: list(v) for k, v in spec._asdict().items() if isinstance(v, Source)}:
        return False
    for path, entry in manifest["files"].items():
        stat = (local_dir / path).stat() if (local_dir / path).is_file() else None
        if stat is None or (stat.st_size, stat.st_mtime_ns) != (
            entry["size"],
            entry["mtime_ns"],
        ):
            return False
    return True


def remote_files(source: Source) -> List[RemoteFile]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(
        source.repo, revision=source.revision, files_metadata=True
    )
    return [
        RemoteFile(
            source=source,
            path=s.rfilename,
            size=s.size,
            sha256=s.lfs.sha256 if s.lfs else None,
            blob_id=s.blob_id,
        )
        for s in info.siblings
    ]


def fetch_json(source: Source, path: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(hf_hub_download(source.repo, path, revision=source.revision)).read_text()
    )


def hub_download(remote: RemoteFile, local_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        remote.source.repo,
        remote.path,
        revision=remote.source.revision,
        local_dir=local_dir,
    )


def prepare(
    *, spec_name: str, local_dir: Path, seed_dir: Optional[Path], dry_run: bool
) -> int:
    spec = SPECS[spec_name]
    if manifest_is_current(local_dir, spec_name, spec):
        print(f"up to date: {local_dir} ({spec_name})")
        return 0
    base_index = fetch_json(spec.base, INDEX)
    files = plan_files(
        base_files=remote_files(spec.base),
        engram_files=remote_files(spec.engram),
        base_index=base_index,
        engram_index=fetch_json(spec.engram, INDEX),
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    # Conservative: a seeded file is a copy unless it can be hard-linked.
    missing = sum(
        f.size
        for f in files.values()
        if not (local_dir / f.path).is_file()
        or (local_dir / f.path).stat().st_size != f.size
    )
    free = shutil.disk_usage(local_dir).free
    print(
        f"{spec_name}: {len(files)} files, up to {missing / 2**30:.1f} GiB to "
        f"seed or download, {free / 2**30:.1f} GiB free at {local_dir}"
    )
    if free < missing + DISK_MARGIN_BYTES:
        print("blocked: not enough free disk")
        return 1
    if dry_run:
        return 0
    (local_dir / MANIFEST).unlink(missing_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = dict(
            zip(
                files,
                pool.map(
                    lambda f: ensure_file(
                        f, local_dir=local_dir, seed_dir=seed_dir, download=hub_download
                    ),
                    files.values(),
                ),
            )
        )
    for shard in engram_shards(base_index["weight_map"]):
        dtypes = {t["dtype"] for t in safetensors_header(local_dir / shard).values()}
        print(f"{shard}: dtypes {sorted(dtypes)}")
    (local_dir / CONFIG).write_text(
        json.dumps(
            derived_config(fetch_json(spec.base, CONFIG), spec.drop_quant_fields),
            indent=2,
        )
        + "\n"
    )
    (local_dir / INDEX).write_text(
        json.dumps(derived_index(base_index, local_dir), indent=2) + "\n"
    )
    manifest = dict(
        spec=spec_name,
        sources={k: list(v) for k, v in spec._asdict().items() if isinstance(v, Source)},
        status="complete",
        files={
            path: dict(
                source=f.source.repo,
                size=f.size,
                sha256=f.sha256,
                blob_id=f.blob_id,
                outcome=outcomes[path],
                mtime_ns=(local_dir / path).stat().st_mtime_ns,
            )
            for path, f in sorted(files.items())
        },
    )
    (local_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    counts = {o: list(outcomes.values()).count(o) for o in set(outcomes.values())}
    print(f"complete: {local_dir} ({spec_name}) {counts}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, choices=sorted(SPECS))
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-dir",
        type=Path,
        help="Hard-link matching files from here instead of downloading them",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed_dir is not None and args.seed_dir.resolve() == args.local_dir.resolve():
        parser.error("--seed-dir must differ from --local-dir")
    return prepare(
        spec_name=args.spec,
        local_dir=args.local_dir,
        seed_dir=args.seed_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
