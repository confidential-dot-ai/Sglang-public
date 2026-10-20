"""Download the pinned official V4.1 Flash snapshot without filling a small disk."""

import argparse
import json
import shutil
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir", type=Path, default=Path("/data/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = [
        f
        for f in info.siblings
        if not args.metadata_only or not f.rfilename.endswith(".safetensors")
    ]
    if any(f.size is None for f in files):
        raise RuntimeError(
            "Hugging Face did not provide all file sizes; cannot budget disk space"
        )
    args.local_dir.mkdir(parents=True, exist_ok=True)
    total = sum(f.size for f in files)
    # Existing complete-sized files may be reused; Hugging Face still checks its
    # local download metadata against the pinned revision. Partial files are not
    # counted, making this a conservative bound for a resumed download.
    existing = sum(
        f.size
        for f in files
        if (args.local_dir / f.rfilename).is_file()
        and (args.local_dir / f.rfilename).stat().st_size == f.size
    )
    required = total - existing + 8 * 1024**3
    free = shutil.disk_usage(args.local_dir).free
    manifest = dict(
        repo=REPO,
        revision=info.sha,
        metadata_only=args.metadata_only,
        selected_bytes=total,
        existing_complete_size_bytes=existing,
        free_bytes=free,
        required_free_bytes=required,
        status="planned",
    )
    path = args.local_dir / "download-manifest.json"
    if free < required:
        manifest["status"] = "blocked_disk_space"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Download blocked: need {required / 1e9:.1f} GB free, have {free / 1e9:.1f} GB at {args.local_dir}"
        )
        return 1
    manifest["status"] = "downloading"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=args.local_dir,
        ignore_patterns=["*.safetensors"] if args.metadata_only else None,
        max_workers=4,
    )
    manifest["status"] = "metadata_downloaded" if args.metadata_only else "complete"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['status']}: {args.local_dir} ({info.sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
