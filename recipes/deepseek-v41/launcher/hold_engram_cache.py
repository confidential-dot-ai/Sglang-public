"""Keep immutable Engram checkpoint pages resident for all serving processes.

Run with sufficient RLIMIT_MEMLOCK (e.g. systemd LimitMEMLOCK=infinity).
Read-only mappings avoid copy-on-write allocation when mlock faults pages.
Stopping this process releases the locks; checkpoint files are never modified.
"""

import argparse
import ctypes
import json
import mmap
import os
import signal
import socket
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Write this process's PID here once every shard is locked "
        "(the container healthcheck and launcher read it)",
    )
    args = parser.parse_args()
    if args.ready_file:
        args.ready_file.unlink(missing_ok=True)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    files = sorted({v for k, v in index["weight_map"].items() if ".engram.embed." in k})
    if not files:
        raise ValueError("Checkpoint has no Engram tensors")
    paths = [args.model / name for name in files]
    total = sum(p.stat().st_size for p in paths)
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if total + 32 * 1024**3 > available:
        raise MemoryError(
            "Insufficient available RAM to lock Engram with a 32 GiB reserve"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.mlock.restype = ctypes.c_int
    mappings = []
    for path in paths:
        with path.open("rb") as f:
            mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        view = np.frombuffer(mapping, dtype=np.uint8)
        print(f"Locking {path.name}: {mapping.size()} bytes", flush=True)
        if libc.mlock(view.ctypes.data, view.nbytes):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        mappings.append((mapping, view))
        print(f"Resident: {path.name}", flush=True)
    print(
        f"READY: {total} checkpoint bytes locked read-only in shared CPU page cache",
        flush=True,
    )
    if args.ready_file:
        staging = args.ready_file.with_suffix(".tmp")
        staging.write_text(f"{os.getpid()}\n")
        staging.replace(args.ready_file)
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"READY=1", address)
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
