"""Test the exact Engram host gather in isolation before loading any weights.

The capability check uses libcuda directly, so it also works without the serving
dependencies. On supported hosts, test both anonymous and shared mappings in
eager execution and CUDA graph replay. No privileged host settings are changed.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import mmap
import os
from pathlib import Path


def host_access_error(attributes):
    if not attributes["host_register_supported"]:
        return "cudaHostRegister is unsupported; the current pinned CPU Engram path cannot run here."
    if not attributes["can_map_host_memory"]:
        return (
            "GPU cannot map host memory; Engram's direct host gather cannot run here."
        )
    return None


def capabilities():
    driver = ctypes.CDLL("libcuda.so.1")
    status = driver.cuInit(0)
    if status:
        raise RuntimeError(f"cuInit failed: {status}")
    result = {}
    # CUdevice_attribute constants from the CUDA driver headers.
    for name, attr in (
        ("can_map_host_memory", 19),
        ("pageable_memory_access", 88),
        ("host_register_supported", 99),
    ):
        value = ctypes.c_int()
        status = driver.cuDeviceGetAttribute(ctypes.byref(value), attr, 0)
        if status:
            raise RuntimeError(f"cuDeviceGetAttribute({name}) failed: {status}")
        result[name] = value.value
    return result


def gather_probe():
    import torch

    # Load the actual kernel without SGLang's model/server import dependencies.
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/kernels/ops/embeddings/engram_gather.py"
    )
    spec = importlib.util.spec_from_file_location("dsv41_engram_gather_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = {}
    rows, dim, block = 17, 256, 32
    nbytes = rows * (dim + dim // block)
    for layout in ("private", "shared"):
        fd = os.memfd_create("dsv41-engram-probe", 0) if layout == "shared" else -1
        if fd != -1:
            os.ftruncate(fd, nbytes)
        mm = mmap.mmap(
            fd,
            nbytes,
            flags=mmap.MAP_SHARED
            if fd != -1
            else mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        )
        raw = torch.frombuffer(mm, dtype=torch.uint8)
        weight = raw[: rows * dim].view(torch.float8_e4m3fn).view(rows, dim)
        scale = raw[rows * dim :].view(rows, dim // block)
        weight.copy_(
            (torch.arange(rows * dim).reshape(rows, dim) % 11 - 5).to(
                torch.float8_e4m3fn
            )
        )
        scale.copy_(
            (torch.arange(rows * (dim // block)).reshape(rows, -1) % 7 + 124).to(
                torch.uint8
            )
        )
        # Include the special e8m0 exponent-zero case.
        weight[0, :block] = 1
        scale[0, 0] = 0
        torch.cuda.init()
        status = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), nbytes, 0)
        if int(status):
            raise RuntimeError(
                f"{layout}: cudaHostRegister failed: {int(status)} ({status})"
            )
        try:
            ids = torch.tensor([0, 16, 3, 8, 3], dtype=torch.int64, device="cuda")
            out = torch.empty((len(ids), dim), dtype=torch.bfloat16, device="cuda")

            def reference():
                index = ids.cpu()
                values = weight.float()[index].unflatten(-1, (-1, block))
                scales = scale.view(torch.float8_e8m0fnu).float()[index]
                return (values * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)

            def run():
                module.engram_gather(
                    weight.data_ptr(),
                    scale.data_ptr(),
                    ids,
                    out,
                    dim,
                    block,
                    row_hi=rows,
                )

            run()
            torch.testing.assert_close(out.cpu(), reference(), rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            ids.copy_(torch.tensor([16, 0, 5, 1, 9], device="cuda"))
            graph.replay()
            torch.testing.assert_close(out.cpu(), reference(), rtol=0, atol=0)
            results[layout] = "exact eager and graph replay match"
            del graph
        finally:
            torch.cuda.synchronize()
            status = torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())
            if int(status):
                raise RuntimeError(f"cudaHostUnregister failed: {int(status)}")
            weight = scale = raw = None
            mm.close()
            if fd != -1:
                os.close(fd)
    return results


def main():
    result = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all")}
    try:
        result["capabilities"] = capabilities()
        error = host_access_error(result["capabilities"])
        if error:
            raise RuntimeError(
                error
                + " Disabling pinning is not a fallback on GPUs without pageable host access."
            )
        result["gather"] = gather_probe()
        result["passed"] = True
    except Exception as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
