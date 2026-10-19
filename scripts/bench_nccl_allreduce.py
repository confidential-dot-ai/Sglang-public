#!/usr/bin/env python3
"""Benchmark BF16 all-reduces through torch/NCCL and PyNCCL.

Run this under the intended TP topology. Environment variables such as
``NCCL_PROTO`` and ``NCCL_ALGO`` must be set before launch because NCCL reads
them while creating each communicator.

Example (TP4 on one four-GPU NVLink island):

    PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1,2,3 \
      NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_NVLS_ENABLE=0 \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      scripts/bench_nccl_allreduce.py \
      --cuda-graph

Strict Confidential Computing + NVLS acceptance test (TP4):

    PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1,2,3 \
      NCCL_DEBUG_FILE=/tmp/nccl-cc-nvls.%h.%p.log \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      scripts/bench_nccl_allreduce.py --require-cc-nvls \
      --message-mib 0.125 16 127.125 192 --cuda-graph

``--require-cc-nvls`` reads the hardware CC state directly from NVML (it does
not trust ``SGLANG_CONFIDENTIAL_COMPUTE``), forces the NVLS algorithm so Ring
cannot silently substitute, and fails if NCCL cannot create/use multicast.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Exact BF16 tensors from the TP4 60K cold-prefill trace:
    # 10848 x 6144 = 127.125 MiB and 16384 x 6144 = 192 MiB.
    parser.add_argument(
        "--message-mib", type=float, nargs="+", default=[127.125, 192]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Also write the rank-0 JSON result to this path.",
    )
    parser.add_argument(
        "--require-cc",
        action="store_true",
        help="Require hardware-reported CC with DevTools mode disabled.",
    )
    parser.add_argument(
        "--require-cc-nvls",
        action="store_true",
        help=(
            "Require hardware-reported Confidential Computing, at least four "
            "ranks, and strict NCCL NVLS AllReduce with Simple protocol."
        ),
    )
    parser.add_argument(
        "--pynccl-library",
        action="append",
        default=None,
        help="PyNCCL library to benchmark; may be repeated.",
    )
    return parser.parse_args()


def get_hardware_cc_state() -> dict:
    """Read the real system CC state without honoring SGLang's CI override."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            state = pynvml.nvmlSystemGetConfComputeState()
            return {
                "query_ok": True,
                "enabled": int(getattr(state, "ccFeature", 0)) != 0,
                "cc_feature": int(getattr(state, "ccFeature", 0)),
                "devtools_mode": int(getattr(state, "devToolsMode", 0)),
                "environment": int(getattr(state, "environment", 0)),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        return {
            "query_ok": False,
            "enabled": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _version_number(version: tuple[int, ...]) -> int:
    padded = (*version, 0, 0, 0)
    return padded[0] * 10000 + padded[1] * 100 + padded[2]


def configure_strict_cc_nvls(world_size: int) -> dict:
    """Configure a fail-closed CC/NVLS validation before communicator init."""
    cc_state = require_hardware_cc()
    if world_size < 4:
        raise RuntimeError(
            f"--require-cc-nvls requires at least four ranks; got {world_size}"
        )

    nccl_version = torch.cuda.nccl.version()
    if _version_number(nccl_version) < 22400:
        raise RuntimeError(
            "Strict per-collective NCCL_ALGO selection requires NCCL >= 2.24; "
            f"PyTorch reports {nccl_version}"
        )

    required = {
        "NCCL_NVLS_ENABLE": "1",
        "NCCL_ALGO": "allreduce:NVLS",
        "NCCL_PROTO": "allreduce:Simple",
    }
    for name, value in required.items():
        existing = os.environ.get(name)
        if existing is not None and existing.lower() != value.lower():
            raise RuntimeError(
                f"--require-cc-nvls conflicts with {name}={existing!r}; "
                f"unset it or set {name}={value!r}"
            )
        os.environ[name] = value

    # INIT proves multicast capability/resource setup; NVLS shows multicast
    # allocation/binding; TUNING records the selected collective plan.
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NVLS,TUNING"
    return cc_state


def require_hardware_cc() -> dict:
    """Fail closed unless NVML proves production Confidential Computing."""
    cc_state = get_hardware_cc_state()
    if not cc_state["query_ok"]:
        raise RuntimeError(
            "Cannot validate Confidential Computing because NVML CC-state "
            f"detection failed: {cc_state.get('error')}"
        )
    if not cc_state["enabled"]:
        raise RuntimeError(
            "--require-cc requires hardware CC mode, but NVML reports it disabled"
        )
    if int(cc_state.get("devtools_mode", 1)) != 0:
        raise RuntimeError(
            "--require-cc requires DevTools mode disabled, but NVML reports "
            f"devtools_mode={cc_state.get('devtools_mode')}"
        )
    return cc_state


def check_all_reduce_correctness(
    group: dist.ProcessGroup,
    device: torch.device,
    rank: int,
    world_size: int,
) -> float:
    value = float(rank + 1)
    tensor = torch.full((4096,), value, dtype=torch.bfloat16, device=device)
    dist.all_reduce(tensor, group=group)
    expected = world_size * (world_size + 1) / 2
    max_abs_error = float((tensor.float() - expected).abs().max().item())
    if max_abs_error != 0.0:
        raise RuntimeError(
            "NCCL all-reduce correctness failed: "
            f"expected={expected}, max_abs_error={max_abs_error}"
        )
    return max_abs_error


def make_exact_pattern(
    numel: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rank-varying BF16 data whose TP4 sum is exactly representable."""
    offsets = torch.arange(numel, dtype=torch.int32, device=device)
    base = (offsets.remainder(17) - 8).to(torch.bfloat16)
    input_tensor = base * float(rank + 1)
    rank_sum = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    return input_tensor, base * rank_sum


def timed_ms(
    fn,
    warmup: int,
    iterations: int,
    cpu_group: dist.ProcessGroup,
) -> dict[str, float]:
    """Measure a collective and report both local and slowest-rank latency.

    Uses host wall-clock (``time.perf_counter``) bracketed by
    ``torch.cuda.synchronize()`` rather than ``torch.cuda.Event``-based GPU
    timing. Under real (ready) Confidential Computing, CUDA event timestamps
    have been observed to produce physically-impossible negative deltas --
    consistent with CC degrading GPU timer precision as a side-channel
    mitigation -- which breaks downstream log-ratio scoring. Host timing adds
    one synchronize's worth of overhead per measurement but stays valid under
    CC.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    local_ms = (end - start) * 1000.0 / iterations
    rank_times = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(rank_times, local_ms, group=cpu_group)
    return {
        "local_ms": local_ms,
        "max_rank_ms": max(rank_times),
        "min_rank_ms": min(rank_times),
    }


def add_bandwidths(timing: dict[str, float], message_bytes: int, world_size: int) -> None:
    critical_seconds = timing["max_rank_ms"] / 1000.0
    algorithm_gbps = message_bytes / critical_seconds / 1e9
    timing["algorithm_gbps"] = algorithm_gbps
    timing["bus_gbps"] = algorithm_gbps * 2.0 * (world_size - 1) / world_size


def capture_all_reduce(
    communicator: PyNcclCommunicator,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    cpu_group: dist.ProcessGroup,
) -> torch.cuda.CUDAGraph:
    # All ranks must enter NCCL graph capture in the same order. The eager
    # warmup also ensures lazy peer connections are established first.
    with communicator.change_state(enable=True):
        communicator.outplace_all_reduce(input_tensor, out_tensor=output_tensor)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with communicator.change_state(enable=True), torch.cuda.graph(graph):
        communicator.outplace_all_reduce(input_tensor, out_tensor=output_tensor)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    return graph


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    cc_state = get_hardware_cc_state()
    if args.require_cc_nvls:
        cc_state = configure_strict_cc_nvls(world_size)
    elif args.require_cc:
        cc_state = require_hardware_cc()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="gloo")
    atexit.register(dist.destroy_process_group)
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    assert coordinator.cpu_group is not None
    assert coordinator.device_group is not None
    rank = dist.get_rank(coordinator.cpu_group)
    correctness_max_abs_error = check_all_reduce_correctness(
        coordinator.device_group,
        device,
        rank,
        world_size,
    )

    communicators = []
    for library_path in args.pynccl_library or ["libnccl.so.2"]:
        communicator = PyNcclCommunicator(
            coordinator.cpu_group,
            device,
            library_path=library_path,
        )
        if not communicator.available:
            raise RuntimeError(f"PyNCCL is unavailable from {library_path}")
        communicators.append(
            (library_path, communicator.nccl.ncclGetVersion(), communicator)
        )

    results = []
    for message_mib in args.message_mib:
        numel = int(message_mib * 1024 * 1024) // 2
        numel = numel // 8 * 8
        torch_input = torch.zeros(numel, dtype=torch.bfloat16, device=device)
        pynccl_input, expected = make_exact_pattern(numel, rank, device)

        torch_ms = timed_ms(
            lambda: dist.all_reduce(torch_input, group=coordinator.device_group),
            args.warmup,
            args.iterations,
            coordinator.cpu_group,
        )
        add_bandwidths(torch_ms, numel * 2, world_size)

        pynccl_results = []
        for library_path, version, communicator in communicators:
            output_tensor = torch.empty_like(pynccl_input)
            with communicator.change_state(enable=True):
                eager_ms = timed_ms(
                    lambda: communicator.outplace_all_reduce(
                        pynccl_input, out_tensor=output_tensor
                    ),
                    args.warmup,
                    args.iterations,
                    coordinator.cpu_group,
                )
            add_bandwidths(eager_ms, numel * 2, world_size)
            eager_max_abs_error = float(
                (output_tensor.float() - expected.float()).abs().max().item()
            )
            if eager_max_abs_error != 0.0:
                raise RuntimeError(
                    f"PyNCCL eager correctness failed for {message_mib} MiB "
                    f"through {library_path}: max_abs_error={eager_max_abs_error}"
                )

            graph_ms = None
            graph_max_abs_error = None
            if args.cuda_graph:
                graph = capture_all_reduce(
                    communicator,
                    pynccl_input,
                    output_tensor,
                    coordinator.cpu_group,
                )
                graph_ms = timed_ms(
                    graph.replay,
                    args.warmup,
                    args.iterations,
                    coordinator.cpu_group,
                )
                add_bandwidths(graph_ms, numel * 2, world_size)
                graph_max_abs_error = float(
                    (output_tensor.float() - expected.float()).abs().max().item()
                )
                if graph_max_abs_error != 0.0:
                    raise RuntimeError(
                        f"PyNCCL CUDA-graph correctness failed for {message_mib} "
                        f"MiB through {library_path}: "
                        f"max_abs_error={graph_max_abs_error}"
                    )

            pynccl_results.append(
                {
                    "library": library_path,
                    "version": version,
                    "eager_outplace_ms": eager_ms,
                    "eager_max_abs_error": eager_max_abs_error,
                    "graph_replay_ms": graph_ms,
                    "graph_max_abs_error": graph_max_abs_error,
                }
            )

        results.append(
            {
                "message_mib": numel * 2 / (1024 * 1024),
                "torch_inplace_ms": torch_ms,
                "pynccl": pynccl_results,
            }
        )

    if dist.get_rank(coordinator.cpu_group) == 0:
        payload = {
            "world_size": world_size,
            "hardware_cc_state": cc_state,
            "require_cc": args.require_cc or args.require_cc_nvls,
            "require_cc_nvls": args.require_cc_nvls,
            "torch_nccl_version": torch.cuda.nccl.version(),
            "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE"),
            "nccl_algo": os.environ.get("NCCL_ALGO"),
            "nccl_proto": os.environ.get("NCCL_PROTO"),
            "nccl_buffsize": os.environ.get("NCCL_BUFFSIZE"),
            "nccl_min_ctas": os.environ.get("NCCL_MIN_CTAS"),
            "nccl_max_ctas": os.environ.get("NCCL_MAX_CTAS"),
            "correctness_max_abs_error": correctness_max_abs_error,
            "results": results,
        }
        rendered = json.dumps(payload, indent=2)
        print(rendered)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
