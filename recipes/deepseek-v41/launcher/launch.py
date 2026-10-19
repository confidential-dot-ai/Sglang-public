"""Reproducible single-node DeepSeek V4.1 N prefill + M decode PD bring-up.

Standard-library only: --dry-run never imports SGLang or touches CUDA. This is
a qualification candidate, not a measured performance profile. See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# A container image has no .git; recipes/deepseek-v41/container/build.sh writes this.
SOURCE_RECORD = ROOT / "SOURCE_RECORD.json"
GIB = 1024**3
# The prefill graph cap the server applies when none is given (deepseek_v4_hook.py).
CAPTURED_INDEXER_MAX_SEQ_LEN = 16384
# Worker TP sizes. TP1 cannot hold the ~294 GiB of weights on one B200 or B300.
TP_SIZES = (2, 4, 8)

# Per-rank memory of a DSV4.1 FP8 worker, measured on the B200 CC VM from the
# release-21eacbd216 TP2 pair's logs (2026-10-19). Pool sizing (--memory-profile)
# derives KV pools from these; HBM fit still needs a real boot.
#   memory.total 183359 MiB; avail at weight-load start 175.25 GiB.
#   Weights: target 142.88 + DSpark draft 3.88 GiB per rank at TP2.
#   Decode pool: 4,194,304 tokens and 64 SWA rings took 8.10 + 0.13 GiB.
#   Prefill pool: 1,572,864 tokens and 16 + 256 SWA rings took 5.56 + 0.24 GiB.
#   Graphs: prefill 12.68 GiB at chunk 12288 (15.28 at 16384), decode 1.75 GiB.
GPU_PROFILES = {
    "B200": dict(hbm_gib=183359 / 1024, measured=True),
    # 288 GB nominal; confirm memory.total on the first B300 boot.
    "B300": dict(hbm_gib=288e9 / GIB, measured=False),
}
CONTEXT_OVERHEAD_GIB = 3.81
WEIGHTS_GIB_TP2 = 146.76
KV_BYTES_PER_TOKEN = 1938
SWA_RING_BYTES = int(10.5 * 2**20)
DECODE_GRAPH_GIB = 1.75
POOL_ROUND_TOKENS = 65536
# A CED-split PP2 prefill stage holds only its encoder layers, every expert at
# TP1/EP1, and no DSpark draft. Weights are checkpoint tensor bytes (6.88-7.04
# GiB per layer; the B200 TP2 rank measures 0.998x half the checkpoint), plus
# embedding and vision tower on stage 0 and the LM head on stage 1. KV per token
# counts each stage's compressed owners: a 584 B latent slot (448 fp8 + 64x2
# bf16 rope + 8) and a 68 B FP4 index slot, one per two tokens at ratio 2.
# Layers 0-7 own kv/index source 2; layers 8-20 own sources 8, 14 (ratio 2)
# and 20 (ratio 1). All 40 layers give 1902 B, 1938 B with the DSpark draft
# (measured). Estimates; only the default 8,13 partition is modelled.
CED_PP2_STAGES = (
    dict(layers="0-7", weights_gib=57.33, kv_bytes=326, ring_share=8 / 40),
    dict(layers="8-20", weights_gib=90.69, kv_bytes=1304, ring_share=13 / 40),
)


def pool_size(text):
    return text if text == "auto" else int(text)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Local V4.1 checkpoint")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--num-prefill", type=int, default=1, help="Prefill workers")
    p.add_argument("--num-decode", type=int, default=1, help="Decode workers")
    for role in ("prefill", "decode"):
        p.add_argument(
            f"--{role}-tp",
            type=int,
            choices=TP_SIZES,
            default=2,
            help=f"TP (= EP) size of each {role} worker",
        )
    p.add_argument(
        "--gpus",
        help="N*prefill-tp + M*decode-tp GPUs, prefill workers first, in worker "
        "order; default 0..count-1",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8030,
        help="Router port; prefill i uses +1+i, decode j +1+N+j",
    )
    p.add_argument(
        "--bootstrap-port", type=int, default=8998, help="Prefill i uses this +i"
    )
    p.add_argument(
        "--router-metrics-port",
        type=int,
        default=29000,
        help="Router Prometheus port; distinct per replica on one host",
    )
    p.add_argument(
        "--nccl-port",
        type=int,
        default=29500,
        help="Prefill i uses +i, decode j +N+j",
    )
    p.add_argument("--context-length", type=int, default=16384)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--prefill-requests", type=int, default=8)
    p.add_argument("--decode-requests", type=int, default=16)
    p.add_argument(
        "--decode-graph-extra-bs",
        type=lambda text: [int(x) for x in text.split(",") if x],
        default=[],
        metavar="BS[,BS...]",
        help="Decode graph batch sizes added to the powers of two, so common "
        "batches do not pad up to the next power",
    )
    p.add_argument(
        "--decode-polling-interval",
        type=int,
        default=1,
        help="Decode polls KV-transfer state every this many scheduler steps",
    )
    p.add_argument(
        "--tokenizer-workers",
        type=int,
        default=1,
        help="Tokenizer processes per worker; each one tokenizes a ~38K-token "
        "chat prompt in about 0.18 s, which caps a single process near 5.7 req/s",
    )
    p.add_argument(
        "--prefill-schedule-policy",
        choices=("fcfs", "hrrn", "lpm", "shortest-prefill-first"),
        default="fcfs",
        help="Prefill waiting-queue order; hrrn favours short uncached prompts "
        "but ages long ones by the prefill tokens processed while they wait",
    )
    p.add_argument(
        "--request-time-stats",
        action="store_true",
        help="Log per-request phase durations (bootstrap, queue, forward, transfer) "
        "on both roles; diagnostic",
    )
    for role in ("prefill", "decode"):
        p.add_argument(
            f"--{role}-env",
            action="append",
            default=[],
            metavar="SGLANG_KEY=VALUE",
            help="Worker environment override; SGLANG_* or NCCL_* keys only",
        )
    p.add_argument(
        "--ced-split",
        action="store_true",
        help="CED split: prefill runs only the encoder layers (no DSpark); decode "
        "replays each prompt tail through every layer",
    )
    p.add_argument(
        "--prefill-pp",
        type=int,
        default=1,
        help="Pipeline stages for each prefill worker on its two GPUs (TP1 per stage)",
    )
    p.add_argument(
        "--prefill-pp-partition",
        default=None,
        help="Layers per prefill pipeline stage (SGLANG_PP_LAYER_PARTITION); stages "
        "must split at kv_source layers. With --ced-split the default is 8,13.",
    )
    p.add_argument(
        "--prefill-hoist-per-batch",
        action="store_true",
        help="Not available on cc-main (its rewritten low-ratio indexer has its own "
        "hoisting); kept so old commands fail clearly",
    )
    p.add_argument(
        "--prefill-graph-eager-indexer",
        action="store_true",
        help="Replaced by --prefill-graph-max-seq-len; kept so old commands fail "
        "clearly",
    )
    p.add_argument(
        "--prefill-graph-max-seq-len",
        type=int,
        metavar="N",
        help="Longest context (cached prefix plus new tokens) the split prefill graphs "
        "replay; longer contexts prefill the whole step eagerly. Default: "
        f"--context-length when it exceeds the server's {CAPTURED_INDEXER_MAX_SEQ_LEN} "
        "cap (without --prefill-bounded-replay), so every allowed context replays; "
        "otherwise the server's cap. Above that cap the prefill ratio-1/2 index "
        "layers run as eager breaks (SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER). "
        "Needs --split-cuda-graphs; not validated with --prefill-bounded-replay",
    )
    p.add_argument(
        "--prefill-swa-prefix-tails",
        type=int,
        help="Prefill-only --swa-prefix-tails; default keeps the server's 4x request cap",
    )
    p.add_argument(
        "--prefill-hicache-ratio",
        type=float,
        help="Enable HiCache on prefill with a host pool this many times its GPU KV "
        "pool; decode has no radix cache to extend",
    )
    p.add_argument(
        "--prefill-hicache-size-gib",
        type=int,
        help="HiCache host pool per prefill rank in GiB (--hicache-size) instead of "
        "--prefill-hicache-ratio; at B300 pool sizes a ratio would exceed host RAM",
    )
    p.add_argument(
        "--prefill-hicache-io-backend",
        choices=("direct", "kernel"),
        default="direct",
        help="HiCache host<->GPU copies: direct uses the copy engines; under CC the "
        "kernel backend took 72 ms per transfer and 48%% of prefill GPU time",
    )
    p.add_argument(
        "--prefill-hicache-write-policy",
        choices=("write_back", "write_through", "write_through_selective"),
        default="write_back",
        help="write_back copies KV to host only when the GPU pool evicts it",
    )
    p.add_argument(
        "--max-total-tokens",
        type=pool_size,
        default=65536,
        help="Per worker KV cap; auto derives it from --memory-profile",
    )
    p.add_argument(
        "--decode-max-total-tokens",
        type=pool_size,
        help="Decode KV cap; defaults to --max-total-tokens; auto as above",
    )
    p.add_argument(
        "--memory-profile",
        choices=tuple(GPU_PROFILES),
        help="GPU memory model for auto pool sizes and the launch plan's memory "
        "estimate; preflight checks the GPUs' memory.total against it",
    )
    for role, gib in (("prefill", 10.0), ("decode", 6.0)):
        p.add_argument(
            f"--{role}-transient-reserve-gib",
            type=float,
            default=gib,
            help=f"HBM {role} keeps free of KV pools for activations and kernel "
            "workspaces (the eager indexer's logits tiles, up to 2 GiB each); "
            "used only by --memory-profile sizing",
        )
    p.add_argument("--mem-fraction", type=float, default=0.80)
    p.add_argument("--dspark-block-size", type=int, default=5)
    p.add_argument(
        "--oracle-commit-length",
        type=int,
        default=0,
        help="Benchmark-only forced-token replay; 0 disables it",
    )
    p.add_argument(
        "--no-dspark", action="store_true", help="Matched target-only control"
    )
    p.add_argument(
        "--candidate-indexer",
        choices=("torch", "deepgemm"),
        default="torch",
        help="DeepGEMM is an unqualified A/B candidate; requires sparse MQA kernels",
    )
    p.add_argument(
        "--host-layout", choices=("auto", "shared", "private"), default="auto"
    )
    p.add_argument(
        "--transport",
        choices=("rdma", "intra-node-nvlink", "tcp"),
        default="intra-node-nvlink",
    )
    p.add_argument("--ib-device")
    p.add_argument(
        "--devtools-mode",
        choices=("off", "on"),
        default="off",
        help="Required existing GPU DevTools mode; checks the mode without changing it",
    )
    p.add_argument(
        "--gpu-models",
        default="B300,GB300",
        help="Comma-separated GPU name substrings accepted by preflight",
    )
    p.add_argument("--parallel-load", action="store_true")
    p.add_argument(
        "--custom-all-reduce",
        choices=("off", "on"),
        default="off",
        help="on: TP groups use CustomAllReduceV2 (CUDA IPC push/pull) instead of the "
        "NCCL control, and preflight probes it with probe_custom_all_reduce.py; "
        "off passes --disable-custom-all-reduce. FlashInfer fusion stays off",
    )
    p.add_argument(
        "--prefill-bounded-replay",
        action="store_true",
        help="Run graph prefill layers 21-39 over padded SWA tail buckets",
    )
    p.add_argument(
        "--split-cuda-graphs",
        action="store_true",
        help="Breakable prefill/decode graphs with eager CPU Engram staging",
    )
    p.add_argument(
        "--kernel-cache-dir",
        type=Path,
        help="Reuse a shared kernel cache from a compatible prior run",
    )
    p.add_argument("--smoke-timeout", type=int, default=600)
    p.add_argument(
        "--resident-engram-pid",
        type=int,
        help="PID holding complete read-only mlocked Engram shards",
    )
    p.add_argument(
        "--cpu-engram-staging",
        action="store_true",
        help="Experimental eager CPU gather from shared checkpoint mappings",
    )
    p.add_argument(
        "--compact-host-memory",
        action="store_true",
        help="Before each worker starts, drop the page cache and compact host memory "
        "(sudo -n), then require free 2 MiB blocks for the CC driver's CUDA init",
    )
    p.add_argument("--host-reserve-gib", type=float, default=32)
    p.add_argument("--startup-timeout", type=int, default=18000)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/dsv41-pd"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    return p


def worker_specs(args):
    """Per-worker role, name, TP size and ports; prefill workers first."""
    n, m = args.num_prefill, args.num_decode
    specs = [
        dict(
            role="prefill",
            name="prefill" if n == 1 else f"prefill{i}",
            tp=args.prefill_tp,
            port=args.port + 1 + i,
            nccl_port=args.nccl_port + i,
            bootstrap_port=args.bootstrap_port + i,
        )
        for i in range(n)
    ]
    # Decode serves no bootstrap registry; its flag is only the default prefill
    # target for requests that omit bootstrap_port.
    specs += [
        dict(
            role="decode",
            name="decode" if m == 1 else f"decode{j}",
            tp=args.decode_tp,
            port=args.port + 1 + n + j,
            nccl_port=args.nccl_port + n + j,
            bootstrap_port=args.bootstrap_port,
        )
        for j in range(m)
    ]
    return specs


def weights_gib(tp):
    """Per-rank weights; only TP2 is measured, others assume even sharding + 5%."""
    return WEIGHTS_GIB_TP2 if tp == 2 else WEIGHTS_GIB_TP2 * 2 / tp * 1.05


def swa_rings(args, role):
    """Requests plus prefix tails: each holds one sliding-window ring per rank."""
    if role == "decode":
        return args.decode_requests
    tails = args.prefill_swa_prefix_tails
    return args.prefill_requests * (5 if tails is None else 1) + (tails or 0)


def rank_budget(args, role, *, weights, kv_bytes, ring_share=1.0, tokens=None):
    """One rank's HBM budget in GiB; pool_tokens is the largest KV pool (rounded
    down) that leaves the role's transient reserve free and keeps weights plus
    pools within --mem-fraction of HBM. With `tokens`, it reports that pool."""
    hbm = GPU_PROFILES[args.memory_profile]["hbm_gib"]
    graphs = 0.0
    if args.split_cuda_graphs:
        # Linear in the largest prefill bucket: 12.68 GiB at 12288, 15.28 at 16384.
        graphs = (
            12.68 + (args.chunk_size - 12288) * (15.28 - 12.34) / 4096
            if role == "prefill"
            else DECODE_GRAPH_GIB
        )
    transient = getattr(args, f"{role}_transient_reserve_gib")
    rings_gib = swa_rings(args, role) * SWA_RING_BYTES * ring_share / GIB
    room = min(
        hbm - CONTEXT_OVERHEAD_GIB - weights - graphs - transient,
        hbm * args.mem_fraction - weights,
    )
    max_tokens = (
        max(0, int((room - rings_gib) * GIB // kv_bytes))
        // POOL_ROUND_TOKENS
        * POOL_ROUND_TOKENS
    )
    tokens = max_tokens if tokens is None else tokens
    pool_gib = tokens * kv_bytes / GIB + rings_gib
    return dict(
        hbm_gib=round(hbm, 2),
        weights_gib=round(weights, 2),
        graphs_gib=round(graphs, 2),
        transient_reserve_gib=transient,
        kv_bytes_per_token=kv_bytes,
        pool_tokens=tokens,
        pool_gib=round(pool_gib, 2),
        max_pool_tokens=max_tokens,
        free_gib=round(hbm - CONTEXT_OVERHEAD_GIB - weights - graphs - pool_gib, 2),
        contexts_resident=tokens // args.context_length,
    )


def pool_estimate(args, role, tp, tokens=None):
    """Per-rank HBM budget of one worker under --memory-profile.

    A PP2 CED prefill is budgeted per stage; both stages share one token pool,
    so the tighter stage sets max_pool_tokens.
    """
    if role == "prefill" and args.prefill_pp > 1:
        if not args.ced_split or args.prefill_pp_partition not in (None, "8,13"):
            raise ValueError(
                "--memory-profile models PP prefill only as a CED split with the "
                "default 8,13 partition"
            )

        def stage_budget(stage, stage_tokens):
            return dict(
                layers=stage["layers"],
                **rank_budget(
                    args,
                    role,
                    weights=stage["weights_gib"],
                    kv_bytes=stage["kv_bytes"],
                    ring_share=stage["ring_share"],
                    tokens=stage_tokens,
                ),
            )

        max_tokens = min(
            stage_budget(stage, None)["max_pool_tokens"] for stage in CED_PP2_STAGES
        )
        stages = [
            stage_budget(stage, max_tokens if tokens is None else tokens)
            for stage in CED_PP2_STAGES
        ]
        return dict(
            pool_tokens=stages[0]["pool_tokens"],
            max_pool_tokens=max_tokens,
            contexts_resident=stages[0]["contexts_resident"],
            stages=stages,
            estimated="per-stage weights, KV bytes, graphs and SWA share",
        )
    return rank_budget(
        args, role, weights=weights_gib(tp), kv_bytes=KV_BYTES_PER_TOKEN, tokens=tokens
    )


def hicache_host_bytes(args, tp, tokens):
    """Host RAM of one prefill worker's HiCache pools (all ranks); 0 if off."""
    ranks = args.prefill_pp if args.prefill_pp > 1 else tp
    if args.prefill_hicache_size_gib is not None:
        return args.prefill_hicache_size_gib * GIB * ranks
    if args.prefill_hicache_ratio is None:
        return 0
    rings = swa_rings(args, "prefill") * SWA_RING_BYTES
    if args.prefill_pp > 1 and args.ced_split:
        # Each stage mirrors only its own layers' device pool.
        return sum(
            int(
                args.prefill_hicache_ratio
                * (tokens * stage["kv_bytes"] + rings * stage["ring_share"])
            )
            for stage in CED_PP2_STAGES
        )
    device = tokens * KV_BYTES_PER_TOKEN + rings
    return int(args.prefill_hicache_ratio * device) * ranks


def resolve_pool_sizes(args):
    """Replace "auto" pool sizes with the --memory-profile estimate, once."""
    for name, role, tp in (
        ("max_total_tokens", "prefill", args.prefill_tp),
        ("decode_max_total_tokens", "decode", args.decode_tp),
    ):
        if getattr(args, name) != "auto":
            continue
        if args.memory_profile is None:
            raise ValueError(f"--{name.replace('_', '-')} auto needs --memory-profile")
        setattr(args, name, pool_estimate(args, role, tp)["max_pool_tokens"])


def validate(args):
    if args.prefill_hoist_per_batch:
        raise ValueError(
            "--prefill-hoist-per-batch is not ported to cc-main: upstream rewrote "
            "the DeepSeek-V4.1 low-ratio prefill indexer it optimized"
        )
    if args.prefill_graph_eager_indexer:
        raise ValueError(
            "--prefill-graph-eager-indexer is replaced by --prefill-graph-max-seq-len N"
        )
    if args.decode_polling_interval < 1:
        raise ValueError("--decode-polling-interval must be at least 1")
    if args.tokenizer_workers < 1:
        raise ValueError("--tokenizer-workers must be at least 1")
    if any(bs < 1 for bs in args.decode_graph_extra_bs):
        raise ValueError("--decode-graph-extra-bs sizes must be positive")
    if args.decode_graph_extra_bs and not args.split_cuda_graphs:
        raise ValueError("--decode-graph-extra-bs requires --split-cuda-graphs")
    if args.prefill_graph_max_seq_len is not None:
        if args.prefill_graph_max_seq_len <= 0:
            raise ValueError("prefill_graph_max_seq_len must be positive")
        if not args.split_cuda_graphs:
            raise ValueError("--prefill-graph-max-seq-len requires --split-cuda-graphs")
        if (
            args.prefill_graph_max_seq_len > CAPTURED_INDEXER_MAX_SEQ_LEN
            and args.prefill_bounded_replay
        ):
            raise ValueError(
                "--prefill-graph-max-seq-len above "
                f"{CAPTURED_INDEXER_MAX_SEQ_LEN} is not validated with "
                "--prefill-bounded-replay"
            )
    if args.prefill_hicache_ratio is not None and args.prefill_hicache_ratio <= 1:
        raise ValueError("--prefill-hicache-ratio must exceed 1")
    if args.prefill_hicache_size_gib is not None:
        if args.prefill_hicache_ratio is not None:
            raise ValueError(
                "--prefill-hicache-size-gib and --prefill-hicache-ratio are exclusive"
            )
        if args.prefill_hicache_size_gib <= 0:
            raise ValueError("--prefill-hicache-size-gib must be positive")
    if args.ced_split and args.prefill_bounded_replay:
        raise ValueError(
            "--ced-split and --prefill-bounded-replay are mutually exclusive"
        )
    if args.prefill_bounded_replay and not args.split_cuda_graphs:
        raise ValueError("--prefill-bounded-replay requires --split-cuda-graphs")
    if (
        args.oracle_commit_length < 0
        or args.oracle_commit_length > args.dspark_block_size + 1
    ):
        raise ValueError("Oracle commit must fit the DSpark verification window")
    if args.split_cuda_graphs and not args.cpu_engram_staging:
        raise ValueError("Split graphs require CPU staging")
    if args.resident_engram_pid is not None and args.resident_engram_pid <= 0:
        raise ValueError("Resident Engram PID must be positive")
    if (args.parallel_load or args.resident_engram_pid) and not args.cpu_engram_staging:
        raise ValueError(
            "Parallel loading and resident-page credit require CPU staging"
        )
    if args.num_prefill <= 0 or args.num_decode <= 0:
        raise ValueError("num_prefill and num_decode must be positive")
    if args.prefill_pp > 1 and args.prefill_tp != 2:
        raise ValueError("--prefill-pp splits a two-GPU worker; needs --prefill-tp 2")
    resolve_pool_sizes(args)
    num_gpus = args.num_prefill * args.prefill_tp + args.num_decode * args.decode_tp
    gpu_error = f"--gpus must contain {num_gpus} distinct nonnegative indices"
    try:
        gpus = (
            [int(x) for x in args.gpus.split(",")]
            if args.gpus is not None
            else list(range(num_gpus))
        )
    except ValueError as exc:
        raise ValueError(gpu_error) from exc
    if len(gpus) != num_gpus or len(set(gpus)) != num_gpus or min(gpus) < 0:
        raise ValueError(gpu_error)
    specs = worker_specs(args)
    ports = [
        args.port,
        *(spec["port"] for spec in specs),
        *(spec["bootstrap_port"] for spec in specs if spec["role"] == "prefill"),
        *(spec["nccl_port"] for spec in specs),
        args.router_metrics_port,
    ]
    if len(set(ports)) != len(ports) or any(not 1024 <= x <= 65535 for x in ports):
        raise ValueError(
            "Router, worker, bootstrap, NCCL and router metrics ports must be distinct and in 1024..65535"
        )
    for name in (
        "context_length",
        "chunk_size",
        "prefill_requests",
        "decode_requests",
        "max_total_tokens",
        "dspark_block_size",
        "startup_timeout",
        "smoke_timeout",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.decode_max_total_tokens is not None and args.decode_max_total_tokens <= 0:
        raise ValueError("decode_max_total_tokens must be positive")
    if not 0 < args.mem_fraction < 1 or args.host_reserve_gib < 0:
        raise ValueError("Require 0 < mem-fraction < 1 and host-reserve-gib >= 0")
    if (
        args.chunk_size > args.context_length
        or args.context_length > args.max_total_tokens
    ):
        raise ValueError("Require chunk-size <= context-length <= max-total-tokens")
    return gpus, ports


def prefill_graph_max_seq_len(args):
    """The prefill graph cap: the explicit one, else the whole context once it is
    past the captured indexer's window. A capped step past the cap runs every layer
    eagerly, and under CC that host time dwarfs the GPU work, while the eager index
    layers make a longer cap cost no static graph memory (their scoring is sized by
    the live rows, and the page table is rebuilt per replay)."""
    if args.prefill_graph_max_seq_len is not None:
        return args.prefill_graph_max_seq_len
    if (
        args.split_cuda_graphs
        and not args.prefill_bounded_replay
        and args.context_length > CAPTURED_INDEXER_MAX_SEQ_LEN
    ):
        return args.context_length
    return None


def prefill_parallelism(command, *, pp):
    if pp == 1:
        return command
    if pp != 2:
        raise ValueError("--prefill-pp supports 1 or 2 stages on a two-GPU worker")
    command = list(command)
    command[command.index("--tp-size") + 1] = "1"
    command[command.index("--ep-size") + 1] = "1"
    return command + ["--pp-size", str(pp)]


def without_speculative_args(command):
    out = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        if item.startswith("--speculative-"):
            skip = True
            continue
        out.append(item)
    return out


def source_record():
    """HEAD and `git status --porcelain` of the tree this launcher runs from."""
    if SOURCE_RECORD.exists():
        return json.loads(SOURCE_RECORD.read_text())
    git = ["git", "-C", str(ROOT)]
    return {
        "revision": subprocess.check_output(
            [*git, "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": subprocess.check_output([*git, "status", "--porcelain"], text=True),
    }


# The client must drop an idle pooled connection before the server closes it. With
# uvicorn's 5 s default, a router POST on a connection uvicorn was closing failed with
# "error sending request" and was not retried; decode then waited 300 s for KV that
# prefill never got (2026-10-17, about once per router run). The image's router has no
# pool-idle flag; its HTTP client keeps idle connections up to reqwest's 90 s default.
ROUTER_POOL_IDLE_S = 90
SERVER_KEEP_ALIVE_S = 300


def build_plan(args):
    gpus, _ = validate(args)
    revision = source_record()["revision"]
    common_env = {
        "PYTHONPATH": str(ROOT / "python"),
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN": "1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": args.host_layout,
        # Avoid one instance evicting the other's checkpoint page cache.
        "SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE": "0",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH": "0",
        "SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER": (
            "1" if args.candidate_indexer == "deepgemm" else "0"
        ),
        "SGLANG_RUST_SERVER": "0",
        "SGLANG_TIMEOUT_KEEP_ALIVE": str(SERVER_KEEP_ALIVE_S),
        "SGLANG_DSV41_ENGRAM_CPU_STAGING": "1" if args.cpu_engram_staging else "0",
    }
    if args.transport == "intra-node-nvlink":
        # Mooncake checks MC_FORCE_TCP's presence; even "0" forces TCP.
        common_env.update(
            SGLANG_MOONCAKE_CUSTOM_MEM_POOL="INTRA_NODE_NVLINK",
            MC_INTRANODE_NVLINK="true",
            MOONCAKE_PROTOCOL="rdma",
        )
    elif args.transport == "tcp":
        common_env.update(MC_FORCE_TCP="1", MOONCAKE_PROTOCOL="tcp")
    elif args.transport == "rdma":
        common_env.update(MOONCAKE_PROTOCOL="rdma")
    if args.oracle_commit_length:
        common_env["SGLANG_ORACLE_FIXED_COMMIT_LEN"] = str(args.oracle_commit_length)
    common = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model.resolve()),
        "--served-model-name",
        "dsv41",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--tp-size",
        "2",
        "--ep-size",
        "2",
        "--dp-size",
        "1",
        "--attention-backend",
        "dsv4",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "auto",
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-bootstrap-port",
        str(args.bootstrap_port),
        "--context-length",
        str(args.context_length),
        "--chunked-prefill-size",
        str(args.chunk_size),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--page-size",
        "256",
        "--kv-cache-dtype",
        "fp8_e4m3",
        *(["--disable-custom-all-reduce"] if args.custom_all_reduce == "off" else []),
        "--enforce-disable-flashinfer-allreduce-fusion",
        "--random-seed",
        "0",
    ]
    if args.ib_device:
        common += ["--disaggregation-ib-device", args.ib_device]
    if not args.no_dspark:
        common += [
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            str(args.dspark_block_size),
        ]
    bootstrap_index = common.index("--disaggregation-bootstrap-port") + 1
    workers = []
    first_gpu = 0
    for spec in worker_specs(args):
        role = spec["role"]
        worker_gpus = gpus[first_gpu : first_gpu + spec["tp"]]
        first_gpu += spec["tp"]
        # Prefill workers share one cache: their commands, hence kernels, match.
        cache = (
            args.kernel_cache_dir.resolve()
            if args.kernel_cache_dir
            else args.output_dir.resolve() / "cache" / revision / role
        )
        env = dict(
            common_env,
            CUDA_VISIBLE_DEVICES=",".join(map(str, worker_gpus)),
            TRITON_CACHE_DIR=str(cache / "triton"),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
            FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
            CUDA_CACHE_PATH=str(cache / "nv"),
            SGLANG_CACHE_DIR=str(cache),
        )
        if args.prefill_bounded_replay:
            env["SGLANG_OPT_DSV41_PREFILL_TAIL"] = "1"
        if args.ced_split:
            env["SGLANG_OPT_DSV41_CED_SPLIT"] = "1"
        if role == "prefill" and args.prefill_hicache_ratio is not None:
            # SWA-only evictions here are mostly superseded conversation tails; the
            # synchronous CC copy that demoted them took 8% of scheduler time for no
            # cache hits (2026-10-18). Leaf demotes still back up evicted tails.
            env["SGLANG_ENABLE_HICACHE_SWA_TOMBSTONE_BACKUP"] = "0"
        if (
            role == "prefill"
            and (prefill_graph_max_seq_len(args) or 0) > CAPTURED_INDEXER_MAX_SEQ_LEN
        ):
            env["SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER"] = "1"
        if role == "prefill" and args.prefill_pp > 1:
            # Layers 8-13 read layer 8's compressed KV and top-k, so the encoder
            # (layers 0-20) splits at the kv_source layer 8.
            env["SGLANG_PP_LAYER_PARTITION"] = args.prefill_pp_partition or (
                "8,13"
                if args.ced_split
                else ("14,26" if args.prefill_bounded_replay else "")
            )
            if not env["SGLANG_PP_LAYER_PARTITION"]:
                del env["SGLANG_PP_LAYER_PARTITION"]
        for assignment in getattr(args, f"{role}_env", []):
            key, sep, value = assignment.partition("=")
            if (
                not sep
                or not key.startswith(("SGLANG_", "NCCL_"))
                or not key.isidentifier()
            ):
                raise ValueError(f"Invalid worker environment override: {assignment!r}")
            env[key] = value
        requests = args.prefill_requests if role == "prefill" else args.decode_requests
        command = [*common]
        command[bootstrap_index] = str(spec["bootstrap_port"])
        for flag in ("--tp-size", "--ep-size"):
            command[command.index(flag) + 1] = str(spec["tp"])
        if role == "prefill":
            command = prefill_parallelism(command, pp=args.prefill_pp)
            if args.ced_split:
                command = without_speculative_args(command)
        command += [
            "--disaggregation-mode",
            role,
            "--port",
            str(spec["port"]),
            "--nccl-port",
            str(spec["nccl_port"]),
            "--max-running-requests",
            str(requests),
            "--max-total-tokens",
            str(
                args.decode_max_total_tokens
                if role == "decode" and args.decode_max_total_tokens
                else args.max_total_tokens
            ),
        ]
        if args.tokenizer_workers != 1:
            command += ["--tokenizer-worker-num", str(args.tokenizer_workers)]
        if role == "prefill" and args.prefill_schedule_policy != "fcfs":
            command += ["--schedule-policy", args.prefill_schedule_policy]
        if args.request_time_stats:
            command += ["--enable-request-time-stats-logging"]
        if role == "decode" and args.decode_polling_interval != 1:
            command += [
                "--disaggregation-decode-polling-interval",
                str(args.decode_polling_interval),
            ]
        if role == "prefill" and args.prefill_swa_prefix_tails is not None:
            command += ["--swa-prefix-tails", str(args.prefill_swa_prefix_tails)]
        if role == "prefill" and (
            args.prefill_hicache_ratio is not None
            or args.prefill_hicache_size_gib is not None
        ):
            # layer_first is the layout M3 qualified under CC; an explicit ratio or
            # size also turns off host-memory auto-sizing, which rejects the DSV4 pool.
            command += [
                "--enable-hierarchical-cache",
                *(
                    ["--hicache-ratio", str(args.prefill_hicache_ratio)]
                    if args.prefill_hicache_size_gib is None
                    else ["--hicache-size", str(args.prefill_hicache_size_gib)]
                ),
                "--hicache-mem-layout",
                "layer_first",
                "--hicache-io-backend",
                args.prefill_hicache_io_backend,
                "--hicache-write-policy",
                args.prefill_hicache_write_policy,
            ]
        if args.split_cuda_graphs:
            # Powers of two up to the chunk, plus the chunk itself so a full chunk
            # never pads to the next bucket.
            prefill_buckets = sorted(
                {1 << p for p in range(8, 15) if 1 << p <= args.chunk_size}
                | ({args.chunk_size} if args.chunk_size >= 256 else set())
            )
            if not prefill_buckets:
                raise ValueError("Split prefill graphs require chunk-size >= 256")
            decode_buckets = sorted(
                {1 << power for power in range(requests.bit_length())}
                | {bs for bs in args.decode_graph_extra_bs if bs <= requests}
                | {requests}
            )
            prefill_graph = {"backend": "breakable", "bs": prefill_buckets}
            max_seq_len = prefill_graph_max_seq_len(args)
            if max_seq_len is not None:
                prefill_graph["max_seq_len"] = max_seq_len
            graph_config = {
                "prefill": (
                    prefill_graph if role == "prefill" else {"backend": "disabled"}
                ),
                "decode": (
                    {"backend": "breakable", "bs": decode_buckets}
                    if role == "decode"
                    else {"backend": "disabled"}
                ),
            }
            command += ["--cuda-graph-config", json.dumps(graph_config)]
            # Explicit both ways: unset, the server turns it on for supported
            # configurations. Prefill runs it only under --prefill-bounded-replay
            # (a CED-split prefill must keep every encoder row).
            if role == "decode" or args.prefill_bounded_replay:
                command += ["--enable-decoder-swa-bounded-replay"]
            else:
                command += ["--no-enable-decoder-swa-bounded-replay"]
        elif role == "decode" and not args.cpu_engram_staging:
            command += ["--cuda-graph-max-bs-decode", str(requests)]
        else:
            command += ["--disable-decode-cuda-graph"]
        if not args.split_cuda_graphs:
            command += [
                "--disable-prefill-cuda-graph",
                "--enable-decoder-swa-bounded-replay",
            ]
        worker = dict(
            name=spec["name"],
            command=command,
            env=env,
            url=f"http://127.0.0.1:{spec['port']}",
        )
        if role == "prefill" and args.num_prefill > 1:
            worker["bootstrap_port"] = spec["bootstrap_port"]
        workers.append(worker)
    router = build_router(args=args, workers=workers)
    n, m, ptp, dtp = args.num_prefill, args.num_decode, args.prefill_tp, args.decode_tp
    prefill = (
        f"PP{args.prefill_pp}xTP1{' CED' if args.ced_split else ''} prefill"
        if args.prefill_pp > 1
        else f"TP{ptp} prefill"
    )
    if n == m == 1 and ptp == dtp == 2 and args.prefill_pp == 1:
        topology = "1x(TP2 prefill + TP2 decode)"
    else:
        topology = f"{n}x({prefill}) + {m}x(TP{dtp} decode)"
    plan = dict(
        revision=revision,
        status="unqualified",
        physical_gpus=gpus,
        topology=topology + "; CPU Engram; DP=1 per worker",
        collectives=dict(
            custom_all_reduce=args.custom_all_reduce,
            flashinfer_allreduce_fusion="off",
        ),
        workers=workers,
        router=router,
    )
    if args.memory_profile is not None:
        plan["memory"] = memory_summary(args)
    return plan


def memory_summary(args):
    """Per-rank HBM and host-RAM estimate per role for the launch plan."""
    profile = GPU_PROFILES[args.memory_profile]
    roles = {}
    for role, tp, tokens in (
        ("prefill", args.prefill_tp, args.max_total_tokens),
        ("decode", args.decode_tp, args.decode_max_total_tokens or args.max_total_tokens),
    ):
        roles[role] = pool_estimate(args, role, tp, tokens)
    host = args.num_prefill * hicache_host_bytes(
        args, args.prefill_tp, args.max_total_tokens
    )
    return dict(
        profile=args.memory_profile,
        profile_measured=profile["measured"],
        per_rank=roles,
        hicache_host_gib_all_prefill=round(host / GIB, 1),
        note="Estimate from B200 TP2 measurements; HBM fit requires a real boot.",
    )


def build_router(args, workers):
    prefill_args = []
    for spec in worker_specs(args)[: args.num_prefill]:
        prefill_args += [
            "--prefill",
            f"http://127.0.0.1:{spec['port']}",
            str(spec["bootstrap_port"]),
        ]
    # The manual policy pins each X-SMG-Routing-Key to one worker per role, keeping
    # a conversation's prefix cache on one prefill (and its decode radix cache, if
    # enabled, on one decode); unkeyed requests fall back to random.
    policy = ["--prefill-policy", "manual"] if args.num_prefill > 1 else []
    if args.num_decode > 1:
        policy += ["--decode-policy", "manual"]
    if policy:
        policy += ["--assignment-mode", "min_group"]
    decode_args = []
    for worker in workers[args.num_prefill :]:
        decode_args += ["--decode", worker["url"]]
    return dict(
        name="router",
        env={"PYTHONPATH": str(ROOT / "python")},
        url=f"http://127.0.0.1:{args.port}",
        command=[
            args.python,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            *prefill_args,
            *decode_args,
            *policy,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--prometheus-port",
            str(args.router_metrics_port),
        ],
    )


def checkpoint_budget(model, tp_sizes=(2,)):
    """Exact configured Engram payload; NOT a prediction of runtime HBM use."""
    raw = (model / "config.json").read_bytes()
    config = json.loads(raw)
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("Expected a deepseek_v41 checkpoint")
    config = {**config.get("text_config", {}), **config}
    rows = config.get("engram_num_embeddings", [])
    layers = config.get("engram_layer_ids", [])
    dim = config.get("engram_head_dim", 0)
    if (
        not rows
        or len(rows) != len(layers)
        or dim <= 0
        or dim % 32
        or any(x <= 0 for x in rows)
    ):
        raise ValueError("Missing or invalid Engram table dimensions in config.json")
    for tp in sorted(set(tp_sizes)):
        for name in ("num_attention_heads", "o_groups", "n_routed_experts"):
            value = config.get(name)
            if not isinstance(value, int) or value <= 0 or value % tp:
                raise ValueError(
                    f"TP{tp}/EP{tp} requires positive {name} divisible by {tp}, "
                    f"got {value!r}"
                )
    # MXFP4 tables (nvidia/DeepSeek-V4.1-Flash-NVFP4) pack two values per byte.
    quant = config.get("quantization_config", {})
    fp4 = str(quant.get("engram_embed_quant") or "").lower() == "mxfp4"
    per_instance = sum(rows) * ((dim // 2 if fp4 else dim) + dim // 32)
    shards = list(model.glob("*.safetensors"))
    if not shards:
        raise ValueError("No local safetensors checkpoint shards found")
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        expected = set(index["weight_map"].values())
        missing = sorted(expected - {p.name for p in shards})
        if missing:
            raise ValueError(
                f"Checkpoint download is incomplete: {len(missing)} indexed shards missing; "
                f"first missing shard: {missing[0]}"
            )
        shards = [p for p in shards if p.name in expected]
        tensor_bytes = index.get("metadata", {}).get("total_size", 0)
        if sum(p.stat().st_size for p in shards) < tensor_bytes:
            raise ValueError(
                "Checkpoint shards are truncated: smaller than indexed tensor bytes"
            )
    return dict(
        config_sha256=hashlib.sha256(raw).hexdigest(),
        engram_bytes_per_instance=per_instance,
        engram_bytes_pd_pair=2 * per_instance,
        checkpoint_files=len(shards),
        checkpoint_file_bytes=sum(p.stat().st_size for p in shards),
        note="Two TP groups allocate two tables. Excludes loader temporaries, OS, KV, graphs, and workspaces; HBM fit requires a real boot.",
    )


def clean_environment(overrides):
    # Do not let an earlier M3/PD experiment silently change the contract.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SGLANG_", "SGL_", "MC_", "MOONCAKE_", "NCCL_"))
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("PYTHONPATH", None)
    env.update(overrides)
    return env


def required_versions():
    requirements = (ROOT / "python/pyproject.toml").read_text()
    versions = {}
    for name in ("torch", "transformers", "flashinfer_python", "sglang-kernel"):
        match = re.search(
            r'"' + re.escape(name) + r'(?:\[[^]]+\])?==([^"; ]+)', requirements
        )
        if match is None:
            raise ValueError(f"Cannot resolve branch dependency pin for {name}")
        versions[name.replace("_", "-")] = match[1]
    return versions


def resident_engram_bytes(model, pid):
    """Credit measured resident bytes in read-only locked checkpoint mappings."""
    index = json.loads((model / "model.safetensors.index.json").read_text())
    files = {
        str((model / v).resolve())
        for k, v in index["weight_map"].items()
        if ".engram.embed." in k
    }
    credited = {}
    for section in re.split(
        r"(?=^[0-9a-f]+-[0-9a-f]+ )", Path(f"/proc/{pid}/smaps").read_text(), flags=re.M
    ):
        lines = section.splitlines()
        if not lines:
            continue
        header = lines[0].split(maxsplit=5)
        if len(header) != 6 or header[5] not in files:
            continue
        path = Path(header[5])
        stat = path.stat()
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in lines[1:]
            if ":" in line
        }
        if (
            "w" not in header[1]
            and int(header[2], 16) == 0
            and int(header[4]) == stat.st_ino
            and header[3] == f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
            and "lo" in fields.get("VmFlags", "").split()
            and int(fields.get("Size", "0").split()[0]) * 1024 >= stat.st_size
        ):
            credited[str(path)] = min(
                stat.st_size, int(fields.get("Rss", "0").split()[0]) * 1024
            )
    if set(credited) != files:
        raise ValueError(
            "Residency helper does not hold all Engram shards read-only and locked"
        )
    return sum(credited.values())


def custom_all_reduce_probe_gpus(plan):
    return [
        worker["env"]["CUDA_VISIBLE_DEVICES"]
        for worker in plan["workers"]
        # A PP worker runs TP1 per stage, so it has no TP all-reduce to probe.
        if "--pp-size" not in worker["command"]
    ]


def probe_custom_all_reduce(args, plan, gpus):
    worker = next(
        w for w in plan["workers"] if w["env"]["CUDA_VISIBLE_DEVICES"] == gpus
    )
    out = args.output_dir.resolve() / f"custom-ar-probe-{gpus.replace(',', '-')}"
    result = subprocess.run(
        [
            args.python,
            str(Path(__file__).with_name("probe_custom_all_reduce.py")),
            "--precompile",
            "--output-dir",
            str(out),
        ],
        env=clean_environment(worker["env"]),
        text=True,
        capture_output=True,
        timeout=1200,
        cwd=ROOT,
    )
    lines = (result.stdout or result.stderr).strip().splitlines()
    verdict = lines[-1] if lines else "no output"
    if result.returncode:
        raise RuntimeError(f"exit {result.returncode}: {verdict}; reports in {out}")
    return verdict


def preflight(args, plan):
    results, errors = {}, []

    def check(name, fn):
        print(f"Preflight: {name}", flush=True)
        try:
            results[name] = fn()
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    def run(command, env=None, timeout=60):
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, env=env, cwd=ROOT
        )
        if result.returncode:
            raise RuntimeError(
                f"exit {result.returncode}: {result.stdout}\n{result.stderr}".strip()
            )
        return result.stdout.strip()

    groups = args.num_prefill + args.num_decode
    check(
        "checkpoint",
        lambda: checkpoint_budget(args.model, (args.prefill_tp, args.decode_tp)),
    )
    if args.cpu_engram_staging and "checkpoint" in results:
        budget = results["checkpoint"]
        budget["engram_bytes_pd_pair"] = budget["engram_bytes_per_instance"]
        budget["note"] = (
            "Eager staging shares immutable checkpoint page cache across P/D; excludes OS, loader and GPU memory."
        )
    elif groups > 2 and "checkpoint" in results:
        budget = results["checkpoint"]
        budget["engram_bytes_pd_pair"] = groups * budget["engram_bytes_per_instance"]
        budget["note"] = (
            f"{groups} TP groups allocate one table each; excludes OS, loader and GPU memory."
        )
    check(
        "inventory",
        lambda: run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        ),
    )
    idle_gpus = []
    if "inventory" in results:
        devices = {
            int(row.split(",")[0]): [v.strip() for v in row.split(",")]
            for row in results["inventory"].splitlines()
        }
        for gpu in plan["physical_gpus"]:
            row = devices.get(gpu)
            if (
                row is None
                or not any(x in row[1] for x in args.gpu_models.split(","))
                or int(row[-1]) > 256
            ):
                errors.append(
                    f"GPU {gpu}: must be an idle {args.gpu_models} GPU (<=256 MiB used); found {row}"
                )
            else:
                idle_gpus.append(gpu)
            if row is not None and args.memory_profile is not None:
                expected = GPU_PROFILES[args.memory_profile]["hbm_gib"]
                total = int(row[3]) / 1024
                if abs(total - expected) > 0.02 * expected:
                    errors.append(
                        f"GPU {gpu}: memory.total {total:.1f} GiB is not the "
                        f"{args.memory_profile} profile's {expected:.1f} GiB"
                    )
    for name, option, required in (
        ("cc", "-f", "CC status: ON"),
        ("devtools", "-d", f"DevTools Mode: {args.devtools_mode.upper()}"),
        ("multi_gpu_cc", "-mgm", "Multi-GPU Mode: NVLE"),
    ):
        check(name, lambda option=option: run(["nvidia-smi", "conf-compute", option]))
        if name in results:
            prefix = required.split(":")[0] + ":"
            states = [
                line.strip() for line in results[name].splitlines() if prefix in line
            ]
            if not states or any(line != required for line in states):
                errors.append(f"{name}: requires {required!r}; found {results[name]!r}")
    check("topology", lambda: run(["nvidia-smi", "topo", "-m"]))
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    results["host_available_bytes"] = available
    locked_credit = 0
    if args.resident_engram_pid:
        check(
            "resident_engram_bytes",
            lambda: resident_engram_bytes(args.model, args.resident_engram_pid),
        )
        locked_credit = results.get("resident_engram_bytes", 0)
    # Every prefill worker pins its own HiCache host pools (one per rank).
    hicache_host = args.num_prefill * hicache_host_bytes(
        args, args.prefill_tp, args.max_total_tokens
    )
    results["hicache_host_bytes"] = hicache_host
    if (
        "checkpoint" in results
        and max(0, results["checkpoint"]["engram_bytes_pd_pair"] - locked_credit)
        + hicache_host
        + args.host_reserve_gib * GIB
        > available
    ):
        errors.append(
            "CPU memory cannot hold required Engram pages, the prefill HiCache host "
            "pools and the configured host reserve"
        )
    for port in validate(args)[1]:

        def port_check(port=port):
            with socket.socket() as s:
                # Like the servers, allow restart while old connections linger
                # in TIME_WAIT; an active listener still fails this bind.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
            return "free"

        check(f"port_{port}", port_check)
    targets = required_versions()
    results["required_versions"] = targets
    version_code = (
        "import json, importlib.metadata as m\n"
        "versions = {}\n"
        f"for name in {list(targets)!r}:\n"
        "    try: versions[name] = m.version(name)\n"
        "    except m.PackageNotFoundError: versions[name] = None\n"
        "print(json.dumps(versions))\n"
    )
    check(
        "installed_versions", lambda: json.loads(run([args.python, "-c", version_code]))
    )
    if "installed_versions" in results:
        for name, expected in targets.items():
            actual = results["installed_versions"].get(name)
            if actual is None or actual.split("+")[0] != expected:
                errors.append(
                    f"Dependency {name}: branch requires {expected}, installed {actual}"
                )
    dependency_code = (
        "import json, importlib.metadata as m; "
        "import sglang, torch, transformers, flashinfer, mooncake.engine, sglang_router; "
        "print(json.dumps({'sglang':sglang.__file__, 'torch':torch.__version__, "
        "'transformers':transformers.__version__, 'flashinfer':m.version('flashinfer-python')}))"
    )
    if args.candidate_indexer == "deepgemm":
        dependency_code += (
            "; import deep_gemm; "
            "assert callable(deep_gemm.get_paged_sparse_mqa_logits_metadata); "
            "assert callable(deep_gemm.fp8_fp4_paged_sparse_mqa_logits)"
        )
    check(
        "imports",
        lambda: run(
            [args.python, "-c", dependency_code],
            clean_environment(plan["workers"][0]["env"]),
            120,
        ),
    )
    # A small subprocess per GPU tests real host access before allocating tables.
    # An unsupported mapping must never be bypassed by disabling pinning.
    for gpu in idle_gpus:
        env = clean_environment(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "TRITON_CACHE_DIR": str(args.output_dir.resolve() / "probe-cache"),
            }
        )
        check(
            f"engram_gpu_{gpu}",
            lambda env=env: run(
                [
                    args.python,
                    str(
                        Path(__file__).with_name(
                            "probe_staged_engram.py"
                            if args.cpu_engram_staging
                            else "probe_engram.py"
                        )
                    ),
                ]
                + (["--split-graphs"] if args.split_cuda_graphs else []),
                env,
                120,
            ),
        )
    if args.custom_all_reduce == "on":
        for gpus in custom_all_reduce_probe_gpus(plan):
            if all(int(gpu) in idle_gpus for gpu in gpus.split(",")):
                check(
                    f"custom_all_reduce_gpus_{gpus.replace(',', '_')}",
                    lambda gpus=gpus: probe_custom_all_reduce(args, plan, gpus),
                )
    results["errors"] = errors
    results["passed"] = not errors
    results["attestation"] = (
        "Mode checks only; CPU/GPU cryptographic attestation must be recorded separately."
    )
    return results


# The server falls back to NCCL silently when CustomAllReduceV2 setup fails, so
# the launcher reads two log lines: GroupCoordinator warns with the first when
# setup raises, and CustomAllReduceV2 logs the second on rank 0 once it is up.
CUSTOM_AR_FAILED_LINE = "Setup Custom allreduce failed"
CUSTOM_AR_READY_LINE = "All Reduce config:"


def require_custom_all_reduce(log_path):
    text = log_path.read_text(errors="replace")
    if CUSTOM_AR_FAILED_LINE in text:
        raise RuntimeError(f"{log_path.name}: custom all-reduce setup failed (NCCL)")
    if CUSTOM_AR_READY_LINE not in text:
        raise RuntimeError(f"{log_path.name}: CustomAllReduceV2 never initialized")


# Under CC, cuInit's secure-channel setup (ccslContextInitViaChannel) allocates
# order-9 (2 MiB) contiguous kernel pages. Mlocked Engram shards plus one
# worker's weight load leave the Normal zone without any, and the next worker's
# cuInit then fails as "No accelerator". Measured on the B200 CC VM, 2026-10-15.
CC_CONTIGUOUS_ORDER = 9
# Arbitrary floor; the two failing decode launches had zero such blocks.
MIN_FREE_CONTIGUOUS_BLOCKS = 64


def free_contiguous_blocks(buddyinfo, *, order=CC_CONTIGUOUS_ORDER):
    """Normal-zone free memory in blocks of 2**order pages that need no compaction."""
    blocks = 0
    for line in buddyinfo.splitlines():
        fields = line.split()
        if len(fields) > 4 and fields[3] == "Normal":
            counts = [int(count) for count in fields[4:]]
            blocks += sum(
                count << (size - order)
                for size, count in enumerate(counts)
                if size >= order
            )
    return blocks


def compact_host_memory():
    command = [
        "sh",
        "-c",
        "sync; echo 1 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory",
    ]
    # The container image runs as root (privileged) and ships no sudo.
    subprocess.run(
        command if os.geteuid() == 0 else ["sudo", "-n", *command], check=True
    )
    blocks = free_contiguous_blocks(Path("/proc/buddyinfo").read_text())
    if blocks < MIN_FREE_CONTIGUOUS_BLOCKS:
        raise RuntimeError(
            f"only {blocks} free 2 MiB host blocks after compaction "
            f"(need {MIN_FREE_CONTIGUOUS_BLOCKS}); CC cuInit would fail"
        )
    return blocks


def wait_ready(url, processes, timeout, path="/health"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, proc in processes:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited with {proc.returncode}; inspect its log"
                )
        try:
            with urllib.request.urlopen(url + path, timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{url} did not become ready within {timeout}s")


def owned_cleanup_pids(processes, gpus, resident_engram_pid):
    """Resolve exact scheduler PIDs on the assigned GPUs and our direct children."""
    parents = {proc.pid for _, proc in processes if proc.poll() is None}
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    allowed = {
        uuid.strip()
        for line in inventory.splitlines()
        for index, uuid in [line.split(",")]
        if int(index) in gpus
    }
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    owned = set(parents)
    for line in apps.splitlines():
        pid, uuid = line.split(",")
        pid = int(pid)
        if uuid.strip() not in allowed or pid == resident_engram_pid:
            continue
        try:
            # Each worker was started in its own session. A child may survive
            # its parent; the session id still proves which launch owns it.
            if os.getsid(pid) in {proc.pid for _, proc in processes}:
                owned.add(pid)
        except ProcessLookupError:
            pass
    owned.discard(resident_engram_pid)
    return owned


def serve(args, plan):
    processes, logs = [], []

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        groups = (
            [plan["workers"]]
            if args.parallel_load
            else [[item] for item in plan["workers"]]
        )
        for group in [*groups, [plan["router"]]]:
            if args.compact_host_memory and group[0]["name"] != "router":
                blocks = compact_host_memory()
                print(f"Compacted host memory: {blocks} free 2 MiB blocks", flush=True)
            for item in group:
                log = (args.output_dir / f"{item['name']}.log").open("w")
                logs.append(log)
                proc = subprocess.Popen(
                    item["command"],
                    env=clean_environment(item["env"]),
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append((item["name"], proc))
                print(
                    f"Started {item['name']} PID {proc.pid}; waiting for readiness",
                    flush=True,
                )
            for item in group:
                wait_ready(
                    item["url"],
                    processes,
                    args.startup_timeout,
                    path="/readiness" if item["name"] == "router" else "/health",
                )
                if item["name"] != "router":
                    with urllib.request.urlopen(
                        item["url"] + "/server_info", timeout=30
                    ) as response:
                        (
                            args.output_dir / f"{item['name']}-server-info.json"
                        ).write_bytes(response.read())
                    has_tp_group = "--pp-size" not in item["command"]
                    if args.custom_all_reduce == "on" and has_tp_group:
                        log_path = args.output_dir / f"{item['name']}.log"
                        require_custom_all_reduce(log_path)
        # Exercise the full router -> prefill -> decode path, beyond /health.
        request = urllib.request.Request(
            plan["router"]["url"] + "/v1/completions",
            data=json.dumps(
                dict(
                    model="dsv41",
                    prompt="The capital of France is",
                    max_tokens=8,
                    temperature=0,
                )
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=args.smoke_timeout) as response:
            result = json.load(response)
        (args.output_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        if (
            not result.get("choices")
            or not result["choices"][0].get("text", "").strip()
        ):
            raise RuntimeError("PD smoke returned no generated text")
        print(
            f"PD smoke passed: {plan['router']['url']}/v1 (model dsv41). Performance remains unqualified.",
            flush=True,
        )
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited with {proc.returncode}")
            time.sleep(2)
    finally:
        # Signal only verified exact PIDs, including orphaned GPU schedulers.
        gpus, _ = validate(args)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in owned_cleanup_pids(processes, gpus, args.resident_engram_pid):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                for _, proc in processes:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
        for log in logs:
            log.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=False)
        record = source_record()
        plan["git_status"] = record["status"]
        if "diff_sha256" in record:
            plan["source_diff_sha256"] = record["diff_sha256"]
        (args.output_dir / "launch-plan.json").write_text(
            json.dumps(plan, indent=2) + "\n"
        )
        print(
            "Checking checkpoint, hardware, dependencies, and CPU Engram access",
            flush=True,
        )
        checks = preflight(args, plan)
        (args.output_dir / "preflight.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        if not checks["passed"]:
            print("Preflight failed:\n" + "\n".join(checks["errors"]), file=sys.stderr)
            return 1
        if not args.preflight_only:
            serve(args, plan)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
