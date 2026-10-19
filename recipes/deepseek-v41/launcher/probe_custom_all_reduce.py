"""Probe CustomAllReduceV2 against NCCL on a decode-sized BF16 payload.

Builds the TP group with the server's own constructor and custom all-reduce on,
so a failed IPC/symmetric-memory setup shows up as the same silent NCCL fallback
the server would take. Then compares the custom path with NCCL bit for bit and
times both, eager and in CUDA graphs. Latency uses host wall-clock around a
final synchronize: CUDA event timestamps are unreliable under CC.

Exit codes: 0 pass, 1 mismatch or error, 2 custom all-reduce did not initialize,
3 hang or timeout. Name the GPUs explicitly; the probe refuses to guess:

    CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=python timeout 120 \
        python recipes/deepseek-v41/launcher/probe_custom_all_reduce.py --precompile
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import logging
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PASS, FAIL, NOT_INITIALIZED, HANG = 0, 1, 2, 3


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[384, 192, 96, 6, 1],
        help="Token rows per payload; the first is the headline shape. 384 is "
        "64 requests x 6 DSpark verify positions.",
    )
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument(
        "--graph-reps",
        type=int,
        default=32,
        help="All-reduces captured per CUDA graph, to amortize replay launch cost",
    )
    p.add_argument("--soak-payloads", type=int, default=64)
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument(
        "--timeout",
        type=int,
        default=100,
        help="Hard limit for the ranks; they are killed and reported as a hang",
    )
    p.add_argument(
        "--precompile",
        action="store_true",
        help="Build the JIT kernels first under --precompile-timeout, so a cold "
        "cache cannot read as a hang",
    )
    p.add_argument("--precompile-timeout", type=int, default=900)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    p.add_argument("--precompile-only", action="store_true", help=argparse.SUPPRESS)
    return p


# Parent: spawn the ranks, enforce the deadline, read their reports.


def visible_gpu_count():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the GPUs under test")
    return len(visible.split(","))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def kill_groups(procs):
    if not procs:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for proc in procs:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(5)


# Returns each exit code, or None when the deadline hit and every group was killed.
def run_children(*, commands, env, timeout, logs):
    procs = [
        subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for command, log in zip(commands, logs)
    ]
    deadline = time.monotonic() + timeout
    try:
        while any(proc.poll() is None for proc in procs):
            if time.monotonic() > deadline:
                kill_groups(procs)
                return None
            time.sleep(0.5)
        return [proc.returncode for proc in procs]
    finally:
        kill_groups([proc for proc in procs if proc.poll() is None])


# Returns None on success, else the probe's exit code.
def precompile(args, out_dir):
    env = dict(os.environ, MAX_JOBS=os.environ.get("MAX_JOBS", "4"))
    command = [sys.executable, __file__, "--precompile-only"]
    with (out_dir / "precompile.log").open("w") as log:
        codes = run_children(
            commands=[command], env=env, timeout=args.precompile_timeout, logs=[log]
        )
    if codes is None:
        print(f"HANG: JIT precompile exceeded {args.precompile_timeout}s")
        return HANG
    if codes != [0]:
        print(f"FAIL: JIT precompile exited {codes}; see {out_dir}/precompile.log")
        return FAIL
    return None


def verdict(reports, *, timed_out, world_size):
    stages = {r: rep.get("stage") for r, rep in reports.items()}
    if timed_out:
        return HANG, f"HANG: killed at the deadline; last stage per rank {stages}"
    if len(reports) != world_size:
        return FAIL, f"FAIL: only ranks {sorted(reports)} wrote a report"
    errors = {r: rep["error"] for r, rep in reports.items() if rep.get("error")}
    if errors:
        return FAIL, f"FAIL: {errors}"
    stuck = [r for r, rep in reports.items() if "custom_all_reduce" not in rep]
    if stuck:
        return FAIL, f"FAIL: ranks {stuck} exited before group init; stages {stages}"
    ca = [rep["custom_all_reduce"] for rep in reports.values()]
    if not all(c["initialized"] for c in ca):
        warnings = [w for c in ca for w in c["warnings"]]
        return NOT_INITIALIZED, f"NOT INITIALIZED: {ca[0]['summary']} {warnings}"
    bad = [r for r, rep in reports.items() if not rep["correctness"]["all_match"]]
    if bad:
        return FAIL, f"MISMATCH on ranks {bad}: see correctness in the report"
    return PASS, "PASS: CustomAllReduceV2 initialized and matches NCCL"


def parent(args):
    world_size = visible_gpu_count()
    if world_size < 2:
        raise ValueError("Need at least two visible GPUs")
    # Unwind on an outer `timeout` so run_children still kills the rank sessions.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="probe-custom-ar-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.precompile:
        code = precompile(args, out_dir)
        if code is not None:
            return code
    port = free_port()
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    commands = [
        [sys.executable, __file__, *forwarded_args(args), "--worker-rank", str(rank)]
        + ["--output-dir", str(out_dir)]
        for rank in range(world_size)
    ]
    logs = [(out_dir / f"rank{r}.log").open("w") for r in range(world_size)]
    started = time.monotonic()
    try:
        codes = run_children(
            commands=commands, env=env, timeout=args.timeout, logs=logs
        )
    finally:
        for log in logs:
            log.close()
    reports = {
        r: json.loads((out_dir / f"rank{r}.json").read_text())
        for r in range(world_size)
        if (out_dir / f"rank{r}.json").exists()
    }
    code, text = verdict(reports, timed_out=codes is None, world_size=world_size)
    summary = dict(verdict=text, exit_code=code, output_dir=str(out_dir))
    summary["wall_seconds"] = round(time.monotonic() - started, 1)
    summary["reports"] = reports
    print(json.dumps(summary, indent=2))
    print(text)
    return code


def forwarded_args(args):
    out = ["--rows", *map(str, args.rows), "--hidden", str(args.hidden)]
    out += ["--iters", str(args.iters), "--graph-reps", str(args.graph_reps)]
    out += ["--soak-payloads", str(args.soak_payloads), "--timeout", str(args.timeout)]
    return out + (["--skip-latency"] if args.skip_latency else [])


# Worker: one rank, one GPU.


# Saved at every stage so a hang still leaves evidence of where it stopped.
class Report:
    def __init__(self, path):
        self.path = path
        self.data = dict(stage="start")

    def stage(self, name):
        self.data["stage"] = name
        self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))


def make_payload(*, rows, hidden, seed, device, exact):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if exact:
        # Integer-valued, so the 2-rank sum is exact in BF16 whatever the order.
        ints = torch.randint(-8, 9, (rows, hidden), generator=gen, device=device)
        return ints.to(torch.bfloat16)
    return torch.randn(rows, hidden, generator=gen, device=device).to(torch.bfloat16)


def nccl_reference(group, x):
    import torch.distributed as dist

    ref = x.clone()
    dist.all_reduce(ref, group=group.device_group)
    return ref


def compare(out, ref, *, world_size):
    diff = (out.float() - ref.float()).abs()
    # Two operands round once, so any order gives the same bits;
    # more operands can differ by an ulp.
    if world_size == 2:
        bad = out != ref
    else:
        bad = diff > ref.float().abs() * 2.0**-7
    return dict(
        mismatched=int(bad.sum()),
        numel=out.numel(),
        max_abs_diff=float(diff.max()),
    )


def check_eager(*, group, ca, rows, hidden, rank, count, exact):
    import torch

    device = group.device
    results = []
    for i in range(count):
        x = make_payload(
            rows=rows, hidden=hidden, seed=1000 * i + rank, device=device, exact=exact
        )
        saved = x.clone()
        out = ca.custom_all_reduce(x)
        ref = nccl_reference(group, saved)
        results.append((out, ref, torch.equal(x, saved)))
    torch.cuda.synchronize()
    worst = dict(mismatched=0, numel=0, max_abs_diff=0.0)
    for out, ref, input_intact in results:
        stats = compare(out, ref, world_size=group.world_size)
        worst["mismatched"] += stats["mismatched"] + (0 if input_intact else 1)
        worst["numel"] += stats["numel"]
        worst["max_abs_diff"] = max(worst["max_abs_diff"], stats["max_abs_diff"])
    return worst


def check_graph(*, group, ca, rows, hidden, rank, algo, replays=8):
    import torch

    device = group.device
    static_in = make_payload(
        rows=rows, hidden=hidden, seed=rank, device=device, exact=False
    )
    ca.override_algo = algo
    try:
        ca.custom_all_reduce(static_in)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with ca.capture(), torch.cuda.graph(graph):
            static_out = ca.custom_all_reduce(static_in)
    finally:
        ca.override_algo = None
    worst = dict(mismatched=0, numel=0, max_abs_diff=0.0)
    for i in range(replays):
        fresh = make_payload(
            rows=rows,
            hidden=hidden,
            seed=7000 + 31 * i + rank,
            device=device,
            exact=False,
        )
        static_in.copy_(fresh)
        graph.replay()
        ref = nccl_reference(group, fresh)
        torch.cuda.synchronize()
        stats = compare(static_out, ref, world_size=group.world_size)
        worst["mismatched"] += stats["mismatched"]
        worst["numel"] += stats["numel"]
        worst["max_abs_diff"] = max(worst["max_abs_diff"], stats["max_abs_diff"])
    return worst


def host_us(fn, *, iters, divisor, cpu_group):
    import torch
    import torch.distributed as dist

    samples = []
    for _ in range(3):
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        local = (time.perf_counter() - start) / iters / divisor * 1e6
        gathered = [None] * dist.get_world_size(cpu_group)
        dist.all_gather_object(gathered, local, group=cpu_group)
        samples.append(max(gathered))
    return dict(median_us=statistics.median(samples), min_us=min(samples))


def capture_graph(*, reps, body, ca=None, pynccl=None):
    import torch

    with contextlib.ExitStack() as stack:
        if pynccl is not None:
            stack.enter_context(pynccl.change_state(enable=True))
        if ca is not None:
            stack.enter_context(ca.capture())
        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps):
                body()
    torch.cuda.synchronize()
    return graph


def time_rows(*, group, ca, rows, hidden, args, algos):
    import torch
    import torch.distributed as dist

    device = group.device
    cpu_group = group.cpu_group
    # One reused input keeps L2 hot, as in decode where the previous kernel
    # just wrote it.
    x = make_payload(
        rows=rows, hidden=hidden, seed=group.rank, device=device, exact=False
    )
    y = torch.empty_like(x)
    zeros = torch.zeros_like(x)
    pynccl = group.pynccl_comm
    iters, reps = args.iters, args.graph_reps
    result = dict(rows=rows, nbytes=x.nbytes)

    def nccl_out():
        pynccl.outplace_all_reduce(x, out_tensor=y)

    with pynccl.change_state(enable=True):
        result["nccl_pynccl_eager"] = host_us(
            nccl_out, iters=iters, divisor=1, cpu_group=cpu_group
        )
    result["nccl_torch_eager_inplace"] = host_us(
        lambda: dist.all_reduce(zeros, group=group.device_group),
        iters=iters,
        divisor=1,
        cpu_group=cpu_group,
    )
    for name, divisor in (("graph_x1", 1), ("graph_amortized", reps)):
        graph = capture_graph(reps=divisor, body=nccl_out, pynccl=pynccl)
        result[f"nccl_pynccl_{name}"] = host_us(
            graph.replay, iters=iters, divisor=divisor, cpu_group=cpu_group
        )
    result["custom_eager"] = host_us(
        lambda: ca.custom_all_reduce(x), iters=iters, divisor=1, cpu_group=cpu_group
    )
    for label, algo in algos:
        ca.override_algo = algo
        try:
            for name, divisor in (("graph_x1", 1), ("graph_amortized", reps)):
                graph = capture_graph(
                    reps=divisor, body=lambda: ca.custom_all_reduce(x), ca=ca
                )
                result[f"custom_{label}_{name}"] = host_us(
                    graph.replay, iters=iters, divisor=divisor, cpu_group=cpu_group
                )
        finally:
            ca.override_algo = None
    return result


def describe_custom_all_reduce(*, group, capture, nbytes):
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        dispatch_custom_allreduce,
    )
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    ca = group.ca_comm
    cls = dispatch_custom_allreduce(group=group.cpu_group, device=group.device)
    info = dict(
        dispatch_class=cls.__name__,
        ca_comm_class=None if ca is None else type(ca).__name__,
        warnings=[m for level, m in capture.records if level >= logging.WARNING],
        log=[m for level, m in capture.records if level < logging.WARNING],
    )
    info["initialized"] = isinstance(ca, CustomAllReduceV2) and not ca.disabled
    if not info["initialized"]:
        info["summary"] = (
            f"ca_comm is {info['ca_comm_class']} (dispatch picked {cls.__name__}), "
            "not an enabled CustomAllReduceV2"
        )
        return info
    info["summary"] = "CustomAllReduceV2 is the TP group's ca_comm"
    info["has_multicast"] = ca.has_multicast
    info["max_push_size"] = ca.max_push_size
    info["max_pull_size"] = ca.max_pull_size
    info["num_push_blocks"] = ca.config.num_push_blocks
    info["num_pull_blocks"] = ca.config.num_pull_blocks
    info["graph_mode_supported"] = ca._is_graph_mode_supported
    info["eager_pick"] = str(ca._pick_config(nbytes, False))
    info["graph_pick"] = str(ca._pick_config(nbytes, True))
    return info


def worker(args):
    import torch

    rank = args.worker_rank
    report = Report(args.output_dir / f"rank{rank}.json")
    stack_file = (args.output_dir / f"rank{rank}.stack").open("w")
    faulthandler.dump_traceback_later(max(args.timeout - 10, 5), file=stack_file)
    # The default SIGALRM action ends an orphaned rank if the parent is gone.
    signal.alarm(args.timeout + 10)
    capture = LogCapture()
    logging.getLogger("sglang").addHandler(capture)
    logging.getLogger("sglang").setLevel(logging.INFO)
    try:
        run_worker(args=args, rank=rank, report=report, capture=capture)
    except Exception as exc:
        report.data["error"] = f"{type(exc).__name__}: {exc}"
        report.save()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_file.close()
    torch.cuda.synchronize()


def run_worker(*, args, rank, report, capture):
    import torch

    import sglang.srt.distributed.parallel_state as ps
    from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
    from sglang.srt.utils.common import is_confidential_compute

    world_size = visible_gpu_count()
    torch.cuda.set_device(rank)
    report.data.update(
        rank=rank,
        world_size=world_size,
        gpu=torch.cuda.get_device_name(),
        compute_capability=list(torch.cuda.get_device_capability()),
        confidential_compute=is_confidential_compute(),
        confidential_compute_env_override=os.environ.get("SGLANG_FORCE_CONFIDENTIAL_COMPUTE"),
        torch=torch.__version__,
        payload=dict(rows=args.rows, hidden=args.hidden, dtype="bfloat16"),
    )
    report.stage("dist_init")
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=rank,
        backend="nccl",
    )
    report.stage("group_init")
    start = time.perf_counter()
    group = ps.init_model_parallel_group(
        group_ranks=[list(range(world_size))],
        local_rank=rank,
        backend="nccl",
        use_pynccl=True,
        use_custom_allreduce=True,
        group_name="tp",
    )
    nbytes = args.rows[0] * args.hidden * 2
    custom = describe_custom_all_reduce(group=group, capture=capture, nbytes=nbytes)
    custom["init_seconds"] = time.perf_counter() - start
    report.data["custom_all_reduce"] = custom
    report.stage("group_ready")
    if not custom["initialized"]:
        return
    ca = group.ca_comm
    report.data["correctness"] = run_correctness(
        args=args, group=group, ca=ca, algo_type=AllReduceAlgo, report=report
    )
    if not args.skip_latency:
        report.data["latency_us"] = run_latency(
            args=args, group=group, ca=ca, algo_type=AllReduceAlgo, report=report
        )
    report.stage("done")
    ca.close()
    torch.distributed.destroy_process_group()


def run_correctness(*, args, group, ca, algo_type, report):
    import torch

    report.stage("correctness_eager")
    hidden, rank = args.hidden, group.rank
    out = dict(eager={}, graph={})
    for rows in args.rows:
        shape = torch.empty(rows, hidden, dtype=torch.bfloat16, device=group.device)
        if not ca.should_custom_ar(shape):
            out["eager"][rows] = dict(skipped="should_custom_ar is False")
            continue
        out["eager"][rows] = dict(
            random=check_eager(
                group=group,
                ca=ca,
                rows=rows,
                hidden=hidden,
                rank=rank,
                count=args.soak_payloads,
                exact=False,
            ),
            exact_ints=check_eager(
                group=group,
                ca=ca,
                rows=rows,
                hidden=hidden,
                rank=rank,
                count=4,
                exact=True,
            ),
        )
    report.stage("correctness_graph")
    headline = args.rows[0]
    for label, algo in (
        ("push", algo_type.ONE_SHOT_PUSH),
        ("pull_1shot", algo_type.ONE_SHOT_PULL),
        ("pull_2shot", algo_type.TWO_SHOT_PULL),
    ):
        report.stage(f"correctness_graph_{label}")
        try:
            out["graph"][label] = check_graph(
                group=group,
                ca=ca,
                rows=headline,
                hidden=hidden,
                rank=rank,
                algo=algo,
            )
        except Exception as exc:
            out["graph"][label] = dict(error=f"{type(exc).__name__}: {exc}")
    failures = [
        v
        for case in out["eager"].values()
        for v in (case.get("random"), case.get("exact_ints"))
        if v is not None and v["mismatched"]
    ] + [v for v in out["graph"].values() if v.get("error") or v.get("mismatched")]
    out["all_match"] = not failures
    return out


def run_latency(*, args, group, ca, algo_type, report):
    import torch

    report.stage("latency")
    algos = (
        ("push", algo_type.ONE_SHOT_PUSH),
        ("pull_1shot", algo_type.ONE_SHOT_PULL),
        ("pull_2shot", algo_type.TWO_SHOT_PULL),
    )
    out = []
    for i, rows in enumerate(args.rows):
        shape = torch.empty(
            rows, args.hidden, dtype=torch.bfloat16, device=group.device
        )
        if not ca.should_custom_ar(shape):
            continue
        report.stage(f"latency_rows_{rows}")
        shape_algos = algos if i == 0 else ()
        out.append(
            time_rows(
                group=group,
                ca=ca,
                rows=rows,
                hidden=args.hidden,
                args=args,
                algos=shape_algos,
            )
        )
    return out


def precompile_only():
    import torch

    from sglang.kernels.ops.communication import all_reduce
    from sglang.srt.distributed.device_communicators import (
        custom_all_reduce_utils as utils,
    )

    world_size = visible_gpu_count()
    torch.cuda.set_device(0)
    all_reduce.IPCManager()
    all_reduce._init_communicator()
    all_reduce.get_all_reduce_module(torch.bfloat16, world_size)
    # The first server launch on a GPU set runs this cross-process cudaIpc check
    # and caches it per CUDA_VISIBLE_DEVICES; under CC it takes about 45 s.
    peer = utils.gpu_p2p_access_check(0, 1)
    print(f"precompiled; peer access 0->1: {peer}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.precompile_only:
        precompile_only()
        return PASS
    if args.worker_rank is not None:
        worker(args)
        return PASS
    try:
        return parent(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return FAIL


if __name__ == "__main__":
    sys.exit(main())
