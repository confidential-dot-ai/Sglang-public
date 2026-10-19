"""Tiny checkpoint test for eager CPU staging; --cpu-only needs no GPU."""

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
spec = importlib.util.spec_from_file_location(
    "engram_staging", ROOT / "python/sglang/srt/layers/engram_staging.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def staged_dequant(table, indices):
    return (table.weight.float() * table.scale.float())[indices].to(torch.bfloat16)


def check_prefetched_rows(prefetcher):
    """Rows gathered on the worker thread equal the blocking gather, byte for byte."""
    hash_ids = torch.tensor([[[3, 0, 3], [1, 2, 0]], [[2, 2, 1], [0, 3, 3]]])
    prefetcher.start(hash_ids)
    for column, table in prefetcher.tables.items():
        indices = hash_ids[:, column]
        expected = table.gather_bytes(indices)
        actual = table._take_prefetched(indices)
        assert actual is not None
        for got, want in zip(actual, expected):
            assert torch.equal(got, want)
        # Rows are handed out once: a second lookup must gather again.
        assert table._take_prefetched(indices) is None
    prefetcher.start(hash_ids)
    for column, table in prefetcher.tables.items():
        # A different ids tensor never reuses the prefetched rows.
        assert table._take_prefetched(hash_ids[:, column].flip(0)) is None
    prefetcher.start(hash_ids)
    prefetcher.start(hash_ids[:0])
    assert all(t.prefetched is None for t in prefetcher.tables.values())


def count_blocking_gathers(tables):
    calls = []
    for table in tables:
        original = table.gather_bytes

        def counted(indices, original=original):
            calls.append(1)
            return original(indices)

        table.gather_bytes = counted
    return calls


def check_prefetched_lookup(prefetcher, ids):
    tables = list(prefetcher.tables.values())
    hash_ids = torch.stack([ids, ids.flip(1)], dim=1).cuda()
    reference = [staged_dequant(t, hash_ids[:, c].cpu()) for c, t in enumerate(tables)]
    calls = count_blocking_gathers(tables)
    prefetcher.start(hash_ids)
    for column, table in enumerate(tables):
        actual = table.lookup(hash_ids[:, column]).cpu()
        torch.testing.assert_close(actual, reference[column], rtol=0, atol=0)
    assert not calls, "prefetched lookups fell back to the blocking gather"
    # No prefetch pending (and a stale one for other ids): the blocking path runs.
    prefetcher.start(hash_ids)
    other = hash_ids.flip(0)
    for column, table in enumerate(tables):
        actual = table.lookup(other[:, column]).cpu()
        torch.testing.assert_close(
            actual, staged_dequant(table, other[:, column].cpu()), rtol=0, atol=0
        )
    assert len(calls) == len(tables)
    for table in tables:
        del table.gather_bytes


def capture_breakable(forward):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = BreakableCUDAGraph()
    with BreakableCUDAGraphCapture(graph, stream=stream):
        output = forward()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def check_prefetched_graph(prefetcher, ids):
    """Replay with fresh ids on every call: the prefetch break must hand each
    layer the rows of this replay's ids, bit-identical to the blocking path."""
    tables = list(prefetcher.tables.values())
    static_ids = ids.cuda()

    def forward(prefetch):
        hash_ids = torch.stack([(static_ids + 1) % 4, (static_ids + 2) % 4], dim=1)
        if prefetch:
            prefetcher.start(hash_ids)
        first = tables[0].lookup(hash_ids[:, 0])
        second = tables[1].lookup(hash_ids[:, 1])
        return first * 2 + second

    blocking, blocking_out = capture_breakable(lambda: forward(False))
    prefetched, prefetched_out = capture_breakable(lambda: forward(True))
    assert (len(blocking._segments), len(blocking._break_fns)) == (3, 2)
    assert (len(prefetched._segments), len(prefetched._break_fns)) == (4, 3)
    calls = count_blocking_gathers(tables)
    for current in (ids, ids.flip(0), ids.flip(1), torch.zeros_like(ids), ids):
        static_ids.copy_(current)
        reference = staged_dequant(tables[0], (current + 1) % 4) * 2 + staged_dequant(
            tables[1], (current + 2) % 4
        )
        blocking.replay()
        torch.testing.assert_close(blocking_out.cpu(), reference, rtol=0, atol=0)
        old_calls = len(calls)
        prefetched.replay()
        torch.testing.assert_close(prefetched_out.cpu(), reference, rtol=0, atol=0)
        assert len(calls) == old_calls, "graph replay used the blocking gather"
    for table in tables:
        del table.gather_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--split-graphs", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weight = torch.arange(-64, 64).reshape(4, 32).to(torch.float8_e4m3fn)
        scale = torch.tensor([[126], [127], [128], [129]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        weight14 = torch.arange(63, -65, -1).reshape(4, 32).to(torch.float8_e4m3fn)
        scale14 = torch.tensor([[129], [126], [128], [127]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        tensors = {
            "layers.1.engram.embed.weight": weight,
            "layers.1.engram.embed.scale": scale,
            "layers.14.engram.embed.weight": weight14,
            "layers.14.engram.embed.scale": scale14,
        }
        save_file(tensors, root / "table.safetensors")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "table.safetensors" for k in tensors}})
        )
        table = module.StagedEngramTable(root, 1, 4, 32)
        ids = torch.tensor([[3, 0, 3], [1, 2, 0]])
        w, s = table.gather_bytes(ids)
        torch.testing.assert_close(w, weight.view(torch.uint8)[ids.flatten()])
        torch.testing.assert_close(s, scale.view(torch.uint8)[ids.flatten()])
        assert table.gather_bytes(ids[:0])[0].shape == (0, 32)
        table14 = module.StagedEngramTable(root, 14, 4, 32)
        prefetcher = module.StagedEngramPrefetcher(
            tables={0: table, 1: table14}, gather_threads=2
        )
        # inline_max_tokens=0 sends every step through the worker thread.
        threaded = module.StagedEngramPrefetcher(
            tables={0: table, 1: table14}, gather_threads=2, inline_max_tokens=0
        )
        for each in (prefetcher, threaded):
            check_prefetched_rows(each)
        try:
            module.StagedEngramTable(root, 1, 5, 32)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong checkpoint dimensions accepted")
        # nvidia/DeepSeek-V4.1-Flash-NVFP4: two e2m1 values per byte, same scales.
        fp4_root = root / "fp4"
        fp4_root.mkdir()
        packed = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
        fp4_tensors = {
            "layers.1.engram.embed.weight": packed,
            "layers.1.engram.embed.scale": scale,
        }
        save_file(fp4_tensors, fp4_root / "table.safetensors")
        (fp4_root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "table.safetensors" for k in fp4_tensors}})
        )
        fp4_table = module.StagedEngramTable(fp4_root, 1, 4, 32, fp4=True)
        fp4_bytes = fp4_table.gather_bytes(ids)[0]
        torch.testing.assert_close(fp4_bytes, packed[ids.flatten()])
        try:
            module.StagedEngramTable(root, 1, 4, 32, fp4=True)
        except ValueError:
            pass
        else:
            raise AssertionError("FP8 Engram table accepted as MXFP4")
        if not args.cpu_only:
            from sglang.srt.layers.engram import dequantize_engram_rows

            fp4_expected = dequantize_engram_rows(packed, scale, fp4=True)[ids]
            torch.testing.assert_close(
                fp4_table.lookup(ids.cuda()).cpu(),
                fp4_expected.to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
            expected = (weight.float() * scale.float())[ids].to(torch.bfloat16)
            for current in (ids, ids.flip(0)):
                actual = table.lookup(current.cuda()).cpu()
                reference = expected if current is ids else expected.flip(0)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            assert table.lookup(ids[:0].cuda()).shape == (0, 3, 32)
            for each in (prefetcher, threaded):
                check_prefetched_lookup(each, ids)
            if args.split_graphs:
                from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
                    BreakableCUDAGraph,
                    BreakableCUDAGraphCapture,
                )

                static_ids = ids.cuda()

                def forward():
                    first = table.lookup((static_ids + 1) % 4)
                    second = table.lookup((static_ids + 2) % 4)
                    return first * 2 + second

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = BreakableCUDAGraph()
                with BreakableCUDAGraphCapture(graph, stream=stream):
                    output = forward()
                torch.cuda.current_stream().wait_stream(stream)
                assert len(graph._segments) == 3
                assert len(graph._break_fns) == 2
                # Change graph-produced lookup IDs on every replay. This catches
                # stale eager inputs, dangling bridges, and missing stream order.
                for current in (ids, ids.flip(0), ids.flip(1), torch.zeros_like(ids)):
                    static_ids.copy_(current)
                    graph.replay()
                    reference = (weight.float() * scale.float()).to(torch.bfloat16)
                    reference = (
                        reference[(current + 1) % 4] * 2 + reference[(current + 2) % 4]
                    )
                    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
                for each in (prefetcher, threaded):
                    check_prefetched_graph(each, ids)
                from sglang.srt.layers.logits_processor import LogitsProcessorOutput
                from sglang.srt.model_executor.runner.shape_key import ShapeKey
                from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
                    BreakableCudaGraphBackend,
                )

                backend = BreakableCudaGraphBackend(
                    SimpleNamespace(
                        device_module=torch.cuda,
                        cuda_graph_output_capacity_rows=lambda output: None,
                        cuda_graph_output_rows=lambda output: None,
                        model_runner=SimpleNamespace(
                            tp_group=SimpleNamespace(barrier=lambda: None)
                        ),
                    )
                )
                keys = [ShapeKey(size=2), ShapeKey(size=1)]
                with backend.capture_session(stream):
                    for key in keys:

                        def structured_forward(size=key.size):
                            hidden = table.lookup(static_ids[:size])
                            return LogitsProcessorOutput(
                                next_token_logits=hidden.sum(dim=1),
                                hidden_states=hidden,
                            )

                        backend.capture_one(key, structured_forward)
                for key in (keys[0], keys[1], keys[0]):
                    current = ids.flip(1) if key.size == 1 else ids.flip(0)
                    static_ids.copy_(current)
                    with backend.replay_session():
                        result = backend.replay(key, None)
                    reference = (weight.float() * scale.float())[
                        current[: key.size]
                    ].to(torch.bfloat16)
                    torch.testing.assert_close(
                        result.hidden_states.cpu(), reference, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        result.next_token_logits.cpu(),
                        reference.sum(dim=1),
                        rtol=0,
                        atol=0,
                    )
                    assert result.next_token_logprobs is None
                backend.cleanup()
        print(json.dumps({"passed": True, "cpu_only": args.cpu_only}))


if __name__ == "__main__":
    main()
