"""Tests for the breakable CUDA graph (BCG) runner.

Two test classes:
- TestBreakableCUDAGraphBasic / TestCopyOutput / TestBreakGraphHelper:
  unit tests for the core capture / replay mechanism (simple tensor ops).
- TestBreakableCudaGraph: integration test — spin up Qwen3-8B with
  --cuda-graph-backend-prefill=breakable and check mgsm_en accuracy.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.runner_utils.forward_batch import (
    set_forward_batch,
)
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    prefill_graph_scope,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration — large suite to fit the integration test's server startup.
register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-c-test-large-8-gpu-amd-mi35x")


class TestBreakableCUDAGraphBasic(CustomTestCase):
    """Test basic breakable CUDA graph capture and replay."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.eager_on_graph = staticmethod(eager_on_graph)
        cls.device = torch.device("cuda:0")

    def test_qsa_indexer_uses_live_lengths(self):
        from sglang.srt.models import qwen4_exp

        context = SimpleNamespace(forward_batch=SimpleNamespace(rows=8))
        layer = SimpleNamespace(_qsa_prefill_topk_bridge=None)
        layer._compute_qsa_topk_indices_eager = (
            lambda hidden_states, forward_batch, **kw: hidden_states[
                : forward_batch.rows
            ]
        )
        x = torch.zeros((8, 2), dtype=torch.int32, device=self.device)
        inputs = dict(layer=layer, positions=None)
        graph = self.BreakableCUDAGraph()
        with patch.object(
            qwen4_exp, "get_tc_piecewise_forward_context", return_value=context
        ):
            with self.BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
                result = qwen4_exp._breakable_qsa_indexer(hidden_states=x + 1, **inputs)
            for n in (3, 8):
                context.forward_batch = SimpleNamespace(rows=n)
                x.fill_(2)
                graph.replay()
                self.assertEqual(result.tolist(), [[3, 3]] * n + [[0, 0]] * (8 - n))

    def test_no_break_capture_replay(self):
        """Capture and replay without any graph breaks should work like normal CUDA graph."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            y.copy_(x + 1.0)

        # Replay with new input
        x.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 6.0, device=self.device)))

    def test_single_break(self):
        """A single graph break should split capture into two segments."""
        x = torch.zeros(4, device=self.device)
        intermediate = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def eager_op(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            intermediate.copy_(x + 1.0)
            broken = eager_op(intermediate)
            y.copy_(broken + 3.0)

        # Replay with new input
        x.fill_(10.0)
        graph.replay()
        torch.cuda.synchronize()
        # x=10 -> intermediate=11 -> eager: 11*2=22 -> y=22+3=25
        self.assertTrue(torch.allclose(y, torch.full((4,), 25.0, device=self.device)))

    def test_multiple_breaks(self):
        """Multiple graph breaks should produce correct chained results."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def add_one(src):
            return src + 1.0

        @self.eager_on_graph(enable=True)
        def double(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t1 = x + 1.0  # graph segment 1
            t2 = add_one(t1)  # break 1: eager
            t3 = t2 + 1.0  # graph segment 2
            t4 = double(t3)  # break 2: eager
            y.copy_(t4)  # graph segment 3

        # Replay: x=5 -> +1=6 -> add_one=7 -> +1=8 -> double=16
        x.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 16.0, device=self.device)))

    def test_eager_on_graph_disabled(self):
        """@eager_on_graph(enable=False) should be a no-op passthrough."""

        @self.eager_on_graph(enable=False)
        def my_fn(x):
            return x + 1.0

        # Should just be the original function
        t = torch.tensor([1.0, 2.0], device=self.device)
        result = my_fn(t)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_eager_on_graph_outside_capture(self):
        """@eager_on_graph called outside capture should run the function directly."""

        @self.eager_on_graph(enable=True)
        def my_fn(x):
            return x + 1.0

        t = torch.tensor([1.0, 2.0], device=self.device)
        result = my_fn(t)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_replay_updates_output(self):
        """Replay should produce different results when input buffers change."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def scale(src):
            return src * 3.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            t2 = scale(t)
            y.copy_(t2)

        # First replay: x=0 -> 0+1=1 -> 1*3=3
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 3.0, device=self.device)))

        # Second replay: x=10 -> 10+1=11 -> 11*3=33
        x.fill_(10.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 33.0, device=self.device)))

    def test_side_stream_join_across_break(self):
        """A side-stream producer may be joined after a graph break."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)
        stream = torch.cuda.Stream(self.device)

        @self.eager_on_graph(enable=True)
        def identity(src):
            return src

        graph = self.BreakableCUDAGraph()
        capture_stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=capture_stream):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                side_output = x + 1.0
            identity(x)
            torch.cuda.current_stream().wait_stream(stream)
            y.copy_(side_output * 2.0)

        x.fill_(5.0)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 12.0, device=self.device)))

    def test_eager_output_is_held_strongly_for_replay_bridge(self):
        """The replay closure must keep the eager output bridge buffer alive."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def scale(src):
            return src * 3.0

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            broken = scale(t)
            y.copy_(broken)

        replay_closure = graph._break_fns[0].__closure__ or ()
        self.assertTrue(
            any(cell.cell_contents is broken for cell in replay_closure),
            "eager output bridge buffer must be strongly captured",
        )

    def test_attention_narrows_padded_positions(self):

        num_tokens = 3
        padded_num_tokens = 5
        forward_batch = SimpleNamespace(
            global_num_token_non_padded_cpu=num_tokens,
            mha_return_lse=False,
            _attn_output=None,
            out_cache_loc=torch.arange(padded_num_tokens, device=self.device),
            positions=torch.arange(padded_num_tokens, device=self.device),
        )
        observed = {}

        def attention_forward(query, key, value, layer, batch, save_kv_cache):
            observed["positions"] = batch.positions.clone()
            observed["out_cache_loc"] = batch.out_cache_loc.clone()
            return torch.ones_like(query)

        output = torch.full((padded_num_tokens, 2), float("nan"), device=self.device)
        with (
            set_forward_batch(forward_batch),
            prefill_graph_scope(full_graph=False, raw_num_tokens=num_tokens),
            patch(
                "sglang.srt.layers.radix_attention.get_attn_backend",
                return_value=SimpleNamespace(forward=attention_forward),
            ),
        ):
            RadixAttention(1, 2, 1.0, 1, 0)._eager_attention(
                torch.zeros((padded_num_tokens, 2), device=self.device),
                torch.zeros((padded_num_tokens, 1, 2), device=self.device),
                torch.zeros((padded_num_tokens, 1, 2), device=self.device),
                output,
                True,
            )

        expected = torch.arange(num_tokens, device=self.device)
        torch.testing.assert_close(observed["positions"], expected)
        torch.testing.assert_close(observed["out_cache_loc"], expected)
        self.assertEqual(forward_batch.positions.shape[0], padded_num_tokens)
        self.assertEqual(forward_batch.out_cache_loc.shape[0], padded_num_tokens)


class TestCopyOutput(CustomTestCase):
    """Test the _copy_output helper for structured output writeback."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            _copy_output,
        )

        cls._copy_output = staticmethod(_copy_output)
        cls.device = torch.device("cuda:0")

    def test_tensor_copy(self):
        dst = torch.zeros(4, device=self.device)
        src = torch.ones(4, device=self.device) * 5.0
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(torch.allclose(dst, src))

    def test_dict_copy(self):
        dst = {
            "a": torch.zeros(4, device=self.device),
            "b": torch.zeros(4, device=self.device),
        }
        src = {
            "a": torch.ones(4, device=self.device),
            "b": torch.ones(4, device=self.device) * 2.0,
        }
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(torch.allclose(dst["a"], torch.ones(4, device=self.device)))
        self.assertTrue(
            torch.allclose(dst["b"], torch.ones(4, device=self.device) * 2.0)
        )

    def test_object_copy(self):
        class FakeOutput:
            def __init__(self, t, label):
                self.tensor = t
                self.label = label

        dst = FakeOutput(torch.zeros(4, device=self.device), "old")
        src = FakeOutput(torch.ones(4, device=self.device) * 3.0, "new")
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(
            torch.allclose(dst.tensor, torch.ones(4, device=self.device) * 3.0)
        )
        self.assertEqual(dst.label, "new")

    def test_non_tensor_fallback(self):
        result = self._copy_output(42, 99)
        self.assertEqual(result, 99)


class TestBreakGraphHelper(CustomTestCase):
    """Test the break_graph() convenience function."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            break_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.break_graph = staticmethod(break_graph)
        cls.device = torch.device("cuda:0")

    def test_break_graph_inserts_segment(self):
        """break_graph() should insert a graph break even though it does nothing."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        graph = self.BreakableCUDAGraph()
        stream = torch.cuda.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            self.break_graph()
            y.copy_(t + 2.0)

        x.fill_(10.0)
        graph.replay()
        torch.cuda.synchronize()
        # x=10 -> +1=11 -> break -> +2=13
        self.assertTrue(torch.allclose(y, torch.full((4,), 13.0, device=self.device)))


class TestBreakableCudaGraphBackendLogitsOutput(CustomTestCase):
    """BreakableCudaGraphBackend's output-buffer helpers must support
    LogitsProcessorOutput -- this is what capture_one() actually receives
    from a model forward, and the four dispatch methods previously raised
    TypeError("Unsupported BCG output type") for it (only torch.Tensor,
    PPProxyTensors, tuple, and list were handled). Regression test for a
    live crash: DSPARK decode/verify graph capture under
    --cuda-graph-backend-decode breakable hit this on the very first
    captured shape."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )

        cls.LogitsProcessorOutput = LogitsProcessorOutput
        # capture_one()'s dispatch methods are pure functions of self._logits_
        # output_tensor_field_names / recursion -- no other instance state is
        # touched, so a bare, un-initialized instance is a faithful unit under
        # test without needing a real model_runner/cuda_graph_runner.
        cls.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        from types import SimpleNamespace
        cls.backend._cuda_graph_runner = SimpleNamespace(
            cuda_graph_output_rows=lambda output: None,
            cuda_graph_output_capacity_rows=lambda output: None,
        )
        cls.device = torch.device("cuda:0")

    def test_output_rows_ignores_non_tensor_fields(self):
        out = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 8, device=self.device),
            hidden_states=torch.randn(3, 4, device=self.device),
        )
        self.assertEqual(self.backend._output_rows(out, cap=8), 3)

    def test_alloc_full_buffer_sizes_tensor_fields_only(self):
        out = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 8, device=self.device),
            hidden_states=torch.randn(3, 4, device=self.device),
        )
        buf = self.backend._alloc_full_buffer(out, size=8)
        self.assertEqual(tuple(buf.next_token_logits.shape), (8, 8))
        self.assertEqual(tuple(buf.hidden_states.shape), (8, 4))
        self.assertIsNone(buf.next_token_logprobs)
        self.assertIsNone(buf.customized_info)

    def test_copy_and_slice_round_trip(self):
        out = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 8, device=self.device),
            hidden_states=torch.randn(3, 4, device=self.device),
        )
        buf = self.backend._alloc_full_buffer(out, size=8)
        self.backend._copy_output_to_buffer(out, buf, num_tokens=3)
        stored = self.backend._slice_output(buf, num_tokens=3)
        self.assertTrue(torch.allclose(stored.next_token_logits, out.next_token_logits))
        self.assertTrue(torch.allclose(stored.hidden_states, out.hidden_states))

    def test_non_tensor_fields_carried_through_unchanged(self):
        out = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(2, 8, device=self.device),
            customized_info={"foo": [1, 2, 3]},
        )
        buf = self.backend._alloc_full_buffer(out, size=8)
        self.assertEqual(buf.customized_info, {"foo": [1, 2, 3]})
        self.backend._copy_output_to_buffer(out, buf, num_tokens=2)
        stored = self.backend._slice_output(buf, num_tokens=2)
        self.assertEqual(stored.customized_info, {"foo": [1, 2, 3]})

    def test_copy_raises_on_tensor_field_mismatch_across_shapes(self):
        """A later-captured shape whose LogitsProcessorOutput has a
        different set of live tensor fields than the shared buffer must
        raise, not silently corrupt the buffer -- this is the same
        cross-shape structural-consistency contract PPProxyTensors and
        list/tuple outputs already enforce below."""
        buf = self.backend._alloc_full_buffer(
            self.LogitsProcessorOutput(
                next_token_logits=torch.randn(2, 8, device=self.device)
            ),
            size=8,
        )
        mismatched = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(2, 8, device=self.device),
            hidden_states=torch.randn(2, 4, device=self.device),
        )
        with self.assertRaises(ValueError):
            self.backend._copy_output_to_buffer(mismatched, buf, num_tokens=2)

    def test_buffer_rows_come_from_output_not_shape_key(self):
        """Regression: the decode runner keys shapes by REQUEST count while
        TARGET_VERIFY emits bs * num_draft_tokens output rows; sizing the
        shared buffer off shape_key.size truncated verify outputs to bs rows
        ("next_token_logits row count mismatch... Expected 9, got 1",
        2026-08-25 live incident). Buffer capacity must come from the output
        tensors themselves."""
        backend = type(self.backend).__new__(type(self.backend))
        backend._shared_output_buffer = None
        backend._shared_output_rows = 0
        backend._cuda_graph_runner = self.backend._cuda_graph_runner
        verify_out = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(18, 8, device=self.device),
            hidden_states=torch.randn(18, 4, device=self.device),
        )
        rows = backend._ensure_shared_output_buffer(verify_out)
        self.assertEqual(rows, 18)
        self.assertEqual(
            tuple(backend._shared_output_buffer.next_token_logits.shape), (18, 8)
        )
        # A later, smaller shape reuses the buffer without reallocating.
        smaller = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(9, 8, device=self.device),
            hidden_states=torch.randn(9, 4, device=self.device),
        )
        buffer_before = backend._shared_output_buffer
        self.assertEqual(backend._ensure_shared_output_buffer(smaller), 18)
        self.assertIs(backend._shared_output_buffer, buffer_before)
        self.assertEqual(backend._output_rows(smaller, 18), 9)
        # A later LARGER shape violates the capture-order contract loudly.
        larger = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(27, 8, device=self.device)
        )
        with self.assertRaises(RuntimeError):
            backend._ensure_shared_output_buffer(larger)

    def test_max_output_rows_takes_max_across_tensor_fields(self):
        out = self.LogitsProcessorOutput(
            next_token_logits=torch.randn(18, 8, device=self.device),
            hidden_states=torch.randn(9, 4, device=self.device),
        )
        self.assertEqual(type(self.backend)._max_output_rows(out), 18)
        self.assertEqual(
            type(self.backend)._max_output_rows(
                [out, torch.randn(20, 2, device=self.device), None]
            ),
            20,
        )
        self.assertEqual(type(self.backend)._max_output_rows(None), 0)


class TestBreakableCudaGraph(CustomTestCase):
    """Integration: Qwen3-8B with --cuda-graph-backend-prefill=breakable on mgsm_en."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-backend-prefill=breakable",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=200,
            num_threads=1024,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"mgsm_en accuracy with breakable CUDA graph: {score:.3f}")

        self.assertGreaterEqual(score, 0.80)


if __name__ == "__main__":
    unittest.main()
