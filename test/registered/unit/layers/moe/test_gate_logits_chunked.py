import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    compute_gate_logits_chunked,
    try_fused_gate_topk_chunked,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeGate:
    """Mimics ReplicatedLinear's (output, bias) return shape."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.calls = []

    def __call__(self, x: torch.Tensor):
        self.calls.append(x.shape[0])
        return x.to(torch.float32) @ self.weight, None


class TestComputeGateLogitsChunked(unittest.TestCase):
    def test_matches_unchunked_below_cap(self):
        gate = _FakeGate(torch.randn(16, 4, dtype=torch.float32))
        hidden = torch.randn(8, 16, dtype=torch.bfloat16)
        expected, _ = gate(hidden.to(torch.float32))
        gate.calls.clear()
        got = compute_gate_logits_chunked(gate, hidden)
        self.assertTrue(torch.equal(expected, got))
        self.assertEqual(gate.calls, [8])  # single unchunked call

    def test_bit_exact_when_chunked(self):
        gate = _FakeGate(torch.randn(32, 6, dtype=torch.float32))
        hidden = torch.randn(357, 32, dtype=torch.bfloat16)
        expected, _ = gate(hidden.to(torch.float32))
        gate.calls.clear()
        with patch("sglang.srt.layers.moe.utils.GATE_LOGITS_ROW_CAP", 100):
            got = compute_gate_logits_chunked(gate, hidden)
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(torch.equal(expected, got))
        self.assertEqual(gate.calls, [100, 100, 100, 57])

    def test_never_materializes_more_than_cap_rows_at_once(self):
        gate = _FakeGate(torch.randn(8, 2, dtype=torch.float32))
        hidden = torch.randn(250, 8, dtype=torch.bfloat16)
        with patch("sglang.srt.layers.moe.utils.GATE_LOGITS_ROW_CAP", 64):
            compute_gate_logits_chunked(gate, hidden)
        self.assertTrue(all(c <= 64 for c in gate.calls))
        self.assertEqual(sum(gate.calls), 250)

    def test_exact_cap_boundary_single_call(self):
        gate = _FakeGate(torch.randn(4, 2, dtype=torch.float32))
        hidden = torch.randn(64, 4, dtype=torch.bfloat16)
        with patch("sglang.srt.layers.moe.utils.GATE_LOGITS_ROW_CAP", 64):
            compute_gate_logits_chunked(gate, hidden)
        self.assertEqual(gate.calls, [64])


class _FakeTopK:
    """Row-independent topk over logits, returning StandardTopKOutput --
    the property the fusion's chunk-concat correctness rests on."""

    def __init__(self, k: int):
        self.k = k

    def __call__(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits, self.k, dim=-1)
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32),
            router_logits=router_logits,
        )


class _CutlassBackend:
    def is_flashinfer_cutlass(self):
        return True


class _OtherBackend:
    def is_flashinfer_cutlass(self):
        return False


class TestTryFusedGateTopkChunked(unittest.TestCase):
    def _fused(self, gate, topk, hidden, backend=_CutlassBackend(), cap=100):
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend",
            return_value=backend,
        ), patch("sglang.srt.layers.moe.utils.GATE_LOGITS_ROW_CAP", cap):
            return try_fused_gate_topk_chunked(gate, topk, hidden)

    def test_bit_exact_vs_unfused(self):
        gate = _FakeGate(torch.randn(32, 12, dtype=torch.float32))
        topk = _FakeTopK(k=4)
        hidden = torch.randn(357, 32, dtype=torch.bfloat16)
        expected = topk(hidden, gate(hidden.to(torch.float32))[0])
        gate.calls.clear()
        fused = self._fused(gate, topk, hidden)
        self.assertIsNotNone(fused)
        self.assertTrue(torch.equal(expected.topk_weights, fused.topk_weights))
        self.assertTrue(torch.equal(expected.topk_ids, fused.topk_ids))
        self.assertEqual(gate.calls, [100, 100, 100, 57])

    def test_router_logits_is_empty_with_expert_dim(self):
        gate = _FakeGate(torch.randn(16, 8, dtype=torch.float32))
        fused = self._fused(gate, _FakeTopK(4), torch.randn(250, 16))
        self.assertEqual(tuple(fused.router_logits.shape), (0, 8))
        self.assertEqual(fused.router_logits.dtype, torch.float32)

    def test_small_batch_returns_none(self):
        gate = _FakeGate(torch.randn(16, 8, dtype=torch.float32))
        fused = self._fused(gate, _FakeTopK(4), torch.randn(100, 16), cap=100)
        self.assertIsNone(fused)
        self.assertEqual(gate.calls, [])  # no wasted work before the bail

    def test_non_whitelisted_backend_returns_none(self):
        gate = _FakeGate(torch.randn(16, 8, dtype=torch.float32))
        fused = self._fused(
            gate, _FakeTopK(4), torch.randn(250, 16), backend=_OtherBackend()
        )
        self.assertIsNone(fused)
        self.assertEqual(gate.calls, [])

    def test_uninitialized_flags_returns_none(self):
        gate = _FakeGate(torch.randn(16, 8, dtype=torch.float32))
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend",
            side_effect=RuntimeError("flags not initialized"),
        ), patch("sglang.srt.layers.moe.utils.GATE_LOGITS_ROW_CAP", 100):
            fused = try_fused_gate_topk_chunked(
                gate, _FakeTopK(4), torch.randn(250, 16)
            )
        self.assertIsNone(fused)

    def test_non_standard_topk_output_returns_none(self):
        gate = _FakeGate(torch.randn(16, 8, dtype=torch.float32))

        def weird_topk(hidden_states, router_logits):
            return (router_logits.topk(4).values, None, router_logits)  # plain tuple

        fused = self._fused(gate, weird_topk, torch.randn(250, 16))
        self.assertIsNone(fused)


if __name__ == "__main__":
    unittest.main()
