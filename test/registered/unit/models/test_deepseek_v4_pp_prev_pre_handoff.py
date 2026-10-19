"""Regression test for the Single-Pass mHC `prev_pre` hand-off across a
Pipeline-Parallel stage boundary (`DeepseekV4Model._forward_layers_hc_pre_from_prev`).

Before the fix, `prev_pre` was hard-initialized to `None` on every call
regardless of an incoming value from a previous PP stage -- splitting a layer
range into two calls (as PP does) silently produced different results than
running the same layers in one call. This test builds a minimal fake layer
stack (no real weights, no distributed group, no GPU required) with a
deterministic `forward_hc_pre_from_prev` that genuinely depends on the
incoming `prev_pre`, and checks that splitting the call in two, threading the
first call's returned `prev_pre` into the second as `prev_pre_in`, exactly
reproduces the one-call reference -- and that the old (pre-fix) behavior of
starting the second call with `prev_pre_in=None` would NOT reproduce it, so
this test would have caught the original bug.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC_MULT = 4
HIDDEN = 8
NUM_LAYERS = 6
SPLIT_AT = 3


class _FakeLayer:
    """Deterministic stand-in for DeepseekV4DecoderLayer: `forward_hc_pre_from_prev`
    genuinely mixes the incoming pre-mix into both its outputs, so a wrong
    (None) pre-mix at a split point changes the numeric result -- this is
    what makes the test able to fail against the pre-fix code."""

    engram = None
    hc_cfg = None

    def forward_hc_pre_from_prev(
        self,
        *,
        positions,
        state,
        input_ids,
        forward_batch,
        input_ids_global,
        **_cross_layer_fusion,
    ):
        hidden_states, prev_pre = state.residual, state.pre
        if prev_pre is None:
            prev_pre = torch.zeros(hidden_states.shape[0], HC_MULT)
        new_pre = prev_pre + hidden_states.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        bump = new_pre.mean(dim=-1).view(-1, 1, 1)
        return HcState(hidden_states + bump, new_pre)


def _bare_model(start_layer: int, end_layer: int) -> DeepseekV4Model:
    """Mirrors the `object.__new__(...)` + manual-attribute pattern already
    used elsewhere in this test suite (e.g. dsv4 backend tests' `_backend()`
    helper) to exercise one real method without constructing the whole model."""
    m = object.__new__(DeepseekV4Model)
    m.start_layer = start_layer
    m.end_layer = end_layer
    m.layers = [_FakeLayer() for _ in range(NUM_LAYERS)]
    # Not deepseek_v41, so the V4.1-only cross-layer fusions stay off.
    m.config = SimpleNamespace(model_type="deepseek_v4", vision_n_layers=0)
    m.late_layer_start = None
    m.engram_hasher = None
    m.engram_prefetcher = None
    m.dspark_layers_to_capture = None
    m.pp_group = SimpleNamespace(world_size=2)
    return m


class TestPrevPreHandoff(CustomTestCase):
    def _run(self, model: DeepseekV4Model, hidden_states, prev_pre_in):
        return model._forward_layers_hc_pre_from_prev(
            positions=None,
            hidden_states=hidden_states,
            forward_batch=None,
            input_ids=None,
            input_ids_global=None,
            capture_dspark=False,
            dspark_aux_hidden_states=[],
            prev_pre_in=prev_pre_in,
        )

    def test_split_call_matches_single_call_reference(self):
        torch.manual_seed(0)
        hidden = torch.randn(5, HC_MULT, HIDDEN)

        reference_model = _bare_model(0, NUM_LAYERS)
        ref_hidden, ref_pre, ref_tail = self._run(reference_model, hidden.clone(), None)
        self.assertIsNone(ref_tail)

        stage0 = _bare_model(0, SPLIT_AT)
        mid_hidden, mid_pre, _ = self._run(stage0, hidden.clone(), None)

        stage1 = _bare_model(SPLIT_AT, NUM_LAYERS)
        split_hidden, split_pre, _ = self._run(stage1, mid_hidden, mid_pre)

        torch.testing.assert_close(split_hidden, ref_hidden)
        torch.testing.assert_close(split_pre, ref_pre)

    def test_dropping_the_handoff_reproduces_the_original_bug(self):
        """Proves this test is sensitive to the bug: re-running stage 1 with
        prev_pre_in=None (the pre-fix hard-coded behavior) must NOT match the
        single-call reference -- otherwise this test couldn't have caught it."""
        torch.manual_seed(0)
        hidden = torch.randn(5, HC_MULT, HIDDEN)

        reference_model = _bare_model(0, NUM_LAYERS)
        ref_hidden, _, _ = self._run(reference_model, hidden.clone(), None)

        stage0 = _bare_model(0, SPLIT_AT)
        mid_hidden, _mid_pre, _ = self._run(stage0, hidden.clone(), None)

        stage1_buggy = _bare_model(SPLIT_AT, NUM_LAYERS)
        buggy_hidden, _, _ = self._run(stage1_buggy, mid_hidden, None)

        self.assertFalse(torch.allclose(buggy_hidden, ref_hidden))

    def test_pp_size_one_default_is_unaffected(self):
        """prev_pre_in defaults to None, so the ordinary (no-PP) call site,
        which never passes it, is byte-for-byte unchanged by the fix."""
        torch.manual_seed(1)
        hidden = torch.randn(3, HC_MULT, HIDDEN)

        model_default = _bare_model(0, NUM_LAYERS)
        default_hidden, default_pre, _ = model_default._forward_layers_hc_pre_from_prev(
            positions=None,
            hidden_states=hidden.clone(),
            forward_batch=None,
            input_ids=None,
            input_ids_global=None,
            capture_dspark=False,
            dspark_aux_hidden_states=[],
        )

        model_explicit = _bare_model(0, NUM_LAYERS)
        explicit_hidden, explicit_pre, _ = self._run(
            model_explicit, hidden.clone(), None
        )

        torch.testing.assert_close(default_hidden, explicit_hidden)
        torch.testing.assert_close(default_pre, explicit_pre)


if __name__ == "__main__":
    unittest.main()
