"""Admission tests for DeepSeek-V4 decode/verify TBO."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4Model

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StubForwardBatch:
    def __init__(self, mode: ForwardMode, *, can_run=True, children=True):
        self.can_run_tbo = can_run
        self.global_forward_mode = mode
        self.tbo_children = [object(), object()] if children else None


def _bare_model(pp_size=1):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    model._tbo_compiled_shapes = set()
    model.hc_pre_from_prev_sublayer = False
    model.pp_group = SimpleNamespace(world_size=pp_size)
    return model


def _check(
    model, forward_batch, *, capture, signature=(1, 8, 1, 1024), is_hip=False, cp_active=False
):
    with patch(
        "sglang.srt.models.deepseek_v4.get_is_capture_mode", lambda: capture
    ), patch(
        "sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", lambda fb: cp_active
    ), patch(
        "sglang.srt.models.deepseek_v4._is_hip", is_hip
    ), patch(
        "sglang.srt.models.deepseek_v4.get_parallel",
        lambda: SimpleNamespace(attn_dp_size=2),
    ), patch(
        "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
        lambda: SimpleNamespace(is_none=lambda: True),
    ), patch(
        "sglang.srt.layers.moe.is_tbo_enabled", lambda: True
    ), patch.object(
        DeepseekV4Model, "_tbo_shape_signature", staticmethod(lambda fb: signature)
    ), patch.object(
        DeepseekV4Model,
        "_all_gather_tbo_signature",
        staticmethod(lambda sig, fb: sig),
    ):
        return model._can_run_tbo(forward_batch)


class TestDeepseekV4TboCanRun(CustomTestCase):
    def test_decode_and_one_token_verify_have_distinct_compile_keys(self):
        common = dict(
            batch_size=8,
            extend_seq_lens_cpu=None,
            seq_lens_cpu=torch.tensor([1024] * 8),
        )
        decode = SimpleNamespace(
            **common,
            forward_mode=ForwardMode.DECODE,
            global_forward_mode=ForwardMode.DECODE,
            spec_info=None,
        )
        verify = SimpleNamespace(
            **common,
            forward_mode=ForwardMode.TARGET_VERIFY,
            global_forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=1),
        )
        self.assertNotEqual(
            DeepseekV4Model._tbo_shape_signature(decode),
            DeepseekV4Model._tbo_shape_signature(verify),
        )

    def test_cp_and_alternate_mhc_scheme_decline_before_shape_latch(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            for capture in (False, True):
                with self.subTest(mode=mode, capture=capture):
                    model = _bare_model()
                    fb = _StubForwardBatch(mode)
                    self.assertFalse(_check(model, fb, capture=capture, cp_active=True))
                    model.hc_pre_from_prev_sublayer = True
                    self.assertFalse(_check(model, fb, capture=capture))
                    self.assertEqual(model._tbo_compiled_shapes, set())

    def test_prefill_capture_behavior_is_unchanged(self):
        model = _bare_model()
        fb = _StubForwardBatch(ForwardMode.EXTEND)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                with envs.SGLANG_ENABLE_DEEPSEEK_V4_TBO_CUDA_GRAPH.override(enabled):
                    self.assertTrue(_check(model, fb, capture=True))

    def test_decode_and_verify_graph_tbo_require_coordinated_opt_in(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                model = _bare_model()
                fb = _StubForwardBatch(mode)
                with envs.SGLANG_ENABLE_DEEPSEEK_V4_TBO_CUDA_GRAPH.override(False):
                    self.assertFalse(_check(model, fb, capture=True))
                with envs.SGLANG_ENABLE_DEEPSEEK_V4_TBO_CUDA_GRAPH.override(True):
                    self.assertTrue(_check(model, fb, capture=True))

    def test_hip_graph_tbo_stays_disabled(self):
        # DeepseekV4HipRadixBackend.tbo_supports_cuda_graph is permanently
        # false, so the model body must not take a TBO capture path there.
        model = _bare_model()
        fb = _StubForwardBatch(ForwardMode.DECODE)
        with envs.SGLANG_ENABLE_DEEPSEEK_V4_TBO_CUDA_GRAPH.override(True):
            self.assertFalse(_check(model, fb, capture=True, is_hip=True))

    def test_eager_decode_and_verify_compile_before_overlap(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                model = _bare_model()
                fb = _StubForwardBatch(mode)
                self.assertFalse(_check(model, fb, capture=False))
                self.assertTrue(_check(model, fb, capture=False))

    def test_latch_is_shape_specific(self):
        model = _bare_model()
        fb = _StubForwardBatch(ForwardMode.DECODE)
        self.assertFalse(
            _check(model, fb, capture=False, signature=(1, 8, 1, 1024))
        )
        self.assertTrue(_check(model, fb, capture=False, signature=(1, 8, 1, 1024)))
        self.assertFalse(
            _check(model, fb, capture=False, signature=(1, 16, 1, 1024))
        )

    def test_missing_children_or_scheduler_permission_declines(self):
        model = _bare_model()
        for fb in (
            _StubForwardBatch(ForwardMode.DECODE, children=False),
            _StubForwardBatch(ForwardMode.DECODE, can_run=False),
        ):
            self.assertFalse(_check(model, fb, capture=False))

    def test_pipeline_parallel_and_unsupported_modes_decline(self):
        self.assertFalse(
            _check(
                _bare_model(pp_size=2),
                _StubForwardBatch(ForwardMode.DECODE),
                capture=False,
            )
        )
        self.assertFalse(
            _check(
                _bare_model(),
                _StubForwardBatch(ForwardMode.IDLE),
                capture=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
