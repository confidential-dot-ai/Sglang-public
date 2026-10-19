"""Stage-boundary tests for DeepSeek-V4 decode/verify TBO."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.batch_overlap import operations_strategy as ops_strategy
from sglang.srt.batch_overlap.operations import _convert_operations_to_stages
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayerBase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StubA2ABackend:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self) -> bool:
        return self._is_none


class _StubAttention:
    def op_attn(self, state):
        pass


class _StubMlp:
    def op_gate(self, state):
        pass

    def op_select_experts(self, state):
        pass

    def op_dispatch_a(self, state):
        pass

    def op_dispatch_b(self, state):
        pass

    def op_experts(self, state):
        pass

    def op_combine_a(self, state):
        pass

    def op_shared_experts(self, state):
        pass

    def op_combine_b(self, state):
        pass

    def op_output(self, state):
        pass


class _StubDeepseekV4Layer(DeepseekV4DecoderLayerBase):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.self_attn = _StubAttention()
        self.mlp = _StubMlp()

    def op_mhc_prepare_attn(self, state, **kwargs):
        pass

    def op_mhc_post_attn_pre_mlp(self, state):
        pass

    def op_gather_a(self, state):
        pass

    def op_gather_b(self, state):
        pass

    def op_moe(self, state):
        pass

    def op_combine_a(self, state):
        pass

    def op_combine_b(self, state):
        pass

    def op_mhc_postprocess(self, state):
        pass


def _build(mode=ForwardMode.EXTEND, *, ep=False):
    with patch(
        "sglang.srt.layers.moe.get_moe_a2a_backend",
        lambda: _StubA2ABackend(is_none=not ep),
    ):
        return ops_strategy.OperationsStrategy.init_new_tbo(
            [_StubDeepseekV4Layer()], mode
        )


def _stage_names(strategy):
    return [
        [op.debug_name for op in stage]
        for stage in _convert_operations_to_stages(strategy.operations)
    ]


class TestDeepseekV4OperationsStrategy(CustomTestCase):
    def test_prefill_retains_unstaggered_schedule(self):
        strategy = _build(ForwardMode.EXTEND)
        self.assertEqual(strategy.tbo_delta_stages, 0)
        stages = _stage_names(strategy)
        self.assertIn("mhc_prepare_attn", stages[0])
        self.assertIn("attn", stages[0])

    def test_decode_and_verify_use_dsv4_stage_boundaries(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                strategy = _build(mode)
                self.assertEqual(strategy.tbo_delta_stages, 2)
                stages = _stage_names(strategy)

                def stage_of(name):
                    return next(i for i, stage in enumerate(stages) if name in stage)

                # DSV4 attention is monolithic. Its safe stagger boundary is
                # after mHC input preparation, not inside attention.
                self.assertLess(stage_of("mhc_prepare_attn"), stage_of("attn"))
                self.assertLess(stage_of("gather_a"), stage_of("gather_b"))
                self.assertLess(stage_of("combine_a"), stage_of("combine_b"))

    def test_ep_decode_and_verify_keep_dispatch_combine_splits(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                stages = _stage_names(_build(mode, ep=True))

                def stage_of(name):
                    return next(i for i, stage in enumerate(stages) if name in stage)

                self.assertLess(stage_of("dispatch_a"), stage_of("dispatch_b"))
                self.assertLess(stage_of("combine_a"), stage_of("combine_b"))

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            _build(ForwardMode.IDLE)


if __name__ == "__main__":
    unittest.main()
