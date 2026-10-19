"""Backend opt-in flags for the ragged-verify graphs.

Runs in the GPU suite because importing the backend modules pulls GPU-only
wheels (sgl_kernel) at module scope, which fail to import on CPU runners.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

    def test_ragged_implementing_backends_declare_the_flag(self):
        """Every backend with a ragged-verify metadata path must opt in; a
        dropped flag silently disables ragged graphs for that backend (the
        runner falls back to eager with no other test going red)."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        for backend in (
            TRTLLMHAAttnBackend,
            DeepseekV4AttnBackend,
            FlashAttentionBackend,
        ):
            with self.subTest(backend=backend.__name__):
                self.assertTrue(backend.supports_ragged_verify_graph)

    def test_flashinfer_and_tbo_capabilities_compose(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
            WrapperDispatch,
        )
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        flashinfer = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        flashinfer.dispatch_reason = None
        self.assertTrue(flashinfer.supports_ragged_verify_graph)
        flashinfer.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        self.assertFalse(flashinfer.supports_ragged_verify_graph)

        primary = SimpleNamespace(supports_ragged_verify_graph=False)
        tbo = TboAttnBackend.__new__(TboAttnBackend)
        tbo.primary = primary
        self.assertFalse(tbo.supports_ragged_verify_graph)
        primary.supports_ragged_verify_graph = True
        self.assertTrue(tbo.supports_ragged_verify_graph)

    def test_flashinfer_compact_verify_key_is_token_tier(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        static = SimpleNamespace(ragged_verify_layout=None)
        compact_18 = SimpleNamespace(
            ragged_verify_layout=SimpleNamespace(graph_num_tokens=18)
        )
        compact_27 = SimpleNamespace(
            ragged_verify_layout=SimpleNamespace(graph_num_tokens=27)
        )
        self.assertEqual(FlashInferAttnBackend._verify_graph_key(4, static), 4)
        self.assertEqual(
            FlashInferAttnBackend._verify_graph_key(4, compact_18),
            ("ragged", 18),
        )
        self.assertEqual(
            FlashInferAttnBackend._verify_graph_key(7, compact_18),
            ("ragged", 18),
        )
        self.assertNotEqual(
            FlashInferAttnBackend._verify_graph_key(4, compact_18),
            FlashInferAttnBackend._verify_graph_key(4, compact_27),
        )

    def test_flashinfer_compact_host_plan_matches_capped_device_layout(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            _ragged_verify_host_plan_metadata,
        )

        layout = SimpleNamespace(
            bs=2,
            graph_num_tokens=18,
            verify_lens_cpu=[3, 5],
        )
        spec_info = SimpleNamespace(
            ragged_verify_layout=layout,
            draft_token_num=9,
            live_seq_lens_cpu=torch.tensor([100, 200], dtype=torch.int64),
        )
        qo, q_lens, kv_lens = _ragged_verify_host_plan_metadata(
            spec_info=spec_info,
            seq_lens_cpu=torch.tensor([103, 205, 1, 1], dtype=torch.int64),
            padded_bs=4,
        )
        self.assertEqual(q_lens.tolist(), [3, 5, 5, 5])
        self.assertEqual(qo.tolist(), [0, 3, 8, 13, 18])
        self.assertEqual(kv_lens.tolist(), [103, 205, 6, 6])

        # With no padding slot the leftover is assigned to the last row, then
        # capped. The remaining token-tier rows are deliberate ghosts.
        qo, q_lens, kv_lens = _ragged_verify_host_plan_metadata(
            spec_info=spec_info,
            seq_lens_cpu=torch.tensor([103, 205], dtype=torch.int64),
            padded_bs=2,
        )
        self.assertEqual(q_lens.tolist(), [3, 9])
        self.assertEqual(qo.tolist(), [0, 3, 12])
        self.assertEqual(kv_lens.tolist(), [103, 209])

        # A later seqlen-tier capture reuses the already-fast wrapper. Its
        # synthetic spec has no live prefix mirror and capture seq_lens_cpu is
        # committed-only, so Q must be added rather than subtracted.
        synthetic_spec = SimpleNamespace(
            ragged_verify_layout=SimpleNamespace(
                bs=2,
                graph_num_tokens=18,
                verify_lens_cpu=[9, 9],
            ),
            draft_token_num=9,
            draft_token=None,
            live_seq_lens_cpu=None,
        )
        qo, q_lens, kv_lens = _ragged_verify_host_plan_metadata(
            spec_info=synthetic_spec,
            seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int64),
            padded_bs=2,
        )
        self.assertEqual(qo.tolist(), [0, 9, 18])
        self.assertEqual(q_lens.tolist(), [9, 9])
        self.assertEqual(kv_lens.tolist(), [10, 10])

    def test_flashinfer_static_dflash_reused_capture_adds_verify_width(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            _static_verify_host_kv_lens,
        )
        from sglang.srt.speculative.spec_info import SpecInputType

        synthetic_spec = SimpleNamespace(
            spec_input_type=SpecInputType.DFLASH_VERIFY,
            draft_token=None,
            live_seq_lens_cpu=None,
        )
        actual = _static_verify_host_kv_lens(
            spec_info=synthetic_spec,
            seq_lens_cpu_i32=torch.tensor([1, 1], dtype=torch.int32),
            num_tokens_per_req=9,
        )
        self.assertEqual(actual.tolist(), [10, 10])

        live_spec = SimpleNamespace(
            spec_input_type=SpecInputType.DFLASH_VERIFY,
            draft_token=torch.empty(18, dtype=torch.int64),
            live_seq_lens_cpu=torch.tensor([100, 200], dtype=torch.int64),
        )
        actual = _static_verify_host_kv_lens(
            spec_info=live_spec,
            seq_lens_cpu_i32=torch.tensor([109, 209], dtype=torch.int32),
            num_tokens_per_req=9,
        )
        self.assertEqual(actual.tolist(), [109, 209])

    def test_flashinfer_compact_ghost_rows_are_zeroed(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            _zero_ragged_verify_ghost_rows,
        )

        output = torch.arange(10, dtype=torch.float32).view(5, 2)
        output[3:].fill_(float("nan"))
        qo_indptr = torch.tensor([0, 2, 3], dtype=torch.int32)
        actual = _zero_ragged_verify_ghost_rows(output, qo_indptr)
        self.assertTrue(torch.equal(actual[:3], output[:3]))
        self.assertTrue(torch.equal(actual[3:], torch.zeros_like(actual[3:])))

    def test_dflash_flashinfer_geometry_caps_unassigned_token_tail(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[3, 5],
            device=torch.device("cpu"),
            grid=[18],
        )
        spec_info = DFlashVerifyInput(
            draft_token=torch.zeros(18, dtype=torch.int64),
            positions=torch.zeros(18, dtype=torch.int64),
            draft_token_num=9,
            ragged_verify_layout=layout,
        )
        translator = mock.MagicMock()
        plan = object()
        kv_indices, kv_indptr, qo_indptr, mask = spec_info.generate_attn_arg_prefill(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            paged_kernel_lens=torch.tensor([100, 200], dtype=torch.int32),
            paged_kernel_lens_sum=300,
            translator=translator,
            plan=plan,
        )

        self.assertEqual(qo_indptr.tolist(), [0, 3, 12])
        self.assertEqual(kv_indptr.tolist(), [0, 103, 312])
        self.assertEqual(kv_indices.numel(), 318)
        self.assertIsNone(mask)
        # One pack over both rows, each widened by its capped verify length.
        translator.pack_read_stream.assert_called_once()
        call = translator.pack_read_stream.call_args
        self.assertIs(call.args[0], plan)
        self.assertEqual(call.kwargs["seq_lens"].tolist(), [103, 209])
        self.assertEqual(call.kwargs["indptr"].tolist(), [0, 103, 312])
        self.assertIs(call.kwargs["out"], kv_indices)

if __name__ == "__main__":
    unittest.main()
