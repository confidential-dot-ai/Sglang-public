"""Unit tests for _handle_dspark's DP-attention + MoE A2A backend guard.

flashinfer MoE A2A (MoeAlltoAll) requires DP attention (dp_size == tp_size),
and was previously blocked unconditionally whenever DSpark used DP attention
-- a leftover conservative guard from DSpark's original commit, not a
correctness requirement (see TODO.md Track J14b). It's relaxed to allow
moe_a2a_backend='flashinfer' specifically, gated on
SGLANG_RAGGED_VERIFY_MODE=static: the flashinfer combine assumes every DP
rank dispatches the same padded token count per verify step, which only
holds under static (non-compact) verify.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _make_dp_server_args(*, moe_a2a_backend: str) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = _BUNDLED_MODEL_PATH
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    # Bundled draft path avoids unrelated draft-path-resolution errors --
    # this test only cares about the dp_attention/moe_a2a_backend branch.
    server_args.speculative_draft_model_path = _BUNDLED_MODEL_PATH
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=_bundled_hf_config())
    server_args.enable_dp_attention = True
    server_args.enable_dp_lm_head = True
    server_args.attn_cp_size = 1
    server_args.moe_a2a_backend = moe_a2a_backend
    server_args.speculative_moe_a2a_backend = None
    return server_args


class TestDsparkDpAttentionA2AGuard(CustomTestCase):
    def test_none_backend_still_allowed(self):
        server_args = _make_dp_server_args(moe_a2a_backend="none")
        _handle_dspark(server_args)  # must not raise

    def test_flashinfer_backend_allowed_under_static_verify(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            server_args = _make_dp_server_args(moe_a2a_backend="flashinfer")
            _handle_dspark(server_args)  # must not raise

    def test_flashinfer_backend_rejected_under_non_static_verify(self):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            server_args = _make_dp_server_args(moe_a2a_backend="flashinfer")
            with self.assertRaisesRegex(ValueError, "SGLANG_RAGGED_VERIFY_MODE"):
                _handle_dspark(server_args)

    def test_deepep_backend_still_rejected(self):
        server_args = _make_dp_server_args(moe_a2a_backend="deepep")
        with self.assertRaisesRegex(ValueError, "moe_a2a_backend"):
            _handle_dspark(server_args)

    def test_mooncake_backend_still_rejected(self):
        server_args = _make_dp_server_args(moe_a2a_backend="mooncake")
        with self.assertRaisesRegex(ValueError, "moe_a2a_backend"):
            _handle_dspark(server_args)


if __name__ == "__main__":
    unittest.main()
