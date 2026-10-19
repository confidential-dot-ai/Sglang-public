"""FlashInfer autotune records are keyed by everything that changes the tuned profile."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner import flashinfer_autotune
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_pd_roles_have_separate_autotune_records(tmp_path):
    mr = SimpleNamespace(
        device="cuda",
        dtype="bfloat16",
        is_draft_worker=False,
        tp_size=2,
        attn_dp_size=1,
        moe_ep_size=2,
        tp_rank=0,
        dp_rank=0,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    role = SimpleNamespace(disaggregation_mode="prefill")
    with (
        patch.object(
            flashinfer_autotune.torch.cuda,
            "get_device_capability",
            return_value=(10, 3),
        ),
        patch.object(
            flashinfer_autotune,
            "get_model",
            return_value=SimpleNamespace(model_path="model", quantization="mxfp4"),
        ),
        patch.object(
            flashinfer_autotune,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(moe_runner_backend="flashinfer_mxfp4")
            ),
        ),
        patch.object(
            flashinfer_autotune,
            "get_parallel",
            return_value=SimpleNamespace(pp_size=1, pp_rank=0),
        ),
        patch.object(flashinfer_autotune, "get_disagg", return_value=role),
        patch.object(
            flashinfer_autotune, "get_flashinfer_autotune_skip_ops", return_value=set()
        ),
        patch.object(
            flashinfer_autotune.envs.SGLANG_CACHE_DIR, "get", return_value=str(tmp_path)
        ),
    ):
        prefill = flashinfer_autotune.flashinfer_autotune_cache_path(mr)
        prefill.write_text("prefill tactics")
        role.disaggregation_mode = "decode"
        decode = flashinfer_autotune.flashinfer_autotune_cache_path(mr)
        assert decode != prefill
        decode.write_text("decode tactics")
        assert prefill.read_text() == "prefill tactics"
