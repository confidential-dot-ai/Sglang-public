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
            return_value=SimpleNamespace(
                tp_size=2,
                pp_size=1,
                attn_dp_size=1,
                moe_ep_size=2,
                tp_rank=0,
                pp_rank=0,
                dp_rank=0,
            ),
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


def _extend_runner():
    model = SimpleNamespace(
        wants_prefill_autotune=lambda: True,
        autotune_prefill_kernels=lambda num_tokens, dtype: 7,
    )
    mr = SimpleNamespace(
        model=model,
        dtype="bfloat16",
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        attn_backend=SimpleNamespace(extend_dummy_seqs_capped_by_req_pool=False),
        canary_manager=None,
    )
    return SimpleNamespace(
        model_runner=mr,
        _alloc_dummy_decode_buffers=lambda *a, **k: object(),
        _dummy_run=lambda **k: None,
    )


def test_model_prefill_hook_does_not_skip_the_opted_in_extend_pass():
    """DSV4.1's prefill hook tunes only its dense linears; returning early on it
    left the fused MoE on tactics tuned for <= 16 tokens at 1K-12K-token prefill."""
    from contextlib import nullcontext

    from sglang.srt.environ import envs

    for opted_in, expect_extend in ((False, False), (True, True)):
        forwards = []
        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.override(opted_in),
            patch.object(flashinfer_autotune, "max_prefill_buffer_tokens", return_value=12288),
            patch.object(
                flashinfer_autotune, "flashinfer_autotune_context", lambda *a, **k: nullcontext()
            ),
            patch.object(
                flashinfer_autotune,
                "run_flashinfer_autotune_forward",
                lambda mr, fn, **k: forwards.append(fn),
            ),
            patch.object(
                flashinfer_autotune,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
        ):
            flashinfer_autotune.maybe_flashinfer_autotune_extend(
                _extend_runner(), decode_num_tokens=16
            )
        assert bool(forwards) == expect_extend, opted_in
