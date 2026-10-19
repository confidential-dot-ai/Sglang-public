from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre, mhc_post, mhc_pre
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@pytest.fixture
def stated_tp_group():
    """Provide a TP-group placeholder for kernels with mocked symmetric memory."""
    from sglang.srt.runtime_context import get_parallel

    with get_parallel().override(tp_group=None):
        yield


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("num_tokens", [0, 1, 6, 8, 17, 32, 64])
@pytest.mark.parametrize("use_norm", [False, True])
def test_mhc_fused_post_pre_matches_unfused(
    monkeypatch, hidden_size, num_tokens, use_norm, stated_tp_group
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TileLang mHC kernels")

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    # Disable symmetric-memory allocation for this single-process kernel test.
    monkeypatch.setattr(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(mhc, "is_allocation_symmetric", lambda: False)
    torch.manual_seed(0)
    device = torch.device("cuda")
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    residual = (
        torch.randn(
            num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    post_prev = torch.rand(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_prev = (
        torch.rand(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
        * 0.25
    )
    fn = (
        torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01
    )
    hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=device, dtype=torch.float32)
    norm_weight = (
        torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
        if use_norm
        else None
    )
    norm_eps = 1e-6 if use_norm else None

    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_repeat = 2

    residual_ref = post_ref = comb_ref = layer_ref = None
    if num_tokens > 0:
        residual_ref = mhc_post(x, residual, post_prev, comb_prev)
        post_ref, comb_ref, layer_ref = mhc_pre(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            2.0,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    residual_out, post_out, comb_out, layer_out = mhc_fused_post_pre(
        x,
        residual,
        post_prev,
        comb_prev,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        2.0,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    if hidden_size == 4096 and num_tokens in (0, 1, 6, 17):
        _check_glm_boundary(
            x,
            residual,
            post_prev,
            comb_prev,
            fn,
            hc_scale,
            hc_base,
            use_norm=use_norm,
        )

    torch.cuda.synchronize()
    if num_tokens == 0:
        assert residual_out.shape == residual.shape
        assert post_out.shape == (0, hc_mult, 1)
        assert comb_out.shape == (0, hc_mult, hc_mult)
        assert layer_out.shape == (0, hidden_size)
        assert residual_out.dtype == torch.bfloat16
        assert post_out.dtype == torch.float32
        assert comb_out.dtype == torch.float32
        assert layer_out.dtype == torch.bfloat16
        return

    assert residual_ref is not None
    assert post_ref is not None
    assert comb_ref is not None
    assert layer_ref is not None
    assert residual_out.shape == residual_ref.shape
    assert post_out.shape == post_ref.shape
    assert comb_out.shape == comb_ref.shape
    assert layer_out.shape == layer_ref.shape

    torch.testing.assert_close(residual_out, residual_ref, atol=0, rtol=0)
    torch.testing.assert_close(post_out, post_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(comb_out, comb_ref, atol=1e-3, rtol=1e-3)
    layer_atol = 2e-2 if use_norm else 2e-3
    layer_rtol = 2e-2 if use_norm else 2e-3
    torch.testing.assert_close(layer_out, layer_ref, atol=layer_atol, rtol=layer_rtol)


def _check_glm_boundary(x, residual, post, comb, fn, scale, base, *, use_norm):
    from sglang.srt.environ import envs
    from sglang.srt.layers.communicator import MHCState
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        mhc=True,
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    layer.hc_ffn_fn = torch.nn.Parameter(fn)
    layer.hc_ffn_scale = torch.nn.Parameter(scale)
    layer.hc_ffn_base = torch.nn.Parameter(base)
    norm = RMSNorm(x.shape[-1], eps=1e-6).to(x) if use_norm else None
    states = [
        MHCState(
            hc_mult=4,
            hc_attn_pre=layer.hc_attn_pre,
            hc_ffn_pre=layer.hc_ffn_pre,
            hc_post=layer.hc_post,
            hc_ffn_post_pre=callback,
            h_res=comb.flatten(1),
            h_post=post.flatten(1),
        )
        for callback in (None, layer.hc_ffn_post_pre)
    ]
    # Literal, not derived from the cutoff constant: deriving it makes this a
    # mirror that stays green when the cutoff moves. None is the empty batch,
    # which attn_to_mlp short-circuits before reaching the callback.
    fused_expected = {1: True, 6: True, 17: False}[x.shape[0]] if x.shape[0] else None
    with envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(True):
        if x.shape[0] > 0:
            declined = (
                layer.hc_ffn_post_pre(
                    hidden_states=x,
                    residual=residual.flatten(1),
                    h_res=comb.flatten(1),
                    h_post=post.flatten(1),
                    out_norm_weight=None,
                    out_norm_eps=None,
                )
                is None
            )
            assert declined is not fused_expected, (
                f"num_tokens={x.shape[0]} fused={not declined}, "
                f"expected fused={fused_expected}"
            )
        outputs = [s.attn_to_mlp(x, residual.flatten(1), norm) for s in states]
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
    torch.testing.assert_close(states[0].h_res, states[1].h_res, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(states[0].h_post, states[1].h_post, atol=1e-3, rtol=1e-3)
    # The next combine must consume the FFN mixing matrices, not attention's.
    torch.testing.assert_close(
        states[0].mlp_combine(x, outputs[0][1]),
        states[1].mlp_combine(x, outputs[1][1]),
        atol=2e-3,
        rtol=2e-2,
    )


# DeepSeek-V4.1 HC=4 coefficients: 24 mixes over hc_mult * hidden_size inputs.
_MIX_K, _MIX_N, _SINKHORN_ITERS, _HC_EPS = 20480, 24, 20, 1e-6


def _hc_mix_inputs(m):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(m, _MIX_K, device="cuda", generator=gen).bfloat16()
    fn = torch.randn(_MIX_N, _MIX_K, device="cuda", generator=gen) * 0.01
    scale = torch.rand(3, device="cuda", generator=gen)
    base = torch.randn(_MIX_N, device="cuda", generator=gen) * 0.1
    return x, fn, scale, base


def _hc_mix_reference(x, fn, scale, base):
    x, fn, scale, base = (t.double() for t in (x, fn, scale, base))
    mix = x @ fn.T * torch.rsqrt(x.square().mean(-1, keepdim=True) + _HC_EPS)
    pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + _HC_EPS
    post = 2 * torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8])
    comb = (mix[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4)
    comb = torch.softmax(comb, -1) + _HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + _HC_EPS)
    for _ in range(_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + _HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + _HC_EPS)
    return pre, post, comb


def _hc_mix_run(x, fn, scale, base):
    return mhc.hc_mix_stats_sinkhorn(
        x, fn, scale, base, 4, _SINKHORN_ITERS, _HC_EPS, _HC_EPS
    )


@pytest.mark.parametrize("m", [1, 7, 84, 127])
def test_hc_mix_stats_sinkhorn_matches_float64_reference(m):
    inputs = _hc_mix_inputs(m)
    for got, want in zip(_hc_mix_run(*inputs), _hc_mix_reference(*inputs)):
        torch.testing.assert_close(got.double(), want, rtol=0, atol=2e-6)


def test_hc_mix_stats_sinkhorn_rows_are_batch_invariant():
    # Decode/verify batches of any size must give each row the same bits.
    x, fn, scale, base = _hc_mix_inputs(127)
    full = _hc_mix_run(x, fn, scale, base)
    for m in (1, 8, 84):
        for got, want in zip(_hc_mix_run(x[:m].contiguous(), fn, scale, base), full):
            assert torch.equal(got, want[:m])
    for row in (0, 63, 126):
        alone = _hc_mix_run(x[row : row + 1].contiguous(), fn, scale, base)
        for got, want in zip(alone, full):
            assert torch.equal(got, want[row : row + 1])

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
