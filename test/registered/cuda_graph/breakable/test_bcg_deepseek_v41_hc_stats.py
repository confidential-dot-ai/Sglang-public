"""A DeepSeek-V4.1 decoder layer's mHC replayed from a breakable CUDA graph matches
eager execution, for >=4096-row prefill and for decode and target-verify batches.

Regression: those modes computed the attention sublayer's mHC coefficients on a
side stream forked before the attention's eager graph break, so the coefficient
kernels escaped the capture and replay mixed the residual streams with stale
values: garbage prefill for every prompt padded into the 4096-token bucket, and
decode that could not copy from context beyond the sliding window. The guard is
`deepseek_v4_mhc.use_stats_stream`, consulted by the CUDA boundary's mHC read.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC, HIDDEN = 4, 5120


def _bare_layer(device):
    """The mHC state of one V4.1 decoder layer, as its CUDA boundary reads and
    updates it; attention and MoE are stubbed."""
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models import deepseek_v4_mhc as mhc

    gen = torch.Generator(device=device).manual_seed(0)

    def param(*shape, scale, shift=0.0):
        value = torch.randn(*shape, device=device, generator=gen) * scale + shift
        return nn.Parameter(value, requires_grad=False)

    cfg = mhc.HcConfig(
        mult=HC,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-20,
        hidden=HIDDEN,
        pre_from_prev=True,
        cp_prefill=False,
    )

    def sublayer():
        norm = RMSNorm(HIDDEN, eps=1e-20).to(device=device, dtype=torch.bfloat16)
        noise = torch.randn(HIDDEN, device=device, generator=gen)
        norm.weight.data.copy_(1 + 0.1 * noise)
        return mhc.HcSubLayer(
            cfg,
            param(24, HC * HIDDEN, scale=1e-2),
            param(3, scale=0.1, shift=1.0),
            param(24, scale=0.1),
            norm,
        )

    return SimpleNamespace(
        boundary_pre_from_prev=True,
        boundary_mhc_fusions=False,
        hc_cfg=cfg,
        hc_stats_stream=torch.cuda.Stream(device),
        attn_hc=sublayer(),
        ffn_hc=sublayer(),
        self_attn=SimpleNamespace(
            attn_tp_size=1, accepts_mxfp8_swizzled_input=lambda: False
        ),
        mlp=SimpleNamespace(tp_size=1),
    )


def _layer_forward(layer, hidden, prev_pre, forward_batch):
    """Both sublayers through the boundary's mHC read and update, the attention
    as an eager break of the graph; the residual and the FFN's pre out."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        eager_on_graph,
    )
    from sglang.srt.models import deepseek_v4_boundary as boundary
    from sglang.srt.models import deepseek_v4_mhc as mhc

    @eager_on_graph(True)
    def attention_break(x):
        # The prefill attention runs as an eager break of the graph.
        return x * 0.5 + 0.25

    call = boundary.LayerCall()
    state = mhc.HcState(hidden, prev_pre)
    for attention, sublayer in ((True, attention_break), (False, lambda x: x * 0.75)):
        read = boundary.HcRead(layer, attention).bind(forward_batch, call)
        x, read_state = read.read(state, None)
        state = boundary.HcUpdate(layer, attention).post(sublayer(x), read_state)
    return state.residual, state.pre


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "the V4.1 mHC prefill kernels and stats stream are Blackwell-only",
)
class TestBreakablePrefillHcStats(CustomTestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_parallel

        self.enterContext(
            get_parallel().override(attn_dp_size=1, attn_cp_size=1, tp_size=1)
        )

    def test_replay_matches_eager(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        # Prefill from the 4096 bucket up, a DSPARK verify batch (8 x 6 tokens), decode.
        for mode, rows in (
            (ForwardMode.EXTEND, 4096),
            (ForwardMode.TARGET_VERIFY, 48),
            (ForwardMode.DECODE, 64),
        ):
            with self.subTest(mode=mode.name, rows=rows):
                self._check_replay(mode, rows)

    @torch.no_grad()
    def _check_replay(self, mode, rows):
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            context,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
        )

        device = torch.device("cuda")
        layer = _bare_layer(device)
        forward_batch = SimpleNamespace(forward_mode=mode)
        hidden = torch.empty(rows, HC, HIDDEN, device=device, dtype=torch.bfloat16)
        prev_pre = torch.empty(rows, HC, device=device)

        def load(seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            hidden.copy_(torch.randn(hidden.shape, device=device, generator=gen))
            prev_pre.copy_(torch.rand(prev_pre.shape, device=device, generator=gen))

        def forward():
            return _layer_forward(layer, hidden, prev_pre, forward_batch)

        load(0)
        graph = BreakableCUDAGraph()
        with context.enable_breakable_cuda_graph():
            forward()  # warm up, as the runner does before capturing
            torch.cuda.synchronize()
            with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream(device)):
                graph_out, graph_pre = forward()
        for seed in (1, 2):
            load(seed)
            # Recycle the allocator blocks the capture freed, as later work would.
            scratch = [torch.randn(rows, 512, device=device) for _ in range(8)]
            with context.enable_breakable_cuda_graph():
                graph.replay()
            want_out, want_pre = forward()  # eager prefill
            torch.cuda.synchronize()
            del scratch
            torch.testing.assert_close(graph_pre, want_pre, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(graph_out, want_out, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
