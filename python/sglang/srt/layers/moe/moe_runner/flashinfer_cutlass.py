"""FlashInfer CUTLASS MoE fused funcs.

This module owns the FlashInfer ``cutlass_fused_moe`` calls used by the
unquantized, ModelOpt FP8, ModelOpt NVFP4, and CUTLASS MXFP4 MoE paths, plus
the shared ``flashinfer_mxfp4`` dispatcher. Quantization methods prepare a
small quant_info payload and route through ``MoeRunner``.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import next_power_of_2

# FlashInfer's own default (flashinfer.fused_moe.core.cutlass_fused_moe's
# tune_max_num_tokens parameter) is 8192 -- this is the top bucket its
# AutoTuner profiles (MoERunner.refine_tuning_config's dynamic_tensor_specs),
# and the profiling pass allocates a CUTLASS grouped-GEMM workspace sized for
# that top bucket. Passing the real batch's next_power_of_2 uncapped (as
# below) forces that profiling-time workspace to scale with whatever the
# current --chunked-prefill-size/concurrent-batch happens to be; at 131072+
# tokens this overflows the workspace allocator inside
# FusedMoeRunner::getWorkspaceInfo and kills the scheduler process outright
# (observed live: crashes at both 262144 and 131072 chunk sizes). Capping at
# FlashInfer's own tested default is safe for correctness: map_to_hybrid_bucket
# already clamps any real runtime token count down to the nearest tuned
# bucket for dispatch, so a real batch larger than this cap just reuses the
# tactic tuned at the cap -- normal, expected MoE-kernel behavior, not a
# correctness gap.
_MOE_TUNE_MAX_TOKENS_CAP = 8192

# Real per-forward-call row cap, separate from the tuning cap above.
# FusedMoeRunner::getWorkspaceInfo (TensorRT-LLM cpp/tensorrt_llm/thop/
# moeOp.cpp) resizes its CUTLASS grouped-GEMM workspace to fit `num_rows`
# on EVERY real forward call -- uncapped, freeing the old buffer first --
# independent of tune_max_num_tokens (which only bounds the AutoTuner's
# own one-time profiling-pass allocation at startup; a real forward call
# with a bigger row count still resizes this buffer regardless of what
# was tuned for). A single call with ~97k-100k rows (25+ concurrent
# sequences batched together under a large --chunked-prefill-size) crashed
# the prefill scheduler process outright at both 131072 and 262144, even
# with the tuning cap above already landed (2026-07-24 investigation, see
# tdx-checkpoints/128 and its predecessors). Most likely a genuine
# VRAM-headroom OOM under --mem-fraction-static 0.80 rather than a hard
# kernel limit, but the original crash's own OOM message was never
# captured (log truncated by the next relaunch) -- so this is not
# confirmed root-caused, only worked around.
#
# _run_flashinfer_cutlass below splits any real forward call wider than
# this into sequential <=-this-many-row sub-calls, each independently
# workspace-bounded, so --chunked-prefill-size can grow for prefill/
# decode-pipeline throughput without growing what the MoE runner's C++
# allocator ever has to size for in one shot. Chosen equal to the
# separately-validated-safe --chunked-prefill-size (32768) from the same
# investigation, not derived from any workspace-size math -- there is no
# formula available from the visible (compiled, not patched) C++ for what
# the *actual* safe bound is, so this reuses the one row count already
# proven not to crash.
#
# UNTESTED as of this comment: written with the host's 8 GPUs held by a
# separate workload, so no live validation was possible. Before trusting
# this in production, validate against a real concurrent flood at
# >32768-token chunk sizes for: (a) no crash, (b) output correctness vs.
# the known-safe single-call 32768 baseline (chunked-vs-unchunked MoE
# output should match bit-for-bit modulo kernel nondeterminism -- diff
# them), and (c) actual throughput gain, since row-chunking trades N
# sequential kernel launches for the single-shot workspace risk it avoids
# and could net negative if the sequential overhead dominates.
#
# Lowered from 32768 to 8192 (2026-07-27 investigation): at 32768 (equal to
# our --chunked-prefill-size), this cap never actually triggers the split --
# every real forward call already fits in one sub-call, so the workspace
# still gets resized up to whatever real chunk sizes reach as concurrency
# ramps toward the 32768 cap. Measured live: real c64 traffic against the
# known-safe 900k-token-cap config grew per-GPU memory.used by a consistent
# ~2.97-2.98 GiB, monotonically, plateauing once chunk sizes stopped
# growing -- consistent with the caching allocator retaining the
# largest-ever workspace resize rather than shrinking back down.
# _MOE_TUNE_MAX_TOKENS_CAP=8192 is already the AutoTuner's own top tuned
# bucket, so this reuses a bucket already proven to be profiled/tuned for
# rather than picking an arbitrary smaller number. Testing whether this
# actually bounds that growth, and whether (b)/(c) above hold at this
# cap -- see TODO.md.
_MOE_FORWARD_ROW_CAP = 8192

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class FlashInferCutlassMoeQuantInfo(MoeQuantInfo):
    """Payload for FlashInfer CUTLASS fused MoE.

    ``quant_type`` selects the input/weight conventions:
      - ``"bf16"``: unquantized weights, BF16/FP16 input, no quant scales.
      - ``"fp8"``: FP8 weights, FP8-quantized input, per-tensor scales.
      - ``"fp4"``: NVFP4 packed weights and optional NVFP4 packed input.
    """

    quant_type: str
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_scales: Optional[list[torch.Tensor]] = None
    output_dtype: Optional[torch.dtype] = None
    # Optional per-expert SwiGLU overrides, fp32 [num_local_experts].
    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None
    apply_routed_scaling_factor: bool = True


def materialize_swiglu_params_for_cutlass(
    runner_config: MoeRunnerConfig,
    num_local_experts: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Per-expert SwiGLU (alpha, beta, limit) tensors for the CUTLASS kernel.

    Returns all-None unless a clamp limit is configured. ``gemm1_alpha``
    implies the GPT-OSS-style ``+1`` up term, so beta then defaults to 1.0;
    without alpha, alpha/beta stay silu-neutral at 1.0 / 0.0. The kernel
    clamps dequantized values, so the limit is in physical units (no
    g1_alphas conversion, unlike trtllm-gen). SiTU carries its clamp in the
    activation itself.
    """
    if runner_config.activation == "situ":
        return None, None, None
    clamp_limit = runner_config.gemm1_clamp_limit or runner_config.swiglu_limit
    if clamp_limit is None:
        return None, None, None
    alpha = runner_config.gemm1_alpha if runner_config.gemm1_alpha is not None else 1.0
    beta = runner_config.gemm1_beta
    if beta is None:
        beta = 1.0 if runner_config.gemm1_alpha is not None else 0.0
    return (
        torch.full(
            (num_local_experts,), float(alpha), dtype=torch.float32, device=device
        ),
        torch.full(
            (num_local_experts,), float(beta), dtype=torch.float32, device=device
        ),
        torch.full(
            (num_local_experts,),
            float(clamp_limit),
            dtype=torch.float32,
            device=device,
        ),
    )


@dataclass
class FlashInferCutlassMxfp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload for CUTLASS MXFP4 MoE.

    SM90 consumes either W4A16-interleaved weights/scales or the Humming-style
    W4A8 layouts. SM120 consumes packed MXFP4 weights and block-interleaved
    scales with MXFP8 activations.
    """

    # SM90 weights are interleaved; SM120 weights remain checkpoint-packed.
    w13_weight: torch.Tensor  # [E, 2*N, K/2]
    w2_weight: torch.Tensor  # [E, K, N/2]

    # E8M0 block scales in the layout selected by the quantization method.
    w13_weight_scale: torch.Tensor  # [E, 2*N, K/32]
    w2_weight_scale: torch.Tensor  # [E, K, N/32]

    # A non-None global scale selects the SM120 MXFP8 activation path.
    mxfp4_weight_global_scale: Optional[torch.Tensor] = None

    # A complete non-None triplet selects the SM90 Humming W4A8 path. The
    # residuals are FP32 [num_local_experts] and already include the fixed 2^6
    # compensation required by FlashInfer's epilogue.
    w13_humming_residual_scale: Optional[torch.Tensor] = None
    w2_humming_residual_scale: Optional[torch.Tensor] = None
    humming_fc2_act_scale: Optional[torch.Tensor] = None

    # Per-expert bias. GPT-OSS has both; DSv4 leaves both None.
    w13_bias: Optional[torch.Tensor] = None  # bf16 [E, 2*N]
    w2_bias: Optional[torch.Tensor] = None  # bf16 [E, K]

    # Optional per-expert SwiGLU overrides, fp32 [E].
    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None

    # Bailing clamps after SiLU, which the kernel only implements in its
    # SwigluStep variant.
    use_swiglu_step: bool = False

    # GPT-OSS pads its input hidden dim up to the (pre-padded) loaded weight
    # width and trims the output back. DSv4 leaves this as ``None`` (no pad).
    padded_hidden: Optional[int] = None


def _flashinfer_cutlass_fused_moe():
    if not is_flashinfer_available():
        raise RuntimeError(
            "flashinfer_cutlass MoE runner backend requires flashinfer to be installed."
        )
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    return cutlass_fused_moe, ActivationType


@functools.lru_cache(maxsize=1)
def _flashinfer_supports_fused_finalize() -> bool:
    """cutlass_fused_moe's ``use_fused_finalize`` kwarg needs FlashInfer
    >=0.6.16 (#28354); older pinned installs (e.g. this repo's validated
    0.6.14) raise TypeError on an unknown kwarg. Detected once via
    inspect.signature rather than a hardcoded version cutoff so this tracks
    whatever FlashInfer build is actually installed."""
    cutlass_fused_moe, _ = _flashinfer_cutlass_fused_moe()
    return "use_fused_finalize" in inspect.signature(cutlass_fused_moe).parameters


def _fused_finalize_kwargs() -> dict:
    if not _flashinfer_supports_fused_finalize():
        return {}
    return {"use_fused_finalize": envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()}


def _activation_type(runner_config: MoeRunnerConfig):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type

    _, ActivationType = _flashinfer_cutlass_fused_moe()
    activation = ActivationType(
        get_activation_type(
            runner_config.activation,
            is_gated=runner_config.is_gated,
        )
    )
    supported = {
        ActivationType.Swiglu,
        ActivationType.Geglu,
        ActivationType.Relu2,
        ActivationType.Identity,
    }
    assert activation in supported, (
        f"Activation {runner_config.activation!r} "
        f"(is_gated={runner_config.is_gated}) maps to {activation.name}, "
        "which is not supported by flashinfer cutlass moe."
    )
    return activation


def _maybe_apply_routed_scaling_factor(
    output: torch.Tensor,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    if (
        quant_info.apply_routed_scaling_factor
        and runner_config.routed_scaling_factor is not None
    ):
        output.mul_(runner_config.routed_scaling_factor)
    return output


def _prescale_router_weight_on_input(
    dispatch_output: StandardDispatchOutput | FlashinferDispatchOutput,
    runner_config: MoeRunnerConfig,
) -> StandardDispatchOutput | FlashinferDispatchOutput:
    if not runner_config.apply_router_weight_on_input:
        return dispatch_output

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights

    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported when activations are "
            "quantized before dispatch (flashinfer_cutlass fp4 all-gather path)."
        )

    assert topk_weights.dim() == 2 and topk_weights.shape[-1] == 1, (
        "apply_router_weight_on_input requires topk=1"
    )

    hidden_states = dispatch_output.hidden_states * topk_weights.to(
        dispatch_output.hidden_states.dtype
    )
    unit_scales = torch.ones_like(topk_weights, dtype=torch.float32)
    return dispatch_output._replace(
        hidden_states=hidden_states,
        topk_output=topk_output._replace(topk_weights=unit_scales),
    )


def _prepare_input(
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype, int]:
    x = dispatch_output.hidden_states
    x_sf = dispatch_output.hidden_states_scale

    if quant_info.quant_type == "fp8":
        assert quant_info.quant_scales is not None and len(quant_info.quant_scales) == 4
        x, _ = scaled_fp8_quant(x, quant_info.quant_scales[3])
        x_sf = None
        output_dtype = quant_info.output_dtype or dispatch_output.hidden_states.dtype
        output_col = dispatch_output.hidden_states.shape[1]
    elif quant_info.quant_type == "fp4":
        output_dtype = quant_info.output_dtype or torch.bfloat16
        output_col = x.shape[1]
        if x_sf is not None and runner_config.is_gated:
            output_col *= 2
    else:
        assert quant_info.quant_type == "bf16"
        output_dtype = quant_info.output_dtype or x.dtype
        output_col = x.shape[1]

    return x, x_sf, output_dtype, output_col


def _run_flashinfer_cutlass_call(
    *,
    flashinfer_cutlass_fused_moe,
    output: torch.Tensor,
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    activation_type,
    enable_alltoall: bool,
) -> None:
    """One workspace-bounded call -- `x.shape[0]` must already be
    <= _MOE_FORWARD_ROW_CAP by the time this is reached.

    Explicitly assigns the kernel's returned tensor into `output` via a
    slice-copy rather than trusting the `output=` kwarg to have written
    in place: the original (pre-chunking) call site did
    `output = flashinfer_cutlass_fused_moe(output=output, ...)[0]` and used
    THAT return value going forward, not the passed-in buffer directly --
    so whether this kernel aliases its `output=` argument or returns a
    fresh tensor is not established here, and an explicit copy is correct
    either way (a plain kwarg-write assumption would silently no-op if the
    kernel doesn't alias)."""
    result = flashinfer_cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight,
        fc2_expert_weights=w2_weight,
        output_dtype=output_dtype,
        input_sf=x_sf,
        quant_scales=quant_scales,
        ep_size=runner_config.moe_ep_size,
        ep_rank=runner_config.moe_ep_rank,
        tp_size=runner_config.moe_tp_size,
        tp_rank=runner_config.moe_tp_rank,
        tune_max_num_tokens=min(next_power_of_2(x.shape[0]), _MOE_TUNE_MAX_TOKENS_CAP),
        activation_type=activation_type,
        enable_alltoall=enable_alltoall,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        **_fused_finalize_kwargs(),
    )[0]
    if result.data_ptr() != output.data_ptr():
        output.copy_(result)


def _run_flashinfer_cutlass(
    *,
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
) -> torch.Tensor:
    flashinfer_cutlass_fused_moe, _ = _flashinfer_cutlass_fused_moe()

    dispatch_output = _prescale_router_weight_on_input(dispatch_output, runner_config)
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights.to(torch.float32)
    topk_ids = topk_output.topk_ids
    x, x_sf, output_dtype, output_col = _prepare_input(
        dispatch_output, quant_info, runner_config
    )

    if output is None:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            output = torch.empty(
                x.shape[0],
                output_col,
                dtype=output_dtype,
                device=x.device,
            )

    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    quant_scales = quant_info.quant_scales
    if quant_info.quant_type == "fp4":
        w13_weight = w13_weight.view(torch.long)
        w2_weight = w2_weight.view(torch.long)
        assert quant_scales is not None and len(quant_scales) == 6
        quant_scales = [
            quant_scales[0],
            quant_scales[1].view(torch.int32),
            quant_scales[2],
            quant_scales[3],
            quant_scales[4].view(torch.int32),
            quant_scales[5],
        ]

    activation_type = _activation_type(runner_config)
    num_rows = x.shape[0]

    if num_rows <= _MOE_FORWARD_ROW_CAP:
        _run_flashinfer_cutlass_call(
            flashinfer_cutlass_fused_moe=flashinfer_cutlass_fused_moe,
            output=output,
            x=x,
            x_sf=x_sf,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            output_dtype=output_dtype,
            quant_scales=quant_scales,
            quant_info=quant_info,
            runner_config=runner_config,
            activation_type=activation_type,
            enable_alltoall=enable_alltoall,
        )
    else:
        # Real forward call exceeds the workspace-safe row cap -- split
        # into sequential sub-calls, each independently bounded, instead
        # of handing the whole batch to a single CUTLASS grouped-GEMM
        # call. See _MOE_FORWARD_ROW_CAP's comment for why this exists
        # and what it trades off.
        for start in range(0, num_rows, _MOE_FORWARD_ROW_CAP):
            end = min(start + _MOE_FORWARD_ROW_CAP, num_rows)
            _run_flashinfer_cutlass_call(
                flashinfer_cutlass_fused_moe=flashinfer_cutlass_fused_moe,
                output=output[start:end],
                x=x[start:end],
                x_sf=x_sf[start:end] if x_sf is not None else None,
                topk_ids=topk_ids[start:end],
                topk_weights=topk_weights[start:end],
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                output_dtype=output_dtype,
                quant_scales=quant_scales,
                quant_info=quant_info,
                runner_config=runner_config,
                activation_type=activation_type,
                enable_alltoall=enable_alltoall,
            )

    if quant_info.quant_type in ("bf16", "fp8"):
        _maybe_apply_routed_scaling_factor(output, quant_info, runner_config)
    return output


@register_fused_func("none", "flashinfer_cutlass")
def fused_experts_none_to_flashinfer_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, FlashInferCutlassMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    )

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutlass")
def fused_experts_flashinfer_to_flashinfer_cutlass(
    dispatch_output: FlashinferDispatchOutput | StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput | StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    # All-gather prefill must use ordinary expert execution and reduce-scatter
    # combine, without the decode A2A workspace or kernel layout.
    if isinstance(dispatch_output, StandardDispatchOutput):
        return fused_experts_none_to_flashinfer_cutlass(
            dispatch_output, quant_info, runner_config
        )

    assert isinstance(quant_info, FlashInferCutlassMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    )

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
        output=dispatch_output.moe_output,
        enable_alltoall=True,
    )
    return FlashinferCombineInput(hidden_states=output)


@register_fused_func("none", "flashinfer_mxfp4")
def fused_experts_none_to_flashinfer_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Dispatch flashinfer_mxfp4 by quant-info type.

    Both mxfp4 paths register under this single ``("none", "flashinfer_mxfp4")``
    key but call different kernels.
    """
    if isinstance(quant_info, FlashInferCutlassMxfp4MoeQuantInfo):
        return _fused_experts_flashinfer_mxfp4_cutlass(
            dispatch_output, quant_info, runner_config
        )

    # Keep one fused-op registration for the shared backend while loading the
    # TRT-LLM implementation only when its quant-info type is dispatched.
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
    )

    if isinstance(quant_info, FlashInferTrtllmGenMxfp4MoeQuantInfo):
        return _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output, quant_info, runner_config
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"
    )


def _fused_experts_flashinfer_mxfp4_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferCutlassMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    x = dispatch_output.hidden_states
    if x.shape[0] == 0:
        return StandardCombineInput(hidden_states=x)

    flashinfer_cutlass_fused_moe, ActivationType = _flashinfer_cutlass_fused_moe()

    topk_output = dispatch_output.topk_output

    # Under ``--moe-runner-backend flashinfer_mxfp4`` topk may be in bypassed
    # form (the SM100 trtllm-gen path does routing internally). The CUTLASS
    # SM90 path needs explicit topk_ids / topk_weights; materialize here.
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # GPT-OSS: pad input hidden dim up to the loaded weight width. DSv4
    # leaves padded_hidden as None (or equal to origin_hidden), no pad.
    origin_hidden = x.shape[-1]
    padded_hidden = quant_info.padded_hidden
    do_pad = padded_hidden is not None and padded_hidden != origin_hidden
    if do_pad:
        x = torch.nn.functional.pad(
            x,
            (0, padded_hidden - origin_hidden),
            mode="constant",
            value=0.0,
        )

    weight_global_scale = quant_info.mxfp4_weight_global_scale
    use_mxfp8_act_scaling = weight_global_scale is not None
    w13_humming_residual_scale = quant_info.w13_humming_residual_scale
    w2_humming_residual_scale = quant_info.w2_humming_residual_scale
    humming_fc2_act_scale = quant_info.humming_fc2_act_scale
    humming_scales = (
        w13_humming_residual_scale,
        w2_humming_residual_scale,
        humming_fc2_act_scale,
    )
    use_wfp4afp8_humming = any(scale is not None for scale in humming_scales)
    if use_wfp4afp8_humming and not all(scale is not None for scale in humming_scales):
        raise ValueError(
            "SM90 Humming MXFP4 MoE requires both expert residual scales "
            "and the FC2 activation scale."
        )
    if use_wfp4afp8_humming and use_mxfp8_act_scaling:
        raise ValueError("SM90 Humming and SM120 MXFP8 scaling are mutually exclusive.")
    input_sf = None
    fc1_expert_weights = quant_info.w13_weight
    fc2_expert_weights = quant_info.w2_weight
    if weight_global_scale is not None:
        from flashinfer import mxfp8_quantize

        x, input_sf = mxfp8_quantize(
            x,
            is_sf_swizzled_layout=True,
            alignment=32,
        )
        fc1_expert_weights = fc1_expert_weights.view(torch.int64)
        fc2_expert_weights = fc2_expert_weights.view(torch.int64)
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            weight_global_scale,
            quant_info.w2_weight_scale.view(torch.int32),
            weight_global_scale,
        ]
    elif use_wfp4afp8_humming:
        assert w13_humming_residual_scale is not None
        assert w2_humming_residual_scale is not None
        assert humming_fc2_act_scale is not None
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            w13_humming_residual_scale,
            humming_fc2_act_scale,
            quant_info.w2_weight_scale.view(torch.int32),
            w2_humming_residual_scale,
        ]
    else:
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            quant_info.w2_weight_scale.view(torch.int32),
        ]

    out_hidden = padded_hidden if do_pad else origin_hidden
    output_dtype = torch.bfloat16
    # FlashInfer 0.6.17 intentionally reverted the Humming API. Do not pass the
    # new keyword at all on the existing W4A16/MXFP8 paths, so those paths keep
    # working with SGLang's currently pinned release.
    humming_kwargs = {"use_wfp4afp8_humming": True} if use_wfp4afp8_humming else {}
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)

    flashinfer_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc2_expert_weights=fc2_expert_weights,
        output_dtype=output_dtype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        fc1_expert_biases=quant_info.w13_bias,
        fc2_expert_biases=quant_info.w2_bias,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        tp_size=runner_config.moe_tp_size,
        tp_rank=runner_config.moe_tp_rank,
        ep_size=runner_config.moe_ep_size,
        ep_rank=runner_config.moe_ep_rank,
        use_w4_group_scaling=not use_mxfp8_act_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        activation_type=(
            ActivationType.SwigluStep
            if quant_info.use_swiglu_step
            else ActivationType.Swiglu
        ),
        tune_max_num_tokens=min(next_power_of_2(x.shape[0]), _MOE_TUNE_MAX_TOKENS_CAP),
        output=out,
        **_fused_finalize_kwargs(),
        **humming_kwargs,
    )

    if do_pad:
        out = out[:, :origin_hidden].contiguous()

    return StandardCombineInput(hidden_states=out)
