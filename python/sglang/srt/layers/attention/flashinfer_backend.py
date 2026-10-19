from __future__ import annotations

from sglang.srt.runtime_context import (
    get_buffer,
    get_exec,
    get_parallel,
    get_platform,
)

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import json
import logging
import os
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.utils import (
    assert_buffer_fits,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccessKind,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    is_ccd_capturing,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
    resolve_draft_decode_window,
)
from sglang.srt.utils import (
    get_cuda_graph_max_batch_size,
    get_int_env_var,
    is_flashinfer_available,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
    from flashinfer.cascade import merge_state

    from sglang.kernels.ops.attention.merge_state import merge_state_triton

    # FlashInfer's MergeState CUDA kernel uses blockDim = (head_dim/vec_size, num_heads).
    # When num_heads is large (e.g. with DP attention where attention_tp_size=1), the
    # total threads per block can exceed CUDA's limit of 1024 and the kernel launch fails
    # with `invalid configuration argument`. Fall back to the in-tree Triton implementation,
    # which uses (token, head) as the launch grid and is therefore unaffected.
    _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK = 1024

    def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:
        # Mirrors flashinfer's vec_size selection in include/flashinfer/attention/cascade.cuh.
        vec_size = max(16 // element_size, head_dim // 32)
        bdx = head_dim // vec_size
        if bdx <= 0:
            return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx

    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)


def _maybe_publish_draft_ring_locs(swa_kv_pool, forward_batch) -> None:
    """Publish (req, pos) -> ring slot for the locs THIS forward will write.

    A ring-pooled draft owns its full->SWA mapping, rewritten by the DSpark
    injector -- but the injector runs AFTER a forward, while the translate
    calls below run BEFORE it, on out_cache_loc entries nobody has published
    yet. Without this, every fresh write loc resolves to the -1 sentinel and
    the KV store kernel dies on its index assert (or, worse, one rank dies
    and its TP peers hang in the next collective).

    Same slot formula as the injector's _maybe_publish_ring_mapping
    (slot = req * ring + pos % ring), so the injector's later publish of the
    same locs is idempotent, and its window-gating (-1 for tokens older than
    the window) still lands afterwards for future READERS. Fresh writes are
    by definition the newest tokens of their request, so no gating here.

    On any layout this function cannot attribute (ragged shapes it does not
    recognize), it publishes nothing -- the -1 sentinel then fails loudly in
    the kernel, which is the designed failure mode, rather than guessing an
    attribution and corrupting a live slot silently.
    """
    ring = getattr(swa_kv_pool, "draft_swa_ring_stride", None)
    mapping = getattr(swa_kv_pool, "full_to_swa_index_mapping", None)
    if ring is None or mapping is None:
        return
    locs = forward_batch.out_cache_loc
    reqs = forward_batch.req_pool_indices
    pos = forward_batch.positions
    if locs is None or reqs is None or pos is None:
        return
    n_tok = int(locs.shape[0])
    n_req = int(reqs.shape[0])
    if n_tok == 0 or n_req == 0 or pos.shape[0] < n_tok:
        return
    reqs = reqs.to(torch.int64)
    window_keep = None
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and forward_batch.extend_seq_lens is not None
    ):
        ext = forward_batch.extend_seq_lens.to(torch.int64)
        per_tok_req = torch.repeat_interleave(reqs, ext)
        if per_tok_req.shape[0] != n_tok:
            return
        # Window-gate, mirroring the injector: a chunk longer than the ring
        # stride writes positions p and p+stride into the SAME slot in one
        # kernel call, and which write wins is undefined -- a stale token
        # could survive in a live slot. Tokens already outside the window at
        # the chunk's end are never read from the cache again (intra-chunk
        # attention uses the in-batch K/V, not the ring), so leave them
        # unpublished; their -1 rows are dropped by SWAKVPool's skip filter.
        window = getattr(swa_kv_pool, "draft_swa_window", None)
        if window is not None and forward_batch.seq_lens is not None:
            final = torch.repeat_interleave(
                forward_batch.seq_lens.to(torch.int64) - 1, ext
            )
            if final.shape[0] == n_tok:
                window_keep = pos.to(torch.int64)[:n_tok] > final - int(window)
    elif n_tok % n_req == 0:
        per_tok_req = reqs.repeat_interleave(n_tok // n_req)
    else:
        # Graph-bucket padding: req rows padded at the tail while
        # out_cache_loc keeps only real rows (see the InnerQ lagged-consume
        # trim for the same layout). Recover the width from spec_info.
        width = int(getattr(forward_batch.spec_info, "draft_token_num", 0) or 0)
        if width <= 0 or n_tok % width != 0 or n_tok // width > n_req:
            return
        per_tok_req = reqs[: n_tok // width].repeat_interleave(width)
    slot = per_tok_req * int(ring) + pos.to(torch.int64)[:n_tok] % int(ring)
    # Publish only provably-in-range rows. At runtime every row passes: req
    # pool indices are < max_running_requests, which is exactly the ring's
    # row count, and locs come from the allocator. But this helper also runs
    # on CUDA-graph capture-WARMUP batches (eager passes, so no stream-
    # capture flag distinguishes them) whose req/pos/loc buffers hold
    # arbitrary values -- publishing slots computed from those poisons the
    # mapping that the captured write then reads back (boots 11/13, capture-
    # phase store_kvcache asserts). Filtering, rather than a capture flag,
    # keeps one invariant for every phase: nothing out-of-ring ever enters
    # the mapping.
    swa_size = getattr(getattr(swa_kv_pool, "swa_kv_pool", None), "size", None)
    locs64 = locs.to(torch.int64)
    ok = (slot >= 0) & (locs64 >= 0) & (locs64 < mapping.shape[0] - 1)
    if window_keep is not None:
        ok &= window_keep
    if swa_size is not None:
        ok &= slot < int(swa_size)
    if not bool(ok.all()):
        locs64 = locs64[ok]
        slot = slot[ok]
    mapping[locs64] = slot


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


def _static_verify_host_kv_lens(
    *,
    spec_info: SpecInput,
    seq_lens_cpu_i32: torch.Tensor,
    num_tokens_per_req: int,
) -> torch.Tensor:
    """Resolve effective host K lengths for a uniform fast verify plan.

    Live DFlash target batches retain ``live_seq_lens_cpu`` and expose
    prefix+verify in ``seq_lens_cpu``. Synthetic capture/idle inputs have no
    live mirror and expose only the committed prefix, so repeated captures
    that reuse an already-fast wrapper must add the uniform verify width.
    Other speculative inputs keep the historical interpretation unchanged.
    """
    if spec_info.spec_input_type != SpecInputType.DFLASH_VERIFY:
        return seq_lens_cpu_i32
    live_prefix = getattr(spec_info, "live_seq_lens_cpu", None)
    if live_prefix is not None:
        return seq_lens_cpu_i32
    draft_token = getattr(spec_info, "draft_token", None)
    # draft_token is a required (non-Optional) DFlashVerifyInput field, so it
    # is never None -- DSPARK's reused draft-block-proposal template
    # (make_draft_block_spec_info) sets it to an empty placeholder tensor
    # rather than omitting it. Check content, not identity, or every such
    # synthetic template wrongly takes the "live but broken" branch below.
    if draft_token is not None and draft_token.numel() > 0:
        raise RuntimeError(
            "live DFlash fast verify plan is missing committed " "live_seq_lens_cpu"
        )
    return seq_lens_cpu_i32 + int(num_tokens_per_req)


def _ragged_verify_host_plan_metadata(
    *,
    spec_info: SpecInput,
    seq_lens_cpu: torch.Tensor,
    padded_bs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact host mirrors for a compact target-verify plan.

    Device metadata assigns at most ``draft_token_num`` rows to each captured
    request slot and leaves the remaining token-bucket tail as ghosts. The
    fast plan must see that same variable-width Q layout; rebuilding a
    uniform-width indptr silently plans different rows.

    Live ``seq_lens_cpu`` is already prefix+verify (DSpark extends it before
    ForwardBatch creation). Padding slots still hold the graph sentinel
    prefix, so their synthetic capped Q length is added below.
    """
    layout = getattr(spec_info, "ragged_verify_layout", None)
    if layout is None:
        raise ValueError("ragged host plan metadata requires a ragged layout")
    raw_bs = int(layout.bs)
    if padded_bs < raw_bs:
        raise ValueError(
            f"captured FlashInfer slots {padded_bs} < ragged batch {raw_bs}"
        )

    live_prefix = getattr(spec_info, "live_seq_lens_cpu", None)
    if live_prefix is not None and int(live_prefix.numel()) != raw_bs:
        raise RuntimeError(
            "compact FlashInfer graph replay has mismatched committed host "
            f"lengths: {int(live_prefix.numel())} for ragged bs={raw_bs}"
        )

    if layout.verify_lens_cpu is not None:
        raw_q_lens = [int(length) for length in layout.verify_lens_cpu]
    else:
        # Device-only layout construction deliberately avoids D2H. Recover
        # the same lengths from the retained committed host prefix and the
        # already-effective host K lengths.
        if live_prefix is None:
            raise RuntimeError(
                "compact FlashInfer graph replay requires host verify lengths: "
                "ragged layout has no verify_lens_cpu and spec_info has no "
                "matching live_seq_lens_cpu"
            )
        raw_q_lens = (
            seq_lens_cpu[:raw_bs].to(torch.int64) - live_prefix[:raw_bs].to(torch.int64)
        ).tolist()

    cap = int(spec_info.draft_token_num)
    if any(length < 0 or length > cap for length in raw_q_lens):
        raise RuntimeError(
            f"invalid compact verify lengths {raw_q_lens}; expected 0..{cap}"
        )

    q_lens = list(raw_q_lens)
    leftover = int(layout.graph_num_tokens) - sum(q_lens)
    if leftover < 0:
        raise RuntimeError(
            f"ragged verify lengths sum to {sum(q_lens)} above graph tier "
            f"{layout.graph_num_tokens}"
        )
    num_padding = padded_bs - raw_bs
    if num_padding > 0:
        base, remainder = divmod(leftover, num_padding)
        q_lens.extend(
            min(cap, base + (1 if index < remainder else 0))
            for index in range(num_padding)
        )
    elif q_lens:
        # Match padded_to_bucket(..., cap=cap): assign the leftover to the
        # last real row, then clamp it to the fixed per-request capture width.
        q_lens[-1] = min(cap, q_lens[-1] + leftover)

    q_lens_host = torch.tensor(q_lens, dtype=torch.int32, device="cpu")
    qo_indptr_host = torch.zeros(padded_bs + 1, dtype=torch.int32, device="cpu")
    qo_indptr_host[1:] = torch.cumsum(q_lens_host, dim=0)

    effective_kv_lens = seq_lens_cpu[:padded_bs].to(torch.int32).clone()
    if raw_bs:
        _draft_token = getattr(spec_info, "draft_token", None)
        if live_prefix is not None:
            raw_prefix = live_prefix[:raw_bs].to(torch.int32)
        elif _draft_token is None or _draft_token.numel() == 0:
            # Repeated seqlen-tier/variant captures reuse the wrapper after
            # begin_forward has become fast. Their synthetic spec has no live
            # prefix mirror and capture seq_lens_cpu is still the committed K
            # prefix, not prefix+Q. draft_token is a required (non-Optional)
            # DFlashVerifyInput field, so a synthetic/reused template (e.g.
            # DSPARK's draft-block-proposal spec_info) sets it to an empty
            # placeholder tensor rather than omitting it -- check content,
            # not identity.
            raw_prefix = effective_kv_lens[:raw_bs]
        else:
            raise RuntimeError(
                "live compact DFlash fast verify plan is missing committed "
                "live_seq_lens_cpu"
            )
        # The capped padded layout can assign graph-tier tail rows to the last
        # real request. Its K length must grow with that padded Q length, not
        # remain at the effective length of only the original live queries.
        effective_kv_lens[:raw_bs] = raw_prefix + q_lens_host[:raw_bs]
    if num_padding > 0:
        effective_kv_lens[raw_bs:] += q_lens_host[raw_bs:]
    return qo_indptr_host, q_lens_host, effective_kv_lens


def _zero_ragged_verify_ghost_rows(
    output: torch.Tensor, qo_indptr_device: torch.Tensor
) -> torch.Tensor:
    """Zero compact-token bucket rows not assigned to any request.

    FlashInfer sizes its returned tensor from ``q`` rather than from the final
    qo indptr.  A capped compact layout can therefore leave a tail of output
    rows that no attention CTA writes.  Those rows still flow through the
    following dense layers/MoE in the captured model body, so make them
    deterministic without a host read of the graph-live indptr.
    """
    live_rows = qo_indptr_device[-1].to(torch.int64)
    row_is_live = torch.arange(output.shape[0], device=output.device) < live_rows
    row_is_live = row_is_live.view(-1, *([1] * (output.ndim - 1)))
    return torch.where(row_is_live, output, output.new_zeros(()))


@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    prefix_len_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_len: int = 0
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """Check if multi-item scoring is enabled."""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None


# Reuse this workspace buffer across all flashinfer wrappers

# Safety margin on the computed split-kv worst case for the dedicated
# full-CG prefill workspace (absorbs allocator alignment and minor
# flashinfer sizing drift across versions). Sizing logic lives in
# FlashInferAttnBackend._full_cg_prefill_workspace_bytes.
FULL_CG_PREFILL_WORKSPACE_MARGIN = 1.25

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None

# None = unresolved; resolved once in fast_prefill_plan from the compiled
# module's arity (FlashInfer <= 0.6.14 lacks the trailing uniform_q_len).
_FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN = None

# One plan side stream per device, shared by every wrapper's side-stream plan.
_PLAN_SIDE_STREAMS: Dict[int, torch.cuda.Stream] = {}
# Per-wrapper side-stream plan state, keyed weakly so wrapper lifetime rules it.
_PLAN_SIDE_STATES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class _PlanScratchSlot:
    """One scratch int-workspace + pinned pair with its reuse-hazard events."""

    def __init__(self, int_workspace: torch.Tensor, pin_workspace: torch.Tensor):
        self.scratch = torch.empty_like(int_workspace)
        self.pin = torch.empty(
            pin_workspace.shape, dtype=pin_workspace.dtype, pin_memory=True
        )
        # Fresh events report complete, so first use waits on nothing.
        self.h2d_done = torch.cuda.Event()
        self.d2d_done = torch.cuda.Event()


class _PlanSideState:
    def __init__(self, wrapper):
        self.slots = [
            _PlanScratchSlot(
                wrapper._int_workspace_buffer,
                wrapper._pin_memory_int_workspace_buffer,
            )
            for _ in range(2)
        ]
        self.toggle = 0


def _plan_side_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _PLAN_SIDE_STREAMS.get(device.index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _PLAN_SIDE_STREAMS[device.index] = stream
    return stream


def _side_stream_plan(self, args: list):
    """Run the compiled plan without blocking the host on a deep main stream.

    Under GPU confidential computing, pinned host memory is managed-backed, so
    the cudaMemcpyAsync inside the compiled plan blocks the HOST until the
    target stream can execute it. On the main stream that means draining every
    queued kernel -- including DP-attention allgathers gated on a peer rank's
    prefill chunk -- before this rank can even enqueue its own tick (py-spy
    2026-08-26: 87% of three decode ranks' event loops at that call under
    mixed prefill+decode load).

    Fix: give plan a private scratch int-workspace + pinned pair and issue it
    on a shallow per-device side stream (the tvm-ffi env stream must be set
    explicitly -- `torch.cuda.stream(...)` alone does not reach it, see
    cc_conditional_decode.py's CCD note), then hand the schedule to the real
    graph-referenced workspace with a stream-ordered device-to-device copy on
    the main stream: after the previous replay's reads, before the coming
    one.

    The side stream must stay IDLE, never event-waiting: a pinned-source H2D
    blocks the host until the target stream can execute it (measured on this
    CC VM; D2D and event records enqueue normally). Both reuse hazards are
    therefore closed with host-side event syncs that are complete by
    construction: the slot's previous H2D ran on the idle side stream at its
    own plan call, and the main-stream D2D read of this scratch is two of
    this wrapper's plans (= two ticks) old -- resolve_seq_lens forces the
    intervening tick's main-stream work to finish before the scheduler can
    reach this plan again.
    """
    import tvm_ffi

    state = _PLAN_SIDE_STATES.get(self)
    if state is None:
        state = _PlanSideState(self)
        _PLAN_SIDE_STATES[self] = state
    slot = state.slots[state.toggle]
    state.toggle ^= 1

    device = self._int_workspace_buffer.device
    side = _plan_side_stream(device)
    # Pinned reuse: this slot's previous H2D must have executed before the
    # compiled plan's CPU pass rewrites the pinned buffer.
    slot.h2d_done.synchronize()
    # Scratch reuse: the main stream's D2D read of this scratch (two plans
    # ago) must precede the new H2D into it. A GPU-side side-stream wait
    # would re-block the host at the H2D itself -- see the docstring.
    slot.d2d_done.synchronize()
    args = list(args)
    args[1] = slot.scratch
    args[2] = slot.pin
    with torch.cuda.stream(side):
        with tvm_ffi.use_torch_stream():
            plan_info = _call_compiled_plan(self, args)
        slot.h2d_done.record(side)
    main = torch.cuda.current_stream(device)
    main.wait_event(slot.h2d_done)
    self._int_workspace_buffer.copy_(slot.scratch, non_blocking=True)
    slot.d2d_done.record(main)
    return plan_info


def _call_compiled_plan(self, args: list):
    """Invoke the compiled plan, resolving the uniform_q_len arity once."""
    global _FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN
    if _FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN is False:
        args = args[:-1]
    try:
        plan_info = self._cached_module.plan(*args)
        if _FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN is None:
            _FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN = True
        return plan_info
    except TypeError:
        if _FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN is not None:
            raise
        _FLASHINFER_PLAN_TAKES_UNIFORM_Q_LEN = False
        return self._cached_module.plan(*args[:-1])


def fast_prefill_plan(
    self,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    page_size: int,
    head_dim_vo: Optional[int] = None,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_left: int = -1,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    prefix_len_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_len: int = 0,
    max_item_len_ptr: Optional[torch.Tensor] = None,
    # Required host-known metadata: lets us skip the per-replay device-to-host
    # copies upstream plan() always issues. Keyword-only with no default so a
    # caller that forgets them fails at the call boundary, not with a cryptic
    # None crash deeper in.
    *,
    qo_indptr_host: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_lens_host: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
) -> None:
    """Sync-free ``BatchPrefillWithPagedKVCacheWrapper.plan`` for the EAGLE
    draft-extend CUDA graph (FlashInfer fa2, cuda-graph mode only).

    Upstream plan() always does qo/paged_kv/last_page_len ``.to("cpu")`` to build
    its host scheduling metadata, a blocking D2H that drains the GPU queue every
    replay. The caller passes host-known qo/kv layout in, so we call the underlying
    ``_cached_module.plan`` directly with no readback; the ``_plan_info`` produced
    is identical to plan()'s.
    """
    assert self.is_cuda_graph_enabled, "fast_prefill_plan is cuda-graph only"
    assert (
        getattr(self, "_backend", None) == "fa2"
    ), "fast_prefill_plan supports the fa2 backend only"
    assert (
        getattr(self, "_cached_module", None) is not None
    ), "fast_prefill_plan requires _cached_module from a prior real plan() (capture)"

    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    batch_size = len(paged_kv_last_page_len)

    total_num_rows = int(qo_indptr_host[-1])
    self._qo_indptr_last = total_num_rows
    self._max_q_len = max_q_len
    self._max_kv_len = max_kv_len

    if self._max_total_num_rows is None:
        self._max_total_num_rows = total_num_rows

    self._batch_size = batch_size
    self._num_qo_heads = num_qo_heads
    self._num_kv_heads = num_kv_heads
    self._prefix_len_ptr = prefix_len_ptr
    self._token_pos_in_items_ptr = token_pos_in_items_ptr
    self._token_pos_in_items_len = token_pos_in_items_len
    self._max_item_len_ptr = max_item_len_ptr

    # Refresh the cuda-graph input buffers (device-to-device, non-blocking).
    self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
    self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
    self._paged_kv_last_page_len_buf.copy_(
        paged_kv_last_page_len, non_blocking=non_blocking
    )
    self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )

    self._cached_q_data_type = q_data_type
    self._cached_kv_data_type = (
        kv_data_type if kv_data_type is not None else q_data_type
    )
    self._cached_o_data_type = o_data_type
    self._block_tables = None

    # CC conditional decode (CCD) captures a whole chunk's worth of ticks as
    # ONE CUDA graph body; `_cached_module.plan()` below does its own
    # internal host->device copy from `_pin_memory_int_workspace_buffer`
    # (a FlashInfer-owned pinned CPU buffer). Under GPU confidential
    # computing, pinned host memory is transparently backed by CUDA Managed
    # memory (tdx-checkpoints/085/086), and a captured memcpy node sourced
    # from Managed memory fails cudaGraphInstantiate
    # (cudaGraphInstantiateMultipleDevicesNotSupported -- see
    # tdx-checkpoints/098). plan()'s output only encodes a schedule for the
    # batch shape / upper-bound row count (`_max_total_num_rows`, already
    # cached across calls above); the actual per-tick data (qo/paged_kv
    # indptr, kv indices, last-page-len) is supplied via the device-to-device
    # buffer refreshes just above, which run unconditionally every call.
    # CCD's own priming tick (prime_and_capture, eager, immediately before
    # capture begins) already calls this same function once for this exact
    # wrapper+shape with a valid plan_info as a result -- skip the redundant,
    # uncapturable replan here and reuse it verbatim.
    if is_ccd_capturing():
        assert self._plan_info is not None, (
            "fast_prefill_plan: no prior plan_info available under CCD "
            "capture -- the eager priming tick must run (and populate "
            "_plan_info) before body_fn capture begins."
        )
    else:
        args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_lens_host,
            self._max_total_num_rows or total_num_rows,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim_qk,
            head_dim_vo,
            causal,
            window_left,
            fixed_split_size if fixed_split_size is not None else -1,
            False,  # disable_split_kv
            0,  # num_colocated_ctas
            0,  # uniform_q_len (absent in FlashInfer <= 0.6.14 -- see below)
        ]
        # FlashInfer <= 0.6.14's compiled plan() takes 19 args (through
        # num_colocated_ctas); uniform_q_len was appended upstream. The
        # tvm_ffi function exposes no Python signature to inspect, so the
        # arity is resolved once from the TypeError and cached process-wide
        # (_call_compiled_plan); omitting the trailing hint on old builds is
        # the old behavior.
        if (
            not envs.SGLANG_OPT_DISABLE_FLASHINFER_SIDE_STREAM_PLAN.get()
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._plan_info = _side_stream_plan(self, args)
        else:
            self._plan_info = _call_compiled_plan(self, args)


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    # kv_indptr/qo_indptr are preallocated at (req pool + 1); an extend batch
    # can never carry more seqs than the pool.
    extend_dummy_seqs_capped_by_req_pool: bool = True

    @property
    def supports_ragged_verify_graph(self) -> bool:
        # Compact target verify is implemented for the ordinary full-attn
        # wrapper and the decoder-only SWA pair. Encoder cross-attention has a
        # different K layout and remains fail-closed.
        return self.dispatch_reason is not WrapperDispatch.CROSS_ATTENTION

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.kv_index_translator = model_runner.kv_index_translator
        self._swa_kv_pool: Optional[BaseSWAKVPool] = self._resolve_swa_kv_pool(
            model_runner
        )
        self.use_sliding_window_kv_pool = self._swa_kv_pool is not None
        self.enable_mis = get_exec().features.enable_mis

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        self.kv_cache_quant_method = self.token_to_kv_pool.get_kv_cache_quant_method()
        self.prefill_kv_access = self.kv_cache_quant_method.resolve_attention_access(
            "prefill", "flashinfer"
        )
        self.decode_kv_access = self.kv_cache_quant_method.resolve_attention_access(
            "decode", "flashinfer"
        )
        prefill_backend = model_runner.prefill_attention_backend_str
        decode_backend = model_runner.decode_attention_backend_str
        if self.__class__ is FlashInferAttnBackend:
            if prefill_backend == "flashinfer":
                self._check_kv_attention_access("prefill", self.prefill_kv_access)
            if decode_backend == "flashinfer":
                self._check_kv_attention_access("decode", self.decode_kv_access)

        self.prefill_uses_dequant_workspace = (
            self.prefill_kv_access is not None
            and self.prefill_kv_access.kind
            == KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
        )
        self.decode_uses_dequant_workspace = (
            self.decode_kv_access is not None
            and self.decode_kv_access.kind
            == KVCacheAttentionAccessKind.DEQUANT_WORKSPACE
        )
        self.is_nvfp4_kvcache = any(
            access is not None and access.scale_recipe == "nvfp4"
            for access in (self.prefill_kv_access, self.decode_kv_access)
        )
        self.dq_page_table = None
        self.dq_paged_kernel_lens = None
        self.cpu_req_pool_indices = None
        # FP4 fake-quant prefill/decode exposes an FP8 workspace to FlashInfer.
        self.flashinfer_kv_cache_dtype = (
            torch.float8_e4m3fn
            if (
                self.prefill_uses_dequant_workspace
                or self.decode_uses_dequant_workspace
            )
            else model_runner.kv_cache_dtype
        )

        # Parse constants
        # kv_buffer_dtype, not kv_cache_dtype: FP4 pools dequantize to bf16 in
        # get_kv_buffer, so that's the dtype flashinfer actually receives --
        # planning around the configured (packed) dtype makes flashinfer's
        # plan/runtime dtype check fail on the first real forward.
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.token_to_kv_pool.kv_buffer_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
        )
        self.max_context_len = model_runner.model_config.context_len
        self.page_size = model_runner.page_size
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            get_exec().deterministic.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()

        # Allocate buffers
        # different from flashinfer zero_init_global_workspace_buffer
        global_workspace_buffer = get_buffer(
            "flashinfer_workspace",
            lambda: torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            ),
        )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = get_cuda_graph_max_batch_size(model_runner.req_to_token_pool.size)
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if get_platform().is_sm100:
            fmha_backend = "fa2"
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on SM100 GPUs. The
            # current FlashInfer SM100 CUTLASS FMHA dispatch only instantiates
            # 64x64, 128x128, and 192x128 head dimensions. Keep unsupported
            # shapes (for example Qwen3.5's 256x256) on the FA2 fallback.
            cutlass_supported_head_dims = {
                (64, 64),
                (128, 128),
                (192, 128),
            }
            head_dims = (
                model_runner.model_config.head_dim,
                model_runner.model_config.v_head_dim,
            )
            if (
                head_dims in cutlass_supported_head_dims
                and not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
            ):
                fmha_backend = "cutlass"
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend=self.decode_backend,
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend
        # Plain EXTEND under full prefill CUDA graph: one wrapper set
        # shared across all captured num_tokens buckets (bs fixed at 1).
        # Created lazily on first capture in _prepare_cuda_graph_metadata.
        self.full_cg_prefill_wrappers: Optional[
            List[BatchPrefillWithPagedKVCacheWrapper]
        ] = None

    def _check_kv_attention_access(self, phase: str, access) -> None:
        if access is not None:
            return
        method_name = getattr(self.kv_cache_quant_method, "name", "unknown")
        available = self.kv_cache_quant_method.describe_attention_accesses(phase)
        raise ValueError(
            f"KV cache method {method_name!r} does not support {phase} with "
            f"flashinfer attention backend. Available {phase} accesses: {available}."
        )

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[BaseSWAKVPool]:
        """Return the SWA KV pool to translate against, or None for non-SWA models.

        EAGLE-like draft workers share the target allocator for token bookkeeping,
        but own a separate draft KV pool. Do not use the target allocator's SWA
        mapping for that draft pool. FROZEN_KV MTP is the exception: its draft
        path reads target KV directly, so it still needs the allocator pool when
        the active pool is not SWA.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, BaseSWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        return kvcache if isinstance(kvcache, BaseSWAKVPool) else None

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """Process multi-item scoring tensors for FlashInfer attention.

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        precomputed_indices = forward_batch.multi_item_delimiter_indices
        if precomputed_indices is None:
            return MultiItemScoringParams()

        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0
        device = forward_batch.input_ids.device

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            delimiter_indices_cpu = precomputed_indices[i]
            if len(delimiter_indices_cpu) == 0:
                seq_start = seq_end
                continue

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
            prefix_len = first_delim + (
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0
            )
            prefix_len_ptr.append(prefix_len)

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            suffix_len = seq_len - first_delim
            relative_positions = delimiter_indices - first_delim

            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
            bucket_idx = torch.searchsorted(
                relative_positions, suffix_range, right=True
            )
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
            pos_within_item = suffix_range - last_delim

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

            forward_batch.positions[seq_start + first_delim : seq_end] = (
                prefix_len + pos_within_item - 1
            )

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)

        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
                req_pool_indices=req_pool_indices,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[
                    self._verify_graph_key(bs, spec_info)
                ],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_dllm_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=not self.use_paged,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        elif forward_mode.is_draft_extend_v2():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_extend():
            # Plain EXTEND under full prefill CUDA graph. plan() runs
            # out-of-graph against capture-stable wrappers; captured kernels
            # read the refreshed state at replay. Must stay below the
            # target-verify / draft-extend / dllm branches (also is_extend()).
            # Split-kv must stay on — its block_valid_mask is the only
            # early-exit for the captured fixed grid's padded/stale tiles.
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=forward_batch.extend_prefix_lens[:bs],
                prefill_wrappers=self.full_cg_prefill_wrappers,
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        else:
            raise ValueError("Invalid forward mode")

        if (
            in_capture
            and forward_mode.is_decode_or_idle()
            and not envs.SGLANG_DISABLE_FAST_ATTENTION_PLAN.get()
        ):
            # fast_decode_plan needs _cached_module from the initial begin_forward
            # above, so install it only after that first plan has run.
            for w in self.decode_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_decode_plan, w)

        if (
            in_capture
            and not envs.SGLANG_DISABLE_FAST_ATTENTION_PLAN.get()
            and forward_mode.is_draft_extend_v2()
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA/cross-attn keep the plain plan().
            and self.dispatch_reason is None
        ):
            # Like decode: swap in fast_prefill_plan for replay, after the real
            # plan() above set up _cached_module (host metadata supplied per-replay
            # in call_begin_forward).
            for w in self.draft_extend_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        if (
            in_capture
            and not envs.SGLANG_DISABLE_FAST_ATTENTION_PLAN.get()
            and forward_mode.is_target_verify()
            and spec_info is not None
            and spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY
            and getattr(spec_info, "custom_mask", None) is None
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA is admitted for DSpark's draft model (1024-token window) when
            # the call_begin_forward fast branch's host-known KV layout
            # precondition (prefix_lens == seq_lens, non-ragged) holds --
            # cross-attn still keeps the plain plan().
            and self.dispatch_reason in (None, WrapperDispatch.SLIDING_WINDOW)
        ):
            # DFLASH target-verify replays are shape-static per
            # (bs, draft_token_num): qo_indptr is a constant arange stride of
            # num_tokens_per_req, and the batch carries seq_lens_cpu =
            # prefix + draft_token_num (dspark_draft._run_forward /
            # dspark_verify.run_non_compact / dflash_worker_v2 all add the
            # verify window host-side), which equals the device kv length
            # generate_attn_arg_prefill produces. The host-kwargs assembly in
            # call_begin_forward therefore applies verbatim; installing the
            # sync-free plan removes three blocking .to("cpu") reads per
            # replay that otherwise stall the CPU behind the in-flight graph.
            # EAGLE target-verify keeps the plain plan(): its spec input is
            # not DFLASH_VERIFY, and this branch keys off the capture-time
            # spec_info of these per-bs wrappers. CC conditional decode's
            # captured WHILE-loop body relies on the same path (DFlashVerify-
            # Input.prepare_for_verify substitutes draft_input.nxt_kv_lens_cpu
            # for the host-known upper bound under CC conditional decode,
            # dflash_worker_v2.py ~line 1700).
            for w in self.prefill_cuda_graph_metadata[
                self._verify_graph_key(bs, spec_info)
            ]:
                w.begin_forward = partial(fast_prefill_plan, w)

        # Refill the SWA write-target buffer from the live out_cache_loc before
        # replay (bound onto the metadata at capture below).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            # Replay only, never capture: a capture-phase forward_batch holds
            # dummy buffers (req_pool_indices/positions are arbitrary), and
            # publishing slots computed from them lands out-of-ring indices in
            # the mapping that the captured write then reads back (boot-11
            # capture assert). Capture in-range-ness is provided by the draft
            # worker's _ring_mapping_capture_safe fill(0) instead.
            if not in_capture:
                _maybe_publish_draft_ring_locs(self._swa_kv_pool, forward_batch)
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            if in_capture:
                self.cuda_graph_swa_out_cache_loc[:n].zero_()
            else:
                self.cuda_graph_swa_out_cache_loc[:n].copy_(
                    self.kv_index_translator.sliding_window_write_loc_for(
                        forward_batch.out_cache_loc
                    )
                )
            if in_capture:
                self.forward_metadata.swa_out_cache_loc = (
                    self.cuda_graph_swa_out_cache_loc[:n]
                )

    def _prepare_dequant_workspace_metadata_for_extend(
        self, forward_batch: ForwardBatch, use_ragged: bool = False
    ):
        """Prepare FlashInfer metadata for an FP4 dequant workspace.

        Some FP4 recipes store packed KV but expose an FP8 workspace to
        FlashInfer prefill. This builds the workspace page table, exact paged
        lengths, and CPU request ids needed to populate that workspace before
        the prefill kernel runs.
        """
        self.dq_page_table = None
        self.dq_paged_kernel_lens = None
        self.cpu_req_pool_indices = None
        if not (
            self.prefill_uses_dequant_workspace
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            return

        # Ragged prefill handles current-chunk K/V with raw tensors, so the
        # paged side only contains cached prefix lengths. Non-ragged prefill
        # uses the dequant workspace for prefix + current chunk, so it needs
        # full sequence lengths. These CPU length containers may arrive as
        # Python lists or CPU tensors depending on the metadata builder.
        paged_seq_lens_cpu = (
            forward_batch.extend_prefix_lens_cpu
            if use_ragged
            else forward_batch.seq_lens_cpu
        )
        raw_paged_seq_lens = (
            paged_seq_lens_cpu
            if isinstance(paged_seq_lens_cpu, list)
            else paged_seq_lens_cpu.tolist()
        )
        paged_seq_lens = [
            int(seq_len.item()) if isinstance(seq_len, torch.Tensor) else int(seq_len)
            for seq_len in raw_paged_seq_lens
        ]
        if sum(paged_seq_lens) <= 0:
            self.cpu_req_pool_indices = forward_batch.req_pool_indices.to(
                "cpu", non_blocking=True
            )
            return

        # dq_buffer layout is page-aligned: each request occupies
        # ceil(seq_len/page_size)*page_size slots, starting after a page_size
        # dummy prefix. dq_page_table maps only actual token positions and skips
        # padding gaps; dq_paged_kernel_lens stores real lengths so FlashInfer
        # causal offsets use seq_len - q_len, not page_align(seq_len) - q_len.
        seq_lens_with_scratch = paged_seq_lens + [256]
        starts = []
        next_start = self.page_size
        for seq_len in seq_lens_with_scratch:
            starts.append(next_start)
            padded_len = (
                (seq_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            next_start += padded_len

        device = forward_batch.req_pool_indices.device
        indices = [
            torch.arange(start, start + seq_len, device=device, dtype=torch.int32)
            for start, seq_len in zip(starts, seq_lens_with_scratch)
            if seq_len > 0
        ]
        self.dq_page_table = torch.cat(indices) if indices else None
        self.dq_paged_kernel_lens = torch.tensor(
            paged_seq_lens,
            dtype=torch.int32,
            device=device,
        )
        self.cpu_req_pool_indices = forward_batch.req_pool_indices.to(
            "cpu", non_blocking=True
        )

    def _kv_write_scales(self, layer: RadixAttention):
        if self.kv_cache_quant_method.needs_global_scale():
            return None, None
        return layer.k_scale, layer.v_scale

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            _maybe_publish_draft_ring_locs(self._swa_kv_pool, forward_batch)
            swa_out_cache_loc = self.kv_index_translator.sliding_window_write_loc_for(
                forward_batch.out_cache_loc
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
                req_pool_indices=forward_batch.req_pool_indices,
            )
            self.forward_metadata = DecodeMetadata(
                self.decode_wrappers, swa_out_cache_loc=swa_out_cache_loc
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify,
                False,
                False,
                swa_out_cache_loc=swa_out_cache_loc,
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.enable_mis:
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = (
                    not self.enable_deterministic
                    and not is_in_tc_piecewise_cuda_graph()
                    and not self.use_paged
                )
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.enable_mis:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self._prepare_dequant_workspace_metadata_for_extend(
                forward_batch, use_ragged
            )

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                custom_kv_indices=self.dq_page_table,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
                swa_out_cache_loc=swa_out_cache_loc,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            # TOKENS, not pages: no page_size divisor, so this is
            # max_num_tokens * max_context_len * 4B and grows linearly with
            # context, then is cloned per wrapper below. max_num_tokens is
            # max_bs * captured_req_width (decode_cuda_graph_runner), so a
            # spec-decode ladder multiplies it: 128 * 8 = 1024 with DSPARK,
            # giving 1 GiB per wrapper at 256K context and 4 GiB at 1M.
            # A 4.00 GiB request here is what raised OutOfMemoryError on a
            # 266 GiB B300 during 1M capture (2026-09-04). Budget it before
            # raising --context-length or CUDA_GRAPH_MAX_BS.
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # SWA write-target buffer; refilled and bound onto forward_metadata in
        # init_forward_metadata_out_graph before each replay.
        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
            if self.use_sliding_window_kv_pool
            else None
        )

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:
        return [
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
            )
            for i in range(self.num_wrappers)
        ]

    def get_cuda_graph_decode_wrappers(
        self,
        *,
        bs: int,
        num_tokens: int,
    ) -> list:
        wrappers = self.decode_cuda_graph_metadata.get(bs)
        if wrappers is None:
            self._prepare_cuda_graph_metadata(
                bs,
                num_tokens,
                ForwardMode.DECODE,
                spec_info=None,
            )
            wrappers = self.decode_cuda_graph_metadata[bs]
        return wrappers

    def _create_prefill_wrappers(self, bs: int, use_custom_mask: bool = False) -> list:
        # FlashInfer's prefill wrapper decides mask mode based on whether
        # `custom_mask_buf` is initialized (not whether a custom mask is provided).
        # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
        # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
        # FlashInfer will treat the (zero) buffer as a real mask and block attention.
        wrappers = []
        for i in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    backend=self.prefill_backend,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    **extra,
                )
            )
        return wrappers

    @staticmethod
    def _full_cg_prefill_workspace_bytes(
        num_slots: int,
        max_num_tokens: int,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> int:
        """Split-kv worst-case float-workspace demand for the plain-EXTEND
        cudagraph wrappers, mirroring flashinfer's PrefillPlan sizing
        (scheduler.cuh, enable_cuda_graph=True) for the largest captured
        bucket:

          cta_tile_q = FA2DetermineCtaTileQ(max packed qo len, head_dim)
          tiles      = ceil(max_rows * gqa / cta_tile_q) + batch_size - 1
          padded     = max(2 * num_SMs / num_kv_heads, tiles)
          tmp_v      = num_qo_heads * padded * cta_tile_q * head_dim * fp32
          tmp_s      = num_qo_heads * padded * cta_tile_q * fp32

        Split-kv must stay enabled for these wrappers — its
        block_valid_mask is what lets the padded/stale tiles of the fixed
        captured grid exit early at replay; without it every replay
        re-runs capture-sized attention (measured ~6.5 ms/layer). If a
        future flashinfer outgrows the margin, plan() fails loudly at
        startup ("Increase the workspace buffer size").
        """
        gqa_group_size = num_qo_heads // num_kv_heads
        max_qo_len = (max_num_tokens - num_slots + 1) * gqa_group_size
        if max_qo_len > 64 and head_dim < 256:
            cta_tile_q = 128
        elif max_qo_len > 16:
            cta_tile_q = 64
        else:
            cta_tile_q = 16
        tiles = -(-max_num_tokens * gqa_group_size // cta_tile_q) + num_slots - 1
        num_sm = torch.cuda.get_device_properties(device).multi_processor_count
        padded_batch_size = max((2 * num_sm) // num_kv_heads, tiles)
        per_row = num_qo_heads * padded_batch_size * cta_tile_q * 4
        tmp_v = per_row * head_dim
        tmp_s = per_row
        return int((tmp_v + tmp_s) * FULL_CG_PREFILL_WORKSPACE_MARGIN)

    def _create_full_cg_prefill_wrappers(
        self, num_slots: int, max_num_tokens: int
    ) -> list:
        """Wrappers for plain EXTEND captured under a full prefill CUDA
        graph. plan() must keep its internal state at capture-stable
        addresses (use_cuda_graph=True); the decode-side cuda-graph
        wrappers permanently pin the shared workspace via their own
        plans, so these get a dedicated workspace sized from the largest
        captured bucket. The request-slot count is fixed at capture (the
        runner pads real batches up to it with zero-length sentinel
        requests); kv indices cover up to num_slots sequences of
        max_context_len.
        """
        device = self.workspace_buffer.device
        self.full_cg_prefill_req_slots = num_slots
        upd = self.indices_updater_prefill
        workspace_bytes = self._full_cg_prefill_workspace_bytes(
            num_slots,
            max_num_tokens,
            num_qo_heads=upd.num_qo_heads,
            num_kv_heads=upd.num_kv_heads,
            head_dim=upd.head_dim,
            device=device,
        )
        logger.info(
            "Full-CG prefill workspace: %.0f MB (max bucket %d tokens, "
            "%d request slots)",
            workspace_bytes / (1024 * 1024),
            max_num_tokens,
            num_slots,
        )
        self.full_cg_prefill_workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        self.full_cg_prefill_qo_indptr = [
            torch.zeros((num_slots + 1,), dtype=torch.int32, device=device)
            for _ in range(self.num_wrappers)
        ]
        self.full_cg_prefill_kv_indptr = [
            torch.zeros((num_slots + 1,), dtype=torch.int32, device=device)
            for _ in range(self.num_wrappers)
        ]
        # call_begin_forward materializes paged_kernel_lens_sum + 256
        # indices; size the fixed buffer for the worst case.
        self.full_cg_prefill_kv_indices = [
            torch.zeros(
                (num_slots * self.max_context_len + 256,),
                dtype=torch.int32,
                device=device,
            )
            for _ in range(self.num_wrappers)
        ]
        return [
            BatchPrefillWithPagedKVCacheWrapper(
                self.full_cg_prefill_workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend=self.prefill_backend,
                qo_indptr_buf=self.full_cg_prefill_qo_indptr[i],
                paged_kv_indptr_buf=self.full_cg_prefill_kv_indptr[i],
                paged_kv_indices_buf=self.full_cg_prefill_kv_indices[i],
                paged_kv_last_page_len_buf=self.kv_last_page_len[:num_slots],
            )
            for i in range(self.num_wrappers)
        ]

    def _prepare_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        # Reuse an already-captured bs's wrappers instead of reallocating
        # (mirrors the is_extend() branch's `if None` pattern below). A bs can
        # be captured more than once per session (one graph per sum bucket or
        # graph variant), and the graph-capture memory pool assumes each bs is
        # allocated exactly once ("capture large shapes first so smaller shapes
        # reuse memory"). Reallocating on a repeat capture orphans the first
        # capture's buffers while its recorded CUDA graph still needs them at
        # replay (an illegal memory access on the first replay).
        if forward_mode.is_decode_or_idle():
            if bs in self.decode_cuda_graph_metadata:
                decode_wrappers = self.decode_cuda_graph_metadata[bs]
            else:
                decode_wrappers = self._create_decode_wrappers(bs, num_tokens)
                self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            graph_key = (
                self._verify_graph_key(bs, spec_info)
                if forward_mode.is_target_verify()
                else bs
            )
            if graph_key in self.prefill_cuda_graph_metadata:
                prefill_wrappers = self.prefill_cuda_graph_metadata[graph_key]
            else:
                use_custom_mask = (
                    forward_mode.is_target_verify()
                    and spec_info is not None
                    and getattr(spec_info, "custom_mask", None) is not None
                )
                prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
                self.prefill_cuda_graph_metadata[graph_key] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(
                prefill_wrappers, forward_mode.is_dllm_extend(), False
            )
        elif forward_mode.is_draft_extend_v2():
            # Draft-extend: causal paged prefill over the full sequence (no mask).
            if bs in self.draft_extend_cuda_graph_metadata:
                prefill_wrappers = self.draft_extend_cuda_graph_metadata[bs]
            else:
                prefill_wrappers = self._create_prefill_wrappers(
                    bs, use_custom_mask=False
                )
                self.draft_extend_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        elif forward_mode.is_extend():
            if self.full_cg_prefill_wrappers is None:
                self.full_cg_prefill_wrappers = self._create_full_cg_prefill_wrappers(
                    bs, num_tokens
                )
            self.forward_metadata = PrefillMetadata(
                self.full_cg_prefill_wrappers, False, False
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    @staticmethod
    def _verify_graph_key(bs: int, spec_info: Optional[SpecInput]):
        """Request bucket for static verify; token bucket for compact verify.

        Compact token tiers can share the same captured request-slot count,
        but FlashInfer wrappers retain plan buffers sized/recorded for the Q
        tier. Sequence-length variants of the same token tier deliberately
        reuse one wrapper because their dense-attention geometry is identical.
        """
        layout = getattr(spec_info, "ragged_verify_layout", None)
        if layout is None:
            return bs
        return ("ragged", int(layout.graph_num_tokens))

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        assert not (
            self.prefill_uses_dequant_workspace and layer.is_cross_attention
        ), "FP4 dequant KV cache is not supported for cross-attention"

        # We perform dequant for chunk prefill/cache reuse.
        pool = self.token_to_kv_pool
        if self.prefill_uses_dequant_workspace:
            kv_cache = pool.get_flashinfer_dequant_workspace_kv_buffer(
                layer,
                self.req_to_token_pool.req_to_token,
                self.cpu_req_pool_indices,
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                self.page_size,
                prepare_workspace=self.dq_page_table is not None,
                use_ragged=self.forward_metadata.use_ragged,
                k_cur=k,
                v_cur=v,
            )
        else:
            kv_cache = pool.get_kv_buffer(layer.layer_id)

        # use paged attention
        if not self.forward_metadata.use_ragged:
            if k is not None and save_kv_cache:
                assert v is not None
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

            causal = (
                not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            )
            o = prefill_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                kv_cache,
                causal=causal,
                sm_scale=layer.scaling,
                # Disable sliding window attention for multi-item scoring:
                # - Sliding window could cut across item boundaries, breaking semantic coherence
                # - Multi-item sequences need full attention to properly handle delimiter tokens
                # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                #   provide more precise attention control than simple sliding windows
                # - Item-aware masking takes precedence over window-based masking
                window_left=(
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                ),
                logits_soft_cap=logits_soft_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        else:
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `self.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            if k is None and v is None:
                assert not self.prefill_uses_dequant_workspace, "KV cache must be provided for ragged attention when using FP4 dequant KV cache"
                k = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False

            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                swa_window_left = (
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                )
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    kv_cache,
                    causal=False,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )

                o, _ = _safe_merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

        o = o.view(-1, layer.tp_q_head_num * layer.head_dim)
        ragged_layout = getattr(
            getattr(forward_batch, "spec_info", None),
            "ragged_verify_layout",
            None,
        )
        if forward_batch.forward_mode.is_target_verify() and ragged_layout is not None:
            o = _zero_ragged_verify_ghost_rows(o, ragged_layout.qo_indptr_device)
        return o

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

        if self.decode_uses_dequant_workspace:
            kv_cache = (
                self.token_to_kv_pool.get_flashinfer_decode_dequant_workspace_kv_buffer(
                    layer,
                    self.req_to_token_pool.req_to_token,
                    forward_batch.req_pool_indices,
                    (
                        forward_batch.seq_lens_cpu
                        if forward_batch.seq_lens_cpu is not None
                        else forward_batch.seq_lens
                    ),
                )
            )
        else:
            kv_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_cache,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        # Plan with the max per-layer head count: FlashInfer bakes num_qo_heads
        # into the plan, and layers running more heads than planned are
        # silently corrupted. Over-planning is safe.
        self.num_qo_heads = (
            model_runner.model_config.get_max_num_attention_heads()
            // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        # See FlashInferAttnBackend.__init__: plan with the dtype the pool's
        # get_kv_buffer actually returns (bf16 for FP4 dequant-on-read pools).
        self.data_type = model_runner.token_to_kv_pool.kv_buffer_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self._swa_kv_pool = attn_backend._swa_kv_pool

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        req_pool_indices: torch.Tensor,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        req_pool_indices: torch.Tensor,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            req_pool_indices=req_pool_indices,
        )

    def update_sliding_window(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        req_pool_indices: torch.Tensor,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                req_pool_indices=req_pool_indices,
            )

    def update_cross_attention(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        req_pool_indices: torch.Tensor,
    ):
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                kv_lens_cpu = seq_lens_cpu
            else:
                # Cross-attention: attend to encoder tokens only
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()
                kv_lens_cpu = encoder_lens_cpu

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=kv_lens_cpu,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
                req_pool_indices=req_pool_indices,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
        *,
        req_pool_indices: torch.Tensor,
    ):
        # Unified SWA wrapper-0: gather from the swa canonical directly -- its
        # entries are already swa-side kernel-facing ids, so the in-place
        # full->swa translate below must not run on top of them.
        translator = self.attn_backend.kv_index_translator
        use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated
        if spec_info is None or getattr(spec_info, "kv_indptr", None) is None:
            bs = len(paged_kernel_lens)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            translator.fill_packed_read_stream(
                req_pool_indices=req_pool_indices,
                seq_lens=paged_kernel_lens,
                indptr=kv_indptr,
                total_tokens=paged_kernel_lens_sum,
                out=kv_indices,
                kv_start_idx=kv_start_idx,
                sliding_window=use_swa_source,
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool and not use_swa_source:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        wrapper_uses_fast_decode_plan = (
            hasattr(wrapper.begin_forward, "func")
            and wrapper.begin_forward.func == fast_decode_plan
        )

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        # Plan with the max per-layer head count: FlashInfer bakes num_qo_heads
        # into the plan, and layers running more heads than planned are
        # silently corrupted. Over-planning is safe.
        self.num_qo_heads = (
            model_runner.model_config.get_max_num_attention_heads()
            // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        # See FlashInferAttnBackend.__init__: plan with the dtype the pool's
        # get_kv_buffer actually returns (bf16 for FP4 dequant-on-read pools).
        self.data_type = model_runner.token_to_kv_pool.kv_buffer_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        # Kept ONLY for the spec-info branches (generate_attn_arg_prefill),
        # which are static-pool-only: unified memory asserts spec off. The
        # normal builders source from the per-batch KVIndexTable.
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
    ):
        if use_ragged:
            assert prefix_lens is not None
            paged_kernel_lens = prefix_lens
            if extend_prefix_lens_cpu is not None:
                # Host-known prefix lens; avoids a per-step D2H sync.
                paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
            else:
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
            seq_lens_cpu=seq_lens_cpu,
            custom_kv_indices=custom_kv_indices,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
    ):
        if custom_kv_indices is not None:
            raise RuntimeError(
                "NVFP4 custom KV indices are only supported by the single-wrapper FlashInfer path."
            )
        if prefix_lens is None:
            num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
            # Spec verify keeps its query block outside seq_lens, so an unset
            # prefix means the whole sequence is already-cached prefix.
            prefix_is_full_seq = num_accept_tokens is None
            prefix_lens = (
                seq_lens
                if prefix_is_full_seq
                else seq_lens
                - num_accept_tokens[: seq_lens.shape[0]].to(
                    device=seq_lens.device, dtype=seq_lens.dtype
                )
            )
        else:
            prefix_is_full_seq = False
        sliding_window_size = self.sliding_window_size
        assert sliding_window_size is not None
        for wrapper_id in range(2):
            swa_paged_custom_mask = None
            if wrapper_id == 0:
                if use_ragged:
                    # K for extend tokens is written after the paged wrapper runs, so
                    # the paged wrapper sees prefix-only. Trim to the last `window` tokens
                    # (required for SWATokenToKVPoolAllocator; also keeps mask O(window)).
                    effective_start = torch.clamp(
                        prefix_lens - sliding_window_size, min=0
                    )
                    paged_kernel_lens = prefix_lens - effective_start
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = effective_start
                    swa_paged_custom_mask = self._build_swa_prefix_custom_mask(
                        prefix_lens, seq_lens, effective_start
                    )
                else:
                    # window attention use paged only; the trim below is
                    # request-granular, exactness comes from plan-time window_left
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        sliding_window_size + seq_lens - prefix_lens,
                    )
                    if prefix_is_full_seq and seq_lens_cpu is not None:
                        # prefix_lens is seq_lens, so the trim is min(seq_lens, window);
                        # summing the host mirror avoids draining the stream.
                        paged_kernel_lens_sum = int(
                            torch.clamp(seq_lens_cpu, max=sliding_window_size).sum()
                        )
                    else:
                        paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = seq_lens - paged_kernel_lens
            elif use_ragged:
                # Extend K/V lands after the paged pass: plan over the prefix only.
                paged_kernel_lens = prefix_lens
                if extend_prefix_lens_cpu is not None:
                    paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
                else:
                    paged_kernel_lens_sum = prefix_lens.sum().item()
                kv_start_idx = None
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=swa_paged_custom_mask,
                # paged-only SWA path only; ragged keeps its custom prefix
                # mask, spec-verify keeps its tree mask
                window_left=(
                    sliding_window_size
                    if (wrapper_id == 0 and not use_ragged and spec_info is None)
                    else -1
                ),
                seq_lens_cpu=seq_lens_cpu,
                is_swa_windowed_wrapper=(wrapper_id == 0),
            )

    def _build_swa_prefix_custom_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Custom SWA mask for the paged wrapper in the ragged merge_state EXTEND path.

        Paged KV covers absolute positions [kv_start_idx[i], prefix_lens[i]).
        Returns None when every key is in-window for every extend query.
        """
        window = self.sliding_window_size
        if window is None or window < 0:
            return None

        prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
        extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()
        kv_start_cpu = kv_start_idx.detach().cpu().tolist()
        if all(p == 0 for p in prefix_lens_cpu):
            return None

        device = prefix_lens.device
        mask_parts: List[torch.Tensor] = []
        need_mask = False
        for prefix_len, extend_len, kv_start in zip(
            prefix_lens_cpu, extend_lens_cpu, kv_start_cpu
        ):
            paged_len = int(prefix_len - kv_start)  # = min(prefix_len, window)
            if paged_len == 0 or extend_len == 0:
                continue
            q_abs = torch.arange(extend_len, device=device).view(-1, 1) + prefix_len
            k_abs = torch.arange(paged_len, device=device).view(1, -1) + kv_start
            block = (k_abs >= (q_abs - window)).to(torch.uint8)
            if not bool(block.all()):
                need_mask = True
            mask_parts.append(block.view(-1))

        if not need_mask or not mask_parts:
            return None
        return torch.cat(mask_parts)

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
    ):
        if custom_kv_indices is not None:
            raise RuntimeError(
                "NVFP4 custom KV indices are not supported for cross-attention."
            )
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=(
                    cross_attention_custom_mask if wrapper_id == 1 else None
                ),
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        custom_kv_indices: Optional[torch.Tensor] = None,
        window_left: int = -1,
        is_swa_windowed_wrapper: bool = False,
    ):
        bs = len(seq_lens)
        # Unified SWA wrapper-0: gather from the swa canonical directly -- its
        # entries are already swa-side kernel-facing ids, so the in-place
        # full->swa translate below must not run on top of them.
        translator = self.attn_backend.kv_index_translator
        use_swa_source = use_sliding_window_kv_pool and translator.reads_are_translated
        if spec_info is None:
            assert prefix_lens is not None
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            # custom_kv_indices uses exact dq_paged_kernel_lens so FlashInfer causal
            # offsets are based on real token counts, not page-aligned padding.
            if (
                custom_kv_indices is not None
                and self.attn_backend.dq_paged_kernel_lens is not None
            ):
                kv_indptr[1 : bs + 1] = torch.cumsum(
                    self.attn_backend.dq_paged_kernel_lens, dim=0
                )
            else:
                kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if custom_kv_indices is not None:
                kv_indices = custom_kv_indices
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum + 256,
                    dtype=torch.int32,
                    device=req_pool_indices.device,
                )
                translator.fill_packed_read_stream(
                    req_pool_indices=req_pool_indices,
                    seq_lens=paged_kernel_lens,
                    indptr=kv_indptr,
                    total_tokens=paged_kernel_lens_sum,
                    out=kv_indices,
                    kv_start_idx=kv_start_idx,
                    sliding_window=use_swa_source,
                )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            assert isinstance(spec_info, SpecInput)
            if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                        kv_start_idx=kv_start_idx,
                    )
                )
            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                    )
                )

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool and not use_swa_source:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        # fast_prefill_plan (installed at capture) is sync-free: it needs the
        # host-known qo/kv layout from the caller. Assert rather than silently
        # fall back to plan()'s blocking D2H on the replay hot-path.
        paged_plan_kwargs = {}
        num_tokens_per_req = (
            spec_info.num_tokens_per_req if spec_info is not None else None
        )
        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
        if uses_fast_prefill:
            assert (
                use_custom_mask is None
            ), "fast_prefill_plan does not support custom_mask; keep the plain plan()"
            assert (
                seq_lens_cpu is not None
            ), "fast_prefill_plan replay requires host-known seq_lens_cpu (got None)"
            assert (
                num_tokens_per_req is not None and num_tokens_per_req > 0
            ), f"fast_prefill_plan replay requires num_tokens_per_req > 0 (got {num_tokens_per_req})"
            ragged_layout = getattr(spec_info, "ragged_verify_layout", None)
            if ragged_layout is not None:
                (
                    qo_indptr_host,
                    q_lens_host_i32,
                    effective_kv_lens_host_i32,
                ) = _ragged_verify_host_plan_metadata(
                    spec_info=spec_info,
                    seq_lens_cpu=seq_lens_cpu,
                    padded_bs=bs,
                )
                if is_swa_windowed_wrapper:
                    assert self.sliding_window_size is not None
                    prefix_lens_host_i32 = effective_kv_lens_host_i32 - q_lens_host_i32
                    kv_lens_host_i32 = (
                        torch.clamp(
                            prefix_lens_host_i32,
                            max=self.sliding_window_size,
                        )
                        + q_lens_host_i32
                    )
                else:
                    kv_lens_host_i32 = effective_kv_lens_host_i32
                max_q_len = max(1, int(q_lens_host_i32.max()))
            else:
                seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
                qo_indptr_host = torch.arange(
                    0,
                    (bs + 1) * num_tokens_per_req,
                    step=num_tokens_per_req,
                    dtype=torch.int32,
                    device="cpu",
                )
                kv_lens_host_i32 = _static_verify_host_kv_lens(
                    spec_info=spec_info,
                    seq_lens_cpu_i32=seq_lens_cpu_i32,
                    num_tokens_per_req=num_tokens_per_req,
                )
                max_q_len = num_tokens_per_req
            if is_swa_windowed_wrapper and ragged_layout is None:
                # update_sliding_window's non-ragged wrapper-0 formula is
                # min(seq_lens, sliding_window_size + seq_lens - prefix_lens);
                # this fast path only covers callers where prefix_lens falls
                # back to seq_lens (no spec_info.num_accept_tokens) and the
                # non-ragged branch runs -- the same preconditions
                # update_sliding_window itself uses -- assert them rather
                # than silently mis-size the window or drop the ragged
                # branch's custom mask handling on a future caller.
                assert (
                    spec_info is not None
                    and getattr(spec_info, "num_accept_tokens", None) is None
                ), (
                    "fast_prefill_plan SWA branch assumes prefix_lens == "
                    "seq_lens (no num_accept_tokens on spec_info)"
                )
                assert not use_ragged, "fast_prefill_plan SWA branch only covers the non-ragged wrapper-0 formula"
                assert self.sliding_window_size is not None
                if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                    prefix_lens_host_i32 = seq_lens_cpu_i32.clone()
                    live_prefix = getattr(spec_info, "live_seq_lens_cpu", None)
                    if live_prefix is not None:
                        live_bs = int(live_prefix.numel())
                        prefix_lens_host_i32[:live_bs] = live_prefix.to(torch.int32)
                    kv_lens_host_i32 = (
                        torch.clamp(
                            prefix_lens_host_i32,
                            max=self.sliding_window_size,
                        )
                        + num_tokens_per_req
                    )
                else:
                    kv_lens_host_i32 = torch.clamp(
                        seq_lens_cpu_i32, max=self.sliding_window_size
                    )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32, device="cpu")
            kv_indptr_host[1:] = torch.cumsum(kv_lens_host_i32, dim=0)
            paged_plan_kwargs = dict(
                qo_indptr_host=qo_indptr_host,
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=kv_lens_host_i32,
                max_q_len=max_q_len,
                max_kv_len=int(kv_lens_host_i32.max()),
            )

        if window_left >= 0:
            # selects the module with the per-element window mask compiled in
            paged_plan_kwargs["window_left"] = window_left

        if spec_info is not None and envs.SGLANG_DSPARK_DEBUG_FORCE_SINGLE_SPLIT.get():
            # Diagnostic-only: force one non-split KV tile, so the multi-tile
            # split-KV LSE merge path is never exercised. Isolates whether
            # that merge -- not causal masking itself -- is the source of a
            # graph-replay-only correctness gap now that this call passes
            # causal=True.
            fixed_split_size = 1 << 20

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            # plan()/begin_forward() default causal=False; the actual
            # attention run (forward_extend, this file, ~line 1235) always
            # computes and passes its own causal=True for standard decoder
            # layers, independent of what plan() was given. For a spec_info
            # verify forward (DSpark/DFlash: linear window, custom_mask
            # always None, so causal masking is the ONLY thing restricting
            # each of the verify_w query rows to its causally-valid keys)
            # that leaves plan scheduled for non-causal work while run
            # executes causal -- a plan/run mismatch, not present for the
            # normal-extend branch (spec_info is None) so left untouched
            # here pending the same audit.
            causal=(spec_info is not None),
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            **paged_plan_kwargs,
        )

        if spec_info is not None:
            _stash_dspark_verify_plan_info(wrapper_paged)
        if spec_info is not None and envs.SGLANG_DSPARK_DEBUG_LOG_PLAN_INFO.get():
            _log_dspark_verify_plan_info(wrapper_paged)
        if spec_info is not None:
            _maybe_dump_dspark_verify_plan(
                bs=bs,
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
            )


_dspark_verify_plan_dump_ct = 0


def _maybe_dump_dspark_verify_plan(
    *,
    bs: int,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
) -> None:
    """Record the attention metadata the verify forward will consume.

    qo_indptr/kv_indptr/kv_indices here are the SAME graph-owned buffers the
    captured kernel dereferences at replay (the indices updater writes into
    them, then this plan call reads them), so a stale or wrong value logged
    here is exactly a stale or wrong value the attention kernel saw. Pairs
    with the worker-side verify_fwd records (same JSONL) via write order and
    kv_len to diff intended vs consumed inputs per tick.
    """
    path = envs.SGLANG_DSPARK_DEBUG_ACCEPT_DUMP_PATH.get()
    if not path:
        return
    # .cpu() below would be an illegal D2H inside graph capture.
    if torch.cuda.is_current_stream_capturing():
        return
    global _dspark_verify_plan_dump_ct
    if _dspark_verify_plan_dump_ct > 3000:
        return
    _dspark_verify_plan_dump_ct += 1
    qo = qo_indptr[: bs + 1].cpu().tolist()
    kvp = kv_indptr[: bs + 1].cpu().tolist()
    kv_len0 = int(kvp[1]) if len(kvp) > 1 else 0
    tail_start = max(0, kv_len0 - 12)
    record = {
        "kind": "verify_plan",
        "ct": _dspark_verify_plan_dump_ct,
        "device": kv_indices.device.index,
        "bs": int(bs),
        "qo_indptr": qo,
        "kv_indptr": kvp,
        "kv_indices_head": kv_indices[:4].cpu().tolist(),
        "kv_indices_tail": kv_indices[tail_start:kv_len0].cpu().tolist(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


_LAST_DSPARK_PLAN_INFO: dict = {}


def _stash_dspark_verify_plan_info(wrapper_paged) -> None:
    # Diagnostic-only: unconditionally snapshot the just-computed plan_info
    # scalars (PrefillPlanInfo::ToVector's leading fields, see
    # _log_dspark_verify_plan_info) into a module-global every time a
    # spec_info (DSPARK verify) plan() call runs -- both the graph replay's
    # per-tick plan and the shadow-eager-verify plan. Cheap (five int()
    # reads of an already-materialized host vector), so left unguarded by
    # a debug flag; the worker-side dump pulls this right after each of the
    # two calls to compare graph-time vs eager-time scheduling decisions
    # for the identical (bs, window) shape.
    global _LAST_DSPARK_PLAN_INFO
    plan_info = wrapper_paged._plan_info
    _LAST_DSPARK_PLAN_INFO = {
        "padded_batch_size": int(plan_info[0]),
        "total_num_rows": int(plan_info[1]),
        "cta_tile_q": int(plan_info[3]),
        "enable_cuda_graph": bool(plan_info[-2]),
        "split_kv": bool(plan_info[-1]),
        "is_capturing": bool(torch.cuda.is_current_stream_capturing()),
    }
    if envs.SGLANG_DSPARK_DEBUG_PROBE_FI_WORKSPACE.get():
        _LAST_DSPARK_PLAN_INFO.update(_probe_fi_workspace(wrapper_paged, plan_info))


def _probe_fi_workspace(wrapper_paged, plan_info) -> dict:
    """Read the split-KV scratch arrays back out of the int workspace buffer.

    PrefillPlan lays these out with AlignedAllocator at the byte offsets it
    reports in plan_info, then memcpys the host staging buffer over in one
    shot. Slicing the device buffer at those same offsets is therefore an
    exact read of what the attention kernel will dereference. Safe here
    because plan() runs host-side outside graph capture (is_capturing=False
    on every observed verify tick), so the implied device sync is legal.
    """
    # PrefillPlanInfo::ToVector index map (scheduler.cuh).
    (
        padded_batch_size,
        total_num_rows,
        _total_num_rows_offset,
        _cta_tile_q,
        _request_indices_offset,
        _qo_tile_indices_offset,
        kv_tile_indices_offset,
        merge_indptr_offset,
        _o_indptr_offset,
        kv_chunk_size_ptr_offset,
        _v_offset,
        _s_offset,
        block_valid_mask_offset,
        _enable_cuda_graph,
        split_kv,
    ) = [int(v) for v in plan_info]
    if not split_kv:
        return {"ws_probe": "split_kv=False (no mask/merge arrays allocated)"}
    buf = wrapper_paged._int_workspace_buffer.view(torch.uint8)

    def _i32(offset: int, count: int) -> list:
        # clone() before view(): a slice carries a storage_offset, and
        # view(dtype) rejects that; the clone gives a fresh zero-offset
        # storage. Debug-only path, so the extra copy is irrelevant.
        raw = buf[offset : offset + 4 * count].clone()
        return raw.view(torch.int32).tolist()

    mask = buf[
        block_valid_mask_offset : block_valid_mask_offset + padded_batch_size
    ].tolist()
    n_valid = sum(1 for v in mask if v)
    kv_chunk_size = _i32(kv_chunk_size_ptr_offset, 1)[0]
    merge_indptr = _i32(merge_indptr_offset, total_num_rows + 1)
    return {
        "ws_padded_batch_size": padded_batch_size,
        # The decisive number: how many of the padded work-units the merge
        # will actually fold in. Must equal the real (unpadded) work-unit
        # count; anything larger means padding slots contribute garbage.
        "ws_block_valid_true": n_valid,
        "ws_block_valid_head": mask[:12],
        "ws_kv_chunk_size": kv_chunk_size,
        "ws_merge_indptr": merge_indptr[: min(len(merge_indptr), 12)],
        "ws_merge_indptr_last": merge_indptr[-1] if merge_indptr else None,
        "ws_kv_tile_indices_head": _i32(
            kv_tile_indices_offset, min(padded_batch_size, 12)
        ),
    }


def get_last_dspark_verify_plan_info() -> dict:
    return dict(_LAST_DSPARK_PLAN_INFO)


_dspark_verify_plan_info_log_ct = 0


def _log_dspark_verify_plan_info(wrapper_paged) -> None:
    # PrefillPlanInfo::ToVector (FlashInfer scheduler.cuh): [padded_batch_size,
    # total_num_rows, total_num_rows_offset, cta_tile_q, ..., enable_cuda_graph,
    # split_kv] -- the last two are always the trailing two entries.
    global _dspark_verify_plan_info_log_ct
    if _dspark_verify_plan_info_log_ct >= 5:
        return
    _dspark_verify_plan_info_log_ct += 1
    plan_info = wrapper_paged._plan_info
    logger.warning(
        "[dspark plan_info %d] padded_batch_size=%s total_num_rows=%s "
        "cta_tile_q=%s enable_cuda_graph=%s split_kv=%s",
        _dspark_verify_plan_info_log_ct,
        plan_info[0],
        plan_info[1],
        plan_info[3],
        plan_info[-2],
        plan_info[-1],
    )


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size

        max_bs = get_cuda_graph_max_batch_size(
            model_runner.req_to_token_pool.size * self.topk
        )
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.draft_window_size, self.draft_sink_size = resolve_draft_decode_window(
            model_runner
        )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        required_kv_indices_len = draft_kv_indices_used_len(
            seq_lens_sum, self.topk, bs, self.speculative_num_steps
        )
        assert_buffer_fits(
            required_kv_indices_len,
            kv_indices_buffer.shape[1],
            "EAGLE draft kv_indices row (size max_bs * topk * max_context_len)",
            bs=bs,
            seq_lens_sum=seq_lens_sum,
        )

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
            self.draft_window_size,
            self.draft_sink_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # generate_draft_decode_kv_indices packs topk per-branch sequences per row,
        # so the row needs the topk factor -- same as the eager init_forward_metadata
        # (batch_size * topk * max_context_len). Dropping it overflows the buffer.
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        bs = forward_batch.batch_size

        def call_fn(i, fb):
            inner_fb = build_inner_fb_view(fb, bs=bs, forward_mode=ForwardMode.DECODE)
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
