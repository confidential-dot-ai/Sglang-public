# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DecodeCudaGraphRunner — runs DECODE / TARGET_VERIFY / DLLM_EXTEND under
a pluggable backend.

Backend selection comes from cuda_graph_config.decode:
  - "full"      — default, FullCudaGraphBackend: one
                      torch.cuda.CUDAGraph per shape.
  - "breakable" — experimental, BreakableCudaGraphBackend:
                      segmented capture (no torch.compile).
  - "tc_piecewise"     — not implemented for decode; logs a one-shot warning
                      and falls back to "full".
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import inspect
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.compilation import torch_compile_decoration
from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.distributed.parallel_state import (
    graph_capture,
)
from sglang.srt.dllm.config import DllmConfig
import itertools
import json
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.graph_variants import (
    AttentionGraphVariants,
    create_attention_graph_variants,
    create_dsv41_candidate_graph_variants,
)
from sglang.srt.layers.cp.utils import is_mla_cp_enabled
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    is_dp_gatherv_configured,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    compute_local_num_token_non_padded,
    enable_num_token_non_padded,
    get_required_capture_hidden_mode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
    freeze_gc,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    maybe_flashinfer_autotune_speculative_draft,
)
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    is_ccd_capturing,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.runner_utils.buffers import (
    DecodeInputBuffers,
)
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_attention_variant,
    _set_capture_lora_variant,
    model_capture_mode,
)
from sglang.srt.model_executor.runner_utils.deepep_adapter import (
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
)
from sglang.srt.model_executor.runner_utils.shared_read_event import make_external_event
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    require_attn_tp_gather,
    require_mlp_tp_gather,
)
from sglang.srt.utils.device_timer import device_timer_ctx
from sglang.srt.utils.profile_utils import (
    export_cuda_graph_capture_trace,
    graph_capture_profile_dir,
)

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


_replay_view_dump_ct = itertools.count()
_prev_out_cache_loc = None


def _pool_ptr_record(*, model_runner, tag: str) -> dict:
    """Identity of the pools the captured kernels dereference.

    The replay view's metadata is verified correct, so the remaining way to get
    wrong attention is for the graph to read a different buffer than the live
    one -- a pointer baked at capture that no longer matches. Recording both
    sides makes that a direct comparison instead of an inference.
    """
    rt = model_runner.req_to_token_pool.req_to_token
    rec = {
        "kind": "pool_ptrs",
        "tag": tag,
        "is_draft": bool(model_runner.is_draft_worker),
        "req_to_token_ptr": rt.data_ptr(),
        "req_to_token_shape": list(rt.shape),
    }
    kv = model_runner.token_to_kv_pool
    try:
        kbuf = kv.get_key_buffer(0)
        rec["kv_k0_ptr"] = kbuf.data_ptr()
        rec["kv_k0_shape"] = list(kbuf.shape)
        rec["kv_k0_dtype"] = str(kbuf.dtype)
    except Exception as exc:  # diagnostic only; pool layouts vary by backend
        rec["kv_err"] = repr(exc)[:160]
    return rec


def _maybe_dump_pool_ptrs(*, model_runner, tag: str) -> None:
    path = envs.SGLANG_DSPARK_DEBUG_ACCEPT_DUMP_PATH.get()
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(_pool_ptr_record(model_runner=model_runner, tag=tag)) + "\n")


def _slot_stats(buf, slots) -> dict:
    """Occupancy of `buf` at exactly `slots`, for a pool buffer allocated zeroed.

    zero_rows is the discriminator: a slot the write path skipped is still all
    zeros, so zero_rows == n_rows on a live decode step means "this family was
    never written for these tokens".
    """
    sample = buf[slots]
    if sample.dtype in (torch.uint8, torch.int8, torch.int32, torch.int64):
        sample = sample.to(torch.int64).abs()
    else:
        sample = sample.float().abs()
    per_row = sample.sum(dim=tuple(range(1, sample.dim()))) if sample.dim() > 1 else sample
    return {
        "absmax": round(float(sample.max()), 5),
        "zero_rows": int((per_row == 0).sum()),
        "n_rows": int(per_row.numel()),
    }


def _kv_family_stats(*, pool, slots) -> dict:
    """Sample every KV buffer family of a hybrid (dense/sparse/index) KV pool.

    Sub-pools are addressed directly rather than through get_key_buffer(), which
    dispatches on layer id and would only ever reach one of them.
    """
    out: dict = {}

    def _add(name, buf):
        if buf is None:
            return
        try:
            out[name] = _slot_stats(buf, slots)
        except Exception as exc:  # diagnostic only; layouts vary by config
            out[name] = {"err": repr(exc)[:120]}

    for sub_name in ("dense_pool", "sparse_pool", "index_kv_pool", "index_k_pool"):
        sub = getattr(pool, sub_name, None)
        if sub is None:
            continue
        for attr in (
            "k_buffer",
            "v_buffer",
            "k_scale_buffer",
            "v_scale_buffer",
            "k_scale_swz_buffer",
            "v_scale_swz_buffer",
        ):
            bufs = getattr(sub, attr, None)
            if not bufs:
                continue
            _add(f"{sub_name}.{attr}", bufs[0])
    return out


def _maybe_dump_replay_view(
    *,
    forward_batch,
    fb_view,
    model_runner,
    raw_bs: int,
    padded_bs: int,
    raw_num_token: int,
    padded_num_tokens: int,
) -> None:
    """Diff the replay view the captured graph will read against the live batch.

    The target-verify forward computes wrong logits under graph replay on every
    attention backend (sparse-nvfp4, sparse-triton, dense), which rules out the
    attention kernels and points at the metadata they all consume. This compares
    the two in the same call -- no cross-run trajectory mismatch -- so any field
    the replay view gets wrong is named directly rather than inferred.
    """
    path = envs.SGLANG_DSPARK_DEBUG_ACCEPT_DUMP_PATH.get()
    if not path:
        return
    if not forward_batch.forward_mode.is_target_verify():
        return
    ct = next(_replay_view_dump_ct)
    if ct > 40:
        return

    def _diff(name, truth, view, n):
        if truth is None or view is None:
            return {name: "missing" if truth is None else "view_missing"}
        t = truth[:n].tolist()
        v = view[:n].tolist()
        return {
            name: "MATCH" if t == v else "MISMATCH",
            **({} if t == v else {f"{name}_truth": t[:12], f"{name}_view": v[:12]}),
        }

    rec = {
        "kind": "replay_view",
        "ct": ct,
        "raw_bs": int(raw_bs),
        "padded_bs": int(padded_bs),
        "raw_num_token": int(raw_num_token),
        "padded_num_tokens": int(padded_num_tokens),
        "is_draft": bool(model_runner.is_draft_worker),
    }
    rec.update(_diff("positions", forward_batch.positions, fb_view.positions, raw_num_token))
    rec.update(_diff("input_ids", forward_batch.input_ids, fb_view.input_ids, raw_num_token))
    rec.update(_diff("seq_lens", forward_batch.seq_lens, fb_view.seq_lens, raw_bs))
    rec.update(
        _diff("out_cache_loc", forward_batch.out_cache_loc, fb_view.out_cache_loc, raw_num_token)
    )
    rec.update(
        _diff(
            "req_pool_indices",
            forward_batch.req_pool_indices,
            fb_view.req_pool_indices,
            raw_bs,
        )
    )
    rec.update(_pool_ptr_record(model_runner=model_runner, tag="replay"))
    rec["kind"] = "replay_view"

    # Page table row: the position -> KV-slot mapping the captured attention
    # kernel walks. Metadata being correct does not imply this row is; a stale
    # or zeroed tail here is exactly "attention cannot see the history", which
    # is what the damped logits and '\n' predictions look like.
    req_idx = int(forward_batch.req_pool_indices[0])
    seq_len = int(forward_batch.seq_lens[0])
    row = model_runner.req_to_token_pool.req_to_token[req_idx, :seq_len]
    rec["page_row_seq_len"] = seq_len
    rec["page_row_zeros"] = int((row == 0).sum())
    rec["page_row_head"] = row[:8].tolist()
    rec["page_row_tail"] = row[-8:].tolist()
    rec["page_row_max"] = int(row.max()) if seq_len else None
    # Capture-time dummy batches reach this path too, and their seq_lens are
    # the fill value (1), which makes every diff match trivially. Only views
    # carrying actual_forward_mode are live replays -- the same discriminator
    # the sparse attention backends use. Without this the dump is dominated by
    # capture batches and says nothing about replay.
    rec["is_live_replay"] = hasattr(fb_view, "actual_forward_mode")
    rec["seq_len0"] = seq_len

    # Did the captured graph actually write KV for last step's verify tokens?
    # Metadata and page tables are verified correct, so the remaining way to
    # get wrong attention is for the KV at those (correct) slots to be absent
    # or stale. Read the PREVIOUS step's out_cache_loc now that its graph has
    # run: a written slot has non-zero data, an unwritten one stays at pool
    # init (every buffer family below is allocated with torch.zeros).
    #
    # Every family is sampled, not just one: get_key_buffer(0) resolves to the
    # DENSE sub-pool (layer 0 is dense, bf16), so a clean result there says
    # nothing about the FP4 sparse pool, its block scales, the incrementally
    # maintained 128x4 swizzle cache, or the fp8 index cache -- and those are
    # exactly the buffers whose staleness would damp attention on every
    # backend at once (sparse-nvfp4, sparse-triton, and dense all read the
    # same sparse KV; only the topk differs). set_kv_buffer's capture branch
    # also diverges from its eager branch here: V and V-scale writes move to
    # alt_stream, and the swizzle update splits into two unfused launches.
    global _prev_out_cache_loc
    if _prev_out_cache_loc is not None:
        prev = _prev_out_cache_loc
        rec["prev_kv_slots"] = prev[:8].tolist()
        rec["prev_kv_n_rows"] = int(prev.numel())
        rec["prev_kv"] = _kv_family_stats(
            pool=model_runner.token_to_kv_pool, slots=prev
        )
    _prev_out_cache_loc = forward_batch.out_cache_loc[:raw_num_token].clone()

    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def dspark_dp_verify_sum_capture_enabled(
    *,
    spec_algorithm: SpeculativeAlgorithm,
    require_mlp_tp_gather: bool,
    is_target_verify_capture: bool,
    is_draft_worker: bool,
) -> bool:
    """TODO.md J14c Milestone 1: DSPARK TARGET_VERIFY under DP attention with
    a captured SUM_LEN dp_padding_mode, bucketed on the cross-DP-rank total
    token count instead of the MAX_LEN forced by
    DpPaddingMode.get_default_mode_in_cuda_graph. Off by default
    (SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE); every other combination of
    spec algorithm / topology / forward mode takes the pre-existing MAX_LEN
    path unchanged.
    """
    if not (
        spec_algorithm.is_dspark()
        and require_mlp_tp_gather
        and is_target_verify_capture
        and not is_draft_worker
    ):
        return False
    if not envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get():
        return False
    if is_dp_gatherv_configured():
        raise ValueError(
            "SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE requires "
            "SGLANG_DP_USE_GATHERV=0 (unset): its gatherv path sizes the "
            "DP-gather buffer from a Python-side size list, which is not "
            "CUDA-graph-replay-safe for a shape that must vary run to run."
        )
    return True


def build_dp_verify_sum_bucket_ladder(
    capture_bs: list[int], dp_size: int, window: int, num_tokens_per_req: int
) -> dict[int, list[int]]:
    """For each local capture bucket (request-count units), the small set of
    cross-DP-rank total TOKEN counts ("sum buckets") to capture a SUM_LEN-
    mode graph against.

    Keys stay in request-count units (matching self.capture_bs, so callers
    can look up by the same local bucket they already compute via
    _pad_to_bucket). Values are TOKEN counts (request count * num_tokens_per_req,
    9 for DSPARK's draft width) because that's what capture_prepare's
    num_tokens / synthetic_dp_verify_sum_split's `total` and
    forward_batch.original_global_num_tokens_cpu are all denominated in --
    mixing the two units here is exactly the capture-time crash this
    docstring update fixes (AssertionError: sum_bucket total below this
    rank's own local tokens, first live boot, TODO.md J14c).

    **Every local bucket maps to the same sum ladder, and that is load
    bearing, not a simplification.** The sum bucket picked for a step becomes
    the size of the DP-gathered buffer, i.e. the size of a collective every
    rank takes part in. Ranks hold different request counts, so if the ladder
    were keyed on the caller's own local bucket (as it was until TODO.md
    J14d), two ranks in one step could resolve different sums -- or one could
    resolve none and fall back to eager while its peers replayed graphs. Both
    are rank divergence under DP lockstep: the observed skew (2,3,4,5) gave
    per-rank sums [None, None, 144, 180], a size-mismatched all-reduce
    between the two graph ranks plus two ranks somewhere else entirely.
    Keying every bucket to one shared ladder makes lookup a pure function of
    `real_sum`, which is itself derived from the all-gathered vector every
    rank already holds identically -- so agreement is structural rather than
    something skew can break.

    That shared ladder holds exactly **one** entry: the largest total any step
    can produce, `dp_size * max(capture_bs) * num_tokens_per_req`. Two
    independent reasons force this rather than a range of totals:

    * The sum is the DP-gathered buffer length, and the model specializes on
      it, so each distinct total is a fresh set of compiled kernels. Capturing
      a matrix of totals made every rank stall in `cuModuleLoad` at the first
      shape (TODO.md J14d) -- a single total reuses one specialization, which
      is the regime that has always captured cleanly.
    * Smaller totals are not even representable at capture. All ranks capture
      the same shape, so a total below `dp_size * local_tokens` forces
      synthetic_dp_verify_sum_split to hand the other ranks zero -- a buffer
      with room for one rank, every rank claiming offset 0. The one total here
      always splits into positive per-rank shares.

    A single oversized buffer costs collective bandwidth versus sizing it to
    the true total, but it keeps the win this feature exists for: each rank's
    own tensors stay at its own local bucket through all 60 decoder layers,
    instead of MAX_LEN padding every rank up to the cross-rank max. Replay
    offsets come from the real per-rank counts on the GPU, so the unused tail
    is simply not read.

    `window` is no longer a coverage knob and is accepted only so existing
    callers/env keep working; it is deliberately unused.
    """
    del window  # see docstring: coverage is fixed, not tunable
    largest_total = dp_size * max(capture_bs) * num_tokens_per_req
    return {bucket: [largest_total] for bucket in sorted(capture_bs)}


def build_ragged_dp_verify_sum_bucket_ladder(
    capture_num_tokens: list[int], dp_size: int, window: int
) -> dict[int, list[int]]:
    """SUM_LEN capture ladder for compact verify.

    Unlike :func:`build_dp_verify_sum_bucket_ladder`, both the keys and the
    values here are already token-denominated.  ``ShapeKey.size`` is a packed
    query-token bucket for compact graphs, and ``ShapeKey.sum_bucket`` is a
    token-denominated captured collective total that covers the sum of the
    per-rank packed graph-token buckets.  The existing capture path uses one
    rank-identical largest-total specialization; it is not necessarily the
    literal vector sum.  Keeping the conversion in this small wrapper prevents
    the static request-count ladder from silently multiplying compact buckets
    by the draft width a second time.
    """
    return build_dp_verify_sum_bucket_ladder(
        capture_num_tokens,
        dp_size=dp_size,
        window=window,
        num_tokens_per_req=1,
    )


def synthetic_dp_verify_sum_split(
    my_tokens: int, total: int, dp_size: int, my_rank: int
) -> list[int]:
    """A `global_num_tokens_cpu`-shaped placeholder list, length `dp_size`,
    summing to exactly `total` -- CANONICAL: every rank computes the identical
    list, with no dependence on `my_rank`.

    Rank-invariance is the load-bearing property, not a style choice (TODO.md
    J14d, fifth follow-up). The previous version pinned `my_rank`'s slot to
    `my_tokens` and spread the remainder over the peers, so whenever
    `my_tokens < total / dp_size` each rank traced a DIFFERENT (rotated) list
    -- and capture empirically hangs (symmetric `cuModuleLoad` stall on all
    ranks) exactly when the per-rank traced splits differ, while identical
    splits capture cleanly every time. The synthetic list leaks into the
    traced program through its CPU consumers (`sum(...)` sizes the gathered
    buffer; `max(...)` sizes flashinfer's per-rank dispatch geometry;
    `set_dp_buffer_len` republishes the list to any model-side reader), so
    the only way to make every consumer rank-symmetric at trace time --
    without enumerating them -- is to hand every rank the same list.

    Correctness at REPLAY is unaffected by the placeholder values: the real
    per-rank gather/scatter offsets are recomputed fresh from the real
    (post-replay-copy) global_num_tokens_gpu contents via get_dp_local_info's
    on-device cumsum (dp_attention.py:349-364), and load_batch overwrites the
    CPU/GPU lists with the real skewed counts. The placeholder only has to
    give the trace a fixed, rank-identical geometry that upper-bounds every
    real split -- which `total = dp_size * max(capture_bs) *
    num_tokens_per_req` (build_dp_verify_sum_bucket_ladder's single entry)
    guarantees: each canonical slot is `max(capture_bs) * num_tokens_per_req
    >= my_tokens` for every capturable local bucket.

    `my_tokens`/`my_rank` are retained for the coverage assertions only.
    """
    assert total >= dp_size * my_tokens, (
        f"sum_bucket total {total} cannot give every rank a canonical slot "
        f"covering this rank's own local tokens {my_tokens} at dp_size "
        f"{dp_size}; build_dp_verify_sum_bucket_ladder should never produce "
        f"this and the capture loop's feasibility filter must skip it"
    )
    base = total // dp_size
    extra = total - base * dp_size
    # Deterministic by index, never by my_rank: identical on every rank.
    split = [base + (1 if i < extra else 0) for i in range(dp_size)]
    assert split[my_rank] >= my_tokens and sum(split) == total
    return split


def resolve_dp_verify_sum_graph_buckets(
    *,
    global_num_reqs: list[int],
    dp_rank: int,
    num_tokens_per_req: int,
    capture_bs: list[int],
    sum_buckets_by_local: dict[int, list[int]],
    max_bs: int,
) -> Optional[tuple[int, int]]:
    """The (local_bucket, sum_bucket) this rank replays under, or None when no
    captured pair covers the batch (caller falls back to eager).

    `global_num_reqs` is forward_batch.original_global_num_tokens_cpu, whose
    name is misleading on this path: for a decode-family round it holds per-rank
    REQUEST counts, not token counts. DSPARK keeps ScheduleBatch.forward_mode ==
    DECODE through spec_prepare_for_decode (TARGET_VERIFY is only applied later,
    inside ForwardBatch.init_new), so prepare_mlp_sync_batch_raw's is_decode()
    branch is what fills it: `num_tokens = local_batch.batch_size()`. The
    pre-existing MAX_LEN admission check says the same in its own comment
    ("per-rank request counts on decode-family rounds -- no width division").

    So the local bucket takes that count directly, and only the ladder lookup
    converts: build_dp_verify_sum_bucket_ladder's keys are request counts but
    its VALUES are token counts, so real_sum is scaled by num_tokens_per_req.
    Dividing the per-rank count instead (the inverse of this) collapsed every
    batch below num_tokens_per_req to bucket 1, replaying a 1-request graph for
    a 4-request batch -- TODO.md J14d's "next_token_logits row count mismatch
    for DFlash verify adjustments. Expected 36, got 9."

    Shared by the admission check and the replay path so the two can never
    disagree about which pair a batch maps to.

    Every decision here is a pure function of `global_num_reqs`, which the
    MLP-sync all-gather already leaves identical on every rank. That is
    required, not incidental: DP ranks advance in lockstep, so "use a graph"
    and "which sum bucket" must come out the same everywhere or the step
    deadlocks. In particular the max_bs test covers *all* ranks rather than
    just this one -- a rank over the ladder cannot replay, and a rank that
    replayed while a peer went eager would leave that peer waiting in a
    collective the graph never performs (TODO.md J14d).
    """
    if any(num_reqs > max_bs for num_reqs in global_num_reqs):
        return None
    local_bucket = BaseCudaGraphRunner._pad_to_bucket(
        int(global_num_reqs[dp_rank]), capture_bs
    )
    real_sum = sum(global_num_reqs) * num_tokens_per_req
    sum_bucket = lookup_dp_verify_sum_bucket(
        sum_buckets_by_local, local_bucket, real_sum
    )
    if sum_bucket is None:
        return None
    return local_bucket, sum_bucket


def resolve_ragged_dp_verify_sum_graph_buckets(
    *,
    global_graph_num_tokens: Optional[list[int]],
    dp_rank: int,
    local_graph_num_tokens: int,
    capture_num_tokens: list[int],
    sum_buckets_by_local: dict[int, list[int]],
    max_num_tokens: int,
) -> Optional[tuple[int, int]]:
    """Resolve compact verify's token-denominated ``ShapeKey`` pair.

    ``global_graph_num_tokens`` must be the already-rounded graph-token bucket
    vector shared by all DP ranks.  These are not live accepted/query totals:
    the graph executes the padded rows (including an idle rank's dummy rows),
    so the DP collective offsets and its captured buffer shape must use the
    graph buckets too.  The live total remains represented by the ragged
    layout and is used only to mask ghost rows.

    The local key comes from the layout selected by the planner, then is
    checked against this rank's entry in the shared vector.  That equality is
    load bearing: accepting a layout/vector mismatch would make admission and
    replay select a graph whose local tensor shape disagrees with the DP
    offsets baked into the step.
    """
    if not global_graph_num_tokens or not 0 <= dp_rank < len(
        global_graph_num_tokens
    ):
        return None
    if any(
        num_tokens <= 0 or num_tokens > max_num_tokens
        for num_tokens in global_graph_num_tokens
    ):
        return None
    if local_graph_num_tokens not in capture_num_tokens:
        return None
    if int(global_graph_num_tokens[dp_rank]) != local_graph_num_tokens:
        return None

    graph_sum = sum(int(num_tokens) for num_tokens in global_graph_num_tokens)
    sum_bucket = lookup_dp_verify_sum_bucket(
        sum_buckets_by_local, local_graph_num_tokens, graph_sum
    )
    if sum_bucket is None:
        return None
    return local_graph_num_tokens, sum_bucket


def lookup_dp_verify_sum_bucket(
    sum_buckets_by_local: dict[int, list[int]], local_bucket: int, real_sum: int
) -> Optional[int]:
    """Smallest captured sum bucket for `local_bucket` that covers `real_sum`,
    or None if `real_sum` falls outside the captured window (clean miss --
    the caller falls back to the pre-existing eager SUM_LEN path, exactly as
    any other can_run_graph()==False case already does)."""
    sums = sum_buckets_by_local.get(local_bucket)
    if not sums:
        return None
    index = bisect.bisect_left(sums, real_sum)
    if index >= len(sums):
        return None
    return sums[index]


def ragged_verify_compact_graphs_enabled(spec_algorithm: SpeculativeAlgorithm) -> bool:
    if not spec_algorithm.supports_ragged_verify():
        return False
    from sglang.srt.speculative.ragged_verify import ragged_verify_compact_enabled

    return ragged_verify_compact_enabled()


def build_replay_fb_view(
    forward_batch: ForwardBatch,
    buffers: DecodeInputBuffers,
    bs: int,
    raw_bs: int,
    num_tokens: int,
    global_num_tokens_cpu: Optional[list[int]],
    seq_len_fill_value: int,
    capture_forward_mode: ForwardMode,
    is_encoder_decoder: bool,
) -> SimpleNamespace:
    """Construct a ForwardBatch-like view for backend replay-side init.

    Combines the original forward_batch (for unpadded / per-iter
    fields like spec_info, out_cache_loc, and the runtime
    actual_forward_mode) with the padded capture-time buffers from
    buffers (for req_pool_indices, seq_lens, seq_lens_cpu,
    positions, encoder_lens).

    forward_mode is the capture-time mode (used by backends for
    bucket / dispatch decisions); actual_forward_mode is the
    runtime mode (may be IDLE while the captured graph targets DECODE
    — DSV4's replay metadata prep uses this for IDLE substitution).

    Subsumes the _replay_forward_batch side channel that DSV4 used to
    read out-of-band before the init_forward_metadata 3-method ABC.
    """
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=capture_forward_mode,
        actual_forward_mode=forward_batch.forward_mode,
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        req_pool_indices=buffers.req_pool_indices[:bs],
        seq_lens=buffers.seq_lens[:bs],
        seq_lens_sum=(
            None
            if forward_batch.seq_lens_sum is None
            else forward_batch.seq_lens_sum + (bs - raw_bs) * seq_len_fill_value
        ),
        # Propagate mirror absence: the pinned buffer is not refreshed when the
        # batch has no CPU mirror; a stale non-None tensor defeats None-guards.
        seq_lens_cpu=(
            None if forward_batch.seq_lens_cpu is None else buffers.seq_lens_cpu[:bs]
        ),
        num_padding=bs - raw_bs,
        encoder_lens=buffers.encoder_lens[:bs] if is_encoder_decoder else None,
        out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
        out_cache_loc_virtual=forward_batch.out_cache_loc_virtual,
        out_cache_loc_dsv4=getattr(forward_batch, "out_cache_loc_dsv4", None),
        max_seq_len_override=forward_batch.max_seq_len_override,
        # The mamba-track registry slot (VIRTUAL ids) is the v2p translate SOURCE
        # for the backend, which copies the result into its own static buffer and
        # reads THAT in the decode track-save — this slot is never mutated. None
        # when mamba-track is disabled, slice to [:bs] like every other buffer
        mamba_track_indices=(
            None
            if buffers.mamba_track_indices is None
            else buffers.mamba_track_indices[:bs]
        ),
        spec_info=forward_batch.spec_info,
        lora_ids=(
            None
            if forward_batch.lora_ids is None
            else forward_batch.lora_ids + [None] * (bs - len(forward_batch.lora_ids))
        ),
        can_run_decode_cuda_graph=forward_batch.can_run_decode_cuda_graph,
        global_num_tokens_cpu=global_num_tokens_cpu,
        dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
    )


def _is_cuda_oom(err: BaseException) -> bool:
    """True for an out-of-memory failure, which capture degrades on rather than dies.

    torch.OutOfMemoryError does not cover every path: FlashInfer workspace and
    driver-level graph allocations surface as a plain RuntimeError whose text
    carries the CUDA message, so match that too.
    """
    return isinstance(err, getattr(torch, "OutOfMemoryError", ())) or (
        "out of memory" in str(err).lower()
    )


def _validate_compact_ragged_graph_experiments(
    *,
    ragged_verify_mode: bool,
    dp_verify_sum_capture: bool = False,
) -> None:
    """Fail closed for graph experiments whose ownership assumes static Q."""
    if not ragged_verify_mode:
        return
    # Compact SUM capture is supported: its local and global keys are both
    # token-denominated (see resolve_ragged_dp_verify_sum_graph_buckets).
    del dp_verify_sum_capture


def _tbo_cuda_graph_enabled(
    *,
    enable_two_batch_overlap: bool,
    ragged_verify_mode: bool,
    dp_verify_sum_capture: bool,
    attention_backend_supports_tbo_graph: bool,
) -> bool:
    """Whether a decode graph may bake TBO child geometry into its body."""
    return (
        enable_two_batch_overlap
        and not ragged_verify_mode
        and not dp_verify_sum_capture
        and attention_backend_supports_tbo_graph
    )


def _set_replay_tbo_state(
    fb_view,
    *,
    enable_two_batch_overlap: bool,
    tbo_cuda_graph_enabled: bool,
    padded_bs: int,
) -> None:
    """Make replay metadata agree with the body captured for this graph.

    Static graph-TBO needs a non-``None`` split marker so
    :class:`TboAttnBackend` rebuilds both child backends from the padded replay
    view.  A registry-extracted view can also inherit live eager children from
    its ``ForwardBatch`` template; those children have the raw/DP-padded split,
    not necessarily the CUDA-graph bucket split, so they must never be used to
    plan a captured graph.

    Compact, SUM, and other non-TBO graph families clear both fields because
    their captured model body uses only the primary backend.
    """
    if tbo_cuda_graph_enabled:
        fb_view.tbo_split_seq_index = int(padded_bs) // 2
        fb_view.tbo_children = None
    elif enable_two_batch_overlap:
        fb_view.tbo_split_seq_index = None
        fb_view.tbo_children = None


@dataclass(frozen=True)
class _ReplayDispatchState:
    """Per-stage host state needed to dispatch the prepared graph exactly."""

    raw_bs: int
    raw_num_token: int
    bs: int
    ragged_graph_size: int
    replay_graph_key: ShapeKey
    replay_dp_verify_sum_bucket: Optional[int]
    padded_num_tokens: int
    ragged_layout_key: Optional[tuple[Optional[int], int]] = None
    staged_ragged_layout: Optional[object] = None


class DecodeCudaGraphRunner(BaseCudaGraphRunner):
    """Decode-phase CUDA graph runner.

    Owns: static input buffers (DecodeInputBuffers), capture-bs list,
    attention backend, two-batch-overlap plugin, DeepEP adapter, and the
    pluggable self.backend that handles the actual capture/replay.
    """

    # Class-level so the capture-degradation state is always readable: the
    # draft/extend subclasses reuse capture() without running this __init__.
    # Both defaults mean "nothing degraded", i.e. pre-existing behavior.
    # frozenset (rebound, never mutated) keeps the default from being shared.
    _cuda_graph_state_ready: bool = True
    _skipped_bs: frozenset = frozenset()

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        record_nolora_graph: bool = False,
    ):
        super().__init__(model_runner)
        self.record_nolora_graph = record_nolora_graph

        # In-graph metadata prep: shared buffers -> in-graph private data
        self.in_graph_metadata_prep_done: Optional[torch.cuda.Event] = None

        # --- core state ------------------------------------------------
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        self.disable_padding = get_exec().graph.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_mlp_tp_gather = (
            require_mlp_tp_gather() and not self._forward_is_dp_local(model_runner)
        )
        self.require_attn_tp_gather = require_attn_tp_gather()
        # Composite predicates derive from the instance values so the dp-local
        # draft exemption above stays consistent (require_gathered_buffer ==
        # mlp_tp_gather or attn_tp_gather; require_mlp_sync adds dp attention).
        self.require_gathered_buffer = (
            self.require_mlp_tp_gather or self.require_attn_tp_gather
        )
        self.require_mlp_sync = (
            get_parallel().enable_dp_attention or self.require_gathered_buffer
        )
        # Draft attention backends are not wrapped for TBO capture.
        self.enable_two_batch_overlap = (
            get_exec().overlap.enable_two_batch_overlap
            and not model_runner.is_draft_worker
        )
        self.use_ngram_embedding = model_runner.ngram_embedding_manager.enabled
        self.speculative_algorithm = get_spec().speculative_algorithm
        self.enable_profile_cuda_graph = get_exec().graph.enable_profile_cuda_graph

        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        # True if the DSA or MLA prefill-CP flavor is active. These flavors
        # feed a zigzag-split rank-local layout into the runner; MHA-arch
        # prefill CP (Qwen3/Qwen2 MoE via PR #18233) keeps an
        # attn_tp-replicated layout and is intentionally excluded so the
        # attn_tp-local num_token_non_padded adjustment still runs for it.
        self.enable_prefill_cp = is_dsa_enable_prefill_cp() or is_mla_cp_enabled()

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None
        self.dllm_uses_input_embeds = (
            self.is_dllm and self.dllm_config.requires_separate_context_encoding
        )
        self.attn_backend = attn_backend or model_runner.attn_backend
        self.speculative_num_steps = (
            get_spec().speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.speculative_num_draft_tokens = (
            get_spec().speculative_num_draft_tokens
            if speculative_num_draft_tokens is None
            else speculative_num_draft_tokens
        )

        # --- capture mode + tokens-per-bs ------------------------------
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = self.return_hidden_states_mode
        # Static capture width.
        self.captured_req_width = model_runner.decode_num_tokens_per_req(
            num_draft_tokens=self.speculative_num_draft_tokens
        )
        if model_runner.spec_algorithm.is_speculative():
            if self.model_runner.is_draft_worker:
                # Draft workers can use TARGET_VERIFY mode.
                if not self.model_runner.spec_algorithm.supports_target_verify_for_draft():
                    raise RuntimeError("This should not happen")
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND

        self.attention_graph_variants: Optional[AttentionGraphVariants] = (
            create_attention_graph_variants(model_runner.model_config.hf_config)
            or create_dsv41_candidate_graph_variants(
                model_runner, self.capture_forward_mode, self.captured_req_width
            )
        )

        # --- bucket sizes ---------------------------------------------
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.captured_req_width
        )
        if self.dllm_uses_input_embeds:
            max_requests = min(
                self.dllm_config.max_running_requests, max(self.capture_bs)
            )
            self.capture_bs = sorted(
                {bs for bs in self.capture_bs if bs <= max_requests} | {max_requests}
            )
            self.compile_bs = [bs for bs in self.compile_bs if bs <= max_requests]
        self.max_bs = max(self.capture_bs)
        # TODO.md J14c Milestone 1: set on the full load_batch path (the
        # only place with access to real global_num_tokens); reused as-is by
        # the pre-planned path below, matching how self.bs is already reused
        # there without recomputation.
        self._replay_dp_verify_sum_bucket: Optional[int] = None
        # Built on first idle TARGET_VERIFY replay; see
        # _idle_replay_verify_spec_info.
        self._idle_verify_spec_info = None
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)

        self.ragged_verify_mode = (
            ragged_verify_compact_graphs_enabled(self.model_runner.spec_algorithm)
            and (self.capture_forward_mode == ForwardMode.TARGET_VERIFY)
            and not self.model_runner.is_draft_worker
        )
        self.capture_num_tokens: Optional[list[int]] = (
            self._build_ragged_verify_token_buckets()
            if self.ragged_verify_mode
            else None
        )
        self._ragged_graph_size = 0
        # One persistent layout per execution lane and token bucket, shared
        # by every sequence-length/DSA/LoRA graph variant for that bucket.
        # Its verify_lens / qo_indptr tensors are baked into the captured
        # graphs and refreshed in place before replay. PDMux lanes must not
        # share them because their graph replays may overlap.
        self._captured_ragged_layouts: dict[tuple[Optional[int], int], object] = {}
        if self.ragged_verify_mode and (
            model_runner.lora_manager is not None or self.disable_padding
        ):
            raise ValueError(
                "Compact ragged verify does not support LoRA or "
                "disable-cuda-graph-padding (bs pads to the captured "
                "tier); disable SGLANG_RAGGED_VERIFY_MODE or the conflicting "
                "feature."
            )

        # TODO.md J14c Milestone 1: DSPARK TARGET_VERIFY under DP attention,
        # captured with SUM_LEN dp_padding_mode bucketed on the cross-rank
        # total token count, instead of the MAX_LEN forced by
        # DpPaddingMode.get_default_mode_in_cuda_graph. See
        # dspark_dp_verify_sum_capture_enabled's docstring; off by default.
        self.dspark_dp_verify_sum_capture_enabled = dspark_dp_verify_sum_capture_enabled(
            spec_algorithm=self.model_runner.spec_algorithm,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            is_target_verify_capture=(
                self.capture_forward_mode == ForwardMode.TARGET_VERIFY
            ),
            is_draft_worker=self.model_runner.is_draft_worker,
        )
        _validate_compact_ragged_graph_experiments(
            ragged_verify_mode=self.ragged_verify_mode,
            dp_verify_sum_capture=self.dspark_dp_verify_sum_capture_enabled,
        )
        # Compact request boundaries vary inside a total-token bucket, while
        # TBO child tensor views are fixed when the graph is captured. SUM_LEN
        # capture likewise permits different local buckets across DP ranks,
        # invalidating _sync_tbo_child_dp_lens's capture-time replicated child
        # sizes. The attention backend also has its own audited graph-support
        # flag. Keep that gate in the runner so "TBO graph enabled" means that
        # both the model body and child attention backends actually capture
        # TBO. Global TBO remains enabled for exact eager fallbacks.
        primary_attn_backend = getattr(
            self.attn_backend, "primary", self.attn_backend
        )
        attention_backend_supports_tbo_graph = bool(
            getattr(primary_attn_backend, "tbo_supports_cuda_graph", True)
        )
        self.tbo_cuda_graph_enabled = _tbo_cuda_graph_enabled(
            enable_two_batch_overlap=self.enable_two_batch_overlap,
            ragged_verify_mode=self.ragged_verify_mode,
            dp_verify_sum_capture=self.dspark_dp_verify_sum_capture_enabled,
            attention_backend_supports_tbo_graph=(
                attention_backend_supports_tbo_graph
            ),
        )
        self.dp_verify_sum_buckets: Optional[dict[int, list[int]]] = (
            (
                build_ragged_dp_verify_sum_bucket_ladder(
                    self.capture_num_tokens,
                    self.dp_size,
                    envs.SGLANG_DSPARK_DP_VERIFY_SUM_WINDOW.get(),
                )
                if self.ragged_verify_mode
                else build_dp_verify_sum_bucket_ladder(
                    self.capture_bs,
                    self.dp_size,
                    envs.SGLANG_DSPARK_DP_VERIFY_SUM_WINDOW.get(),
                    self.captured_req_width,
                )
            )
            if self.dspark_dp_verify_sum_capture_enabled
            else None
        )
        # Attention backend
        self.max_num_token = (
            max(self.capture_num_tokens)
            if self.capture_num_tokens is not None
            else self.max_bs * self.captured_req_width
        )
        # This one allocation is sized by context length, not by the tier
        # ladder: flashinfer's cuda_graph_kv_indices is
        # max_num_tokens * max_context_len * 4B with no page_size divisor, then
        # cloned per wrapper -- 1 GiB per wrapper at 256K and 4 GiB at 1M, which
        # is what OOM-killed the 1M boot on 2026-09-04. It runs once, before the
        # tier loop, so the per-shape skip below cannot catch it. Losing it
        # costs graphs, not correctness: no shape key is ever registered, so
        # can_run_graph refuses every batch and decode runs eager.
        self._cuda_graph_state_ready = True
        try:
            self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
        except Exception as err:
            if not _is_cuda_oom(err):
                raise
            self._cuda_graph_state_ready = False
            logger.warning(
                "CUDA graph state allocation OOM'd at max_bs=%s "
                "max_num_token=%s (%r) -- DISABLING CUDA graphs; decode will "
                "run eager. To recover them, lower CONTEXT_LENGTH or "
                "CUDA_GRAPH_MAX_BS so the graph KV-index buffer fits.",
                self.max_bs,
                self.max_num_token,
                err,
            )
        self._warn_if_dp_verify_sum_graph_coverage_is_partial()

        # Init PDMux if needed
        self.maybe_init_pdmux()
        self.seq_len_fill_value = (
            self.attn_backend.get_cuda_graph_seq_len_fill_value()
            if self.dllm_config is None
            else self.dllm_config.block_size
        )

        # Non-zero encoder length ensures cross-attention kernels are captured in the graph.
        self.encoder_len_fill_value = (
            getattr(model_runner.model_config.hf_config, "max_source_positions", 0)
            if self.is_encoder_decoder
            else 0
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.lora_manager is not None:
            # Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.
            # Phase 1 (MoE buffers) was handled earlier in ModelRunner via
            # lora_manager.init_cuda_graph_moe_buffers().
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_req=self.captured_req_width,
            )

        enable_mamba_track = (
            get_exec().mamba.enable_mamba_extra_buffer
            and self.model_runner.spec_algorithm.is_none()
        )

        if self.require_gathered_buffer:
            assert self.require_mlp_tp_gather or self.require_attn_tp_gather

        # --- buffers ---------------------------------------------------
        next_token_logits_buffer = self.model_runner.graph_shared_output.get_logits_buffer(
            self.model_runner.model_config.vocab_size, rows=self.max_num_token
        )
        self.buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=self.model_runner.model_config.hidden_size,
            next_token_logits_buffer=next_token_logits_buffer,
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_req=self.captured_req_width,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            ne_token_table=(
                model_runner.ngram_embedding_manager.table
                if self.use_ngram_embedding
                else None
            ),
            hc_hidden_size=getattr(
                self.model_runner.model_config, "hc_hidden_size", None
            ),
            pp_proxy_topk_size=self.model_runner.get_pp_proxy_topk_size(),
            pp_proxy_residual_num_blocks=(
                self.model_runner.get_pp_proxy_residual_num_blocks()
            ),
            pp_proxy_dspark_hidden_size=(
                self.model_runner.get_pp_proxy_dspark_hidden_size()
            ),
        )
        self.buffers.share_buffers()
        # FB-shared slot registry adopting DecodeInputBuffers storage (same
        # physical tensors, stable data_ptr for capture vs replay). Provides
        # the unified fill_from / slot access surface for capture/replay.
        self.buffer_registry: CudaGraphBufferRegistry = build_decode_registry(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            seq_len_fill_value=self.seq_len_fill_value,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            is_encoder_decoder=self.is_encoder_decoder,
            encoder_len_fill_value=self.encoder_len_fill_value,
            enable_num_token_non_padded=enable_num_token_non_padded(),
            require_gathered_buffer=self.require_gathered_buffer,
            enable_prefill_cp=self.enable_prefill_cp,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            attn_tp_sharded_fn=self.model_runner.attn_tp_sequence_sharded,
            dp_size=self.dp_size,
            source=self.buffers,
            dspark_dp_verify_sum_capture_enabled=self.dspark_dp_verify_sum_capture_enabled,
        )
        self._dspark_metadata_double_buffer = bool(
            envs.SGLANG_DSPARK_DOUBLE_BUFFER_METADATA.get()
            and self.model_runner.spec_algorithm.is_dspark()
            and not self.model_runner.is_draft_worker
        )
        self._metadata_timing_interval = (
            envs.SGLANG_DSPARK_METADATA_TIMING_INTERVAL.get()
        )
        self._metadata_timing_count = 0
        self._metadata_timing_sum = defaultdict(float)
        self._metadata_timing_max = defaultdict(float)
        self._metadata_gpu_samples = deque()
        self._staged_batches = {}
        self._metadata_stage_cursor = 0
        if self._dspark_metadata_double_buffer:
            self._metadata_stream = torch.cuda.Stream(device=self.device)
            self._metadata_stages = []
            for _ in range(2):
                registry = build_decode_registry(
                    device=self.device,
                    max_bs=self.max_bs,
                    max_num_token=self.max_num_token,
                    seq_len_fill_value=self.seq_len_fill_value,
                    cache_loc_dtype=self._cache_loc_dtype(),
                    enable_mamba_track=enable_mamba_track,
                    is_encoder_decoder=self.is_encoder_decoder,
                    encoder_len_fill_value=self.encoder_len_fill_value,
                    enable_num_token_non_padded=enable_num_token_non_padded(),
                    require_gathered_buffer=self.require_gathered_buffer,
                    enable_prefill_cp=self.enable_prefill_cp,
                    require_mlp_tp_gather=self.require_mlp_tp_gather,
                    dp_size=self.dp_size,
                    share_pool=False,
                    source=None,
                    dspark_dp_verify_sum_capture_enabled=self.dspark_dp_verify_sum_capture_enabled,
                )
                self._metadata_stages.append(
                    {
                        "registry": registry,
                        "consumed": None,
                        "producer_ready": None,
                        "dispatch_state": None,
                        "ragged_layouts": {},
                    }
                )
            logger.info(
                "Enabled DSPARK double-buffered CUDA-graph metadata preparation"
            )

        # Captures the per-replay attention-metadata prep into a small CUDA
        # graph; see metadata_glue_graph.py for the correctness contract.
        # Force-off for DFlash-family spec: verify installs host-fed fast
        # plans (sync-free begin_forward that recomputes plan inputs on the
        # host every replay), and capturing one freezes the capture-time
        # plan — drafts go stale and accept length collapses to ~1.
        enable_metadata_glue = envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get()
        if enable_metadata_glue and model_runner.spec_algorithm.is_dflash_family():
            logger.warning(
                "SGLANG_ENABLE_METADATA_GLUE_GRAPH is incompatible with "
                "DFlash-family speculative decoding (host-fed fast verify "
                "plans must re-run on the host every replay); disabling the "
                "metadata glue graph."
            )
            enable_metadata_glue = False
        self._metadata_glue = (
            MetadataGlueGraph(self.device) if enable_metadata_glue else None
        )

        # --- backend ---------------------------------------------------
        self.backend = resolve_decode_backend(self)

        # --- capture --------------------------------------------------
        self._skipped_capture_shapes: list = []
        # Padded batch sizes whose capture OOM'd and so have no graph;
        # consulted by can_run_graph so such a bucket is never admitted to
        # replay.
        self._skipped_bs = frozenset()
        if not self._cuda_graph_state_ready:
            logger.warning(
                "Skipping CUDA graph capture entirely: graph state is "
                "unavailable (see the OOM above). Decode runs eager."
            )
        else:
            try:
                with model_capture_mode():
                    self.capture()
            except RuntimeError as e:
                raise Exception(
                    f"Capture cuda graph failed: {e}\n"
                    f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
                )
        if self._dspark_metadata_double_buffer and self.ragged_verify_mode:
            self._initialize_metadata_stage_ragged_layouts()

    def _next_token_logits_buffer_capacity_rows(self, max_num_tokens: int) -> int:
        """Rows reserved for the largest shared logits output."""
        return max_num_tokens

    def _record_in_graph_metadata_prep_done(self):
        # Purely a marker at this point in the graph; where the shared reads
        # actually end is the attn backend's call.
        if not self.device_module.is_current_stream_capturing():
            # Warmup shares this body. Breakable capture still plants: it opens
            # segment 1 on context entry and every segment re-arms the node.
            # Routed through device_module so XPU (torch.xpu) is picked up
            # instead of hitting torch.cuda dummy stubs on non-CUDA builds.
            return
        if self.in_graph_metadata_prep_done is None:
            self.in_graph_metadata_prep_done = make_external_event(self.device_module)
        event = self.in_graph_metadata_prep_done
        if event is not None:
            # Stays None without external-event support, so the read-end
            # resolution below never hands out an unrecorded event.
            event.record()

    def _replay_attn_backend(self) -> AttentionBackend:
        # Under pdmux each stream replays on its own group member.
        if self.enable_pdmux:
            return self.model_runner.decode_attn_backend_group[get_current_stream_idx()]
        return self.attn_backend

    def _resolve_shared_read_ends(self, attn_backend, forward_mode) -> SharedReadEnds:
        declared = attn_backend.shared_read_ends(forward_mode)
        if (
            declared is SharedReadEnds.IN_REPLAY
            and self.in_graph_metadata_prep_done is None
        ):
            # TODO: this lands EARLIER than declared; POST_REPLAY is the sound one.
            return SharedReadEnds.PRE_REPLAY
        return declared

    def _publish_read_done(self, in_graph: bool):
        """Hand the scheduler's WAR barrier the event marking this phase's
        shared-buffer reads as done."""
        if in_graph:
            # Reads end at the in-graph marker: wire it through, don't re-record.
            self.model_runner.shared_read_done_event = self.in_graph_metadata_prep_done
        else:
            read_done = self.device_module.Event()
            read_done.record()
            self.model_runner.shared_read_done_event = read_done

    def _build_ragged_verify_token_buckets(self) -> list[int]:
        buckets = sorted({bs * self.captured_req_width for bs in self.capture_bs})
        assert buckets and buckets[0] > 0, f"{buckets=}"
        return buckets

    def _autotune_buffers(self):
        """Reuse these static decode buffers (sized to max_bs) for the warmup
        flashinfer-autotune dummy forward instead of allocating a throwaway set
        — see BaseRunner._autotune_buffers / BaseRunner._dummy_run.

        The dummy forward derives its shape from max_bs and must match these
        buffers exactly; _dummy_run asserts that. Every autotune-reachable
        decode shape (plain decode, spec target-verify) matches. DLLM would not
        (its buffers hold block_size tokens/bs while the dummy run derives 1),
        but DLLM does not use a flashinfer MoE backend, so autotune never runs
        for it and this is never reached there.
        """
        return self.buffers, self.max_bs

    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                # Same context-sized allocation as the primary backend above,
                # once per stream group; degrade to eager rather than die.
                try:
                    attn_backend.init_cuda_graph_state(
                        self.max_bs, self.max_num_token
                    )
                except Exception as err:
                    if not _is_cuda_oom(err):
                        raise
                    self._cuda_graph_state_ready = False
                    logger.warning(
                        "CUDA graph state allocation OOM'd for a PDMux stream "
                        "group at max_bs=%s max_num_token=%s (%r) -- DISABLING "
                        "CUDA graphs; decode will run eager.",
                        self.max_bs,
                        self.max_num_token,
                        err,
                    )
                    return

    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(
        self,
        size,
        stream_idx=None,
        variant_label=None,
        attention_variant=None,
        sum_bucket=None,
    ):
        return ShapeKey(
            size=size,
            stream_idx=stream_idx,
            variant_label=variant_label,
            attention_variant=attention_variant,
            sum_bucket=sum_bucket,
        )

    def _capture_graph_size(self, *, bs: int, num_tokens: int) -> int:
        return num_tokens if self.ragged_verify_mode else bs

    def _global_num_tokens_for_graph(self, num_tokens: int) -> Optional[list[int]]:
        if self.require_mlp_tp_gather:
            return [num_tokens] * self.dp_size
        if self.require_attn_tp_gather:
            return [num_tokens]
        return None

    def _resolve_attention_variant(self, forward_batch: ForwardBatch) -> Optional[str]:
        variants = self.attention_graph_variants
        return variants.select(forward_batch) if variants is not None else None

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if not self.record_nolora_graph:
            return None
        if forward_batch.lora_ids is not None and any(
            uid is not None for uid in forward_batch.lora_ids
        ):
            return "lora"
        return "nolora"

    @staticmethod
    def _forward_is_dp_local(model_runner) -> bool:
        """The DSpark dense draft runs attn-TP-local (draft_tp_context): each
        DP rank drafts independently with no cross-DP collective, so its
        hand-built batches carry no dp-global metadata and must key graphs by
        local batch size. Everything else keeps the dp-global padding path."""
        if not model_runner.is_draft_worker:
            return False
        if not model_runner.spec_algorithm.is_dspark():
            return False
        from sglang.srt.speculative.dspark_components.dspark_config import (
            draft_is_deepseek_v4,
        )

        return not draft_is_deepseek_v4()

    def _ragged_capture_slots(self, num_tokens: int) -> int:
        if envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get():
            return num_tokens // self.captured_req_width
        return min(num_tokens, self.max_bs)

    def _all_compact_dp_ranks_fit_graph_slots(
        self,
        *,
        global_num_reqs: Optional[list[int]],
        global_graph_num_tokens: Optional[list[int]],
    ) -> bool:
        """Rank-identical capacity gate for compact SUM graph admission.

        A token bucket can be inside the captured grid while still exposing
        fewer request slots than a rank needs (for example, 129 one-token
        requests rounded to a 144-token bucket with ``max_bs == 128``).  The
        local ``batch_size`` check alone would make only that rank fall back
        to eager while its peers replay a collective-bearing graph.  Both
        vectors are already DP-shared, so validate every rank structurally.
        """
        if (
            not global_num_reqs
            or not global_graph_num_tokens
            or len(global_num_reqs) != len(global_graph_num_tokens)
        ):
            return False
        return all(
            0 <= int(num_reqs) <= self._ragged_capture_slots(int(num_tokens))
            for num_reqs, num_tokens in zip(
                global_num_reqs, global_graph_num_tokens
            )
        )

    def _ragged_layout_key(
        self, num_tokens: int, stream_idx: Optional[int] = None
    ) -> tuple[Optional[int], int]:
        if self.enable_pdmux:
            if stream_idx is None:
                stream_idx = get_current_stream_idx()
            if stream_idx is None:
                raise RuntimeError("PDMux ragged layout requires an execution lane")
        else:
            stream_idx = None
        return stream_idx, num_tokens

    def _capture_ragged_verify_layout(
        self, num_tokens: int, stream_idx: Optional[int] = None
    ):
        if not self.ragged_verify_mode:
            return None
        if envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get():
            return None
        # Sequence-length tiering captures several graphs for the same token
        # bucket. They must all retain the same staged layout buffers: making
        # a fresh layout here would overwrite the sole dictionary entry, so
        # replay would refresh only the last tier while earlier graphs kept
        # their synthetic capture-time indptr (and raw pointers whose tensor
        # owners had been dropped). Replays are serialized on this runner, so
        # sharing one buffer set across tiers is both sufficient and cheaper.
        layout_key = self._ragged_layout_key(num_tokens, stream_idx)
        layout = self._captured_ragged_layouts.get(layout_key)
        if layout is not None:
            return layout
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyLayout,
            build_capture_verify_lens,
        )

        verify_lens_cpu = build_capture_verify_lens(
            num_tokens=num_tokens,
            num_slots=self._ragged_capture_slots(num_tokens),
            num_draft_tokens=self.captured_req_width,
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=self.device,
            grid=self.capture_num_tokens,
        )
        self._captured_ragged_layouts[layout_key] = layout
        return layout

    def _new_private_ragged_layout(self, captured_layout):
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        return RaggedVerifyLayout(
            verify_lens=torch.empty_like(captured_layout.verify_lens),
            graph_num_tokens=captured_layout.graph_num_tokens,
            extend_start_loc=torch.empty_like(captured_layout.extend_start_loc),
            qo_indptr_device=torch.empty_like(captured_layout.qo_indptr_device),
            cap=self.captured_req_width,
        )

    def _initialize_metadata_stage_ragged_layouts(self) -> None:
        """Give each alternating metadata stage private graph-layout storage."""
        for stage in self._metadata_stages:
            layouts = stage["ragged_layouts"]
            for layout_key, captured_layout in self._captured_ragged_layouts.items():
                layouts[layout_key] = self._new_private_ragged_layout(captured_layout)

    @staticmethod
    def _copy_ragged_layout_buffers(destination, source) -> None:
        destination.verify_lens.copy_(source.verify_lens)
        destination.extend_start_loc.copy_(source.extend_start_loc)
        destination.qo_indptr_device.copy_(source.qo_indptr_device)

    def _prepare_ragged_layout_for_graph(self, ragged_layout, captured_layout):
        layout = ragged_layout
        if layout.bs != captured_layout.bs or layout.cap is None:
            layout = layout.padded_to_bucket(
                padded_bs=captured_layout.bs, cap=self.captured_req_width
            )
        return layout

    def _stage_ragged_verify_layout(self, ragged_layout, graph_size_key: int) -> None:
        """Refresh graph-owned layout buffers on the replay's current stream."""
        layout_key = self._ragged_layout_key(graph_size_key)
        captured_layout = self._captured_ragged_layouts.get(layout_key)
        if captured_layout is None:
            return
        prepared = self._prepare_ragged_layout_for_graph(
            ragged_layout, captured_layout
        )
        self._copy_ragged_layout_buffers(captured_layout, prepared)

    def _stage_private_ragged_verify_layout(
        self, stage: dict, ragged_layout, graph_size_key: int
    ) -> tuple[Optional[object], Optional[tuple[Optional[int], int]]]:
        """Fill a stage-owned snapshot without touching graph-owned buffers."""
        layout_key = self._ragged_layout_key(graph_size_key)
        captured_layout = self._captured_ragged_layouts.get(layout_key)
        if captured_layout is None:
            return None, None
        private_layout = stage["ragged_layouts"].get(layout_key)
        if private_layout is None:
            raise RuntimeError(
                "metadata stage has no private ragged layout for captured key "
                f"{layout_key}; all graph layouts must be allocated at startup"
            )
        prepared = self._prepare_ragged_layout_for_graph(
            ragged_layout, captured_layout
        )
        self._copy_ragged_layout_buffers(private_layout, prepared)
        return private_layout, layout_key

    def _promote_private_ragged_verify_layout(
        self,
        private_layout,
        layout_key: tuple[Optional[int], int],
    ) -> None:
        captured_layout = self._captured_ragged_layouts.get(layout_key)
        if captured_layout is None:
            raise RuntimeError(
                "captured ragged layout disappeared before metadata promotion: "
                f"{layout_key}"
            )
        self._copy_ragged_layout_buffers(captured_layout, private_layout)

    @staticmethod
    def _max_dp_batch_size(forward_batch: ForwardBatch) -> int:
        request_counts = forward_batch.original_global_num_tokens_cpu
        if request_counts is None:
            raise RuntimeError(
                "DP CUDA graph replay requires raw per-rank request counts"
            )
        return max(request_counts)

    def can_run_graph(self, forward_batch: ForwardBatch):
        # No graph state means no shape was ever captured. Must be refused
        # here rather than left to backend.can_run: with padding enabled the
        # is_bs_supported branch below never consults the backend, so a "yes"
        # reaches replay's bare self._graphs[shape_key] as a KeyError.
        if not self._cuda_graph_state_ready:
            return False
        # Disable for token embedding overrides (dynamic per-request)
        if forward_batch.replace_embeds is not None:
            return False
        # The caller asked for an eager forward (e.g. a graph-vs-eager
        # comparison of the same verify batch).
        if forward_batch.force_eager_decode:
            return False

        ragged_layout = (
            resolve_ragged_verify_layout(forward_batch)
            if self.ragged_verify_mode
            else None
        )
        if ragged_layout is not None:
            return self._can_run_ragged_verify_graph(forward_batch, ragged_layout)
        if self.ragged_verify_mode and forward_batch.forward_mode.is_target_verify():
            return False

        # Uniform-width replay invariant: the batch's actual per-request width
        # must match this runner's capture width; anything else falls back to
        # eager. (Unset widths pass: not every path fills the field yet.)
        spec_info = forward_batch.spec_info
        if (
            spec_info is not None
            and spec_info.num_tokens_per_req > 0
            and spec_info.num_tokens_per_req != self.captured_req_width
        ):
            return False

        if self.dspark_dp_verify_sum_capture_enabled:
            return self._can_run_dp_verify_sum_graph(forward_batch)

        if self.require_mlp_tp_gather:
            cuda_graph_bs = self._max_dp_batch_size(forward_batch)
        else:
            cuda_graph_bs = forward_batch.batch_size

        graph_key = self._make_graph_key(
            cuda_graph_bs,
            stream_idx=get_current_stream_idx() if self.enable_pdmux else None,
            variant_label=self._resolve_lora_variant(forward_batch),
            attention_variant=(
                self._resolve_attention_variant(forward_batch)
                if self.disable_padding
                else None
            ),
        )

        is_bs_supported = (
            self.backend.can_run(forward_batch, graph_key)
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        # A shape skipped for OOM stays in capture_bs, so with padding enabled
        # the branch above still pads onto it and replay indexes a graph that
        # was never captured. Empty on a healthy boot, so this changes nothing
        # there; on a degraded one, answering "eager" is always safe.
        if is_bs_supported and self._skipped_bs:
            padded_bs = self._pad_to_bucket(cuda_graph_bs, self.capture_bs)
            if padded_bs in self._skipped_bs:
                return False

        if self.require_mlp_sync:
            is_bs_supported = (
                is_bs_supported and forward_batch.can_run_decode_cuda_graph
            )

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        is_tbo_supported = (
            forward_batch.can_run_tbo if self.tbo_cuda_graph_enabled else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.captured_req_width
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and is_ngram_supported
        )

    def _warn_if_dp_verify_sum_graph_coverage_is_partial(self) -> None:
        """Warn when the SUM_LEN verify graphs cannot cover every batch this
        scheduler can build, because a miss is far more expensive here than
        the usual "just run eager".

        can_run_graph returning False is a correct, already-tested fallback in
        isolation. Under DP attention it is not free: the ranks advance in
        lockstep, so a rank that drops to eager and hits a *cold* Triton
        autotune for a query shape the captured graphs never warmed (e.g. a
        sparse score kernel autotuned on the query length) blocks every
        other rank inside its DP collective for as long as that compile takes.
        TODO.md J14d caught exactly this: one rank sat in cuModuleLoadData for
        10+ minutes while the other three waited in all_gather/broadcast.

        Compare against model_runner's *resolved* max_running_requests, not
        the server_args one: SGLang divides the user-supplied value across DP
        ranks (--max-running-requests 32 at dp_size 4 admits 8 per scheduler),
        and capture_bs is clamped to the same per-rank bound via
        req_to_token_pool.size. Reading the undivided value made this warn on
        correctly-configured DP deployments, where capture_bs tops out at
        exactly the per-rank limit and coverage is in fact complete.

        Warn instead of failing closed: a deliberately shrunken ladder is a
        valid diagnostic configuration, it is only the silent version that is
        a trap.
        """
        if not self.dspark_dp_verify_sum_capture_enabled:
            return
        max_running_requests = self.model_runner.max_running_requests
        if max_running_requests is None or self.max_bs >= max_running_requests:
            return
        logger.warning(
            "DSPARK DP-verify SUM_LEN capture covers per-rank batches up to "
            "%d requests, but this scheduler admits up to %d. Any step where a "
            "rank exceeds %d falls back to eager; under DP attention that rank "
            "stalls every other rank in its collective while it autotunes any "
            "sparse-attention shape the captured graphs never warmed. Raise "
            "--cuda-graph-max-bs to at least --max-running-requests (%d) for "
            "production runs.",
            self.max_bs,
            max_running_requests,
            self.max_bs,
            max_running_requests,
        )

    def _can_run_dp_verify_sum_graph(self, forward_batch: ForwardBatch) -> bool:
        """TODO.md J14c Milestone 1 admission check: this rank's own local
        bucket and the real cross-rank sum must both fall inside the
        windowed (local_bucket, sum_bucket) capture matrix built at __init__
        (self.dp_verify_sum_buckets). A miss (either bucket, or the local
        request count exceeding max_bs) returns False, same contract as
        every other can_run_graph branch -- the caller falls back to the
        pre-existing eager SUM_LEN path, which is already correct.

        See resolve_dp_verify_sum_graph_buckets for why the per-rank counts
        this reads are request-denominated, and which operand the token
        conversion belongs on.
        """
        if (
            resolve_dp_verify_sum_graph_buckets(
                global_num_reqs=forward_batch.original_global_num_tokens_cpu,
                dp_rank=get_parallel().attn_dp_rank,
                num_tokens_per_req=self.captured_req_width,
                capture_bs=self.capture_bs,
                sum_buckets_by_local=self.dp_verify_sum_buckets,
                max_bs=self.max_bs,
            )
            is None
        ):
            return False

        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )
        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        return is_encoder_lens_supported and capture_hidden_mode_matches

    def _can_run_ragged_verify_graph(self, forward_batch: ForwardBatch, ragged_layout):
        if not self.attn_backend.supports_ragged_verify_graph:
            return False

        admission_tokens = ragged_layout.graph_num_tokens
        is_tokens_supported = admission_tokens <= self.capture_num_tokens[
            -1
        ] and forward_batch.batch_size <= self._ragged_capture_slots(admission_tokens)

        is_dp_supported = (
            forward_batch.can_run_decode_cuda_graph if self.require_mlp_sync else True
        )

        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        capture_hidden_mode_matches = (
            forward_batch.capture_hidden_mode <= self.capture_hidden_mode
        )

        sum_bucket_supported = True
        if self.dspark_dp_verify_sum_capture_enabled:
            global_graph_num_tokens = getattr(
                forward_batch.spec_info, "dp_graph_num_tokens", None
            )
            sum_bucket_supported = (
                resolve_ragged_dp_verify_sum_graph_buckets(
                    global_graph_num_tokens=global_graph_num_tokens,
                    dp_rank=get_parallel().attn_dp_rank,
                    local_graph_num_tokens=admission_tokens,
                    capture_num_tokens=self.capture_num_tokens,
                    sum_buckets_by_local=self.dp_verify_sum_buckets,
                    max_num_tokens=self.capture_num_tokens[-1],
                )
                is not None
                and global_graph_num_tokens == forward_batch.global_num_tokens_cpu
                and self._all_compact_dp_ranks_fit_graph_slots(
                    global_num_reqs=forward_batch.original_global_num_tokens_cpu,
                    global_graph_num_tokens=global_graph_num_tokens,
                )
            )

        return (
            is_tokens_supported
            and is_dp_supported
            and is_encoder_lens_supported
            and capture_hidden_mode_matches
            and sum_bucket_supported
        )

    def _graph_batch_capture_active(self) -> bool:
        """Whether the per-batch-size capture-trace feature is active.

        Gated by SGLANG_GRAPH_BATCH_CAPTURE. The original single-trace export
        (SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE) takes precedence: when both are
        set we fall back to the original behavior.
        """
        return (
            envs.SGLANG_GRAPH_BATCH_CAPTURE.get()
            and not envs.SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE.get()
        )

    def _init_profile_context_and_memory_record(self):
        if self._graph_batch_capture_active():
            # Per-batch-size capture traces (SGLANG_GRAPH_BATCH_CAPTURE): a
            # scheduled profiler is stepped once per batch size (see
            # FullCudaGraphBackend.capture_one) and on_trace_ready writes one
            # chrome trace per bs.
            rank = get_parallel().tp_rank
            runner_name = type(self).__name__
            trace_dir = graph_capture_profile_dir()
            os.makedirs(trace_dir, exist_ok=True)

            # Track which BS is currently being captured for trace file naming
            self._profile_bs_list = list(reversed(self.capture_bs))
            self._profile_bs_idx = 0

            def on_trace_ready(prof):
                bs = self._profile_bs_list[self._profile_bs_idx]
                trace_file = os.path.join(
                    trace_dir, f"{runner_name}_bs_{bs}_rank{rank}.json.gz"
                )
                prof.export_chrome_trace(trace_file)
                logger.info(f"Saved trace for bs={bs} to {trace_file}")
                self._profile_bs_idx += 1

            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                # Schedule: wait=2 (skip 2 dummy runs), warmup=0, active=1
                # (capture run); repeat=0 repeats the cycle so each batch size
                # gets its own trace.
                schedule=torch.profiler.schedule(wait=2, warmup=0, active=1, repeat=0),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                profile_memory=True,
                on_trace_ready=on_trace_ready,
            )
        else:
            # a single unscheduled pass over the whole
            # capture. The combined trace (if any) is exported in
            # _post_process_after_profile via export_cuda_graph_capture_trace,
            # gated by SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE.
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            )
        torch.cuda.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.cuda.memory._dump_snapshot("cuda_graph_runner_memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by CUDA Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)

        # single-trace export for the whole capture pass; no-op unless
        # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE is set. In per-bs mode
        # (SGLANG_GRAPH_BATCH_CAPTURE) that env is unset, so this stays a no-op
        # and the per-bs on_trace_ready handles export instead.
        export_cuda_graph_capture_trace(
            prof_context,
            runner_name=type(self).__name__,
        )

    def capture_prepare(
        self,
        size: int,
        stream_idx: Optional[int] = None,
        num_tokens: Optional[int] = None,
        sum_bucket: Optional[int] = None,
    ):
        """Build the dummy decode ForwardBatch for capture at size (=bs),
        populate static input buffers, choose the active attn backend, and
        optionally build pp_proxy_tensors.

        num_tokens defaults to the uniform bs * num_tokens_per_req; ragged
        verify capture passes the decoupled (slots, tier tokens) pair.

        sum_bucket (TODO.md J14c Milestone 1): when set (only reachable via
        self.dspark_dp_verify_sum_capture_enabled), capture this DP-attention
        DSPARK verify shape with SUM_LEN dp_padding_mode against a
        cross-rank total of `sum_bucket` tokens, instead of the MAX_LEN
        capture's `[num_tokens] * dp_size` assumption. None (default)
        preserves existing MAX_LEN behavior exactly.

        Returns (forward_batch, attn_backend, pp_proxy_tensors);
        pp_proxy_tensors is None unless pp_size > 1.
        """
        bs = size
        buffers: DecodeInputBuffers = self.buffers
        if num_tokens is None:
            num_tokens = bs * self.captured_req_width

        # Registry-owned FB-shared slots come through the registry (which
        # shares physical storage with self.buffers via source=...); the rest
        # still come off buffers directly.
        registry = self.buffer_registry

        def _slot(name):
            return registry.get_slot(name).slice_for(bs, num_tokens)

        input_ids = _slot("input_ids")
        req_pool_indices = _slot("req_pool_indices")
        seq_lens = _slot("seq_lens")
        seq_lens_cpu = _slot("seq_lens_cpu")
        out_cache_loc = _slot("out_cache_loc")
        positions = _slot("positions")
        encoder_lens = (
            _slot("encoder_lens") if registry.has_slot("encoder_lens") else None
        )
        mrope_positions = _slot("mrope_positions")
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
        rids_int = buffers.rids_int[:bs] if buffers.rids_int is not None else None
        bootstrap_room_ids_int = (
            buffers.bootstrap_room_ids_int[:bs]
            if buffers.bootstrap_room_ids_int is not None
            else None
        )

        # Localize the count when this bucket is attn-TP sharded (SP on).
        attn_tp_sharded = self.model_runner.attn_tp_sequence_sharded(num_tokens)
        if buffers.num_token_non_padded is not None:
            buffers.num_token_non_padded[...] = num_tokens
            if not self.enable_prefill_cp and attn_tp_sharded:
                local = compute_local_num_token_non_padded(
                    global_num_token_non_padded=buffers.num_token_non_padded,
                    num_tokens_per_dp=num_tokens,
                    sharded=True,
                )
                buffers.num_token_non_padded.copy_(local)

        pp_proxy_tensors = None
        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather and sum_bucket is not None:
            global_num_tokens_cpu = synthetic_dp_verify_sum_split(
                num_tokens, sum_bucket, self.dp_size, get_parallel().attn_dp_rank
            )
        else:
            global_num_tokens_cpu = self._global_num_tokens_for_graph(num_tokens)

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=input_ids.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens, stream_idx=stream_idx)
        self.capture_hidden_mode = get_required_capture_hidden_mode(
            self.capture_hidden_mode,
            spec_info,
        )

        if self.model_runner.lora_manager is not None:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # mamba state tracking (registry-owned when enabled)
        mamba_track_indices = (
            _slot("mamba_track_indices")
            if registry.has_slot("mamba_track_indices")
            else None
        )
        mamba_track_mask = (
            _slot("mamba_track_mask") if registry.has_slot("mamba_track_mask") else None
        )

        if stream_idx is None:
            attn_backend = self.attn_backend
        else:
            assert self.enable_pdmux
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=None,
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=(
                DpPaddingMode.SUM_LEN
                if sum_bucket is not None
                else DpPaddingMode.get_default_mode_in_cuda_graph()
            ),
            global_dp_buffer_len=global_dp_buffer_len,
            global_num_tokens_cpu=global_num_tokens_cpu,
            can_run_decode_cuda_graph=True,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            # Maintained only under expert parallelism; None elsewhere so routing
            # does not mask every row against a never-filled zero count.
            num_token_non_padded=(
                buffers.num_token_non_padded if enable_num_token_non_padded() else None
            ),
            attn_tp_sequence_sharded=attn_tp_sharded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
        )

        # Trip the coordinator so the hisparse code path is captured into the
        # graph; backends read it from self.model_runner.hisparse_coordinator.
        forward_batch.hisparse_coordinator = self.model_runner.hisparse_coordinator
        if forward_batch.hisparse_coordinator is not None:
            forward_batch.hisparse_coordinator.num_real_reqs.fill_(bs)

        if buffers.ngram_embedding_info is not None:
            forward_batch.ngram_embedding_info = buffers.ngram_embedding_info.slice(bs)

        return forward_batch, attn_backend, pp_proxy_tensors

    def capture(self) -> None:
        self.warmup()
        # warmup() may disable torch.compile for a model whose _can_torch_compile
        # is False; recompute the compile bucket so capture matches.
        if self.enable_torch_compile and not (get_flags().capture.enable_torch_compile):
            self.enable_torch_compile = False
            _, self.compile_bs = get_batch_sizes_to_capture(
                self.model_runner, self.captured_req_width
            )
        profile_context = empty_context()
        # Holds the active torch profiler during capture so the backend can
        # advance its schedule (profiler.step()) per batch size. Only the
        # scheduled per-bs profiler (SGLANG_GRAPH_BATCH_CAPTURE) needs stepping;
        # the original unscheduled pass leaves this None.
        self._profiler = None
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()
            if self._graph_batch_capture_active():
                self._profiler = profile_context

        # share_buffers() coalesces seq_lens / seq_lens_cpu through the process-
        # wide pool, so they may alias a buffer seeded by an earlier runner (the
        # eager registry fills them with 0). The capture-time attention-metadata
        # plan reads these as the per-request KV length, and the prefill wrapper
        # (DLLM_EXTEND) asserts kv_len >= qo_len, so restore the fill value the
        # captured graph needs before capturing.
        self.buffers.seq_lens.fill_(self.seq_len_fill_value)
        self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
        # Capture runs real forwards, so a mid-serving recapture would index --
        # and write KV -- through the previous batch's live values. Replay is
        # already covered by the registry's padding policy.
        self.buffers.reset_index_buffers()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        _maybe_dump_pool_ptrs(model_runner=self.model_runner, tag="capture")
        with freeze_gc(get_exec().graph.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with (
                    graph_capture(
                        stream=get_or_create_global_graph_capture_stream()
                    ) as graph_capture_context,
                    profile_context as prof,
                ):
                    self.stream = graph_capture_context.stream
                    with self.backend.capture_session(self.stream):
                        self._capture_one_stream()
            else:
                for i, sg in enumerate(self.stream_groups):
                    with (
                        graph_capture(stream=sg[1]) as graph_capture_context,
                        profile_context as prof,
                    ):
                        self.stream = graph_capture_context.stream
                        with self.backend.capture_session(self.stream):
                            self._capture_one_stream(i)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)
        self._profiler = None

        # No pool-side pin to clear: the captured full-physical write loc rides the
        # backend's `ForwardMetadata.out_cache_loc_full_physical` (-> KVWriteLoc.full_loc).

        skipped = getattr(self, "_skipped_capture_shapes", [])
        if skipped:
            logger.warning(
                "CUDA graph capture finished with %d shape(s) SKIPPED due to "
                "OOM; those batch shapes will replay eager: bs %s. To recover "
                "them, lower MAX_TOTAL_TOKENS or the CUDA graph max batch size.",
                len(skipped),
                sorted({_bs for _bs, _nt, _sb in skipped}),
            )

    def _capture_one_stream(self, stream_idx: Optional[int] = None) -> None:
        avail_mem = get_available_gpu_memory(
            self.model_runner.device,
            self.model_runner.gpu_id,
            empty_cache=False,
        )
        lora_variants = (
            [("lora", True), ("nolora", False)]
            if self.record_nolora_graph
            else [(None, None)]
        )
        variants = self.attention_graph_variants
        attention_variants = (
            variants.capture_labels if variants is not None else (None,)
        )
        # Reverse so cuda graphs share memory better.
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_parallel().tp_rank == 0
            else reversed(self.capture_bs)
        )
        for bs in capture_range:
            if get_parallel().tp_rank == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                capture_range.set_description(
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )

            # TODO.md J14c Milestone 1: capture one graph per sum_bucket
            # for this local bucket instead of the single MAX_LEN-implicit
            # shape. [None] preserves existing behavior exactly
            # (capture_one_shape's sum_bucket=None path).
            #
            # The ladder is shared by every local bucket so that lookup
            # agrees across ranks (build_dp_verify_sum_bucket_ladder), which
            # leaves corners that cannot physically occur. Skip any total
            # too small to give every rank a canonical slot covering this
            # local bucket (synthetic_dp_verify_sum_split is rank-invariant,
            # so its per-rank slot is total / dp_size, and it asserts this
            # same precondition). resolve_dp_verify_sum_graph_buckets can
            # never return a skipped one: the single ladder entry is
            # dp_size times the largest local capture-token bucket, which
            # always passes for every capturable static or compact shape.
            capture_num_tokens = bs * self.captured_req_width
            local_sum_key = capture_num_tokens if self.ragged_verify_mode else bs
            sum_bucket_range = (
                [
                    sum_bucket
                    for sum_bucket in self.dp_verify_sum_buckets.get(
                        local_sum_key, []
                    )
                    if sum_bucket >= self.dp_size * capture_num_tokens
                ]
                if self.dspark_dp_verify_sum_capture_enabled
                else [None]
            )
            for sum_bucket in sum_bucket_range:
                for variant_label, _variant_has_lora in lora_variants:
                    _set_capture_lora_variant(variant_label)
                    for attention_variant in attention_variants:
                        _set_capture_attention_variant(attention_variant)
                        with torch_compile_decoration.patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs,
                            num_tokens=capture_num_tokens,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            # Guards the whole shape, not just
                            # backend.capture_one: capture_prepare, the
                            # out-graph metadata init and the autotune
                            # warmup all allocate, and any of them can be
                            # the one that OOMs.
                            # Subclass capture_one_shape overrides inherit
                            # the guard by sitting under this call.
                            try:
                                self.capture_one_shape(
                                    bs,
                                    forward,
                                    stream_idx,
                                    variant_label,
                                    attention_variant,
                                    sum_bucket=sum_bucket,
                                )
                            except Exception as err:
                                self._skip_capture_shape_or_raise(
                                    err,
                                    bs,
                                    capture_num_tokens,
                                    sum_bucket,
                                )
        _set_capture_attention_variant(None)

    def _skip_capture_shape_or_raise(
        self,
        err: Exception,
        bs: int,
        num_tokens: int,
        sum_bucket: Optional[int],
    ) -> None:
        """Record an OOM'd capture shape as eager-only, or re-raise a real bug.

        The shape key is only registered in the backend after a successful
        capture, so a skipped shape is simply absent from backend.can_run() and
        those batches replay eager -- slower for them, but the server boots and
        every request stays servable. Anything that is NOT an OOM is a real bug
        and still dies here.
        """
        is_oom = _is_cuda_oom(err)
        mem = ""
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            alloc_b = torch.cuda.memory_allocated()
            resv_b = torch.cuda.memory_reserved()
            mem = (
                " | driver free %.2f/%.2f GB, torch allocated "
                "%.2f GB, torch reserved %.2f GB "
                "(reserved-allocated %.2f GB is "
                "fragmentation/graph-pool)"
                % (
                    free_b / 1e9,
                    total_b / 1e9,
                    alloc_b / 1e9,
                    resv_b / 1e9,
                    (resv_b - alloc_b) / 1e9,
                )
            )
        except Exception:
            mem = " | (memory introspection failed)"
        if not is_oom:
            logger.error(
                "CUDA graph capture failed at bs=%s num_tokens=%s "
                "sum_bucket=%s: %r%s",
                bs,
                num_tokens,
                sum_bucket,
                err,
                mem,
            )
            raise err
        logger.warning(
            "CUDA graph capture OOM at bs=%s num_tokens=%s "
            "sum_bucket=%s -- SKIPPING this shape; "
            "these batches will run eager.%s",
            bs,
            num_tokens,
            sum_bucket,
            mem,
        )
        self._skipped_capture_shapes.append((bs, num_tokens, sum_bucket))
        # capture_bs is never pruned, so this bucket stays a padding target;
        # can_run_graph reads this to keep it out of replay.
        self._skipped_bs = self._skipped_bs | {bs}
        # Give the next shape the best chance: drop whatever the failed
        # capture reserved before moving on.
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
        attention_variant: Optional[str] = None,
        sum_bucket: Optional[int] = None,
    ):
        num_tokens = size * self.captured_req_width
        bs = self._ragged_capture_slots(num_tokens) if self.ragged_verify_mode else size

        # Sanity-check: --debug-cuda-graph requires breakable backend.
        if get_exec().graph.debug_cuda_graph:
            assert isinstance(self.backend, BreakableCudaGraphBackend), (
                "Breakable CUDA graph is required for --debug-cuda-graph"
            )

        forward_batch, attn_backend, pp_proxy_tensors = self.capture_prepare(
            bs,
            stream_idx=stream_idx,
            num_tokens=num_tokens,
            sum_bucket=sum_bucket,
        )

        # All setup hooks below read get_attn_backend() (TboForwardBatchPreparer,
        # DeepEP adapter, …) so they must run inside the same ForwardContext
        # that wraps the warmup/capture forward.
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            # Gated on the runner's own flag, not just is_tbo_enabled(): the
            # draft worker shares this runner class but has no TBO-wrapped
            # backend (see __init__), so its captures must stay non-TBO.
            if self.tbo_cuda_graph_enabled:
                self.tbo_plugin.capture_one_batch_size(
                    forward_batch, num_tokens=num_tokens
                )

            if forward_batch.lora_ids is not None:
                self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

            attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

            def run_once():
                # Graph-recordable metadata-prep hook. The unified memory pool
                # records ZERO translate nodes here: all its read/write translates
                # run eagerly in `init_forward_metadata_out_graph` (replay-prep), so
                # the captured graph reads already-physical locs. Base no-op for triton.
                attn_backend.init_forward_metadata_in_graph(forward_batch)
                self._record_in_graph_metadata_prep_done()

                # No invalidate_loc_cache() here: the unified pool translates its
                # locs in `init_forward_metadata_out_graph`, so no cache to invalidate.

                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )
                set_dp_buffer_len(
                    forward_batch.global_dp_buffer_len,
                    num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                    forward_batch.global_num_tokens_cpu,
                )
                set_is_extend_in_batch(False)

                kwargs = {}
                if (
                    self.pp_size > 1
                    and "pp_proxy_tensors" in inspect.signature(forward).parameters
                ):
                    kwargs["pp_proxy_tensors"] = PPProxyTensors(
                        {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                    )
                if (
                    (
                        self.dllm_uses_input_embeds
                        or (
                            self.model_runner.spec_algorithm.is_dflash_family()
                            and self.model_runner.is_draft_worker
                        )
                    )
                    and "input_embeds" in inspect.signature(forward).parameters
                    and not hasattr(self.model_runner.model, "forward_embed")
                ):
                    kwargs["input_embeds"] = self.buffers.input_embeds[:num_tokens]

                out = forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
                for capture_hook in self.model_runner.capture_tail_hooks:
                    capture_hook(self, out, forward_batch, num_tokens)
                return out

            self.deepep_adapter.capture(is_extend_in_batch=False)
            canary_ctx = (
                c.with_active_single_forward_manager(0)
                if (c := self.model_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            # Full-physical write loc lives in the attention metadata (the backend's
            # `out_cache_loc_full_physical` -> KVWriteLoc.full_loc), so the runner
            # wires no buffer here. (SWA write loc rides the `swa_out_cache_loc` rail.)

            with canary_ctx:
                shape_key = self._make_graph_key(
                    self._capture_graph_size(bs=bs, num_tokens=num_tokens),
                    stream_idx,
                    variant_label,
                    attention_variant,
                    sum_bucket=sum_bucket,
                )
                # Adaptive runners may own a different backend than model_runner.
                post_warmup_hook = getattr(
                    attn_backend,
                    "on_after_cuda_graph_warmup",
                    None,
                )
                maybe_flashinfer_autotune_speculative_draft(
                    self,
                    run_once,
                    post_warmup_hook=post_warmup_hook,
                    run_lm_head=True,
                )
                try:
                    self.backend.capture_one(
                        shape_key,
                        run_once,
                        capture_inputs=None,
                        post_warmup_hook=post_warmup_hook,
                    )
                except Exception as err:
                    self._skip_capture_shape_or_raise(
                        err, bs, num_tokens, sum_bucket
                    )

    def _validate_capture_hidden_mode(self, forward_batch: ForwardBatch) -> None:
        if self.capture_hidden_mode < forward_batch.capture_hidden_mode:
            raise RuntimeError(
                "The runtime hidden-state mode exceeds the fixed CUDA graph "
                f"capture mode ({self.capture_hidden_mode.name})."
            )

    def _snapshot_replay_dispatch_state(
        self,
        *,
        padded_num_tokens: int,
        ragged_layout_key: Optional[tuple[Optional[int], int]] = None,
        staged_ragged_layout: Optional[object] = None,
    ) -> _ReplayDispatchState:
        return _ReplayDispatchState(
            raw_bs=self.raw_bs,
            raw_num_token=self.raw_num_token,
            bs=self.bs,
            ragged_graph_size=self._ragged_graph_size,
            replay_graph_key=self._replay_graph_key,
            replay_dp_verify_sum_bucket=self._replay_dp_verify_sum_bucket,
            padded_num_tokens=padded_num_tokens,
            ragged_layout_key=ragged_layout_key,
            staged_ragged_layout=staged_ragged_layout,
        )

    def _restore_replay_dispatch_state(self, state: _ReplayDispatchState) -> None:
        self.raw_bs = state.raw_bs
        self.raw_num_token = state.raw_num_token
        self.bs = state.bs
        self._ragged_graph_size = state.ragged_graph_size
        self._replay_graph_key = state.replay_graph_key
        self._replay_dp_verify_sum_bucket = state.replay_dp_verify_sum_bucket

    def _promote_staged_batch(
        self, staged: dict, forward_batch: ForwardBatch
    ) -> None:
        stage = staged["stage"]
        state = stage["dispatch_state"]
        if not isinstance(state, _ReplayDispatchState):
            raise RuntimeError("metadata stage has no replay dispatch state")
        if getattr(self, "enable_pdmux", False):
            replay_lane = get_current_stream_idx()
            if replay_lane != state.replay_graph_key.stream_idx:
                raise RuntimeError(
                    "metadata batch was staged for PDMux lane "
                    f"{state.replay_graph_key.stream_idx}, but replay is on "
                    f"lane {replay_lane}"
                )

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(staged["ready"])
        promote_start = torch.cuda.Event(enable_timing=True)
        promote_end = torch.cuda.Event(enable_timing=True)
        promote_start.record(current_stream)
        self.buffer_registry.copy_from_registry(
            stage["registry"],
            padded_bs=state.bs,
            padded_num_tokens=state.padded_num_tokens,
        )
        if state.staged_ragged_layout is not None:
            if state.ragged_layout_key is None:
                raise RuntimeError("staged ragged layout has no captured layout key")
            self._promote_private_ragged_verify_layout(
                state.staged_ragged_layout, state.ragged_layout_key
            )
        self._restore_replay_dispatch_state(state)

        # Backend-private metadata (FlashInfer plans, page tables, host
        # launch state, TBO child state, etc.) is intentionally NOT prepared
        # on the background metadata stream.  The generic attention-backend
        # contract gives the runner no way to clone or promote that state, so
        # preparing batch N+1 there can overwrite buffers still consumed by
        # batch N's replay.  Rebuild it here, after the private registry and
        # compact layout have been promoted, on the replay stream that will
        # launch the graph.  Registry/layout copies still overlap through the
        # alternating stages; backend-owned state remains correctly ordered.
        if self.tbo_cuda_graph_enabled:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=state.bs,
                num_token_non_padded=state.raw_num_token,
                spec_info=forward_batch.spec_info,
            )
        fb_view = self.buffer_registry.extract_buffer(
            padded_bs=state.bs,
            padded_num_tokens=state.padded_num_tokens,
            forward_batch_template=forward_batch,
        )
        fb_view.forward_mode = self.capture_forward_mode
        fb_view.actual_forward_mode = forward_batch.forward_mode
        fb_view.num_padding = state.bs - state.raw_bs
        fb_view.seq_lens_sum = (
            None
            if forward_batch.seq_lens_sum is None
            else forward_batch.seq_lens_sum
            + (state.bs - state.raw_bs) * self.seq_len_fill_value
        )
        if fb_view.spec_info is None and self.capture_forward_mode == (
            ForwardMode.TARGET_VERIFY
        ):
            fb_view.spec_info = self._idle_replay_verify_spec_info()
        _set_replay_tbo_state(
            fb_view,
            enable_two_batch_overlap=self.enable_two_batch_overlap,
            tbo_cuda_graph_enabled=self.tbo_cuda_graph_enabled,
            padded_bs=state.bs,
        )
        attn_started_ns = time.perf_counter_ns()
        self._replay_attn_backend().init_forward_metadata_out_graph(fb_view)
        self.record_metadata_timing(
            "attention_metadata_cpu_ms",
            (time.perf_counter_ns() - attn_started_ns) / 1e6,
        )
        if self.model_runner.hisparse_coordinator is not None:
            self.model_runner.hisparse_coordinator.num_real_reqs.fill_(state.raw_bs)

        promote_end.record(current_stream)
        stage["consumed"] = promote_end
        stage["dispatch_state"] = None
        self._metadata_gpu_samples.append(
            ("promote_gpu_ms", promote_start, promote_end)
        )
        self._drain_metadata_gpu_timings()

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        *,
        _registry: Optional[CudaGraphBufferRegistry] = None,
        _metadata_stage: Optional[dict] = None,
    ):
        if self.dllm_uses_input_embeds and forward_batch.input_embeds is None:
            raise ValueError(
                "Diffusion graph replay requires prepared input embeddings"
            )
        if _metadata_stage is not None and _registry is None:
            raise ValueError("metadata staging requires a private buffer registry")
        ragged_layout = (
            resolve_ragged_verify_layout(forward_batch)
            if self.ragged_verify_mode
            else None
        )
        is_ragged = ragged_layout is not None

        # Staging owns only its private registry/layout. DeepEP and attention
        # backends retain process/runner-private replay state, so touch them
        # only on the actual replay path (which later promotes this stage).
        if _metadata_stage is None:
            self.deepep_adapter.replay()

        if not forward_batch.needs_forward_metadata_init():
            staged = self._staged_batches.pop(id(forward_batch), None)
            if staged is not None:
                self._promote_staged_batch(staged, forward_batch)
                return
            # Pre-planned (plan-stream load_batch already ran).
            # In speculative decoding, these two fields are still needed.
            graph_size_key = (
                self._ragged_graph_size
                if is_ragged
                else self._capture_graph_size(
                    bs=self.bs, num_tokens=self.bs * self.captured_req_width
                )
            )
            if is_ragged:
                assert self.raw_num_token == ragged_layout.graph_num_tokens, (
                    f"stale ragged raw_num_token {self.raw_num_token} != "
                    f"{ragged_layout.graph_num_tokens}"
                )
                self._stage_ragged_verify_layout(ragged_layout, graph_size_key)
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            # seq_lens must be refreshed for graphs whose captured body READS
            # it (sparse target-verify backends build prefix_lens from
            # forward_batch.seq_lens in-graph). Without this, every replay ran
            # the sparse indexer with the capture-time fill value (1), so
            # top-k selected only KV block 0 regardless of the real sequence
            # -- the root cause of the DSpark graph-only output corruption.
            # Same fill-then-copy contract as eagle_draft_cuda_graph_runner /
            # multi_layer_eagle_draft_extend_cuda_graph_runner (padding slots
            # must hold the fill value, not a previous tick's stale lengths).
            raw_bs_now = forward_batch.seq_lens.shape[0]
            self.buffers.seq_lens.fill_(self.seq_len_fill_value)
            self.buffers.seq_lens[:raw_bs_now].copy_(forward_batch.seq_lens)
            if forward_batch.seq_lens_cpu is not None:
                self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
                self.buffers.seq_lens_cpu[:raw_bs_now].copy_(
                    forward_batch.seq_lens_cpu
                )
            if (
                pp_proxy_tensors is not None
                and self.buffers.pp_proxy_tensors is not None
            ):
                # PP + spec verify: the pre-planned load ran without the proxy
                # (eagle_prepare_for_verify has no access to it), so the
                # graph's proxy input buffers must be refreshed here -- the
                # captured graph reads these rows (mirrors fill_from's
                # side-slot copy).
                for k, v in pp_proxy_tensors.tensors.items():
                    buf = self.buffers.pp_proxy_tensors.get(k)
                    if buf is not None:  # skip markers like __msg_type__
                        buf[: v.shape[0]].copy_(v)
            if (
                not is_ragged
                and (
                    self.dllm_uses_input_embeds
                    or (
                        self.model_runner.spec_algorithm.is_dflash_family()
                        and self.model_runner.is_draft_worker
                    )
                )
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            variant_label = self._resolve_lora_variant(forward_batch)
            attention_variant = self._resolve_attention_variant(forward_batch)
            stream_idx = get_current_stream_idx() if self.enable_pdmux else None
            self._replay_graph_key = self._make_graph_key(
                graph_size_key,
                stream_idx,
                variant_label,
                attention_variant,
                sum_bucket=self._replay_dp_verify_sum_bucket,
            )
            return

        buffers = self.buffers
        staged_ragged_layout = None
        staged_ragged_layout_key = None
        self._validate_capture_hidden_mode(forward_batch)

        raw_bs = forward_batch.batch_size
        # A full load resolves this batch's key from scratch.  In particular,
        # do not let a previous static or compact SUM replay leak its
        # sum_bucket into an alternate non-SUM key.
        self._replay_dp_verify_sum_bucket = None

        if is_ragged:
            raw_num_token = ragged_layout.graph_num_tokens
            graph_size_key = self._ragged_graph_num_tokens(raw_num_token)
            assert graph_size_key == ragged_layout.graph_num_tokens, (
                f"ragged verify tier mismatch: runner tier {graph_size_key} != "
                f"layout graph_num_tokens {ragged_layout.graph_num_tokens}"
            )
            bs = self._ragged_capture_slots(graph_size_key)
            assert bs >= raw_bs, (
                f"ragged capture slots {bs} (tier {graph_size_key}) < raw_bs "
                f"{raw_bs}; the planner must reject this batch before replay"
            )
            padded_num_tokens = graph_size_key
            if self.dspark_dp_verify_sum_capture_enabled:
                global_graph_num_tokens = getattr(
                    forward_batch.spec_info, "dp_graph_num_tokens", None
                )
                assert (
                    global_graph_num_tokens == forward_batch.global_num_tokens_cpu
                ), (
                    "compact verify graph-token vector diverged between spec "
                    f"metadata {global_graph_num_tokens} and ForwardBatch DP "
                    f"counts {forward_batch.global_num_tokens_cpu}"
                )
                assert self._all_compact_dp_ranks_fit_graph_slots(
                    global_num_reqs=forward_batch.original_global_num_tokens_cpu,
                    global_graph_num_tokens=global_graph_num_tokens,
                ), (
                    "can_run_graph admitted a compact SUM vector whose request "
                    "counts do not fit the captured slots: requests="
                    f"{forward_batch.original_global_num_tokens_cpu}, graph_tokens="
                    f"{global_graph_num_tokens}"
                )
                buckets = resolve_ragged_dp_verify_sum_graph_buckets(
                    global_graph_num_tokens=global_graph_num_tokens,
                    dp_rank=get_parallel().attn_dp_rank,
                    local_graph_num_tokens=graph_size_key,
                    capture_num_tokens=self.capture_num_tokens,
                    sum_buckets_by_local=self.dp_verify_sum_buckets,
                    max_num_tokens=self.capture_num_tokens[-1],
                )
                assert buckets is not None, (
                    "can_run_graph admitted compact global graph-token buckets="
                    f"{global_graph_num_tokens} with local layout "
                    f"bucket={graph_size_key}, but load_batch's identical "
                    "lookup missed"
                )
                resolved_local_bucket, self._replay_dp_verify_sum_bucket = buckets
                assert resolved_local_bucket == graph_size_key
            if _metadata_stage is None:
                self._stage_ragged_verify_layout(ragged_layout, graph_size_key)
            else:
                (
                    staged_ragged_layout,
                    staged_ragged_layout_key,
                ) = self._stage_private_ragged_verify_layout(
                    _metadata_stage, ragged_layout, graph_size_key
                )
        else:
            raw_num_token = raw_bs * self.captured_req_width
            if self.dspark_dp_verify_sum_capture_enabled:
                # TODO.md J14c Milestone 1: this rank's own local bucket (not
                # the cross-rank max) plus the covering sum_bucket for the real
                # total. can_run_graph admitted this batch through the very
                # same helper, so a miss here means the two calls disagreed.
                buckets = resolve_dp_verify_sum_graph_buckets(
                    global_num_reqs=forward_batch.original_global_num_tokens_cpu,
                    dp_rank=get_parallel().attn_dp_rank,
                    num_tokens_per_req=self.captured_req_width,
                    capture_bs=self.capture_bs,
                    sum_buckets_by_local=self.dp_verify_sum_buckets,
                    max_bs=self.max_bs,
                )
                assert buckets is not None, (
                    "can_run_graph admitted global_num_reqs="
                    f"{forward_batch.original_global_num_tokens_cpu} but "
                    "load_batch's identical lookup missed"
                )
                bs, self._replay_dp_verify_sum_bucket = buckets
            elif self.require_mlp_tp_gather:
                max_batch_size = self._max_dp_batch_size(forward_batch)
                bs = self._pad_to_bucket(max_batch_size, self.capture_bs)
            else:
                bs = self._pad_to_bucket(raw_bs, self.capture_bs)
            padded_num_tokens = bs * self.captured_req_width
            graph_size_key = self._capture_graph_size(
                bs=bs, num_tokens=padded_num_tokens
            )

        if self.dspark_dp_verify_sum_capture_enabled:
            # The registry's graph-live global-count slot copies the real
            # per-rank distribution only for SUM_LEN.  Decode graph replay
            # skips ForwardBatch.prepare_mlp_sync_batch, so set the mode here
            # for both the established static path and compact token buckets.
            forward_batch.dp_padding_mode = DpPaddingMode.SUM_LEN

        registry = _registry or self.buffer_registry
        fill_started_ns = time.perf_counter_ns()
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=bs,
            raw_num_tokens=raw_num_token,
            padded_num_tokens=padded_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        self.record_metadata_timing(
            "registry_fill_cpu_ms", (time.perf_counter_ns() - fill_started_ns) / 1e6
        )

        if (
            not is_ragged
            and (
                self.dllm_uses_input_embeds
                or (
                    self.model_runner.spec_algorithm.is_dflash_family()
                    and self.model_runner.is_draft_worker
                )
            )
            and forward_batch.input_embeds is not None
        ):
            buffers.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
        # Padded tokens aren't read, so skip zeroing. Ragged input_ids arrive
        # from the planner already padded to the tier, invalid slots zeroed.
        if self.tbo_cuda_graph_enabled and _metadata_stage is None:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if (
            not is_ragged
            and forward_batch.forward_mode.is_idle()
            and forward_batch.spec_info is not None
        ):
            forward_batch.spec_info.custom_mask = buffers.custom_mask

        attn_backend = (
            self._replay_attn_backend() if _metadata_stage is None else None
        )
        if _registry is None:
            fb_view = build_replay_fb_view(
                forward_batch=forward_batch,
                buffers=buffers,
                bs=bs,
                raw_bs=raw_bs,
                num_tokens=padded_num_tokens,
                global_num_tokens_cpu=self._global_num_tokens_for_graph(padded_num_tokens),
                seq_len_fill_value=self.seq_len_fill_value,
                capture_forward_mode=self.capture_forward_mode,
                is_encoder_decoder=self.is_encoder_decoder,
            )
        else:
            fb_view = registry.extract_buffer(
                padded_bs=bs,
                padded_num_tokens=padded_num_tokens,
                forward_batch_template=forward_batch,
            )
            fb_view.forward_mode = self.capture_forward_mode
            fb_view.actual_forward_mode = forward_batch.forward_mode
            fb_view.num_padding = bs - raw_bs
            fb_view.seq_lens_sum = (
                None
                if forward_batch.seq_lens_sum is None
                else forward_batch.seq_lens_sum
                + (bs - raw_bs) * self.seq_len_fill_value
            )
        if fb_view.spec_info is None and self.capture_forward_mode == (
            ForwardMode.TARGET_VERIFY
        ):
            fb_view.spec_info = self._idle_replay_verify_spec_info()
        _set_replay_tbo_state(
            fb_view,
            enable_two_batch_overlap=self.enable_two_batch_overlap,
            tbo_cuda_graph_enabled=self.tbo_cuda_graph_enabled,
            padded_bs=bs,
        )
        if _metadata_stage is None:
            if (
                self.model_runner.lora_manager is not None
                and self.model_runner.lora_manager.enable_dp_attention
            ):
                self.model_runner.lora_manager.prepare_lora_batch(
                    cast(ForwardBatch, fb_view)
                )
            # Glue-graph fast path: pointer-stable prep (static buffers + pool
            # tensors only) is captured per key; guards keep every python-visible
            # branch inside the backends constant for that key.
            attn_started_ns = time.perf_counter_ns()
            if (
                self._metadata_glue is not None
                and not self._metadata_glue.disabled
                and raw_bs == bs
                and not self.enable_two_batch_overlap
                and not self.enable_pdmux
                and self.model_runner.lora_manager is None
            ):
                # actual_forward_mode belongs in the key even though the captured
                # graph always targets capture_forward_mode: DSV4's replay prep
                # substitutes seq_lens / seq_lens_cpu / seq_lens_sum /
                # req_pool_indices / out_cache_loc when the runtime mode is IDLE,
                # so IDLE and active DECODE are different python branches and must
                # not share a captured graph.
                self._metadata_glue.run(
                    attn_backend,
                    fb_view,
                    (
                        bs,
                        str(self.capture_forward_mode),
                        str(fb_view.actual_forward_mode),
                    ),
                )
            else:
                attn_backend.init_forward_metadata_out_graph(fb_view)
            self.record_metadata_timing(
                "attention_metadata_cpu_ms",
                (time.perf_counter_ns() - attn_started_ns) / 1e6,
            )
        _maybe_dump_replay_view(
            forward_batch=forward_batch,
            fb_view=fb_view,
            model_runner=self.model_runner,
            raw_bs=raw_bs,
            padded_bs=bs,
            raw_num_token=raw_num_token,
            padded_num_tokens=padded_num_tokens,
        )
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        if is_ragged:
            self._ragged_graph_size = graph_size_key

        if (
            _metadata_stage is None
            and self.model_runner.hisparse_coordinator is not None
        ):
            self.model_runner.hisparse_coordinator.num_real_reqs.fill_(raw_bs)

        variant_label = self._resolve_lora_variant(forward_batch)
        attention_variant = self._resolve_attention_variant(forward_batch)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        self._replay_graph_key = self._make_graph_key(
            graph_size_key,
            stream_idx,
            variant_label,
            attention_variant,
            sum_bucket=self._replay_dp_verify_sum_bucket,
        )
        return self._snapshot_replay_dispatch_state(
            padded_num_tokens=padded_num_tokens,
            ragged_layout_key=staged_ragged_layout_key,
            staged_ragged_layout=staged_ragged_layout,
        )

    def stage_batch(self, forward_batch: ForwardBatch) -> None:
        """Prepare one DSPARK verify batch in an alternating private registry."""
        if not self._dspark_metadata_double_buffer:
            self.load_batch(forward_batch)
            return
        stage = self._metadata_stages[self._metadata_stage_cursor]
        self._metadata_stage_cursor ^= 1
        if stage["dispatch_state"] is not None:
            raise RuntimeError(
                "metadata stage was reused before its prior batch was promoted"
            )
        producer_stream = torch.cuda.current_stream(self.device)
        producer_ready = torch.cuda.Event()
        producer_ready.record(producer_stream)
        stage["producer_ready"] = producer_ready
        with torch.cuda.stream(self._metadata_stream):
            if stage["consumed"] is not None:
                self._metadata_stream.wait_event(stage["consumed"])
            # ForwardBatch tensor sources (compact input_ids, positions,
            # out_cache_loc, and ragged layout tensors) were produced on the
            # caller's current stream. Establish ownership before this stream
            # reads any of them into its private stage.
            self._metadata_stream.wait_event(producer_ready)
            gpu_start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            gpu_start.record(self._metadata_stream)
            started_ns = time.perf_counter_ns()
            dispatch_state = self.load_batch(
                forward_batch,
                _registry=stage["registry"],
                _metadata_stage=stage,
            )
            if not isinstance(dispatch_state, _ReplayDispatchState):
                raise RuntimeError(
                    "metadata staging did not produce replay dispatch state"
                )
            stage["dispatch_state"] = dispatch_state
            self.record_metadata_timing(
                "load_batch_cpu_ms", (time.perf_counter_ns() - started_ns) / 1e6
            )
            ready.record(self._metadata_stream)
        self._metadata_gpu_samples.append(
            ("staging_gpu_ms", gpu_start, ready)
        )
        self._staged_batches[id(forward_batch)] = {
            "stage": stage,
            "ready": ready,
        }
        self._drain_metadata_gpu_timings()

    def discard_staged_batch(self, forward_batch: ForwardBatch) -> None:
        """Release metadata staged for a batch that later falls back eager.

        Graph eligibility is checked once during speculative pre-planning and
        again immediately before execution.  If the second check declines,
        the private stage must become reusable even though it was never
        promoted into the graph-owned registry.  Preserve the ready event as
        the stage's reuse barrier so its queued copies finish before a later
        batch overwrites the same private buffers.
        """
        staged = self._staged_batches.pop(id(forward_batch), None)
        if staged is None:
            return
        stage = staged["stage"]
        stage["consumed"] = staged["ready"]
        stage["dispatch_state"] = None

    def record_metadata_timing(self, name: str, value_ms: float) -> None:
        if self._metadata_timing_interval <= 0:
            return
        self._metadata_timing_sum[name] += value_ms
        self._metadata_timing_max[name] = max(
            self._metadata_timing_max[name], value_ms
        )

    def _drain_metadata_gpu_timings(self) -> None:
        while self._metadata_gpu_samples:
            name, start, end = self._metadata_gpu_samples[0]
            if not end.query():
                break
            self._metadata_gpu_samples.popleft()
            self.record_metadata_timing(name, start.elapsed_time(end))

    def finish_metadata_timing_sample(self) -> None:
        interval = self._metadata_timing_interval
        if interval <= 0:
            return
        self._metadata_timing_count += 1
        self._drain_metadata_gpu_timings()
        if self._metadata_timing_count < interval:
            return
        means = {
            name: total / interval
            for name, total in self._metadata_timing_sum.items()
        }
        logger.info(
            "[DSPARK metadata timing] count=%d mean_ms=%s max_ms=%s pending_gpu=%d",
            interval,
            means,
            dict(self._metadata_timing_max),
            len(self._metadata_gpu_samples),
        )
        self._metadata_timing_count = 0
        self._metadata_timing_sum.clear()
        self._metadata_timing_max.clear()

    def _ragged_graph_num_tokens(self, total_verify_tokens: int) -> int:
        from sglang.srt.speculative.ragged_verify import round_up_grid

        return round_up_grid(total_verify_tokens, self.capture_num_tokens)

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        timer_ctx = device_timer_ctx(
            self.model_runner.device_timer, forward_batch.forward_mode.name.lower()
        )
        shared_read_ends = self._resolve_shared_read_ends(
            self._replay_attn_backend(), forward_batch.forward_mode
        )
        # Skip the read-done publish while a CC conditional decode WHILE body is
        # capturing this trace: Event.record()/wait_event() are illegal node
        # types inside a captured conditional body, and dflash_cc_conditional_
        # decode.run_ccd_chunk already discards whatever this would have set
        # (last_shared_read_runner.shared_read_done_event = None) once capture
        # completes, since an event recorded against this chunk's private,
        # non-persistent stream would otherwise be a dangling reference on a
        # later tick anyway. UNKNOWN short-circuits the PRE/IN/POST_REPLAY
        # dispatch below (records nothing, scheduler keeps the coarse fence).
        if is_ccd_capturing():
            shared_read_ends = SharedReadEnds.UNKNOWN
        with timer_ctx, self.backend.replay_session():
            self.load_batch(forward_batch, pp_proxy_tensors)
            if envs.SGLANG_LOG_DECODE_GRAPH_KEY.get():
                logger.info(
                    "Decode graph replay: worker=%s key_size=%s (%s) mode=%s raw_bs=%d%s",
                    "draft" if self.model_runner.is_draft_worker else "target",
                    self._replay_graph_key.size,
                    "num_tokens" if self.ragged_verify_mode else "bs",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    (
                        f" slots={self._ragged_capture_slots(self._replay_graph_key.size)}"
                        if self.ragged_verify_mode
                        else ""
                    ),
                )
            if shared_read_ends is SharedReadEnds.PRE_REPLAY:
                self._publish_read_done(in_graph=False)

            output = self.backend.replay(self._replay_graph_key, forward_batch)

            if shared_read_ends is SharedReadEnds.IN_REPLAY:
                self._publish_read_done(in_graph=True)

            if shared_read_ends is SharedReadEnds.POST_REPLAY:
                self._publish_read_done(in_graph=False)

            output = self._process_output_after_replay(output, forward_batch)

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[: self.raw_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[: self.raw_num_token]
                    if output.next_token_logits is not None
                    else None
                )

            # Preserve extension fields produced by the eager output processor.
            return dataclasses.replace(
                output,
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            # Slice in token rows, not request rows: under speculative verify
            # each request carries captured_req_width tokens (identical for
            # plain decode, where captured_req_width == 1).
            return PPProxyTensors(
                {
                    k: v[: self.bs * self.captured_req_width]
                    for k, v in output.tensors.items()
                }
            )

    def _process_output_after_replay(self, output, forward_batch: ForwardBatch):
        """Optional eager tail executed after graph replay and inside its timer."""
        return output

    def _idle_replay_verify_spec_info(self):
        """Verify input for a rank replaying a TARGET_VERIFY graph with no
        local requests, cached because its only field is config-static.

        Under DP attention the ranks step together, so when peers verify, a
        rank holding zero requests still has to run the collective. Its
        ScheduleBatch comes from prepare_for_idle (schedule_batch.py), which
        never sets spec_info, and the replay view reports the *captured* mode
        (TARGET_VERIFY) rather than IDLE -- so the attention backend asks for a
        verify width the batch cannot supply and a sparse backend
        raises ("TARGET_VERIFY forward without a spec_info.draft_token_num",
        TODO.md J14d). The width was never request-derived though: capture used
        self.captured_req_width (get_spec_info's dflash branch), a server-arg
        constant, so the idle replay can state the same thing.

        Only reachable where the alternative is that RuntimeError, so this
        cannot change the behavior of a configuration that works today.
        ragged_verify_layout stays None: a ragged batch takes the
        _can_run_ragged_verify_graph path and never lands here.
        """
        if self._idle_verify_spec_info is None:
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            from sglang.srt.speculative.dflash_utils import (
                resolve_dflash_verify_mask_policy,
            )

            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            self._idle_verify_spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.captured_req_width,
                custom_mask=(
                    self.buffers.custom_mask if build_custom_mask else None
                ),
                capture_hidden_mode=CaptureHiddenMode.FULL,
                ragged_verify_layout=None,
            )
        return self._idle_verify_spec_info

    def get_spec_info(
        self, num_tokens: int, stream_idx: Optional[int] = None
    ):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.model_runner.spec_algorithm.is_standalone()
                    else CaptureHiddenMode.FULL
                )
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=get_spec().speculative_eagle_topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=capture_mode,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
                # MTP models (e.g. deepseek_nextn) read spec_info.hidden_states
                spec_info.hidden_states = torch.zeros(
                    (num_tokens, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                    device=self.model_runner.device,
                )
        elif self.model_runner.spec_algorithm.is_dflash_family():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            from sglang.srt.speculative.dflash_utils import (
                resolve_dflash_verify_mask_policy,
            )

            # Avoid enabling custom-mask modes during graph capture for backends that
            # can express DFLASH verify via their built-in causal path.
            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.captured_req_width,
                custom_mask=(
                    None
                    if (self.model_runner.is_draft_worker or not build_custom_mask)
                    else self.buffers.custom_mask
                ),
                capture_hidden_mode=(
                    CaptureHiddenMode.NULL
                    if self.model_runner.is_draft_worker
                    else CaptureHiddenMode.FULL
                ),
                ragged_verify_layout=self._capture_ragged_verify_layout(
                    num_tokens, stream_idx=stream_idx
                ),
            )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                custom_mask=self.buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                draft_token_num=self.captured_req_width,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info
