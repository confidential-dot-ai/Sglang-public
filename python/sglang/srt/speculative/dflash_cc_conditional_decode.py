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
"""CC conditional decode wiring for DFlash: runs `chunk_cap` decode/verify
ticks entirely on-GPU per host round-trip, via cc_conditional_decode.py's
CUDA-graph WHILE-loop primitive, instead of one host round-trip per tick.

Strictly opt-in (SGLANG_CC_CONDITIONAL_DECODE); every non-CCD code path is
byte-for-byte unaffected. See tdx-checkpoints/059-063 for the full design
history and validated risk resolutions this module builds on:
  - 059: nested/child graph capture mechanism (add_child_graph_node).
  - 060: DFlash-aware finish/predicate kernel (variable accept length).
  - 061/062: why forward_batch_generation's accept/commit block can be
    reused almost as-is (FullCudaGraphBackend.replay() is the one
    injection point, everything else is either already capture-safe or a
    documented no-op for this benchmark's config), and why prepare_for_decode/
    load_batch/recapture_if_needed must run as a real, eager "priming" tick
    before the chunk capture, not inside it.
  - 063: DFlashDraftInputV2.prepare_for_decode(num_blocks=...) for
    chunk-sized KV reservation.
  - This file's own memory-pool-sharing spike (ccd_phase3_pool_spike.py):
    draft + verify graphs sharing FullCudaGraphBackend's global pool,
    embedded as child nodes inside a WHILE loop capturing under its own,
    separate pool -- validated correct, including in-capture allocation and
    repeated chunk relaunches.

NOT YET DONE (tracked in the plan, not implemented here): the CCD
eligibility gate that decides whether a given running batch may use this
path at all (all-greedy sampling only, no compact draft cache, no mamba
verify-commit, no logprobs, grammar only via the device-FSM backend,
draft+verify graphs already captured
for this exact bs) and the scheduler call site that invokes it once per
chunk instead of once per tick (Phase 5). This module provides the
mechanism; wiring it into the live decode loop is separate, deliberately
gated work.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.model_executor.cc_conditional_decode import (
    CCConditionalDecode,
    CondGraph,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    ccd_capturing_context,
)
from sglang.srt.oracle_decode.forced_token_pool import stash_ccd_forced_lookup
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

# Master opt-in for the whole CC conditional decode path. Same flag name
# dflash_worker_v2.py / full_cuda_graph_backend.py already read (each reads
# it independently, matching the SGLANG_CC_INGRAPH_GREEDY_SAMPLE precedent
# this design follows) -- default OFF, every module's behavior is then
# byte-for-byte unchanged. The scheduler call site (checkpoint 067) is the
# only place that needs to check this directly; the other two modules gate
# their own internal behavior on it already.
SGLANG_CC_CONDITIONAL_DECODE = get_bool_env_var("SGLANG_CC_CONDITIONAL_DECODE")

# How many captured iterations per chunk (on top of the one real priming
# tick -- checkpoint 062/064). Larger = fewer host round-trips per token but
# more KV pre-reservation (chunk_cap * block_size slots, checkpoint 063) and
# a bigger blast radius if a single chunk's captured graph turns out to be
# wrong. Start small; this is exactly the kind of knob Phase 6/7's
# correctness + real-engine validation should tune, not something to guess
# a large value for up front.
SGLANG_CC_CONDITIONAL_DECODE_CHUNK_CAP = int(
    os.environ.get("SGLANG_CC_CONDITIONAL_DECODE_CHUNK_CAP", "4")
)

# Minimum output tokens a request must already have generated before CCD
# may run on a batch it's part of (task #149). Measured live: at 64 output
# tokens CCD's per-token TPOT is 22.33ms vs plain decode's 18.56ms (CCD
# LOSES -- its per-chunk fixed overhead isn't amortized yet); at 512 tokens
# it's 10.87ms vs 17.34ms (CCD wins 1.6x); the crossover sits somewhere in
# that unmeasured 64-512 gap. 256 is the midpoint of that gap, not a
# pinned-down crossover point -- tune with a real 128/256-token sweep before
# trusting this default in production. All-or-nothing per-batch, matching
# every other gate in check_ccd_eligible: one request in the running batch
# below the threshold falls the WHOLE batch back to normal decode for that
# tick, same as any other ineligibility reason. This means a continuously-
# batched server with a steady trickle of freshly-admitted short requests
# may rarely satisfy this gate -- CCD then only kicks in once every request
# currently in the batch has individually crossed the threshold.
SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN = int(
    os.environ.get("SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN", "256")
)


def check_ccd_eligible(batch: "ScheduleBatch", worker: "DFlashWorkerV2") -> Optional[str]:
    """Strict eligibility gate. Returns None if `batch` may use CC
    conditional decode, otherwise a human-readable reason it can't --
    callers must fall back to the normal per-tick eager path unchanged,
    never guess or partially apply CCD.
    """
    # CCD only ever captures forward_batch_generation's decode/verify branch
    # (checkpoints 061-064). This must match forward_batch_generation's own
    # routing condition exactly (dflash_worker_v2.py: `if
    # batch.forward_mode.is_extend() or batch.is_extend_in_batch:` selects
    # its PREFILL branch instead) -- found the hard way (checkpoint 068):
    # the first real execution of this code crashed inside the prefill
    # branch's `torch.tensor(batch.extend_lens, ..., device=device)`, a
    # host-list-to-GPU materialization CUDA graph capture forbids. This
    # gate never checked forward_mode at all before that crash.
    if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        return "CCD only applies to decode/verify ticks, not prefill/extend"
    sampling_info = batch.sampling_info
    if sampling_info is None or not sampling_info.is_all_greedy:
        return "CCD requires all-greedy sampling (non-greedy verify path is not captured)"
    if getattr(batch, "return_logprob", False):
        return "CCD does not support return_logprob"
    if getattr(batch, "has_grammar", False):
        # Grammar-constrained decoding is CCD-compatible ONLY when every
        # grammar in the batch is a DeviceFSMGrammar (grammar_backend=
        # 'device_fsm'): its mask application and state advancement are
        # fixed-shape tensor ops on persistent device buffers that DFlash's
        # verify step runs in-graph (dflash_worker_v2's device_fsm_rows
        # blocks), so a captured chunk replays them correctly. Every other
        # backend (xgrammar/outlines/llguidance) advances a host-resident,
        # per-request CPU matcher between ticks and rebuilds its bitmask on
        # the host every step -- exactly the host round-trip / per-request
        # Python branching CCD's captured region forbids; without this check
        # CCD would silently emit unconstrained tokens for those requests.
        # (In practice validate_dflash_request already keeps host-backend
        # grammar requests out of DFlash entirely; this stays as CCD's own
        # last line of defense, matching the gate's strictness elsewhere.)
        from sglang.srt.constrained.device_token_fsm import DeviceFSMGrammar

        for req in batch.reqs:
            grammar = getattr(req, "grammar", None)
            if grammar is not None and not isinstance(grammar, DeviceFSMGrammar):
                return (
                    "CCD supports grammar-constrained decoding only via "
                    "grammar_backend='device_fsm' (device-resident token FSM); "
                    f"got {type(grammar).__name__}"
                )
    if worker.use_compact_draft_cache and (
        worker.draft_window_size is None
        or worker.draft_window_size > SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN
    ):
        # Compact draft cache is CCD-compatible ONLY when draft_window_size
        # can never exceed the min-output-len floor every request in this
        # batch has already cleared -- dflash_worker_v2.py's compact branch
        # clamps every request's draft-visible length to draft_window_size
        # (DFlashWorkerV2._compute_compact_draft_seq_lens), so once a
        # request has generated >= draft_window_size tokens (guaranteed by
        # the check below, given this condition), that clamp is always
        # saturated for the entire lifetime of a captured chunk. That
        # invariant is what makes the compact branch's fixed-shape padded
        # gather/assign (_gather_req_to_token_segments_padded,
        # assign_req_to_token_pool_padded_func -- see tdx-checkpoints/120
        # for why the earlier ragged, boolean-mask version crashed under
        # capture) correctly sized: max_len is a static bound derived from
        # draft_window_size, and every request's real per-request length
        # must never exceed it. A misconfiguration where draft_window_size
        # could exceed the min-output-len floor would violate that, so
        # this checks it explicitly rather than assuming it silently.
        return (
            "CCD requires draft_window_size <= "
            "SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN under compact draft "
            f"cache (draft_window_size={worker.draft_window_size}, "
            f"min_output_len={SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN})"
        )
    if worker.use_compact_draft_cache and batch.seq_lens_cpu is None:
        # The compact branch derives its EXACT host-side seq_lens_cpu from
        # batch.seq_lens_cpu (see dflash_worker_v2.py: flashinfer's
        # fast_prefill_plan consumes seq_lens_cpu as exact geometry, so a
        # static upper bound there corrupts the KV layout -> IMA). Without a
        # host mirror the only exact source is a D2H sync, which the capture
        # region forbids -- so degrade to eager decode instead.
        return "CCD with compact draft cache requires batch.seq_lens_cpu"
    if worker._need_mamba_verify_commit:
        return "CCD does not support mamba verify-commit models"
    if worker.ccd_lm_head_requires_collective():
        # TP>1 with per-rank added (extended) vocab: the vocab-parallel
        # LM-head sample/score paths would fall through to their NCCL
        # all-gather fallback INSIDE the captured WHILE body -- collectives
        # record event nodes that are illegal in a conditional body
        # (cudaErrorInvalidValue at graph instantiation, checkpoint 122's
        # crash class). Gate here so the config degrades to eager decode
        # instead of crashing capture.
        return (
            "CCD requires a TP-replicable target LM head (added/extended "
            "vocab under TP>1 needs an in-body NCCL all-gather)"
        )
    # Worker-capability hook: a dflash-family worker other than
    # DFlashWorkerV2 (today: DSparkWorkerV2, which the scheduler's
    # is_dflash_family() gate routes here too) may have static-config
    # reasons of its own that its forward_batch_generation cannot run
    # inside the captured WHILE body (host-scheduled ragged verify,
    # active observability sinks, eager draft sampling...). Optional via
    # getattr so DFlashWorkerV2 -- which encodes its constraints in the
    # dedicated checks above -- is byte-for-byte unaffected.
    worker_capture_reason_fn = getattr(worker, "ccd_capture_incompatible_reason", None)
    if worker_capture_reason_fn is not None:
        worker_capture_reason = worker_capture_reason_fn()
        if worker_capture_reason is not None:
            return worker_capture_reason
    if batch.spec_info is None or getattr(batch.spec_info, "nxt_kv_lens_cpu", None) is None:
        return "CCD requires an already-prepared DFlashDraftInputV2 spec_info"
    # process_batch_result_decode dereferences logits_output for these three
    # per-request flags; the synthetic per-iteration results this module
    # builds carry logits_output=None (verified safe against every other
    # caller in that pipeline -- _normalize_decode_outputs/_resolve_spec_v2_tokens
    # only touch next_token_ids/accept_lens, _maybe_collect_customized_info
    # explicitly None-checks). These three must be excluded batch-wide.
    for req in batch.reqs:
        if req.return_hidden_states or req.return_sampling_mask:
            return "CCD does not support return_hidden_states/return_sampling_mask"
        if len(req.output_ids) < SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN:
            return (
                f"CCD requires every request in the batch to have generated "
                f">= {SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN} output tokens "
                f"already (req has {len(req.output_ids)})"
            )
    return None


def partition_ccd_eligible(
    batch: "ScheduleBatch", worker: "DFlashWorkerV2"
) -> Tuple[Optional[str], List[int]]:
    """Per-request partition of CCD eligibility, for scheduler.py's
    split-batch path (run CCD on whichever subset of `batch.reqs`
    individually qualifies, eager on the rest, same tick) -- an important
    case check_ccd_eligible's whole-batch gate handles poorly: a batch with
    a continuous mix of long-running "veteran" requests (which do qualify)
    and freshly-admitted "newcomer" ones (which never do, since every
    request starts at 0 output tokens) NEVER passes check_ccd_eligible as a
    whole, even though most of its requests individually would. Under
    sustained multi-turn traffic with constant request churn, that mix is
    the *steady state*, not an edge case -- check_ccd_eligible alone means
    CCD effectively never fires.

    Returns `(batch_level_reason, eligible_indices)`.

    `batch_level_reason` is set (and `eligible_indices` always empty) only
    for conditions that can't vary per-request -- forward_mode is one
    property of the whole tick, worker.use_compact_draft_cache/
    ._need_mamba_verify_commit are static model config, spec_info readiness
    is a batch-structural precondition. None of these can be satisfied by
    some requests and not others, so a failure here means no request in
    `batch` may use CCD this tick, full stop.

    Otherwise `eligible_indices` lists the positions in `batch.reqs` that
    individually satisfy every remaining, genuinely per-request condition
    (greedy sampling, no logprob/hidden-states/sampling-mask, grammar
    backend, minimum already-generated output length) -- deliberately the
    same conditions check_ccd_eligible checks batch-wide, just evaluated
    row by row instead of requiring unanimous agreement. Positions not
    listed must run the normal eager per-tick path this same call --
    reqs = [batch.reqs[i] for i in eligible_indices] and its complement are
    what scheduler.py's split path filters into two sub-batches via
    ScheduleBatch.copy()/.filter_batch().
    """
    if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        return "CCD only applies to decode/verify ticks, not prefill/extend", []
    if worker.use_compact_draft_cache and (
        worker.draft_window_size is None
        or worker.draft_window_size > SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN
    ):
        # See check_ccd_eligible's matching guard for the full rationale:
        # compact draft cache is CCD-safe only once draft_window_size can
        # never exceed the min-output-len floor, since that's what bounds
        # the compact branch's fixed-shape padded gather/assign (see
        # tdx-checkpoints/120 for the ragged version's live capture crash
        # and the padded-kernel fix that replaced it).
        return (
            "CCD requires draft_window_size <= "
            "SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN under compact draft "
            f"cache (draft_window_size={worker.draft_window_size}, "
            f"min_output_len={SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN})"
        ), []
    if worker.use_compact_draft_cache and batch.seq_lens_cpu is None:
        # Matches check_ccd_eligible's gate (batch-level, so no request can
        # qualify). Without it here, run_ccd_chunk's own re-check would
        # RAISE on the eligible sub-batch instead of falling back.
        return "CCD with compact draft cache requires batch.seq_lens_cpu", []
    if worker._need_mamba_verify_commit:
        return "CCD does not support mamba verify-commit models", []
    if worker.ccd_lm_head_requires_collective():
        # Matches check_ccd_eligible's gate (see its comment): worker-level
        # static config, so no request can qualify -- and without this,
        # run_ccd_chunk's own re-check would RAISE on the eligible
        # sub-batch instead of falling back.
        return (
            "CCD requires a TP-replicable target LM head (added/extended "
            "vocab under TP>1 needs an in-body NCCL all-gather)"
        ), []
    # Matches check_ccd_eligible's worker-capability hook (see its comment):
    # worker-level static config, so no request can qualify -- and without
    # this, run_ccd_chunk's own re-check would RAISE on the eligible
    # sub-batch instead of falling back.
    worker_capture_reason_fn = getattr(worker, "ccd_capture_incompatible_reason", None)
    if worker_capture_reason_fn is not None:
        worker_capture_reason = worker_capture_reason_fn()
        if worker_capture_reason is not None:
            return worker_capture_reason, []
    if batch.spec_info is None or getattr(batch.spec_info, "nxt_kv_lens_cpu", None) is None:
        return "CCD requires an already-prepared DFlashDraftInputV2 spec_info", []

    from sglang.srt.constrained.device_token_fsm import DeviceFSMGrammar

    eligible = []
    for i, req in enumerate(batch.reqs):
        if req.sampling_params.top_k > 1:
            continue  # non-greedy: verify path not captured
        if req.return_logprob:
            continue
        if req.return_hidden_states or req.return_sampling_mask:
            continue  # synthetic per-tick results carry logits_output=None
        grammar = getattr(req, "grammar", None)
        if grammar is not None and not isinstance(grammar, DeviceFSMGrammar):
            continue  # host-resident grammar backends need a real per-tick round trip
        if len(req.output_ids) < SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN:
            continue  # too new; not yet worth a captured chunk's own priming cost
        eligible.append(i)
    return None, eligible


def try_split_ccd_batch(
    batch: "ScheduleBatch", worker: "DFlashWorkerV2"
) -> Optional[Tuple["ScheduleBatch", "ScheduleBatch", List[int], List[int]]]:
    """Split `batch` into (eligible_batch, ineligible_batch,
    eligible_indices, ineligible_indices) for scheduler.py's split-tick
    path, or None if the split shouldn't happen this tick (caller falls
    back to the existing, unchanged check_ccd_eligible-gated whole-batch
    path). The two index lists give each sub-batch row's position in the
    ORIGINAL batch -- merge_ccd_split_results uses them to scatter both
    halves' results back into original row order, so the caller's batch
    object never needs reordering (any in-place reorder would be undone by
    run_batch's _forward_isolation restore anyway, leaving the result and
    the batch permanently misaligned -- found by review, not live).

    Returns None (no split) when:
      - partition_ccd_eligible's batch-level gate fails (no request may use
        CCD this tick regardless).
      - eligible_indices is empty or covers the whole batch -- both cases
        are already handled correctly by check_ccd_eligible's existing
        whole-batch logic (empty -> plain eager; whole -> plain CCD), so
        there is nothing for a split to gain.
      - any of a short list of narrow, deliberate V1 scope guards below
        trip. Each guard sidesteps a real merge-correctness question this
        function does not attempt to solve yet (see comments); tripping one
        falls back to today's proven whole-batch behavior, never guesses.

    Callers MUST give the two returned sub-batches independent objects
    before doing anything else with them -- ScheduleBatch.copy() is
    unsuitable here (it's an intentionally partial, output-side-only
    snapshot: "Only contain fields that will be used by process_batch_result",
    missing seq_lens/sampling_info/etc, so it cannot back a real forward).
    This function instead uses copy.copy() (shallow top-level clone) plus an
    explicit independent copy.copy() of `spec_info`/`sampling_info` for each
    side -- ScheduleBatch.filter_batch()'s own field mutations are all
    rebindings (`self.x = self.x[keep_indices]`, which creates a new tensor
    via advanced indexing, never an in-place write), so a shallow object
    copy is sufficient to stop the two sub-batches from aliasing each
    other's nested state; without it, filtering the second sub-batch would
    silently clobber the first (both `.copy.copy(batch)`s otherwise share
    the exact same `spec_info`/`sampling_info` Python objects, and
    `filter_batch` mutates whatever `self.spec_info`/`self.sampling_info`
    point to).
    """
    reason, eligible_indices = partition_ccd_eligible(batch, worker)
    n = len(batch.reqs)
    if os.environ.get("SGLANG_CCD_SPLIT_DEBUG") == "1":
        import sys as _sys_splitdbg

        _sys_splitdbg.stderr.write(
            f"CCD_PARTITION_DEBUG: n={n} eligible={len(eligible_indices)} "
            f"reason={reason!r}\n"
        )
        _sys_splitdbg.stderr.flush()
    if reason is not None or len(eligible_indices) == 0 or len(eligible_indices) == n:
        return None

    eligible_set = set(eligible_indices)
    ineligible_indices = [i for i in range(n) if i not in eligible_set]

    # V1 scope guards -- each sidesteps a real question this function does
    # not attempt to answer yet; fall back rather than guess.
    # By the time run_batch calls this, batch.sampling_info is a
    # copy_for_forward() copy: penalizer_orchestrator is ALWAYS None there
    # (deliberately dropped, with any active penalties pre-accumulated --
    # found live across the split path's first two real executions; the
    # unit-test mocks originally supplied an orchestrator object, a
    # mock-fidelity gap). The pre-accumulation lands in TWO buffers, and
    # either alone signals active penalties: acc_scaling_penalties
    # (multiplicative -- repetition_penalty is the only one) and
    # acc_additive_penalties (frequency/presence/min_new_tokens -- built
    # unconditionally whenever the orchestrator is_required, so a batch
    # with only additive penalties has acc_scaling_penalties=None).
    # SamplingBatchInfo.filter_batch row-filters NEITHER buffer, so
    # splitting would leave both sub-batches holding the full-width tensor
    # and misalign (or shape-crash) penalty rows. Bail on either.
    if getattr(batch.sampling_info, "acc_scaling_penalties", None) is not None:
        return None
    if getattr(batch.sampling_info, "acc_additive_penalties", None) is not None:
        return None
    _penalizer = getattr(batch.sampling_info, "penalizer_orchestrator", None)
    if _penalizer is not None and _penalizer.is_required:
        # SamplingBatchInfo.filter_batch calls
        # penalizer_orchestrator.filter(...); this module hasn't audited
        # whether that mutates shared state in a way that's safe to run
        # independently on two sibling sub-batches derived from the same
        # orchestrator (unlike the plain-tensor fields, which are all
        # rebind-not-mutate and therefore fine to copy.copy() around).
        # This project's oracle/agentic-trace workloads run greedy/
        # forced-token decoding with no frequency/presence penalties
        # configured, so this guard costs nothing in practice today.
        return None
    if batch.model_config.is_encoder_decoder:
        return None
    if batch.mamba_track_indices is not None:
        # Mamba extra-buffer tracking is per-request state this function's
        # split/merge does not thread through; not exercised by this
        # project's models (DFlash + a plain transformer target), so
        # falling back here rather than guessing is free in practice.
        return None
    if getattr(batch, "dp_cooperation_info", None) is not None:
        return None
    if getattr(batch, "hisparse_coordinator", None) is not None:
        return None
    for i in ineligible_indices:
        req = batch.reqs[i]
        if req.return_logprob or req.return_hidden_states or req.return_sampling_mask:
            # Both sub-batches' results get merged with logits_output=None
            # (matching build_synthetic_result's own established
            # convention) -- safe only because, same as check_ccd_eligible,
            # every request that would dereference logits_output is kept
            # out. partition_ccd_eligible already guarantees this for the
            # eligible side; this guard extends the same guarantee to the
            # ineligible side, which partition_ccd_eligible does not itself
            # constrain (a request lands there for many reasons, e.g. being
            # a brand-new admit, not only for wanting logprobs).
            return None

    if os.environ.get("SGLANG_CCD_SPLIT_DEBUG") == "1":
        import sys as _sys_splitdbg2

        _sys_splitdbg2.stderr.write("CCD_PARTITION_DEBUG: guards passed, splitting\n")
        _sys_splitdbg2.stderr.flush()

    eligible_batch = copy.copy(batch)
    eligible_batch.spec_info = copy.copy(batch.spec_info)
    eligible_batch.sampling_info = copy.copy(batch.sampling_info)
    eligible_batch.filter_batch(keep_indices=eligible_indices)
    # SamplingBatchInfo.filter_batch never recomputes is_all_greedy (set
    # once at build). partition_ccd_eligible admits only greedy rows into
    # eligible_indices, so the filtered sub-batch IS all-greedy by
    # construction even when the original mixed batch wasn't -- without
    # this, run_ccd_chunk's check_ccd_eligible re-check would raise on the
    # stale False for exactly the mixed batches this split exists to serve.
    eligible_batch.sampling_info.is_all_greedy = True

    ineligible_batch = copy.copy(batch)
    ineligible_batch.spec_info = copy.copy(batch.spec_info)
    ineligible_batch.sampling_info = copy.copy(batch.sampling_info)
    ineligible_batch.filter_batch(keep_indices=ineligible_indices)

    return eligible_batch, ineligible_batch, eligible_indices, ineligible_indices


def merge_ccd_split_results(
    eligible_result,
    ineligible_result,
    eligible_indices: List[int],
    ineligible_indices: List[int],
    *,
    device,
):
    """Combine the eligible sub-batch's CCD-chunk terminal result with the
    ineligible sub-batch's normal eager result into one GenerationBatchResult
    in the ORIGINAL batch's row order, by scattering each half's rows back to
    the positions try_split_ccd_batch partitioned them from.

    Original order (not eligible-then-ineligible concatenation) is the load-
    bearing property: the caller's ScheduleBatch keeps its original row order
    (run_batch's _forward_isolation restores every field after the forward,
    so no in-forward reorder could survive to process_batch_result anyway),
    and process_batch_result maps result rows to batch.reqs positionally.

    Device handling: the eligible terminal result's tensors are CPU
    (build_synthetic_result / read_history's D2H), the eager result's are
    device-resident. Everything is scattered on `device` (eligible rows H2D
    -- small [ne(,block)] int tensors, eager context, legal) so the merged
    result follows the standard "device until copy_to_cpu" convention.

    next_draft_input is rebuilt in original order from both halves' own
    next_draft_input (run_ccd_chunk sets the terminal result's to the
    chunk's live in-place-carried spec_info; the eager path always sets
    one), so the caller's post-isolation `batch.spec_info =
    batch_result.next_draft_input` and future-map relay both see rows
    aligned with the original batch.

    logits_output is always None on the merged result, same convention as
    build_synthetic_result: safe here specifically because
    try_split_ccd_batch's own guards keep every return_logprob/
    return_hidden_states/return_sampling_mask request out of both halves
    (eligible via partition_ccd_eligible, ineligible via try_split_ccd_batch's
    own explicit guard) -- process_batch_result_decode never dereferences it
    for any request in either half.
    """
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

    ne = len(eligible_indices)
    ni = len(ineligible_indices)
    n = ne + ni
    block = int(eligible_result.speculative_num_draft_tokens)

    elig_idx = torch.tensor(eligible_indices, dtype=torch.long, device=device)
    inel_idx = torch.tensor(ineligible_indices, dtype=torch.long, device=device)

    def scatter_rows(elig_t: torch.Tensor, inel_t: torch.Tensor) -> torch.Tensor:
        # Output dtype follows the eager half: that's what every downstream
        # consumer already receives on a non-split tick.
        inel_t = inel_t.to(device)
        elig_t = elig_t.to(device=device, dtype=inel_t.dtype)
        out = torch.empty(
            (n, *inel_t.shape[1:]), dtype=inel_t.dtype, device=device
        )
        out[elig_idx] = elig_t
        out[inel_idx] = inel_t
        return out

    next_token_ids = scatter_rows(
        eligible_result.next_token_ids.view(ne, block),
        ineligible_result.next_token_ids.view(ni, block),
    ).reshape(-1)
    accept_lens = scatter_rows(
        eligible_result.accept_lens.view(ne),
        ineligible_result.accept_lens.view(ni),
    )

    elig_spec = eligible_result.next_draft_input
    inel_spec = ineligible_result.next_draft_input
    bonus_tokens = scatter_rows(
        elig_spec.bonus_tokens.view(ne, *elig_spec.bonus_tokens.shape[1:]),
        inel_spec.bonus_tokens.view(ni, *inel_spec.bonus_tokens.shape[1:]),
    )
    new_seq_lens = scatter_rows(
        elig_spec.new_seq_lens.view(ne),
        inel_spec.new_seq_lens.view(ni),
    )
    next_draft_input = make_draft_input_v2(
        bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens
    )

    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        can_run_cuda_graph=(
            eligible_result.can_run_cuda_graph and ineligible_result.can_run_cuda_graph
        ),
        speculative_num_draft_tokens=block,
        accept_lens=accept_lens,
        next_draft_input=next_draft_input,
        new_seq_lens=new_seq_lens,
    )


def build_synthetic_result(
    *,
    next_token_ids: torch.Tensor,
    accept_lens: torch.Tensor,
    speculative_num_draft_tokens: int,
):
    """One CC conditional decode captured iteration's output, packaged as a
    GenerationBatchResult -- exactly the shape process_batch_result_decode /
    _resolve_spec_v2_tokens already expect from a real tick, so neither
    needs to change (checkpoints 065/066). `next_token_ids`/`accept_lens`
    must already be CPU tensors (DflashCCDChunk.read_history's one combined
    D2H copy provides this). `logits_output=None` is safe ONLY because
    check_ccd_eligible already excludes every per-request flag that would
    dereference it (return_logprob/return_hidden_states/return_sampling_mask)
    -- audited against process_batch_result_decode's full body, not assumed.
    """
    from sglang.srt.managers.scheduler import GenerationBatchResult

    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        can_run_cuda_graph=True,
        copy_done=None,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        accept_lens=accept_lens,
    )
    return None


class DflashCCDChunk:
    """One capturable WHILE-loop chunk for a fixed-bs, fixed-request-set
    DFlash decode batch. Construct once per chunk (bs and requests must not
    change mid-chunk -- that's what makes it capturable), `prime_and_capture`
    once, then `launch` repeatedly for as many chunks as this exact batch
    stays eligible and unchanged.
    """

    def __init__(
        self,
        worker: "DFlashWorkerV2",
        *,
        bs: int,
        chunk_cap: int,
        eos_ids: torch.Tensor,
        max_new: torch.Tensor,
        device: torch.device,
    ):
        self.worker = worker
        self.bs = bs
        self.chunk_cap = chunk_cap
        self.block_size = int(worker.block_size)
        self.device = device

        self.ccd = CCConditionalDecode()

        # WHILE-loop control state.
        self.n_run = torch.ones(1, dtype=torch.int32, device=device)
        self.d_step = torch.zeros(1, dtype=torch.int32, device=device)
        self.d_cap = torch.full((1,), chunk_cap, dtype=torch.int32, device=device)

        # Per-request continuation state (checkpoint 060's finish kernel).
        # A fresh chunk always starts un-finished; the caller re-derives the
        # authoritative finished/out_len state from the real Req objects
        # between chunks (see module docstring: this kernel is a
        # best-effort continuation predicate, not the source of truth).
        self.out_len = torch.zeros(bs, dtype=torch.int32, device=device)
        self.finished = torch.zeros(bs, dtype=torch.uint8, device=device)
        self.eos_ids = eos_ids
        self.max_new = max_new

        # Written by body_fn each captured iteration; read by the finish kernel.
        # Scratch for the CURRENT iteration only -- the finish kernel's
        # contract (checkpoint 060) is unchanged, it never sees the history.
        self.commit_lens_buf = torch.zeros(bs, dtype=torch.int32, device=device)
        self.out_tokens_buf = torch.zeros(
            (bs, self.block_size), dtype=torch.int64, device=device
        )

        # Per-iteration history, one row per captured tick, so every token
        # generated during the chunk reaches the client stream -- not just
        # the last iteration's (checkpoint 065's accumulation gap). Written
        # via a dynamic-index (`d_step`-valued) index_copy_ each iteration;
        # validated standalone in ccd_phase5_accum_spike.py before landing
        # here. Read back host-side once, after the whole chunk completes
        # (see read_history), never inside the capture.
        self.history_commit_lens = torch.zeros(
            (chunk_cap, bs), dtype=torch.int32, device=device
        )
        self.history_out_tokens = torch.zeros(
            (chunk_cap, bs, self.block_size), dtype=torch.int64, device=device
        )

        # TEMPORARY debug instrumentation (tdx-checkpoints/071/074/075): captures
        # the target model's own per-position argmax (target_predict) for every
        # replayed iteration, same in-graph accumulation pattern as history_*
        # above, so it can be diffed against the eager baseline's target_predict
        # to tell apart "the graph computed the wrong logits" from "commit_lens
        # was miscomputed from correct logits." Remove once root-caused.
        self.history_target_predict = torch.zeros(
            (chunk_cap, bs, self.block_size), dtype=torch.int64, device=device
        )

        self._finish_ka = None
        self._cond_graph = None
        self.chunk_batch: Optional["ScheduleBatch"] = None

    def prime_and_capture(self, chunk_batch: "ScheduleBatch"):
        """Run one real, eager (uncaptured) decode/verify tick against
        `chunk_batch` -- this is the "priming" step (checkpoint 062):
        forces recapture_if_needed/forward-metadata planning to settle and
        the draft/verify CUDA graphs for this exact bs to already exist, so
        the WHILE-loop capture that follows only ever exercises the
        lightweight, already-planned fast paths. Also reserves KV slots for
        the whole chunk up front (checkpoint 063:
        prepare_for_decode(num_blocks=chunk_cap), covering the priming tick
        plus every captured iteration).

        Consumes real tokens for this batch (it's a genuine decode step,
        not a no-op) -- callers must process its GenerationBatchResult the
        same way they would any other tick's result, not discard it.
        """
        reason = check_ccd_eligible(chunk_batch, self.worker)
        if reason is not None:
            raise RuntimeError(f"DflashCCDChunk: batch is not CCD-eligible: {reason}")

        self.chunk_batch = chunk_batch
        draft_input = chunk_batch.spec_info

        # Oracle mode: build the forced-token lookup exactly once, here,
        # before either the priming tick or the capture that follows ever
        # runs -- not inside body_fn, and not left to forward_batch_
        # generation's own fallback (which would silently rebuild on every
        # real host call, but only ONCE, at trace time, if that call were
        # itself inside the captured region -- see dflash_worker_v2.py's
        # oracle block for why that's wrong). Requests and their
        # forced_token_ids are fixed for a chunk's whole lifetime (same
        # invariant that makes the chunk capturable at all), so building
        # once here and stashing it on chunk_batch is both sufficient and
        # correct: every captured tick's gather_oracle_window/
        # compute_oracle_commit_via_draft_logprob call already indexes by
        # `prefix_lens` (== chunk_batch.seq_lens, updated in place every
        # tick below and in body_fn), not by anything rebuilt per-tick.
        # stash_ccd_forced_lookup also fingerprints the request list (rids):
        # this dynamic attribute survives filter_batch/merge_batch/copy.copy
        # on the live batch object untouched, so consumers
        # (resolve_ccd_forced_lookup) must be able to detect that the
        # batch's composition changed since the stash and rebuild instead of
        # silently replaying one request's forced rows against another.
        stash_ccd_forced_lookup(chunk_batch, self.device)

        # Reserve the whole chunk's KV slots in this one host-side call
        # (checkpoint 063). +1 block already baked into prepare_for_decode's
        # own safety margin on top of num_blocks.
        draft_input.prepare_for_decode(chunk_batch, num_blocks=self.chunk_cap)

        # The real, eager priming tick. Not captured; its output is the
        # first tick's real result, returned to the caller like any other.
        priming_result = self.worker.forward_batch_generation(chunk_batch)

        # forward_batch_generation's own decode/verify branch sets
        # chunk_batch.forward_mode = ForwardMode.TARGET_VERIFY as an internal
        # side effect (DFlashVerifyInput.prepare_for_verify) -- normally
        # invisible because _forward_isolation's per-tick snapshot/restore
        # resets it before the next real tick. This module calls
        # forward_batch_generation twice (priming tick, then again inside
        # body_fn) within ONE outer isolation scope, so that automatic reset
        # never happens between them. Without this explicit reset, body_fn's
        # call sees forward_mode=TARGET_VERIFY, and since
        # ForwardMode.is_extend() is True for TARGET_VERIFY too, it
        # misroutes into forward_batch_generation's PREFILL branch instead
        # -- found by running this for the first time (checkpoint 068): it
        # crashed on `torch.tensor(batch.extend_lens, ..., device=device)`,
        # a host-list-to-GPU materialization CUDA graph capture forbids.
        #
        # Same mechanism, different field: DFlashVerifyInput.prepare_for_verify
        # also does `batch.spec_info = self` (the DFlashVerifyInput instance,
        # not the DFlashDraftInputV2 `draft_input` this whole class is built
        # around) as its own transient side effect, also normally invisible
        # because of the same per-tick isolation reset. Found immediately
        # after fixing forward_mode above: body_fn's call then failed
        # forward_batch_generation's own `isinstance(draft_input,
        # DFlashDraftInputV2)` check with a RuntimeError, for the identical
        # reason. Restore both.
        chunk_batch.forward_mode = ForwardMode.DECODE
        chunk_batch.spec_info = draft_input

        # From here on, chunk_batch.seq_lens / chunk_batch.spec_info.bonus_tokens
        # must be updated IN PLACE for the rest of this chunk's lifetime
        # (capture traces this code path once; every subsequent replay reads
        # whatever these same tensor objects hold at replay time -- see
        # checkpoint 062/063 and this module's docstring).
        chunk_batch.seq_lens.copy_(priming_result.new_seq_lens)
        draft_input.bonus_tokens.copy_(priming_result.next_draft_input.bonus_tokens)

        self._finish_ka = self.ccd.make_finish_args(
            out_tokens=self.out_tokens_buf.data_ptr(),
            commit_lens=self.commit_lens_buf.data_ptr(),
            block_size=self.block_size,
            eos_ids=self.eos_ids.data_ptr(),
            n_eos=int(self.eos_ids.shape[0]),
            out_len=self.out_len.data_ptr(),
            max_new=self.max_new.data_ptr(),
            finished=self.finished.data_ptr(),
            d_step=self.d_step.data_ptr(),
            n_run=self.n_run.data_ptr(),
            bs=self.bs,
        )
        finish_args_ptr = self._finish_ka.build()

        def body_fn(stream_ptr):
            with ccd_capturing_context():
                result = self.worker.forward_batch_generation(chunk_batch)
            # In-place carry -- see the note above prime_and_capture's
            # priming-tick copies; same requirement, every captured iteration.
            chunk_batch.seq_lens.copy_(result.new_seq_lens)
            draft_input.bonus_tokens.copy_(result.next_draft_input.bonus_tokens)
            self.commit_lens_buf.copy_(result.accept_lens)
            self.out_tokens_buf.copy_(
                result.next_token_ids.view(self.bs, self.block_size)
            )
            # Accumulate into this iteration's history slot BEFORE the finish
            # kernel advances d_step -- read d_step's pre-increment value.
            step_idx = self.d_step.to(torch.int64)
            self.history_commit_lens.index_copy_(
                0, step_idx, self.commit_lens_buf.unsqueeze(0)
            )
            self.history_out_tokens.index_copy_(
                0, step_idx, self.out_tokens_buf.unsqueeze(0)
            )
            # TEMPORARY debug instrumentation -- see history_target_predict's
            # own comment above. result.logits_output is the raw (uncleared)
            # verify output at this point -- forward_batch_generation only
            # nulls out hidden_states, next_token_logits is still populated.
            if result.logits_output is not None:
                target_predict = torch.argmax(
                    result.logits_output.next_token_logits, dim=-1
                ).view(self.bs, self.block_size)
                self.history_target_predict.index_copy_(
                    0, step_idx, target_predict.unsqueeze(0)
                )
            self.ccd.launch_finish(finish_args_ptr, self.bs, stream_ptr)

        def warmup_fn(stream_ptr):
            # The CCD-replicated LM head weight (checkpoint 109) is built
            # eagerly during THIS priming tick's draft greedy-sample call,
            # on whatever stream was ambient at that point -- not s_body,
            # which doesn't exist yet. body_fn's first read of it (inside
            # the draft greedy sampler's CCD fast path) would otherwise be
            # the first cross-stream touch, captured as an illegal
            # WaitEvent node. Touch it once, eagerly, here instead.
            full_weight = getattr(
                self.worker, "_ccd_replicated_lm_head_weight", None
            )
            import os as _os_wdbg
            if _os_wdbg.environ.get("SGLANG_CCD_LMHEAD_DEBUG") == "1":
                import sys as _sys_wdbg
                _sys_wdbg.stderr.write(
                    f"CCD_WARMUP_DEBUG: warmup_fn called, full_weight is None={full_weight is None}"
                    + (f" shape={tuple(full_weight.shape)}" if full_weight is not None else "")
                    + "\n"
                )
                _sys_wdbg.stderr.flush()
            if full_weight is not None:
                full_weight[:1].contiguous()

        self._cond_graph = self.ccd.capture(
            body_fn=body_fn,
            finish_ka=self._finish_ka,
            n_run=self.n_run,
            d_step=self.d_step,
            d_cap=self.d_cap,
            torch_mod=torch,
            warmup_fn=warmup_fn,
        )

        # body_fn's own single capture-time trace of forward_batch_generation
        # hits the exact same DFlashVerifyInput.prepare_for_verify side effect
        # the priming-tick reset above exists for -- capture() traces body_fn
        # exactly once (as real, eager Python code) to record the graph, and
        # that one trace's call to forward_batch_generation sets
        # chunk_batch.forward_mode/spec_info to the transient verify state,
        # with nothing inside body_fn to reset it back afterward. Harmless
        # for replay itself (the GPU op sequence was already fixed at
        # capture time, replaying doesn't re-run this Python branch logic),
        # but run_ccd_chunk reads batch.spec_info in Python *after*
        # launch()+read_history() to build the chunk's terminal
        # next_draft_input -- without this second reset, that read sees the
        # stale DFlashVerifyInput left over from capture, not a
        # DFlashDraftInputV2 (found live: AttributeError, 'DFlashVerifyInput'
        # object has no attribute 'bonus_tokens', in _relay_forward_payload).
        chunk_batch.forward_mode = ForwardMode.DECODE
        chunk_batch.spec_info = draft_input

        return priming_result

    def launch(self, stream: torch.cuda.Stream):
        """Replay the captured chunk (chunk_cap further ticks) on `stream`.
        Caller resets n_run/d_step/finished/out_len before each launch
        unless intentionally continuing a partially-finished chunk."""
        if self._cond_graph is None:
            raise RuntimeError("DflashCCDChunk: capture() has not been called yet")
        self._cond_graph.launch(stream.cuda_stream)

    def read_history(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """One-time D2H readback of every captured iteration's output.

        Call after launch() and after the caller has ordered its current
        stream behind the launch stream (e.g.
        `torch.cuda.current_stream().wait_stream(launch_stream)`) -- `.cpu()`
        only synchronizes against the stream it's issued on, not against an
        unrelated stream the graph replayed on.

        Returns `chunk_cap` (next_token_ids, accept_lens) CPU-tensor pairs,
        one per captured iteration, in order. Feed each one through the
        existing per-tick result-processing pipeline
        (`batch_result_processor.py`'s `process_batch_result_decode` /
        `_resolve_spec_v2_tokens`) exactly as if it were a separate real
        tick's `GenerationBatchResult.next_token_ids`/`.accept_lens`
        (checkpoint 065) -- that pipeline already handles variable per-request
        accept length, it does not need to change. The priming tick's result
        (returned by `prime_and_capture`) is a real `GenerationBatchResult`
        already and should be processed the normal way, not through this method.
        """
        commit_lens_cpu = self.history_commit_lens.cpu()
        out_tokens_cpu = self.history_out_tokens.cpu()
        return [
            (out_tokens_cpu[t].reshape(-1), commit_lens_cpu[t])
            for t in range(self.chunk_cap)
        ]

    def read_target_predict_history(self) -> List[torch.Tensor]:
        """TEMPORARY debug instrumentation -- see history_target_predict's
        comment. Same D2H-readback contract as read_history(). Remove once
        the correctness divergence (checkpoint 071/074/075) is root-caused."""
        target_predict_cpu = self.history_target_predict.cpu()
        return [target_predict_cpu[t].reshape(-1) for t in range(self.chunk_cap)]

    def destroy(self) -> None:
        """Release the captured graph's native CUDA resources (task #136).
        Call exactly once, after `launch()` + every readback (`read_history()`
        / `read_target_predict_history()`) has completed -- see
        `CondGraph.destroy()`. A no-op if `prime_and_capture` was never
        called (nothing was captured)."""
        if self._cond_graph is not None:
            self._cond_graph.destroy()


def run_ccd_chunk(
    worker: "DFlashWorkerV2",
    batch: "ScheduleBatch",
    *,
    chunk_cap: int,
    eos_ids: torch.Tensor,
    max_new: torch.Tensor,
    stream: torch.cuda.Stream,
    on_publish: Optional[Callable[[torch.Tensor], None]] = None,
) -> List:
    """Run one CC conditional decode chunk against `batch` -- the real, live
    running batch, mutated in place (matches forward_batch_generation's own
    contract; caller re-derives what should carry to the next real tick from
    the return value, per _forward_isolation's existing pattern -- checkpoint
    065). Caller must have already confirmed
    `check_ccd_eligible(batch, worker) is None`.

    Returns `chunk_cap + 1` GenerationBatchResult objects in order: the real
    priming tick's result first, then `chunk_cap` synthetic results unpacked
    from the captured chunk's accumulated history
    (build_synthetic_result/read_history). Every one of them should be
    processed through the existing per-tick result pipeline exactly like a
    normal tick's result -- feed them through in order, do not skip or batch
    them together.

    `stream` should be the caller's current forward stream (the same stream
    `batch`'s real per-tick forward calls already run on) -- launching the
    chunk there means `read_history()`'s `.cpu()` calls are correctly
    ordered after the launch with no extra cross-stream wait needed.

    `on_publish`, if given, is called once with the chunk's terminal
    `batch.seq_lens` after the chunk completes -- normally
    `partial(future_map.publish, future_indices)` (checkpoint: task #133).
    None of this function's `chunk_cap + 1` internal ticks (the priming tick
    or any captured iteration) go through the normal per-tick `on_publish`
    plumbing `forward_batch_generation` otherwise takes, so without this,
    `FutureMap.new_seq_lens_buf` never advances for a CCD-driven request.
    The *next* scheduler tick's `resolve_seq_lens_cpu` (called unconditionally
    at the top of every `run_batch`, gated only on `spec_info.future_indices
    is not None`) then rewinds `batch.seq_lens` back to whatever stale value
    was last published -- silently discarding this whole chunk's in-place
    advance and feeding the model a truncated attention window on the next
    chunk. Confirmed live: with this unpublished, CCD output degrades into
    repetition/wrong facts a few chunks in; a live A/B against CCD-off
    output on the same prompts was clean. Publishing once with the terminal
    value is sufficient -- `FutureMap.publish` only ever cares about the
    latest value at each index, same as N per-tick publishes would leave it.

    KNOWN LIMITATION (deliberate, not an oversight): this always does a
    fresh `prime_and_capture` -- it does not cache/reuse a captured chunk
    across separate `run_ccd_chunk` calls. A captured chunk becomes stale
    the instant `batch.reqs` is rebound by `filter_batch`/`merge_batch`
    (admission/eviction), and safely detecting that isn't implemented yet.
    Capture overhead is paid every call, amortized only within that one
    chunk's `chunk_cap` iterations, not across separate chunks.

    Cross-chunk capture reuse now EXISTS for the plain-decode analog
    (model_executor/ccspec_decode.py, SGLANG_CCSPEC_CAPTURE_REUSE) and was
    DELIBERATELY NOT extended to CCD in that pass, for auditability
    reasons, not feasibility ones. CCSpec's captured body reads a small,
    chunk-OWNED carry set (cur_input_ids / slot_table / seq_lens_buf /
    positions / out_loc / forced lookup / eos), each re-armable by an
    in-place D2D rewrite of a buffer the chunk itself allocated. CCD's
    captured body is a full `worker.forward_batch_generation` trace: its
    per-chunk-varying device inputs are created INSIDE
    `DFlashDraftInputV2.prepare_for_decode(num_blocks=chunk_cap)` (the
    chunk-wide KV reservation and its req_to_token writes) and inside the
    draft/verify prep paths (verify input construction, compact-cache
    padded gathers, device-FSM state, the draft model's own metadata) --
    none of them are chunk-owned pre-allocated buffers, so there is no
    pointer-stable rewrite target; a reuse re-arm would first have to
    enumerate every tensor that one-time trace closed over. Additionally,
    keeping a DflashCCDChunk alive across scheduler ticks changes the
    lifetime analysis this function already had to patch once (the
    shared_read_done_event dangling-stream fix below assumes the
    chunk's private streams die with this call). Until that full audit is
    done, shipping CCD reuse would violate the "never launch a graph whose
    validity wasn't proven" rule; see the plan / checkpoint 067.
    """
    bs = batch.batch_size()
    chunk = DflashCCDChunk(
        worker,
        bs=bs,
        chunk_cap=chunk_cap,
        eos_ids=eos_ids,
        max_new=max_new,
        device=batch.device,
    )
    priming_result = chunk.prime_and_capture(batch)

    # body_fn's captured trace of forward_batch_generation runs the target
    # runner's normal decode_cuda_graph_runner.execute() code, including its
    # own shared_read_done_event bookkeeping (an event recorded on
    # WHATEVER stream is current at that point in the trace -- here, `chunk`'s
    # own private body stream, not model_runner's regular forward stream).
    # That event object gets published onto the long-lived
    # last_shared_read_runner (checkpoint: scheduler.py's _apply_war_barrier
    # reads it on a LATER, unrelated tick), but `chunk` itself -- and its
    # private streams/mempool -- has no persistent owner (see this
    # function's own KNOWN LIMITATION note) and becomes eligible for
    # garbage collection once this function returns. A later wait_event()
    # on an event recorded against a since-destroyed stream is a dangling
    # reference, not merely stale data -- found live: `event.wait(stream)`
    # -> CUDA error: invalid argument, in a scheduler tick well after this
    # chunk had already finished. Clear it so _apply_war_barrier takes its
    # existing, already-correct fallback (wait_stream on the real forward
    # stream) instead of trusting a reference this module can't guarantee
    # stays valid.
    war_runner = worker.last_shared_read_runner
    if war_runner is not None:
        war_runner.shared_read_done_event = None

    chunk.launch(stream)
    history = chunk.read_history()
    target_predict_history = chunk.read_target_predict_history()

    if os.environ.get("SGLANG_CCD_SPLIT_DEBUG") == "1":
        # Livelock forensics (checkpoint 123): one line per chunk with the
        # carry state -- if seq_lens/out_len stop advancing across chunks
        # while chunks keep running, this shows exactly which quantity is
        # frozen and what each captured tick claimed to commit.
        import sys as _sys_chunkdbg

        _hist_commits = [
            accept_lens.tolist() for _tok, accept_lens in history
        ]
        _sys_chunkdbg.stderr.write(
            "CCD_CHUNK_DEBUG: "
            f"seq_lens={batch.seq_lens.tolist()} "
            f"out_len={chunk.out_len.tolist()} "
            f"d_step={int(chunk.d_step.item())} "
            f"finished={chunk.finished.tolist()} "
            f"max_new={chunk.max_new.tolist()} "
            f"prim_new_seq_lens={priming_result.new_seq_lens.tolist() if priming_result.new_seq_lens is not None else None} "
            f"hist_commit_lens={_hist_commits} "
            f"output_ids_len={[len(r.output_ids) for r in batch.reqs]}\n"
        )
        _sys_chunkdbg.stderr.flush()

    # Release the captured graph's native CUDA resources now -- every
    # readback above is a synchronous .cpu() copy, so the launch is fully
    # complete and nothing will touch the graph exec again (task #136: this
    # was never destroyed, so every chunk permanently leaked its private
    # MemPool -- see CondGraph.destroy()'s own comment).
    chunk.destroy()

    # TEMPORARY debug instrumentation for the CCD correctness-divergence
    # investigation (tdx-checkpoints/071/074/075) -- see dflash_worker_v2.py's
    # _ccd_debug_dump_tick for the eager-path counterpart. Host-side only,
    # safe (chunk.launch()/read_history() have already fully returned by
    # this point, no capture in progress). Remove once root-caused.
    from sglang.srt.speculative.dflash_worker_v2 import _ccd_debug_dump_tick

    for (next_token_ids, accept_lens), target_predict in zip(
        history, target_predict_history
    ):
        _ccd_debug_dump_tick(
            "ccd_replay", next_token_ids, accept_lens, target_predict
        )

    results = [priming_result]
    for next_token_ids, accept_lens in history:
        results.append(
            build_synthetic_result(
                next_token_ids=next_token_ids,
                accept_lens=accept_lens,
                speculative_num_draft_tokens=chunk.block_size,
            )
        )
    # batch.seq_lens / batch.spec_info already hold the chunk's terminal
    # state in place (checkpoint 062/064's carry design) -- surface it on
    # the last result so the caller's normal post-forward bookkeeping
    # (batch.spec_info = result.next_draft_input, etc.) sees it, matching
    # what a real final tick's GenerationBatchResult would carry.
    #
    # nxt_kv_lens_cpu/_sum are the one exception (task #141): they're
    # this chunk's own num_blocks=chunk_cap reservation from
    # prepare_for_decode() at chunk start (prime_and_capture, above), not
    # part of what a real final tick would carry -- every other caller
    # (dflash_disaggregation.py's docstring, DFlashDraftInputV2.merge_batch)
    # depends on this field staying transient, single-tick state that's
    # None between real scheduler ticks and only recomputed fresh by the
    # next tick's own prepare_for_decode() call. Left set here, it survives
    # into the next normal-path merge_batch() (new requests admitted from
    # prefill merging into this running batch) as a stale chunk-cap-sized
    # reservation the incoming batch's freshly-built spec_info (always None
    # per that invariant) can't match -- confirmed live: `assert
    # spec_info.nxt_kv_lens_cpu is not None` in merge_batch crashes
    # both engines the first time a chunk boundary lands concurrently with
    # new requests merging in (e.g. any real concurrent-load benchmark).
    # Reset here so the handed-back object matches what a real final tick's
    # next_draft_input actually looks like; the next CCD chunk's own
    # prepare_for_decode(num_blocks=chunk_cap) recomputes this from scratch
    # regardless, so nothing downstream reads the pre-reset value.
    batch.spec_info.nxt_kv_lens_cpu = None
    batch.spec_info.nxt_kv_lens_sum = None
    results[-1].next_draft_input = batch.spec_info
    results[-1].new_seq_lens = batch.seq_lens
    if on_publish is not None:
        on_publish(batch.seq_lens)

    # TEMPORARY debug instrumentation (task #136, CCD memory-growth
    # investigation). Remove once root-caused.
    if os.environ.get("SGLANG_CCD_MEM_DEBUG") == "1":
        import gc
        import sys as _sys_memdbg

        gc.collect()
        live_chunks = sum(1 for o in gc.get_objects() if type(o) is DflashCCDChunk)
        live_ccd = sum(1 for o in gc.get_objects() if type(o) is CCConditionalDecode)
        live_condgraph = sum(1 for o in gc.get_objects() if type(o) is CondGraph)
        _sys_memdbg.stderr.write(
            f"CCD_MEM_DEBUG: alloc={torch.cuda.memory_allocated() / 1e6:.1f}MB "
            f"reserved={torch.cuda.memory_reserved() / 1e6:.1f}MB "
            f"live_chunks={live_chunks} live_ccd={live_ccd} "
            f"live_condgraph={live_condgraph}\n"
        )
        _sys_memdbg.stderr.flush()

    return results
