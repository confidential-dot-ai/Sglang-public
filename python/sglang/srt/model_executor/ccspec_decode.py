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
"""CCSpec: CCD-style conditional WHILE-loop CUDA-graph decode chunks for
PLAIN (non-speculative) decode -- the host-round-trip amortization of
speculative/dflash_cc_conditional_decode.py (CCD), without any draft model.

One `run_ccspec_chunk` call runs `chunk_cap + 1` plain greedy decode ticks
per host round-trip: one real, eager "priming" tick (the tick the scheduler
already prepared via `prepare_for_decode`), then up to `chunk_cap` further
ticks entirely on-GPU inside one captured WHILE conditional graph. On the
TDX + GPU-CC hosts this project targets, every host<->device crossing pays
the software AES-GCM bounce-buffer tax and pinned memory is silently broken
(`is_pinned()` always False), so nothing inside the captured body may do an
H2D copy, a `.item()`/`.cpu()`/`.tolist()`, a data-dependent-shape
allocation, or an NCCL collective -- the same hard constraints CCD's module
docstring and KNOWN CONSTRAINTS establish, inherited here verbatim.

V1 scope (deliberately narrow, mirroring CCD v1):
  - Whole-batch only; strictly opt-in via SGLANG_CCSPEC_DECODE.
  - Fresh capture per chunk BY DEFAULT: a captured chunk is only valid for
    the exact request set / bs it was captured against. Cross-chunk capture
    reuse now exists behind its own opt-in (SGLANG_CCSPEC_CAPTURE_REUSE,
    default OFF): staleness detection is an explicit validity fingerprint
    (ccspec_capture_fingerprint + CCSpecChunk.reuse_mismatch_reason) plus a
    per-chunk re-arm of every carried device buffer
    (CCSpecChunk.rearm_for_reuse); any mismatch destroys and recaptures.
  - All-or-nothing stop: the first captured tick whose sampled token hits
    EOS for ANY row ends the chunk after that tick is recorded. Every
    emitted tick is therefore valid for every row, so the synthetic
    per-tick results need no per-row accept/trim logic -- the plain-decode
    result processor appends exactly one token per row per result and has
    no accept_lens concept. The EOS-carrying tick itself IS emitted (the
    EOS token must reach the scheduler); ticks after it never run.
  - Greedy argmax only, bypassing the full Sampler: the eligibility gate
    excludes everything the Sampler adds on top of argmax (penalties,
    logit bias, custom processors, grammar masks, logprobs), so
    `argmax(next_token_logits)` + `apply_forced_tokens` reproduces
    `ModelRunner.sample` exactly for eligible batches.
  - Oracle (forced-token) support comes essentially free: the CCD forced
    lookup machinery (`stash_ccd_forced_lookup` / `resolve_ccd_forced_lookup`)
    is already capture-safe and rid-fingerprinted; the body applies it the
    same way `ModelRunner.sample` does.
  - Host-side stop conditions beyond model EOS (per-request stop_token_ids,
    stop strings, ignore_eos interplay) remain host-authoritative exactly
    as in CCD: the device stop is a best-effort continuation predicate; a
    request the host finishes mid-chunk has its later in-chunk tokens
    dropped by the result processor's existing finished-row skip, and its
    over-allocated KV freed by the same release path that already covers
    normal overlap's one-tick over-run (this chunk just widens that window
    to at most chunk_cap ticks).

Scheduler contract (mirrors CCD's whole-batch branch exactly): run_batch's
overlap branch queues `results[:-1]` on `_ccd_pending_extra_results` (with
`_ccd_pending_extra_snapshot = None` -- whole-batch, so the event loop's
`batch.copy()` fallback is exactly right), returns `results[-1]` as the
tick's batch_result, and the existing `_take_ccd_extras`/`_drain_ccd_extras`
machinery and tripwire drain the rest with ZERO changes.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.model_executor.cc_conditional_decode import CCConditionalDecode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.oracle_decode.forced_token_pool import (
    apply_forced_tokens,
    stash_ccd_forced_lookup,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# Master opt-in. Read once at import, same idiom as CCD's
# SGLANG_CC_CONDITIONAL_DECODE -- default OFF, every non-CCSpec code path is
# then byte-for-byte unchanged.
SGLANG_CCSPEC_DECODE = get_bool_env_var("SGLANG_CCSPEC_DECODE")

# Captured iterations per chunk (on top of the one real priming tick). Same
# rationale as CCD's chunk cap: larger = fewer host round-trips per token
# but more KV pre-reservation (bs * chunk_cap slots) and a bigger blast
# radius per captured graph. Start small.
SGLANG_CCSPEC_CHUNK_CAP = int(os.environ.get("SGLANG_CCSPEC_CHUNK_CAP", "4"))

# Minimum output tokens every request must already have generated before
# CCSpec may run on the batch -- same crossover rationale as CCD's floor
# (per-chunk fixed capture overhead isn't amortized on short outputs; see
# the measurement notes on SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN).
SGLANG_CCSPEC_MIN_OUTPUT_LEN = int(
    os.environ.get("SGLANG_CCSPEC_MIN_OUTPUT_LEN", "256")
)

# Cross-chunk capture reuse (strictly opt-in, default OFF -- with it unset,
# every code path below is byte-for-byte the v1 fresh-capture-per-chunk
# behavior). When ON, run_ccspec_chunk keeps the last chunk's captured
# WHILE-loop graph alive and relaunches it for the next chunk IFF an
# explicit validity fingerprint proves the capture is still exact for the
# new chunk (see ccspec_capture_fingerprint + CCSpecChunk.reuse_mismatch_
# reason); on ANY mismatch the cached graph is destroyed and a fresh
# capture runs -- never guess, never partially re-arm. This removes the
# ~10s-of-ms host-side capture serialization from every chunk after the
# first, which on the TDX+GPU-CC hosts is the dominant per-chunk fixed
# cost the MIN_OUTPUT_LEN crossover exists to amortize.
SGLANG_CCSPEC_CAPTURE_REUSE = get_bool_env_var("SGLANG_CCSPEC_CAPTURE_REUSE")


def check_ccspec_eligible(
    batch: "ScheduleBatch",
    model_runner: "ModelRunner",
    *,
    chunk_cap: Optional[int] = None,
) -> Optional[str]:
    """Strict eligibility gate, mirroring check_ccd_eligible's contract:
    returns None if `batch` may run a CCSpec chunk, else a human-readable
    reason. Callers must fall back to the normal per-tick path unchanged --
    never guess or partially apply CCSpec. Never raises; conditions that
    would make the chunk *fail* (allocator exhaustion, req_to_token
    overflow) are returned as reasons, not discovered as exceptions.
    """
    if chunk_cap is None:
        chunk_cap = SGLANG_CCSPEC_CHUNK_CAP

    # -- Tick-shape gates -------------------------------------------------
    # The captured body reproduces exactly one PLAIN decode tick; anything
    # extend/verify/idle-shaped routes elsewhere in run_batch and must
    # never reach the chunk.
    if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        return "CCSpec only applies to plain decode ticks, not prefill/extend"
    if not batch.forward_mode.is_decode():
        return "CCSpec only applies to plain decode ticks (got non-decode mode)"
    if not batch.spec_algorithm.is_none():
        return "CCSpec is the plain-decode path; speculative batches use CCD"
    if getattr(batch, "dllm_config", None) is not None:
        return "CCSpec does not support diffusion-LM decode"
    if batch.batch_size() == 0:
        return "CCSpec requires a non-empty batch"
    if batch.model_config.is_encoder_decoder:
        return "CCSpec does not support encoder-decoder models"
    # DP-attention / MLP-sync batches carry per-rank padding + collective
    # sync obligations the captured body cannot replay.
    if getattr(batch, "global_num_tokens", None) is not None:
        return "CCSpec does not support DP-attention / MLP-sync batches"
    if getattr(batch, "hisparse_coordinator", None) is not None:
        return "CCSpec does not support hisparse-coordinated batches"
    # Mamba extra-buffer tracking recomputes mamba_track_mask (an H2D of
    # `seq_lens_cpu % interval`) in prepare_for_decode EVERY tick; the
    # captured ticks bypass prepare_for_decode entirely, so the mask would
    # be stale for them. The backend gate below already excludes every
    # mamba-capable backend; this is defense in depth.
    if getattr(batch, "mamba_track_indices", None) is not None:
        return "CCSpec does not support mamba extra-buffer tracking"

    # -- Parallelism gates ------------------------------------------------
    # TP>1: the plain path's LogitsProcessor computes vocab-parallel logits
    # then AllGathers -- a collective INSIDE the captured WHILE body, which
    # records event nodes that are illegal in a conditional body
    # (cudaErrorInvalidValue at instantiation; tdx-checkpoints/080/122's
    # crash class). CCD solves this with a pre-replicated LM-head weight
    # built by the DFlash worker; the plain path has no owner for that
    # machinery yet, so V1 simply requires TP=1. PP>1 additionally returns
    # proxy tensors instead of logits on non-last ranks.
    ps = getattr(model_runner, "ps", None)
    if ps is not None and int(getattr(ps, "tp_size", 1)) > 1:
        return (
            "CCSpec v1 requires TP=1 (the plain-path LM-head AllGather is an "
            "in-body collective; no replicated-weight machinery on this path)"
        )
    if ps is not None and int(getattr(ps, "pp_size", 1)) > 1:
        return "CCSpec v1 requires PP=1"

    # -- Sampling gates ---------------------------------------------------
    # These jointly guarantee that greedy argmax over next_token_logits +
    # apply_forced_tokens is EXACTLY what ModelRunner.sample would produce.
    sampling_info = batch.sampling_info
    if sampling_info is None or not sampling_info.is_all_greedy:
        return "CCSpec requires all-greedy sampling"
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and penalizer.is_required:
        return "CCSpec does not support active sampling penalizers"
    if getattr(sampling_info, "acc_scaling_penalties", None) is not None:
        return "CCSpec does not support accumulated scaling penalties"
    if getattr(sampling_info, "acc_additive_penalties", None) is not None:
        return "CCSpec does not support accumulated additive penalties"
    if getattr(sampling_info, "logit_bias", None) is not None:
        return "CCSpec does not support logit_bias (argmax bypasses it)"
    if getattr(sampling_info, "has_custom_logit_processor", False):
        return "CCSpec does not support custom logit processors"

    # -- Per-request gates ------------------------------------------------
    if getattr(batch, "has_grammar", False):
        # V1: reject ALL grammar backends, including DeviceFSM. Unlike
        # DFlash (which advances DeviceFSM state in-graph during verify),
        # the plain path advances grammar host-side per tick in
        # process_batch_result_decode's _accept_grammar_tokens -- there is
        # no in-graph plain-path FSM advance to capture, so a captured
        # chunk would emit unconstrained tokens for grammar rows.
        return "CCSpec does not support grammar-constrained requests (v1)"
    for req in batch.reqs:
        if getattr(req, "grammar", None) is not None:
            return "CCSpec does not support grammar-constrained requests (v1)"
        if req.return_logprob:
            return "CCSpec does not support return_logprob"
        if req.return_hidden_states or req.return_sampling_mask:
            return "CCSpec does not support return_hidden_states/return_sampling_mask"
        if len(req.output_ids) < SGLANG_CCSPEC_MIN_OUTPUT_LEN:
            return (
                f"CCSpec requires every request to have generated >= "
                f"{SGLANG_CCSPEC_MIN_OUTPUT_LEN} output tokens already "
                f"(req has {len(req.output_ids)})"
            )
        # Max-new stop design: the chunk emits 1 (priming) + up to
        # chunk_cap tokens, and the device-side stop is EOS-only (see
        # module docstring). Requiring remaining >= chunk_cap + 1 makes it
        # arithmetically impossible for any row to exceed its max_new
        # budget before the chunk's final tick: the worst case (no EOS,
        # emitted == chunk_cap) lands exactly ON the budget at the final
        # tick, which the host then finishes normally when it processes
        # that result. With remaining == chunk_cap the final captured tick
        # would emit one token PAST the budget -- so that case is rejected.
        max_new = getattr(req.sampling_params, "max_new_tokens", None)
        if max_new is not None and max_new - len(req.output_ids) < chunk_cap + 1:
            return (
                f"CCSpec requires remaining max_new_tokens >= chunk_cap + 1 "
                f"= {chunk_cap + 1} (req has {max_new - len(req.output_ids)})"
            )

    # -- Attention backend gate -------------------------------------------
    # The captured body traces the backend's decode path with NO per-tick
    # host metadata refresh; only backends that explicitly declare their
    # decode path sync-free and fixed-shape under a frozen (in_capture=True)
    # metadata plan may opt in. Default False on the base class.
    attn_backend = getattr(model_runner, "attn_backend", None)
    if not getattr(attn_backend, "supports_ccspec_capture", False):
        return (
            "CCSpec requires an attention backend declaring "
            "supports_ccspec_capture=True "
            f"(got {type(attn_backend).__name__})"
        )

    # -- KV reservation gates ---------------------------------------------
    allocator = batch.token_to_kv_pool_allocator
    if getattr(allocator, "page_size", 1) != 1:
        return "CCSpec v1 requires page_size == 1 (single-call slot reservation)"
    bs = batch.batch_size()
    needed = bs * chunk_cap
    # Conservative: don't count tree-cache-evictable tokens. If the alloc
    # would (or might) fail, fall back rather than raise mid-run_batch.
    if allocator.available_size() < needed:
        return (
            f"CCSpec cannot reserve {needed} KV slots "
            f"(available {allocator.available_size()})"
        )
    if batch.seq_lens_cpu is None:
        return "CCSpec requires a live seq_lens_cpu mirror"
    # The pre-written req_to_token rows span positions [S0, S0 + chunk_cap)
    # per request; all must fit in the pool's row width.
    pool_width = batch.req_to_token_pool.req_to_token.shape[1]
    if int(batch.seq_lens_cpu.max()) + chunk_cap > pool_width:
        return "CCSpec chunk would overflow req_to_token row width"

    return None


def ccspec_capture_fingerprint(
    batch: "ScheduleBatch",
    *,
    chunk_cap: int,
    eos_ids_host: List[int],
    model_runner: "ModelRunner",
) -> tuple:
    """Validity fingerprint of everything a captured CCSpec chunk graph is
    frozen against that is NOT re-armed per chunk (host-only data; no device
    reads). A cached capture may be relaunched for a new chunk only if this
    tuple compares EQUAL between capture time and reuse time -- see
    CCSpecChunk.reuse_mismatch_reason for the two extra non-hashable checks
    (forced-token stash identity, live-eligibility) layered on top.

    Field-by-field justification (each field pins one class of tensor the
    captured graph reads by pointer, whose validity cannot be restored by
    the per-chunk re-arm):
      - bs: every carried buffer (cur_input_ids/slot_table/history/...) is
        shaped [.., bs]; a different bs is a different graph.
      - (rid, req_pool_idx) per request, IN ORDER: row r of every carried
        buffer and of req_to_token's pre-writes belongs to batch.reqs[r].
        rid pins request identity AND (since sampling_params/
        forced_token_ids are immutable post-admission) the per-request
        sampling config; req_pool_idx pins the req_to_token row the
        captured attention gathers for that batch row (the graph baked
        fb.req_pool_indices's capture-time VALUES into its gather; equal
        host values == equal gather semantics). Ordering matters: the same
        set in a different order permutes every row.
      - chunk_cap: baked into d_cap, slot_table/history shapes, and the
        req_to_token pre-write width.
      - eos_ids: the continue-predicate compares against the capture-time
        eos buffer; only equal VALUES keep it correct (the scheduler builds
        a fresh device tensor each tick, so identity can't be used).
      - req_to_token data_ptr: the captured attention kernels read the pool
        tensor by raw pointer; a rebound pool invalidates every capture.
      - device / model_runner / attn_backend identity: the graph is a
        recording of THIS runner's weights and THIS backend's frozen
        (in_capture=True, _max_seqlen_k=max_context_len worst-case) decode
        plan on THIS device.

    Deliberately NOT in the fingerprint (because the per-chunk re-arm
    refreshes them -- see CCSpecChunk.rearm_for_reuse's audit table):
    seq_lens values, KV slot reservations, req_to_token row contents,
    the priming token, and the loop-control state.
    """
    return (
        int(batch.batch_size()),
        tuple((req.rid, int(req.kv.req_pool_idx)) for req in batch.reqs),
        int(chunk_cap),
        tuple(int(e) for e in eos_ids_host),
        int(batch.req_to_token_pool.req_to_token.data_ptr()),
        str(batch.device),
        id(model_runner),
        id(getattr(model_runner, "attn_backend", None)),
    )


def ccspec_slot_positions(seq_lens: torch.Tensor, chunk_cap: int) -> torch.Tensor:
    """Positions at which the chunk's reserved slot table rows are written
    into req_to_token, shape [chunk_cap, bs] (row i = captured tick i+1).

    Derivation: at chunk start `seq_lens` holds S0, the PRIMING tick's
    post-prepare_for_decode value -- prepare_for_decode already allocated
    the priming tick's own slot and wrote it at position S0 - 1 (its `locs`
    is the pre-increment seq_lens). Captured tick j (1-based) writes its KV
    at position (S0 - 1) + j, so slot row i (0-based, tick i+1) belongs at
    position S0 + i.
    """
    offsets = torch.arange(
        chunk_cap, device=seq_lens.device, dtype=torch.int64
    ).unsqueeze(1)
    return seq_lens.to(torch.int64).unsqueeze(0) + offsets


def ccspec_continue_flag(
    next_ids: torch.Tensor, eos_ids: torch.Tensor
) -> torch.Tensor:
    """All-or-nothing WHILE-loop continue predicate for one captured tick:
    int32 [1] tensor holding 0 iff ANY row's sampled token is an EOS id,
    else 1. Pure fixed-shape tensor ops (explicit broadcast compare, not
    torch.isin, so the captured op sequence is trivially auditable).
    """
    # [bs, n_eos] compare -> any over both dims. Scalar bool -> int32 [1].
    eos_hit = (next_ids.unsqueeze(1) == eos_ids.to(next_ids.dtype).unsqueeze(0)).any()
    return (~eos_hit).to(torch.int32).reshape(1)


def build_ccspec_synthetic_result(next_token_ids: torch.Tensor):
    """One captured tick's output as a plain-decode GenerationBatchResult:
    `next_token_ids` [bs] (CPU for drained extras, device for the chunk's
    terminal result -- the terminal one flows through the normal
    relay/copy_to_cpu path exactly like a real tick's), logits_output=None,
    NO accept_lens (the non-spec result path has no accept concept;
    process_batch_result_decode appends next_token_ids[i] unconditionally
    per row). logits_output=None is safe ONLY because check_ccspec_eligible
    excludes every flag that would dereference it (return_logprob /
    return_hidden_states / return_sampling_mask) -- same audit as CCD's
    build_synthetic_result.
    """
    from sglang.srt.managers.utils import GenerationBatchResult

    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        can_run_cuda_graph=True,
        copy_done=None,
    )


def reconcile_ccspec_host_lengths(batch: "ScheduleBatch", emitted: int) -> None:
    """Reconcile ScheduleBatch host-side length bookkeeping after a chunk
    that emitted `emitted` captured ticks (0 <= emitted <= chunk_cap).

    Arithmetic derivation (off-by-one here corrupts the KV of every
    subsequent tick, so spell it out): at run_batch entry the scheduler's
    prepare_for_decode had already advanced this tick's state -- seq_lens =
    seq_lens_cpu = S0 (covers the PRIMING tick), kv_committed_len /
    kv.kv_allocated_len / decode_batch_idx already include the priming
    tick's +1. The captured body then advanced the DEVICE batch.seq_lens in
    place by exactly `emitted` (one add_(1) per executed iteration), i.e.
    terminal device seq_lens == S0 + emitted. This function applies the
    identical +emitted to every HOST mirror prepare_for_decode normally
    advances once per tick, so the next tick's prepare_for_decode (+1 on
    top) sees exactly the state N ordinary ticks would have left:

      seq_lens_cpu:   S0 -> S0 + emitted   (device tensor already there)
      orig_seq_lens:  +emitted             (device; rebind like prepare does)
      seq_lens_sum:   None                 (lazily recomputed, same contract)
      per-request decode_batch_idx / kv_committed_len / kv.kv_allocated_len:
                      +emitted each        (prepare_for_decode/alloc_for_decode
                                            do +1 per tick; the chunk's extra
                                            slots are net `emitted` per row
                                            after run_ccspec_chunk frees the
                                            (chunk_cap - emitted) unused rows)
    """
    if emitted == 0:
        return
    batch.seq_lens_cpu = batch.seq_lens_cpu + emitted
    batch.orig_seq_lens = batch.orig_seq_lens + emitted
    batch.seq_lens_sum = None
    for req in batch.reqs:
        req.decode_batch_idx += emitted
        req.kv.kv_committed_len += emitted
        req.kv.kv_allocated_len += emitted


class CCSpecChunk:
    """One capturable WHILE-loop chunk for a fixed-bs, fixed-request-set
    plain decode batch. Default (v1) lifecycle: construct once per chunk;
    `prime_and_capture` once; `arm` + `launch` once; `read_history`;
    `destroy` -- fresh capture per chunk.

    Under SGLANG_CCSPEC_CAPTURE_REUSE (opt-in), run_ccspec_chunk instead
    keeps one instance alive across consecutive chunks: per later chunk it
    checks `reuse_mismatch_reason` (strict fingerprint -- any mismatch
    destroys and recaptures), then `rearm_for_reuse` + `arm` + `launch` +
    `read_history`, skipping the capture entirely.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        *,
        bs: int,
        chunk_cap: int,
        eos_ids: torch.Tensor,
        device: torch.device,
        eos_ids_host: Optional[List[int]] = None,
    ):
        self.model_runner = model_runner
        self.bs = bs
        self.chunk_cap = chunk_cap
        self.eos_ids = eos_ids
        self.device = device
        # Host copy of eos_ids for the reuse fingerprint (None when reuse
        # is off -- the caller then never took the .tolist() sync).
        self.eos_ids_host = eos_ids_host
        self.fingerprint: Optional[tuple] = None

        # WHILE-loop control state (read by the cond kernel via raw pointers;
        # updated by torch ops inside the body -- CCSpec needs no custom
        # finish kernel, unlike CCD's variable-accept-length machinery).
        self.n_run = torch.ones(1, dtype=torch.int32, device=device)
        self.d_step = torch.zeros(1, dtype=torch.int32, device=device)
        self.d_cap = torch.full((1,), chunk_cap, dtype=torch.int32, device=device)

        # Device carry state, all pre-allocated; the body only ever writes
        # these in place (capture traces Python once -- only tensor ops on
        # these fixed buffers replay).
        self.cur_input_ids = torch.zeros(bs, dtype=torch.int64, device=device)
        self.positions_buf = torch.zeros(bs, dtype=torch.int64, device=device)
        self.out_loc_buf = torch.zeros(bs, dtype=torch.int64, device=device)
        # Per-tick token history; one combined D2H after launch.
        self.history = torch.zeros(
            (chunk_cap, bs), dtype=torch.int64, device=device
        )

        self.slot_table: Optional[torch.Tensor] = None  # [chunk_cap, bs] int64
        self.body_fb: Optional[ForwardBatch] = None
        self.batch: Optional["ScheduleBatch"] = None
        # The capture-time batch.seq_lens tensor: the ONE buffer the captured
        # body advances in place and the captured attention reads. prepare_
        # for_decode REBINDS batch.seq_lens to a new tensor every tick
        # (`self.seq_lens = self.seq_lens + 1`), so on reuse this buffer is
        # stale and must be refreshed from the live tensor (D2D) before
        # launch and copied back after -- see rearm_for_reuse /
        # sync_seq_lens_back.
        self.seq_lens_buf: Optional[torch.Tensor] = None
        self._cond_graph = None

    def _reserve_and_prewrite_slots(
        self, batch: "ScheduleBatch", *, fresh_capture: bool
    ) -> None:
        """Reserve this chunk's bs * chunk_cap KV slots (ONE host call) and
        pre-write them into req_to_token, outside any capture.

        Pre-write safety (both fresh and reuse): row i lands at position
        S0 + i (see ccspec_slot_positions' derivation). Writing ahead is
        safe: attention at captured tick j reads req_to_token rows only up
        to seq_lens-1 == S0 + j - 1, i.e. exactly the rows ticks <= j own;
        rows past `emitted` are never read (seq_lens never covers them) and
        the next ordinary prepare_for_decode overwrites position
        S0 + emitted before the sequence ever grows over it -- so freeing
        the unused slots after launch needs no req_to_token rollback (see
        run_ccspec_chunk).

        fresh_capture=True binds self.slot_table to the new allocation (the
        buffer the capture will bake in). fresh_capture=False (reuse) MUST
        instead REWRITE the already-captured buffer in place -- the graph
        gathers out_cache_loc from self.slot_table by pointer, so binding a
        new tensor would leave the replay reading last chunk's (freed)
        slots.
        """
        from sglang.srt.mem_cache.allocation import alloc_token_slots

        flat = alloc_token_slots(batch.tree_cache, self.bs * self.chunk_cap)
        if fresh_capture:
            self.slot_table = flat.to(torch.int64).view(self.chunk_cap, self.bs)
        else:
            if self.slot_table is None:
                raise RuntimeError(
                    "CCSpecChunk: reuse re-arm before any capture (no slot_table)"
                )
            # In-place value rewrite; pointer identity with the captured
            # gather is the load-bearing property.
            self.slot_table.copy_(flat.view(self.chunk_cap, self.bs))

        req_to_token = batch.req_to_token_pool.req_to_token
        pos = ccspec_slot_positions(batch.seq_lens, self.chunk_cap)
        rows = (
            batch.req_pool_indices.to(torch.int64)
            .unsqueeze(0)
            .expand(self.chunk_cap, self.bs)
        )
        req_to_token[rows, pos] = self.slot_table.to(req_to_token.dtype)

    def reuse_mismatch_reason(
        self,
        batch: "ScheduleBatch",
        *,
        chunk_cap: int,
        eos_ids_host: List[int],
        model_runner: "ModelRunner",
    ) -> Optional[str]:
        """Return None iff this chunk's captured graph is provably still
        valid for `batch` (so rearm_for_reuse + launch may run), else a
        human-readable mismatch reason -- the caller must then destroy this
        chunk and capture fresh. Host-only Python; never touches the device.

        Three layers, all mandatory:
          1. The capture must exist and have recorded a fingerprint.
          2. The recorded fingerprint must equal the batch's current one
             (see ccspec_capture_fingerprint for what each field pins).
          3. The forced-token lookup the captured body reads must be THE
             stash object currently on the batch (identity, not equality:
             a restash rebinds new device tensors the graph cannot see),
             and the stash's own rid fingerprint must match the current
             request list. Both None (no oracle mode) also matches.
        """
        if self._cond_graph is None or self.fingerprint is None:
            return "no live capture / no recorded fingerprint"
        current = ccspec_capture_fingerprint(
            batch,
            chunk_cap=chunk_cap,
            eos_ids_host=eos_ids_host,
            model_runner=model_runner,
        )
        if current != self.fingerprint:
            return (
                "fingerprint mismatch (request set/order, req_pool_indices, "
                "bs, chunk_cap, eos_ids, pool, or runner changed)"
            )
        if self.body_fb is None:
            return "no captured ForwardBatch"
        stash = getattr(batch, "_ccd_forced_lookup", None)
        if stash is not self.body_fb.forced_token_lookup:
            return "forced-token stash was rebuilt/rebound since capture"
        stash_rids = getattr(batch, "_ccd_forced_lookup_rids", None)
        if stash_rids != [req.rid for req in batch.reqs]:
            return "forced-token stash rid fingerprint does not match batch"
        return None

    def rearm_for_reuse(self, batch: "ScheduleBatch", model_worker):
        """Per-chunk re-arm of every carried device buffer whose value
        changes between chunks, WITHOUT recapture. Caller must have already
        confirmed reuse_mismatch_reason(...) is None -- this method re-runs
        only the cheap live-eligibility gate (allocator headroom, max_new
        budgets, row-width overflow change per tick and are not part of the
        static fingerprint).

        Re-arm audit (buffer the captured graph reads -> refreshed how):
          - slot_table: new KV reservation each chunk; values REWRITTEN in
            the same pre-allocated buffer (_reserve_and_prewrite_slots with
            fresh_capture=False).
          - req_to_token rows: re-pre-written at the new positions S0'+i.
          - cur_input_ids: this chunk's priming token (D2D copy).
          - seq_lens_buf: prepare_for_decode rebound batch.seq_lens since
            capture; copy the live values INTO the captured buffer (D2D).
            The graph keeps reading/advancing the captured buffer;
            sync_seq_lens_back mirrors the terminal value out after launch.
          - fb.req_pool_indices: values proven equal by the fingerprint
            (host (rid, req_pool_idx) pairs); if the live tensor was
            rebound to a new-but-equal buffer, defensively refresh the
            captured buffer anyway (D2D) so replay never depends on the
            rebound tensor's lifetime.
          - positions_buf / out_loc_buf: no re-arm -- recomputed inside the
            captured body every iteration from seq_lens_buf / slot_table.
          - n_run / d_step: re-armed by the caller's arm() from the priming
            token, exactly as on a fresh capture.
          - history: rows [0, emitted) are overwritten by the launch; stale
            rows past `emitted` are never read (read_history bounds).
          - eos_ids / forced_token_lookup: static by fingerprint; not
            touched (a rebuild would rebind tensors the graph cannot see).

        Returns the priming tick's real GenerationBatchResult (process it
        like any other tick's), mirroring prime_and_capture's contract.
        """
        reason = check_ccspec_eligible(
            batch, self.model_runner, chunk_cap=self.chunk_cap
        )
        if reason is not None:
            raise RuntimeError(
                f"CCSpecChunk: reuse re-arm on ineligible batch: {reason}"
            )
        if self.seq_lens_buf is None or self.body_fb is None:
            raise RuntimeError("CCSpecChunk: reuse re-arm before capture")
        self.batch = batch

        # NOTE: no stash_ccd_forced_lookup here. reuse_mismatch_reason
        # proved the existing stash matches this exact request list and is
        # the very object the captured body reads; restashing would rebind
        # fresh device tensors -- the priming tick would then apply forced
        # tokens from the new tensors while the replayed ticks read the old
        # ones. The priming tick's init_new resolves the (validated) stash.
        self._reserve_and_prewrite_slots(batch, fresh_capture=False)

        priming_result = model_worker.forward_batch_generation(batch)
        self.cur_input_ids.copy_(priming_result.next_token_ids)

        # Refresh the captured seq_lens buffer from the live (rebound)
        # tensor. D2D only. Plain-decode forwards never mutate seq_lens, so
        # ordering this after the priming tick is equivalent to before it.
        if batch.seq_lens is not self.seq_lens_buf:
            self.seq_lens_buf.copy_(batch.seq_lens)
        fb_rpi = self.body_fb.req_pool_indices
        if (
            fb_rpi is not None
            and fb_rpi.data_ptr() != batch.req_pool_indices.data_ptr()
        ):
            fb_rpi.copy_(batch.req_pool_indices)

        return priming_result

    def sync_seq_lens_back(self, batch: "ScheduleBatch") -> None:
        """After a reuse launch: the graph advanced the CAPTURED seq_lens
        buffer to S0 + emitted; mirror that into the live batch.seq_lens
        (rebound since capture) so the scheduler's shared post-branch code
        (future_map publish of `batch.seq_lens + 1`, the next tick's
        prepare_for_decode `+ 1`) sees exactly what the fresh-capture path
        leaves behind (where the two tensors alias and this is a no-op)."""
        if batch.seq_lens is not self.seq_lens_buf:
            batch.seq_lens.copy_(self.seq_lens_buf)

    def prime_and_capture(self, batch: "ScheduleBatch", model_worker):
        """Reserve the chunk's KV, run the real eager priming tick, set up
        the device carry state, and capture the WHILE-loop body. Returns the
        priming tick's real GenerationBatchResult (process it like any other
        tick's -- it consumed real tokens)."""
        reason = check_ccspec_eligible(
            batch, self.model_runner, chunk_cap=self.chunk_cap
        )
        if reason is not None:
            raise RuntimeError(f"CCSpecChunk: batch is not CCSpec-eligible: {reason}")
        self.batch = batch

        # Oracle mode: build the forced-token lookup exactly once, before
        # the priming tick and the capture -- rebuilding inside the captured
        # region would freeze a fresh unpinned H2D copy at trace time (the
        # exact CCD failure documented in forced_token_pool.py). init_new
        # below resolves this stash (rid-fingerprinted) for both the priming
        # tick's ForwardBatch and the body's.
        stash_ccd_forced_lookup(batch, self.device)

        # KV reservation -- slot-row-0 choice: the priming tick uses the
        # batch's OWN out_cache_loc, already allocated and req_to_token-
        # written by prepare_for_decode before run_batch (that allocation IS
        # this tick's slot; re-deriving it here would double-count the
        # scheduler's accounting). We therefore reserve chunk_cap EXTRA rows
        # for the captured ticks only, in ONE host call.
        self._reserve_and_prewrite_slots(batch, fresh_capture=True)

        # The real, eager priming tick -- exactly the call run_batch's
        # normal plain-decode branch would have made. Consumes this tick's
        # prepared state (input_ids from the future-map resolve,
        # out_cache_loc from prepare_for_decode).
        priming_result = model_worker.forward_batch_generation(batch)

        # Seed the carry: captured tick 1's input is the priming token.
        self.cur_input_ids.copy_(priming_result.next_token_ids)

        # Build the body's ForwardBatch once, eagerly, from the live batch,
        # then repoint the per-tick-varying fields at the carried buffers.
        # fb.seq_lens aliases batch.seq_lens (init_new passes it by
        # reference), which the body advances in place -- the same carry
        # design as CCD (checkpoint 062/063).
        fb = ForwardBatch.init_new(
            batch, self.model_runner, return_hidden_states_before_norm=False
        )
        fb.input_ids = self.cur_input_ids
        fb.positions = self.positions_buf
        fb.out_cache_loc = self.out_loc_buf
        self.body_fb = fb
        # Pin the capture-time seq_lens buffer BY OBJECT (== fb.seq_lens ==
        # batch.seq_lens right now): the recorded graph ops target this
        # exact allocation forever; the live batch.seq_lens attribute gets
        # rebound to a new tensor every subsequent prepare_for_decode.
        seq_lens_buf = batch.seq_lens
        self.seq_lens_buf = seq_lens_buf

        # Reuse-mode bookkeeping: the fingerprint the NEXT chunk must equal
        # for this capture to be relaunched (host-only; None when reuse is
        # off so the default path takes no eos_ids sync).
        if self.eos_ids_host is not None:
            self.fingerprint = ccspec_capture_fingerprint(
                batch,
                chunk_cap=self.chunk_cap,
                eos_ids_host=self.eos_ids_host,
                model_runner=self.model_runner,
            )

        # Freeze attention metadata under capture semantics (fixed shapes
        # sized to worst case, no host reads of live lengths) -- the same
        # in_capture=True call the normal decode CUDA-graph capture sites
        # make. Nothing refreshes metadata per captured tick; that is
        # exactly what supports_ccspec_capture certifies is safe for this
        # backend's decode path.
        self.model_runner.attn_backend.init_forward_metadata_out_graph(
            fb, in_capture=True
        )

        def body_fn(stream_ptr):
            # ONE plain decode tick, mirroring prepare_for_decode +
            # model forward + greedy sample field-for-field:
            #   seq_lens += 1            (prepare_for_decode's advance)
            #   positions = seq_lens - 1 (ForwardBatch.init_new's decode rule)
            #   out_cache_loc = slot_table[d_step]  (this tick's reserved row,
            #       gathered by the DEVICE step counter -- replays correctly)
            # seq_lens_buf IS batch.seq_lens at trace time (pinned above);
            # using the pinned handle makes the buffer the replay reads
            # explicit -- identical captured ops either way.
            seq_lens_buf.add_(1)
            self.positions_buf.copy_(torch.clamp(seq_lens_buf - 1, min=0))
            step = self.d_step.to(torch.int64)
            self.out_loc_buf.copy_(
                self.slot_table.index_select(0, step).view(self.bs)
            )

            # Graph-recordable per-tick metadata hook (no-op for the
            # backends currently allowed by supports_ccspec_capture; kept
            # for contract fidelity with the 3-method backend init API).
            self.model_runner.attn_backend.init_forward_metadata_in_graph(fb)

            # Model layers resolve the attention backend via the ambient
            # ForwardContext (get_attn_backend) -- publish one for the trace
            # if the caller hasn't (mirrors ModelRunner._forward_raw).
            if has_forward_context():
                ctx = contextlib.nullcontext()
            else:
                ctx = forward_context(
                    ForwardContext(attn_backend=self.model_runner.attn_backend)
                )
            with ctx:
                logits_output = self.model_runner.model.forward(
                    fb.input_ids, fb.positions, fb
                )

            # Greedy sample. Eligibility already excluded everything
            # ModelRunner.sample adds beyond argmax + forced tokens.
            next_ids = torch.argmax(logits_output.next_token_logits, dim=-1)
            next_ids = apply_forced_tokens(
                next_ids, fb.positions, fb.forced_token_lookup
            )

            # Record BEFORE advancing d_step -- history row == this tick's
            # pre-increment step index (same pattern as CCD's index_copy_).
            self.history.index_copy_(0, step, next_ids.unsqueeze(0))

            # All-or-nothing stop + carries.
            self.n_run.copy_(ccspec_continue_flag(next_ids, self.eos_ids))
            self.cur_input_ids.copy_(next_ids)
            self.d_step.add_(1)

        self._cond_graph = CCConditionalDecode().capture(
            body_fn=body_fn,
            finish_ka=None,  # CCSpec's stop is pure torch ops; no finish kernel
            n_run=self.n_run,
            d_step=self.d_step,
            d_cap=self.d_cap,
            torch_mod=torch,
        )
        return priming_result

    def arm(self) -> None:
        """Reset loop control before launch. If the PRIMING tick already
        emitted an EOS for any row, the WHILE predicate must be false on
        entry so ZERO captured ticks run (the chunk then degenerates to a
        normal single-tick run_batch)."""
        self.d_step.zero_()
        self.n_run.copy_(ccspec_continue_flag(self.cur_input_ids, self.eos_ids))

    def launch(self, stream: torch.cuda.Stream) -> None:
        if self._cond_graph is None:
            raise RuntimeError("CCSpecChunk: capture() has not been called yet")
        self._cond_graph.launch(stream.cuda_stream)

    def read_history(self) -> Tuple[int, torch.Tensor]:
        """One combined D2H after launch: returns (emitted, history_cpu
        [chunk_cap, bs] int64). Must be called with the current stream being
        the launch stream (read_history's stream-ordering contract, same as
        CCD's: `.cpu()` only synchronizes against the stream it's issued
        on). `emitted == d_step` -- one increment per executed body
        iteration, so rows [0, emitted) of history are valid.
        """
        history_cpu = self.history.cpu()
        emitted = int(self.d_step.cpu().item())
        return emitted, history_cpu

    def destroy(self) -> None:
        """Release the captured graph's native CUDA resources (graph exec +
        private mem pool). Same leak class as CondGraph.destroy documents:
        Python GC alone never frees the driver-side exec. The carry/history
        tensors are ordinary allocator tensors created BEFORE capture and
        stay valid afterwards (results may hold views into history)."""
        if self._cond_graph is not None:
            self._cond_graph.destroy()
            self._cond_graph = None


# SGLANG_CCSPEC_CAPTURE_REUSE only: the single cached chunk (one scheduler
# process runs one running batch, and the TP=1 eligibility gate means one
# rank). None when reuse is off, before the first chunk, and after any
# fingerprint mismatch (which destroys the stale capture before recapturing).
_reuse_cached_chunk: Optional[CCSpecChunk] = None


def run_ccspec_chunk(
    model_worker,
    batch: "ScheduleBatch",
    *,
    chunk_cap: int,
    eos_ids: torch.Tensor,
    stream: torch.cuda.Stream,
    on_publish: Optional[Callable[[torch.Tensor], None]] = None,
) -> List:
    """Run one CCSpec chunk against `batch` (the live running batch, mutated
    in place). Caller must have already confirmed check_ccspec_eligible is
    None. Returns `1 + emitted` GenerationBatchResult objects in order: the
    real priming tick's result, then one synthetic plain-decode result per
    emitted captured tick. Feed every one through the existing per-tick
    result pipeline in order (scheduler: last one is the tick's
    batch_result, the rest go through the CCD extras drain).

    Terminal-state contract for the scheduler's shared post-branch code
    (traced, not guessed):
      - `future_map.publish(future_indices, batch.seq_lens + 1)`: for a
        normal plain tick this publishes S0 + 1, i.e. "the value the NEXT
        tick's prepare_for_decode will hold post-advance". The body carried
        batch.seq_lens to S0 + emitted in place, so the shared publish
        writes S0 + emitted + 1 -- exactly the next tick's post-prepare
        value. (For non-spec batches new_seq_lens_buf currently has no
        reader -- resolve_seq_lens_cpu gates on spec_info -- but the
        published value stays convention-correct regardless.)
      - `_relay_forward_payload(future_indices, batch_result)`: stashes
        batch_result.next_token_ids into output_tokens_buf; the next tick's
        resolve_forward_inputs gathers it as input_ids. It must therefore
        carry the LAST emitted tick's tokens, on DEVICE (stash is a
        device-side scatter). The terminal result's next_token_ids is the
        device history row for exactly this reason; copy_to_cpu then
        converts it for result processing like any real tick's.

    `on_publish`, if given, is called once with the terminal batch.seq_lens
    (API parity with run_ccd_chunk; the scheduler's plain-decode branch
    relies on the shared publish line instead and passes nothing).
    """
    global _reuse_cached_chunk
    model_runner = model_worker.model_runner
    bs = batch.batch_size()
    reuse = SGLANG_CCSPEC_CAPTURE_REUSE

    chunk: Optional[CCSpecChunk] = None
    eos_ids_host: Optional[List[int]] = None
    if reuse:
        # Host copy of eos_ids for the value fingerprint (the scheduler
        # builds a fresh device tensor from static model config each tick,
        # so identity can't prove anything). Tiny pre-launch D2H, reuse
        # mode only.
        eos_ids_host = [int(e) for e in eos_ids.cpu().tolist()]
        if _reuse_cached_chunk is not None:
            # Take ownership: the cache is re-populated only after this
            # chunk fully succeeds (launch + readback). If anything below
            # raises mid-re-arm, the half-armed capture is abandoned (never
            # relaunched) instead of lingering for a later call whose
            # fingerprint check couldn't see the partial state.
            cached = _reuse_cached_chunk
            _reuse_cached_chunk = None
            mismatch = cached.reuse_mismatch_reason(
                batch,
                chunk_cap=chunk_cap,
                eos_ids_host=eos_ids_host,
                model_runner=model_runner,
            )
            if mismatch is None:
                chunk = cached
            else:
                # Stale capture: destroy BEFORE recapturing -- never keep a
                # graph whose fingerprint check failed, and never launch one
                # whose check was skipped.
                cached.destroy()

    if chunk is not None:
        priming_result = chunk.rearm_for_reuse(batch, model_worker)
    else:
        chunk = CCSpecChunk(
            model_runner,
            bs=bs,
            chunk_cap=chunk_cap,
            eos_ids=eos_ids,
            device=batch.device,
            eos_ids_host=eos_ids_host,
        )
        priming_result = chunk.prime_and_capture(batch, model_worker)
    chunk.arm()
    chunk.launch(stream)
    emitted, history_cpu = chunk.read_history()

    if reuse:
        # Keep the capture alive for the next chunk; mirror the terminal
        # seq_lens out of the captured buffer into the live (possibly
        # rebound) batch.seq_lens first -- no-op when they alias (the
        # fresh-capture case).
        chunk.sync_seq_lens_back(batch)
        _reuse_cached_chunk = chunk
    else:
        # Release graph exec + private pool now -- readback above is a
        # synchronous .cpu(), so the launch is fully complete (CCD task
        # #136's leak class).
        chunk.destroy()

    # Return the unused reserved rows to the allocator. No req_to_token
    # rollback needed: rows past `emitted` are provably unread until the
    # next prepare_for_decode overwrites position S0 + emitted first (see
    # _reserve_and_prewrite_slots' pre-write comment).
    if emitted < chunk_cap:
        free_tail = chunk.slot_table[emitted:].reshape(-1)
        if reuse:
            # The allocator may hold the passed tensor by REFERENCE (e.g.
            # TokenToKVPoolAllocator.free appends it to free_group inside a
            # free group); under reuse the next chunk's re-arm REWRITES
            # slot_table in place, which would corrupt the free list's
            # view. Hand the allocator its own copy.
            free_tail = free_tail.clone()
        batch.token_to_kv_pool_allocator.free(free_tail)

    # Host bookkeeping: +emitted on every mirror prepare_for_decode would
    # have advanced per tick (full derivation in the helper's docstring).
    reconcile_ccspec_host_lengths(batch, emitted)

    results = [priming_result]
    # Extras (all but the last emitted tick) carry CPU tensors from the one
    # combined D2H; the drain path's copy_to_cpu is then a no-op for them.
    for t in range(emitted - 1):
        results.append(build_ccspec_synthetic_result(history_cpu[t]))
    if emitted > 0:
        terminal_tokens = chunk.history[emitted - 1]
        if reuse:
            # Decouple the terminal DEVICE tokens from the history buffer:
            # under reuse the NEXT chunk's launch overwrites history, and
            # the scheduler's non-CC result copy runs on copy_stream, only
            # ordered after forward work enqueued so far -- a later launch
            # into the same buffer could race it. The fresh-capture path
            # never reuses the buffer, so it keeps the zero-copy view.
            terminal_tokens = terminal_tokens.clone()
        # Terminal result: DEVICE tokens, so the shared relay/copy path
        # treats it exactly like a real tick's result (see docstring).
        results.append(build_ccspec_synthetic_result(terminal_tokens))

    if on_publish is not None:
        on_publish(batch.seq_lens)

    return results
