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
"""Oracle-mode ("teacher-forced" decode) token lookup.

A request with `SamplingParams.forced_token_ids` set skips real sampling
during decode and instead emits the exact tokens supplied with the request,
at the position they'd naturally be generated. This replays a known dataset
trace's real compute profile (forward-pass cost, KV-cache growth,
prefix-cache reuse on later turns) without depending on the model's own
sampling actually reproducing that text.

Two consumers:
- Plain (non-speculative) decode/prefill: `ModelRunner.sample()` calls
  `apply_forced_tokens()` after the real sampler runs (single token/row/tick).
- DFlash speculative decode (eager, non-CCD only -- see below):
  `DFlashWorkerV2.forward_batch_generation()` runs the real draft model, the
  real target-verify forward pass, and their real (unforced) accept-length
  computation completely unchanged -- nothing about either forward pass is
  touched, so their compute cost is identical to normal serving. Only
  afterward, for oracle-mode rows, does `compute_oracle_commit_via_draft_logprob()`
  override `out_tokens`/`commit_lens` -- gated by the draft model's own real
  log-probability of the oracle tokens (not a fixed/full window and not
  argmax equality; oracle tokens usually come from a different model than
  the one being served, so real, data-dependent draft confidence is the
  realistic proxy for "how many tokens would a real speculative-decode tick
  have bundled here"). `compute_forced_block()` is the simpler,
  unconditional-full-window predecessor, kept for callers that don't need
  the data-dependent commit length.

CCD (`speculative/dflash_cc_conditional_decode.py`) captures
`forward_batch_generation` in a CUDA graph and replays it multiple times per
single host launch. `build_forced_token_lookup()` does a fresh Python-side
build (new host lists, new device tensor) every call -- correct for the
eager path (a real host round-trip every tick), but if it ran *inside* a
captured region it would only actually execute once, during capture: replay
re-runs the captured GPU kernels only, not the Python code that built their
inputs, so the lookup's contents would silently freeze at whatever a single
fresh allocation happened to hold. `DflashCCDChunk.prime_and_capture()`
therefore builds the lookup exactly once, *before* either the priming tick
or the capture that follows (sized for the whole chunk via `chunk_batch.reqs`,
which cannot change mid-chunk -- the same invariant that makes the chunk
capturable at all) and stashes it on `chunk_batch._ccd_forced_lookup`.
`forward_batch_generation`'s oracle block checks for this attribute first,
falling back to a fresh per-call build only when absent (the eager path,
unchanged). Everything downstream of the lookup (`gather_oracle_window`,
`compute_oracle_commit_via_draft_logprob`) was already indexed purely by
`prefix_lens` (== `batch.seq_lens`, updated in place every captured tick --
confirmed to advance correctly in-graph since CCD's own attention math
depends on it), so building the lookup once and reusing it is sufficient;
no per-tick code needed to change. `check_ccd_eligible()` no longer excludes
forced_token_ids batches.

No-op by construction when no request in a batch sets `forced_token_ids`:
`build_forced_token_lookup` returns None, and `apply_forced_tokens`/
`compute_forced_block` return their input unchanged (or None) in that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclass
class ForcedTokenLookup:
    """Per-row (batch-aligned) forced-token state, already on `device`.

    ``token_ids``: ``[bs, max_len]`` int64, right-padded with zeros.
    ``valid_len``: ``[bs]`` int32, ``0`` for rows with no forcing -- this is
        the only field that gates lookup, so the zero padding above is inert.
    ``start_offset``: ``[bs]`` int64, ``len(origin_input_ids)`` per row --
        the sequence position at which that row's forced tokens begin.
    """

    token_ids: torch.Tensor
    valid_len: torch.Tensor
    start_offset: torch.Tensor


def build_forced_token_lookup(
    reqs: List["Req"], device: torch.device
) -> Optional[ForcedTokenLookup]:
    """Build a batch-aligned forced-token lookup from `reqs`, or None if no
    row in this batch has `sampling_params.forced_token_ids` set (the common
    case -- callers should skip all further oracle-mode work on None)."""

    forced_lists = [
        getattr(req.sampling_params, "forced_token_ids", None) for req in reqs
    ]
    max_len = max((len(f) for f in forced_lists if f), default=0)
    if max_len == 0:
        return None

    bs = len(reqs)
    token_ids = torch.zeros((bs, max_len), dtype=torch.int64)
    valid_len = torch.zeros(bs, dtype=torch.int32)
    start_offset = torch.zeros(bs, dtype=torch.int64)

    for i, (req, forced) in enumerate(zip(reqs, forced_lists)):
        if not forced:
            continue
        token_ids[i, : len(forced)] = torch.tensor(forced, dtype=torch.int64)
        valid_len[i] = len(forced)
        start_offset[i] = len(req.origin_input_ids)

    return ForcedTokenLookup(
        token_ids=token_ids.to(device, non_blocking=True),
        valid_len=valid_len.to(device, non_blocking=True),
        start_offset=start_offset.to(device, non_blocking=True),
    )


def stash_ccd_forced_lookup(batch, device: torch.device) -> None:
    """Build and stash a batch-aligned lookup on `batch` for CCD's captured
    region to reuse (see resolve_ccd_forced_lookup). Stashes the batch's
    request identity (rids) alongside it: ForcedTokenLookup rows are
    positional, so a lookup is only valid for the exact request list it was
    built from. ScheduleBatch.filter_batch/merge_batch/copy.copy all
    propagate dynamic attributes untouched, so without the fingerprint a
    lookup stashed before a composition change would silently replay one
    request's forced trace into another's rows."""
    batch._ccd_forced_lookup = build_forced_token_lookup(batch.reqs, device)
    batch._ccd_forced_lookup_rids = [req.rid for req in batch.reqs]


def resolve_ccd_forced_lookup(batch, device: torch.device):
    """Return the CCD-stashed lookup if it matches `batch`'s CURRENT request
    list, else a freshly built one. The fingerprint check is host-only
    Python: during a CCD capture trace the stash is one line old and always
    matches (reuse path -- no capture-unsafe H2D rebuild); on any eager call
    after the batch's composition changed (requests finished, newcomers
    merged, a CCD split filtered rows), the stale stash is ignored and the
    rebuild runs in ordinary eager context where H2D is legal."""
    lookup = getattr(batch, "_ccd_forced_lookup", None)
    if lookup is not None and getattr(batch, "_ccd_forced_lookup_rids", None) == [
        req.rid for req in batch.reqs
    ]:
        return lookup
    return build_forced_token_lookup(batch.reqs, device)


def apply_forced_tokens(
    next_token_ids: torch.Tensor,
    positions: torch.Tensor,
    lookup: Optional[ForcedTokenLookup],
) -> torch.Tensor:
    """Overwrite `next_token_ids` with the forced token wherever `lookup` has
    one valid at this row's current `positions`; passes through unchanged
    everywhere else (including every row when `lookup` is None).

    `positions` is the position of the token just consumed this step (as
    passed to `Sampler.forward`, i.e. `forward_batch.positions` in decode
    mode or `forward_batch.seq_lens - 1` in prefill mode) -- the produced
    token lands at `positions + 1`.
    """
    if lookup is None:
        return next_token_ids

    idx = positions.to(torch.int64) + 1 - lookup.start_offset
    in_range = (idx >= 0) & (idx < lookup.valid_len.to(torch.int64))
    idx_clamped = idx.clamp(min=0, max=lookup.token_ids.shape[1] - 1)
    forced = lookup.token_ids.gather(1, idx_clamped.unsqueeze(1)).squeeze(1)
    return torch.where(in_range, forced, next_token_ids)


def gather_oracle_window(
    prefix_lens: torch.Tensor,
    width: int,
    lookup: ForcedTokenLookup,
) -> tuple:
    """Gather a `width`-wide window of oracle tokens per row, starting at
    the global sequence position in `prefix_lens` (name kept for API
    stability; callers must pass the position of the FIRST TOKEN THE WINDOW
    STREAMS). In `DFlashWorkerV2` that is ``batch.seq_lens + 1``:
    ``batch.seq_lens`` (== kv_committed_len) is the position of the
    last-streamed token -- the pending bonus, ``candidates[:, 0]`` -- and
    ``out_tokens[:, j]`` streams at position ``seq_lens + 1 + j``, the same
    ``+ 1`` convention as `apply_forced_tokens`. (An earlier revision passed
    bare ``seq_lens`` here, re-emitting the already-streamed token at the
    head of every window: one duplicated token per prefill->decode handoff,
    which also broke decode-radix cross-turn prefix reuse.)

    Returns `(oracle_tokens[bs, width] int64, in_range[bs, width] bool)`.
    `in_range[r, i]` is False once row `r`'s forced sequence is exhausted at
    that offset (no forced token left to place there) -- callers use this to
    fall back to real generation for the exhausted tail, same fallback
    behavior as plain-decode oracle mode running past `forced_token_ids`'s
    end. Shared by `compute_forced_block` (unconditional-window commit) and
    `compute_oracle_commit_via_draft_logprob` (draft-confidence-gated commit).
    """
    idx0 = prefix_lens.to(torch.int64) - lookup.start_offset  # [bs]
    offsets = torch.arange(width, device=prefix_lens.device)  # [width]
    idx = idx0.unsqueeze(1) + offsets.unsqueeze(0)  # [bs, width]
    valid_len = lookup.valid_len.to(torch.int64).unsqueeze(1)  # [bs, 1]
    in_range = (idx >= 0) & (idx < valid_len)
    idx_clamped = idx.clamp(min=0, max=lookup.token_ids.shape[1] - 1)
    oracle_tokens = lookup.token_ids.gather(1, idx_clamped)
    return oracle_tokens, in_range


def compute_forced_block(
    prefix_lens: torch.Tensor,
    block_size: int,
    lookup: Optional[ForcedTokenLookup],
) -> Optional[tuple]:
    """DFlash-block analog of `apply_forced_tokens`: force the *entire*
    available window every tick, no matter what the draft/target models
    would have produced. Superseded by `compute_oracle_commit_via_draft_logprob`
    for realistic (data-dependent, not artificially-full) commit lengths --
    kept for callers that want the simpler, unconditional behavior.

    Returns `(forced_tokens[bs, block_size] int64, forced_commit_lens[bs]
    int32, override_mask[bs] bool)`, or None if `lookup` is None. Caller
    applies via `torch.where(override_mask, forced_*, real_*)`.
    """
    if lookup is None:
        return None
    forced_tokens, in_range = gather_oracle_window(prefix_lens, block_size, lookup)
    forced_commit_lens = in_range.to(torch.int32).sum(dim=1)
    override_mask = forced_commit_lens > 0
    return forced_tokens, forced_commit_lens, override_mask


def compute_oracle_commit_via_draft_logprob(
    prefix_lens: torch.Tensor,
    draft_log_probs: torch.Tensor,
    log_prob_threshold: float,
    lookup: Optional[ForcedTokenLookup],
) -> Optional[tuple]:
    """Data-dependent DFlash-block oracle commit length, gated by the DRAFT
    model's own real log-probability of the oracle token at each block
    position -- not argmax equality. The oracle tokens were generated by a
    different model than the one being served, so requiring the draft's
    argmax to literally equal the oracle token would almost always fail past
    the first position, collapsing commit_len to 1 every tick regardless of
    how plausible the token actually was to this model.

    `draft_log_probs` is `[bs, block_size]`: the draft's real log P(token) for
    the oracle token at each position, computed by the caller (dflash_worker_v2.py,
    via a TP-aware chunked log-softmax reusing the draft's real hidden
    states -- this function only turns that per-position confidence signal
    into a commit length; it doesn't touch the draft/target forward passes,
    so it can't affect their real compute cost).

    Returns `(oracle_tokens[bs, block_size], commit_lens[bs] int32,
    override_mask[bs] bool)`, or None if `lookup` is None. `commit_lens` is
    the longest prefix where `draft_log_probs >= log_prob_threshold` (within
    the oracle sequence's remaining range), floored at 1 wherever the row has
    at least one oracle token available this tick -- always make forward
    progress, mirroring DFlash's own bonus-token guarantee, even when the
    draft model is not at all confident about the ground truth.
    """
    if lookup is None:
        return None
    block_size = draft_log_probs.shape[1]
    oracle_tokens, in_range = gather_oracle_window(prefix_lens, block_size, lookup)
    confident = (draft_log_probs >= log_prob_threshold) & in_range
    commit_lens = confident.to(torch.int32).cumprod(dim=1).sum(dim=1)
    row_active = in_range[:, 0]
    commit_lens = torch.where(
        row_active, commit_lens.clamp(min=1), torch.zeros_like(commit_lens)
    )
    return oracle_tokens, commit_lens, row_active


def compute_oracle_commit_fixed_len(
    prefix_lens: torch.Tensor,
    block_size: int,
    fixed_commit_len: int,
    lookup: Optional[ForcedTokenLookup],
) -> Optional[tuple]:
    """Fixed-length DFlash-block oracle commit: emit exactly
    `fixed_commit_len` forced tokens per tick (capped by the window and the
    forced sequence's remaining length), skipping the draft-confidence gate
    entirely.

    Rationale: `compute_oracle_commit_via_draft_logprob`'s gate scores a
    trace generated by a DIFFERENT model, so at the default threshold it
    almost never trusts more than the floor of 1 token/tick -- oracle
    benchmarks then exercise accept-length-1 dynamics, far below the
    3-4-token accept lengths real (self-trace) speculative serving
    measures. Pinning the commit length to that historical average lets
    oracle replay exercise realistic accept-length dynamics. The caller
    should also skip the draft log-prob scoring pass when using this
    (its output would be unused); the draft forward itself still runs --
    its KV/hidden-state bookkeeping is load-bearing for verify.

    Deterministic by design: a per-tick random length would freeze at
    trace time under CCD's CUDA-graph capture and replay identical values
    anyway, so a fixed value is both simpler and honest about what a
    captured region can express.

    Same return contract as compute_oracle_commit_via_draft_logprob:
    `(oracle_tokens[bs, block_size], commit_lens[bs] int32,
    override_mask[bs] bool)`, or None if `lookup` is None. Pure static-shape
    tensor ops throughout -- capture-safe by construction.
    """
    if lookup is None:
        return None
    oracle_tokens, in_range = gather_oracle_window(prefix_lens, block_size, lookup)
    # Longest contiguous in-range prefix per row, capped at the requested
    # fixed length; floor of 1 wherever the row has any forced token left
    # (forward-progress guarantee, same as the gated variant).
    avail = in_range.to(torch.int32).cumprod(dim=1).sum(dim=1)
    commit_lens = avail.clamp(max=max(1, int(fixed_commit_len)))
    row_active = in_range[:, 0]
    commit_lens = torch.where(
        row_active, commit_lens.clamp(min=1), torch.zeros_like(commit_lens)
    )
    return oracle_tokens, commit_lens, row_active
