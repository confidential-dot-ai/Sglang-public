"""DFLASH PD-disaggregation draft-input reconstruction.

Mirrors eagle_disaggregation.py's contract (build_disagg_draft_input),
adapted to DFLASH's actual mechanism: the "legacy Eagle-shaped" fields on
DFlashDraftInputV2 (topk_p/topk_index/hidden_states/etc.) are unused for
DFLASH -- real draft state relays via FutureMap.

NOTE: this function does NOT materialize the draft model's KV cache for the
prompt tokens -- it only builds the *next tick's* draft input. An earlier
version of this docstring claimed draft KV "is already materialized locally
on the decode engine... during PD handoff (the target hidden states cross
via NIXL)" -- that was false; no code in this file or its call chain ever
called DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc. The actual
materialization now happens in disaggregation/decode.py's
get_new_prebuilt_batch(), via _materialize_dflash_draft_kv_for_prebuilt(),
called right after prepare_for_prebuilt() and before process_prebuilt()
(which calls into this function). See that method's docstring for the
mechanism (a redundant local target-model extend pass, forcing
forward_mode=EXTEND to reuse DFlashWorkerV2.forward_batch_generation's
existing extend-branch logic).

Reuses make_draft_input_v2 -- the exact function Scheduler's normal overlap
decode path calls after every real step (scheduler.py: `batch.spec_info =
batch_result.next_draft_input`) to build the *next* tick's draft input --
rather than hand-constructing an equivalent DFlashDraftInputV2. That
normal-path object always has nxt_kv_lens_cpu unset (None): it's
transient, single-tick state that ScheduleBatch.prepare_for_decode()
recomputes from scratch every tick (at the end of update_running_batch,
called *after* filter_batch/merge_batch have already run against whatever
nxt_kv_lens_cpu was on the batch coming in). An earlier version of
this file called spec_info.prepare_for_decode(batch) here to give this
sub-batch a "real" nxt_kv_lens_cpu before merging -- that was wrong,
not just unnecessary: it planted a small, batch-local reservation into a
field that's supposed to stay None until the *merged* batch's own
prepare_for_decode() call recomputes it fresh over every request. Confirmed
live: DFlashDraftInputV2.merge_batch's `elif spec_info.nxt_kv_lens_cpu
is not None: self.nxt_kv_lens_cpu = spec_info.nxt_kv_lens_cpu`
took that small sub-batch reservation as if it represented the *entire*
merged batch, and the next filter_batch call crashed indexing into it
("IndexError: index N is out of bounds for dimension 0 with size 1").

`future_indices` is different -- unlike nxt_kv_lens_cpu, the normal
path explicitly restores it right after the same batch.spec_info reset
(`batch.spec_info.future_indices = future_indices`, scheduler.py), so it
must also be set here, exactly as eagle's own disagg builder does.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    spec_info = make_draft_input_v2(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens,
    )
    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(
            spec_info.future_indices,
            batch.seq_lens,
            future_indices_cpu=batch.req_pool_indices_cpu,
        )
        future_map.stash(
            spec_info.future_indices,
            RelayPayload(bonus_tokens=last_tokens_tensor),
            future_indices_cpu=batch.req_pool_indices_cpu,
        )
    return spec_info
