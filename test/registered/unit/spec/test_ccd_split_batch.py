"""Unit tests for try_split_ccd_batch / merge_ccd_split_results -- no server,
no model loading, no GPU.

The core regression this file guards against: ScheduleBatch.copy() is an
intentionally partial, output-side-only snapshot ("Only contain fields that
will be used by process_batch_result") and is unsuitable for building a
second, independently-forward-able sub-batch. try_split_ccd_batch instead
uses copy.copy() (a real, if shallow, clone) plus an explicit independent
copy.copy() of spec_info/sampling_info -- without that second step, two
copy.copy()-derived sub-batches would still alias the same spec_info Python
object, and filtering the second sub-batch would silently clobber state the
first one already filtered. test_split_gives_each_side_independent_spec_info
below is the direct regression test for that specific bug.
"""

import dataclasses
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.dflash_cc_conditional_decode import (  # noqa: E402
    SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
    merge_ccd_split_results,
    try_split_ccd_batch,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

VETERAN_LEN = SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN + 1
NEWCOMER_LEN = 1

AUTO_FILL_EXCLUDED_FIELDS = ["reqs"]


def make_schedule_batch(bs: int, **overrides) -> ScheduleBatch:
    """Same pattern as test_schedule_batch_out_of_place.py's helper: build a
    minimal ScheduleBatch with every Tensor/List field auto-filled to a
    bs-shaped placeholder, then apply explicit overrides."""
    batch = ScheduleBatch(reqs=overrides.pop("reqs"))
    for field in dataclasses.fields(ScheduleBatch):
        name = field.name
        if name in overrides or name in AUTO_FILL_EXCLUDED_FIELDS:
            continue
        annotation = str(field.type)
        if "List" in annotation or "list[" in annotation:
            setattr(batch, name, [f"{name}-{i}" for i in range(bs)])
        elif "Tensor" in annotation:
            setattr(batch, name, torch.arange(bs, dtype=torch.int64))
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


def _make_spec_info(bs: int, future_indices=None) -> DFlashDraftInputV2:
    return DFlashDraftInputV2(
        topk_p=torch.zeros(bs, 1),
        topk_index=torch.zeros(bs, 1, dtype=torch.int64),
        bonus_tokens=torch.arange(bs, dtype=torch.int64) + 100,
        new_seq_lens=torch.arange(bs, dtype=torch.int64) + 200,
        hidden_states=torch.zeros(bs, 1),
        future_indices=future_indices,
    )


def _make_req(*, output_len=VETERAN_LEN, return_logprob=False, top_k=1):
    return types.SimpleNamespace(
        sampling_params=types.SimpleNamespace(
            top_k=top_k, max_new_tokens=1000, forced_token_ids=None
        ),
        return_logprob=return_logprob,
        return_hidden_states=False,
        return_sampling_mask=False,
        output_ids=[0] * output_len,
        grammar=None,
    )


def _permissive_sampling_info():
    """A MagicMock stands in for SamplingBatchInfo: filter_batch/merge_batch
    calls are recorded no-ops (sufficient for these structural tests).

    penalizer_orchestrator is pinned to None because that is what REAL
    decode batches carry when no request has penalty sampling params -- the
    common case, and the one production actually hit. The original mock
    supplied an orchestrator object here, which hid an unconditional
    .is_required dereference until the split path's first-ever live
    execution crashed on it (mock-fidelity gap; see
    test_returns_none_when_penalizer_required for the non-None case)."""
    sampling_info = MagicMock()
    sampling_info.penalizer_orchestrator = None
    # Both penalty buffers are None on real copy_for_forward() copies unless
    # penalties are active -- and an unpinned MagicMock attribute is truthy,
    # which would trip the corresponding split guard on every test.
    # acc_scaling_penalties = multiplicative (repetition_penalty);
    # acc_additive_penalties = additive (frequency/presence/min_new_tokens),
    # built unconditionally whenever the orchestrator is_required, so it is
    # the buffer that's non-None for additive-only penalty batches.
    sampling_info.acc_scaling_penalties = None
    sampling_info.acc_additive_penalties = None
    sampling_info.is_all_greedy = True
    return sampling_info


def _make_worker():
    return types.SimpleNamespace(
        use_compact_draft_cache=False,
        _need_mamba_verify_commit=False,
        ccd_lm_head_requires_collective=lambda: False,
    )


def _make_decode_batch(reqs, spec_info, *, req_pool_indices=None):
    bs = len(reqs)
    return make_schedule_batch(
        bs,
        reqs=reqs,
        device="cpu",
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        sampling_info=_permissive_sampling_info(),
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        return_logprob=False,
        has_grammar=False,
        spec_info=spec_info,
        spec_algorithm=types.SimpleNamespace(
            is_dflash_family=lambda: True, is_none=lambda: False
        ),
        req_pool_indices=(
            req_pool_indices
            if req_pool_indices is not None
            else torch.arange(bs, dtype=torch.int64)
        ),
        req_pool_indices_cpu=torch.arange(bs, dtype=torch.int64),
        mamba_track_indices=None,
        dp_cooperation_info=None,
        hisparse_coordinator=None,
    )


class TestTrySplitCcdBatch(CustomTestCase):
    def test_returns_none_when_partition_batch_level_reason_set(self):
        reqs = [_make_req(), _make_req()]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        batch.forward_mode = ForwardMode.EXTEND
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_returns_none_when_all_eligible(self):
        reqs = [_make_req(), _make_req()]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_returns_none_when_all_ineligible(self):
        reqs = [_make_req(output_len=NEWCOMER_LEN), _make_req(output_len=NEWCOMER_LEN)]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_returns_none_when_penalizer_required(self):
        reqs = [_make_req(output_len=VETERAN_LEN), _make_req(output_len=NEWCOMER_LEN)]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        batch.sampling_info.penalizer_orchestrator = types.SimpleNamespace(
            is_required=True
        )
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_returns_none_when_scaling_penalties_active(self):
        """copy_for_forward() with an active repetition_penalty leaves
        acc_scaling_penalties as a [bs, vocab] tensor (orchestrator=None) --
        filter_batch does not row-filter it, so splitting must bail."""
        reqs = [_make_req(output_len=VETERAN_LEN), _make_req(output_len=NEWCOMER_LEN)]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        batch.sampling_info.acc_scaling_penalties = torch.ones(2, 8)
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_returns_none_when_additive_penalties_active(self):
        """The additive-only case: frequency/presence/min_new_tokens
        penalties pre-accumulate into acc_additive_penalties while
        acc_scaling_penalties stays None (repetition_penalty is the only
        multiplicative penalizer) -- the guard must catch this buffer too,
        for exactly the same not-row-filtered reason."""
        reqs = [_make_req(output_len=VETERAN_LEN), _make_req(output_len=NEWCOMER_LEN)]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        batch.sampling_info.acc_additive_penalties = torch.zeros(2, 8)
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_returns_none_when_ineligible_req_wants_logprob(self):
        reqs = [
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=NEWCOMER_LEN, return_logprob=True),
        ]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        self.assertIsNone(try_split_ccd_batch(batch, _make_worker()))

    def test_split_produces_correctly_sized_sub_batches(self):
        reqs = [
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=NEWCOMER_LEN),
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=NEWCOMER_LEN),
        ]
        batch = _make_decode_batch(reqs, _make_spec_info(4))
        split = try_split_ccd_batch(batch, _make_worker())
        self.assertIsNotNone(split)
        eligible_batch, ineligible_batch, eligible_indices, ineligible_indices = split
        self.assertEqual(eligible_batch.batch_size(), 2)
        self.assertEqual(ineligible_batch.batch_size(), 2)
        self.assertEqual(eligible_indices, [0, 2])
        self.assertEqual(ineligible_indices, [1, 3])
        self.assertTrue(
            torch.equal(eligible_batch.req_pool_indices, torch.tensor([0, 2]))
        )
        self.assertTrue(
            torch.equal(ineligible_batch.req_pool_indices, torch.tensor([1, 3]))
        )

    def test_split_gives_each_side_independent_spec_info(self):
        """Regression test: ScheduleBatch.copy() (shared spec_info reference)
        would let filtering the second sub-batch silently corrupt the
        first's already-filtered state. try_split_ccd_batch must hand each
        side its own independent DFlashDraftInputV2 object."""
        reqs = [
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=NEWCOMER_LEN),
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=NEWCOMER_LEN),
        ]
        # future_indices=None so DFlashDraftInputV2.filter_batch takes its
        # full-reslice branch (topk_p/bonus_tokens/etc), not the
        # future_indices-early-return branch -- exercises the aliasing bug
        # at its widest surface.
        batch = _make_decode_batch(reqs, _make_spec_info(4, future_indices=None))
        eligible_batch, ineligible_batch, _, _ = try_split_ccd_batch(
            batch, _make_worker()
        )

        self.assertIsNot(eligible_batch.spec_info, ineligible_batch.spec_info)
        self.assertEqual(eligible_batch.spec_info.bonus_tokens.shape[0], 2)
        self.assertEqual(ineligible_batch.spec_info.bonus_tokens.shape[0], 2)
        self.assertTrue(
            torch.equal(eligible_batch.spec_info.bonus_tokens, torch.tensor([100, 102]))
        )
        self.assertTrue(
            torch.equal(
                ineligible_batch.spec_info.bonus_tokens, torch.tensor([101, 103])
            )
        )

        # Mutating one side's spec_info must not affect the other's --
        # the concrete symptom the aliasing bug would produce.
        eligible_batch.spec_info.bonus_tokens.fill_(999)
        self.assertTrue(
            torch.equal(
                ineligible_batch.spec_info.bonus_tokens, torch.tensor([101, 103])
            )
        )

    def test_split_gives_each_side_independent_sampling_info(self):
        reqs = [
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=NEWCOMER_LEN),
        ]
        batch = _make_decode_batch(reqs, _make_spec_info(2))
        eligible_batch, ineligible_batch, _, _ = try_split_ccd_batch(
            batch, _make_worker()
        )
        self.assertIsNot(eligible_batch.sampling_info, ineligible_batch.sampling_info)

    def test_eligible_sub_batch_is_marked_all_greedy(self):
        """SamplingBatchInfo.filter_batch never recomputes is_all_greedy, so
        a mixed-greedy batch (is_all_greedy=False) would leave the filtered
        eligible sub-batch carrying a stale False -- and run_ccd_chunk's
        check_ccd_eligible re-check RAISES on that, for exactly the mixed
        batches the split exists to serve. try_split_ccd_batch must set it
        True by construction (partition admits only greedy rows)."""
        reqs = [
            _make_req(output_len=VETERAN_LEN),
            _make_req(output_len=VETERAN_LEN, top_k=4),  # sampled -> ineligible
            _make_req(output_len=NEWCOMER_LEN),
        ]
        batch = _make_decode_batch(reqs, _make_spec_info(3))
        batch.sampling_info.is_all_greedy = False  # mixed batch
        split = try_split_ccd_batch(batch, _make_worker())
        self.assertIsNotNone(split)
        eligible_batch, _, eligible_indices, ineligible_indices = split
        self.assertEqual(eligible_indices, [0])
        self.assertEqual(ineligible_indices, [1, 2])
        self.assertIs(eligible_batch.sampling_info.is_all_greedy, True)


class TestMergeCcdSplitResults(CustomTestCase):
    """merge_ccd_split_results must scatter both halves' rows back to their
    ORIGINAL batch positions (per eligible_indices/ineligible_indices), not
    concatenate eligible-then-ineligible: the caller's ScheduleBatch keeps
    its original row order (run_batch's _forward_isolation restore reverts
    any in-forward reorder), and process_batch_result maps result rows to
    batch.reqs positionally."""

    def _make_result(
        self, next_token_ids, accept_lens, bonus_tokens, new_seq_lens, draft_tokens=2
    ):
        return types.SimpleNamespace(
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            can_run_cuda_graph=True,
            speculative_num_draft_tokens=draft_tokens,
            next_draft_input=types.SimpleNamespace(
                bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens
            ),
        )

    def test_scatters_rows_back_to_original_positions(self):
        """Interleaved split (eligible rows 0 and 2, ineligible rows 1 and
        3): every merged tensor's row i must be original row i's data."""
        block = 2
        eligible_result = self._make_result(
            # rows for original positions [0, 2], block=2
            next_token_ids=torch.tensor([10, 11, 30, 31]),
            accept_lens=torch.tensor([2, 1], dtype=torch.int32),
            bonus_tokens=torch.tensor([100, 300]),
            new_seq_lens=torch.tensor([1000, 3000]),
            draft_tokens=block,
        )
        ineligible_result = self._make_result(
            # rows for original positions [1, 3]
            next_token_ids=torch.tensor([20, 21, 40, 41]),
            accept_lens=torch.tensor([1, 2], dtype=torch.int32),
            bonus_tokens=torch.tensor([200, 400]),
            new_seq_lens=torch.tensor([2000, 4000]),
            draft_tokens=block,
        )
        merged = merge_ccd_split_results(
            eligible_result,
            ineligible_result,
            [0, 2],
            [1, 3],
            device="cpu",
        )
        self.assertIsNone(merged.logits_output)
        self.assertTrue(
            torch.equal(
                merged.next_token_ids,
                torch.tensor([10, 11, 20, 21, 30, 31, 40, 41]),
            )
        )
        self.assertTrue(
            torch.equal(merged.accept_lens, torch.tensor([2, 1, 1, 2], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                merged.next_draft_input.bonus_tokens,
                torch.tensor([100, 200, 300, 400]),
            )
        )
        self.assertTrue(
            torch.equal(
                merged.next_draft_input.new_seq_lens,
                torch.tensor([1000, 2000, 3000, 4000]),
            )
        )
        self.assertTrue(
            torch.equal(merged.new_seq_lens, torch.tensor([1000, 2000, 3000, 4000]))
        )
        self.assertEqual(merged.speculative_num_draft_tokens, block)

    def test_merged_dtype_follows_eager_half(self):
        """The eligible terminal result's tensors come off read_history's
        D2H (CPU) and may differ in dtype from the eager half's device
        tensors; the merged result must follow the eager half's dtype --
        that's what every downstream consumer receives on a non-split
        tick."""
        merged = merge_ccd_split_results(
            self._make_result(
                next_token_ids=torch.tensor([1, 2], dtype=torch.int64),
                accept_lens=torch.tensor([1], dtype=torch.int64),
                bonus_tokens=torch.tensor([7], dtype=torch.int64),
                new_seq_lens=torch.tensor([5], dtype=torch.int64),
            ),
            self._make_result(
                next_token_ids=torch.tensor([3, 4], dtype=torch.int64),
                accept_lens=torch.tensor([2], dtype=torch.int32),
                bonus_tokens=torch.tensor([8], dtype=torch.int64),
                new_seq_lens=torch.tensor([6], dtype=torch.int64),
            ),
            [0],
            [1],
            device="cpu",
        )
        self.assertEqual(merged.accept_lens.dtype, torch.int32)
        self.assertTrue(torch.equal(merged.accept_lens, torch.tensor([1, 2], dtype=torch.int32)))

    def test_can_run_cuda_graph_is_and_of_both_sides(self):
        eligible_result = self._make_result(
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([9]),
            torch.tensor([1]),
            draft_tokens=1,
        )
        ineligible_result = self._make_result(
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor([9]),
            torch.tensor([1]),
            draft_tokens=1,
        )
        ineligible_result.can_run_cuda_graph = False
        merged = merge_ccd_split_results(
            eligible_result, ineligible_result, [0], [1], device="cpu"
        )
        self.assertFalse(merged.can_run_cuda_graph)


if __name__ == "__main__":
    unittest.main()
