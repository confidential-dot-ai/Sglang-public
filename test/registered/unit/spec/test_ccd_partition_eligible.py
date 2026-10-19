"""Unit tests for partition_ccd_eligible -- no server, no model loading, no GPU.

Covers the case check_ccd_eligible's whole-batch gate handles poorly: a
decode batch under sustained multi-turn traffic is a continuous mix of
long-running "veteran" requests (individually CCD-eligible) and freshly-
admitted "newcomer" ones (never eligible, since every request starts at 0
output tokens) -- that mix is the steady state, not an edge case, so
check_ccd_eligible alone means CCD effectively never fires. These tests
verify partition_ccd_eligible correctly separates the two instead of
letting one newcomer poison the whole batch.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.constrained.device_token_fsm import DeviceFSMGrammar, DeviceTokenFSM
from sglang.srt.speculative.dflash_cc_conditional_decode import (
    SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
    check_ccd_eligible,
    partition_ccd_eligible,
)
from sglang.test.test_utils import CustomTestCase

VETERAN_LEN = SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN + 1
NEWCOMER_LEN = 1

# Sentinel so _make_batch can tell "caller passed None on purpose" from
# "caller didn't specify" -- None is a meaningful value for seq_lens_cpu.
_UNSET = object()


def _make_req(
    *,
    output_len=VETERAN_LEN,
    top_k=1,
    return_logprob=False,
    return_hidden_states=False,
    return_sampling_mask=False,
    grammar=None,
    forced_token_ids=None,
):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            top_k=top_k, forced_token_ids=forced_token_ids
        ),
        return_logprob=return_logprob,
        return_hidden_states=return_hidden_states,
        return_sampling_mask=return_sampling_mask,
        output_ids=[0] * output_len,
        grammar=grammar,
    )


def _make_batch(
    reqs,
    *,
    forward_mode_is_extend=False,
    is_extend_in_batch=False,
    seq_lens_cpu=_UNSET,
):
    # seq_lens_cpu defaults to a real host mirror: ScheduleBatch always
    # carries the attribute, and the compact-draft-cache CCD path derives
    # its exact per-request lengths from it (see
    # DFlashWorkerV2._compute_compact_draft_seq_lens_cpu). Pass None
    # explicitly to exercise the gate that keeps that path off a batch with
    # no host mirror.
    if seq_lens_cpu is _UNSET:
        seq_lens_cpu = torch.tensor([128] * len(reqs), dtype=torch.int32)
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: forward_mode_is_extend),
        is_extend_in_batch=is_extend_in_batch,
        sampling_info=SimpleNamespace(
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs)
        ),
        return_logprob=any(r.return_logprob for r in reqs),
        has_grammar=any(r.grammar for r in reqs),
        spec_info=SimpleNamespace(reserved_seq_lens_cpu=None),
        seq_lens_cpu=seq_lens_cpu,
        reqs=reqs,
    )


def _make_worker():
    return SimpleNamespace(
        use_compact_draft_cache=False,
        _need_mamba_verify_commit=False,
        draft_window_size=None,
        ccd_lm_head_requires_collective=lambda: False,
    )


def _make_device_fsm_grammar():
    fsm = DeviceTokenFSM.compile(
        transitions={0: {1: 0}},
        start_state=0,
        final_states={0},
        vocab_size=8,
        eos_token_ids={7},
    )
    return DeviceFSMGrammar(fsm)


class TestPartitionCcdEligible(CustomTestCase):
    def test_batch_level_gate_excludes_everyone(self):
        """forward_mode/worker/spec_info failures can't vary per-request --
        batch_level_reason set, eligible_indices always empty."""
        batch = _make_batch([_make_req(), _make_req()], forward_mode_is_extend=True)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNotNone(reason)
        self.assertEqual(eligible, [])

    def test_all_veterans_all_eligible(self):
        batch = _make_batch([_make_req(), _make_req(), _make_req()])
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 1, 2])
        # Must agree with the whole-batch gate on the all-eligible case.
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))

    def test_veteran_newcomer_mix_partitions_correctly(self):
        """The steady-state case: check_ccd_eligible fails the whole batch,
        but partition_ccd_eligible correctly isolates the veterans."""
        reqs = [
            _make_req(output_len=VETERAN_LEN),  # veteran
            _make_req(output_len=NEWCOMER_LEN),  # newcomer, just transferred in
            _make_req(output_len=VETERAN_LEN),  # veteran
            _make_req(output_len=NEWCOMER_LEN),  # newcomer
        ]
        batch = _make_batch(reqs)
        worker = _make_worker()

        # Sanity: this is exactly the shape check_ccd_eligible rejects
        # wholesale -- otherwise this test would not be exercising the case
        # partition_ccd_eligible exists to fix.
        whole_batch_reason = check_ccd_eligible(batch, worker)
        self.assertIsNotNone(whole_batch_reason)

        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 2])

    def test_non_greedy_request_excluded_but_does_not_block_others(self):
        reqs = [_make_req(top_k=1), _make_req(top_k=5), _make_req(top_k=1)]
        batch = _make_batch(reqs)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 2])

    def test_return_logprob_request_excluded_but_does_not_block_others(self):
        reqs = [
            _make_req(return_logprob=True),
            _make_req(return_logprob=False),
        ]
        batch = _make_batch(reqs)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [1])

    def test_return_hidden_states_and_sampling_mask_excluded(self):
        reqs = [
            _make_req(return_hidden_states=True),
            _make_req(return_sampling_mask=True),
            _make_req(),
        ]
        batch = _make_batch(reqs)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [2])

    def test_host_grammar_excluded_device_fsm_grammar_included(self):
        host_grammar = object()
        reqs = [
            _make_req(grammar=host_grammar),
            _make_req(grammar=_make_device_fsm_grammar()),
            _make_req(grammar=None),
        ]
        batch = _make_batch(reqs)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [1, 2])

    def test_worker_level_gate_excludes_everyone(self):
        batch = _make_batch([_make_req(), _make_req()])
        worker = SimpleNamespace(
            use_compact_draft_cache=True,
            _need_mamba_verify_commit=False,
            draft_window_size=None,
        )
        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertIsNotNone(reason)
        self.assertEqual(eligible, [])

    def test_compact_draft_cache_excluded_when_window_exceeds_min_output_len(self):
        """draft_window_size > the min-output-len floor breaks the invariant
        that makes the compact branch's padded gather/assign correctly
        sized (see check_ccd_eligible's own comment) -- must still exclude
        in that case."""
        batch = _make_batch([_make_req(), _make_req()])
        worker = SimpleNamespace(
            use_compact_draft_cache=True,
            _need_mamba_verify_commit=False,
            draft_window_size=SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN + 1,
        )
        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertIsNotNone(reason)
        self.assertEqual(eligible, [])
        self.assertIsNotNone(check_ccd_eligible(batch, worker))

    def test_compact_draft_cache_allowed_when_window_within_min_output_len(self):
        """The fixed case: draft_window_size <= the min-output-len floor
        means every CCD-eligible request has already saturated the compact
        window's clamp, so the padded gather/assign's static bound is
        always correctly sized -- compact draft cache must NOT be excluded
        here."""
        batch = _make_batch([_make_req(), _make_req()])
        worker = SimpleNamespace(
            use_compact_draft_cache=True,
            _need_mamba_verify_commit=False,
            draft_window_size=SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
            ccd_lm_head_requires_collective=lambda: False,
        )
        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 1])
        self.assertIsNone(check_ccd_eligible(batch, worker))

    def test_compact_draft_cache_excluded_without_host_seq_lens(self):
        """Compact draft cache derives its exact seq_lens_cpu from
        batch.seq_lens_cpu (a static upper bound there corrupts flashinfer's
        KV layout -> IMA, tdx-checkpoints/134). With no host mirror the only
        exact source would be a D2H sync, which capture forbids -- so the
        batch must degrade to eager rather than reach that code."""
        batch = _make_batch([_make_req(), _make_req()], seq_lens_cpu=None)
        worker = SimpleNamespace(
            use_compact_draft_cache=True,
            _need_mamba_verify_commit=False,
            draft_window_size=SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
            ccd_lm_head_requires_collective=lambda: False,
        )
        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertIsNotNone(reason)
        self.assertIn("seq_lens_cpu", reason)
        self.assertEqual(eligible, [])
        self.assertIsNotNone(check_ccd_eligible(batch, worker))

    def test_non_compact_cache_unaffected_by_missing_host_seq_lens(self):
        """The gate is scoped to the compact path -- the non-windowed CCD
        branch uses reserved_seq_lens_cpu and must not be restricted by it."""
        batch = _make_batch([_make_req(), _make_req()], seq_lens_cpu=None)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 1])

    def test_no_eligible_requests_returns_empty_list_not_a_reason(self):
        """All-newcomer batch: no batch-level gate fails, but nothing
        qualifies individually -- callers must see this as 'no CCD this
        tick', not mistake the empty list for a batch-level failure."""
        reqs = [_make_req(output_len=NEWCOMER_LEN), _make_req(output_len=NEWCOMER_LEN)]
        batch = _make_batch(reqs)
        reason, eligible = partition_ccd_eligible(batch, _make_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [])
