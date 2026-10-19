"""Unit tests for srt/oracle_decode/forced_token_pool.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")

from types import SimpleNamespace

import torch

from sglang.srt.oracle_decode.forced_token_pool import (
    apply_forced_tokens,
    build_forced_token_lookup,
    compute_forced_block,
    compute_oracle_commit_fixed_len,
    compute_oracle_commit_via_draft_logprob,
    gather_oracle_window,
    resolve_ccd_forced_lookup,
    stash_ccd_forced_lookup,
)
from sglang.test.test_utils import CustomTestCase


def _make_req(origin_len: int, forced_token_ids=None, rid="r0"):
    """A minimal stand-in for Req: only origin_input_ids, rid and
    sampling_params.forced_token_ids are read by forced_token_pool.py."""
    return SimpleNamespace(
        origin_input_ids=[0] * origin_len,
        rid=rid,
        sampling_params=SimpleNamespace(forced_token_ids=forced_token_ids),
    )


class TestBuildForcedTokenLookup(CustomTestCase):

    def test_no_request_forced_returns_none(self):
        """No row has forced_token_ids set -> None, so callers skip all further work."""
        reqs = [_make_req(3), _make_req(4)]
        self.assertIsNone(build_forced_token_lookup(reqs, torch.device("cpu")))

    def test_mixed_batch_shapes(self):
        """One forced row, one plain row -> lookup covers both, right-padded to max_len."""
        reqs = [_make_req(5, [11, 12, 13]), _make_req(7)]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        self.assertIsNotNone(lookup)
        self.assertEqual(tuple(lookup.token_ids.shape), (2, 3))
        self.assertEqual(lookup.valid_len.tolist(), [3, 0])
        self.assertEqual(lookup.start_offset.tolist(), [5, 0])
        self.assertEqual(lookup.token_ids[0].tolist(), [11, 12, 13])


class TestApplyForcedTokens(CustomTestCase):

    def test_forced_row_overrides_sampled_token_at_each_position(self):
        """Row 0 is forced ([11, 12, 13] starting after a 5-token prompt); row 1 is
        plain decode and must always keep whatever the real sampler produced."""
        reqs = [_make_req(5, [11, 12, 13]), _make_req(7)]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        sampled = torch.tensor([999, 777])

        cases = [
            (4, 11),  # consumed pos 4 -> produced pos 5 -> forced[0]
            (5, 12),  # produced pos 6 -> forced[1]
            (6, 13),  # produced pos 7 -> forced[2]
            (7, 999),  # forced sequence exhausted -> falls back to the real sample
        ]
        for consumed_pos, expected_row0 in cases:
            positions = torch.tensor([consumed_pos, consumed_pos])
            out = apply_forced_tokens(sampled.clone(), positions, lookup)
            self.assertEqual(
                out[0].item(), expected_row0, msg=f"consumed_pos={consumed_pos}"
            )
            self.assertEqual(out[1].item(), 777, msg="plain row must be untouched")

    def test_none_lookup_is_pass_through(self):
        """apply_forced_tokens must be a strict no-op when lookup is None."""
        sampled = torch.tensor([999, 777])
        out = apply_forced_tokens(sampled.clone(), torch.tensor([1, 1]), None)
        self.assertTrue(torch.equal(out, sampled))

    def test_forced_token_before_start_offset_not_applied(self):
        """A position before the forced sequence's start (still consuming the
        prompt) must not be overridden, even though a lookup exists for the row."""
        reqs = [_make_req(5, [11, 12, 13])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        sampled = torch.tensor([999])
        out = apply_forced_tokens(sampled.clone(), torch.tensor([2]), lookup)
        self.assertEqual(out[0].item(), 999)


class TestComputeForcedBlock(CustomTestCase):
    """DFlash block-granularity analog of TestApplyForcedTokens above."""

    def test_full_window_and_advancing_ticks(self):
        # Row 0: prompt len 5, forced [11..16] (6 tokens). Row 1: no forcing.
        reqs = [_make_req(5, [11, 12, 13, 14, 15, 16]), _make_req(7)]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        block_size = 4

        # Tick 1: nothing generated yet for either request.
        forced_tokens, commit_lens, mask = compute_forced_block(
            torch.tensor([5, 7]), block_size, lookup
        )
        self.assertEqual(forced_tokens[0].tolist(), [11, 12, 13, 14])
        self.assertEqual(commit_lens.tolist(), [4, 0])
        self.assertEqual(mask.tolist(), [True, False])

        # Tick 2: row 0 already committed 4 tokens -> only 2 remain.
        forced_tokens2, commit_lens2, mask2 = compute_forced_block(
            torch.tensor([9, 8]), block_size, lookup
        )
        self.assertEqual(forced_tokens2[0][:2].tolist(), [15, 16])
        self.assertEqual(commit_lens2[0].item(), 2)
        self.assertTrue(mask2[0].item())

        # Tick 3: row 0's forced sequence is now exhausted -> falls back to
        # real generation (mask False), same fallback as apply_forced_tokens.
        _, commit_lens3, mask3 = compute_forced_block(
            torch.tensor([11, 8]), block_size, lookup
        )
        self.assertEqual(commit_lens3.tolist(), [0, 0])
        self.assertEqual(mask3.tolist(), [False, False])

    def test_none_lookup_is_pass_through(self):
        self.assertIsNone(compute_forced_block(torch.tensor([5]), 4, None))


class TestGatherOracleWindow(CustomTestCase):
    """Shared window-gathering helper used by both compute_forced_block and
    compute_oracle_commit_via_draft_logprob."""

    def test_window_and_in_range_mask(self):
        reqs = [_make_req(5, [11, 12, 13]), _make_req(7)]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        tokens, in_range = gather_oracle_window(torch.tensor([5, 7]), 4, lookup)
        self.assertEqual(
            tokens[0].tolist(), [11, 12, 13, 13]
        )  # padding beyond valid_len
        self.assertEqual(in_range[0].tolist(), [True, True, True, False])
        self.assertEqual(in_range[1].tolist(), [False, False, False, False])


class TestComputeOracleCommitViaDraftLogprob(CustomTestCase):
    """Draft-confidence-gated commit length: the realistic replacement for
    compute_forced_block's unconditional full-window force, since oracle
    tokens usually come from a different model than the one being served."""

    def test_commit_len_stops_at_first_low_confidence_position(self):
        # Row 0: prompt len 5, forced [11..15]. Row 1: no forcing.
        reqs = [_make_req(5, [11, 12, 13, 14, 15]), _make_req(7)]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        threshold = -2.0
        # Row 0 confident (>= threshold) for the first 2 positions, then drops.
        draft_log_probs = torch.tensor(
            [[-0.1, -0.5, -5.0, -6.0], [-0.1, -0.1, -0.1, -0.1]]
        )
        oracle_tokens, commit_lens, mask = compute_oracle_commit_via_draft_logprob(
            torch.tensor([5, 7]), draft_log_probs, threshold, lookup
        )
        self.assertEqual(oracle_tokens[0].tolist(), [11, 12, 13, 14])
        self.assertEqual(commit_lens.tolist(), [2, 0])
        self.assertEqual(mask.tolist(), [True, False])

    def test_commit_len_floors_at_one_even_with_zero_confidence(self):
        """Real speculative decode always commits at least the bonus token;
        oracle mode must too, or a fully-unconfident tick makes no progress."""
        reqs = [_make_req(5, [11, 12, 13, 14, 15])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        draft_log_probs = torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
        _, commit_lens, mask = compute_oracle_commit_via_draft_logprob(
            torch.tensor([5]), draft_log_probs, -2.0, lookup
        )
        self.assertEqual(commit_lens.tolist(), [1])
        self.assertTrue(mask[0].item())

    def test_exhausted_row_does_not_floor_to_one(self):
        """Once the oracle sequence itself is exhausted (not just low
        confidence), there's nothing left to force -- must not floor to 1."""
        reqs = [_make_req(5, [11, 12])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        draft_log_probs = torch.tensor([[-0.1, -0.1, -0.1, -0.1]])
        _, commit_lens, mask = compute_oracle_commit_via_draft_logprob(
            torch.tensor([7]), draft_log_probs, -2.0, lookup
        )  # prefix_lens=7 is already past start_offset(5)+valid_len(2)=7
        self.assertEqual(commit_lens.tolist(), [0])
        self.assertFalse(mask[0].item())

    def test_none_lookup_is_pass_through(self):
        self.assertIsNone(
            compute_oracle_commit_via_draft_logprob(
                torch.tensor([5]), torch.zeros(1, 4), -2.0, None
            )
        )


class TestComputeOracleCommitFixedLen(CustomTestCase):
    """Fixed-length commit: the SGLANG_ORACLE_FIXED_COMMIT_LEN mode that
    replaces the draft-confidence gate with a constant commit length, so
    oracle benchmarks exercise realistic accept-length dynamics instead of
    the gate's near-permanent floor of 1 on foreign-model traces."""

    def test_commits_exactly_fixed_len_when_enough_tokens_remain(self):
        reqs = [_make_req(5, [11, 12, 13, 14, 15]), _make_req(7)]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        oracle_tokens, commit_lens, mask = compute_oracle_commit_fixed_len(
            torch.tensor([5, 7]), 4, 3, lookup
        )
        self.assertEqual(oracle_tokens[0].tolist(), [11, 12, 13, 14])
        self.assertEqual(commit_lens.tolist(), [3, 0])
        self.assertEqual(mask.tolist(), [True, False])

    def test_caps_at_remaining_forced_tokens(self):
        """Near the end of the forced sequence there may be fewer tokens
        left than the fixed length -- commit only what exists."""
        reqs = [_make_req(5, [11, 12])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        _, commit_lens, mask = compute_oracle_commit_fixed_len(
            torch.tensor([5]), 4, 3, lookup
        )
        self.assertEqual(commit_lens.tolist(), [2])
        self.assertTrue(mask[0].item())

    def test_caps_at_block_window(self):
        """fixed_len larger than the block window can never commit more
        than the window holds."""
        reqs = [_make_req(5, [11, 12, 13, 14, 15, 16, 17, 18])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        _, commit_lens, _ = compute_oracle_commit_fixed_len(
            torch.tensor([5]), 4, 99, lookup
        )
        self.assertEqual(commit_lens.tolist(), [4])

    def test_exhausted_row_commits_zero_not_one(self):
        """Same contract as the gated variant: past the forced sequence's
        end there is nothing to force -- no floor-to-1."""
        reqs = [_make_req(5, [11, 12])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        _, commit_lens, mask = compute_oracle_commit_fixed_len(
            torch.tensor([7]), 4, 3, lookup
        )
        self.assertEqual(commit_lens.tolist(), [0])
        self.assertFalse(mask[0].item())

    def test_mid_sequence_window_offsets_correctly(self):
        """prefix_lens past start_offset indexes the forced sequence at the
        right position, same as gather_oracle_window's own contract."""
        reqs = [_make_req(5, [11, 12, 13, 14, 15])]
        lookup = build_forced_token_lookup(reqs, torch.device("cpu"))
        oracle_tokens, commit_lens, _ = compute_oracle_commit_fixed_len(
            torch.tensor([7]), 4, 2, lookup
        )
        self.assertEqual(oracle_tokens[0].tolist()[:3], [13, 14, 15])
        self.assertEqual(commit_lens.tolist(), [2])

    def test_none_lookup_returns_none(self):
        self.assertIsNone(
            compute_oracle_commit_fixed_len(torch.tensor([5]), 4, 3, None)
        )


class TestCcdForcedLookupStashResolve(CustomTestCase):
    """The CCD stash (`_ccd_forced_lookup`) is a dynamic attribute on the
    live ScheduleBatch: filter_batch/merge_batch/copy.copy all propagate it
    untouched, so a lookup stashed before a batch-composition change points
    its positional rows at the wrong requests. resolve_ccd_forced_lookup
    must detect that via the rid fingerprint and rebuild."""

    def _batch(self, reqs):
        return SimpleNamespace(reqs=reqs)

    def test_resolve_reuses_stash_when_request_list_unchanged(self):
        reqs = [_make_req(5, [11, 12], rid="a"), _make_req(7, rid="b")]
        batch = self._batch(reqs)
        stash_ccd_forced_lookup(batch, torch.device("cpu"))
        resolved = resolve_ccd_forced_lookup(batch, torch.device("cpu"))
        self.assertIs(resolved, batch._ccd_forced_lookup)

    def test_resolve_rebuilds_when_rows_filtered(self):
        """Request 'a' finished and was filtered out: the stale bs=2 stash
        must NOT be reused for the bs=1 batch (row 0 would replay a's
        forced trace into b's row)."""
        req_a = _make_req(5, [11, 12], rid="a")
        req_b = _make_req(7, [31, 32], rid="b")
        batch = self._batch([req_a, req_b])
        stash_ccd_forced_lookup(batch, torch.device("cpu"))
        batch.reqs = [req_b]  # what ScheduleBatch.filter_batch leaves behind
        resolved = resolve_ccd_forced_lookup(batch, torch.device("cpu"))
        self.assertIsNot(resolved, batch._ccd_forced_lookup)
        self.assertEqual(tuple(resolved.token_ids.shape), (1, 2))
        self.assertEqual(resolved.token_ids[0].tolist(), [31, 32])

    def test_resolve_rebuilds_when_same_size_but_different_requests(self):
        """The dangerous silent case: same bs after one finishes and one
        joins -- shapes match, rows don't. The fingerprint must catch it."""
        req_a = _make_req(5, [11, 12], rid="a")
        req_b = _make_req(7, [31, 32], rid="b")
        req_c = _make_req(9, [51, 52], rid="c")
        batch = self._batch([req_a, req_b])
        stash_ccd_forced_lookup(batch, torch.device("cpu"))
        batch.reqs = [req_b, req_c]  # a finished, c admitted, bs still 2
        resolved = resolve_ccd_forced_lookup(batch, torch.device("cpu"))
        self.assertIsNot(resolved, batch._ccd_forced_lookup)
        self.assertEqual(resolved.token_ids[0].tolist(), [31, 32])
        self.assertEqual(resolved.token_ids[1].tolist(), [51, 52])

    def test_resolve_without_stash_builds_fresh(self):
        reqs = [_make_req(5, [11, 12], rid="a")]
        batch = self._batch(reqs)
        resolved = resolve_ccd_forced_lookup(batch, torch.device("cpu"))
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.token_ids[0].tolist(), [11, 12])

    def test_resolve_without_stash_and_no_forcing_returns_none(self):
        batch = self._batch([_make_req(5, rid="a")])
        self.assertIsNone(resolve_ccd_forced_lookup(batch, torch.device("cpu")))
