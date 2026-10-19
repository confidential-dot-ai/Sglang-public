"""Unit tests for the pure-Python helpers behind TODO.md J14c Milestone 1
(SUM_LEN-bucketed CUDA graph capture for DSPARK TARGET_VERIFY under DP
attention) -- CPU-only, no GPU/distributed context required.

Root cause this workstream addresses: decode_cuda_graph_runner.py forces
DpPaddingMode.MAX_LEN for every graph-replayed step, so DP ranks pad their
local batch (and DSPARK's 9x-per-request verify width) up to the cross-rank
max on every layer. These helpers build a bounded (local_bucket, sum_bucket)
capture window so DP-attention DSPARK verify can replay under SUM_LEN
instead. See decode_cuda_graph_runner.py's
dspark_dp_verify_sum_capture_enabled / build_dp_verify_sum_bucket_ladder /
lookup_dp_verify_sum_bucket / synthetic_dp_verify_sum_split.
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
    build_dp_verify_sum_bucket_ladder,
    build_ragged_dp_verify_sum_bucket_ladder,
    dspark_dp_verify_sum_capture_enabled,
    lookup_dp_verify_sum_bucket,
    resolve_dp_verify_sum_graph_buckets,
    resolve_ragged_dp_verify_sum_graph_buckets,
    synthetic_dp_verify_sum_split,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CAPTURE_BS = [1, 2, 4, 8, 12, 16, 24, 32]
# DSPARK's speculative_num_draft_tokens default -- the ladder's keys are
# request-count buckets (self.capture_bs's own units) but its VALUES are
# TOKEN counts (extend_num_tokens = requests * num_tokens_per_req); mixing
# the two was the exact bug the first live boot crashed on (TODO.md J14c).
_NUM_TOKENS_PER_REQ = 9
_RAGGED_CAPTURE_NUM_TOKENS = [
    bucket * _NUM_TOKENS_PER_REQ for bucket in _CAPTURE_BS
]


class TestBuildDpVerifySumBucketLadder(CustomTestCase):
    def test_every_local_bucket_shares_one_ladder(self):
        """The property the whole feature rests on. The sum bucket sizes a
        collective, so lookup must not depend on the caller's own local
        bucket -- ranks hold different request counts, and a per-bucket
        ladder let two ranks in the same step resolve different sums (skew
        (2,3,4,5) gave [None, None, 144, 180], TODO.md J14d)."""
        for window in (0, 1, 3, 7):
            ladder = build_dp_verify_sum_bucket_ladder(
                _CAPTURE_BS,
                dp_size=4,
                window=window,
                num_tokens_per_req=_NUM_TOKENS_PER_REQ,
            )
            distinct = {tuple(sums) for sums in ladder.values()}
            self.assertEqual(
                len(distinct),
                1,
                f"window={window} produced per-bucket ladders: {distinct}",
            )

    def test_ladder_is_independent_of_window(self):
        """window is retained only so existing callers/env keep working.
        Full coverage is the only rank-safe option, so it must not be able to
        narrow the ladder back into the divergent regime."""
        ladders = [
            build_dp_verify_sum_bucket_ladder(
                _CAPTURE_BS,
                dp_size=4,
                window=window,
                num_tokens_per_req=_NUM_TOKENS_PER_REQ,
            )
            for window in (0, 1, 2, 7, 99)
        ]
        for other in ladders[1:]:
            self.assertEqual(ladders[0], other)

    def test_single_total_covers_every_reachable_step(self):
        """One total, and it must be the largest reachable one: every rank is
        capped at max(capture_bs) requests, so no step can exceed
        dp_size * max_bucket * num_tokens_per_req. A smaller entry would send
        busy steps to eager for no reason."""
        ladder = build_dp_verify_sum_bucket_ladder(
            _CAPTURE_BS, dp_size=4, window=0, num_tokens_per_req=_NUM_TOKENS_PER_REQ
        )
        expected = 4 * max(_CAPTURE_BS) * _NUM_TOKENS_PER_REQ
        for sums in ladder.values():
            self.assertEqual(sums, [expected])

    def test_capture_split_never_starves_a_peer_rank(self):
        """Why a range of totals is not an option. All ranks capture the same
        shape, so a total below dp_size * local_tokens can only be expressed by
        handing peers a zero share -- a gathered buffer with room for one rank
        and every rank claiming offset 0. That degenerate shape is what the
        multi-total matrix captured first, and every rank then stalled in
        cuModuleLoad (TODO.md J14d)."""
        for dp_size in (2, 4, 8):
            ladder = build_dp_verify_sum_bucket_ladder(
                _CAPTURE_BS,
                dp_size=dp_size,
                window=0,
                num_tokens_per_req=_NUM_TOKENS_PER_REQ,
            )
            for local_bucket, sums in ladder.items():
                my_tokens = local_bucket * _NUM_TOKENS_PER_REQ
                for total in sums:
                    with self.subTest(dp_size=dp_size, bucket=local_bucket):
                        shares = synthetic_dp_verify_sum_split(
                            my_tokens, total, dp_size, 0
                        )
                        self.assertTrue(
                            all(share > 0 for share in shares),
                            f"zero share in {shares} for bucket {local_bucket}",
                        )

    def test_one_graph_per_local_bucket(self):
        """Capture cost is the other half of the fix: a matrix of totals meant
        a fresh kernel specialization per total, which is where capture hung.
        One total keeps graph count at len(capture_bs), the regime that has
        always captured cleanly."""
        ladder = build_dp_verify_sum_bucket_ladder(
            _CAPTURE_BS, dp_size=4, window=0, num_tokens_per_req=_NUM_TOKENS_PER_REQ
        )
        captured = [
            (bucket, total)
            for bucket, sums in ladder.items()
            for total in sums
            if total >= bucket * _NUM_TOKENS_PER_REQ
        ]
        self.assertEqual(len(captured), len(_CAPTURE_BS))


class TestLookupDpVerifySumBucket(CustomTestCase):
    def test_returns_smallest_covering_bucket(self):
        buckets = {8: [24, 32, 40]}
        self.assertEqual(lookup_dp_verify_sum_bucket(buckets, 8, 24), 24)
        self.assertEqual(lookup_dp_verify_sum_bucket(buckets, 8, 25), 32)
        self.assertEqual(lookup_dp_verify_sum_bucket(buckets, 8, 40), 40)

    def test_miss_above_window_returns_none_not_error(self):
        """Completeness contract: exceeding the captured window must be a
        clean miss (caller falls back to eager), never an exception -- a
        real forward_batch reaching this must not crash the scheduler."""
        buckets = {8: [24, 32, 40]}
        self.assertIsNone(lookup_dp_verify_sum_bucket(buckets, 8, 41))

    def test_unknown_local_bucket_returns_none(self):
        self.assertIsNone(lookup_dp_verify_sum_bucket({8: [32]}, 16, 32))


class TestSyntheticDpVerifySumSplit(CustomTestCase):
    def test_sums_to_total_and_covers_own_slot(self):
        """Canonical-split semantics (TODO.md J14d fifth follow-up): the own
        slot COVERS `my_tokens` (>=) rather than pinning to it -- pinning is
        what made the traced split differ per rank and hang capture."""
        for dp_size in (2, 3, 4, 8):
            for my_rank in range(dp_size):
                for my_tokens, slack in ((4, 0), (3, 1), (0, 9), (2, 7)):
                    total = dp_size * my_tokens + slack
                    split = synthetic_dp_verify_sum_split(
                        my_tokens, total, dp_size, my_rank
                    )
                    with self.subTest(dp_size=dp_size, my_rank=my_rank, total=total):
                        self.assertEqual(len(split), dp_size)
                        self.assertGreaterEqual(split[my_rank], my_tokens)
                        self.assertEqual(sum(split), total)

    def test_split_is_identical_on_every_rank(self):
        """THE load-bearing property (TODO.md J14d fifth follow-up): capture
        hangs iff ranks trace different splits. The list must be a pure
        function of (total, dp_size) -- `my_rank` and `my_tokens` must not
        influence a single element."""
        for dp_size in (2, 4, 8):
            for total in (dp_size * 36, dp_size * 36 + 3, dp_size * 100):
                splits = {
                    tuple(
                        synthetic_dp_verify_sum_split(
                            my_tokens, total, dp_size, my_rank
                        )
                    )
                    for my_rank in range(dp_size)
                    for my_tokens in (0, 9, total // dp_size)
                }
                with self.subTest(dp_size=dp_size, total=total):
                    self.assertEqual(
                        len(splits),
                        1,
                        f"split varies with my_rank/my_tokens: {splits}",
                    )

    def test_uneven_remainder_still_sums_exactly(self):
        """Remainder not evenly divisible by dp_size is a real off-by-one
        trap for the extra-distribution -- assert the total is still exact,
        not short by the undistributed remainder."""
        split = synthetic_dp_verify_sum_split(
            my_tokens=1, total=10, dp_size=4, my_rank=0
        )
        self.assertEqual(sum(split), 10)
        self.assertGreaterEqual(split[0], 1)

    def test_refuses_total_below_canonical_coverage(self):
        """A total that cannot give every rank a slot covering `my_tokens`
        must fail loudly at capture-prep time, not trace an undersized
        geometry (the capture loop's feasibility filter skips these)."""
        with self.assertRaises(AssertionError):
            synthetic_dp_verify_sum_split(my_tokens=5, total=16, dp_size=4, my_rank=0)

    def test_dp_size_one_no_other_ranks(self):
        split = synthetic_dp_verify_sum_split(
            my_tokens=5, total=5, dp_size=1, my_rank=0
        )
        self.assertEqual(split, [5])


class TestLadderFeedsSyntheticSplitWithoutUnitMismatch(CustomTestCase):
    """Bug regression: the first live boot of this feature crashed every DP
    rank during CUDA graph capture with
    ``AssertionError: sum_bucket total 104 is below this rank's own local
    tokens 288`` -- build_dp_verify_sum_bucket_ladder produced sums in
    request-count units (dp_size * local_bucket) while capture_prepare feeds
    synthetic_dp_verify_sum_split the LOCAL TOKEN count (bs *
    num_tokens_per_req) as `my_tokens` and a ladder value as `total`. Fixed
    by scaling the ladder's values by num_tokens_per_req. This test drives
    every (local_bucket, sum_bucket) pair the capture loop would actually
    visit through synthetic_dp_verify_sum_split exactly as capture_prepare
    does, so a reintroduced unit mismatch fails here instead of on the next
    live GPU boot."""

    def test_every_ladder_entry_is_synthetic_split_safe(self):
        for dp_size in (2, 4, 8):
            ladder = build_dp_verify_sum_bucket_ladder(
                _CAPTURE_BS,
                dp_size=dp_size,
                window=2,
                num_tokens_per_req=_NUM_TOKENS_PER_REQ,
            )
            for local_bucket, sums in ladder.items():
                my_tokens = local_bucket * _NUM_TOKENS_PER_REQ
                # Mirror the capture loop's feasibility filter: skip totals
                # too small to give every rank a canonical slot covering this
                # local bucket (the loop uses the same dp_size-scaled bound).
                for sum_bucket in [s for s in sums if s >= dp_size * my_tokens]:
                    rank_splits = set()
                    for my_rank in range(dp_size):
                        with self.subTest(
                            dp_size=dp_size,
                            local_bucket=local_bucket,
                            sum_bucket=sum_bucket,
                            my_rank=my_rank,
                        ):
                            split = synthetic_dp_verify_sum_split(
                                my_tokens, sum_bucket, dp_size, my_rank
                            )
                            self.assertEqual(sum(split), sum_bucket)
                            self.assertGreaterEqual(split[my_rank], my_tokens)
                            rank_splits.add(tuple(split))
                    # Canonical-split invariance across ranks, checked on the
                    # exact (bucket, total) pairs the capture loop visits.
                    self.assertEqual(len(rank_splits), 1)


def _gate_kwargs(**overrides):
    kwargs = dict(
        spec_algorithm=SpeculativeAlgorithm.DSPARK,
        require_mlp_tp_gather=True,
        is_target_verify_capture=True,
        is_draft_worker=False,
    )
    kwargs.update(overrides)
    return kwargs


class TestDspparkDpVerifySumCaptureEnabledGate(CustomTestCase):
    """Completeness/negative-branch contract: every one of these conditions
    must independently gate the feature off. A predicate that degrades to
    always-true (or drops one of these checks) would silently activate a
    SUM_LEN capture path for a topology/algorithm it was never designed for
    -- e.g. plain decode or a non-DSPARK spec algorithm under DP attention,
    which the mirror-buffer's SUM_LEN gather/scatter assumptions were never
    validated against."""

    def test_enabled_when_every_condition_holds(self):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs, patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.is_dp_gatherv_configured",
            return_value=False,
        ):
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = True
            self.assertTrue(dspark_dp_verify_sum_capture_enabled(**_gate_kwargs()))

    def test_disabled_by_default_env(self):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs:
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = (
                False
            )
            self.assertFalse(dspark_dp_verify_sum_capture_enabled(**_gate_kwargs()))

    def test_disabled_for_non_dspark_algorithm(self):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs:
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = True
            self.assertFalse(
                dspark_dp_verify_sum_capture_enabled(
                    **_gate_kwargs(spec_algorithm=SpeculativeAlgorithm.EAGLE3)
                )
            )

    def test_disabled_without_mlp_tp_gather(self):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs:
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = True
            self.assertFalse(
                dspark_dp_verify_sum_capture_enabled(
                    **_gate_kwargs(require_mlp_tp_gather=False)
                )
            )

    def test_disabled_outside_target_verify_capture(self):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs:
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = True
            self.assertFalse(
                dspark_dp_verify_sum_capture_enabled(
                    **_gate_kwargs(is_target_verify_capture=False)
                )
            )

    def test_disabled_for_draft_worker(self):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs:
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = True
            self.assertFalse(
                dspark_dp_verify_sum_capture_enabled(
                    **_gate_kwargs(is_draft_worker=True)
                )
            )

    def test_raises_when_gatherv_configured(self):
        """Building block #4's caveat: SGLANG_DP_USE_GATHERV's gather path
        sizes offsets from a Python-side list, not a GPU tensor -- not
        CUDA-graph-replay-safe for a shape that must vary run to run. This
        must fail loud at construction time, not silently corrupt output."""
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs"
        ) as mock_envs, patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.is_dp_gatherv_configured",
            return_value=True,
        ):
            mock_envs.SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE.get.return_value = True
            with self.assertRaisesRegex(ValueError, "SGLANG_DP_USE_GATHERV"):
                dspark_dp_verify_sum_capture_enabled(**_gate_kwargs())


def _resolve_kwargs(**overrides):
    kwargs = dict(
        global_num_reqs=[4, 4, 4, 4],
        dp_rank=0,
        num_tokens_per_req=_NUM_TOKENS_PER_REQ,
        capture_bs=_CAPTURE_BS,
        sum_buckets_by_local=build_dp_verify_sum_bucket_ladder(
            _CAPTURE_BS, dp_size=4, window=0, num_tokens_per_req=_NUM_TOKENS_PER_REQ
        ),
        max_bs=max(_CAPTURE_BS),
    )
    kwargs.update(overrides)
    return kwargs


class TestResolveDpVerifySumGraphBuckets(CustomTestCase):
    """Regression cover for TODO.md J14d's second live crash:
    `next_token_logits row count mismatch for DFlash verify adjustments.
    Expected 36, got 9` -- 36 = 4 real requests x 9 draft tokens, 9 = the
    1-request graph that got replayed instead.

    original_global_num_tokens_cpu holds per-rank REQUEST counts on a
    decode-family round (prepare_mlp_sync_batch_raw's is_decode() branch sets
    `num_tokens = local_batch.batch_size()`). The pre-fix code divided that
    count by num_tokens_per_req before bucketing, so every per-rank batch
    smaller than the 9-token draft width -- i.e. essentially all of them --
    collapsed onto local_bucket 1.
    """

    def test_local_bucket_is_the_request_count_not_a_token_division(self):
        local_bucket, _ = resolve_dp_verify_sum_graph_buckets(
            **_resolve_kwargs(global_num_reqs=[4, 4, 4, 4])
        )
        self.assertEqual(local_bucket, 4)

    def test_every_request_count_under_the_draft_width_keeps_its_own_bucket(self):
        """The pre-fix division mapped all of these onto one bucket; each must
        now round up to its own captured shape."""
        for num_reqs in range(1, _NUM_TOKENS_PER_REQ + 1):
            with self.subTest(num_reqs=num_reqs):
                local_bucket, _ = resolve_dp_verify_sum_graph_buckets(
                    **_resolve_kwargs(global_num_reqs=[num_reqs] * 4)
                )
                self.assertGreaterEqual(local_bucket, num_reqs)
                self.assertEqual(
                    local_bucket, min(b for b in _CAPTURE_BS if b >= num_reqs)
                )

    def test_replayed_graph_always_has_room_for_every_real_request(self):
        """The invariant the crash violated: whatever pair we admit, the
        local bucket must cover this rank's real request count, so the graph's
        logits rows (bucket * draft width) are never fewer than the sampler's
        (real requests * draft width)."""
        for dp_rank in range(4):
            for skew in ([1, 2, 3, 4], [4, 1, 1, 1], [8, 8, 8, 8], [2, 5, 3, 6]):
                with self.subTest(dp_rank=dp_rank, skew=skew):
                    resolved = resolve_dp_verify_sum_graph_buckets(
                        **_resolve_kwargs(global_num_reqs=skew, dp_rank=dp_rank)
                    )
                    if resolved is None:
                        continue
                    local_bucket, _ = resolved
                    self.assertGreaterEqual(local_bucket, skew[dp_rank])

    def test_sum_bucket_is_token_denominated(self):
        """Ladder keys are request counts, values are token counts -- the sum
        lookup is the one place the conversion belongs. Comparing a
        request-count sum against a token-count ladder is the unit bug that
        produced 'Expected 36, got 9'."""
        _, sum_bucket = resolve_dp_verify_sum_graph_buckets(
            **_resolve_kwargs(global_num_reqs=[4, 4, 4, 4])
        )
        self.assertEqual(sum_bucket, 4 * max(_CAPTURE_BS) * _NUM_TOKENS_PER_REQ)
        self.assertGreaterEqual(sum_bucket, 4 * 4 * _NUM_TOKENS_PER_REQ)

    def test_returns_none_when_local_count_exceeds_captured_ladder(self):
        """A clean miss, not an exception: the caller falls back to eager."""
        over = max(_CAPTURE_BS) + 1
        self.assertIsNone(
            resolve_dp_verify_sum_graph_buckets(
                **_resolve_kwargs(global_num_reqs=[over, 1, 1, 1])
            )
        )

    def test_admission_and_replay_cannot_disagree(self):
        """Both call sites go through this one helper, so the same inputs must
        map to the same pair -- load_batch asserts on exactly this."""
        for skew in ([1, 1, 1, 1], [3, 2, 4, 1], [8, 4, 2, 1]):
            with self.subTest(skew=skew):
                kwargs = _resolve_kwargs(global_num_reqs=skew, dp_rank=2)
                self.assertEqual(
                    resolve_dp_verify_sum_graph_buckets(**kwargs),
                    resolve_dp_verify_sum_graph_buckets(**kwargs),
                )


class TestResolveRaggedDpVerifySumGraphBuckets(CustomTestCase):
    @staticmethod
    def _ladder(dp_size=4):
        return build_ragged_dp_verify_sum_bucket_ladder(
            _RAGGED_CAPTURE_NUM_TOKENS, dp_size=dp_size, window=0
        )

    def _resolve(self, global_tokens, dp_rank, local_tokens):
        return resolve_ragged_dp_verify_sum_graph_buckets(
            global_graph_num_tokens=list(global_tokens),
            dp_rank=dp_rank,
            local_graph_num_tokens=local_tokens,
            capture_num_tokens=_RAGGED_CAPTURE_NUM_TOKENS,
            sum_buckets_by_local=self._ladder(),
            max_num_tokens=max(_RAGGED_CAPTURE_NUM_TOKENS),
        )

    def test_capture_ladder_keys_and_values_are_tokens(self):
        ladder = self._ladder()
        self.assertEqual(list(ladder), _RAGGED_CAPTURE_NUM_TOKENS)
        self.assertEqual(
            {tuple(sums) for sums in ladder.values()},
            {(4 * max(_RAGGED_CAPTURE_NUM_TOKENS),)},
        )

    def test_capture_and_replay_keys_are_token_denominated(self):
        """Compact ShapeKey.size and ShapeKey.sum_bucket are both tokens.

        Static verify deliberately keeps a request-denominated local key;
        multiplying this compact local key by the draft width again would
        look for bucket 162 instead of the captured 18-token graph.
        """
        global_tokens = [9, 18, 36, 9]
        local_bucket, sum_bucket = self._resolve(global_tokens, 1, 18)
        self.assertEqual(local_bucket, 18)
        self.assertEqual(sum_bucket, 4 * max(_RAGGED_CAPTURE_NUM_TOKENS))

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        capture_key = runner._make_graph_key(18, sum_bucket=sum_bucket)
        replay_key = runner._make_graph_key(local_bucket, sum_bucket=sum_bucket)
        self.assertEqual(capture_key, replay_key)
        self.assertEqual(replay_key.size, 18)
        self.assertEqual(replay_key.sum_bucket, sum_bucket)

    def test_all_ranks_share_sum_key_but_keep_local_token_key(self):
        global_tokens = [9, 18, 36, 72]
        resolved = [
            self._resolve(global_tokens, rank, local_tokens)
            for rank, local_tokens in enumerate(global_tokens)
        ]
        self.assertTrue(all(pair is not None for pair in resolved))
        self.assertEqual([pair[0] for pair in resolved], global_tokens)
        self.assertEqual(len({pair[1] for pair in resolved}), 1)

    def test_idle_rank_uses_its_dummy_graph_bucket_not_zero(self):
        # The planner rounds a gathered idle contribution (0) to the smallest
        # captured graph bucket before this resolver.  The idle graph executes
        # those dummy rows and therefore contributes 9, not zero, to SUM_LEN.
        global_tokens = [9, 18, 36, 72]
        self.assertEqual(self._resolve(global_tokens, 0, 9)[0], 9)
        self.assertIsNone(self._resolve([0, 18, 36, 72], 0, 9))

    def test_layout_and_shared_vector_mismatch_is_a_clean_miss(self):
        self.assertIsNone(self._resolve([9, 18, 36, 72], 1, 36))

    def test_out_of_grid_rank_makes_every_rank_miss(self):
        global_tokens = [9, 18, max(_RAGGED_CAPTURE_NUM_TOKENS) + 1, 9]
        for rank, local_tokens in enumerate((9, 18, 36, 9)):
            self.assertIsNone(self._resolve(global_tokens, rank, local_tokens))

    def test_shared_request_capacity_gate_prevents_rank_local_fallback(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.max_bs = 8
        runner.captured_req_width = _NUM_TOKENS_PER_REQ

        self.assertTrue(
            runner._all_compact_dp_ranks_fit_graph_slots(
                global_num_reqs=[0, 2, 4, 8],
                global_graph_num_tokens=[9, 18, 36, 72],
            )
        )
        # The 9-request rank cannot fit even though its 18-token bucket is a
        # valid captured token key. Every rank must make the same graph miss.
        self.assertFalse(
            runner._all_compact_dp_ranks_fit_graph_slots(
                global_num_reqs=[0, 2, 9, 8],
                global_graph_num_tokens=[9, 18, 18, 72],
            )
        )


class TestDpVerifySumGraphCoverageWarning(CustomTestCase):
    """TODO.md J14d: an uncovered batch is not a cheap fallback under DP
    attention. The rank that drops to eager can hit a cold Triton autotune for
    a sparse-attention shape no captured graph warmed, and every other rank
    blocks in its DP collective until that compile finishes -- observed as a
    10+ minute whole-group stall with one rank inside cuModuleLoadData."""

    @staticmethod
    def _runner(*, enabled=True, max_bs=4, max_running_requests=32):
        return SimpleNamespace(
            dspark_dp_verify_sum_capture_enabled=enabled,
            max_bs=max_bs,
            # The RESOLVED per-rank value, not server_args': SGLang divides
            # --max-running-requests across DP ranks, and reading the
            # undivided one made this warn on correct DP configs.
            model_runner=SimpleNamespace(
                max_running_requests=max_running_requests,
                server_args=SimpleNamespace(max_running_requests=99999),
            ),
        )

    def _warn_calls(self, runner):
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.logger"
        ) as mock_logger:
            DecodeCudaGraphRunner._warn_if_dp_verify_sum_graph_coverage_is_partial(
                runner
            )
            return mock_logger.warning.call_args_list

    def test_warns_when_ladder_cannot_reach_max_running_requests(self):
        calls = self._warn_calls(self._runner(max_bs=4, max_running_requests=32))
        self.assertEqual(len(calls), 1)
        self.assertIn("--cuda-graph-max-bs", calls[0].args[0])

    def test_silent_when_ladder_covers_every_admissible_batch(self):
        self.assertEqual(
            self._warn_calls(self._runner(max_bs=32, max_running_requests=32)), []
        )

    def test_silent_when_feature_is_disabled(self):
        """Every other topology takes the pre-existing MAX_LEN path; this
        warning must not fire for them."""
        self.assertEqual(
            self._warn_calls(
                self._runner(enabled=False, max_bs=4, max_running_requests=32)
            ),
            [],
        )

    def test_silent_when_max_running_requests_is_unset(self):
        self.assertEqual(
            self._warn_calls(self._runner(max_bs=4, max_running_requests=None)), []
        )


class TestDpVerifySumDecisionAgreesAcrossRanks(CustomTestCase):
    """The invariant TODO.md J14d's stalls violated. DP ranks advance in
    lockstep, so for one step every rank must independently reach the SAME
    verdict: all replay a graph with the SAME sum bucket, or all go eager.

    Before the fix the ladder was keyed on the caller's own local bucket, so
    skew (2,3,4,5) resolved per-rank sums [None, None, 144, 180] -- two ranks
    eager, two replaying graphs against differently sized collectives.
    """

    @staticmethod
    def _decide(global_num_reqs, dp_rank):
        return resolve_dp_verify_sum_graph_buckets(
            global_num_reqs=list(global_num_reqs),
            dp_rank=dp_rank,
            num_tokens_per_req=_NUM_TOKENS_PER_REQ,
            capture_bs=_CAPTURE_BS,
            sum_buckets_by_local=build_dp_verify_sum_bucket_ladder(
                _CAPTURE_BS, dp_size=4, window=0, num_tokens_per_req=_NUM_TOKENS_PER_REQ
            ),
            max_bs=max(_CAPTURE_BS),
        )

    _SKEWS = [
        (1, 8, 8, 8),
        (2, 3, 4, 5),
        (8, 8, 8, 8),
        (1, 1, 1, 8),
        (4, 4, 4, 4),
        (1, 2, 3, 4),
        (0, 0, 0, 1),
        (7, 1, 8, 2),
        (32, 1, 1, 1),
    ]

    def test_all_ranks_agree_on_graph_or_eager(self):
        for skew in self._SKEWS:
            with self.subTest(skew=skew):
                verdicts = [self._decide(skew, r) is None for r in range(4)]
                self.assertEqual(
                    len(set(verdicts)),
                    1,
                    f"ranks disagreed on graph-vs-eager for {skew}: {verdicts}",
                )

    def test_all_ranks_agree_on_the_sum_bucket(self):
        """The sum bucket sizes a collective, so a disagreement here is a
        mismatched all-reduce, not just a perf wobble."""
        for skew in self._SKEWS:
            with self.subTest(skew=skew):
                resolved = [self._decide(skew, r) for r in range(4)]
                if resolved[0] is None:
                    continue
                sums = {pair[1] for pair in resolved}
                self.assertEqual(
                    len(sums), 1, f"ranks disagreed on sum bucket for {skew}: {sums}"
                )

    def test_each_rank_still_gets_its_own_local_bucket(self):
        """Agreement is about the shared collective size; the local bucket is
        this rank's own buffer and must still track its real request count."""
        skew = (1, 4, 7, 8)
        for rank, num_reqs in enumerate(skew):
            resolved = self._decide(skew, rank)
            self.assertIsNotNone(resolved)
            self.assertGreaterEqual(resolved[0], num_reqs)

    def test_one_overloaded_rank_sends_everyone_eager_together(self):
        """A rank past the ladder cannot replay, so no rank may replay --
        otherwise the peers wait in a collective the graph never runs."""
        skew = (max(_CAPTURE_BS) + 1, 1, 1, 1)
        for rank in range(4):
            self.assertIsNone(self._decide(skew, rank))


class TestIdleReplayVerifySpecInfo(CustomTestCase):
    """TODO.md J14d: a DP rank holding zero requests still has to run the
    collective when its peers verify, but prepare_for_idle never sets
    spec_info and the replay view reports the captured TARGET_VERIFY mode --
    so a sparse backend used to raise "TARGET_VERIFY forward without a
    spec_info.draft_token_num" and take the whole 4-GPU group down with it."""

    @staticmethod
    def _runner(captured_req_width=9):
        return SimpleNamespace(
            _idle_verify_spec_info=None,
            captured_req_width=captured_req_width,
            model_runner=SimpleNamespace(attn_backend=object()),
            buffers=SimpleNamespace(custom_mask="CUSTOM_MASK_BUFFER"),
        )

    def _build(self, runner, build_custom_mask=False):
        with patch(
            "sglang.srt.speculative.dflash_utils.resolve_dflash_verify_mask_policy",
            return_value=("StubBackend", build_custom_mask),
        ):
            return DecodeCudaGraphRunner._idle_replay_verify_spec_info(runner)

    def test_width_is_the_config_constant_not_request_derived(self):
        """An idle batch must use the captured width without request metadata."""
        for width in (2, 9, 16):
            with self.subTest(width=width):
                spec_info = self._build(self._runner(captured_req_width=width))
                self.assertEqual(spec_info.draft_token_num, width)

    def test_ragged_layout_is_none(self):
        """A ragged batch takes _can_run_ragged_verify_graph instead, and a
        non-None layout is what the sparse backend rejects as unsupported."""
        self.assertIsNone(self._build(self._runner()).ragged_verify_layout)

    def test_cached_across_calls(self):
        """Config-static, so it must not be rebuilt every idle step."""
        runner = self._runner()
        self.assertIs(self._build(runner), self._build(runner))

    def test_custom_mask_follows_backend_policy(self):
        self.assertIsNone(self._build(self._runner(), build_custom_mask=False).custom_mask)
        self.assertEqual(
            self._build(self._runner(), build_custom_mask=True).custom_mask,
            "CUSTOM_MASK_BUFFER",
        )


class TestDsparkVerifyForwardBatchFields(CustomTestCase):
    """init_dspark_target_verify copies ScheduleBatch fields into ForwardBatch by
    name, apart from init_new; a rename upstream (#35640) left it reading a removed
    field, which crashed the first DSPARK verify of every server."""

    @staticmethod
    def _names(cls):
        annotated = set().union(*(vars(c).get("__annotations__", {}) for c in cls.__mro__))
        return annotated | set(dir(cls))

    def test_reads_and_writes_only_existing_fields(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        source = textwrap.dedent(inspect.getsource(ForwardBatch.init_dspark_target_verify))
        tree = ast.parse(source)
        read = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "batch"
        }
        [values] = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "values" for t in node.targets)
        ]
        written = {key.value for key in values.keys}
        self.assertEqual(read - self._names(ScheduleBatch), set())
        self.assertEqual(written - self._names(ForwardBatch), set())


if __name__ == "__main__":
    unittest.main()
