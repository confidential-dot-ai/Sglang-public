import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    compact_dp_sum_graph_token_buckets,
    dp_global_verify_tier_num_tokens,
    local_verify_tier_num_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_info import spec_scale_global_num_tokens
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestLocalVerifyTierNumTokens(CustomTestCase):
    def test_no_budget_returns_sentinel(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=None,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            -1,
        )

    def test_budget_adds_to_anchor_floor(self):
        self.assertEqual(
            local_verify_tier_num_tokens(
                bs=8,
                verify_token_budget=10,
                verify_num_draft_tokens=6,
                min_verify_len=1,
            ),
            18,
        )

    # Clamp/floor variants (verify-all clamp, min_verify_len floor, min=0) are
    # covered by the TestBusyIdleGraphKeyIdentity sweep bounds.


class TestDpGlobalVerifyTierNumTokens(CustomTestCase):
    def test_any_sentinel_pins_everyone(self):
        # The sweep never emits a -1 contribution, so this is the only guard
        # on "any rank without a budget pins everyone"; losing it forks graph
        # keys across DP ranks.
        self.assertIsNone(
            dp_global_verify_tier_num_tokens(global_tier_num_tokens=[100, -1, 50, 0])
        )


class TestCompactDpSumGraphTokenBuckets(CustomTestCase):
    _GRID = [9, 18, 36, 72]

    @classmethod
    def _model_runner(cls, *, enabled=True):
        return SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                dspark_dp_verify_sum_capture_enabled=enabled,
                capture_num_tokens=cls._GRID,
                dp_size=4,
                max_bs=8,
            )
        )

    def test_sum_mode_retains_per_rank_buckets_and_rounds_idle(self):
        self.assertEqual(
            compact_dp_sum_graph_token_buckets(
                model_runner=self._model_runner(),
                global_tier_num_tokens=[0, 10, 35, 72],
            ),
            [9, 18, 36, 72],
        )

    def test_max_mode_keeps_existing_max_collapse(self):
        contributions = [0, 10, 35, 72]
        self.assertIsNone(
            compact_dp_sum_graph_token_buckets(
                model_runner=self._model_runner(enabled=False),
                global_tier_num_tokens=contributions,
            )
        )
        self.assertEqual(
            dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=contributions
            ),
            72,
        )

    def test_missing_or_out_of_grid_requirement_is_shared_miss(self):
        for contributions in (
            None,
            [],
            [0, -1, 18, 9],
            [0, 73, 18, 9],
            [0, 18, 9],
        ):
            with self.subTest(contributions=contributions):
                self.assertIsNone(
                    compact_dp_sum_graph_token_buckets(
                        model_runner=self._model_runner(),
                        global_tier_num_tokens=contributions,
                    )
                )

    def test_request_count_that_exceeds_bucket_slots_is_shared_miss(self):
        self.assertIsNone(
            compact_dp_sum_graph_token_buckets(
                model_runner=self._model_runner(),
                global_tier_num_tokens=[0, 2, 9, 8],
                global_num_reqs=[0, 2, 10, 8],
            )
        )

    def test_graph_bucket_vector_overrides_uniform_request_scaling(self):
        spec_info = SimpleNamespace(
            dp_graph_num_tokens=[9, 18, 36, 72],
            num_tokens_per_req=9,
            num_tokens_for_logprob_per_req=9,
        )
        tokens, logprob_tokens = spec_scale_global_num_tokens(
            spec_info,
            global_num_tokens=[0, 2, 4, 8],
            global_num_tokens_for_logprob=[0, 2, 4, 8],
        )
        self.assertEqual(tokens, [9, 18, 36, 72])
        self.assertEqual(logprob_tokens, tokens)


class TestBusyIdleGraphKeyModeSelection(CustomTestCase):
    @staticmethod
    def _worker(*, sum_enabled):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_is_moe = True
        worker._verify_planner = SimpleNamespace(is_compact_mode=True)
        worker.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                dspark_dp_verify_sum_capture_enabled=sum_enabled,
                capture_num_tokens=[9, 18, 36, 72],
                max_bs=8,
            )
        )
        worker.device = "cpu"
        worker.verify_num_draft_tokens = 9
        return worker

    @staticmethod
    def _batch():
        return SimpleNamespace(
            global_num_tokens=[0, 3, 2, 1],
            global_spec_verify_tier_num_tokens=[0, 20, 12, 7],
        )

    def test_sum_mode_busy_rank_selects_its_local_token_bucket(self):
        worker = self._worker(sum_enabled=True)
        parallel = SimpleNamespace(enable_dp_attention=True, attn_dp_rank=2)
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
            return_value=parallel,
        ):
            self.assertEqual(worker._dp_verify_tier_num_tokens(self._batch()), 18)

    def test_sum_mode_idle_rank_uses_smallest_dummy_graph(self):
        worker = self._worker(sum_enabled=True)
        parallel = SimpleNamespace(enable_dp_attention=True, attn_dp_rank=0)
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
            return_value=parallel,
        ), patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.idle_ragged_layout",
            return_value="idle-layout",
        ) as build_idle:
            self.assertEqual(
                worker._idle_verify_ragged_layout(self._batch()), "idle-layout"
            )
        self.assertEqual(build_idle.call_args.kwargs["tier_num_reqs"], 1)
        self.assertEqual(build_idle.call_args.kwargs["dp_tier_num_tokens"], 9)

    def test_flag_off_preserves_shared_max_bucket(self):
        worker = self._worker(sum_enabled=False)
        parallel = SimpleNamespace(enable_dp_attention=True, attn_dp_rank=0)
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
            return_value=parallel,
        ):
            self.assertEqual(worker._dp_verify_tier_num_tokens(self._batch()), 20)

    def test_sum_mode_invalid_shared_vector_returns_no_planner_floor(self):
        worker = self._worker(sum_enabled=True)
        parallel = SimpleNamespace(enable_dp_attention=True, attn_dp_rank=1)
        batch = self._batch()
        batch.global_spec_verify_tier_num_tokens = [0, 73, 12, 7]
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
            return_value=parallel,
        ):
            self.assertIsNone(worker._dp_verify_tier_num_tokens(batch))

    def test_sum_mode_invalid_shared_vector_forces_idle_static_fallback(self):
        worker = self._worker(sum_enabled=True)
        batch = self._batch()
        batch.global_spec_verify_tier_num_tokens = [0, -1, 12, 7]
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_worker_v2.idle_ragged_layout",
        ) as build_idle:
            self.assertIsNone(worker._idle_verify_ragged_layout(batch))
        build_idle.assert_not_called()

    def test_sum_mode_missing_local_bucket_forces_busy_static_fallback(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner.model_runner = self._worker(sum_enabled=True).model_runner

        # The shared token gather contains a sentinel/out-of-grid value, so
        # the worker passes no local SUM bucket.  Do not build compact's
        # legacy DP-max layout: eager DP counts would remain static-width.
        self.assertIsNone(
            planner.schedule_layout(
                req_pool_indices=torch.arange(2),
                prefix_lens=torch.ones(2),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                global_num_reqs=8,
                dp_tier_num_tokens=None,
            )
        )

    def test_verify_all_layout_uses_local_bucket_only_in_sum_mode(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._is_verify_all = True
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._uniform_layout_cache = {}
        planner.verify_num_draft_tokens = 9
        planner.model_runner = self._worker(sum_enabled=True).model_runner

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_planner.uniform_ragged_layout",
            return_value="sum-layout",
        ) as build_layout:
            result = planner.schedule_layout(
                req_pool_indices=torch.arange(3),
                prefix_lens=torch.ones(3),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                global_num_reqs=8,
                dp_tier_num_tokens=36,
            )
        self.assertEqual(result, "sum-layout")
        self.assertEqual(build_layout.call_args.kwargs["tier_num_reqs"], 3)
        self.assertEqual(build_layout.call_args.kwargs["tier_num_tokens"], 36)

        planner._uniform_layout_cache = {}
        planner.model_runner = self._worker(sum_enabled=False).model_runner
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_planner.uniform_ragged_layout",
            return_value="max-layout",
        ) as build_layout:
            result = planner.schedule_layout(
                req_pool_indices=torch.arange(3),
                prefix_lens=torch.ones(3),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                global_num_reqs=8,
                dp_tier_num_tokens=36,
            )
        self.assertEqual(result, "max-layout")
        self.assertEqual(build_layout.call_args.kwargs["tier_num_reqs"], 8)
        self.assertIsNone(build_layout.call_args.kwargs["tier_num_tokens"])


class TestDraftDpSyncMetadata(CustomTestCase):
    def test_preserves_unscaled_request_counts_for_cuda_graph_admission(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._draft_block_spec_info = SimpleNamespace(
            num_tokens_per_req=6,
            num_tokens_for_logprob_per_req=1,
        )
        proposer.draft_model_runner = SimpleNamespace(device="cpu")
        proposer._num_token_non_padded = torch.empty((1,), dtype=torch.int32)

        forward_batch = SimpleNamespace(input_ids=torch.arange(6))
        batch = SimpleNamespace(
            global_num_tokens=[1, 3, 0, 2],
            global_num_tokens_for_logprob=[1, 3, 0, 2],
            can_run_decode_cuda_graph=True,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=True,
        ):
            proposer._fill_dp_moe_sync_metadata(forward_batch, batch)

        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            [1, 3, 0, 2],
        )
        self.assertEqual(forward_batch.global_num_tokens_cpu, [6, 18, 0, 12])
        # The LOCAL count reuses the proposer's persistent buffer; the GLOBAL
        # count is the invariant every DP rank agrees on.
        self.assertIs(
            forward_batch.num_token_non_padded, proposer._num_token_non_padded
        )
        self.assertEqual(forward_batch.num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 6)
        self.assertEqual(forward_batch.global_num_token_non_padded.item(), 6)
        self.assertEqual(forward_batch.global_num_token_non_padded.dtype, torch.int32)
        self.assertEqual(forward_batch.global_num_token_non_padded_cpu, 6)
        self.assertTrue(forward_batch.can_run_decode_cuda_graph)


class TestBusyIdleGraphKeyIdentity(CustomTestCase):
    def test_busy_and_idle_floors_agree_on_random_topologies(self):
        rng = random.Random(20260703)
        for _ in range(2000):
            verify_num_draft_tokens = rng.randint(2, 8)
            min_verify_len = rng.randint(0, verify_num_draft_tokens - 1)
            effective_min = max(min_verify_len, 1)
            num_ranks = rng.randint(1, 8)
            contributions = []
            num_reqs_per_rank = []
            for _ in range(num_ranks):
                if rng.random() < 0.3:
                    num_reqs_per_rank.append(0)
                    contributions.append(0)
                    continue
                bs = rng.randint(1, 512)
                budget = rng.randint(0, bs * verify_num_draft_tokens)
                num_reqs_per_rank.append(bs)
                contributions.append(
                    local_verify_tier_num_tokens(
                        bs=bs,
                        verify_token_budget=budget,
                        verify_num_draft_tokens=verify_num_draft_tokens,
                        min_verify_len=min_verify_len,
                    )
                )
            tier_num_tokens = dp_global_verify_tier_num_tokens(
                global_tier_num_tokens=contributions
            )
            global_num_reqs = max(num_reqs_per_rank)
            if tier_num_tokens is None:
                self.assertEqual(global_num_reqs, 0)
                continue

            self.assertGreaterEqual(tier_num_tokens, global_num_reqs * effective_min)
            self.assertLessEqual(
                tier_num_tokens, global_num_reqs * verify_num_draft_tokens
            )

            busy_floor = min(tier_num_tokens, global_num_reqs * verify_num_draft_tokens)
            self.assertEqual(busy_floor, tier_num_tokens)

            idle_lens_total = global_num_reqs
            idle_bucket_input = max(idle_lens_total, tier_num_tokens)
            self.assertEqual(idle_bucket_input, tier_num_tokens)


if __name__ == "__main__":
    unittest.main()
