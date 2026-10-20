"""Startup planning of the DeepSeek-V4.1 prefill indexer's logits tiles."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.layers.attention.dsv4.v41_indexer import memory_budget, scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.memory_budget import (
    INDEX_LOGITS_BUDGET_CAP_BYTES,
    INDEX_LOGITS_BUDGET_FLOOR_BYTES,
    eager_indexer_peak_bytes,
    plan_index_logits_budget,
    planned_index_logits_budget,
    set_planned_index_logits_budget,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1 << 30
K = 1024

# (path, context, rows, budget GiB, measured peak GiB): one TP2 rank on B200,
# DeepGEMM fp32 logits; "publish" is the candidate source, the largest layer.
MEASURED = (
    ("full_topk", 1024 * K, 16384, 2, 2.110),
    ("full_topk", 128 * K, 8192, 0.25, 0.277),
    ("publish", 128 * K, 8192, 2, 2.371),
    ("publish", 128 * K, 16384, 0.25, 0.780),
    ("publish", 512 * K, 16384, 2, 2.698),
    ("publish", 1024 * K, 12288, 2, 2.821),
    ("publish", 1024 * K, 16384, 2, 2.985),
    ("publish", 1024 * K, 16384, 0.5, 1.297),
    ("publish", 1024 * K, 16384, 0.25, 1.224),
    ("consume", 1024 * K, 16384, 2, 0.544),
)


class TestEagerIndexerPeakModel(CustomTestCase):
    def test_bounds_every_measured_peak(self):
        for path, ctx, rows, budget, measured in MEASURED:
            with self.subTest(path=path, ctx=ctx, rows=rows, budget=budget):
                peak = eager_indexer_peak_bytes(
                    budget_bytes=int(budget * GIB), rows=rows, context_len=ctx
                )
                self.assertGreaterEqual(peak / GIB, measured)
                if path == "publish":
                    # Not so loose that planning wastes the headroom.
                    self.assertLess(peak / GIB, measured * 1.4)

    def test_tile_is_bounded_by_the_budget_at_any_context(self):
        small = eager_indexer_peak_bytes(
            budget_bytes=GIB, rows=16384, context_len=128 * K
        )
        large = eager_indexer_peak_bytes(
            budget_bytes=GIB, rows=16384, context_len=1024 * K
        )
        # 8x the context adds the per-position term (~0.05 B per row and
        # position), not 8x the tile.
        self.assertLess(large - small, GIB)

    def test_a_short_batch_needs_only_its_own_logits(self):
        peak = eager_indexer_peak_bytes(budget_bytes=2 * GIB, rows=4, context_len=4096)
        self.assertLess(peak, 64 << 20)


class TestPlanIndexLogitsBudget(CustomTestCase):
    def test_keeps_the_cap_with_room(self):
        # The shipped chunk-12288 B200 prefill keeps ~9.7 GiB free after capture.
        budget = plan_index_logits_budget(
            free_bytes=int(9.67 * GIB),
            rows=16384,
            context_len=1024 * K,
            reserve_bytes=4 * GIB,
        )
        self.assertEqual(budget, INDEX_LOGITS_BUDGET_CAP_BYTES)

    def test_halves_until_it_fits(self):
        budget = plan_index_logits_budget(
            free_bytes=6 * GIB, rows=16384, context_len=1024 * K, reserve_bytes=4 * GIB
        )
        self.assertLess(budget, INDEX_LOGITS_BUDGET_CAP_BYTES)
        self.assertGreaterEqual(budget, INDEX_LOGITS_BUDGET_FLOOR_BYTES)
        need = eager_indexer_peak_bytes(
            budget_bytes=budget, rows=16384, context_len=1024 * K
        )
        self.assertLessEqual(need + 4 * GIB, 6 * GIB)
        bigger = eager_indexer_peak_bytes(
            budget_bytes=budget * 2, rows=16384, context_len=1024 * K
        )
        self.assertGreater(bigger + 4 * GIB, 6 * GIB)

    def test_refuses_to_start_when_nothing_fits(self):
        with self.assertRaisesRegex(ValueError, "context_len=1048576.*16384 rows"):
            plan_index_logits_budget(
                free_bytes=4 * GIB,
                rows=16384,
                context_len=1024 * K,
                reserve_bytes=4 * GIB,
            )


class TestPlannedBudget(CustomTestCase):
    def tearDown(self):
        set_planned_index_logits_budget(None)

    def test_defaults_to_the_cap_and_keeps_the_smallest_plan(self):
        set_planned_index_logits_budget(None)
        self.assertEqual(planned_index_logits_budget(), INDEX_LOGITS_BUDGET_CAP_BYTES)
        set_planned_index_logits_budget(GIB)
        set_planned_index_logits_budget(2 * GIB)  # a later, roomier plan
        self.assertEqual(planned_index_logits_budget(), GIB)

    def test_score_tiles_take_the_planned_budget(self):
        seen = []

        def tiles(**kwargs):
            seen.append(kwargs["budget_bytes"])
            return iter(())

        data = type(
            "Data",
            (),
            dict(
                q_fp4=None,
                q_sf=None,
                weights=None,
                request_starts=None,
                compress_lens=None,
                lens_per_request=[],
            ),
        )()
        with patch.object(scoring, "flat_index_logits_tiles", tiles):
            set_planned_index_logits_budget(512 << 20)
            list(scoring.score_tiles(data, kv=None))
            # A test patching the cap lower still wins.
            with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", 128 << 10):
                list(scoring.score_tiles(data, kv=None))
        self.assertEqual(seen, [512 << 20, 128 << 10])
        self.assertIs(
            memory_budget.planned_index_logits_budget,
            scoring.planned_index_logits_budget,
        )


class TestStartupPlanning(CustomTestCase):
    """The post-capture hook: plans prefill, refuses to start when nothing fits,
    and leaves decode and other models alone."""

    def setUp(self):
        from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

        self.setup = cuda_graph_setup
        self.addCleanup(set_planned_index_logits_budget, None)
        set_planned_index_logits_budget(None)

    def _plan(self, *, free_gib, mode="prefill", model_type="deepseek_v41"):
        runner = SimpleNamespace(
            device="cuda",
            gpu_id=0,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type), context_len=1 << 20
            ),
        )
        parallel = SimpleNamespace(
            launch_world_size=1, world_group=SimpleNamespace(cpu_group=None)
        )
        with ExitStack() as stack:
            for name, value in (
                ("get_disagg", lambda: SimpleNamespace(disaggregation_mode=mode)),
                (
                    "get_schedule",
                    lambda: SimpleNamespace(
                        chunked_prefill_size=12288, max_prefill_tokens=16384
                    ),
                ),
                ("get_parallel", lambda: parallel),
                ("get_available_gpu_memory", lambda *a, **kw: free_gib),
            ):
                stack.enter_context(patch.object(self.setup, name, value))
            for name in ("is_sm100_or_newer", "has_dense_fp4_indexer"):
                stack.enter_context(
                    patch(
                        f"sglang.srt.layers.attention.dsv4.v41_indexer.{name}",
                        lambda: True,
                    )
                )
            self.setup.plan_dsv41_index_logits_budget(runner)

    def test_plans_the_cap_with_the_shipped_headroom(self):
        set_planned_index_logits_budget(GIB)  # a plan from an earlier capture
        set_planned_index_logits_budget(None)
        self._plan(free_gib=9.67)
        self.assertEqual(planned_index_logits_budget(), INDEX_LOGITS_BUDGET_CAP_BYTES)

    def test_refuses_to_start_without_room(self):
        with self.assertRaisesRegex(ValueError, "cannot hold the eager indexer"):
            self._plan(free_gib=4.0)

    def test_leaves_decode_and_other_models_alone(self):
        self._plan(free_gib=0.1, mode="decode")
        self._plan(free_gib=0.1, model_type="deepseek_v4")
        self.assertEqual(planned_index_logits_budget(), INDEX_LOGITS_BUDGET_CAP_BYTES)


class TestExpandIndexPageTable(CustomTestCase):
    @staticmethod
    def _int64_reference(page_table, blocks_per_page):
        bs, n = page_table.shape
        base = page_table.to(torch.int64) * blocks_per_page
        offsets = torch.arange(blocks_per_page, dtype=torch.int64)
        expanded = base.unsqueeze(-1) + offsets
        return expanded.reshape(bs, n * blocks_per_page).to(torch.int32)

    def test_matches_the_int64_expansion_bit_for_bit(self):
        gen = torch.Generator().manual_seed(0)
        for ratio, index_page_size in ((1, 128), (1, 64), (1, 32), (2, 128), (4, 32)):
            table = torch.randint(0, 1 << 29, (6, 9), dtype=torch.int32, generator=gen)
            # Products past int32 keep the same low 32 bits either way.
            table[0, :3] = torch.tensor([0, (1 << 30) - 1, (1 << 31) - 1])
            for page_table in (table, table[:4]):  # publish passes a row slice
                with self.subTest(ratio=ratio, index_page_size=index_page_size):
                    got = expand_index_page_table(
                        page_table,
                        full_page_size=256,
                        compress_ratio=ratio,
                        index_page_size=index_page_size,
                    )
                    blocks_per_page = 256 // ratio // index_page_size
                    want = (
                        page_table
                        if blocks_per_page == 1
                        else self._int64_reference(page_table, blocks_per_page)
                    )
                    self.assertEqual(got.dtype, torch.int32)
                    self.assertTrue(got.is_contiguous())
                    self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
