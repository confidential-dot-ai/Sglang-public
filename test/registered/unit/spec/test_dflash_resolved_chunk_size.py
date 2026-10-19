"""Draft hidden-state injection honors the resolved scheduler token budget."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.runtime_context import get_context, get_schedule, get_server_args
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashResolvedChunkSize(CustomTestCase):
    def test_auto_chunk_budget_bounds_injection_and_preserves_rows(self):
        with (
            get_context().override_server_args(
                chunked_prefill_size=None, max_prefill_tokens=7
            ),
            get_schedule().override(chunked_prefill_size=3),
        ):
            chunks = []
            worker = SimpleNamespace(
                server_args=get_server_args(),
                _append_target_hidden_to_draft_kv_by_loc_chunk=lambda **kw: chunks.append(
                    kw
                ),
            )
            hidden = torch.arange(14).view(7, 2)
            locations = torch.arange(7) + 10
            positions = torch.arange(7)
            DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(
                worker,
                target_hidden=hidden,
                cache_loc=locations,
                positions=positions,
            )
            self.assertEqual([c["target_hidden"].shape[0] for c in chunks], [3, 3, 1])
            for key, expected in (
                ("target_hidden", hidden),
                ("cache_loc", locations),
                ("positions", positions),
            ):
                torch.testing.assert_close(
                    torch.cat([c[key] for c in chunks]), expected
                )


if __name__ == "__main__":
    unittest.main()
