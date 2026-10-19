"""The DeepSeek-V4.1 prefill-side tail compares PP stages by per-token KV width."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV41PrefillTailLayout(CustomTestCase):
    def test_signature_normalizes_ratio_one_and_two_pages(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        common = dict(
            mla_compression_ratios=[2, 2, 2, 1],
            page_size=256,
            state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE],
        )
        # Stage 0 owns only ratio-2 sources; stage 1 also owns the ratio-1 one,
        # so their full-page byte counts legitimately differ.
        first = SimpleNamespace(
            **common,
            kv_layer_ids=[0, 1, 0, 1],
            kv_item_lens=[81920, 81920, 8448, 8448],
            state_item_lens=[[640, 640], [2048, 2048]],
        )
        last = SimpleNamespace(
            **common,
            kv_layer_ids=[3, 3, 2, 2],
            kv_item_lens=[163840, 16896, 81920, 8448],
            state_item_lens=[[640, 640, 1280], [2048]],
        )
        with (
            envs.SGLANG_OPT_DSV41_PREFILL_TAIL.override(True),
            get_context().override_server_args(
                speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
            ),
        ):
            layout = get_dsv41_spec_layout(first)
            self.assertEqual(layout, get_dsv41_spec_layout(last))
            self.assertEqual(layout["kv_token_item_sizes"], [66, 640])
            last.kv_item_lens[0] *= 2
            self.assertNotEqual(layout, get_dsv41_spec_layout(last))


if __name__ == "__main__":
    unittest.main()
