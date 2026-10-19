"""TBO spec-info split for DFlash/DSPARK verify inputs (dp-attn-todo TBO
Stage 1).

Regression: split_spec_info previously assumed the Eagle field set
(retrieve_*, seq_lens_cpu) and crashed with AttributeError on a
DFlashVerifyInput, blocking two-batch overlap on the DSPARK config outright.
DFlash verify is linear (non-tree). Static mode uses a flat
[bs * draft_token_num] token layout; compact mode packs variable request
widths and therefore resolves the token split from the ragged qo_indptr,
then rebases each child layout. CPU-only.
"""

import unittest

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    compute_split_indices_for_cuda_graph_replay,
    split_spec_info,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DRAFT_W = 9  # a DSPARK speculative_num_draft_tokens value


def _verify_input(bs: int) -> DFlashVerifyInput:
    num_tokens = bs * _DRAFT_W
    return DFlashVerifyInput(
        draft_token=torch.arange(num_tokens, dtype=torch.int64),
        positions=torch.arange(num_tokens, dtype=torch.int64) + 1000,
        draft_token_num=_DRAFT_W,
    )


def _ragged_verify_input(verify_lens: list[int], graph_num_tokens: int):
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens,
        device=torch.device("cpu"),
        grid=[graph_num_tokens],
        graph_num_tokens_floor=graph_num_tokens,
    )
    return DFlashVerifyInput(
        draft_token=torch.arange(graph_num_tokens, dtype=torch.int64),
        positions=torch.arange(graph_num_tokens, dtype=torch.int64) + 1000,
        draft_token_num=_DRAFT_W,
        ragged_verify_layout=layout,
        live_seq_lens_cpu=torch.arange(len(verify_lens), dtype=torch.int64) + 10,
    )


class TestSplitDflashVerifyInput(CustomTestCase):
    def test_slices_token_indexed_fields_and_keeps_scalars(self):
        spec = _verify_input(bs=4)
        # ubatch A = first 2 requests, ubatch B = last 2, split as TBO does:
        # by token index (seq index * draft width).
        child = split_spec_info(
            spec_info=spec,
            start_seq_index=2,
            end_seq_index=4,
            start_token_index=2 * _DRAFT_W,
            end_token_index=4 * _DRAFT_W,
        )
        self.assertEqual(child.draft_token.tolist(), list(range(18, 36)))
        self.assertEqual(child.positions.tolist(), [1000 + i for i in range(18, 36)])
        self.assertEqual(child.draft_token_num, _DRAFT_W)
        self.assertEqual(child.topk, spec.topk)
        self.assertEqual(child.num_tokens_per_req, spec.num_tokens_per_req)
        self.assertEqual(child.capture_hidden_mode, spec.capture_hidden_mode)
        self.assertEqual(child.spec_input_type, spec.spec_input_type)

    def test_full_range_split_preserves_content(self):
        spec = _verify_input(bs=3)
        child = split_spec_info(
            spec_info=spec,
            start_seq_index=0,
            end_seq_index=3,
            start_token_index=0,
            end_token_index=3 * _DRAFT_W,
        )
        self.assertTrue(torch.equal(child.draft_token, spec.draft_token))
        self.assertTrue(torch.equal(child.positions, spec.positions))

    def test_both_children_partition_the_parent(self):
        """The two ubatches must cover every parent token exactly once --
        an off-by-draft-width here is a silent verify-corruption bug, not a
        crash, so pin it explicitly."""
        spec = _verify_input(bs=5)
        split_at = 2
        kwargs = dict(spec_info=spec, start_seq_index=0, end_seq_index=split_at)
        a = split_spec_info(
            **kwargs, start_token_index=0, end_token_index=split_at * _DRAFT_W
        )
        b = split_spec_info(
            spec_info=spec,
            start_seq_index=split_at,
            end_seq_index=5,
            start_token_index=split_at * _DRAFT_W,
            end_token_index=5 * _DRAFT_W,
        )
        recombined = torch.cat([a.draft_token, b.draft_token])
        self.assertTrue(torch.equal(recombined, spec.draft_token))

    def test_none_passthrough(self):
        self.assertIsNone(
            split_spec_info(
                spec_info=None,
                start_seq_index=0,
                end_seq_index=1,
                start_token_index=0,
                end_token_index=_DRAFT_W,
            )
        )

    def test_ragged_children_split_at_request_boundary_and_rebase(self):
        spec = _ragged_verify_input([4, 1, 3, 2], graph_num_tokens=16)
        split_seq, split_token = compute_split_indices_for_cuda_graph_replay(
            forward_mode=ForwardMode.TARGET_VERIFY,
            cuda_graph_num_tokens=16,
            spec_info=spec,
        )
        self.assertEqual((split_seq, split_token), (2, 5))

        a = split_spec_info(
            spec_info=spec,
            start_seq_index=0,
            end_seq_index=split_seq,
            start_token_index=0,
            end_token_index=split_token,
        )
        b = split_spec_info(
            spec_info=spec,
            start_seq_index=split_seq,
            end_seq_index=4,
            start_token_index=split_token,
            # The final child owns the graph-tier ghost tail.
            end_token_index=16,
        )

        self.assertEqual(a.draft_token.tolist(), list(range(5)))
        self.assertEqual(b.draft_token.tolist(), list(range(5, 16)))
        self.assertEqual(a.ragged_verify_layout.verify_lens_cpu, [4, 1])
        self.assertEqual(b.ragged_verify_layout.verify_lens_cpu, [3, 2])
        self.assertEqual(a.ragged_verify_layout.qo_indptr_device.tolist(), [0, 4, 5])
        self.assertEqual(b.ragged_verify_layout.qo_indptr_device.tolist(), [0, 3, 5])
        self.assertEqual(a.ragged_verify_layout.extend_start_loc.tolist(), [0, 4])
        self.assertEqual(b.ragged_verify_layout.extend_start_loc.tolist(), [0, 3])
        self.assertEqual(a.ragged_verify_layout.graph_num_tokens, 5)
        self.assertEqual(b.ragged_verify_layout.graph_num_tokens, 11)
        self.assertEqual(b.ragged_verify_layout.total_verify_tokens, 5)
        self.assertEqual(a.live_seq_lens_cpu.tolist(), [10, 11])
        self.assertEqual(b.live_seq_lens_cpu.tolist(), [12, 13])
        self.assertTrue(
            torch.equal(torch.cat([a.draft_token, b.draft_token]), spec.draft_token)
        )

    def test_ragged_split_supports_device_only_hostless_layout(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([2, 5, 1], dtype=torch.int32),
            graph_num_tokens=12,
        )
        spec = DFlashVerifyInput(
            draft_token=torch.arange(12),
            positions=torch.arange(12),
            draft_token_num=_DRAFT_W,
            ragged_verify_layout=layout,
        )
        child = split_spec_info(
            spec_info=spec,
            start_seq_index=1,
            end_seq_index=3,
            start_token_index=2,
            end_token_index=12,
        )
        self.assertIsNone(child.ragged_verify_layout.verify_lens_cpu)
        self.assertEqual(child.ragged_verify_layout.verify_lens.tolist(), [5, 1])
        self.assertEqual(child.ragged_verify_layout.qo_indptr_device.tolist(), [0, 5, 6])
        self.assertEqual(child.ragged_verify_layout.graph_num_tokens, 10)
        self.assertEqual(child.ragged_verify_layout.total_verify_tokens, 6)

    def test_rejects_custom_mask(self):
        spec = _verify_input(bs=2)
        spec.custom_mask = torch.ones(4, dtype=torch.bool)
        with self.assertRaises(AssertionError):
            split_spec_info(
                spec_info=spec,
                start_seq_index=0,
                end_seq_index=1,
                start_token_index=0,
                end_token_index=_DRAFT_W,
            )


if __name__ == "__main__":
    unittest.main()
