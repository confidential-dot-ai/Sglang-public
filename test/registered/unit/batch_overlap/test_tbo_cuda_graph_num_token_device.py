"""Regression: the TBO cuda-graph plugin must allocate its children
``num_token_non_padded`` buffer on the model device, not CPU.

``ForwardBatch.num_token_non_padded`` is a scalar tensor on the model device
(see ``ForwardBatch.compute``, which does ``.to(device, ...)``). The eager TBO
split path already honors this -- ``compute_tbo_children_num_token_non_padded_raw``
moves the tensor to ``get_device().device`` -- but
``TboCudaGraphRunnerPlugin`` preallocated its persistent buffer with a bare
``torch.zeros((2,), dtype=torch.int32)``, leaving it on CPU.

During TBO cuda-graph replay that CPU buffer becomes the child's
``num_token_non_padded`` and is compared against a device ``arange`` in the
padded-region mask (``_mask_topk_ids_padded_region`` /
``_zero_topk_weights_padded_region``), raising a device-mismatch error
(an illegal memory access on HIP/aiter MoE).

CPU-only: a ``meta`` device makes the buffer placement observable without a GPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboCudaGraphRunnerPlugin,
    TboForwardBatchPreparer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_device
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestTboCudaGraphNumTokenDevice(CustomTestCase):
    def test_plugin_buffer_on_model_device(self):
        # Use 'meta' so the configured device differs from the implicit CPU
        # default; a bare torch.zeros() would leave the buffer on CPU and fail.
        with get_context().override_server_args():
            with get_device().override(device="meta"):
                plugin = TboCudaGraphRunnerPlugin()

        buf = plugin._tbo_children_num_token_non_padded
        self.assertEqual(tuple(buf.shape), (2,))
        self.assertEqual(buf.dtype, torch.int32)
        self.assertEqual(buf.device.type, "meta")

    def test_graph_and_eager_paths_agree_on_device(self):
        # Both the preallocated cuda-graph buffer and the eager split tensor must
        # land on the same (model) device, matching ForwardBatch's contract.
        with get_context().override_server_args():
            with get_device().override(device="meta"):
                eager = TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                    tbo_split_token_index=3, num_token_non_padded=8
                )
                plugin = TboCudaGraphRunnerPlugin()

        self.assertEqual(
            eager.device.type,
            plugin._tbo_children_num_token_non_padded.device.type,
        )

    def test_ragged_ghost_tail_is_not_counted_as_live(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[4, 1, 3, 2],
            device=torch.device("cpu"),
            grid=[16],
            graph_num_tokens_floor=16,
        )
        batch = SimpleNamespace(
            input_ids=torch.arange(16),
            forward_mode=ForwardMode.TARGET_VERIFY,
            tbo_split_seq_index=2,
            extend_seq_lens_cpu=None,
            spec_info=SimpleNamespace(
                draft_token_num=9,
                ragged_verify_layout=layout,
            ),
        )
        with get_context().override_server_args():
            with get_device().override(device="cpu"):
                counts = TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(
                    batch
                )
        # The final child owns 6 ghost tensor rows, but collectives see only
        # the five live query rows on each side of the request boundary.
        self.assertEqual(counts.tolist(), [5, 5])

    def test_prefill_capture_is_deliberately_non_tbo(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
            tbo_split_seq_index=7,
            tbo_children=[object(), object()],
        )
        with get_context().override_server_args(), get_device().override(
            device="cpu"
        ), patch(
            "sglang.srt.batch_overlap.two_batch_overlap.is_tbo_enabled",
            return_value=True,
        ):
            plugin = TboCudaGraphRunnerPlugin()
            enabled = plugin.capture_one_batch_size(batch, num_tokens=32)
        self.assertFalse(enabled)
        self.assertIsNone(batch.tbo_split_seq_index)
        self.assertIsNone(batch.tbo_children)

    def test_compact_decode_capture_is_deliberately_non_tbo(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[4, 1], device=torch.device("cpu"), grid=[8]
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(ragged_verify_layout=layout),
            tbo_split_seq_index=1,
            tbo_children=[object(), object()],
        )
        with get_context().override_server_args(), get_device().override(
            device="cpu"
        ), patch(
            "sglang.srt.batch_overlap.two_batch_overlap.is_tbo_enabled",
            return_value=True,
        ):
            plugin = TboCudaGraphRunnerPlugin()
            enabled = plugin.capture_one_batch_size(batch, num_tokens=8)
        self.assertFalse(enabled)
        self.assertIsNone(batch.tbo_split_seq_index)
        self.assertIsNone(batch.tbo_children)

    def test_static_decode_capture_behavior_is_unchanged(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(
                draft_token_num=9,
                ragged_verify_layout=None,
            ),
            tbo_split_seq_index=None,
        )
        with get_context().override_server_args(), get_device().override(
            device="cpu"
        ), patch(
            "sglang.srt.batch_overlap.two_batch_overlap.is_tbo_enabled",
            return_value=True,
        ), patch.object(
            TboForwardBatchPreparer,
            "compute_tbo_children_num_token_non_padded",
            return_value=torch.tensor([9, 9], dtype=torch.int32),
        ), patch.object(TboForwardBatchPreparer, "prepare_raw") as prepare_raw:
            plugin = TboCudaGraphRunnerPlugin()
            enabled = plugin.capture_one_batch_size(batch, num_tokens=18)
        self.assertTrue(enabled)
        self.assertEqual(batch.tbo_split_seq_index, 1)
        prepare_raw.assert_called_once()


if __name__ == "__main__":
    unittest.main()
