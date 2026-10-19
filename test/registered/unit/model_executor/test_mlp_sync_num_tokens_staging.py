"""Persistent-staging copy of the MLP-sync per-rank token counts
(copy_mlp_sync_num_tokens_to_gpu): steady-state allocation-free replacement
for the previous per-step ``torch.tensor(values, pin_memory=True)`` upload,
which paid a cudaHostAlloc on every DP-attention step (and under confidential
computing, pinned DMA bounces through encrypted staging anyway).
"""

import torch

from sglang.srt.model_executor.forward_batch_info import (
    copy_mlp_sync_num_tokens_to_gpu,
)
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_STAGING_NAME = "mlp_sync_num_tokens_staging"


class TestCopyMlpSyncNumTokensToGpu(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_copies_values_exactly(self):
        # A CPU dst stands in for the GPU tensor: copy_ semantics are the same.
        dst = torch.zeros(4, dtype=torch.int64)
        copy_mlp_sync_num_tokens_to_gpu(dst, [3, 1, 4, 1])
        self.assertEqual(dst.tolist(), [3, 1, 4, 1])

    def test_staging_buffer_is_reused_across_steps(self):
        dst = torch.zeros(4, dtype=torch.int64)
        copy_mlp_sync_num_tokens_to_gpu(dst, [1, 2, 3, 4])
        first = get_context().resources.buffers[_STAGING_NAME]
        copy_mlp_sync_num_tokens_to_gpu(dst, [5, 6, 7, 8])
        second = get_context().resources.buffers[_STAGING_NAME]
        self.assertIs(first, second, "staging must be leased once, not per step")
        self.assertEqual(dst.tolist(), [5, 6, 7, 8])

    def test_staging_is_pageable_not_pinned(self):
        """The whole point: no cudaHostAlloc. A pageable source also keeps
        non_blocking reuse safe (CUDA stages pageable sources before the
        copy call returns); a reused pinned source would need an event."""
        dst = torch.zeros(2, dtype=torch.int64)
        copy_mlp_sync_num_tokens_to_gpu(dst, [7, 9])
        staging = get_context().resources.buffers[_STAGING_NAME]
        self.assertFalse(staging.is_pinned() if torch.cuda.is_available() else False)
        self.assertEqual(staging.device.type, "cpu")

    def test_rejects_growing_sync_group(self):
        """The sync group size is config-constant; a larger later caller
        means two differently-sized subsystems share the lease -- fail
        loudly instead of silently truncating."""
        dst2 = torch.zeros(2, dtype=torch.int64)
        copy_mlp_sync_num_tokens_to_gpu(dst2, [1, 2])
        dst4 = torch.zeros(4, dtype=torch.int64)
        with self.assertRaises(AssertionError):
            copy_mlp_sync_num_tokens_to_gpu(dst4, [1, 2, 3, 4])

    def test_shorter_group_uses_prefix_slice(self):
        """A smaller-than-lease call (e.g. logprob twin of length 1) copies
        through the buffer prefix without disturbing dst semantics."""
        dst4 = torch.zeros(4, dtype=torch.int64)
        copy_mlp_sync_num_tokens_to_gpu(dst4, [9, 9, 9, 9])
        dst1 = torch.zeros(1, dtype=torch.int64)
        copy_mlp_sync_num_tokens_to_gpu(dst1, [42])
        self.assertEqual(dst1.tolist(), [42])


if __name__ == "__main__":
    import unittest

    unittest.main()
