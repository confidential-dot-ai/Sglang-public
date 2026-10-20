"""async_h2d never synchronizes, including with the pageable staging that
Confidential Computing forces: a blocking pageable upload waits for every
kernel already queued on the stream (~200 ms behind a busy prefill step)."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import common
from sglang.srt.utils.common import async_h2d, async_h2d_stack
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestAsyncH2D(CustomTestCase):
    def _assert_no_sync(self, pin: bool):
        device = torch.device("cuda")
        values = list(range(-3, 61))
        rows = [torch.arange(8, dtype=torch.int32) * i for i in range(4)]
        busy = torch.randn(4096, 4096, device=device)
        with patch.object(common, "is_pin_memory_available", lambda device=None: pin):
            busy @ busy  # queued work a blocking copy would wait for
            torch.cuda.set_sync_debug_mode("error")
            try:
                got = async_h2d(values, dtype=torch.int64, device=device)
                stacked = async_h2d_stack(rows, device=device)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        self.assertEqual(got.tolist(), values)
        self.assertTrue(torch.equal(stacked.cpu(), torch.stack(rows)))

    def test_pageable_staging_does_not_synchronize(self):
        self._assert_no_sync(pin=False)

    def test_pinned_staging_does_not_synchronize(self):
        self._assert_no_sync(pin=True)


if __name__ == "__main__":
    unittest.main()
