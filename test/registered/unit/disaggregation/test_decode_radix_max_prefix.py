"""Decode-radix prefix cap: page aligned, never the whole prompt, and under SWA
ending before the window prefill sends fresh (the CED replay tail included).

    python -m pytest test/registered/unit/disaggregation/test_decode_radix_max_prefix.py -v
"""

import unittest

from sglang.srt.disaggregation.decode import decode_radix_max_prefix
from sglang.srt.disaggregation.utils import ced_tail_start
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE, WINDOW = 256, 128


class TestDecodeRadixMaxPrefix(CustomTestCase):
    def test_without_swa_stops_one_page_short_of_a_page_aligned_prompt(self):
        self.assertEqual(decode_radix_max_prefix(16 * PAGE, PAGE, None), 15 * PAGE)
        self.assertEqual(decode_radix_max_prefix(16 * PAGE + 7, PAGE, None), 16 * PAGE)

    def test_swa_window_is_never_cached(self):
        for prompt_len in (1, PAGE, 37_000, 1_048_576):
            cap = decode_radix_max_prefix(prompt_len, PAGE, WINDOW)
            self.assertEqual(cap % PAGE, 0)
            self.assertLessEqual(cap, max(0, prompt_len - WINDOW))

    def test_ced_replay_tail_starts_at_or_after_the_cap(self):
        with envs.SGLANG_OPT_DSV41_CED_SPLIT.override(True):
            for prompt_len in (PAGE + 1, 37_000, 40_960, 1_048_576):
                cap = decode_radix_max_prefix(prompt_len, PAGE, WINDOW)
                self.assertLessEqual(cap, ced_tail_start(prompt_len, WINDOW, PAGE))
                # The window prefill sends (2 * window + page) is not cached.
                self.assertLessEqual(cap, max(0, prompt_len - (2 * WINDOW + PAGE)))


if __name__ == "__main__":
    unittest.main()
