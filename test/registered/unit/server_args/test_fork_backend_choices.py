"""CLI choices for backends this fork wires in.

A rebase that moved the choice lists into `arg_groups/choices.py` dropped an
entry whose backend stayed implemented, so argparse rejected every launcher
that passed it. These tests fail if an entry goes missing again, or if a
listed backend loses its implementation.
"""

import unittest

from sglang.srt.arg_groups.choices import (
    DISAGG_TRANSFER_BACKEND_CHOICES,
    GRAMMAR_BACKEND_CHOICES,
)
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestForkBackendChoices(unittest.TestCase):
    def test_every_transfer_choice_has_a_backend(self):
        # pd_disaggregation_hook rewrites the alias before TransferBackend(...).
        aliases = {"mooncake_tcp": "mooncake"}
        backends = {b.value for b in TransferBackend}
        for choice in DISAGG_TRANSFER_BACKEND_CHOICES:
            self.assertIn(aliases.get(choice, choice), backends)

    def test_device_fsm_is_a_grammar_backend_choice(self):
        self.assertIn("device_fsm", GRAMMAR_BACKEND_CHOICES)


if __name__ == "__main__":
    unittest.main()
