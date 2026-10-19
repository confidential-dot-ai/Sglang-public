"""Unit tests for validate_dflash_request's grammar admission rules -- no
server, no model loading, no GPU.

Grammar-constrained requests may enter DFlash only under
grammar_backend='device_fsm' (the one backend whose grammars DFlash enforces
in its verify step); under host-CPU backends the verify path would silently
ignore the grammar, so those requests must be rejected at admission. Oracle
mode (forced_token_ids) cannot combine with grammar: forced tokens come from
a recorded trace and have no reason to satisfy the grammar.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

from types import SimpleNamespace

from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.test.test_utils import CustomTestCase


def _make_req(
    json_schema=None, regex=None, ebnf=None, structural_tag=None, forced_token_ids=None
):
    return SimpleNamespace(
        return_logprob=False,
        return_hidden_states=False,
        sampling_params=SimpleNamespace(
            json_schema=json_schema,
            regex=regex,
            ebnf=ebnf,
            structural_tag=structural_tag,
            forced_token_ids=forced_token_ids,
        ),
    )


class TestDflashGrammarAdmission(CustomTestCase):
    def test_plain_request_passes_regardless_of_backend(self):
        for backend in (None, "xgrammar", "device_fsm"):
            self.assertIsNone(validate_dflash_request(_make_req(), False, backend))

    def test_grammar_rejected_under_host_backends(self):
        for backend in (None, "xgrammar", "outlines", "llguidance", "none"):
            reason = validate_dflash_request(_make_req(regex="a+"), False, backend)
            self.assertIsNotNone(reason)
            self.assertIn("device_fsm", reason)

    def test_grammar_admitted_under_device_fsm(self):
        for kwargs in (
            {"regex": "a+"},
            {"json_schema": '{"type": "object"}'},
            # EBNF is admitted here but device_fsm's dispatcher returns
            # InvalidGrammarObject for it, aborting the request visibly at
            # grammar-compile time rather than generating unconstrained.
            {"ebnf": "root ::= 'x'"},
        ):
            self.assertIsNone(
                validate_dflash_request(_make_req(**kwargs), False, "device_fsm")
            )

    def test_oracle_plus_grammar_rejected(self):
        reason = validate_dflash_request(
            _make_req(regex="a+", forced_token_ids=[1, 2, 3]), False, "device_fsm"
        )
        self.assertIsNotNone(reason)
        self.assertIn("forced_token_ids", reason)

    def test_oracle_without_grammar_still_admitted(self):
        self.assertIsNone(
            validate_dflash_request(
                _make_req(forced_token_ids=[1, 2, 3]), False, "device_fsm"
            )
        )
