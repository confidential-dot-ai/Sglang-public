"""DSpark masks its verify logits with the request's grammar, so tool_choice
required/named and response_format requests must reach it rather than be
rejected; DFlash's verify would ignore the grammar, so it keeps rejecting them."""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _req(**constraint):
    params = dict(json_schema=None, regex=None, ebnf=None, structural_tag=None)
    params.update(constraint)
    return SimpleNamespace(
        return_logprob=False,
        return_hidden_states=False,
        sampling_params=SimpleNamespace(**params),
    )


class TestDflashGrammarAdmission(unittest.TestCase):
    def test_dspark_admits_every_grammar_kind_and_dflash_rejects_them(self):
        constraints = {
            "json_schema": '{"type": "object"}',
            "regex": "[0-9]+",
            "ebnf": 'root ::= "a"',
            "structural_tag": '{"format": {}}',
        }
        for kind, value in constraints.items():
            with self.subTest(kind):
                req = _req(**{kind: value})
                self.assertIsNone(
                    validate_dflash_request(
                        req, False, "xgrammar", masks_grammar_in_verify=True
                    )
                )
                self.assertIn(
                    "grammar-constrained",
                    validate_dflash_request(req, False, "xgrammar"),
                )


if __name__ == "__main__":
    unittest.main()
