"""Regression test for OracleTraceDataset.load()'s per-round extra_request_body.

protocol.py resolves `max_new_tokens = max_completion_tokens or max_tokens`.
async_request_openai_chat_completions (the function backing
--backend sglang-oai-chat, the only backend this dataset supports) sets a
run-level "max_completion_tokens": output_len default before merging each
round's extra_request_body on top. "max_new_tokens" -- SamplingParams' own
internal field name, and also the trace JSON's own schema key (see
build_trace.py) -- is not a real ChatCompletionRequest field at all, so
setting it in the wire body is silently dropped; a plain "max_tokens" is
also insufficient since max_completion_tokens outranks it whenever both are
present. Caught live: every round with more forced tokens than the run-level
default 400'd with "forced_token_ids has N tokens, which exceeds
max_new_tokens=<default>" -- only rounds that happened to fit under it
succeeded, silently.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sglang.benchmark.datasets.oracle_trace import OracleTraceDataset
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

from sglang.test.test_utils import CustomTestCase


def _write_combined(tmp_path: Path, conversations) -> str:
    path = tmp_path / "combined.json"
    path.write_text(json.dumps({"metadata": {}, "conversations": conversations}))
    return str(path)


class TestOracleTraceDataset(CustomTestCase):
    def test_extra_request_body_uses_wire_max_tokens_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_combined(
                Path(tmp),
                [
                    [
                        {
                            "messages": [{"role": "user", "content": "hi"}],
                            "forced_token_ids": list(range(300)),
                            "max_new_tokens": 308,
                        }
                    ]
                ],
            )
            args = SimpleNamespace(
                dataset_path=dataset_path,
                num_prompts=1,
                dataset_offset=0,
                agentic_max_turns=None,
            )
            rows = OracleTraceDataset.from_args(args).load(tokenizer=None)
            round0 = rows[0].prompt[0]
            body = round0["extra_request_body"]
            self.assertEqual(body["max_completion_tokens"], 308)
            self.assertEqual(body["max_tokens"], 308)
            self.assertNotIn(
                "max_new_tokens",
                body,
                "must not send the internal field name on the wire -- it's "
                "not a real ChatCompletionRequest field, so it's silently "
                "dropped, leaving the round at serving.py's fixed run-level "
                "max_completion_tokens default",
            )

    def test_max_completion_tokens_falls_back_to_forced_len_plus_slack(self):
        """When a round's trace JSON omits max_new_tokens (older traces),
        max_completion_tokens must still be derived from the real forced
        length, not left at whatever serving.py's run-level default is."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = _write_combined(
                Path(tmp),
                [
                    [
                        {
                            "messages": [{"role": "user", "content": "hi"}],
                            "forced_token_ids": [1, 2, 3, 4, 5],
                        }
                    ]
                ],
            )
            args = SimpleNamespace(
                dataset_path=dataset_path,
                num_prompts=1,
                dataset_offset=0,
                agentic_max_turns=None,
            )
            rows = OracleTraceDataset.from_args(args).load(tokenizer=None)
            body = rows[0].prompt[0]["extra_request_body"]
            self.assertEqual(body["max_completion_tokens"], 5 + 8)
            self.assertEqual(body["max_tokens"], 5 + 8)
