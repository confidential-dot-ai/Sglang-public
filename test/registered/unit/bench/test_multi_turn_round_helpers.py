"""Unit tests for the multi-turn round helpers in sglang/benchmark/serving.py
added for oracle-mode replay (--dataset-name oracle-trace) -- no server, no
model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

import unittest

from sglang.benchmark.serving import (
    _round_messages_and_extra_body,
    _strip_reasoning_for_history,
)
from sglang.test.test_utils import CustomTestCase


class TestRoundMessagesAndExtraBody(CustomTestCase):
    def test_str_round_unchanged(self):
        normalized, extra = _round_messages_and_extra_body("hello")
        self.assertEqual(normalized, [{"role": "user", "content": "hello"}])
        self.assertEqual(extra, {})

    def test_message_list_round_unchanged(self):
        turn = [{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}]
        normalized, extra = _round_messages_and_extra_body(turn)
        self.assertEqual(normalized, turn)
        self.assertEqual(extra, {})

    def test_oracle_round_dict_extracts_messages_and_extra_body(self):
        turn = {
            "messages": [{"role": "user", "content": "tool output"}],
            "extra_request_body": {"forced_token_ids": [1, 2, 3], "max_new_tokens": 11},
        }
        normalized, extra = _round_messages_and_extra_body(turn)
        self.assertEqual(normalized, [{"role": "user", "content": "tool output"}])
        self.assertEqual(extra, {"forced_token_ids": [1, 2, 3], "max_new_tokens": 11})

    def test_oracle_round_dict_without_extra_request_body_defaults_empty(self):
        turn = {"messages": [{"role": "user", "content": "hi"}]}
        normalized, extra = _round_messages_and_extra_body(turn)
        self.assertIsNotNone(normalized)
        self.assertEqual(extra, {})

    def test_single_message_dict_is_not_a_round(self):
        # A single OpenAI message dict ({"role":..., "content":...}) must NOT
        # be mistaken for a round -- it lacks "messages", so single-shot
        # List[Dict] prompts are still correctly excluded from multi-turn
        # detection (see the is_multi_turn check in serving.py).
        normalized, extra = _round_messages_and_extra_body(
            {"role": "user", "content": "hi"}
        )
        self.assertIsNone(normalized)
        self.assertEqual(extra, {})

    def test_invalid_shape_returns_none(self):
        normalized, extra = _round_messages_and_extra_body(12345)
        self.assertIsNone(normalized)
        self.assertEqual(extra, {})


class TestStripReasoningForHistory(CustomTestCase):
    def test_no_think_tag_is_unchanged(self):
        text = "just a plain final answer"
        self.assertEqual(_strip_reasoning_for_history(text), text)

    def test_strips_everything_through_last_think_tag(self):
        text = "reasoning about the task\n</think>\n\nfinal visible answer"
        self.assertEqual(_strip_reasoning_for_history(text), "final visible answer")

    def test_uses_last_think_tag_if_multiple(self):
        text = "outer</think>middle</think>final"
        self.assertEqual(_strip_reasoning_for_history(text), "final")


if __name__ == "__main__":
    unittest.main()
