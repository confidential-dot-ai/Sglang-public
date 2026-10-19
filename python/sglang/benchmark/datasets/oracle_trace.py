import json
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class OracleTraceDataset(BaseDataset):
    """Multi-turn oracle-mode replay of a real agent trajectory dataset.

    Each turn forces the server to emit the trajectory's real recorded
    completion (``forced_token_ids``, see ``SamplingParams.forced_token_ids``)
    instead of sampling one, so throughput/TTFT/ITL/prefix-cache numbers
    reflect a real agentic trace's shape (turn count, output length,
    context growth) rather than a synthetic one-shot benchmark. Requires a
    server built with oracle-mode support and ``--backend sglang-oai-chat``.

    Expects the combined JSON produced by
    ``benchmark/oracle_replay/build_trace.py --combined-output``::

        {
          "metadata": {...},
          "conversations": [
            [   # one conversation == a list of turns
              {"messages": [...new non-assistant messages...],
               "forced_token_ids": [...], "max_new_tokens": N},
              ...
            ],
            ...
          ]
        }

    Each turn's ``messages`` holds only the new non-assistant messages for
    that turn (the previous round's assistant reply is filled in by
    ``wrap_multi_turn_request_func`` from the server's own real output --
    which, since it's forced, is exactly the trajectory's real completion).
    """

    dataset_path: str
    num_requests: int
    offset: int
    max_turns: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "OracleTraceDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            offset=args.dataset_offset,
            # Shared with agentic-trace: same "cap conversation length" knob.
            max_turns=args.agentic_max_turns,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = data.get("conversations", [])
        if not conversations:
            raise ValueError(f"No 'conversations' found in {self.dataset_path}.")

        offset = self.offset % len(conversations)
        if offset:
            conversations = conversations[offset:] + conversations[:offset]

        filtered_dataset: List[DatasetRow] = []
        for conversation in conversations:
            if self.num_requests > 0 and len(filtered_dataset) >= self.num_requests:
                break

            rounds = []
            for turn in conversation:
                if not turn.get("messages") or not turn.get("forced_token_ids"):
                    continue
                round_max_tokens = turn.get(
                    "max_new_tokens", len(turn["forced_token_ids"]) + 8
                )
                rounds.append(
                    {
                        "messages": turn["messages"],
                        "extra_request_body": {
                            "forced_token_ids": turn["forced_token_ids"],
                            # Wire fields, not "max_new_tokens" (that's
                            # SamplingParams' internal field name -- also
                            # the trace JSON's own schema key above, but not
                            # a real ChatCompletionRequest field, so it's
                            # silently dropped on the wire). protocol.py
                            # resolves `max_new_tokens = max_completion_tokens
                            # or max_tokens` -- async_request_openai_chat_
                            # completions (the function backing
                            # --backend sglang-oai-chat, the only backend
                            # this dataset supports) sets a run-level
                            # "max_completion_tokens": output_len default
                            # BEFORE merging this dict in, so plain
                            # "max_tokens" here would be silently outranked
                            # by that default every round. Set both so the
                            # real forced length wins regardless of which
                            # key a given backend path actually reads.
                            "max_completion_tokens": round_max_tokens,
                            "max_tokens": round_max_tokens,
                        },
                    }
                )
            if self.max_turns:
                rounds = rounds[: self.max_turns]
            if not rounds:
                continue

            total_output_len = sum(
                len(r["extra_request_body"]["forced_token_ids"]) for r in rounds
            )
            filtered_dataset.append(
                DatasetRow(
                    prompt=rounds,
                    # Informational only, like agentic-trace: multi-turn
                    # replay ignores this and reads real per-round lengths.
                    prompt_len=0,
                    output_len=total_output_len,
                )
            )

        if not filtered_dataset:
            raise ValueError(
                f"No usable conversations loaded from {self.dataset_path}."
            )

        num_turns = [len(row.prompt) for row in filtered_dataset]
        total_forced = sum(row.output_len for row in filtered_dataset)
        print(
            f"#Conversations: {len(filtered_dataset)} "
            f"(offset={offset}, turns/conv min={min(num_turns)} "
            f"max={max(num_turns)} avg={np.mean(num_turns):.1f})"
        )
        print(f"#Forced tokens total: {total_forced}")
        return filtered_dataset
