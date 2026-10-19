#!/usr/bin/env python3
"""Build explicitly modeled long-conversation oracle replay datasets.

The source oracle corpus contains real agent turns, but most conversations end
before a short throughput run reaches a deep steady state.  This transform:

1. deterministically stitches real continuation deltas until every conversation
   has the requested number of live turns;
2. extends the rendered system instruction with a unique, stable project-history
   archive to place each conversation in the requested served-token band; and
3. resizes each forced completion to preserve the requested output-token share.

The result is useful for a *modeled-mix* capacity experiment.  It is synthetic
and must never be reported as an unmodified production trace.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


class TokenCounter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.cache: dict[str, int] = {}

    def text(self, value: str) -> int:
        value = str(value or "")
        cached = self.cache.get(value)
        if cached is None:
            cached = len(self.tokenizer.encode(value, add_special_tokens=False))
            self.cache[value] = cached
        return cached

    def messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.text(message.get("content") or "") for message in messages)


def continuation_pool(
    conversations: list[list[dict[str, Any]]],
) -> list[tuple[int, int, dict[str, Any]]]:
    pool = []
    for conversation_index, conversation in enumerate(conversations):
        for turn_index, turn in enumerate(conversation[1:], 1):
            if turn.get("forced_token_ids"):
                pool.append((conversation_index, turn_index, turn))
    if not pool:
        raise ValueError("source dataset has no continuation turns")
    return pool


def extend_conversation(
    conversation: list[dict[str, Any]],
    *,
    conversation_index: int,
    pool: list[tuple[int, int, dict[str, Any]]],
    target_turns: int,
    seed: int,
) -> list[dict[str, Any]]:
    turns = copy.deepcopy(conversation[:target_turns])
    rng = random.Random(seed + conversation_index * 1_000_003)
    while len(turns) < target_turns:
        donor_conversation, donor_turn, donor = pool[rng.randrange(len(pool))]
        turn = copy.deepcopy(donor)
        messages = turn.setdefault("messages", [])
        marker = (
            f"[modeled continuation: source conversation {donor_conversation}, "
            f"turn {donor_turn}]\n"
        )
        target_message = next(
            (
                message
                for message in reversed(messages)
                if message.get("role") != "assistant"
            ),
            None,
        )
        if target_message is None:
            messages.append({"role": "user", "content": marker.rstrip()})
        else:
            target_message["content"] = marker + str(
                target_message.get("content") or ""
            )
        turns.append(turn)
    return turns


def fixed_delta_lengths(
    conversation: list[dict[str, Any]], counter: TokenCounter
) -> list[int]:
    lengths = []
    for turn_index, turn in enumerate(conversation):
        messages = turn.get("messages") or []
        # The realmix files explicitly carry the preceding assistant message
        # at the start of continuation turns.  It is rewritten to the resized
        # preceding completion below, so exclude its old length here.
        start = (
            1
            if turn_index and messages and messages[0].get("role") == "assistant"
            else 0
        )
        lengths.append(counter.messages(messages[start:]))
    return lengths


def simulate_lengths(
    fixed_lengths: list[int],
    *,
    archive_tokens: int,
    output_share: float,
    min_output_tokens: int,
    max_output_tokens: int,
) -> tuple[list[int], list[int]]:
    prompts = []
    outputs = []
    history_tokens = archive_tokens
    previous_output = 0
    output_ratio = output_share / (1.0 - output_share)
    for turn_index, fixed_length in enumerate(fixed_lengths):
        if turn_index:
            # concurrent_replay appends the generated assistant reply, while
            # realmix also carries that reply explicitly in the next delta.
            history_tokens += previous_output
        history_tokens += fixed_length
        prompt_tokens = history_tokens
        output_tokens = round(prompt_tokens * output_ratio)
        output_tokens = max(
            min_output_tokens, min(max_output_tokens, output_tokens)
        )
        prompts.append(prompt_tokens)
        outputs.append(output_tokens)
        history_tokens += output_tokens
        previous_output = output_tokens
    return prompts, outputs


def choose_archive_tokens(
    fixed_lengths: list[int],
    *,
    target_served_tokens: int,
    output_share: float,
    min_output_tokens: int,
    max_output_tokens: int,
) -> int:
    def distance(archive_tokens: int) -> float:
        prompts, outputs = simulate_lengths(
            fixed_lengths,
            archive_tokens=archive_tokens,
            output_share=output_share,
            min_output_tokens=min_output_tokens,
            max_output_tokens=max_output_tokens,
        )
        return statistics.mean(p + o for p, o in zip(prompts, outputs))

    if distance(0) >= target_served_tokens:
        return 0
    low, high = 0, target_served_tokens
    while distance(high) < target_served_tokens:
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if distance(middle) < target_served_tokens:
            low = middle
        else:
            high = middle
    return min((low, high), key=lambda value: abs(distance(value) - target_served_tokens))


_ARCHIVE_SENTENCES = (
    "The scheduler records each queue transition before committing the next batch. ",
    "The repository history preserves test failures, patches, and validation notes. ",
    "The cache manifest maps every retained prefix to its owning project session. ",
    "The implementation log describes retries, checkpoints, and measured regressions. ",
    "The review ledger keeps interface contracts and operational constraints explicit. ",
    "The trace archive links each benchmark decision to reproducible source evidence. ",
)


def make_archive_text(
    tokenizer: Any, requested_tokens: int, conversation_index: int
) -> tuple[str, int]:
    if requested_tokens <= 0:
        return "", 0
    ids = tokenizer.encode(
        f"MODELED PROJECT ARCHIVE {conversation_index:04d}. ",
        add_special_tokens=False,
    )
    sentence_index = conversation_index
    while len(ids) < requested_tokens:
        sentence = (
            f"Entry {conversation_index:04d}-{sentence_index:05d}: "
            + _ARCHIVE_SENTENCES[sentence_index % len(_ARCHIVE_SENTENCES)]
        )
        ids.extend(tokenizer.encode(sentence, add_special_tokens=False))
        sentence_index += 1
    text = tokenizer.decode(ids[:requested_tokens], skip_special_tokens=True)
    actual_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return text, actual_tokens


def attach_archive(
    messages: list[dict[str, Any]], archive_text: str
) -> str:
    """Attach modeled history without displacing the rendered system prompt.

    Some chat templates render only the first system/developer message.  Adding
    a new system message at index zero can therefore hide the source trace's
    real instruction block instead of extending it.  Append to the first
    existing instruction message so both pieces survive those templates and
    the source instruction remains a shareable radix prefix across sessions.
    """

    archive = (
        "The following project archive is synthetic and exists only for a "
        "modeled long-context capacity experiment.\n\n" + archive_text
    )
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            existing = str(message.get("content") or "")
            message["content"] = (existing + "\n\n" if existing else "") + archive
            return "merged_into_existing_instruction"

    messages.insert(0, {"role": "system", "content": archive})
    return "inserted_new_system_fallback"


_OUTPUT_SENTENCES = (
    "I will preserve the current contract and validate the next change against the recorded evidence. ",
    "The next implementation step is scoped to the measured bottleneck and keeps unrelated state unchanged. ",
    "After applying the change, I will compare correctness, throughput, and failure counters before proceeding. ",
    "The handoff will distinguish observed production-shaped results from explicitly modeled traffic. ",
)


def resize_forced_tokens(
    tokenizer: Any,
    original: list[int],
    desired_tokens: int,
    *,
    conversation_index: int,
    turn_index: int,
) -> list[int]:
    if len(original) >= desired_tokens:
        return list(original[:desired_tokens])
    result = list(original)
    extension_index = 0
    while len(result) < desired_tokens:
        text = (
            f"\nModeled continuation {conversation_index:04d}-{turn_index:02d}-"
            f"{extension_index:03d}. "
            + _OUTPUT_SENTENCES[extension_index % len(_OUTPUT_SENTENCES)]
        )
        result.extend(tokenizer.encode(text, add_special_tokens=False))
        extension_index += 1
    return result[:desired_tokens]


def materialize_conversation(
    tokenizer: Any,
    counter: TokenCounter,
    conversation: list[dict[str, Any]],
    *,
    conversation_index: int,
    archive_tokens: int,
    output_share: float,
    min_output_tokens: int,
    max_output_tokens: int,
) -> tuple[list[int], list[int], int, str]:
    archive_text, actual_archive_tokens = make_archive_text(
        tokenizer, archive_tokens, conversation_index
    )
    archive_placement = "none"
    if archive_text:
        archive_placement = attach_archive(
            conversation[0].setdefault("messages", []), archive_text
        )

    prompts = []
    outputs = []
    history_tokens = 0
    previous_forced: list[int] | None = None
    output_ratio = output_share / (1.0 - output_share)
    for turn_index, turn in enumerate(conversation):
        messages = turn.setdefault("messages", [])
        if previous_forced is not None:
            previous_text = tokenizer.decode(
                previous_forced, skip_special_tokens=True
            )
            if messages and messages[0].get("role") == "assistant":
                messages[0]["content"] = previous_text
            else:
                messages.insert(
                    0, {"role": "assistant", "content": previous_text}
                )
        history_tokens += counter.messages(messages)
        prompt_tokens = history_tokens
        desired_output = round(prompt_tokens * output_ratio)
        desired_output = max(
            min_output_tokens, min(max_output_tokens, desired_output)
        )
        forced = resize_forced_tokens(
            tokenizer,
            list(turn.get("forced_token_ids") or []),
            desired_output,
            conversation_index=conversation_index,
            turn_index=turn_index,
        )
        turn["forced_token_ids"] = forced
        turn["max_new_tokens"] = len(forced) + 8
        prompts.append(prompt_tokens)
        outputs.append(len(forced))
        history_tokens += len(forced)
        previous_forced = forced
    return prompts, outputs, actual_archive_tokens, archive_placement


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-turns", type=int, default=25)
    parser.add_argument("--target-served-tokens", type=int, required=True)
    parser.add_argument("--output-share", type=float, default=0.0075)
    parser.add_argument("--min-output-tokens", type=int, default=18)
    parser.add_argument("--max-output-tokens", type=int, default=771)
    parser.add_argument("--num-conversations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1323)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.target_turns < 2:
        raise ValueError("--target-turns must be at least 2")
    if args.target_served_tokens <= 0:
        raise ValueError("--target-served-tokens must be positive")
    if not 0 < args.output_share < 1:
        raise ValueError("--output-share must be between 0 and 1")
    if not 0 < args.min_output_tokens <= args.max_output_tokens:
        raise ValueError("invalid output-token bounds")

    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}")

    with source_path.open() as file:
        source_data = json.load(file)
    source_conversations = source_data["conversations"]
    if args.num_conversations:
        source_conversations = source_conversations[: args.num_conversations]
    if not source_conversations:
        raise ValueError("source dataset has no conversations")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    counter = TokenCounter(tokenizer)
    pool = continuation_pool(source_data["conversations"])

    output_conversations = []
    all_prompts: list[int] = []
    all_outputs: list[int] = []
    archive_sizes = []
    archive_placements = []
    for conversation_index, source_conversation in enumerate(source_conversations):
        conversation = extend_conversation(
            source_conversation,
            conversation_index=conversation_index,
            pool=pool,
            target_turns=args.target_turns,
            seed=args.seed,
        )
        fixed_lengths = fixed_delta_lengths(conversation, counter)
        archive_tokens = choose_archive_tokens(
            fixed_lengths,
            target_served_tokens=args.target_served_tokens,
            output_share=args.output_share,
            min_output_tokens=args.min_output_tokens,
            max_output_tokens=args.max_output_tokens,
        )
        prompts, outputs, actual_archive_tokens, archive_placement = (
            materialize_conversation(
                tokenizer,
                counter,
                conversation,
                conversation_index=conversation_index,
                archive_tokens=archive_tokens,
                output_share=args.output_share,
                min_output_tokens=args.min_output_tokens,
                max_output_tokens=args.max_output_tokens,
            )
        )
        output_conversations.append(conversation)
        all_prompts.extend(prompts)
        all_outputs.extend(outputs)
        archive_sizes.append(actual_archive_tokens)
        archive_placements.append(archive_placement)
        if (conversation_index + 1) % 100 == 0:
            print(
                f"built {conversation_index + 1}/{len(source_conversations)} conversations",
                flush=True,
            )

    served = [
        prompt + output for prompt, output in zip(all_prompts, all_outputs)
    ]
    estimated_output_share = sum(all_outputs) / sum(served)
    metadata = dict(source_data.get("metadata") or {})
    metadata["num_conversations"] = len(output_conversations)
    metadata["transform"] = {
        "name": "modeled-long-conversation-v3",
        "synthetic": True,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "target_turns_per_conversation": args.target_turns,
        "target_served_tokens_per_turn": args.target_served_tokens,
        "target_output_share": args.output_share,
        "seed": args.seed,
        "method": [
            "real continuation deltas stitched deterministically to the target turn count",
            (
                "unique stable synthetic project archive appended to the first "
                "rendered system/developer instruction per conversation, preserving "
                "the source instruction as a shared radix prefix"
            ),
            "forced completions truncated or extended with labeled synthetic notes to preserve output share",
        ],
        "reporting_label": "modeled-mix; not an unmodified production trace",
        "estimated_content_token_metrics": {
            "turns": len(served),
            "served_tokens_per_turn_mean": statistics.mean(served),
            "served_tokens_per_turn_p25": percentile(served, 0.25),
            "served_tokens_per_turn_p50": percentile(served, 0.50),
            "served_tokens_per_turn_p75": percentile(served, 0.75),
            "output_share": estimated_output_share,
            "archive_tokens_mean": statistics.mean(archive_sizes),
            "archive_placement_counts": {
                placement: archive_placements.count(placement)
                for placement in sorted(set(archive_placements))
            },
        },
        "caveat": (
            "Content-token estimates omit chat-template overhead. Actual cache hit, "
            "served-token geometry, and output share must be measured by the replay driver."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(
            {"metadata": metadata, "conversations": output_conversations},
            file,
        )
    print(f"wrote {output_path}")
    print(
        f"conversations={len(output_conversations)} turns={len(served)} "
        f"served/turn mean={statistics.mean(served):.1f} "
        f"p50={percentile(served, 0.50)} output_share={estimated_output_share:.5%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
