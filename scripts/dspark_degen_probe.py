#!/usr/bin/env python3
"""Controlled repro matrix for the DSPARK repetition-loop bug.

At temperature 0, DSPARK's committed stream should be bit-identical to
target-only greedy decoding (accept = exact argmax match; bonus = target
argmax). Degeneration therefore means verify-time logits diverge from
decode-time logits. This probe splits the hypothesis space:

  A. solo x2      - same prompt twice at bs=1, nothing else running.
                    Degenerates?  -> bug reproduces without batching.
                    Outputs differ run-to-run? -> nondeterminism in the
                    verify forward even at bs=1.
  B. concurrent   - N copies of the prompt at once (self-batching).
                    Degenerates only here -> batch-shape/padding bug.
  C. mixed load   - prompt + background traffic from other prompts.
                    Degenerates only here -> cross-request contamination
                    (buffer reuse, paging, scheduler overlap).

Usage:
    python3 scripts/dspark_degen_probe.py \
        --samples PATH/samples_....jsonl --doc-id 30 \
        --base-url http://127.0.0.1:30300/v1/chat/completions \
        --max-tokens 6144
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp


def _load_prompt(samples_path: str, doc_id: int) -> list[dict]:
    with open(samples_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("filter") != "flexible-extract":
                continue
            if obj.get("doc_id") == doc_id:
                return json.loads(obj["arguments"]["gen_args_0"]["arg_0"][0])
    raise SystemExit(f"doc_id {doc_id} not found in {samples_path}")


def _degen_onset(text: str) -> int | None:
    """First char index where the next 600 chars hold <=12 distinct 8-grams."""
    window, step = 600, 100
    for i in range(0, max(0, len(text) - window), step):
        chunk = text[i : i + window]
        grams = {chunk[j : j + 8] for j in range(0, len(chunk) - 8, 4)}
        if len(grams) <= 12:
            return i
    return None


async def _gen(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    async with session.post(base_url, json=payload) as resp:
        resp.raise_for_status()
        out = await resp.json()
    return out["choices"][0]["message"]["content"]


def _report(case: str, run: int, text: str) -> None:
    onset = _degen_onset(text)
    print(
        f"[probe] case={case} run={run} len={len(text)} "
        f"degen_onset={onset if onset is not None else 'CLEAN'}",
        flush=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--doc-id", type=int, required=True)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30300/v1/chat/completions"
    )
    parser.add_argument("--model", default="m3")
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--concurrent-copies", type=int, default=4)
    parser.add_argument("--background-load", type=int, default=24)
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    messages = _load_prompt(args.samples, args.doc_id)
    timeout = aiohttp.ClientTimeout(total=3600)
    texts: dict[str, str] = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Case A: two sequential solo runs at bs=1.
        for run in (1, 2):
            t0 = time.monotonic()
            text = await _gen(
                session, args.base_url, args.model, messages, args.max_tokens
            )
            print(f"[probe] solo run {run} took {time.monotonic()-t0:.0f}s")
            _report("solo", run, text)
            texts[f"solo{run}"] = text
        identical = texts["solo1"] == texts["solo2"]
        print(f"[probe] solo determinism: identical={identical}", flush=True)
        if not identical:
            a, b = texts["solo1"], texts["solo2"]
            div = next(
                (i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                min(len(a), len(b)),
            )
            print(
                f"[probe] first divergence at char {div}: "
                f"...{a[max(0,div-40):div]!r} then {a[div:div+20]!r} vs {b[div:div+20]!r}"
            )

        # Case B: N concurrent copies of the same prompt.
        results = await asyncio.gather(
            *(
                _gen(session, args.base_url, args.model, messages, args.max_tokens)
                for _ in range(args.concurrent_copies)
            )
        )
        for i, text in enumerate(results, 1):
            _report("concurrent", i, text)
            texts[f"conc{i}"] = text
        uniq = len({t for t in results})
        print(
            f"[probe] concurrent agreement: {args.concurrent_copies} copies, "
            f"{uniq} distinct outputs",
            flush=True,
        )

        # Case C: probe under background load from other prompts.
        bg_messages = [
            [{"role": "user", "content": f"Count from 1 to 50 in words. v{i}"}]
            for i in range(args.background_load)
        ]
        bg_tasks = [
            asyncio.create_task(
                _gen(session, args.base_url, args.model, m, 2048)
            )
            for m in bg_messages
        ]
        await asyncio.sleep(2)
        text = await _gen(
            session, args.base_url, args.model, messages, args.max_tokens
        )
        for t in bg_tasks:
            t.cancel()
        _report("mixed-load", 1, text)
        texts["mixed"] = text

    if args.out_prefix:
        for name, text in texts.items():
            with open(f"{args.out_prefix}.{name}.txt", "w") as f:
                f.write(text)
        print(f"[probe] texts saved to {args.out_prefix}.*.txt")


if __name__ == "__main__":
    asyncio.run(main())
