"""Replay one rollout's oracle-mode turns against a live server, in order.

Each line from build_trace.py is one turn's full request body (messages +
sampling_params.forced_token_ids). Turns must be sent sequentially against
the same server -- each forced completion becomes real prefix-cache content
for the next turn's request, exactly as in genuine multi-turn serving. Do not
parallelize turns within a rollout; different rollouts are independent and
may run concurrently.
"""

from __future__ import annotations

import argparse
import json
import time

import requests


def replay_rollout(base_url: str, model: str, jsonl_path: str) -> list[dict]:
    results = []
    with open(jsonl_path) as f:
        turns = [json.loads(line) for line in f]

    for i, turn in enumerate(turns):
        body = {
            "model": model,
            "messages": turn["messages"],
            **turn["sampling_params"],
        }
        start = time.monotonic()
        resp = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600)
        resp.raise_for_status()
        elapsed = time.monotonic() - start
        usage = resp.json().get("usage", {})
        result = {
            "turn": i,
            "elapsed_s": elapsed,
            "forced_tokens": len(turn["sampling_params"]["forced_token_ids"]),
            "usage": usage,
        }
        results.append(result)
        print(
            f"turn {i}: {elapsed:.2f}s, forced={result['forced_tokens']}, "
            f"usage={usage}"
        )
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("jsonl_path", help="Output of build_trace.py for one rollout")
    args = ap.parse_args()

    results = replay_rollout(args.base_url, args.model, args.jsonl_path)
    total_forced = sum(r["forced_tokens"] for r in results)
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"\n{len(results)} turns, {total_forced} forced tokens, {total_time:.2f}s total")
    if total_time > 0:
        print(f"{total_forced / total_time:.1f} forced tokens/s (wall-clock, includes prefill)")


if __name__ == "__main__":
    main()
