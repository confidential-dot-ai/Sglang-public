#!/usr/bin/env python3
"""Deterministic and tool-call smoke probe for DeepSeek V4 serving."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = (
    "What is 17 + 26? Answer with just the number.",
    "Name the capital of France in one word.",
    "Write a haiku about the ocean.",
    "List the first 5 prime numbers, comma separated.",
    "Say 'hello world' and nothing else.",
    "Spell the word 'banana' backwards.",
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def parse_completion(data: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    choice = data["choices"][0]
    message = choice["message"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    content = message.get("content") or ""
    return {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": completion_tokens,
        "hit_max_tokens": completion_tokens is not None
        and completion_tokens >= max_tokens,
        "content": content,
        "tool_calls": message.get("tool_calls") or [],
    }


def validate_tool_call(parsed: dict[str, Any]) -> str | None:
    tool_calls = parsed["tool_calls"]
    if not tool_calls:
        return "tool prompt produced no tool call"
    function = tool_calls[0].get("function", {})
    if function.get("name") != "get_weather":
        return f"unexpected tool name: {function.get('name')!r}"
    try:
        arguments = json.loads(function.get("arguments", ""))
    except json.JSONDecodeError as exc:
        return f"tool arguments are not JSON: {exc}"
    if str(arguments.get("location", "")).lower() != "beijing":
        return f"unexpected tool arguments: {arguments!r}"
    return None


def run_case(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    started = time.monotonic()
    data = post_json(f"{base_url.rstrip('/')}/v1/chat/completions", body, timeout)
    parsed = parse_completion(data, max_tokens)
    parsed.update(prompt=prompt, elapsed_s=round(time.monotonic() - started, 3))
    errors: list[str] = []
    if parsed["hit_max_tokens"] or parsed["finish_reason"] == "length":
        errors.append("completion exhausted max_tokens")
    if tools is None and not parsed["content"].strip():
        errors.append("empty text completion")
    if tools is not None:
        tool_error = validate_tool_call(parsed)
        if tool_error:
            errors.append(tool_error)
    parsed["errors"] = errors
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30300")
    parser.add_argument("--model", default="dsv4-flash-0731")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-tool-call", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases: list[dict[str, Any]] = []
    for prompt in PROMPTS:
        try:
            cases.append(
                run_case(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            )
        except Exception as exc:
            cases.append({"prompt": prompt, "errors": [repr(exc)]})
    if not args.skip_tool_call:
        try:
            cases.append(
                run_case(
                    base_url=args.base_url,
                    model=args.model,
                    prompt="What is the weather in Beijing? Use the weather tool.",
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    tools=[WEATHER_TOOL],
                )
            )
        except Exception as exc:
            cases.append({"prompt": "tool-call", "errors": [repr(exc)]})

    failures = [case for case in cases if case.get("errors")]
    payload = {
        "schema_version": 1,
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "passed": not failures,
        "num_cases": len(cases),
        "num_failures": len(failures),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
