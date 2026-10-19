"""Long-context correctness probe for a running DeepSeek-V4.1 server or PD pair.

For each target length it builds one deterministic haystack (seeded pseudo-text)
with three needles ("the access code for <name> is <code>") at 10/50/90% depth
and asks for each needle in turn: the first question prefills the whole context
cold, the next two share its prefix, so they also exercise prefix reuse. A
multi-turn case then appends the answers and asks again, the way a chat grows.
Last, a request just over --context-length must be rejected cleanly with the
server still healthy afterwards.

Every request is greedy and non-thinking, streamed so TTFT is measured. The
JSON written by --out can be fed back with --compare (e.g. a run on another
image or config): any changed answer is reported.

    python probe_long_context.py --url http://127.0.0.1:8050 \\
        --lengths 131072,262144,524288,1040000 --context-length 1048576 \\
        --out /data/.../long-context.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import sys
import time
import urllib.parse

DEPTHS = (0.1, 0.5, 0.9)
NAMES = ("Orion", "Velma", "Tarquin")
WORDS = (
    "river stone lantern harbor meadow copper signal winter orchard beacon "
    "quarry falcon mirror thunder ledger canvas pillar garnet summit violet "
    "anchor bramble cinder delta ember fathom glacier hollow island juniper "
    "kestrel lagoon marble nectar obsidian prairie quiver rampart saffron tundra"
).split()


def _sentence(rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(8, 16))]
    return " ".join(words).capitalize() + ". "


def _code(rng: random.Random) -> str:
    return f"{rng.randint(1000, 9999)}-{rng.choice('BCDFGHJKLMNPQRSTVWXZ')}{rng.choice('BCDFGHJKLMNPQRSTVWXZ')}"


def build_haystack(tokenizer, target_tokens: int, seed: int):
    """Text of about ``target_tokens`` tokens with NAMES' codes at DEPTHS."""
    rng = random.Random(seed)
    codes = {name: _code(rng) for name in NAMES}
    # Estimate tokens per char on a sample, then generate a little more than needed.
    sample = "".join(_sentence(rng) for _ in range(400))
    per_char = len(tokenizer.encode(sample, add_special_tokens=False)) / len(sample)
    chars = int(target_tokens / per_char)
    parts, size = [], 0
    while size < chars:
        s = _sentence(rng)
        parts.append(s)
        size += len(s)
    for depth, name in zip(DEPTHS, NAMES):
        at = int(len(parts) * depth)
        parts.insert(at, f"Remember this: the access code for {name} is {codes[name]}. ")
    return "".join(parts), codes


def _question(name: str) -> str:
    return (
        f"\n\nWhat is the access code for {name}? "
        "Reply with only the code, nothing else."
    )


class Server:
    def __init__(self, url: str, model: str, timeout: float):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.model, self.timeout = model, timeout

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def healthy(self) -> bool:
        try:
            c = self._conn()
            c.request("GET", "/health")
            ok = c.getresponse().status == 200
            c.close()
            return ok
        except OSError:
            return False

    def chat(self, messages, max_tokens: int) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        start = time.perf_counter()
        c = self._conn()
        c.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        resp = c.getresponse()
        if resp.status != 200:
            detail = resp.read().decode(errors="replace")[:500]
            c.close()
            return {"status": resp.status, "error": detail}
        text, ttft, usage, error = [], None, None, None
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                error = event["error"]
                break
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    text.append(piece)
        c.close()
        return {
            "status": 200,
            "error": error,
            "text": "".join(text),
            "ttft_s": ttft,
            "total_s": time.perf_counter() - start,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
        }


def run_length(server, tokenizer, length: int, max_tokens: int, turns: int):
    haystack, codes = build_haystack(tokenizer, length, seed=length)
    cases = []

    def record(case, messages, name):
        r = server.chat(messages, max_tokens)
        r.update(case=case, target_tokens=length, needle=name, expected=codes[name])
        r["correct"] = r.get("status") == 200 and codes[name] in (r.get("text") or "")
        r["healthy_after"] = server.healthy()
        cases.append(r)
        print(
            f"  {length:>8} {case:<14} {name:<8} prompt={r.get('prompt_tokens')} "
            f"ttft={r.get('ttft_s') or 0:.2f}s total={r.get('total_s') or 0:.2f}s "
            f"ok={r['correct']} -> {r.get('text', r.get('error'))!r}",
            flush=True,
        )
        return r

    # Same haystack, one question per needle: the first is cold, the rest hit its prefix.
    for name in NAMES:
        record("needle", [{"role": "user", "content": haystack + _question(name)}], name)
    # A growing conversation over the same haystack.
    messages = [{"role": "user", "content": haystack + _question(NAMES[0])}]
    for turn in range(turns):
        name = NAMES[turn % len(NAMES)]
        r = record(f"multiturn-{turn}", messages, name)
        messages = messages + [
            {"role": "assistant", "content": r.get("text") or ""},
            {"role": "user", "content": _question(NAMES[(turn + 1) % len(NAMES)])},
        ]
    return cases


def run_over_limit(server, tokenizer, context_length: int):
    haystack, _ = build_haystack(tokenizer, context_length + 4096, seed=7)
    r = server.chat([{"role": "user", "content": haystack + _question(NAMES[0])}], 8)
    r.update(case="over-limit", target_tokens=context_length + 4096)
    # Rejected either up front (4xx) or as a stream error; never served or crashed.
    r["correct"] = (r.get("status", 200) >= 400 or bool(r.get("error"))) and not r.get(
        "text"
    )
    r["healthy_after"] = server.healthy()
    print(
        f"  over-limit status={r.get('status')} error={str(r.get('error'))[:160]!r} "
        f"healthy_after={r['healthy_after']}",
        flush=True,
    )
    return r


def compare(results, baseline_path):
    base = {
        (c["case"], c["target_tokens"], c.get("needle")): c
        for c in json.load(open(baseline_path))["cases"]
    }
    diffs = []
    for c in results:
        key = (c["case"], c["target_tokens"], c.get("needle"))
        if key in base and base[key].get("text") != c.get("text"):
            diffs.append(
                {"key": key, "before": base[key].get("text"), "after": c.get("text")}
            )
    return diffs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8050")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--tokenizer", default="/data/DeepSeek-V4.1-Flash")
    p.add_argument("--lengths", default="131072,262144,524288,1040000")
    p.add_argument("--context-length", type=int, default=1048576)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--skip-over-limit", action="store_true")
    p.add_argument("--out")
    p.add_argument("--compare", help="an earlier --out JSON to diff answers against")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    server = Server(args.url, args.model, args.timeout)
    if not server.healthy():
        sys.exit(f"{args.url} is not healthy")

    cases = []
    for length in (int(x) for x in args.lengths.split(",") if x):
        cases += run_length(server, tokenizer, length, args.max_tokens, args.turns)
        if not cases[-1]["healthy_after"]:
            print("server unhealthy; stopping", flush=True)
            break
    if not args.skip_over_limit and server.healthy():
        cases.append(run_over_limit(server, tokenizer, args.context_length))

    report = {
        "url": args.url,
        "context_length": args.context_length,
        "passed": all(c["correct"] and c["healthy_after"] for c in cases),
        "cases": cases,
    }
    if args.compare:
        report["changed_answers"] = compare(cases, args.compare)
        print(f"changed answers vs {args.compare}: {len(report['changed_answers'])}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print("PASSED" if report["passed"] else "FAILED", flush=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
