"""Local benchmark-only PD adapter preserving forced IDs and cache accounting.

Uses the existing worker /generate APIs and PD bootstrap protocol. The public
Rust gateway currently strips unknown forced_token_ids fields. Native response
metadata exposes cache hits even without --enable-cache-report.

With several prefill workers, each request is pinned to one of them by a stable
hash of its X-SMG-Routing-Key header (or, without it, of its first user/system
message), so a conversation's prefix cache stays on a single prefill worker.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import aiohttp
import msgspec
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.encoding_dsv41 import encode_messages

logger = logging.getLogger(__name__)

ROUTING_KEY_HEADER = "X-SMG-Routing-Key"
TOKENIZER_PATH = "/data/DeepSeek-V4.1-Flash"


class PrefillWorker(msgspec.Struct, frozen=True):
    url: str
    bootstrap_host: str
    bootstrap_port: int


def parse_prefill_workers(specs, default_bootstrap_port):
    """Parse `URL` or `URL@BOOTSTRAP_PORT` entries; bare URLs use the default port."""
    workers = []
    for spec in specs:
        url, sep, port = spec.rpartition("@")
        if not sep or not port.isdigit():
            url, port = spec, str(default_bootstrap_port)
        url = url.rstrip("/")
        host = urlparse(url).hostname
        if host is None:
            raise ValueError(f"Invalid prefill URL: {spec!r}")
        workers.append(
            PrefillWorker(url=url, bootstrap_host=host, bootstrap_port=int(port))
        )
    if len({(w.bootstrap_host, w.bootstrap_port) for w in workers}) != len(workers):
        raise ValueError("Prefill workers need distinct bootstrap endpoints")
    return workers


def routing_key(headers, messages):
    key = headers.get(ROUTING_KEY_HEADER)
    if key:
        return key
    for message in messages:
        if message.get("role") in ("user", "system"):
            content = message.get("content")
            return (
                content
                if isinstance(content, str)
                else json.dumps(content, sort_keys=True)
            )
    return ""


def select_prefill(key, num_workers):
    # sha1 rather than hash(): Python salts str hashes per process, which would
    # re-shuffle conversations across proxy restarts and uvicorn workers.
    digest = hashlib.sha1(key.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") % num_workers


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, trust_remote_code=True, local_files_only=True
    )


def render_and_tokenize(tokenizer, messages):
    rendered = encode_messages(messages, thinking_mode="thinking")
    return tokenizer.encode(rendered, add_special_tokens=False)


def _default_session():
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600),
        connector=aiohttp.TCPConnector(limit=256),
    )


def _usage(meta, pmeta):
    prompt = int(meta["prompt_tokens"])
    completion = int(meta["completion_tokens"])
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": int(pmeta["meta_info"]["cached_tokens"])
        },
    }


def build_app(
    *,
    prefill_workers,
    decode_url,
    tokenize,
    forcing=True,
    session_factory=_default_session,
):
    stats = {
        w.url: {
            "bootstrap_port": w.bootstrap_port,
            "requests": 0,
            "prompt_tokens": 0,
            "errors": 0,
        }
        for w in prefill_workers
    }

    @asynccontextmanager
    async def lifespan(app):
        app.state.session = session_factory()
        yield
        await app.state.session.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.get("/stats")
    async def get_stats():
        return {"prefill_workers": stats}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        key = routing_key(request.headers, body["messages"])
        worker = prefill_workers[select_prefill(key, len(prefill_workers))]
        ids = await asyncio.to_thread(tokenize, body["messages"])
        logger.debug(
            "route key=%r prefill=%s bootstrap_port=%d prompt_tokens=%d",
            key[:64],
            worker.url,
            worker.bootstrap_port,
            len(ids),
        )
        counters = stats[worker.url]
        counters["requests"] += 1
        counters["prompt_tokens"] += len(ids)
        native = {
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": body.get(
                    "max_completion_tokens", body.get("max_tokens")
                ),
                "ignore_eos": body.get("ignore_eos", True),
                "temperature": body.get("temperature", 0),
            },
            # Decode pulls KV from this bootstrap endpoint, so it must name the
            # same prefill worker that receives the request.
            "bootstrap_host": worker.bootstrap_host,
            "bootstrap_port": worker.bootstrap_port,
            "bootstrap_room": secrets.randbits(62),
        }
        if forcing:
            native["sampling_params"]["forced_token_ids"] = body["forced_token_ids"]
        session = request.app.state.session

        async def prefill():
            async with session.post(
                worker.url + "/generate", json={**native, "stream": False}
            ) as r:
                d = await r.json()
                if r.status != 200:
                    raise RuntimeError(f"prefill {r.status}: {d}")
                return d

        ptask = asyncio.create_task(prefill())
        if not body.get("stream"):
            try:
                async with session.post(
                    decode_url + "/generate", json={**native, "stream": False}
                ) as r:
                    d = await r.json()
                    if r.status != 200:
                        counters["errors"] += 1
                        return JSONResponse(d, status_code=r.status)
                pmeta = await ptask
                return {
                    "id": d["meta_info"]["id"],
                    "object": "chat.completion",
                    "model": "dsv41",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": d["text"]},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": _usage(d["meta_info"], pmeta),
                    "output_ids": d["output_ids"],
                }
            except Exception:
                counters["errors"] += 1
                raise
            finally:
                if not ptask.done():
                    ptask.cancel()
                await asyncio.gather(ptask, return_exceptions=True)

        async def stream():
            previous = ""
            meta = None
            try:
                async with session.post(
                    decode_url + "/generate", json={**native, "stream": True}
                ) as r:
                    if r.status != 200:
                        raise RuntimeError(f"decode {r.status}: {await r.text()}")
                    async for line in r.content:
                        if not line.startswith(b"data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == b"[DONE]":
                            break
                        d = json.loads(raw)
                        if "error" in d:
                            raise RuntimeError(str(d["error"]))
                        meta = d["meta_info"]
                        if (
                            meta.get("finish_reason", {})
                            and meta["finish_reason"].get("type") == "abort"
                        ):
                            raise RuntimeError(str(meta["finish_reason"]))
                        text = d.get("text", "")
                        if not text.startswith(previous):
                            raise RuntimeError("Non-cumulative native text")
                        delta = text[len(previous) :]
                        previous = text
                        if delta:
                            out = {
                                "id": meta["id"],
                                "object": "chat.completion.chunk",
                                "model": "dsv41",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield "data: " + json.dumps(out) + "\n\n"
                pmeta = await ptask
                if meta is None:
                    raise RuntimeError("No native response metadata")
                yield (
                    "data: "
                    + json.dumps({"choices": [], "usage": _usage(meta, pmeta)})
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"
            except Exception:
                counters["errors"] += 1
                raise
            finally:
                if not ptask.done():
                    ptask.cancel()
                await asyncio.gather(ptask, return_exceptions=True)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Oracle PD benchmark adapter")
    p.add_argument(
        "--prefill-url",
        action="append",
        help="Prefill worker as URL or URL@BOOTSTRAP_PORT; repeat for sticky routing across workers "
        "(default: http://127.0.0.1:8031)",
    )
    p.add_argument("--decode-url", default="http://127.0.0.1:8032")
    p.add_argument(
        "--bootstrap-port",
        type=int,
        default=8998,
        help="Bootstrap port for --prefill-url entries without @PORT",
    )
    p.add_argument("--port", type=int, default=8034)
    # Real greedy sampling and acceptance on the same prompts and output lengths.
    p.add_argument("--no-forcing", action="store_true")
    p.add_argument("--log-level", default="info")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        logger.warning("Ignoring unknown arguments: %s", unknown)
    args.prefill_workers = parse_prefill_workers(
        args.prefill_url or ["http://127.0.0.1:8031"], args.bootstrap_port
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper())
    tokenizer = load_tokenizer()
    app = build_app(
        prefill_workers=args.prefill_workers,
        decode_url=args.decode_url.rstrip("/"),
        tokenize=lambda messages: render_and_tokenize(tokenizer, messages),
        forcing=not args.no_forcing,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
