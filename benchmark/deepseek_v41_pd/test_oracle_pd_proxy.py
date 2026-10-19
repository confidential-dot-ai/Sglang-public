"""CPU-only checks for sticky multi-prefill routing in oracle_pd_proxy.py."""

import json
from collections import Counter

import oracle_pd_proxy as proxy
import pytest
from fastapi.testclient import TestClient

WORKERS = proxy.parse_prefill_workers(
    [
        "http://127.0.0.1:8031@8998",
        "http://127.0.0.1:8032@8999",
        "http://127.0.0.1:8033@9000",
    ],
    default_bootstrap_port=1,
)
DECODE_URL = "http://127.0.0.1:8034"


class FakeResponse:
    def __init__(self, payload, lines=()):
        self.status = 200
        self._payload = payload
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)

    @property
    def content(self):
        async def lines():
            for line in self._lines:
                yield line

        return lines()


class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, json):
        self.posts.append((url, json))
        if not url.startswith(DECODE_URL):
            return FakeResponse({"meta_info": {"cached_tokens": 5}})
        meta = {"id": "r", "prompt_tokens": 3, "completion_tokens": 2}
        if json["stream"]:
            chunk = {"text": "hi", "meta_info": {**meta, "finish_reason": None}}
            return FakeResponse(
                None, [b"data: " + proxy.json.dumps(chunk).encode(), b"data: [DONE]"]
            )
        return FakeResponse({"meta_info": meta, "text": "hi", "output_ids": [7, 8]})

    async def close(self):
        pass


def make_client():
    session = FakeSession()
    app = proxy.build_app(
        prefill_workers=WORKERS,
        decode_url=DECODE_URL,
        tokenize=lambda messages: [1, 2, 3],
        forcing=False,
        session_factory=lambda: session,
    )
    return TestClient(app), session


def body(first_user, *, stream=False, turns=1):
    messages = [{"role": "user", "content": first_user}]
    for i in range(turns - 1):
        messages += [
            {"role": "assistant", "content": f"answer {i}"},
            {"role": "user", "content": f"follow-up {i}"},
        ]
    return {"messages": messages, "max_tokens": 2, "stream": stream}


def test_routing_is_process_stable_and_balanced():
    keys = [f"conv-{i}" for i in range(256)]
    choices = [proxy.select_prefill(key, 3) for key in keys]
    # Literal pins: any per-process salted hash (Python hash()) would move these
    # between proxy restarts and split one conversation across prefill caches.
    assert choices[:8] == [0, 1, 2, 0, 0, 1, 1, 0]
    counts = Counter(choices)
    assert set(counts) == {0, 1, 2}
    assert all(abs(n - 256 / 3) < 0.25 * 256 / 3 for n in counts.values()), counts


def test_bootstrap_endpoint_matches_chosen_prefill_in_both_payloads():
    client, session = make_client()
    with client:
        for i in range(12):
            key = f"conv-{i}"
            chosen = WORKERS[proxy.select_prefill(key, len(WORKERS))]
            for stream in (False, True):
                session.posts.clear()
                response = client.post(
                    "/v1/chat/completions",
                    json=body(f"question {i}", stream=stream),
                    headers={proxy.ROUTING_KEY_HEADER: key},
                )
                assert response.status_code == 200
                (prefill_url, prefill), (decode_url, decode) = sorted(
                    session.posts, key=lambda post: post[0].startswith(DECODE_URL)
                )
                assert prefill_url == chosen.url + "/generate"
                assert decode_url == DECODE_URL + "/generate"
                for payload in (prefill, decode):
                    assert payload["bootstrap_port"] == chosen.bootstrap_port
                    assert payload["bootstrap_host"] == chosen.bootstrap_host
                assert prefill["bootstrap_room"] == decode["bootstrap_room"]
        stats = client.get("/stats").json()["prefill_workers"]
    assert sum(s["requests"] for s in stats.values()) == 24
    for worker in WORKERS:
        assert stats[worker.url]["bootstrap_port"] == worker.bootstrap_port


def test_unkeyed_conversation_stays_on_one_prefill_as_it_grows():
    client, session = make_client()
    with client:
        for first in ("alpha", "beta", "gamma", "delta"):
            urls = set()
            for turns in (1, 2, 5):
                session.posts.clear()
                client.post("/v1/chat/completions", json=body(first, turns=turns))
                urls |= {
                    url for url, _ in session.posts if not url.startswith(DECODE_URL)
                }
            assert urls == {WORKERS[proxy.select_prefill(first, 3)].url + "/generate"}


def test_prefill_url_backward_compatible_and_rejects_shared_bootstrap():
    (legacy,) = proxy.parse_args(
        ["--prefill-url", "http://127.0.0.1:8031", "--bootstrap-port", "8998"]
    ).prefill_workers
    assert (legacy.url, legacy.bootstrap_port) == ("http://127.0.0.1:8031", 8998)
    with pytest.raises(ValueError, match="distinct bootstrap"):
        proxy.parse_prefill_workers(
            ["http://127.0.0.1:8031", "http://127.0.0.1:8032"], 8998
        )
