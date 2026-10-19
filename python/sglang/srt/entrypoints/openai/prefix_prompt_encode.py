r"""Encode a rendered chat prompt by reusing the ids of an earlier prompt that
shares its prefix, with ids identical to ``tokenizer.encode(text, **encode_kwargs)``.

Multi-turn chat re-renders the whole conversation every turn, so each prompt
repeats almost all of the one before it. Cutting text right after an added-token
match cannot change the ids (see ``parallel_prompt_encode``), so a prompt takes
a cached prompt's ids up to an added-token end inside the prefix the two share,
and only the rest is encoded.

That end must also be an added-token end of the cached text. The backend matches
added tokens leftmost-longest, so whether a match starts at position p depends
only on ``text[p : p + max_added_len]``. Two texts that agree on their first L
characters therefore have the same matches starting before
``L - max_added_len + 1``; the cut is taken at an end e <= L - max_added_len,
where both texts have the same matches and so the same ids for ``text[:e]``.

Each process keeps its own cache, bounded by bytes, one entry per conversation:
a prompt that extends a cached one (the next turn) replaces it. The cached
prompt's generation tail may render differently once it is history, so "extends"
allows a few added tokens' worth of unshared tail. Gated by
SGLANG_OPT_PREFIX_PROMPT_ENCODE and SGLANG_PREFIX_PROMPT_ENCODE_CACHE_MB.
"""

import bisect
import logging
import sys
import threading
from array import array
from collections import OrderedDict
from typing import Optional

import msgspec

from sglang.srt.entrypoints.openai.parallel_prompt_encode import (
    _added_spans,
    parallel_prompt_encode,
    split_safe_plan,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Arbitrary; shorter prompts encode in a few ms and are not worth an entry.
_MIN_CHARS = 8192
# Prompts agreeing on this many leading characters share a bucket; the bucket
# is searched for the longest shared prefix. Conversations with one system
# prompt all share a bucket, so it holds about as many as run concurrently.
_BUCKET_CHARS = 4096
_MAX_BUCKET_ENTRIES = 64
# Arbitrary: how many added tokens' worth of a cached prompt's tail may differ
# for a new prompt to count as its next turn.
_EXTEND_TAIL_ADDED = 4
# Prefix comparison stride: each step copies this many characters of both texts.
_LCP_BLOCK = 1 << 16


class _Entry(msgspec.Struct):
    text: str
    # array("i") of token ids; array("q") of added-token ends (characters) and
    # the number of ids before each end, ascending.
    ids: array
    ends: array
    counts: array
    nbytes: int


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo = 0
    while lo < n and a[lo : lo + _LCP_BLOCK] == b[lo : lo + _LCP_BLOCK]:
        lo += _LCP_BLOCK
    hi = min(lo + _LCP_BLOCK, n)
    # Invariant: a[:lo] == b[:lo]; a[:hi + 1] != b[:hi + 1] or hi == n.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class PrefixPromptEncoder:
    def __init__(
        self,
        *,
        tokenizer,
        added_ids: frozenset,
        max_added_len: int,
        max_bytes: int,
        verify: bool,
    ):
        self._tokenizer = tokenizer
        # Ids of the added tokens _added_spans matches, one per match.
        self._added_ids = added_ids
        self._max_added_len = max_added_len
        self._max_bytes = max_bytes
        self._verify = verify
        self._lock = threading.Lock()
        self._lru: OrderedDict[int, _Entry] = OrderedDict()
        self._buckets: dict[tuple, OrderedDict[int, None]] = {}
        self._bucket_of: dict[int, tuple] = {}
        self._next_key = 0
        self._nbytes = 0
        self.hits = 0
        self.misses = 0
        self.encoded_chars = 0
        self.reused_chars = 0

    def encode(self, text: str, encode_kwargs: Optional[dict] = None) -> list[int]:
        encode_kwargs = encode_kwargs or {}
        plan = split_safe_plan(self._tokenizer, encode_kwargs)
        if plan is None or not isinstance(text, str) or len(text) < _MIN_CHARS:
            return parallel_prompt_encode(self._tokenizer, text, encode_kwargs)
        _, _, pattern, npattern, _ = plan
        bucket = (encode_kwargs.get("add_special_tokens", True), text[:_BUCKET_CHARS])
        spliced = self._splice(text, bucket=bucket, pattern=pattern, npattern=npattern)
        if spliced is None:
            ids = parallel_prompt_encode(self._tokenizer, text, encode_kwargs)
            entry = self._make_entry(text, ids, pattern=pattern, npattern=npattern)
            self.misses += 1
            self.encoded_chars += len(text)
            if entry is not None:
                self._insert(entry, bucket=bucket, replaces=None)
            return ids
        ids, entry, source_key, cut, extends = spliced
        if self._verify:
            full = parallel_prompt_encode(self._tokenizer, text, encode_kwargs)
            if full != ids:
                logger.error(
                    "prefix_prompt_encode: spliced ids differ from the full encode "
                    "(%d vs %d ids, cut at %d of %d chars); using the full encode.",
                    len(ids),
                    len(full),
                    cut,
                    len(text),
                )
                self._discard(source_key)
                return full
        self.hits += 1
        self.encoded_chars += len(text) - cut
        self.reused_chars += cut
        self._insert(entry, bucket=bucket, replaces=source_key if extends else None)
        return ids

    def _splice(self, text, *, bucket, pattern, npattern):
        """(ids, new entry, source key, cut, whether text extends the source)
        from the cached prompt sharing the longest prefix, or None."""
        with self._lock:
            candidates = [
                (key, self._lru[key]) for key in self._buckets.get(bucket, ())
            ]
        # A conversation's previous turn is a whole prefix of its next prompt;
        # startswith finds it without copying either text.
        extended = [
            (len(c.text), key, c) for key, c in candidates if text.startswith(c.text)
        ]
        if extended:
            shared, key, cached = max(extended, key=lambda x: x[0])
        elif candidates:
            shared, key, cached = max(
                ((_common_prefix_len(text, c.text), key, c) for key, c in candidates),
                key=lambda x: x[0],
            )
        else:
            return None
        extends = len(cached.text) - shared <= _EXTEND_TAIL_ADDED * self._max_added_len
        cut_limit = shared - self._max_added_len
        j = bisect.bisect_right(cached.ends, cut_limit)
        if j == 0:
            return None
        cut, reused = cached.ends[j - 1], cached.counts[j - 1]
        suffix = text[cut:]
        # The plan admits add_special_tokens=True only when it adds nothing.
        suffix_ids = parallel_prompt_encode(
            self._tokenizer, suffix, {"add_special_tokens": False}
        )
        suffix_ends, suffix_counts = self._checkpoints(
            suffix, suffix_ids, pattern=pattern, npattern=npattern
        )
        if suffix_ends is None:
            return None
        ids = array("i", cached.ids[:reused])
        ids.extend(suffix_ids)
        ends = cached.ends[:j]
        ends.extend(cut + e for e in suffix_ends)
        counts = cached.counts[:j]
        counts.extend(reused + c for c in suffix_counts)
        entry = _Entry(
            text=text,
            ids=ids,
            ends=ends,
            counts=counts,
            nbytes=_nbytes(text, ids, ends),
        )
        return ids.tolist(), entry, key, cut, extends

    def _make_entry(self, text, ids, *, pattern, npattern) -> Optional[_Entry]:
        ends, counts = self._checkpoints(text, ids, pattern=pattern, npattern=npattern)
        if ends is None:
            return None
        ids = array("i", ids)
        return _Entry(
            text=text,
            ids=ids,
            ends=ends,
            counts=counts,
            nbytes=_nbytes(text, ids, ends),
        )

    def _checkpoints(self, text, ids, *, pattern, npattern):
        """Each added-token end in text and the id count before it, or (None,
        None) when the ids do not show one added-token id per match."""
        spans = _added_spans(text, pattern, npattern)
        added = self._added_ids
        positions = [i for i, t in enumerate(ids) if t in added]
        if len(positions) != len(spans):
            return None, None
        return array("q", (e for _, e in spans)), array("q", (p + 1 for p in positions))

    def _insert(self, entry: _Entry, *, bucket, replaces: Optional[int]) -> None:
        if entry.nbytes > self._max_bytes:
            return
        with self._lock:
            if replaces is not None:
                self._drop_locked(replaces)
            key = self._next_key
            self._next_key += 1
            self._lru[key] = entry
            members = self._buckets.setdefault(bucket, OrderedDict())
            members[key] = None
            self._bucket_of[key] = bucket
            self._nbytes += entry.nbytes
            while len(members) > _MAX_BUCKET_ENTRIES:
                self._drop_locked(next(iter(members)))
            while self._nbytes > self._max_bytes:
                self._drop_locked(next(iter(self._lru)))

    def _discard(self, key: int) -> None:
        with self._lock:
            self._drop_locked(key)

    def _drop_locked(self, key: int) -> None:
        entry = self._lru.pop(key, None)
        if entry is None:
            return
        self._nbytes -= entry.nbytes
        bucket = self._bucket_of.pop(key)
        members = self._buckets[bucket]
        del members[key]
        if not members:
            del self._buckets[bucket]


def _nbytes(text: str, ids: array, ends: array) -> int:
    # Each of ends and counts holds one int64 per added-token match.
    return sys.getsizeof(text) + ids.itemsize * len(ids) + 2 * ends.itemsize * len(ends)


def maybe_create_prefix_prompt_encoder(tokenizer) -> Optional[PrefixPromptEncoder]:
    if not envs.SGLANG_OPT_PREFIX_PROMPT_ENCODE.get():
        return None
    plan = split_safe_plan(tokenizer, {})
    if plan is None:
        logger.warning(
            "SGLANG_OPT_PREFIX_PROMPT_ENCODE is set, but this tokenizer cannot be "
            "cut at added tokens; prompts are encoded in full."
        )
        return None
    backend, _, _, npattern, _ = plan
    tokens = backend.get_added_tokens_decoder().items()
    return PrefixPromptEncoder(
        tokenizer=tokenizer,
        # Normalized tokens are matched only when _added_spans has their pattern.
        added_ids=frozenset(
            i
            for i, t in tokens
            if t.content and (not t.normalized or npattern is not None)
        ),
        max_added_len=max((len(t.content) for _, t in tokens), default=0),
        max_bytes=envs.SGLANG_PREFIX_PROMPT_ENCODE_CACHE_MB.get() << 20,
        verify=envs.SGLANG_DEBUG_PREFIX_PROMPT_ENCODE_VERIFY.get(),
    )
