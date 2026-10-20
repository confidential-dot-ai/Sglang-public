"""Prefix-reusing prompt encoding must produce the ids the full encode produces.

`PrefixPromptEncoder` splices a cached prompt's ids with the encode of the rest
of a new prompt, so a cut one character off is a silently different prompt, not
a crash. These cases pin id identity against `tokenizer.encode` on multi-turn
prompts, on cut points next to added tokens (including one added token that
prefixes another, which only the max-added-length margin keeps correct), on
random edits near the shared-prefix end, under cache eviction and concurrent
use, and on DeepSeek-V4.1 conversations when that checkpoint is present.
"""

import json
import os
import random
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai import parallel_prompt_encode as ppe
from sglang.srt.entrypoints.openai import prefix_prompt_encode as pfx
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_TRAIN_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "def main(argv):\n    return 0\n",
    "it's they're I'm we'll you'd she's",
    "中文文本 日本語 émigré naïve",
    "numbers 123 4567 89 1,234.56 <tag/> [bracket]",
]
# "<|a|>" prefixes "<|a|>b": a cut at the end of "<|a|>" in one text is not an
# added-token end in a text that continues with "b".
_SPECIALS = ["<|user|>", "<|assistant|>", "<think>", "</think>", "<|a|>", "<|a|>b"]
_DSV41 = "/data/DeepSeek-V4.1-Flash"
_DATASET = (
    "/home/ubuntu/datasets/oracle-realmix-256-r132-dsv4-modeled-long25-38k-v3.json"
)


@lru_cache(maxsize=1)
def _backend_json():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(ppe._INTRA_SPLIT_REGEX), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        _TRAIN_CORPUS * 20,
        trainers.BpeTrainer(
            vocab_size=600,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
        ),
    )
    backend.add_tokens(
        [AddedToken(t, normalized=False, special=True) for t in _SPECIALS]
    )
    return backend.to_str()


def _tokenizer():
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(_backend_json()))


def _encoder(tokenizer, max_bytes=1 << 30):
    with (
        envs.SGLANG_OPT_PREFIX_PROMPT_ENCODE.override(True),
        envs.SGLANG_PREFIX_PROMPT_ENCODE_CACHE_MB.override(max(1, max_bytes >> 20)),
    ):
        encoder = pfx.maybe_create_prefix_prompt_encoder(tokenizer)
    assert encoder is not None
    encoder._max_bytes = max_bytes
    return encoder


def _small_prompts():
    """Patch the size gates so short test prompts take the cached path."""
    return unittest.mock.patch.multiple(pfx, _MIN_CHARS=0, _BUCKET_CHARS=16)


def _conversation(rng, turns):
    """Prompts of one growing chat, each re-rendered in full like a turn."""
    words = ["alpha", "beta", "gamma", "中文", "émigré", "x=1;", "  ", "\n", "it's"]
    body, prompts = "<|user|>system rules: be precise\n", []
    for _ in range(turns):
        user = " ".join(rng.choice(words) for _ in range(rng.randrange(5, 40)))
        body += f"<|user|>{user}"
        prompts.append(body + "<|assistant|><think>")
        reply = " ".join(rng.choice(words) for _ in range(rng.randrange(5, 40)))
        body += f"<|assistant|>{reply}"
    return prompts


class TestPrefixPromptEncode(CustomTestCase):
    def setUp(self):
        self.tokenizer = _tokenizer()

    def _assert_same(self, encoder, text):
        self.assertEqual(encoder.encode(text), self.tokenizer.encode(text), text)

    def test_multi_turn_prompts_match_full_encode_and_reuse_the_prefix(self):
        with _small_prompts():
            encoder = _encoder(self.tokenizer)
            for seed in range(8):
                for prompt in _conversation(random.Random(seed), turns=12):
                    self._assert_same(encoder, prompt)
        self.assertGreater(encoder.hits, 0)
        self.assertGreater(encoder.reused_chars, encoder.encoded_chars)

    def test_added_token_prefixing_another_needs_the_margin(self):
        head = "<|user|>" + "shared words here " * 8
        first, second = head + "<|a|>tail one", head + "<|a|>b tail two"
        with _small_prompts():
            encoder = _encoder(self.tokenizer)
            self._assert_same(encoder, first)
            self._assert_same(encoder, second)
            # Without the margin the cut lands at the end of "<|a|>", which
            # the second text matches as "<|a|>b".
            encoder = _encoder(self.tokenizer)
            encoder._max_added_len = 0
            encoder.encode(first)
            self.assertNotEqual(encoder.encode(second), self.tokenizer.encode(second))

    def test_random_edits_near_the_shared_prefix_end(self):
        rng = random.Random(7)
        atoms = ["<|user|>", "<|assistant|>", "<|a|>", "<|a|>b", "<think>", "b"]
        atoms += list("ab 中é\n'.<|>")
        with _small_prompts():
            encoder = _encoder(self.tokenizer)
            for _ in range(400):
                base = "".join(rng.choice(atoms) for _ in range(rng.randrange(1, 60)))
                encoder.encode(base)
                cut = rng.randrange(len(base) + 1)
                tail = "".join(rng.choice(atoms) for _ in range(rng.randrange(0, 20)))
                self._assert_same(encoder, base[:cut] + tail)

    def test_eviction_keeps_the_cache_within_its_byte_budget(self):
        with _small_prompts():
            encoder = _encoder(self.tokenizer, max_bytes=20_000)
            for seed in range(20):
                for prompt in _conversation(random.Random(seed), turns=6):
                    self._assert_same(encoder, prompt)
                    self.assertLessEqual(encoder._nbytes, 20_000)
        self.assertEqual(encoder._nbytes, sum(e.nbytes for e in encoder._lru.values()))

    def test_one_entry_per_conversation(self):
        with _small_prompts():
            encoder = _encoder(self.tokenizer)
            for prompt in _conversation(random.Random(3), turns=10):
                encoder.encode(prompt)
        self.assertEqual(len(encoder._lru), 1)

    def test_interleaved_conversations_with_one_system_prompt_keep_their_entries(self):
        """A turn spliced off another conversation's shared system prompt must
        not evict that conversation's entry."""
        chats = [_conversation(random.Random(seed), turns=8) for seed in range(5)]
        with _small_prompts():
            encoder = _encoder(self.tokenizer)
            for turn in range(8):
                for chat in chats:
                    before = encoder.encoded_chars
                    self._assert_same(encoder, chat[turn])
                    # The cut backs off at most one added-token segment, so a
                    # turn re-encodes at most what the last two turns added.
                    if turn > 1:
                        encoded = encoder.encoded_chars - before
                        self.assertLessEqual(
                            encoded, len(chat[turn]) - len(chat[turn - 2])
                        )
        self.assertEqual(len(encoder._lru), len(chats))

    def test_concurrent_encodes_match_full_encode(self):
        prompts = [
            prompt
            for seed in range(6)
            for prompt in _conversation(random.Random(seed), turns=8)
        ]
        with _small_prompts():
            encoder = _encoder(self.tokenizer)
            with ThreadPoolExecutor(8) as pool:
                got = list(pool.map(encoder.encode, prompts))
        self.assertEqual(got, [self.tokenizer.encode(p) for p in prompts])

    def test_off_without_the_flag(self):
        with envs.SGLANG_OPT_PREFIX_PROMPT_ENCODE.override(False):
            self.assertIsNone(pfx.maybe_create_prefix_prompt_encoder(self.tokenizer))

    def test_unsafe_tokenizer_is_refused(self):
        backend = Tokenizer.from_str(_backend_json())
        backend.add_tokens([AddedToken("<rs>", rstrip=True, special=True)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        with envs.SGLANG_OPT_PREFIX_PROMPT_ENCODE.override(True):
            self.assertIsNone(pfx.maybe_create_prefix_prompt_encoder(tokenizer))


@unittest.skipUnless(
    os.path.isdir(_DSV41) and os.path.isfile(_DATASET),
    "needs the DeepSeek-V4.1 tokenizer and the realmix dataset",
)
class TestDeepSeekV41Conversations(CustomTestCase):
    def test_multi_turn_conversations_across_tokenizer_workers(self):
        from sglang.srt.entrypoints.openai import encoding_dsv41
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(_DSV41, trust_remote_code=True)
        with open(_DATASET) as f:
            conversations = json.load(f)["conversations"][:8]
        # Each tokenizer worker has its own cache and sees a random share of turns.
        workers = [_encoder(tokenizer) for _ in range(4)]
        rng = random.Random(0)
        for conversation in conversations:
            history = []
            for turn in conversation:
                history.extend(turn["messages"])
                text = encoding_dsv41.encode_messages(history, thinking_mode="chat")
                worker = rng.choice(workers)
                self.assertEqual(worker.encode(text), tokenizer.encode(text))
                reply = tokenizer.decode(turn["forced_token_ids"])
                history.append({"role": "assistant", "content": reply})
        reused = sum(w.reused_chars for w in workers)
        encoded = sum(w.encoded_chars for w in workers)
        self.assertGreater(reused, 2 * encoded)


if __name__ == "__main__":
    unittest.main()
