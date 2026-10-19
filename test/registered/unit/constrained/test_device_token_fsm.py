"""Unit tests for the device-resident token-DFA grammar
(constrained/device_token_fsm.py) -- no server, no model loading, no GPU.

Covers, in dependency order:
  - DeviceTokenFSM.compile / advance / mask_logits_ against a hand-derived DFA
  - regex -> char DFA -> token DFA compilation (interegular)
  - the DeviceFSMGrammar host mirror's BaseGrammarObject contract
  - the two DFlash verify helpers, including the load-bearing property the
    whole integration rests on: masking the target logits per block position
    makes the existing (unmodified) accept/bonus computation clip the commit
    at the first grammar-illegal draft token, with a grammar-legal bonus.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")

import json

import torch

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.device_token_fsm import (
    DeviceFSMGrammar,
    DeviceFSMGrammarBackend,
    DeviceTokenFSM,
    compile_regex_to_token_fsm,
    dflash_advance_committed_state_,
    dflash_mask_block_logits,
    token_dfa_from_char_fsm,
)
from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from sglang.test.test_utils import CustomTestCase

# Toy vocab shared by most tests: 0 unused, 1='a', 2='b', 3='ab', 4='c',
# 5=EOS (special, no surface string). Grammar: a(ab)*b -- so from the start
# only 'a' (or the fused 'ab', which already completes the match) is legal.
TOKEN_STRINGS = [None, "a", "b", "ab", "c", None]
VOCAB_SIZE = 6
EOS = 5


def _compile_toy_fsm() -> DeviceTokenFSM:
    return compile_regex_to_token_fsm(
        regex="a(ab)*b",
        token_strings=TOKEN_STRINGS,
        vocab_size=VOCAB_SIZE,
        eos_token_ids={EOS},
    )


def _allowed(fsm: DeviceTokenFSM, state_tensor: torch.Tensor) -> list:
    logits = torch.zeros(1, VOCAB_SIZE)
    fsm.mask_logits_(logits, state_tensor)
    return [i for i in range(VOCAB_SIZE) if logits[0, i] == 0]


class TestDeviceTokenFSMCore(CustomTestCase):
    def test_compile_rejects_stuck_state(self):
        # State 1 is reachable, not final, and allows nothing -> generation
        # could never satisfy the grammar from it.
        with self.assertRaises(ValueError):
            DeviceTokenFSM.compile(
                transitions={0: {1: 1}},
                start_state=0,
                final_states=set(),
                vocab_size=VOCAB_SIZE,
                eos_token_ids={EOS},
            )

    def test_advance_matches_host_and_dead_on_missing(self):
        fsm = _compile_toy_fsm()
        states = torch.tensor([fsm.start_state, fsm.start_state], dtype=torch.int32)
        tokens = torch.tensor([1, 4])  # 'a' legal, 'c' illegal
        nxt = fsm.advance(states, tokens)
        self.assertEqual(int(nxt[0]), fsm.advance_host(fsm.start_state, 1))
        self.assertEqual(int(nxt[1]), fsm.dead_state)
        # DEAD is absorbing.
        again = fsm.advance(nxt[1:], torch.tensor([1]))
        self.assertEqual(int(again[0]), fsm.dead_state)

    def test_mask_semantics(self):
        fsm = _compile_toy_fsm()
        start = torch.tensor([fsm.start_state], dtype=torch.int32)
        # From start of a(ab)*b: 'a' steps in; 'ab' completes the whole match.
        self.assertEqual(_allowed(fsm, start), [1, 3])
        after_a = fsm.advance(start, torch.tensor([1]))
        self.assertEqual(_allowed(fsm, after_a), [1, 2, 3])
        # DEAD masks allow-all (safe by the commit-clip property; keeps the
        # argmax well-defined).
        dead = torch.tensor([fsm.dead_state], dtype=torch.int32)
        self.assertEqual(_allowed(fsm, dead), list(range(VOCAB_SIZE)))

    def test_final_state_allows_only_eos_and_sink_loops(self):
        fsm = _compile_toy_fsm()
        s = torch.tensor([fsm.start_state], dtype=torch.int32)
        for t in (1, 2):  # 'a', 'b' -> complete match
            s = fsm.advance(s, torch.tensor([t]))
        self.assertEqual(_allowed(fsm, s), [EOS])
        sink = fsm.advance(s, torch.tensor([EOS]))
        self.assertEqual(int(sink[0]), fsm.accept_sink)
        self.assertEqual(_allowed(fsm, sink), [EOS])
        again = fsm.advance(sink, torch.tensor([EOS]))
        self.assertEqual(int(again[0]), fsm.accept_sink)

    def test_bit31_tokens_roundtrip(self):
        # Token ids 31 and 63 exercise the sign bit of each packed int32 word.
        fsm = DeviceTokenFSM.compile(
            transitions={0: {31: 1}, 1: {63: 0}},
            start_state=0,
            final_states={0},
            vocab_size=64,
            eos_token_ids={7},
        )
        logits = torch.zeros(1, 64)
        fsm.mask_logits_(logits, torch.tensor([fsm.start_state], dtype=torch.int32))
        allowed = [i for i in range(64) if logits[0, i] == 0]
        self.assertEqual(allowed, [7, 31])  # EOS (final state) + token 31


class TestRegexBridge(CustomTestCase):
    def test_token_dfa_from_char_fsm_walk(self):
        # chars: a->1 from 0; b->0 from 1. Token 'ab' walks 0->1->0.
        transitions, initial, finals = token_dfa_from_char_fsm(
            char_transitions={0: {"a": 1}, 1: {"b": 0}},
            char_initial=0,
            char_finals={0},
            token_strings=[None, "a", "b", "ab", ""],
        )
        self.assertEqual(transitions[0], {1: 1, 3: 0})  # empty string disallowed
        self.assertEqual(transitions[1], {2: 0})
        self.assertEqual(initial, 0)
        self.assertEqual(finals, {0})

    def test_unnamed_chars_follow_the_anything_else_edge(self):
        # Alphabet names only 'a'; 'z' is unnamed and must take the catch-all
        # edge. Without it 'z' would be a dead end and token 2 illegal.
        transitions, initial, finals = token_dfa_from_char_fsm(
            char_transitions={0: {"a": 1}},
            char_initial=0,
            char_finals={1},
            token_strings=[None, "a", "z"],
            else_transitions={0: 1},
            explicit_chars={"a"},
        )
        self.assertEqual(transitions[0], {1: 1, 2: 1})

    def test_named_char_without_an_edge_still_rejects(self):
        # 'b' is named by the alphabet, so its absence from state 0 is a real
        # reject and must not silently fall through to the catch-all.
        transitions, _, _ = token_dfa_from_char_fsm(
            char_transitions={0: {"a": 1}},
            char_initial=0,
            char_finals={1},
            token_strings=[None, "a", "b", "z"],
            else_transitions={0: 1},
            explicit_chars={"a", "b"},
        )
        self.assertEqual(transitions[0], {1: 1, 3: 1})  # 'b' excluded, 'z' kept

    def test_json_string_field_admits_a_real_value(self):
        """A JSON string body is a negated class, which interegular routes
        through `anything_else`. Dropping that edge leaves every string-typed
        field able to emit only "", so tool arguments silently come back empty.
        """
        try:
            from outlines_core.fsm.json_schema import build_regex_from_schema
        except ImportError:
            self.skipTest("outlines_core not installed")

        regex = build_regex_from_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                }
            )
        )
        target = '{"location":"Beijing"}'
        # One token per distinct character, so tokenisation cannot be the
        # reason a value is rejected.
        token_strings = [None] + sorted(set(target)) + [None]
        eos = len(token_strings) - 1
        fsm = compile_regex_to_token_fsm(
            regex=regex,
            token_strings=token_strings,
            vocab_size=len(token_strings),
            eos_token_ids={eos},
        )
        g = DeviceFSMGrammar(fsm)
        for ch in target:
            g.accept_token(token_strings.index(ch))

    def test_regex_end_to_end_accepts_and_rejects(self):
        fsm = _compile_toy_fsm()
        g = DeviceFSMGrammar(fsm)
        for t in (1, 3, 2):  # 'a' + 'ab' + 'b' matches a(ab)*b
            g.accept_token(t)
        self.assertFalse(g.is_terminated())
        g.accept_token(EOS)
        self.assertTrue(g.is_terminated())

        g2 = DeviceFSMGrammar(fsm)
        with self.assertRaises(ValueError):
            g2.accept_token(2)  # 'b' first is illegal


class TestDeviceFSMGrammarMirror(CustomTestCase):
    def test_rollback_restores_state(self):
        g = DeviceFSMGrammar(_compile_toy_fsm())
        g.accept_token(1)
        state_after_a = g.state
        g.accept_token(3)
        g.rollback(1)
        self.assertEqual(g.state, state_after_a)
        with self.assertRaises(ValueError):
            g.rollback(5)

    def test_copy_gives_fresh_state_shared_tables(self):
        g = DeviceFSMGrammar(_compile_toy_fsm())
        g.accept_token(1)
        fresh = g.copy()
        self.assertIs(fresh.fsm, g.fsm)
        self.assertEqual(fresh.state, g.fsm.start_state)

    def test_vocab_mask_interface_matches_device_masking(self):
        g = DeviceFSMGrammar(_compile_toy_fsm())
        g.accept_token(1)  # after 'a'
        mask = g.allocate_vocab_mask(vocab_size=VOCAB_SIZE, batch_size=2, device="cpu")
        g.fill_vocab_mask(mask, 1)
        logits = torch.zeros(2, VOCAB_SIZE)
        DeviceFSMGrammar.apply_vocab_mask(logits, mask)
        # Row 0's mask was never filled (all-zero words) -> everything -inf;
        # row 1 mirrors the device-side allowed set after 'a'.
        self.assertTrue(torch.isinf(logits[0]).all())
        self.assertEqual([i for i in range(VOCAB_SIZE) if logits[1, i] == 0], [1, 2, 3])

    def test_try_jump_forward_disabled(self):
        g = DeviceFSMGrammar(_compile_toy_fsm())
        self.assertIsNone(g.try_jump_forward(tokenizer=None))


class TestDflashVerifyHelpers(CustomTestCase):
    """The integration invariants dflash_worker_v2 relies on, exercised with
    the real (unmodified) compute_dflash_correct_drafts_and_bonus."""

    def _masked_target_predict(self, fsm, base_slot, candidates_row, raw_logits):
        logits = raw_logits.clone()
        dflash_mask_block_logits(fsm, base_slot, logits, candidates_row)
        return logits, logits.argmax(dim=-1)

    def test_illegal_draft_clips_commit_and_bonus_is_legal(self):
        fsm = _compile_toy_fsm()
        pool = torch.zeros(8, dtype=torch.int32)
        pool[3] = fsm.advance_host(fsm.start_state, 1)  # base: after 'a'
        slot = pool[3:4]
        # Draft block: [bonus 'a' (already committed), 'ab' (legal),
        # 'c' (ILLEGAL), 'b']. Raw logits love 'c' everywhere, so without
        # masking the target would predict 'c' and accept the illegal draft.
        candidates = torch.tensor([1, 3, 4, 2])
        raw = torch.zeros(4, VOCAB_SIZE)
        raw[:, 4] = 10.0  # 'c' dominates
        raw[:, 3] = 5.0  # 'ab' is the best *legal* token everywhere legal

        masked, target_predict = self._masked_target_predict(fsm, slot, candidates, raw)
        # Position 1 (state after draft 'ab'): 'c' masked out, argmax falls
        # to the best legal token -- which is not the illegal draft 'c', so
        # the accept rule stops there.
        accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates.unsqueeze(0),
            target_predict=target_predict.unsqueeze(0),
        )
        # Draft position 1 ('ab') accepted iff target_predict[0] == 'ab'.
        self.assertEqual(int(target_predict[0]), 3)
        self.assertEqual(int(accept_len[0]), 1)
        # Bonus comes from position accept_len=1, whose state is still valid
        # (DEAD only starts after the illegal token) -- and is grammar-legal.
        after_ab = fsm.advance_host(int(slot[0]), 3)
        self.assertIn(int(bonus[0]), fsm.host_transitions[after_ab])

    def test_advance_committed_only_walks_the_committed_prefix(self):
        fsm = _compile_toy_fsm()
        pool = torch.zeros(8, dtype=torch.int32)
        base = fsm.advance_host(fsm.start_state, 1)  # after 'a'
        pool[5] = base
        slot = pool[5:6]
        # out_tokens ['ab', 'ab', 'b', <junk 'c'>], commit_len=3: the junk
        # tail past the commit must not move the state.
        dflash_advance_committed_state_(
            fsm,
            slot,
            torch.tensor([3, 3, 2, 4]),
            torch.tensor([3], dtype=torch.int32),
        )
        expected = base
        for t in (3, 3, 2):
            expected = fsm.advance_host(expected, t)
        self.assertEqual(int(slot[0]), expected)
        self.assertNotEqual(expected, fsm.dead_state)

    def test_zero_commit_leaves_state_unchanged(self):
        fsm = _compile_toy_fsm()
        pool = torch.zeros(8, dtype=torch.int32)
        base = fsm.advance_host(fsm.start_state, 1)
        pool[0] = base
        dflash_advance_committed_state_(
            fsm,
            pool[0:1],
            torch.tensor([4, 4, 4, 4]),
            torch.tensor([0], dtype=torch.int32),
        )
        self.assertEqual(int(pool[0]), base)

    def test_masking_does_not_write_the_pool(self):
        fsm = _compile_toy_fsm()
        pool = torch.zeros(8, dtype=torch.int32)
        base = fsm.advance_host(fsm.start_state, 1)
        pool[2] = base
        logits = torch.zeros(4, VOCAB_SIZE)
        dflash_mask_block_logits(fsm, pool[2:3], logits, torch.tensor([1, 3, 4, 2]))
        self.assertEqual(int(pool[2]), base)


class _ToyTokenizer:
    """Deterministic stand-in exposing the tokenizer surface
    extract_token_strings/DeviceFSMGrammarBackend need."""

    all_special_ids = [EOS]
    eos_token_id = EOS

    def convert_ids_to_tokens(self, ids):
        return [
            {0: "<pad>", 1: "a", 2: "b", 3: "ab", 4: "c", 5: "</s>"}[i] for i in ids
        ]


class TestReasonerBackendContract(CustomTestCase):
    """ReasonerGrammarBackend wraps whatever grammar backend is configured
    whenever --reasoning-parser is set, and reads these off the BACKEND rather
    than off a grammar object. A backend missing them fails at request time
    with an AttributeError surfaced as "Failed to compile json grammar" -- not
    at startup -- so the combination looks healthy until the first constrained
    request arrives.
    """

    REASONER_BACKEND_ATTRS = (
        "allocate_vocab_mask",
        "move_vocab_mask",
        "apply_vocab_mask",
        "_init_value_dispatch",
        "is_support_token_filter",
        "set_token_filter",
    )

    def test_device_fsm_backend_satisfies_reasoner_contract(self):
        for name in self.REASONER_BACKEND_ATTRS:
            self.assertTrue(
                hasattr(DeviceFSMGrammarBackend, name),
                f"DeviceFSMGrammarBackend is missing {name!r}; "
                "device_fsm cannot be combined with --reasoning-parser",
            )

    def test_device_fsm_is_a_cli_choice(self):
        # A rebase once dropped it from the list, and argparse then rejected
        # every launcher that passes --grammar-backend device_fsm.
        from sglang.srt.arg_groups.choices import GRAMMAR_BACKEND_CHOICES

        self.assertIn("device_fsm", GRAMMAR_BACKEND_CHOICES)

    def test_backend_allocate_vocab_mask_shape(self):
        mask = DeviceFSMGrammarBackend.allocate_vocab_mask(1000, 2, "cpu")
        # one int32 word per 32 vocab entries, one row per sequence
        self.assertEqual(tuple(mask.shape), (2, 32))
        self.assertEqual(mask.dtype, torch.int32)
        self.assertTrue(bool((mask == 0).all()))

    def test_backend_and_grammar_allocate_agree(self):
        g = DeviceFSMGrammar(_compile_toy_fsm())
        self.assertEqual(
            tuple(g.allocate_vocab_mask(1000, 2, "cpu").shape),
            tuple(DeviceFSMGrammarBackend.allocate_vocab_mask(1000, 2, "cpu").shape),
        )


class TestDeviceFSMGrammarBackend(CustomTestCase):
    def test_dispatch_regex_produces_working_grammar(self):
        backend = DeviceFSMGrammarBackend(
            _ToyTokenizer(), vocab_size=VOCAB_SIZE, eos_token_ids={EOS}
        )
        g = backend.dispatch_regex("a(ab)*b")
        self.assertIsInstance(g, DeviceFSMGrammar)
        for t in (1, 2, EOS):
            g.accept_token(t)
        self.assertTrue(g.is_terminated())

    def test_dispatch_ebnf_not_supported(self):
        backend = DeviceFSMGrammarBackend(
            _ToyTokenizer(), vocab_size=VOCAB_SIZE, eos_token_ids={EOS}
        )
        result = backend.dispatch_ebnf("root ::= 'x'")
        self.assertNotIsInstance(result, DeviceFSMGrammar)

    def test_uncompilable_regex_returns_invalid_object(self):
        backend = DeviceFSMGrammarBackend(
            _ToyTokenizer(), vocab_size=VOCAB_SIZE, eos_token_ids={EOS}
        )
        # 'z' is not producible by any token -> the DFA's start state allows
        # nothing -> compile must fail visibly, not permit everything.
        result = backend.dispatch_regex("z+")
        self.assertIsInstance(result, InvalidGrammarObject)
