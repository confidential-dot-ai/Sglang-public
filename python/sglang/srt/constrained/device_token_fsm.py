# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Device-resident token-level DFA grammars, for capture-safe constrained
decoding (grammar inside DFlash verify and CCD's captured WHILE loop).

Every other grammar backend (xgrammar, outlines, llguidance) advances a
host-resident, per-request CPU matcher object between ticks and rebuilds a
token bitmask on the host every step. That is fine for the eager path, but
fundamentally incompatible with CCD (dflash_cc_conditional_decode.py): a CUDA
graph replay re-executes only the recorded GPU kernels, never the Python code
that drove them, so any grammar whose state lives on the host silently
freezes at its capture-time state.

This module takes the other route for the grammar class where it is possible
at all: grammars expressible as a *token-level deterministic finite automaton*
(regex, and JSON schema via schema->regex conversion). A DFA's entire behavior
can be precompiled into two device tensors --

  - ``mask_words``:   per-state packed allowed-token bitmask  [S+1, ceil(V/32)]
  - ``trans_keys`` /
    ``trans_vals``:   the sparse transition map, as a sorted key array
                      (``state * vocab_size + token``) plus next-state values

-- after which *both* per-tick grammar operations (mask the target logits,
advance the state over committed tokens) are pure, fixed-shape tensor ops
(gather / searchsorted / where / masked_fill_) on persistent buffers: exactly
what a captured region can record once and replay forever. Per-request state
is a single int32 living in a persistent pool tensor owned by the DFlash
worker, indexed by ``req.req_pool_idx``.

What is deliberately NOT covered: EBNF / arbitrary CFGs (xgrammar's pushdown
automaton). A PDA's unbounded stack cannot be tabulated into fixed-shape
tensors, so those grammars stay on the host backends and stay excluded from
CCD (check_ccd_eligible).

DFlash/CCD integration invariants (relied on by dflash_worker_v2.py):

  1. The target-verify logits at block position j are masked with the state
     reached by advancing through draft candidates[1..j] (candidates[0] is the
     previous tick's already-committed bonus token, so the base state already
     includes it). A grammar-illegal draft candidate therefore *cannot* equal
     the masked target argmax at its position, so the existing accept/bonus
     kernels clip the commit at the first illegal draft token with no changes,
     and the bonus token -- taken from masked logits at a position whose state
     is still valid -- is always grammar-legal.
  2. The DEAD state (reached by an illegal draft token) uses an allow-all
     mask. This is safe, not permissive: a position can only be in DEAD if an
     earlier candidate was illegal, and (1) already guarantees the commit
     stops before any DEAD position, so DEAD-state masks never influence a
     committed token. Allow-all just keeps the argmax well-defined there.
  3. Committed tokens are always legal (accepted drafts matched a masked
     target prediction; the bonus is masked-legal), so the post-commit state
     scan never lands in DEAD.
  4. After EOS is accepted (from a final state) the FSM sits in an accept
     sink that allows only EOS, with an EOS self-loop -- so CCD ticks that
     keep running a finished row (it does, masking them from output via
     out_len) keep committing EOS harmlessly instead of going off-grammar.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)

logger = logging.getLogger(__name__)


class DeviceTokenFSM:
    """A token-level DFA compiled to tensors. Shared, immutable-after-compile;
    per-request state lives outside this object (an int on the host mirror, a
    pool slot on the device)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        num_real_states: int,
        start_state: int,
        accept_sink: int,
        eos_token_ids: List[int],
        mask_words: torch.Tensor,
        trans_keys: torch.Tensor,
        trans_vals: torch.Tensor,
        host_transitions: Dict[int, Dict[int, int]],
    ):
        self.vocab_size = int(vocab_size)
        self.num_real_states = int(num_real_states)
        self.start_state = int(start_state)
        self.accept_sink = int(accept_sink)
        # DEAD is the extra, last mask row; it has no outgoing transitions, so
        # advance() from DEAD always returns DEAD (missing key -> DEAD).
        self.dead_state = int(num_real_states)
        self.eos_token_ids = sorted(int(e) for e in eos_token_ids)
        self.mask_words = mask_words  # [S+1, W] int32, bit set = allowed
        self.trans_keys = trans_keys  # [E] int64, sorted
        self.trans_vals = trans_vals  # [E] int32
        # Host-side copy of the transition map, for the BaseGrammarObject
        # mirror (accept_token) and for fill_vocab_mask.
        self.host_transitions = host_transitions
        device = mask_words.device
        v = self.vocab_size
        # Per-token word index / bit position, precomputed so mask_logits_
        # does no per-call arange.
        self._tok_word_idx = torch.arange(v, dtype=torch.int64, device=device) // 32
        self._tok_bit = (torch.arange(v, dtype=torch.int64, device=device) % 32).to(
            torch.int32
        )
        # One instance per device, lazily created via on_device().
        self._device_cache: Dict[str, "DeviceTokenFSM"] = {str(device): self}

    @staticmethod
    def compile(
        *,
        transitions: Dict[int, Dict[int, int]],
        start_state: int,
        final_states: Set[int],
        vocab_size: int,
        eos_token_ids: Set[int],
    ) -> "DeviceTokenFSM":
        """Build from an explicit token-level DFA. State ids may be arbitrary
        ints; they are remapped to a dense range. ``transitions[s][t]`` is the
        next state after emitting token ``t`` in state ``s``; a token with no
        entry is disallowed in that state. ``final_states`` are the states
        where the grammar is complete, i.e. where EOS becomes legal.

        Raises ValueError if any reachable non-final state allows no token at
        all (the target argmax would have nothing legal to produce there, so
        generation could never finish the grammar)."""
        if not eos_token_ids:
            raise ValueError("at least one EOS token id is required")
        for eos in eos_token_ids:
            if not (0 <= eos < vocab_size):
                raise ValueError(
                    f"eos_token_id {eos} out of range for vocab_size {vocab_size}"
                )
        all_states = set(transitions.keys()) | {start_state} | set(final_states)
        for outs in transitions.values():
            all_states.update(outs.values())
        remap = {s: i for i, s in enumerate(sorted(all_states))}
        accept_sink = len(remap)  # one extra real state
        num_real_states = len(remap) + 1

        dense: Dict[int, Dict[int, int]] = {remap[s]: {} for s in all_states}
        for s, outs in transitions.items():
            for tok, nxt in outs.items():
                if not (0 <= tok < vocab_size):
                    raise ValueError(
                        f"transition token {tok} out of range for vocab_size {vocab_size}"
                    )
                dense[remap[s]][int(tok)] = remap[nxt]
        for s in final_states:
            for eos in eos_token_ids:
                dense[remap[s]][int(eos)] = accept_sink
        dense[accept_sink] = {int(eos): accept_sink for eos in eos_token_ids}

        for s, outs in dense.items():
            if not outs:
                raise ValueError(
                    f"token-DFA state {s} allows no token (not final, no "
                    f"outgoing transitions); generation could never satisfy "
                    f"the grammar from this state"
                )

        num_words = (vocab_size + 31) // 32
        # +1 row for DEAD (allow-all; see module docstring invariant 2).
        mask_words = torch.zeros(
            (num_real_states + 1, num_words), dtype=torch.int32, device="cpu"
        )
        keys: List[int] = []
        vals: List[int] = []
        mask_np = mask_words.numpy()
        for s, outs in dense.items():
            for tok, nxt in outs.items():
                # c_int32: bit 31 as a raw Python int overflows the int32 view.
                mask_np[s, tok // 32] |= ctypes.c_int32(1 << (tok % 32)).value
                keys.append(s * vocab_size + tok)
                vals.append(nxt)
        mask_words[num_real_states].fill_(-1)  # DEAD: all bits set

        order = sorted(range(len(keys)), key=keys.__getitem__)
        trans_keys = torch.tensor([keys[i] for i in order], dtype=torch.int64)
        trans_vals = torch.tensor([vals[i] for i in order], dtype=torch.int32)

        return DeviceTokenFSM(
            vocab_size=vocab_size,
            num_real_states=num_real_states,
            start_state=remap[start_state],
            accept_sink=accept_sink,
            eos_token_ids=list(eos_token_ids),
            mask_words=mask_words,
            trans_keys=trans_keys,
            trans_vals=trans_vals,
            host_transitions=dense,
        )

    def on_device(self, device) -> "DeviceTokenFSM":
        """The same FSM with its tensors on `device` (cached; tables are
        shared per grammar object across all requests using it)."""
        key = str(torch.device(device))
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        moved = DeviceTokenFSM(
            vocab_size=self.vocab_size,
            num_real_states=self.num_real_states,
            start_state=self.start_state,
            accept_sink=self.accept_sink,
            eos_token_ids=self.eos_token_ids,
            mask_words=self.mask_words.to(device),
            trans_keys=self.trans_keys.to(device),
            trans_vals=self.trans_vals.to(device),
            host_transitions=self.host_transitions,
        )
        self._device_cache[key] = moved
        moved._device_cache = self._device_cache
        return moved

    # ---- host-side (mirror) operations ------------------------------------

    def advance_host(self, state: int, token: int) -> int:
        return self.host_transitions.get(state, {}).get(token, self.dead_state)

    # ---- device-side, capture-safe operations -----------------------------

    def advance(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Next state for each (state, token) pair; DEAD when no transition
        exists. Fixed-shape tensor ops only -- safe inside CUDA graph capture."""
        keys = states.to(torch.int64) * self.vocab_size + tokens.to(torch.int64)
        num_edges = self.trans_keys.numel()
        if num_edges == 0:
            return torch.full_like(states, self.dead_state)
        idx = torch.searchsorted(self.trans_keys, keys)
        idx_c = idx.clamp(max=num_edges - 1)
        hit = (idx < num_edges) & (self.trans_keys[idx_c] == keys)
        nxt = self.trans_vals[idx_c].to(states.dtype)
        return torch.where(hit, nxt, torch.full_like(states, self.dead_state))

    def mask_logits_(self, logits: torch.Tensor, states: torch.Tensor) -> None:
        """In-place: set logits of tokens not allowed in each row's state to
        -inf. `logits` is [..., vocab_size], `states` is [...] (matching
        leading shape). Fixed-shape tensor ops only -- capture-safe."""
        words = self.mask_words[states.to(torch.int64)]  # [..., W]
        picked = words[..., self._tok_word_idx]  # [..., V]
        allowed = (picked >> self._tok_bit) & 1
        logits.masked_fill_(allowed == 0, float("-inf"))


def dflash_mask_block_logits(
    fsm: DeviceTokenFSM,
    base_state: torch.Tensor,
    logits_block: torch.Tensor,
    candidates_row: torch.Tensor,
) -> None:
    """Mask one request's target-verify logits for a whole DFlash block,
    in-place. `base_state` is that request's current-state pool slot view
    ([1], int32; NOT written here), `logits_block` is [block_size, vocab],
    `candidates_row` is [block_size] draft tokens where candidates_row[0] is
    the previous tick's committed bonus token (already reflected in
    base_state, so position 0 is masked with base_state directly).

    The Python loop unrolls to block_size fixed tensor ops at CUDA graph
    capture time; nothing here branches on tensor values from the host."""
    block_size = logits_block.shape[0]
    state = base_state
    fsm.mask_logits_(logits_block[0:1], state)
    for j in range(1, block_size):
        state = fsm.advance(state, candidates_row[j : j + 1])
        fsm.mask_logits_(logits_block[j : j + 1], state)


def dflash_advance_committed_state_(
    fsm: DeviceTokenFSM,
    state_slot: torch.Tensor,
    out_tokens_row: torch.Tensor,
    commit_len: torch.Tensor,
) -> None:
    """Advance one request's pool state over exactly the committed prefix of
    this tick's out_tokens, in-place on the pool slot. `state_slot` is the [1]
    int32 pool view, `out_tokens_row` is [block_size], `commit_len` is [1]
    (int32). Same capture-safety contract as dflash_mask_block_logits."""
    block_size = out_tokens_row.shape[0]
    state = state_slot.clone()
    for j in range(block_size):
        nxt = fsm.advance(state, out_tokens_row[j : j + 1])
        state = torch.where(commit_len > j, nxt, state)
    state_slot.copy_(state)


def _allocate_token_bitmask(vocab_size: int, batch_size: int, device) -> torch.Tensor:
    """One int32 word per 32 vocab entries, one row per sequence.

    Shared by DeviceFSMGrammar and DeviceFSMGrammarBackend: the sampling path
    allocates through the grammar object, while ReasonerGrammarBackend
    allocates through the backend (see DeviceFSMGrammarBackend below).
    """
    return torch.zeros(
        (batch_size, (vocab_size + 31) // 32), dtype=torch.int32, device=device
    )


class DeviceFSMGrammar(BaseGrammarObject):
    """BaseGrammarObject adapter over a DeviceTokenFSM.

    Plays two roles at once:
      - a normal host-side grammar object (accept_token / fill_vocab_mask /
        apply_vocab_mask), so the ordinary non-speculative sampler path and
        the scheduler's result processing work unchanged;
      - the carrier of the shared device tables (`.fsm` / `.fsm.on_device`),
        which DFlash reads for in-graph masking and state advancement.

    The host state here is a *mirror*: during a CCD chunk the device pool is
    authoritative and this mirror lags, then catches up as the chunk's
    per-tick results flow through the scheduler's normal accept_token calls
    (batch_result_processor._accept_grammar_tokens). Both advance the same
    DFA over the same committed tokens, so they always agree between chunks.
    """

    def __init__(self, fsm: DeviceTokenFSM):
        super().__init__()
        self.fsm = fsm
        self.state: int = fsm.start_state
        self._history: List[int] = []

    def accept_token(self, token: int) -> None:
        nxt = self.fsm.advance_host(self.state, token)
        if nxt == self.fsm.dead_state:
            raise ValueError(
                f"token {token} is not allowed by the grammar in state {self.state}"
            )
        self._history.append(self.state)
        self.state = nxt

    def rollback(self, k: int) -> None:
        if k > len(self._history):
            raise ValueError(
                f"cannot roll back {k} tokens; only {len(self._history)} accepted"
            )
        for _ in range(k):
            self.state = self._history.pop()

    def is_terminated(self) -> bool:
        return self.state == self.fsm.accept_sink

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return _allocate_token_bitmask(vocab_size, batch_size, device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        vocab_mask[idx] = self.fsm.mask_words[self.state].to(vocab_mask.device)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        vocab_size = logits.shape[-1]
        tok = torch.arange(vocab_size, dtype=torch.int64, device=logits.device)
        words = vocab_mask[..., tok // 32]
        allowed = (words >> (tok % 32).to(torch.int32)) & 1
        logits.masked_fill_(allowed == 0, float("-inf"))

    def copy(self) -> "DeviceFSMGrammar":
        # Fresh state over the same (shared, immutable) tables -- used by the
        # backend cache to hand the same compiled grammar to a new request.
        return DeviceFSMGrammar(self.fsm)

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        return None

    def __repr__(self):
        return (
            f"DeviceFSMGrammar(states={self.fsm.num_real_states}, "
            f"state={self.state}, edges={self.fsm.trans_keys.numel()})"
        )


def _gpt2_byte_decoder() -> Dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: printable stand-in char -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def extract_token_strings(tokenizer, vocab_size: int) -> List[Optional[str]]:
    """Best-effort per-token surface string for each vocab id, or None for
    tokens that should never be allowed by a grammar (special tokens,
    undecodable ids). Handles the two dominant conventions: byte-level BPE
    ('Ġ'/'Ċ' stand-in alphabet, decoded via the GPT-2 byte map) and
    sentencepiece ('▁' word-boundary marker)."""
    ids = list(range(vocab_size))
    try:
        pieces = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        pieces = [None] * vocab_size
        for i in ids:
            try:
                pieces[i] = tokenizer.convert_ids_to_tokens([i])[0]
            except Exception:
                pieces[i] = None
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    byte_level = any(p is not None and "Ġ" in p for p in pieces)
    byte_decoder = _gpt2_byte_decoder() if byte_level else None

    out: List[Optional[str]] = []
    for i, piece in enumerate(pieces):
        if piece is None or i in special or not isinstance(piece, str):
            out.append(None)
            continue
        if byte_decoder is not None:
            try:
                raw = bytes(byte_decoder[c] for c in piece)
                out.append(raw.decode("utf-8", errors="strict"))
            except (KeyError, UnicodeDecodeError):
                # Partial UTF-8 byte tokens can't be walked through a
                # character DFA; disallow them. (A byte-level DFA would
                # recover these; documented follow-up.)
                out.append(None)
        else:
            out.append(piece.replace("▁", " "))
    return out


def token_dfa_from_char_fsm(
    *,
    char_transitions: Dict[int, Dict[str, int]],
    char_initial: int,
    char_finals: Set[int],
    token_strings: List[Optional[str]],
    else_transitions: Optional[Dict[int, int]] = None,
    explicit_chars: Optional[Set[str]] = None,
) -> Tuple[Dict[int, Dict[int, int]], int, Set[int]]:
    """Lift a character-level DFA to a token-level DFA over a vocabulary:
    token t is allowed in state s iff walking t's characters from s never
    leaves the DFA, and its transition target is the walk's end state.
    Assumes the char DFA is trim (every state can reach a final state), which
    holds for interegular's reduced DFAs. Empty-string tokens are disallowed
    (they would create no-progress self-loops).

    ``else_transitions`` carries interegular's ``anything_else`` edge per state
    and ``explicit_chars`` the characters the alphabet names outright. The
    alphabet partitions every character into either a named symbol or that
    catch-all, so a character the alphabet does not name must follow the
    catch-all edge instead of being treated as a dead end. Dropping it would
    silently narrow every negated character class -- which is how a JSON string
    body is expressed -- down to whichever characters happen to appear
    literally elsewhere in the same pattern."""
    else_transitions = else_transitions or {}
    explicit_chars = explicit_chars or set()
    token_transitions: Dict[int, Dict[int, int]] = {}
    for s in set(char_transitions) | set(else_transitions):
        outs: Dict[int, int] = {}
        for tid, text in enumerate(token_strings):
            if not text:
                continue
            cur = s
            ok = True
            for ch in text:
                step = char_transitions.get(cur)
                if step is not None and ch in step:
                    cur = step[ch]
                    continue
                # Unnamed character: the catch-all edge is the only legal move.
                # A *named* character with no edge here is a genuine reject.
                nxt = else_transitions.get(cur)
                if nxt is None or ch in explicit_chars:
                    ok = False
                    break
                cur = nxt
            if ok:
                outs[tid] = cur
        if outs:
            token_transitions[s] = outs
    return token_transitions, char_initial, set(char_finals)


def compile_regex_to_token_fsm(
    *,
    regex: str,
    token_strings: List[Optional[str]],
    vocab_size: int,
    eos_token_ids: Set[int],
) -> DeviceTokenFSM:
    """regex -> char DFA (interegular) -> token DFA -> DeviceTokenFSM.

    Compile cost is O(states x vocab x token_len) in Python; the backend's
    dispatch cache keeps this a per-(regex, tokenizer) one-time cost."""
    import interegular

    fsm = interegular.parse_pattern(regex).to_fsm().reduce()
    # interegular FSMs map input chars through an alphabet of symbol classes;
    # expand to a plain char -> next-state map per state. `anything_else` is
    # the alphabet's catch-all for every character no symbol names, so it is
    # kept as a separate per-state edge rather than expanded into a concrete
    # (unbounded) character set. The lifter consults it for exactly those
    # unnamed characters. It cannot be dropped: outlines renders a JSON string
    # body as a negated class, which interegular represents through this edge,
    # so discarding it would admit only the empty string for every string-typed
    # field.
    from interegular.fsm import anything_else

    char_transitions: Dict[int, Dict[str, int]] = {}
    else_transitions: Dict[int, int] = {}
    explicit_chars: Set[str] = {
        ch
        for chars in fsm.alphabet.by_transition.values()
        for ch in chars
        if ch is not anything_else
    }
    for state, symbol_map in fsm.map.items():
        outs: Dict[str, int] = {}
        for symbol, chars in fsm.alphabet.by_transition.items():
            nxt = symbol_map.get(symbol)
            if nxt is None:
                continue
            for ch in chars:
                if ch is anything_else:
                    else_transitions[int(state)] = int(nxt)
                else:
                    outs[ch] = int(nxt)
        if outs:
            char_transitions[int(state)] = outs

    token_transitions, initial, finals = token_dfa_from_char_fsm(
        char_transitions=char_transitions,
        char_initial=int(fsm.initial),
        char_finals={int(s) for s in fsm.finals},
        token_strings=token_strings,
        else_transitions=else_transitions,
        explicit_chars=explicit_chars,
    )
    return DeviceTokenFSM.compile(
        transitions=token_transitions,
        start_state=initial,
        final_states=finals,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
    )


class DeviceFSMGrammarBackend(BaseGrammarBackend):
    """Grammar backend producing DeviceFSMGrammar objects (the only grammar
    kind CCD accepts; see check_ccd_eligible). Supports regex directly and
    JSON schema via outlines-core's schema->regex conversion when that
    package is installed. EBNF and structural tags are CFG-class and cannot
    be tabulated into a DFA -- permanently out of scope for this backend."""

    def __init__(self, tokenizer, vocab_size: int, eos_token_ids: Set[int]):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.eos_token_ids = {int(e) for e in eos_token_ids}
        self._token_strings = extract_token_strings(tokenizer, self.vocab_size)

    # ReasonerGrammarBackend -- installed automatically whenever
    # --reasoning-parser is set -- reads these three off the BACKEND rather
    # than off a grammar object, to build the mask for the reasoning span
    # before any grammar applies. XGrammarGrammarBackend exposes them as
    # staticmethods for the same reason; without them device_fsm and any
    # reasoning parser cannot be used together, which fails at request time
    # with AttributeError surfaced as "Failed to compile json grammar", not at
    # startup. Same implementations DeviceFSMGrammar uses.
    @staticmethod
    def allocate_vocab_mask(vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return _allocate_token_bitmask(vocab_size, batch_size, device)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return DeviceFSMGrammar.move_vocab_mask(vocab_mask, device)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        DeviceFSMGrammar.apply_vocab_mask(logits, vocab_mask)

    def _compile_regex(self, regex: str, kind: str) -> BaseGrammarObject:
        try:
            fsm = compile_regex_to_token_fsm(
                regex=regex,
                token_strings=self._token_strings,
                vocab_size=self.vocab_size,
                eos_token_ids=self.eos_token_ids,
            )
        except Exception as e:
            logger.warning("device_fsm: failed to compile %s %r: %s", kind, regex, e)
            return InvalidGrammarObject(f"device_fsm {kind} compile failed: {e}")
        return DeviceFSMGrammar(fsm)

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        return self._compile_regex(key_string, "regex")

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        try:
            from outlines_core.fsm.json_schema import build_regex_from_schema
        except ImportError:
            try:
                from outlines.fsm.json_schema import build_regex_from_schema
            except ImportError:
                return InvalidGrammarObject(
                    "device_fsm json_schema support requires outlines-core "
                    "(schema->regex conversion); install it or use "
                    "--grammar-backend xgrammar for host-side grammar"
                )
        try:
            regex = build_regex_from_schema(key_string)
        except Exception as e:
            return InvalidGrammarObject(f"device_fsm schema->regex failed: {e}")
        return self._compile_regex(regex, "json_schema")

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("ebnf", key_string)

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("structural_tag", key_string)
