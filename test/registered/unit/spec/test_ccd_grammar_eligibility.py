"""Unit test for check_ccd_eligible's grammar handling -- no server, no
model loading, no GPU.

Host-CPU grammars (xgrammar/outlines/llguidance) must exclude the batch:
CCD's captured forward_batch_generation never calls their per-tick host
machinery (update_regex_vocab_mask/accept_token), so it would silently emit
unconstrained tokens for a request the caller explicitly asked to be
grammar-constrained. Device-FSM grammars (grammar_backend='device_fsm') must
NOT exclude the batch: their masking and state advancement are fixed-shape
tensor ops on persistent device buffers that DFlash's verify step runs
in-graph, so a captured chunk replays them correctly. This is unrelated to
oracle mode -- see test_ccd_oracle_eligibility.py for that exclusion.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

from types import SimpleNamespace

from sglang.srt.constrained.device_token_fsm import (
    DeviceFSMGrammar,
    DeviceTokenFSM,
)
from sglang.srt.speculative.dflash_cc_conditional_decode import (
    SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
    check_ccd_eligible,
)
from sglang.test.test_utils import CustomTestCase



def _prepared_spec_info():
    # check_ccd_eligible only runs on a spec_info the previous decode step
    # already prepared, which it recognizes by nxt_kv_lens_cpu being set.
    return SimpleNamespace(reserved_seq_lens_cpu=None, nxt_kv_lens_cpu=[1])

def _make_device_fsm_grammar():
    fsm = DeviceTokenFSM.compile(
        transitions={0: {1: 0}},
        start_state=0,
        final_states={0},
        vocab_size=8,
        eos_token_ids={7},
    )
    return DeviceFSMGrammar(fsm)


def _make_req(grammar=None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(forced_token_ids=None),
        return_hidden_states=False,
        return_sampling_mask=False,
        output_ids=[0] * (SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN + 1),
        grammar=grammar,
    )


def _make_batch(reqs, has_grammar=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
        return_logprob=False,
        has_grammar=has_grammar,
        spec_info=_prepared_spec_info(),
        reqs=reqs,
    )


def _make_worker():
    return SimpleNamespace(
        use_compact_draft_cache=False,
        _need_mamba_verify_commit=False,
        ccd_lm_head_requires_collective=lambda: False,
    )


class TestCcdGrammarEligibility(CustomTestCase):
    def test_otherwise_eligible_batch_passes(self):
        """Sanity: the fixture itself is a valid CCD-eligible batch when
        has_grammar is False -- otherwise the exclusion tests below would
        trivially pass for the wrong reason."""
        batch = _make_batch([_make_req(), _make_req()], has_grammar=False)
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))

    def test_host_grammar_excludes_the_batch(self):
        host_grammar = object()  # any non-DeviceFSMGrammar grammar object
        batch = _make_batch(
            [_make_req(), _make_req(grammar=host_grammar)], has_grammar=True
        )
        reason = check_ccd_eligible(batch, _make_worker())
        self.assertIsNotNone(reason)
        self.assertIn("device_fsm", reason)

    def test_device_fsm_grammar_is_eligible(self):
        batch = _make_batch(
            [_make_req(), _make_req(grammar=_make_device_fsm_grammar())],
            has_grammar=True,
        )
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))

    def test_missing_has_grammar_attribute_defaults_to_not_excluded(self):
        """Batch objects without a has_grammar attribute at all (getattr
        default) must not be mistaken for "has grammar"."""
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
            sampling_info=SimpleNamespace(is_all_greedy=True),
            return_logprob=False,
            spec_info=_prepared_spec_info(),
            reqs=[_make_req()],
        )
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))
