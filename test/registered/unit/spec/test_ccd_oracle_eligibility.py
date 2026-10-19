"""Unit test for check_ccd_eligible's oracle-mode (forced_token_ids)
handling -- no server, no model loading, no GPU.

Oracle-mode batches are CCD-eligible: DflashCCDChunk.prime_and_capture()
builds the forced-token lookup exactly once, before capture, and stashes it
on chunk_batch._ccd_forced_lookup -- every captured tick's oracle-commit
computation already indexed purely by prefix_lens (== batch.seq_lens,
updated in place every tick), so a lookup built once and reused is
sufficient. See oracle_decode/forced_token_pool.py's module docstring."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

from types import SimpleNamespace

from sglang.srt.speculative.dflash_cc_conditional_decode import (
    SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
    check_ccd_eligible,
)
from sglang.test.test_utils import CustomTestCase



def _prepared_spec_info():
    # check_ccd_eligible only runs on a spec_info the previous decode step
    # already prepared, which it recognizes by nxt_kv_lens_cpu being set.
    return SimpleNamespace(reserved_seq_lens_cpu=None, nxt_kv_lens_cpu=[1])

def _make_req(forced_token_ids=None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(forced_token_ids=forced_token_ids),
        return_hidden_states=False,
        return_sampling_mask=False,
        output_ids=[0] * (SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN + 1),
    )


def _make_batch(reqs):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
        return_logprob=False,
        spec_info=_prepared_spec_info(),
        reqs=reqs,
    )


def _make_worker():
    return SimpleNamespace(
        use_compact_draft_cache=False,
        _need_mamba_verify_commit=False,
        ccd_lm_head_requires_collective=lambda: False,
    )


class TestCcdOracleEligibility(CustomTestCase):
    def test_otherwise_eligible_batch_passes(self):
        """Sanity: the fixture itself is a valid CCD-eligible batch when no
        request has forced_token_ids -- otherwise the tests below would
        trivially pass for the wrong reason."""
        batch = _make_batch([_make_req(), _make_req()])
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))

    def test_forced_token_ids_batch_is_eligible(self):
        batch = _make_batch([_make_req(), _make_req(forced_token_ids=[1, 2, 3])])
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))

    def test_all_forced_token_ids_batch_is_eligible(self):
        batch = _make_batch(
            [_make_req(forced_token_ids=[1, 2, 3]), _make_req(forced_token_ids=[4, 5])]
        )
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))

    def test_forced_token_ids_none_is_not_excluded(self):
        """forced_token_ids=None (the default -- no oracle mode) must not be
        mistaken for "has forced_token_ids"."""
        batch = _make_batch([_make_req(forced_token_ids=None)])
        self.assertIsNone(check_ccd_eligible(batch, _make_worker()))
