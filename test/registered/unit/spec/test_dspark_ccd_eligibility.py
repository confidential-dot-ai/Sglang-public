"""Unit tests for CCD eligibility of DSpark workers -- no server, no model
loading, no GPU.

scheduler.py's CCD branches gate on `spec_algorithm.is_dflash_family()`,
which includes DSPARK, so check_ccd_eligible / partition_ccd_eligible run
against DSparkWorkerV2 too. These tests cover the two halves of that
adaptation:

1. The gate's worker-capability hook (`ccd_capture_incompatible_reason`,
   optional via getattr): a worker that defines it can veto capture with a
   precise reason; a worker that doesn't (DFlashWorkerV2, the existing
   mocks) is byte-for-byte unaffected.

2. DSparkWorkerV2's own implementations of the worker contract
   (`ccd_capture_incompatible_reason`, `ccd_lm_head_requires_collective`)
   and DsparkStepObservers.has_capture_unsafe_sink -- exercised as unbound
   methods against SimpleNamespace stubs, so the real gating logic runs
   without constructing a worker (which needs a model and a GPU).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_cc_conditional_decode import (
    SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN,
    check_ccd_eligible,
    partition_ccd_eligible,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.test_utils import CustomTestCase

VETERAN_LEN = SGLANG_CC_CONDITIONAL_DECODE_MIN_OUTPUT_LEN + 1


def _make_req(output_len=VETERAN_LEN):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(top_k=1, forced_token_ids=None),
        return_logprob=False,
        return_hidden_states=False,
        return_sampling_mask=False,
        output_ids=[0] * output_len,
        grammar=None,
    )


def _make_batch(reqs):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
        return_logprob=False,
        has_grammar=False,
        spec_info=SimpleNamespace(
            nxt_kv_lens_cpu=torch.ones(len(reqs), dtype=torch.int32)
        ),
        reqs=reqs,
    )


def _make_dflash_worker():
    """The pre-existing DFlash-shaped mock: deliberately does NOT define
    ccd_capture_incompatible_reason -- DFlashWorkerV2 doesn't either."""
    return SimpleNamespace(
        use_compact_draft_cache=False,
        _need_mamba_verify_commit=False,
        draft_window_size=None,
        ccd_lm_head_requires_collective=lambda: False,
    )


def _make_dspark_worker(capture_reason=None):
    """A DSpark-shaped mock: the same worker contract DSparkWorkerV2 now
    carries explicitly (block_size / compact-cache flags / mamba flag /
    lm-head hook), plus the capability hook."""
    return SimpleNamespace(
        block_size=9,
        use_compact_draft_cache=False,
        _need_mamba_verify_commit=False,
        draft_window_size=None,
        ccd_lm_head_requires_collective=lambda: False,
        ccd_capture_incompatible_reason=lambda: capture_reason,
    )


def _make_dspark_self(
    *,
    schedules_verify_budget=False,
    mode_value="static",
    unsafe_sink=False,
    draft_sampler=object(),
):
    """Stub `self` for calling DSparkWorkerV2's methods unbound."""
    return SimpleNamespace(
        _verify_planner=SimpleNamespace(
            schedules_verify_budget=schedules_verify_budget,
            mode_value=mode_value,
        ),
        _observers=SimpleNamespace(has_capture_unsafe_sink=unsafe_sink),
        _draft_sampler=draft_sampler,
    )


class TestCcdWorkerCapabilityHook(CustomTestCase):
    def test_dflash_shaped_worker_without_hook_unchanged(self):
        """A worker without ccd_capture_incompatible_reason (DFlashWorkerV2)
        must behave exactly as before the hook existed."""
        batch = _make_batch([_make_req(), _make_req()])
        self.assertIsNone(check_ccd_eligible(batch, _make_dflash_worker()))
        reason, eligible = partition_ccd_eligible(batch, _make_dflash_worker())
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 1])

    def test_dspark_shaped_worker_compatible_config_accepted(self):
        batch = _make_batch([_make_req(), _make_req()])
        worker = _make_dspark_worker(capture_reason=None)
        self.assertIsNone(check_ccd_eligible(batch, worker))
        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertIsNone(reason)
        self.assertEqual(eligible, [0, 1])

    def test_hook_reason_gates_whole_batch(self):
        batch = _make_batch([_make_req(), _make_req()])
        veto = "CCD with DSpark requires SGLANG_RAGGED_VERIFY_MODE=static (test)"
        worker = _make_dspark_worker(capture_reason=veto)
        self.assertEqual(check_ccd_eligible(batch, worker), veto)

    def test_hook_reason_is_batch_level_in_partition(self):
        """A capability veto is static worker config: partition must return
        it as the batch-level reason with NO eligible indices, so
        run_ccd_chunk's re-check can never raise on a split sub-batch."""
        batch = _make_batch([_make_req(), _make_req(), _make_req()])
        veto = "CCD with DSpark requires all step observers disabled (test)"
        worker = _make_dspark_worker(capture_reason=veto)
        reason, eligible = partition_ccd_eligible(batch, worker)
        self.assertEqual(reason, veto)
        self.assertEqual(eligible, [])

    def test_pre_hook_gates_still_fire_first(self):
        """The dedicated dflash gates (compact cache, mamba, lm-head) keep
        precedence over the generic hook -- their reasons are more specific."""
        batch = _make_batch([_make_req()])
        worker = _make_dspark_worker(capture_reason="hook reason")
        worker._need_mamba_verify_commit = True
        reason = check_ccd_eligible(batch, worker)
        self.assertIn("mamba", reason)


class TestDSparkWorkerCcdMethods(CustomTestCase):
    """Run DSparkWorkerV2's real methods unbound against stub selves."""

    def test_compatible_static_config_returns_none(self):
        stub = _make_dspark_self()
        self.assertIsNone(DSparkWorkerV2.ccd_capture_incompatible_reason(stub))

    def test_non_static_ragged_mode_vetoes_with_mode_in_reason(self):
        stub = _make_dspark_self(schedules_verify_budget=True, mode_value="compact")
        reason = DSparkWorkerV2.ccd_capture_incompatible_reason(stub)
        self.assertIsNotNone(reason)
        self.assertIn("SGLANG_RAGGED_VERIFY_MODE=static", reason)
        self.assertIn("compact", reason)

    def test_active_observer_sink_vetoes(self):
        stub = _make_dspark_self(unsafe_sink=True)
        reason = DSparkWorkerV2.ccd_capture_incompatible_reason(stub)
        self.assertIsNotNone(reason)
        self.assertIn("observers", reason)

    def test_missing_draft_sampler_vetoes(self):
        stub = _make_dspark_self(draft_sampler=None)
        reason = DSparkWorkerV2.ccd_capture_incompatible_reason(stub)
        self.assertIsNotNone(reason)
        self.assertIn("graph-folded draft sampler", reason)

    def test_lm_head_collective_false_with_folded_sampler(self):
        """With the folded sampler, the shared-LM-head all-gather lives
        inside the pre-captured draft graph -- no in-body collective. Must
        short-circuit before touching the TP group (unit tests have no
        distributed init)."""
        stub = SimpleNamespace(_draft_sampler=object())
        self.assertFalse(DSparkWorkerV2.ccd_lm_head_requires_collective(stub))


class TestObserversCaptureUnsafeSink(CustomTestCase):
    def _make_observers(self, **overrides):
        observers = DsparkStepObservers.__new__(DsparkStepObservers)
        observers._info_dumper = SimpleNamespace(enabled=False)
        observers._block_accept_recorder = None
        observers._sts_collect_path = ""
        for key, value in overrides.items():
            setattr(observers, key, value)
        return observers

    def test_all_sinks_off_is_safe(self):
        self.assertFalse(self._make_observers().has_capture_unsafe_sink)

    def test_info_dumper_enabled_is_unsafe(self):
        observers = self._make_observers(
            _info_dumper=SimpleNamespace(enabled=True)
        )
        self.assertTrue(observers.has_capture_unsafe_sink)

    def test_block_accept_recorder_is_unsafe(self):
        observers = self._make_observers(_block_accept_recorder=object())
        self.assertTrue(observers.has_capture_unsafe_sink)

    def test_sts_collect_path_is_unsafe(self):
        observers = self._make_observers(_sts_collect_path="/tmp/sts")
        self.assertTrue(observers.has_capture_unsafe_sink)
