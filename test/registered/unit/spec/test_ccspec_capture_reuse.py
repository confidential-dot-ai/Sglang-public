"""Unit tests for CCSpec cross-chunk capture reuse
(model_executor/ccspec_decode.py, SGLANG_CCSPEC_CAPTURE_REUSE) -- the
validity fingerprint, the per-chunk re-arm, and the strict
match-or-recapture decision. CPU-runnable: no server, no model, no GPU;
same fixture conventions as test_ccspec_decode.py (in particular the
sampling mocks pin exactly what a real copy_for_forward() batch carries in
the no-penalty case).

Covered here (host-side logic + CPU-tensor buffer semantics):
  - ccspec_capture_fingerprint: equal for an unchanged batch; mismatched on
    every field it pins (bs, request set/order, req_pool_idx, chunk_cap,
    eos_ids, req_to_token pool identity, runner/backend identity).
  - CCSpecChunk.reuse_mismatch_reason: the match case plus each rejection
    (no capture, fingerprint mismatch, forced-token stash rebound, stash
    rid mismatch).
  - CCSpecChunk._reserve_and_prewrite_slots: reuse path REWRITES the
    slot_table values in the SAME storage (pointer identity -- the captured
    graph gathers from it by pointer) and redoes the req_to_token
    pre-writes at the new S0-derived positions; fresh path binds.
  - CCSpecChunk.rearm_for_reuse: priming-token seeding, seq_lens_buf
    refresh from a REBOUND live batch.seq_lens (the prepare_for_decode
    rebind this whole feature exists to survive), req_pool_indices
    defensive refresh, and the loud failure on an ineligible batch.
  - CCSpecChunk.sync_seq_lens_back: terminal value mirrored to the rebound
    live tensor; no-op when aliased.

GPU-only parts (capture, launch, relaunch, read_history) are validated on
the real guest -- see the GPU validation checklist in the change
description.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.ccspec_decode import (  # noqa: E402
    CCSpecChunk,
    ccspec_capture_fingerprint,
    ccspec_slot_positions,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

CHUNK_CAP = 4
EOS = [2]
DEVICE = torch.device("cpu")


def _make_req(rid, pool_idx, *, output_len=10_000):
    return types.SimpleNamespace(
        rid=rid,
        req_pool_idx=pool_idx,
        sampling_params=types.SimpleNamespace(
            top_k=1, max_new_tokens=100_000, forced_token_ids=None
        ),
        return_logprob=False,
        return_hidden_states=False,
        return_sampling_mask=False,
        output_ids=[0] * output_len,
        grammar=None,
    )


def _permissive_sampling_info():
    sampling_info = MagicMock()
    sampling_info.is_all_greedy = True
    sampling_info.penalizer_orchestrator = None
    sampling_info.acc_scaling_penalties = None
    sampling_info.acc_additive_penalties = None
    sampling_info.logit_bias = None
    sampling_info.has_custom_logit_processor = False
    return sampling_info


def _make_model_runner():
    return types.SimpleNamespace(
        attn_backend=types.SimpleNamespace(supports_ccspec_capture=True),
        ps=types.SimpleNamespace(tp_size=1, pp_size=1),
    )


def _make_batch(reqs, *, seq_len=100, pool_width=4096, **overrides):
    bs = len(reqs)
    batch = types.SimpleNamespace(
        reqs=reqs,
        batch_size=lambda: bs,
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
        dllm_config=None,
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        global_num_tokens=None,
        hisparse_coordinator=None,
        sampling_info=_permissive_sampling_info(),
        has_grammar=False,
        token_to_kv_pool_allocator=types.SimpleNamespace(
            page_size=1, available_size=lambda: 10**9
        ),
        seq_lens=torch.full((bs,), seq_len, dtype=torch.int64),
        seq_lens_cpu=torch.full((bs,), seq_len, dtype=torch.int64),
        req_pool_indices=torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64
        ),
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=torch.zeros(
                (max(r.req_pool_idx for r in reqs) + 1 if reqs else 1, pool_width),
                dtype=torch.int32,
            )
        ),
        tree_cache=object(),
        device=DEVICE,
        # Forced-token stash state as prime_and_capture leaves it in the
        # common (no oracle) case.
        _ccd_forced_lookup=None,
        _ccd_forced_lookup_rids=[r.rid for r in reqs],
    )
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


def _fingerprint(batch, runner, *, chunk_cap=CHUNK_CAP, eos=None):
    return ccspec_capture_fingerprint(
        batch,
        chunk_cap=chunk_cap,
        eos_ids_host=eos if eos is not None else EOS,
        model_runner=runner,
    )


def _make_chunk(batch, runner, *, chunk_cap=CHUNK_CAP, captured=True):
    """A CCSpecChunk in the state prime_and_capture leaves it in, minus the
    GPU-only capture (stubbed by a sentinel _cond_graph)."""
    bs = batch.batch_size()
    chunk = CCSpecChunk(
        runner,
        bs=bs,
        chunk_cap=chunk_cap,
        eos_ids=torch.tensor(EOS, dtype=torch.int64),
        device=DEVICE,
        eos_ids_host=list(EOS),
    )
    chunk.batch = batch
    chunk.slot_table = torch.zeros((chunk_cap, bs), dtype=torch.int64)
    chunk.seq_lens_buf = batch.seq_lens
    chunk.body_fb = types.SimpleNamespace(
        forced_token_lookup=batch._ccd_forced_lookup,
        req_pool_indices=batch.req_pool_indices,
    )
    chunk.fingerprint = _fingerprint(batch, runner, chunk_cap=chunk_cap)
    if captured:
        chunk._cond_graph = object()  # sentinel: "a live capture exists"
    return chunk


class TestCaptureFingerprint(CustomTestCase):
    def setUp(self):
        self.runner = _make_model_runner()
        self.reqs = [_make_req("r0", 3), _make_req("r1", 7)]
        self.batch = _make_batch(self.reqs)

    def test_equal_for_unchanged_batch(self):
        self.assertEqual(
            _fingerprint(self.batch, self.runner),
            _fingerprint(self.batch, self.runner),
        )

    def test_seq_lens_progress_does_not_change_fingerprint(self):
        # seq_lens are re-armed per chunk, deliberately NOT fingerprinted:
        # advancing them (what happens between chunks) must keep reuse valid.
        fp0 = _fingerprint(self.batch, self.runner)
        self.batch.seq_lens = self.batch.seq_lens + CHUNK_CAP + 1
        self.batch.seq_lens_cpu = self.batch.seq_lens_cpu + CHUNK_CAP + 1
        self.assertEqual(fp0, _fingerprint(self.batch, self.runner))

    def test_mismatch_on_request_order_swap(self):
        fp0 = _fingerprint(self.batch, self.runner)
        swapped = _make_batch([self.reqs[1], self.reqs[0]])
        swapped.req_to_token_pool = self.batch.req_to_token_pool
        self.assertNotEqual(fp0, _fingerprint(swapped, self.runner))

    def test_mismatch_on_rid_change(self):
        fp0 = _fingerprint(self.batch, self.runner)
        other = _make_batch([_make_req("r0", 3), _make_req("rX", 7)])
        other.req_to_token_pool = self.batch.req_to_token_pool
        self.assertNotEqual(fp0, _fingerprint(other, self.runner))

    def test_mismatch_on_req_pool_idx_change(self):
        fp0 = _fingerprint(self.batch, self.runner)
        other = _make_batch([_make_req("r0", 3), _make_req("r1", 8)])
        other.req_to_token_pool = self.batch.req_to_token_pool
        self.assertNotEqual(fp0, _fingerprint(other, self.runner))

    def test_mismatch_on_bs_change(self):
        fp0 = _fingerprint(self.batch, self.runner)
        shrunk = _make_batch([self.reqs[0]])
        shrunk.req_to_token_pool = self.batch.req_to_token_pool
        self.assertNotEqual(fp0, _fingerprint(shrunk, self.runner))

    def test_mismatch_on_chunk_cap_change(self):
        self.assertNotEqual(
            _fingerprint(self.batch, self.runner, chunk_cap=CHUNK_CAP),
            _fingerprint(self.batch, self.runner, chunk_cap=CHUNK_CAP + 1),
        )

    def test_mismatch_on_eos_change(self):
        self.assertNotEqual(
            _fingerprint(self.batch, self.runner, eos=[2]),
            _fingerprint(self.batch, self.runner, eos=[2, 7]),
        )

    def test_mismatch_on_req_to_token_rebind(self):
        fp0 = _fingerprint(self.batch, self.runner)
        self.batch.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.zeros((8, 4096), dtype=torch.int32)
        )
        self.assertNotEqual(fp0, _fingerprint(self.batch, self.runner))

    def test_mismatch_on_runner_identity(self):
        self.assertNotEqual(
            _fingerprint(self.batch, self.runner),
            _fingerprint(self.batch, _make_model_runner()),
        )


class TestReuseMismatchReason(CustomTestCase):
    def setUp(self):
        self.runner = _make_model_runner()
        self.batch = _make_batch([_make_req("r0", 0), _make_req("r1", 1)])
        self.chunk = _make_chunk(self.batch, self.runner)

    def _reason(self, batch=None, **kw):
        return self.chunk.reuse_mismatch_reason(
            batch if batch is not None else self.batch,
            chunk_cap=kw.pop("chunk_cap", CHUNK_CAP),
            eos_ids_host=kw.pop("eos", EOS),
            model_runner=kw.pop("runner", self.runner),
        )

    def test_match(self):
        self.assertIsNone(self._reason())

    def test_no_capture(self):
        self.chunk._cond_graph = None
        self.assertIn("no live capture", self._reason())

    def test_no_fingerprint(self):
        self.chunk.fingerprint = None
        self.assertIn("no live capture", self._reason())

    def test_fingerprint_mismatch(self):
        other = _make_batch([_make_req("r0", 0), _make_req("rZ", 1)])
        other.req_to_token_pool = self.batch.req_to_token_pool
        other._ccd_forced_lookup_rids = ["r0", "rZ"]
        self.assertIn("fingerprint mismatch", self._reason(batch=other))

    def test_forced_stash_rebound(self):
        # A restash rebinds new device tensors the captured graph cannot
        # see -- identity, not equality, is required.
        self.batch._ccd_forced_lookup = object()
        self.assertIn("stash was rebuilt/rebound", self._reason())

    def test_forced_stash_rid_mismatch(self):
        self.batch._ccd_forced_lookup_rids = ["r0"]
        self.assertIn("rid fingerprint", self._reason())

    def test_never_matches_after_destroy(self):
        self.chunk._cond_graph = None  # what destroy() leaves behind
        self.assertIsNotNone(self._reason())


class TestSlotTableRewrite(CustomTestCase):
    def setUp(self):
        self.runner = _make_model_runner()
        self.batch = _make_batch(
            [_make_req("r0", 2), _make_req("r1", 5)], seq_len=100
        )
        self.chunk = _make_chunk(self.batch, self.runner)
        self.bs = 2

    def _new_slots(self, base):
        return torch.arange(
            base, base + self.bs * CHUNK_CAP, dtype=torch.int64
        )

    def test_reuse_rewrites_in_place_and_prewrites_req_to_token(self):
        flat = self._new_slots(1000)
        before_ptr = self.chunk.slot_table.data_ptr()
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_token_slots",
            return_value=flat.clone(),
        ):
            self.chunk._reserve_and_prewrite_slots(self.batch, fresh_capture=False)
        # Pointer identity: the captured graph gathers from this exact
        # storage; values must be the new reservation.
        self.assertEqual(self.chunk.slot_table.data_ptr(), before_ptr)
        self.assertTrue(
            torch.equal(
                self.chunk.slot_table, flat.view(CHUNK_CAP, self.bs)
            )
        )
        # req_to_token rows: slot for (tick i, row r) at position S0 + i.
        req_to_token = self.batch.req_to_token_pool.req_to_token
        pos = ccspec_slot_positions(self.batch.seq_lens, CHUNK_CAP)
        for i in range(CHUNK_CAP):
            for r, req in enumerate(self.batch.reqs):
                self.assertEqual(
                    int(req_to_token[req.req_pool_idx, int(pos[i, r])]),
                    int(self.chunk.slot_table[i, r]),
                )

    def test_second_rearm_overwrites_first(self):
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_token_slots",
            return_value=self._new_slots(1000),
        ):
            self.chunk._reserve_and_prewrite_slots(self.batch, fresh_capture=False)
        # Next chunk: seq_lens advanced, fresh slots.
        self.batch.seq_lens = self.batch.seq_lens + CHUNK_CAP
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_token_slots",
            return_value=self._new_slots(2000),
        ):
            self.chunk._reserve_and_prewrite_slots(self.batch, fresh_capture=False)
        self.assertEqual(int(self.chunk.slot_table[0, 0]), 2000)
        # Old positions keep the OLD slots (never rolled back -- unread by
        # construction), new positions carry the new ones.
        req_to_token = self.batch.req_to_token_pool.req_to_token
        self.assertEqual(int(req_to_token[2, 100]), 1000)  # first chunk, tick 0
        self.assertEqual(int(req_to_token[2, 104]), 2000)  # second chunk, tick 0

    def test_fresh_capture_binds_new_tensor(self):
        flat = self._new_slots(3000)
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_token_slots",
            return_value=flat,
        ):
            self.chunk._reserve_and_prewrite_slots(self.batch, fresh_capture=True)
        self.assertTrue(
            torch.equal(self.chunk.slot_table, flat.view(CHUNK_CAP, self.bs))
        )

    def test_reuse_before_capture_raises(self):
        self.chunk.slot_table = None
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_token_slots",
            return_value=self._new_slots(0),
        ):
            with self.assertRaisesRegex(RuntimeError, "no slot_table"):
                self.chunk._reserve_and_prewrite_slots(
                    self.batch, fresh_capture=False
                )


class TestRearmForReuse(CustomTestCase):
    def setUp(self):
        self.runner = _make_model_runner()
        self.batch = _make_batch(
            [_make_req("r0", 0), _make_req("r1", 1)], seq_len=200
        )
        self.chunk = _make_chunk(self.batch, self.runner)
        self.priming_tokens = torch.tensor([11, 22], dtype=torch.int64)
        self.worker = types.SimpleNamespace(
            forward_batch_generation=MagicMock(
                return_value=types.SimpleNamespace(
                    next_token_ids=self.priming_tokens
                )
            )
        )

    def _rearm(self):
        with patch(
            "sglang.srt.mem_cache.allocation.alloc_token_slots",
            return_value=torch.arange(8, dtype=torch.int64),
        ):
            return self.chunk.rearm_for_reuse(self.batch, self.worker)

    def test_seeds_priming_token_and_returns_result(self):
        result = self._rearm()
        self.assertIs(result.next_token_ids, self.priming_tokens)
        self.assertTrue(
            torch.equal(self.chunk.cur_input_ids, self.priming_tokens)
        )

    def test_refreshes_rebound_seq_lens_into_captured_buffer(self):
        captured_buf = self.chunk.seq_lens_buf
        # prepare_for_decode's rebind: a NEW tensor with advanced values.
        self.batch.seq_lens = self.batch.seq_lens + 5
        self._rearm()
        self.assertIs(self.chunk.seq_lens_buf, captured_buf)  # never rebound
        self.assertTrue(torch.equal(captured_buf, self.batch.seq_lens))

    def test_aliased_seq_lens_is_noop(self):
        self._rearm()  # batch.seq_lens is still the captured buffer
        self.assertTrue(
            torch.equal(self.chunk.seq_lens_buf, self.batch.seq_lens)
        )

    def test_refreshes_rebound_req_pool_indices(self):
        captured = self.chunk.body_fb.req_pool_indices
        live = captured.clone()  # rebound tensor (equal values, new storage)
        self.batch.req_pool_indices = live
        captured.fill_(-9)  # make the copy observable
        self._rearm()
        self.assertTrue(torch.equal(captured, live))

    def test_raises_on_ineligible_batch(self):
        self.batch.sampling_info.is_all_greedy = False
        with self.assertRaisesRegex(RuntimeError, "ineligible"):
            self._rearm()
        # The priming tick must never have run.
        self.worker.forward_batch_generation.assert_not_called()

    def test_raises_before_capture(self):
        self.chunk.seq_lens_buf = None
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            self._rearm()

    def test_does_not_restash_forced_lookup(self):
        sentinel_lookup = self.batch._ccd_forced_lookup
        sentinel_rids = self.batch._ccd_forced_lookup_rids
        self._rearm()
        self.assertIs(self.batch._ccd_forced_lookup, sentinel_lookup)
        self.assertIs(self.batch._ccd_forced_lookup_rids, sentinel_rids)


class TestSyncSeqLensBack(CustomTestCase):
    def test_mirrors_terminal_value_to_rebound_tensor(self):
        runner = _make_model_runner()
        batch = _make_batch([_make_req("r0", 0)], seq_len=100)
        chunk = _make_chunk(batch, runner)
        # The launch advanced the CAPTURED buffer...
        chunk.seq_lens_buf.add_(3)
        # ...but the live batch.seq_lens was rebound since capture.
        batch.seq_lens = torch.full((1,), 100, dtype=torch.int64)
        chunk.sync_seq_lens_back(batch)
        self.assertEqual(int(batch.seq_lens[0]), 103)

    def test_noop_when_aliased(self):
        runner = _make_model_runner()
        batch = _make_batch([_make_req("r0", 0)], seq_len=100)
        chunk = _make_chunk(batch, runner)
        chunk.seq_lens_buf.add_(2)
        chunk.sync_seq_lens_back(batch)  # aliased: must not double-apply
        self.assertEqual(int(batch.seq_lens[0]), 102)


if __name__ == "__main__":
    unittest.main()
