"""Unit tests for CCSpec (model_executor/ccspec_decode.py) -- CCD-style
conditional WHILE-loop decode chunks for PLAIN (non-speculative) decode.
CPU-runnable: no server, no model loading, no GPU. Follows
test_ccd_split_batch.py's fixture / mock-fidelity conventions -- in
particular, sampling mocks pin penalizer_orchestrator=None,
acc_scaling_penalties=None, acc_additive_penalties=None (what REAL
copy_for_forward() batches carry in the common no-penalty case; a truthy
MagicMock attribute would silently trip the corresponding gate on every
test).

Covered here (host-side logic only):
  - check_ccspec_eligible: the accept case plus every rejection reason,
    including remaining-max-new < chunk_cap + 1 and the backend-flag gate.
  - ccspec_slot_positions: the slot-table / req_to_token precompute
    arithmetic, on CPU tensors.
  - reconcile_ccspec_host_lengths: seq_lens_cpu / per-req bookkeeping for
    emitted < chunk_cap, == chunk_cap, and == 0.
  - build_ccspec_synthetic_result: the plain-decode result contract
    (next_token_ids [bs], NO accept_lens, logits_output None).
  - ccspec_continue_flag: the all-or-nothing emitted-count semantics given
    a history containing an EOS at tick k.

GPU-only parts (capture, launch, read_history) get skipTest-guarded stubs,
like test_ccd_compact_draft_cache_padded.py's CUDA-gated tests -- they
additionally need a loaded model engine, so they are validated on the real
guest (see the GPU validation checklist in the change description).
"""

import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.ccspec_decode import (  # noqa: E402
    SGLANG_CCSPEC_CHUNK_CAP,
    SGLANG_CCSPEC_MIN_OUTPUT_LEN,
    build_ccspec_synthetic_result,
    ccspec_continue_flag,
    ccspec_slot_positions,
    check_ccspec_eligible,
    reconcile_ccspec_host_lengths,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

CHUNK_CAP = 4
VETERAN_LEN = SGLANG_CCSPEC_MIN_OUTPUT_LEN + 1


def _make_req(
    *,
    output_len=VETERAN_LEN,
    max_new_tokens=100000,
    return_logprob=False,
    return_hidden_states=False,
    return_sampling_mask=False,
    grammar=None,
):
    return types.SimpleNamespace(
        sampling_params=types.SimpleNamespace(
            top_k=1, max_new_tokens=max_new_tokens, forced_token_ids=None
        ),
        return_logprob=return_logprob,
        return_hidden_states=return_hidden_states,
        return_sampling_mask=return_sampling_mask,
        output_ids=[0] * output_len,
        grammar=grammar,
    )


def _permissive_sampling_info():
    """See module docstring: pin exactly the fields a real forward-only
    sampling_info copy carries in the no-penalty case."""
    sampling_info = MagicMock()
    sampling_info.is_all_greedy = True
    sampling_info.penalizer_orchestrator = None
    sampling_info.acc_scaling_penalties = None
    sampling_info.acc_additive_penalties = None
    sampling_info.logit_bias = None
    sampling_info.has_custom_logit_processor = False
    return sampling_info


def _make_model_runner(*, supports_ccspec=True, tp_size=1, pp_size=1):
    return types.SimpleNamespace(
        attn_backend=types.SimpleNamespace(supports_ccspec_capture=supports_ccspec),
        ps=types.SimpleNamespace(tp_size=tp_size, pp_size=pp_size),
    )


def _make_batch(reqs, **overrides):
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
        seq_lens_cpu=torch.full((bs,), 100, dtype=torch.int64),
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=torch.zeros((max(bs, 1), 4096), dtype=torch.int32)
        ),
    )
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


class TestCheckCCSpecEligible(CustomTestCase):
    def _check(self, batch, model_runner=None, chunk_cap=CHUNK_CAP):
        if model_runner is None:
            model_runner = _make_model_runner()
        return check_ccspec_eligible(batch, model_runner, chunk_cap=chunk_cap)

    def test_accepts_plain_greedy_decode_batch(self):
        batch = _make_batch([_make_req(), _make_req()])
        self.assertIsNone(self._check(batch))

    def test_rejects_extend_mode(self):
        batch = _make_batch([_make_req()], forward_mode=ForwardMode.EXTEND)
        self.assertIn("not prefill/extend", self._check(batch))

    def test_rejects_extend_in_batch(self):
        batch = _make_batch([_make_req()], is_extend_in_batch=True)
        self.assertIn("not prefill/extend", self._check(batch))

    def test_rejects_idle_mode(self):
        batch = _make_batch([_make_req()], forward_mode=ForwardMode.IDLE)
        self.assertIn("non-decode", self._check(batch))

    def test_rejects_speculative_batch(self):
        batch = _make_batch(
            [_make_req()],
            spec_algorithm=types.SimpleNamespace(is_none=lambda: False),
        )
        self.assertIn("CCD", self._check(batch))

    def test_rejects_empty_batch(self):
        batch = _make_batch([])
        self.assertIn("non-empty", self._check(batch))

    def test_rejects_encoder_decoder(self):
        batch = _make_batch(
            [_make_req()],
            model_config=types.SimpleNamespace(is_encoder_decoder=True),
        )
        self.assertIn("encoder-decoder", self._check(batch))

    def test_rejects_dp_mlp_sync_batch(self):
        batch = _make_batch([_make_req()], global_num_tokens=[1])
        self.assertIn("DP-attention", self._check(batch))

    def test_rejects_tp_gt_1(self):
        batch = _make_batch([_make_req()])
        self.assertIn(
            "TP=1", self._check(batch, _make_model_runner(tp_size=2))
        )

    def test_rejects_pp_gt_1(self):
        batch = _make_batch([_make_req()])
        self.assertIn(
            "PP=1", self._check(batch, _make_model_runner(pp_size=2))
        )

    def test_rejects_non_greedy_sampling(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.is_all_greedy = False
        self.assertIn("all-greedy", self._check(batch))

    def test_rejects_missing_sampling_info(self):
        batch = _make_batch([_make_req()], sampling_info=None)
        self.assertIn("all-greedy", self._check(batch))

    def test_rejects_required_penalizer(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.penalizer_orchestrator = types.SimpleNamespace(
            is_required=True
        )
        self.assertIn("penalizers", self._check(batch))

    def test_allows_non_required_penalizer(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.penalizer_orchestrator = types.SimpleNamespace(
            is_required=False
        )
        self.assertIsNone(self._check(batch))

    def test_rejects_scaling_penalties(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.acc_scaling_penalties = torch.ones(1, 8)
        self.assertIn("scaling penalties", self._check(batch))

    def test_rejects_additive_penalties(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.acc_additive_penalties = torch.zeros(1, 8)
        self.assertIn("additive penalties", self._check(batch))

    def test_rejects_logit_bias(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.logit_bias = torch.zeros(1, 8)
        self.assertIn("logit_bias", self._check(batch))

    def test_rejects_custom_logit_processor(self):
        batch = _make_batch([_make_req()])
        batch.sampling_info.has_custom_logit_processor = True
        self.assertIn("custom logit processors", self._check(batch))

    def test_rejects_batch_level_grammar_flag(self):
        batch = _make_batch([_make_req()], has_grammar=True)
        self.assertIn("grammar", self._check(batch))

    def test_rejects_per_request_grammar_even_device_fsm(self):
        # V1 rejects EVERY grammar backend, including a device-FSM-like one:
        # the plain path advances grammar host-side per tick, so there is no
        # in-graph advance for a captured chunk to replay.
        batch = _make_batch([_make_req(grammar=object())])
        self.assertIn("grammar", self._check(batch))

    def test_rejects_return_logprob(self):
        batch = _make_batch([_make_req(return_logprob=True)])
        self.assertIn("return_logprob", self._check(batch))

    def test_rejects_return_hidden_states(self):
        batch = _make_batch([_make_req(return_hidden_states=True)])
        self.assertIn("return_hidden_states", self._check(batch))

    def test_rejects_return_sampling_mask(self):
        batch = _make_batch([_make_req(return_sampling_mask=True)])
        self.assertIn("return_sampling_mask", self._check(batch))

    def test_rejects_short_output(self):
        batch = _make_batch(
            [_make_req(), _make_req(output_len=SGLANG_CCSPEC_MIN_OUTPUT_LEN - 1)]
        )
        self.assertIn("output tokens", self._check(batch))

    def test_rejects_remaining_max_new_below_cap_plus_one(self):
        # remaining == chunk_cap exactly: the final captured tick would emit
        # one token PAST the budget (derivation in check_ccspec_eligible).
        batch = _make_batch(
            [_make_req(max_new_tokens=VETERAN_LEN + CHUNK_CAP)]
        )
        self.assertIn("remaining max_new_tokens", self._check(batch))

    def test_accepts_remaining_max_new_exactly_cap_plus_one(self):
        batch = _make_batch(
            [_make_req(max_new_tokens=VETERAN_LEN + CHUNK_CAP + 1)]
        )
        self.assertIsNone(self._check(batch))

    def test_accepts_unlimited_max_new(self):
        batch = _make_batch([_make_req(max_new_tokens=None)])
        self.assertIsNone(self._check(batch))

    def test_rejects_backend_without_flag(self):
        batch = _make_batch([_make_req()])
        self.assertIn(
            "supports_ccspec_capture",
            self._check(batch, _make_model_runner(supports_ccspec=False)),
        )

    def test_rejects_paged_allocator(self):
        batch = _make_batch([_make_req()])
        batch.token_to_kv_pool_allocator.page_size = 128
        self.assertIn("page_size", self._check(batch))

    def test_rejects_allocator_without_headroom(self):
        batch = _make_batch([_make_req(), _make_req()])
        # bs * chunk_cap = 8 needed; 7 available -> reason, not a raise.
        batch.token_to_kv_pool_allocator.available_size = lambda: 7
        self.assertIn("cannot reserve", self._check(batch))

    def test_rejects_missing_seq_lens_cpu(self):
        batch = _make_batch([_make_req()], seq_lens_cpu=None)
        self.assertIn("seq_lens_cpu", self._check(batch))

    def test_rejects_req_to_token_overflow(self):
        batch = _make_batch([_make_req()])
        batch.seq_lens_cpu = torch.tensor([4093], dtype=torch.int64)
        # 4093 + 4 > 4096 -> the pre-written rows would run off the pool row.
        self.assertIn("overflow", self._check(batch))

    def test_rejects_dllm(self):
        batch = _make_batch([_make_req()], dllm_config=object())
        self.assertIn("diffusion", self._check(batch))

    def test_rejects_hisparse(self):
        batch = _make_batch([_make_req()], hisparse_coordinator=object())
        self.assertIn("hisparse", self._check(batch))

    def test_rejects_mamba_tracking(self):
        batch = _make_batch(
            [_make_req()], mamba_track_indices=torch.zeros(1, dtype=torch.int64)
        )
        self.assertIn("mamba", self._check(batch))

    def test_default_chunk_cap_comes_from_env(self):
        # chunk_cap=None must fall back to the module-level env default.
        batch = _make_batch(
            [_make_req(max_new_tokens=VETERAN_LEN + SGLANG_CCSPEC_CHUNK_CAP)]
        )
        self.assertIn(
            "remaining max_new_tokens",
            check_ccspec_eligible(batch, _make_model_runner()),
        )


class TestSlotTablePrecompute(CustomTestCase):
    def test_positions_arithmetic(self):
        # S0 = post-prepare seq_lens; priming's own slot sits at S0 - 1
        # (written by prepare_for_decode), so captured tick i+1's row lands
        # at S0 + i.
        seq_lens = torch.tensor([5, 9], dtype=torch.int64)
        pos = ccspec_slot_positions(seq_lens, 3)
        self.assertTrue(
            torch.equal(
                pos, torch.tensor([[5, 9], [6, 10], [7, 11]], dtype=torch.int64)
            )
        )

    def test_prewrite_lands_slots_at_expected_rows(self):
        # Simulate prime_and_capture's pre-write on CPU tensors and verify
        # every (req, position) cell holds the right slot id.
        bs, cap = 2, 3
        req_to_token = torch.zeros((4, 16), dtype=torch.int32)
        req_pool_indices = torch.tensor([1, 3], dtype=torch.int64)
        seq_lens = torch.tensor([5, 9], dtype=torch.int64)
        slot_table = torch.arange(cap * bs, dtype=torch.int64).view(cap, bs) + 100

        pos = ccspec_slot_positions(seq_lens, cap)
        rows = req_pool_indices.unsqueeze(0).expand(cap, bs)
        req_to_token[rows, pos] = slot_table.to(req_to_token.dtype)

        for i in range(cap):
            for r in range(bs):
                self.assertEqual(
                    int(req_to_token[int(req_pool_indices[r]), int(seq_lens[r]) + i]),
                    int(slot_table[i, r]),
                )
        # Nothing below S0 was touched (the priming tick's own row and the
        # live prefix belong to prepare_for_decode, not the chunk).
        self.assertEqual(int(req_to_token[1, :5].abs().sum()), 0)
        self.assertEqual(int(req_to_token[3, :9].abs().sum()), 0)


class TestReconcileHostLengths(CustomTestCase):
    def _make_batch(self, bs=2, s0=100):
        reqs = []
        for _ in range(bs):
            reqs.append(
                types.SimpleNamespace(
                    decode_batch_idx=7,
                    kv_committed_len=s0,
                    kv=types.SimpleNamespace(kv_allocated_len=s0),
                )
            )
        return types.SimpleNamespace(
            reqs=reqs,
            seq_lens_cpu=torch.full((bs,), s0, dtype=torch.int64),
            orig_seq_lens=torch.full((bs,), s0, dtype=torch.int64),
            seq_lens_sum=bs * s0,
        )

    def test_emitted_below_cap(self):
        batch = self._make_batch(bs=2, s0=100)
        reconcile_ccspec_host_lengths(batch, 2)
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([102, 102]))
        )
        self.assertTrue(
            torch.equal(batch.orig_seq_lens, torch.tensor([102, 102]))
        )
        self.assertIsNone(batch.seq_lens_sum)
        for req in batch.reqs:
            self.assertEqual(req.decode_batch_idx, 9)
            self.assertEqual(req.kv_committed_len, 102)
            self.assertEqual(req.kv.kv_allocated_len, 102)

    def test_emitted_equals_cap(self):
        batch = self._make_batch(bs=1, s0=50)
        reconcile_ccspec_host_lengths(batch, CHUNK_CAP)
        self.assertEqual(int(batch.seq_lens_cpu[0]), 50 + CHUNK_CAP)
        self.assertEqual(batch.reqs[0].kv_committed_len, 50 + CHUNK_CAP)
        self.assertIsNone(batch.seq_lens_sum)

    def test_emitted_zero_is_noop(self):
        batch = self._make_batch(bs=1, s0=50)
        reconcile_ccspec_host_lengths(batch, 0)
        # A priming-tick EOS emits zero captured ticks: the tick then WAS a
        # normal single-token tick and no mirror may move.
        self.assertEqual(int(batch.seq_lens_cpu[0]), 50)
        self.assertEqual(batch.seq_lens_sum, 50)
        self.assertEqual(batch.reqs[0].decode_batch_idx, 7)


class TestSyntheticResultContract(CustomTestCase):
    def test_shape_and_fields(self):
        tokens = torch.tensor([11, 22, 33], dtype=torch.int64)
        result = build_ccspec_synthetic_result(tokens)
        self.assertIsNone(result.logits_output)
        self.assertTrue(torch.equal(result.next_token_ids, tokens))
        self.assertEqual(result.next_token_ids.shape, (3,))
        # Plain-decode contract: NO accept_lens (there is no accept concept
        # on the non-spec result path), no draft-token stride.
        self.assertIsNone(result.accept_lens)
        self.assertIsNone(result.speculative_num_draft_tokens)
        self.assertTrue(result.can_run_cuda_graph)
        self.assertIsNone(result.copy_done)
        self.assertIsNone(result.next_draft_input)
        # has_sampled_token_ids drives _relay_forward_payload for the
        # terminal result -- must be True for a tensor payload.
        self.assertTrue(result.has_sampled_token_ids)


class TestAllOrNothingStop(CustomTestCase):
    EOS = torch.tensor([2, 7], dtype=torch.int64)

    def _simulate_emitted(self, history: torch.Tensor, chunk_cap: int) -> int:
        """CPU reference of the captured WHILE loop's control flow: each
        iteration records history[t] then re-arms the predicate via
        ccspec_continue_flag; the cond kernel continues while n_run > 0 and
        step < cap. So the EOS-carrying tick IS emitted and nothing after
        it runs."""
        emitted = 0
        n_run = 1
        for t in range(chunk_cap):
            if n_run == 0:
                break
            emitted += 1
            n_run = int(ccspec_continue_flag(history[t], self.EOS)[0])
        return emitted

    def test_continue_flag_no_eos(self):
        flag = ccspec_continue_flag(torch.tensor([1, 3, 4]), self.EOS)
        self.assertEqual(flag.dtype, torch.int32)
        self.assertEqual(flag.shape, (1,))
        self.assertEqual(int(flag[0]), 1)

    def test_continue_flag_any_row_eos(self):
        self.assertEqual(
            int(ccspec_continue_flag(torch.tensor([1, 7, 4]), self.EOS)[0]), 0
        )

    def test_eos_at_tick_k_emits_k_plus_one(self):
        cap = 4
        history = torch.tensor(
            [[1, 1], [1, 2], [3, 3], [4, 4]], dtype=torch.int64
        )  # EOS (2) for row 1 at tick index 1
        self.assertEqual(self._simulate_emitted(history, cap), 2)

    def test_no_eos_emits_chunk_cap(self):
        cap = 4
        history = torch.ones((cap, 2), dtype=torch.int64)
        self.assertEqual(self._simulate_emitted(history, cap), cap)

    def test_eos_at_first_tick_emits_one(self):
        cap = 4
        history = torch.tensor([[2, 1]] * cap, dtype=torch.int64)
        self.assertEqual(self._simulate_emitted(history, cap), 1)

    def test_priming_eos_arms_loop_off(self):
        # arm() seeds n_run from the PRIMING tokens: an EOS there must stop
        # the loop before any captured tick runs (emitted == 0).
        priming_tokens = torch.tensor([5, 2], dtype=torch.int64)
        self.assertEqual(int(ccspec_continue_flag(priming_tokens, self.EOS)[0]), 0)


class TestCCSpecCaptureGPU(CustomTestCase):
    """Capture/launch need a CUDA device AND a loaded model engine; neither
    exists in unit CI. Stub guarded like test_ccd_compact_draft_cache_padded's
    CUDA-gated tests; the real capture path is validated on the guest via
    the GPU validation checklist (SGLANG_CCSPEC_DECODE=1 live decode run)."""

    def test_capture_and_launch(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        self.skipTest(
            "requires a loaded model engine (CCSpecChunk.prime_and_capture "
            "traces a real model forward); covered by live GPU validation"
        )


if __name__ == "__main__":
    unittest.main()
