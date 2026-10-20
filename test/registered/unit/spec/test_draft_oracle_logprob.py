"""Unit tests for DFlashWorkerV2._compute_draft_oracle_log_prob -- the
draft-model confidence check oracle mode uses to pick a realistic,
data-dependent commit length instead of a fixed/full window or argmax
equality (oracle tokens usually come from a different model than the one
being served). No server, no model loading, no GPU/TP required: the TP=1
(unsharded) path is tested directly; the TP>1 cross-shard math (log-sum-exp
composition + masked-max token selection) is verified standalone against a
manual multi-shard simulation, since it can't be exercised through a real
tp_group without multi-GPU hardware."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.test_utils import CustomTestCase


def _make_worker():
    # _compute_draft_oracle_log_prob's TP=1 path reads no `self.*` state at
    # all, so a bare, un-__init__-ed instance is a valid (and much cheaper)
    # stand-in than constructing a real DFlashWorkerV2.
    return DFlashWorkerV2.__new__(DFlashWorkerV2)


class TestDraftOracleLogProbUnsharded(CustomTestCase):
    def test_matches_reference_log_softmax(self):
        torch.manual_seed(0)
        hidden_dim, vocab = 16, 100
        hidden_states = torch.randn(6, hidden_dim)
        weight = torch.randn(vocab, hidden_dim)
        lm_head = SimpleNamespace(weight=weight)
        oracle_tokens = torch.tensor([3, 17, 42, 99, 0, 55])

        log_probs = _make_worker()._compute_draft_oracle_log_prob(
            hidden_states=hidden_states,
            lm_head=lm_head,
            oracle_tokens=oracle_tokens,
            chunk_size=4,  # smaller than num_tokens=6, exercises the chunk loop
        )

        ref_logits = hidden_states @ weight.T
        ref_log_probs = (
            torch.log_softmax(ref_logits, dim=-1)
            .gather(1, oracle_tokens.unsqueeze(1))
            .squeeze(1)
        )
        self.assertTrue(torch.allclose(log_probs, ref_log_probs, atol=1e-4))

    def test_empty_input_returns_empty(self):
        result = _make_worker()._compute_draft_oracle_log_prob(
            hidden_states=torch.empty(0, 16),
            lm_head=SimpleNamespace(weight=torch.randn(10, 16)),
            oracle_tokens=torch.empty(0, dtype=torch.int64),
        )
        self.assertEqual(result.shape, (0,))


class TestCrossShardLogProbComposition(CustomTestCase):
    """Standalone verification of the two identities the TP>1 branch relies
    on: logsumexp(whole vocab) == logsumexp(per-shard logsumexps), and the
    oracle token's real logit is finite on exactly one shard (the one owning
    its vocab range) so a cross-shard max selects it correctly."""

    def test_sharded_composition_matches_unsharded_reference(self):
        torch.manual_seed(1)
        hidden_dim, vocab, tp_size = 16, 97, 4  # 97 deliberately not divisible by 4
        hidden_states = torch.randn(5, hidden_dim)
        weight = torch.randn(vocab, hidden_dim)
        oracle_tokens = torch.tensor([3, 50, 96, 0, 71])

        ref_logits = hidden_states @ weight.T
        ref_log_probs = (
            torch.log_softmax(ref_logits, dim=-1)
            .gather(1, oracle_tokens.unsqueeze(1))
            .squeeze(1)
        )

        shard_size = (vocab + tp_size - 1) // tp_size
        local_lses, local_logits = [], []
        for r in range(tp_size):
            lo, hi = r * shard_size, min(vocab, (r + 1) * shard_size)
            if lo >= hi:
                local_lses.append(torch.full((5,), float("-inf")))
                local_logits.append(torch.full((5,), float("-inf")))
                continue
            shard_logits = hidden_states @ weight[lo:hi].T
            local_lses.append(torch.logsumexp(shard_logits, dim=-1))
            in_shard = (oracle_tokens >= lo) & (oracle_tokens < hi)
            idx = (oracle_tokens - lo).clamp(min=0, max=hi - lo - 1)
            picked = shard_logits.gather(1, idx.unsqueeze(1)).squeeze(1)
            local_logits.append(
                torch.where(in_shard, picked, torch.full((5,), float("-inf")))
            )

        # In the real code this is tp_group.all_gather_into_tensor; here we
        # just stack, since we're checking the reduction math, not NCCL.
        gathered_lse = torch.stack(local_lses, dim=0)
        gathered_logit = torch.stack(local_logits, dim=0)
        global_lse = torch.logsumexp(gathered_lse, dim=0)
        global_logit = torch.max(gathered_logit, dim=0).values
        computed = global_logit - global_lse

        self.assertTrue(torch.allclose(computed, ref_log_probs, atol=1e-4))
