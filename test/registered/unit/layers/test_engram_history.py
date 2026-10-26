"""Engram history rows: what the scheduler builds for multimodal requests, and
what a CED boundary replay commits. CPU only."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.ngram_embedding_manager import (
    NgramEmbeddingManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 9
# Multimodal pad ids span 62 bits, past what the int32 history holds.
WIDE_PAD = MM_PAD_SHIFT_VALUE + (1 << 61)


def _make_manager():
    """A manager whose trigram hasher has a 16-token vocab, one layer and two
    heads, built without a tokenizer."""
    hasher = EngramHasher.__new__(EngramHasher)
    nn.Module.__init__(hasher)
    hasher.max_ngram_size = 3
    hasher.pad_id = 0
    hasher.image_token_id = IMAGE_TOKEN_ID
    hasher.token_map = torch.arange(16)
    hasher.multipliers = torch.tensor([[3, 5, 7]])
    hasher.primes = torch.tensor([[[11, 13], [17, 19]]])
    hasher.offsets = torch.zeros(1, 4, dtype=torch.int64)
    hasher.init_history(num_req_slots=2, device="cpu")
    return NgramEmbeddingManager(enabled=False, table=None, n=0, engram_hasher=hasher)


class TestEngramHistoryMultimodalPads(CustomTestCase):
    def test_extend_history_clamps_pads_and_hashes_them_as_images(self):
        manager = _make_manager()
        hasher = manager.engram_hasher
        # The prefix hit ends right after an image, so both predecessors are pads.
        req = SimpleNamespace(
            full_untruncated_fill_ids=[1, 2, WIDE_PAD, WIDE_PAD, 3, 4]
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, reqs=[req], prefix_lens=[4]
        )
        manager._prepare_engram_history(batch)
        self.assertEqual(batch.engram_history.tolist(), [[MM_PAD_SHIFT_VALUE] * 2])

        def hash_ids(history):
            return hasher._torch_hash_ids(
                torch.tensor([3, 4]),
                torch.tensor([4, 5]),
                MODE_EXTEND,
                history,
                num_real=2,
                block=1,
                row=torch.tensor([0, 0]),
                starts=torch.tensor([0]),
            )[0]

        image_history = torch.full((1, 2), IMAGE_TOKEN_ID, dtype=torch.int32)
        torch.testing.assert_close(
            hash_ids(batch.engram_history), hash_ids(image_history)
        )

    def test_pd_prebuilt_history_clamps_pads(self):
        manager = _make_manager()
        # The prompt ends with an image and prefill sampled token 7.
        req = SimpleNamespace(
            origin_input_ids=[1, WIDE_PAD, WIDE_PAD],
            output_ids=[7],
            kv=SimpleNamespace(req_pool_idx=1),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.PREBUILT, reqs=[req])
        manager._prepare_engram_history(batch)
        self.assertEqual(
            manager.engram_hasher.history[1].tolist(), [MM_PAD_SHIFT_VALUE] * 2
        )


class TestEngramHistoryBoundaryReplay(CustomTestCase):
    def test_boundary_replay_commits_the_prompt_tail(self):
        """A CED tail replayed from prefill's boundary state runs no Engram layer,
        yet the request's history row must end at its last prompt tokens: its first
        decode step hashes from that row, which otherwise still holds whatever the
        slot's previous request left there."""
        from sglang.srt.models.deepseek_v4 import deepseek_v4_engram_forward_hash_ids

        hasher = _make_manager().engram_hasher
        hasher.history[1] = torch.tensor([5, 6], dtype=torch.int32)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor([1]),
            positions=torch.arange(4),
            extend_seq_lens=torch.tensor([4]),
            extend_start_loc=torch.tensor([0]),
            extend_seq_lens_cpu=[4],
            engram_history=torch.zeros(1, 2, dtype=torch.int32),
            out_cache_loc=torch.tensor([10, 11, 12, 13]),
        )
        hash_ids = deepseek_v4_engram_forward_hash_ids(
            hasher=hasher,
            prefetcher=None,
            input_ids=torch.tensor([1, 2, 3, 4]),
            num_rows=4,
            forward_batch=forward_batch,
            cp_extend=False,
            replay_from_boundary=True,
        )
        self.assertIsNone(hash_ids)
        self.assertEqual(hasher.history[1].tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
