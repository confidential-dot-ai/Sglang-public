import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRejectSamplingFallback(unittest.TestCase):
    def test_cdf_rounding_fallback_uses_largest_residual(self):
        device = torch.device("cuda")
        candidates = torch.tensor([[0, 3]], dtype=torch.int64, device=device)
        retrieve_index = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        predicts = torch.full((2,), -1, dtype=torch.int64, device=device)
        accept_index = torch.zeros((1, 2), dtype=torch.int64, device=device)
        accept_token_num = torch.zeros((1,), dtype=torch.int32, device=device)
        uniform_samples = torch.tensor([[0.5]], dtype=torch.float32, device=device)
        # target_u == norm_sum makes the strict CDF comparison miss every bin.
        uniform_samples_final = torch.tensor([1.0], dtype=torch.float32, device=device)
        target_probs = torch.tensor(
            [[[0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=device,
        )
        draft_probs = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32, device=device
        )

        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_index,
            retrive_next_sibling=retrieve_index,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_samples_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(accept_token_num.item(), 0)
        self.assertEqual(predicts[0].item(), 2)


if __name__ == "__main__":
    unittest.main()
