import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.operations import _StateDict
from sglang.srt.batch_overlap.two_batch_overlap import (
    _model_forward_tbo_merge_value,
)
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4CudaDecoderLayer,
    DeepseekV4PlatformDecoderLayer,
)


class DeepseekV4DsparkTboTest(unittest.TestCase):
    def test_merge_crops_padding_and_restores_parent_order(self):
        output_a = {
            "forward_batch": SimpleNamespace(tbo_parent_token_range=(0, 2)),
            "aux": torch.tensor([[1.0], [2.0], [99.0]]),
        }
        output_b = {
            "forward_batch": SimpleNamespace(tbo_parent_token_range=(2, 5)),
            "aux": torch.tensor([[3.0], [4.0], [5.0], [99.0]]),
        }
        merged = _model_forward_tbo_merge_value(output_a, output_b, "aux", 5)
        torch.testing.assert_close(
            merged, torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        )

    def test_completed_mhc_state_is_added_to_child_capture_list(self):
        # Both decoder layers close the FFN mHC in their TBO postprocess: the
        # CUDA one through its FFN boundary, the platform one through hc_post.
        cuda_layer = SimpleNamespace(
            capture_dspark_aux=True,
            # finish completes the post-FFN mHC streams; the final read of a
            # stream its producer already wrote returns them as is.
            ffn_boundary=SimpleNamespace(finish=lambda hidden, batch: hidden),
        )
        platform_layer = SimpleNamespace(
            capture_dspark_aux=True,
            hc_post=lambda hidden, *_: hidden,
        )
        for cls, layer in (
            (DeepseekV4CudaDecoderLayer, cuda_layer),
            (DeepseekV4PlatformDecoderLayer, platform_layer),
        ):
            with self.subTest(cls.__name__):
                state = _StateDict()
                state.hidden_states_mlp_output = torch.tensor(
                    [[[1.0, 3.0], [3.0, 5.0]]]
                )
                if cls is DeepseekV4PlatformDecoderLayer:
                    state.ffn_residual = None
                    state.ffn_post = None
                    state.ffn_comb = None
                state.dspark_aux_hidden_states = []
                state.positions = torch.tensor([0])
                state.forward_batch = object()
                state.tbo_subbatch_index = 0

                with patch.object(
                    deepseek_v4.residual_batch,
                    "read_output",
                    lambda hidden, batch, read, norm: hidden,
                ):
                    output = cls.op_mhc_postprocess(layer, state)

                self.assertEqual(len(output["dspark_aux_hidden_states"]), 1)
                torch.testing.assert_close(
                    output["dspark_aux_hidden_states"][0], torch.tensor([[2.0, 4.0]])
                )

if __name__ == "__main__":
    unittest.main()
