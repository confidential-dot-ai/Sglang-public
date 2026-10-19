import unittest
from types import SimpleNamespace

import torch

from sglang.srt.batch_overlap.operations import _StateDict
from sglang.srt.batch_overlap.two_batch_overlap import (
    _model_forward_tbo_merge_value,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer


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
        layer = SimpleNamespace(
            capture_dspark_aux=True,
            hc_post=lambda hidden, *_: hidden,
        )
        state = _StateDict()
        state.hidden_states_mlp_output = torch.tensor(
            [[[1.0, 3.0], [3.0, 5.0]]]
        )
        state.ffn_residual = None
        state.ffn_post = None
        state.ffn_comb = None
        state.dspark_aux_hidden_states = []
        state.positions = torch.tensor([0])
        state.forward_batch = object()
        state.tbo_subbatch_index = 0

        output = DeepseekV4DecoderLayer.op_mhc_postprocess(layer, state)

        self.assertEqual(len(output["dspark_aux_hidden_states"]), 1)
        torch.testing.assert_close(
            output["dspark_aux_hidden_states"][0], torch.tensor([[2.0, 4.0]])
        )


if __name__ == "__main__":
    unittest.main()
