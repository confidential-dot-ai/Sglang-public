from unittest.mock import patch

import torch

from sglang.srt.distributed.collective_trace import (
    trace_collective,
    trace_decoder_layer,
)
from sglang.srt.environ import envs


class _FakeGroup:
    unique_name = "tp"
    rank_in_group = 2
    rank = 6
    world_size = 4


def test_disabled_collective_trace_returns_original_function():
    def collective(group, tensor):
        return tensor

    with envs.SGLANG_COLLECTIVE_TRACE.override(False):
        assert trace_collective("all_reduce")(collective) is collective


def test_collective_trace_labels_rank_sequence_shape_and_layer():
    events = []
    group = _FakeGroup()

    with envs.SGLANG_COLLECTIVE_TRACE.override(True):

        @trace_collective("all_reduce")
        def collective(traced_group, tensor):
            assert traced_group is group
            return tensor + 1

        class DecoderLayer(torch.nn.Module):
            def __init__(self, layer_id, layer_collective):
                super().__init__()
                self.layer_id = layer_id
                self.collective = layer_collective

            @trace_decoder_layer
            def forward(self, tensor):
                return self.collective(tensor)

        layer = DecoderLayer(
            layer_id=17,
            layer_collective=lambda tensor: collective(group, tensor),
        )

        with (
            patch(
                "torch.cuda.nvtx.range_push",
                side_effect=lambda label: events.append(("push", label)),
            ),
            patch(
                "torch.cuda.nvtx.range_pop",
                side_effect=lambda: events.append(("pop", None)),
            ),
            patch("torch.compiler.is_compiling", return_value=False),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            result = layer(torch.zeros((3, 8), dtype=torch.bfloat16))

    torch.testing.assert_close(result, torch.ones_like(result))
    assert events[0] == (
        "push",
        "sglang.collective|op=all_reduce|group=tp|seq=0|rank=2|global_rank=6"
        "|world=4|layer=17|bytes=48|dtype=bfloat16|shape=3x8",
    )
    assert events[1:] == [("pop", None)]


def test_layer_trace_is_invisible_to_fullgraph_compile():
    group = _FakeGroup()
    with envs.SGLANG_COLLECTIVE_TRACE.override(True):

        @trace_collective("all_reduce")
        def collective(_group, tensor):
            return tensor + 1

        class DecoderLayer(torch.nn.Module):
            layer_id = 3

            @trace_decoder_layer
            def forward(self, tensor):
                return collective(group, tensor)

        compiled = torch.compile(DecoderLayer(), backend="eager", fullgraph=True)
        result = compiled(torch.zeros(4))

    torch.testing.assert_close(result, torch.ones_like(result))
