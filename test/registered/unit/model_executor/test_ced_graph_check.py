import torch

from sglang.srt.model_executor.runner.ced_graph_check import bitwise_equal


def test_hidden_check_distinguishes_signed_zero_and_dtype():
    positive = torch.tensor([0.0, 1.0])
    negative = torch.tensor([-0.0, 1.0])
    assert torch.equal(positive, negative)
    assert not bitwise_equal(positive, negative)
    assert bitwise_equal(positive, positive.clone())
    assert not bitwise_equal(positive, positive.to(torch.bfloat16))
