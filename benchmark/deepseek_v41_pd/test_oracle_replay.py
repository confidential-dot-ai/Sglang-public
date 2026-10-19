from types import SimpleNamespace

import pytest
import torch

from sglang.srt.oracle_decode.forced_token_pool import (
    apply_forced_tokens,
    build_forced_token_lookup,
    compute_oracle_commit_fixed_len,
    gather_oracle_window,
)
from sglang.srt.sampling.sampling_params import SamplingParams


def req(ids, prompt=10):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(forced_token_ids=ids),
        origin_input_ids=[0] * prompt,
    )


def test_prefill_then_verify_and_tail():
    lookup = build_forced_token_lookup([req(list(range(20, 29))), req(None)], "cpu")
    first = apply_forced_tokens(torch.tensor([99, 98]), torch.tensor([9, 9]), lookup)
    assert first.tolist() == [20, 98]
    # seq_len is the position of the pending first output token.
    verify, valid = gather_oracle_window(torch.tensor([10, 10]), 6, lookup)
    assert verify[0].tolist() == list(range(20, 26))
    assert not valid[1].any()
    output, commit, mask = compute_oracle_commit_fixed_len(
        torch.tensor([11, 11]), 6, 6, lookup
    )
    assert output[0].tolist() == list(range(21, 27))
    assert commit.tolist() == [6, 0]
    assert mask.tolist() == [True, False]
    output, commit, _ = compute_oracle_commit_fixed_len(
        torch.tensor([17, 17]), 6, 6, lookup
    )
    assert output[0, :2].tolist() == [27, 28]
    assert commit.tolist() == [2, 0]


def test_no_oracle_is_noop():
    assert build_forced_token_lookup([req(None)], "cpu") is None


def test_validation_rejects_bad_forcing():
    # cc-main's oracle mode validates forced ids without requiring a server flag.
    SamplingParams(forced_token_ids=[2, 3], max_new_tokens=2).verify(100)
    for ids in [[], [-1], [100]]:
        with pytest.raises(ValueError):
            SamplingParams(forced_token_ids=ids).verify(100)
    with pytest.raises(ValueError, match="max_new_tokens"):
        SamplingParams(forced_token_ids=[1, 2, 3], max_new_tokens=2).verify(100)


def test_protocol_exposes_forcing():
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    request = ChatCompletionRequest(
        model="dsv41",
        messages=[{"role": "user", "content": "test"}],
        forced_token_ids=[2, 3],
    )
    assert request.forced_token_ids == [2, 3]
    assert request.to_sampling_params([], {})["forced_token_ids"] == [2, 3]
