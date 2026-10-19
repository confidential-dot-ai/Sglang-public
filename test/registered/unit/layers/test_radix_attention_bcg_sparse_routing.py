from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import radix_attention as ra
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _layer() -> ra.RadixAttention:
    return ra.RadixAttention(
        num_heads=2,
        head_dim=8,
        scaling=1.0,
        num_kv_heads=1,
        layer_id=0,
    )


def _sparse_kwargs() -> dict:
    return {
        "idx_q": torch.zeros((4, 1, 8)),
        "idx_k": torch.zeros((4, 1, 8)),
    }


def _run_forward(forward_mode: ForwardMode, in_bcg: bool):
    """Drive RadixAttention.forward's sparse extend branch with the module
    seams patched, returning which route it took."""
    calls = []
    backend = SimpleNamespace(
        forward=lambda *a, **k: calls.append("captured_backend") or ("idx", "attn")
    )

    def breakable_op(*a, **k):
        calls.append("breakable_op")

    def plain_op(*a, **k):
        calls.append("plain_op")

    with (
        patch.object(ra, "is_in_breakable_cuda_graph", return_value=in_bcg),
        patch.object(
            ra, "get_tc_piecewise_forward_context", return_value=object()
        ),
        patch.object(ra, "get_attn_backend", return_value=backend),
        patch.object(
            ra, "breakable_unified_sparse_attention_with_output", breakable_op
        ),
        patch.object(ra, "unified_sparse_attention_with_output", plain_op),
    ):
        out = _layer().forward(
            q=torch.zeros((4, 16)),
            k=None,
            v=None,
            forward_batch=SimpleNamespace(forward_mode=forward_mode),
            **_sparse_kwargs(),
        )
    return calls, out


def test_bcg_prefill_takes_the_eager_break_not_the_captured_path() -> None:
    """Regression guard for the prefill-BCG silent-wrongness class: the sparse
    EXTEND forward must route through the eager_on_graph break (fresh extend
    metadata every replay), never record get_attn_backend().forward into the
    captured segment (which bakes stale metadata pointers, the capture bs
    grid, and bucket-sized K coverage)."""
    calls, out = _run_forward(ForwardMode.EXTEND, in_bcg=True)
    assert calls == ["breakable_op"]
    idx_out, attn_out = out
    assert tuple(attn_out.shape) == (4, 16)
    assert tuple(idx_out.shape) == (4, 8)


def test_bcg_target_verify_keeps_the_captured_path() -> None:
    """TARGET_VERIFY metadata is address-stable and refreshed in place at
    replay-prep, so the validated captured route must be preserved."""
    calls, out = _run_forward(ForwardMode.TARGET_VERIFY, in_bcg=True)
    assert calls == ["captured_backend"]
    assert out == ("idx", "attn")


def test_no_bcg_takes_the_plain_op() -> None:
    calls, _ = _run_forward(ForwardMode.EXTEND, in_bcg=False)
    assert calls == ["plain_op"]


def test_breakable_sparse_wrapper_wraps_the_real_op() -> None:
    cells = [
        c.cell_contents
        for c in ra.breakable_unified_sparse_attention_with_output.__closure__
    ]
    assert ra.unified_sparse_attention_with_output in cells
