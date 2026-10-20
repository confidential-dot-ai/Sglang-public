"""CED boundary handoff: prefill's last-window state reaches decode's tail replay
in the late-layer tail's row order, or the request replays its encoder."""

from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.ced_boundary import (
    CedBoundaryLayout,
    build_ced_boundary_capture,
    build_ced_boundary_injection,
    fill_ced_boundary_for_forward,
    finalize_ced_boundary_slot,
    take_ced_boundary_slot,
    window_start,
    write_ced_boundary_slots,
)
from sglang.srt.environ import envs

LAYOUT = CedBoundaryLayout(window=8, hc_mult=2, hidden_size=4)


def req(seq_len, slot, room):
    return SimpleNamespace(
        origin_input_ids=list(range(seq_len)),
        metadata_buffer_index=slot,
        bootstrap_room=room,
        ced_boundary_state=None,
    )


def state_rows(positions, room):
    """A state row is identifiable by (room, absolute position)."""
    pos = torch.tensor(positions, dtype=torch.float32)
    residual = (room * 1000 + pos)[:, None, None].expand(
        -1, LAYOUT.hc_mult, LAYOUT.hidden_size
    )
    pre = (room * 1000 + pos + 0.5)[:, None].expand(-1, LAYOUT.hc_mult)
    return residual.to(torch.bfloat16), pre.contiguous()


def run_chunk(reqs, prefixes, lens, slots):
    """One prefill forward: every extend row's state, captured and scattered."""
    rows_residual, rows_pre = [], []
    for r, prefix, n in zip(reqs, prefixes, lens):
        res, pre = state_rows(range(prefix, prefix + n), r.bootstrap_room)
        rows_residual.append(res)
        rows_pre.append(pre)
    residual = torch.cat(rows_residual)
    pre = torch.cat(rows_pre)
    capture = build_ced_boundary_capture(
        reqs=reqs,
        extend_prefix_lens=prefixes,
        extend_lens=lens,
        window=LAYOUT.window,
        device=torch.device("cpu"),
    )
    if capture is not None:
        fill_ced_boundary_for_forward(
            SimpleNamespace(ced_boundary=capture), residual, pre
        )
    write_ced_boundary_slots(capture=capture, reqs=reqs, slots=slots, layout=LAYOUT)
    return capture


def slots(n=4):
    return torch.zeros((n, LAYOUT.nbytes), dtype=torch.uint8)


def test_layout_views_are_disjoint_and_aligned():
    buf = slots(1)
    header, residual, pre = LAYOUT.views(buf[0])
    assert LAYOUT.nbytes % 64 == 0
    header.fill_(1)
    residual.fill_(2)
    pre.fill_(3)
    assert int(header[0]) == 1 and residual.float().eq(2).all() and pre.eq(3).all()


def test_window_spanning_a_chunk_boundary_round_trips():
    buf = slots()
    r = req(seq_len=20, slot=1, room=7)
    # Chunks [0, 16) and [16, 20): the window [12, 20) spans both.
    first = run_chunk([r], [0], [16], buf)
    assert first.spans == [(0, 12, 4)]
    second = run_chunk([r], [16], [4], buf)
    assert second.spans == [(0, 16, 4)]
    finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
    state = take_ced_boundary_slot(req=r, slots=buf, idx=1, layout=LAYOUT)
    assert state is not None
    want_res, want_pre = state_rows(range(12, 20), 7)
    assert torch.equal(state.residual, want_res)
    assert torch.equal(state.pre, want_pre)
    # Taken once: the header is cleared, so a stale slot cannot be read again.
    assert take_ced_boundary_slot(req=r, slots=buf, idx=1, layout=LAYOUT) is None


def test_middle_chunks_capture_nothing():
    buf = slots()
    r = req(seq_len=100, slot=0, room=3)
    assert run_chunk([r], [0], [32], buf) is None


def test_short_prompt_window_is_the_whole_prompt():
    buf = slots()
    r = req(seq_len=5, slot=2, room=9)
    run_chunk([r], [0], [5], buf)
    finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
    state = take_ced_boundary_slot(req=r, slots=buf, idx=2, layout=LAYOUT)
    assert window_start(5, LAYOUT.window) == 0
    assert state.residual.shape[0] == 5
    assert torch.equal(state.residual, state_rows(range(5), 9)[0])


def test_prefix_cache_hit_inside_the_window_falls_back():
    buf = slots()
    r = req(seq_len=20, slot=0, room=5)
    # The radix prefix covered [0, 16): window rows [12, 16) were never computed.
    run_chunk([r], [16], [4], buf)
    finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
    assert take_ced_boundary_slot(req=r, slots=buf, idx=0, layout=LAYOUT) is None


def test_slot_of_another_request_is_rejected():
    buf = slots()
    r = req(seq_len=10, slot=0, room=5)
    run_chunk([r], [0], [10], buf)
    finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
    other = req(seq_len=10, slot=0, room=6)
    assert take_ced_boundary_slot(req=other, slots=buf, idx=0, layout=LAYOUT) is None


def test_batched_capture_keeps_each_request_in_its_own_slot():
    buf = slots()
    a, b = req(seq_len=12, slot=0, room=1), req(seq_len=30, slot=3, room=2)
    run_chunk([a, b], [0, 20], [12, 10], buf)
    for r in (a, b):
        finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
    sa = take_ced_boundary_slot(req=a, slots=buf, idx=0, layout=LAYOUT)
    sb = take_ced_boundary_slot(req=b, slots=buf, idx=3, layout=LAYOUT)
    assert torch.equal(sa.residual, state_rows(range(4, 12), 1)[0])
    assert torch.equal(sb.residual, state_rows(range(22, 30), 2)[0])


def test_injection_is_the_tail_row_order_or_nothing():
    buf = slots()
    a, b = req(seq_len=12, slot=0, room=1), req(seq_len=6, slot=1, room=2)
    run_chunk([a, b], [0, 0], [12, 6], buf)
    for r in (a, b):
        finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
        r.ced_boundary_state = take_ced_boundary_slot(
            req=r, slots=buf, idx=r.metadata_buffer_index, layout=LAYOUT
        )
    with envs.SGLANG_DSV41_CED_BOUNDARY_CHECK.override(False):
        injection = build_ced_boundary_injection(
            reqs=[a, b], device=torch.device("cpu")
        )
    # Request order, each request's last min(window, seq_len) rows.
    want = torch.cat([state_rows(range(4, 12), 1)[0], state_rows(range(6), 2)[0]])
    assert torch.equal(injection.residual, want)
    assert a.ced_boundary_state is None and b.ced_boundary_state is None

    c, d = req(seq_len=12, slot=2, room=3), req(seq_len=12, slot=3, room=4)
    run_chunk([c], [0], [12], buf)
    finalize_ced_boundary_slot(req=c, slots=buf, layout=LAYOUT)
    c.ced_boundary_state = take_ced_boundary_slot(
        req=c, slots=buf, idx=2, layout=LAYOUT
    )
    d.ced_boundary_state = None
    # One request without a handoff sends the whole batch through the encoder.
    assert build_ced_boundary_injection(reqs=[c, d], device=torch.device("cpu")) is None


def test_retried_prefill_that_rewrites_its_window_falls_back():
    buf = slots()
    r = req(seq_len=10, slot=0, room=4)
    run_chunk([r], [0], [10], buf)
    run_chunk([r], [0], [10], buf)
    finalize_ced_boundary_slot(req=r, slots=buf, layout=LAYOUT)
    assert take_ced_boundary_slot(req=r, slots=buf, idx=0, layout=LAYOUT) is None
