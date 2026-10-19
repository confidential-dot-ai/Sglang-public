"""CED coalescing deadlines must not stop decode or diverge across TP ranks."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.environ import envs


def scheduler(*, age=0.5, count=1, last_tail=False, tp_size=1):
    return SimpleNamespace(
        waiting_queue=[
            SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=10 - age))
            for _ in range(count)
        ],
        last_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: last_tail)
        ),
        ps=SimpleNamespace(tp_size=tp_size),
        tp_cpu_group=object(),
    )


def defer(s, *, idle=False):
    with (
        patch("sglang.srt.disaggregation.decode.time.perf_counter", return_value=10),
        envs.SGLANG_DSV41_CED_TAIL_MAX_WAIT_MS.override(1000),
    ):
        return SchedulerDisaggregationDecodeMixin._defer_ced_tail_batch(
            s, SimpleNamespace(is_empty=lambda: idle), 8
        )


def test_deadline_and_target_release_coalesced_tails():
    assert defer(scheduler(age=0.5))
    assert not defer(scheduler(age=1.1))
    assert not defer(scheduler(age=0.01, count=8))


def test_idle_is_immediate_and_backlog_yields_a_decode_step():
    assert not defer(scheduler(age=0.01), idle=True)
    assert defer(scheduler(age=10, count=16, last_tail=True))


def test_remote_rank_deadline_controls_dispatch_without_cuda_sync():
    s = scheduler(age=0.5, tp_size=2)

    def remote_expired(value, *, op, group):
        assert value.device.type == "cpu"
        assert op == torch.distributed.ReduceOp.MAX
        assert group is s.tp_cpu_group
        assert value.item() == 0
        value.fill_(1)

    with patch(
        "torch.distributed.all_reduce", side_effect=remote_expired
    ) as collective:
        assert not defer(s)
        collective.assert_called_once()
