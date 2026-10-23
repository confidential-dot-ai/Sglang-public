import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeStream:
    def __init__(self, stream_id):
        self.cuda_stream = stream_id


class FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def _make_scheduler(**attrs):
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(attrs)
    return scheduler


class TestPPCommOverlap(CustomTestCase):
    def test_graph_proxy_send_records_forward_reuse_fence(self):
        comm_stream = FakeStream(4)
        work = Mock()
        works = [SimpleNamespace(work=work)]
        scheduler = _make_scheduler(
            pp_comm_stream=comm_stream,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
            device_module=SimpleNamespace(Event=FakeEvent),
        )

        scheduler._pp_commit_comm_work(works, fence_next_forward=True)

        work.wait.assert_called_once_with()
        self.assertEqual(works, [])
        self.assertIs(scheduler.pp_send_done_event.recorded_stream, comm_stream)

    def test_no_fence_event_without_comm_stream(self):
        scheduler = _make_scheduler(
            pp_comm_stream=None,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
        )

        scheduler._pp_commit_comm_work([SimpleNamespace(work=Mock())], True)

        self.assertIsNone(scheduler.pp_send_done_event)

    def test_forward_waits_for_graph_send_and_proxy_receive(self):
        schedule_stream = FakeStream(1)
        send_done_event = object()
        recv_event = object()
        forward_stream = Mock()
        scheduler = _make_scheduler(
            schedule_stream=schedule_stream,
            forward_stream=forward_stream,
            pp_send_done_event=send_done_event,
            pp_proxy_recv_event=recv_event,
        )

        scheduler._pp_wait_forward_dependencies()

        forward_stream.wait_stream.assert_called_once_with(schedule_stream)
        self.assertEqual(
            forward_stream.wait_event.call_args_list,
            [call(send_done_event), call(recv_event)],
        )
        self.assertIsNone(scheduler.pp_send_done_event)
        self.assertIsNone(scheduler.pp_proxy_recv_event)

    def test_overlapped_tensors_stay_owned_by_the_streams_that_use_them(self):
        """With comm overlap, a received proxy must not be reallocated to the next
        recv while the forward still reads it, nor a sent one to the next forward
        while NCCL still reads it (silently wrong prefill activations)."""
        comm, schedule, forward, copy = (FakeStream(i) for i in range(4))
        received = Mock(spec=torch.Tensor, is_cuda=True)
        sent = Mock(spec=torch.Tensor, is_cuda=True)
        pp_group = Mock()
        pp_group.recv_tensor_dict.return_value = {
            "__msg_type__": "proxy",
            "hidden_states": received,
        }
        pp_group.send_tensor_dict.return_value = []
        scheduler = _make_scheduler(
            pp_comm_stream=comm,
            pp_comm_stream_ctx=nullcontext(),
            schedule_stream=schedule,
            forward_stream=forward,
            copy_stream=copy,
            device_module=SimpleNamespace(Event=FakeEvent),
            pp_group=pp_group,
            attn_tp_group=None,
            _pp_tensor_dict_inbox=defaultdict(deque),
        )

        scheduler._pp_recv_typed_dict("proxy")
        scheduler._pp_send_dict_to_next_stage({"hidden_states": sent}, msg_type="proxy")

        self.assertEqual(
            received.record_stream.call_args_list,
            [call(schedule), call(forward), call(copy)],
        )
        sent.record_stream.assert_called_once_with(comm)

    def test_inbox_returns_original_receive_event(self):
        recv_event = object()
        tensor_dict = {"__msg_type__": "output", "value": torch.arange(2)}
        scheduler = _make_scheduler(
            _pp_tensor_dict_inbox=defaultdict(
                deque, {"output": deque([(tensor_dict, recv_event)])}
            ),
        )

        received, event = scheduler._pp_recv_typed_dict("output")

        self.assertIs(received, tensor_dict)
        self.assertIs(event, recv_event)


if __name__ == "__main__":
    unittest.main()
