import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, *, stream_interval=None):
        self.rid = "req"
        self.beam_group = None
        self.weight_version_events = []
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.stream = True
        self.sampling_params = SimpleNamespace(
            stream_interval=stream_interval,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = []
        self.send_token_offset = 0
        self.send_decode_id_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = [100]
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cached_tokens_device = 0
        self.cached_tokens_host = 0
        self.cached_tokens_storage = 0
        self.retraction_count = 0
        self.time_stats = None
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.mm_token_counts_ready = True
        self.customized_info = None
        self.is_retracted = False
        self.return_logprob = True
        self.input_logprob_sent = True
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0
        self.spec_num_block_accept_tokens = 0
        self.spec_num_cap_tokens = 0
        self.spec_correct_drafts_histogram = []
        self.spec_cap_lens_histogram = []
        self.stop_prefix = False
        self.get_mm_token_counts = lambda: (
            self.mm_image_tokens,
            self.mm_audio_tokens,
            self.mm_video_tokens,
        )
        self.logprob = SimpleNamespace(
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
        )

    def commit(self, count):
        for _ in range(count):
            token = len(self.output_ids) + 1
            self.output_ids.append(token)
            self.logprob.output_token_logprobs_val.append(-token / 10)
            self.logprob.output_token_logprobs_idx.append(token)
            self.logprob.output_top_logprobs_val.append([-token / 10])
            self.logprob.output_top_logprobs_idx.append([token])
            self.logprob.output_token_ids_logprobs_val.append([-token / 10])
            self.logprob.output_token_ids_logprobs_idx.append([token])

    def finished(self):
        return self.finished_reason is not None

    @property
    def output_ids_through_stop(self):
        return self.output_ids[: self.finished_len]

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return self.stop_prefix


class TestOutputStreamerStreamInterval(unittest.TestCase):
    def setUp(self):
        for name, value in (
            (
                "get_serving",
                SimpleNamespace(stream_interval=4, weight_version="default"),
            ),
            ("get_parallel", SimpleNamespace(dp_rank=0, attn_tp_rank=0)),
            (
                "get_observability",
                SimpleNamespace(enable_request_time_stats_logging=False),
            ),
        ):
            patcher = patch(
                f"sglang.srt.managers.scheduler_components.output_streamer.{name}",
                return_value=value,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

        self.outputs = []
        self.streamer = SchedulerOutputStreamer(
            send_to_detokenizer=SimpleNamespace(send_output=self.outputs.append),
            tree_cache=None,
            server_args=SimpleNamespace(),
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: False,
        )

    def step(self, req, count):
        req.commit(count)
        self.streamer._stream_output_generation([req], return_logprob=True)

    def assert_emitted(self, chunk_lengths):
        self.assertEqual(
            [len(output.output_ids[0]) for output in self.outputs], chunk_lengths
        )
        tokens = list(range(1, sum(chunk_lengths) + 1))
        for field in ("output_ids", "decode_ids", "output_token_logprobs_idx"):
            self.assertEqual(
                [
                    token
                    for output in self.outputs
                    for token in getattr(output, field)[0]
                ],
                tokens,
                field,
            )
        self.assertEqual(
            [
                value
                for output in self.outputs
                for value in output.output_token_logprobs_val[0]
            ],
            [-token / 10 for token in tokens],
        )

    def test_speculative_blocks_do_not_need_to_land_on_modulo_boundary(self):
        req = _FakeReq()
        # Lengths 1, 4, 8, 12, 16: after the first token, none land on 1 mod 4.
        for count in (1, 3, 4, 4, 4):
            self.step(req, count)
        self.assert_emitted([1, 7, 4, 4])

    def test_varying_acceptance_lengths_preserve_all_tokens_and_logprobs(self):
        req = _FakeReq()
        for count in (1, 2, 1, 5, 1, 7, 4):
            self.step(req, count)
        self.assert_emitted([1, 8, 8, 4])

    def test_first_speculative_block_is_emitted_immediately(self):
        req = _FakeReq()
        self.step(req, 4)
        self.assert_emitted([4])

    def test_single_token_decoding_keeps_first_token_and_interval_cadence(self):
        self.streamer.spec_algorithm = SpeculativeAlgorithm.NONE
        req = _FakeReq()
        for _ in range(9):
            self.step(req, 1)
        self.assert_emitted([1, 4, 4])

    def test_request_interval_overrides_server_default(self):
        req = _FakeReq(stream_interval=2)
        for _ in range(5):
            self.step(req, 1)
        self.assert_emitted([1, 2, 2])

    def test_interval_one_emits_each_nonempty_block_once(self):
        req = _FakeReq(stream_interval=1)
        for count in (0, 3, 0, 4, 0):
            self.step(req, count)
        self.assert_emitted([3, 4])

    def test_stop_prefix_defers_emission_without_advancing_offsets(self):
        req = _FakeReq()
        self.step(req, 1)
        req.stop_prefix = True
        self.step(req, 4)
        self.assertEqual(req.send_token_offset, 1)
        self.assertEqual(req.send_output_token_logprobs_offset, 1)
        req.stop_prefix = False
        self.step(req, 1)
        self.assert_emitted([1, 5])

    def test_stop_prefix_can_defer_the_first_block(self):
        req = _FakeReq()
        req.stop_prefix = True
        self.step(req, 4)
        self.assertEqual(self.outputs, [])
        req.stop_prefix = False
        self.step(req, 1)
        self.assert_emitted([5])

    def test_terminal_flush_trims_speculative_overshoot_and_is_not_duplicated(self):
        req = _FakeReq()
        self.step(req, 1)
        req.stop_prefix = True
        req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "stop"})
        req.finished_len = 3
        self.step(req, 4)  # Tokens 4 and 5 are beyond the stop position.
        self.step(req, 0)  # Overlap scheduling visits the finished request again.
        self.assert_emitted([1, 2])
        self.assertEqual(self.outputs[-1].finished_reasons, [{"type": "stop"}])
        self.assertEqual(req.send_token_offset, 3)
        self.assertEqual(req.send_output_token_logprobs_offset, 3)

    def test_terminal_event_is_sent_even_without_new_tokens(self):
        req = _FakeReq()
        self.step(req, 1)
        req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "abort"})
        self.step(req, 0)
        self.step(req, 0)
        self.assert_emitted([1, 0])
        self.assertEqual(self.outputs[-1].finished_reasons, [{"type": "abort"}])


if __name__ == "__main__":
    unittest.main()
