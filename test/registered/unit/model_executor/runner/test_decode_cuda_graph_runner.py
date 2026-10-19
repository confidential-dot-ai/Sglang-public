"""Unit tests for ``DecodeCudaGraphRunner`` capture-phase profiling — CPU-only.

Two capture-trace modes plus their precedence:

  * **Original single-trace** (``SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE``):
    ``_init_profile_context_and_memory_record`` builds an *unscheduled* profiler
    (``record_shapes`` only, no schedule / no ``on_trace_ready``); the combined
    trace is exported in ``_post_process_after_profile`` via
    ``export_cuda_graph_capture_trace``.
  * **Per-batch-size traces** (``SGLANG_GRAPH_BATCH_CAPTURE``): a *scheduled*
    profiler (``wait=2, warmup=0, active=1, repeat=0``) with the trace-export
    knobs (record_shapes / with_stack / with_flops / profile_memory) and an
    ``on_trace_ready`` hook that writes one trace per captured shape to
    ``<SGLANG_TORCH_PROFILER_DIR>/graph_capture_profile/`` named
    ``{runner_name}_{tag}_rank{rank}.json.gz``, where ``tag`` comes from
    ``_set_profile_trace_tag`` (bs plus whichever variants are active).
  * **Precedence**: when both env vars are set, the original single-trace path
    wins (no per-bs schedule / dir / bookkeeping).

The profiler / CUDA-memory APIs are mocked; the directory + naming + schedule
logic is pure-Python and runs on CPU. The method is invoked unbound against a
lightweight stand-in (with the real precedence helper bound) so no model or
server is constructed.
"""

import os
import tempfile
import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.utils import profile_utils as putils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_CAPTURE_TRACE = "SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE"
_BATCH_CAPTURE = "SGLANG_GRAPH_BATCH_CAPTURE"


def _make_fake_self(capture_bs):
    """Stand-in ``self`` with the real precedence helper bound so the env-var
    gating in ``_init_profile_context_and_memory_record`` applies."""
    fake_self = SimpleNamespace(capture_bs=list(capture_bs))
    fake_self._graph_batch_capture_active = (
        DecodeCudaGraphRunner._graph_batch_capture_active.__get__(fake_self)
    )
    # _profiler is the gate _set_profile_trace_tag checks.
    fake_self._profiler = mock.Mock()
    fake_self._set_profile_trace_tag = (
        DecodeCudaGraphRunner._set_profile_trace_tag.__get__(fake_self)
    )
    fake_self._profile_runner_name = DecodeCudaGraphRunner._profile_runner_name.__get__(
        fake_self
    )
    return fake_self


def _make_runner(cls, *, is_draft_worker):
    """Uninitialized runner of ``cls`` with only the model_runner flag the
    trace-naming helper reads."""
    runner = cls.__new__(cls)
    runner.model_runner = SimpleNamespace(is_draft_worker=is_draft_worker)
    return runner


class TestInitProfileBatchMode(CustomTestCase):
    """SGLANG_GRAPH_BATCH_CAPTURE -> scheduled per-bs profiler."""

    def _invoke(self, *, capture_bs, rank=0, profiler_dir=None):
        fake_self = _make_fake_self(capture_bs)
        env = {_BATCH_CAPTURE: "1"}
        if profiler_dir is not None:
            env["SGLANG_TORCH_PROFILER_DIR"] = profiler_dir
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(tp_rank=rank)
            ),
            mock.patch.object(mod, "profile") as mock_profile,
            mock.patch("torch.profiler.schedule") as mock_schedule,
            mock.patch(
                "torch.cuda.memory._record_memory_history"
            ) as mock_record_history,
        ):
            os.environ.pop(_CAPTURE_TRACE, None)  # original flag off
            if profiler_dir is None:
                os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)
            ctx = DecodeCudaGraphRunner._init_profile_context_and_memory_record(
                fake_self
            )
        self.assertIs(ctx, mock_profile.return_value)
        return fake_self, mock_profile, mock_schedule, mock_record_history

    def test_creates_graph_capture_profile_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._invoke(capture_bs=[1, 2, 4], profiler_dir=tmp)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "graph_capture_profile")))

    def test_primes_empty_trace_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, *_ = self._invoke(capture_bs=[1, 2, 4, 8], profiler_dir=tmp)
            self.assertIsNone(fake_self._profile_trace_tag)


class TestInitProfileOriginalMode(CustomTestCase):
    """No flag, original flag only, or both (precedence) -> unscheduled pass with
    no per-bs schedule / directory / bookkeeping."""

    def _invoke_original(self, *, env):
        fake_self = _make_fake_self([1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            environ = dict(env)
            environ["SGLANG_TORCH_PROFILER_DIR"] = tmp
            with (
                mock.patch.dict(os.environ, environ, clear=False),
                mock.patch.object(
                    mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
                ),
                mock.patch.object(mod, "profile") as mock_profile,
                mock.patch("torch.profiler.schedule") as mock_schedule,
                mock.patch("torch.cuda.memory._record_memory_history"),
            ):
                for k in (_CAPTURE_TRACE, _BATCH_CAPTURE):
                    if k not in environ:
                        os.environ.pop(k, None)
                DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
            kwargs = mock_profile.call_args.kwargs
            # Unscheduled pass: record_shapes only, no schedule / on_trace_ready.
            self.assertTrue(kwargs["record_shapes"])
            self.assertIsNone(kwargs.get("schedule"))
            self.assertIsNone(kwargs.get("on_trace_ready"))
            mock_schedule.assert_not_called()
            self.assertFalse(os.path.isdir(os.path.join(tmp, "graph_capture_profile")))
            self.assertFalse(hasattr(fake_self, "_profile_trace_tag"))

    def test_no_flags(self):
        self._invoke_original(env={})

    def test_original_flag_only(self):
        self._invoke_original(env={_CAPTURE_TRACE: "1"})

    def test_both_flags_original_takes_precedence(self):
        self._invoke_original(env={_CAPTURE_TRACE: "1", _BATCH_CAPTURE: "1"})


class TestOnTraceReadyNaming(CustomTestCase):
    @staticmethod
    def _build_on_trace_ready(*, capture_bs, rank, tmp, runner_name=None):
        fake_self = _make_fake_self(capture_bs)
        if runner_name is not None:
            fake_self._profile_runner_name = lambda: runner_name
        with (
            mock.patch.dict(
                os.environ,
                {"SGLANG_TORCH_PROFILER_DIR": tmp, _BATCH_CAPTURE: "1"},
                clear=False,
            ),
            mock.patch.object(
                mod, "get_parallel", return_value=SimpleNamespace(tp_rank=rank)
            ),
            mock.patch.object(mod, "profile") as mock_profile,
            mock.patch("torch.profiler.schedule"),
            mock.patch("torch.cuda.memory._record_memory_history"),
        ):
            os.environ.pop(_CAPTURE_TRACE, None)
            DecodeCudaGraphRunner._init_profile_context_and_memory_record(fake_self)
        on_trace_ready = mock_profile.call_args.kwargs["on_trace_ready"]
        return fake_self, on_trace_ready

    @staticmethod
    def _flush(on_trace_ready, exported):
        prof = mock.Mock()
        prof.export_chrome_trace.side_effect = exported.append
        on_trace_ready(prof)

    def test_single_variant_keeps_bs_only_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[1, 2, 4], rank=0, tmp=tmp
            )
            trace_dir = os.path.join(tmp, "graph_capture_profile")
            runner = type(fake_self).__name__

            exported = []
            for bs in (4, 2, 1):
                fake_self._set_profile_trace_tag(bs, None, None, None)
                self._flush(on_trace_ready, exported)

            self.assertEqual(
                exported,
                [
                    os.path.join(trace_dir, f"{runner}_bs_4_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_2_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_1_rank0.json.gz"),
                ],
            )

    def test_dual_attention_variants_name_both_traces(self):
        """DSA models capture dense + sparse per bs, so two flushes per bs."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[1, 2], rank=0, tmp=tmp
            )
            trace_dir = os.path.join(tmp, "graph_capture_profile")
            runner = type(fake_self).__name__

            exported = []
            for bs in (2, 1):
                for attention_variant in ("dense", "sparse"):
                    fake_self._set_profile_trace_tag(bs, None, None, attention_variant)
                    self._flush(on_trace_ready, exported)

            self.assertEqual(
                exported,
                [
                    os.path.join(trace_dir, f"{runner}_bs_2_dense_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_2_sparse_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_1_dense_rank0.json.gz"),
                    os.path.join(trace_dir, f"{runner}_bs_1_sparse_rank0.json.gz"),
                ],
            )
            self.assertEqual(len(set(exported)), len(exported))

    def test_more_flushes_than_batch_sizes_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[1, 2], rank=0, tmp=tmp
            )
            exported = []
            for i in range(6):
                fake_self._set_profile_trace_tag(2, None, None, f"v{i}")
                self._flush(on_trace_ready, exported)
            self.assertEqual(len(exported), 6)

    def test_unset_tag_skips_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[8], rank=0, tmp=tmp
            )
            prof = mock.Mock()
            on_trace_ready(prof)  # tag never set by the capture loop
            prof.export_chrome_trace.assert_not_called()

    def test_rank_in_trace_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_self, on_trace_ready = self._build_on_trace_ready(
                capture_bs=[8], rank=3, tmp=tmp
            )
            runner = type(fake_self).__name__
            fake_self._set_profile_trace_tag(8, None, None, None)
            prof = mock.Mock()
            on_trace_ready(prof)
            prof.export_chrome_trace.assert_called_once_with(
                os.path.join(
                    tmp, "graph_capture_profile", f"{runner}_bs_8_rank3.json.gz"
                )
            )


class TestProfileTraceTag(CustomTestCase):
    """_set_profile_trace_tag composition."""

    def test_bs_only_when_no_variants(self):
        fake_self = _make_fake_self([1])
        fake_self._set_profile_trace_tag(16, None, None, None)
        self.assertEqual(fake_self._profile_trace_tag, "bs_16")

    def test_appends_attention_lora_and_stream(self):
        fake_self = _make_fake_self([1])
        fake_self._set_profile_trace_tag(8, 1, "nolora", "sparse")
        self.assertEqual(fake_self._profile_trace_tag, "bs_8_sparse_nolora_stream_1")

    def test_noop_without_scheduled_profiler(self):
        fake_self = _make_fake_self([1])
        fake_self._profiler = None
        fake_self._set_profile_trace_tag(8, None, None, "sparse")
        self.assertFalse(hasattr(fake_self, "_profile_trace_tag"))


class TestProfileRunnerName(CustomTestCase):
    """DSPARK / DFLASH drafts capture with the base class at the target's
    shapes, so their traces must not share the target's runner name."""

    def test_target_keeps_class_name(self):
        runner = _make_runner(DecodeCudaGraphRunner, is_draft_worker=False)
        self.assertEqual(runner._profile_runner_name(), "DecodeCudaGraphRunner")

    def test_base_class_draft_worker_is_prefixed(self):
        runner = _make_runner(DecodeCudaGraphRunner, is_draft_worker=True)
        self.assertEqual(runner._profile_runner_name(), "DraftDecodeCudaGraphRunner")

    def test_draft_subclass_keeps_class_name(self):
        class EAGLEDraftLikeRunner(DecodeCudaGraphRunner):
            pass

        runner = _make_runner(EAGLEDraftLikeRunner, is_draft_worker=True)
        self.assertEqual(runner._profile_runner_name(), "EAGLEDraftLikeRunner")

    def test_target_and_draft_traces_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            exported = []
            for is_draft_worker in (False, True):
                name = _make_runner(
                    DecodeCudaGraphRunner, is_draft_worker=is_draft_worker
                )._profile_runner_name()
                fake_self, on_trace_ready = (
                    TestOnTraceReadyNaming._build_on_trace_ready(
                        capture_bs=[1, 2], rank=0, tmp=tmp, runner_name=name
                    )
                )
                for bs in (2, 1):
                    fake_self._set_profile_trace_tag(bs, None, None, None)
                    TestOnTraceReadyNaming._flush(on_trace_ready, exported)
            self.assertEqual(len(exported), 4)
            self.assertEqual(len(set(exported)), len(exported))


class TestOriginalTraceExport(CustomTestCase):
    """export_cuda_graph_capture_trace (original single combined trace per rank),
    gated by SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE, and the shared dir helper.
    Both trace modes land under graph_capture_profile/."""

    def test_writes_named_trace_when_flag_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(
                    os.environ,
                    {"SGLANG_TORCH_PROFILER_DIR": tmp, _CAPTURE_TRACE: "1"},
                    clear=False,
                ),
                mock.patch.object(
                    putils, "get_parallel", return_value=SimpleNamespace(tp_rank=2)
                ),
            ):
                prof = mock.Mock()
                putils.export_cuda_graph_capture_trace(
                    prof, runner_name="DecodeCudaGraphRunner"
                )
                expected = os.path.join(
                    tmp,
                    "graph_capture_profile",
                    "cuda_graph_capture-DecodeCudaGraphRunner-TP-2.json.gz",
                )
                prof.export_chrome_trace.assert_called_once_with(expected)
                self.assertTrue(
                    os.path.isdir(os.path.join(tmp, "graph_capture_profile"))
                )

    def test_noop_when_flag_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ, {"SGLANG_TORCH_PROFILER_DIR": tmp}, clear=False
            ):
                os.environ.pop(_CAPTURE_TRACE, None)
                prof = mock.Mock()
                putils.export_cuda_graph_capture_trace(
                    prof, runner_name="DecodeCudaGraphRunner"
                )
                prof.export_chrome_trace.assert_not_called()
                self.assertFalse(
                    os.path.isdir(os.path.join(tmp, "graph_capture_profile"))
                )


class TestRaggedCaptureLayoutLifetime(CustomTestCase):
    def _runner(self, *, enable_pdmux=False):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.enable_pdmux = enable_pdmux
        runner._captured_ragged_layouts = {}
        runner.capture_num_tokens = [18, 36]
        runner.captured_req_width = 9
        runner.max_bs = 4
        runner.device = "cpu"
        return runner

    def test_reuses_layout_across_graph_variants(self):
        """Every graph variant for one bucket must retain the buffer replay
        stages, rather than an earlier variant keeping an orphaned raw pointer."""
        runner = self._runner()
        layout = object()
        fake_layout_type = SimpleNamespace(
            from_verify_lens=mock.Mock(return_value=layout)
        )
        with mod.envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.override(
            False
        ), mock.patch(
            "sglang.srt.speculative.ragged_verify.RaggedVerifyLayout",
            fake_layout_type,
        ):
            first = runner._capture_ragged_verify_layout(18)
            second = runner._capture_ragged_verify_layout(18)

        self.assertIs(first, layout)
        self.assertIs(second, layout)
        fake_layout_type.from_verify_lens.assert_called_once()
        self.assertIs(runner._captured_ragged_layouts[(None, 18)], layout)

    def test_pdmux_lanes_do_not_share_staging_buffers(self):
        runner = self._runner(enable_pdmux=True)
        layouts = [object(), object()]
        fake_layout_type = SimpleNamespace(
            from_verify_lens=mock.Mock(side_effect=layouts)
        )
        with mod.envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.override(
            False
        ), mock.patch(
            "sglang.srt.speculative.ragged_verify.RaggedVerifyLayout",
            fake_layout_type,
        ):
            lane0 = runner._capture_ragged_verify_layout(18, stream_idx=0)
            lane1 = runner._capture_ragged_verify_layout(18, stream_idx=1)

        self.assertIs(lane0, layouts[0])
        self.assertIs(lane1, layouts[1])
        self.assertIsNot(lane0, lane1)
        self.assertEqual(fake_layout_type.from_verify_lens.call_count, 2)

    def test_compact_verify_allows_token_denominated_sum_capture(self):
        mod._validate_compact_ragged_graph_experiments(
            ragged_verify_mode=True,
            dp_verify_sum_capture=True,
        )

    def test_static_verify_keeps_sum_capture_available(self):
        mod._validate_compact_ragged_graph_experiments(
            ragged_verify_mode=False,
            dp_verify_sum_capture=True,
        )

    def test_compact_and_sum_graphs_capture_non_tbo_body(self):
        self.assertFalse(
            mod._tbo_cuda_graph_enabled(
                enable_two_batch_overlap=True,
                ragged_verify_mode=True,
                dp_verify_sum_capture=False,
                attention_backend_supports_tbo_graph=True,
            )
        )
        self.assertFalse(
            mod._tbo_cuda_graph_enabled(
                enable_two_batch_overlap=True,
                ragged_verify_mode=False,
                dp_verify_sum_capture=True,
                attention_backend_supports_tbo_graph=True,
            )
        )
        self.assertTrue(
            mod._tbo_cuda_graph_enabled(
                enable_two_batch_overlap=True,
                ragged_verify_mode=False,
                dp_verify_sum_capture=False,
                attention_backend_supports_tbo_graph=True,
            )
        )
        self.assertFalse(
            mod._tbo_cuda_graph_enabled(
                enable_two_batch_overlap=True,
                ragged_verify_mode=False,
                dp_verify_sum_capture=False,
                attention_backend_supports_tbo_graph=False,
            )
        )

    def test_static_tbo_replay_replans_from_padded_view(self):
        stale_live_child = object()
        fb_view = SimpleNamespace(
            tbo_split_seq_index=3,
            tbo_children=[stale_live_child, stale_live_child],
        )

        mod._set_replay_tbo_state(
            fb_view,
            enable_two_batch_overlap=True,
            tbo_cuda_graph_enabled=True,
            padded_bs=16,
        )

        # TboAttnBackend uses the marker to enter its replay split, and None
        # children force it to derive 8+8 views from the padded graph bucket
        # instead of reusing the live batch's potentially different split.
        self.assertEqual(fb_view.tbo_split_seq_index, 8)
        self.assertIsNone(fb_view.tbo_children)

    def test_non_tbo_graph_clears_live_scheduler_split(self):
        fb_view = SimpleNamespace(
            tbo_split_seq_index=3,
            tbo_children=[object(), object()],
        )

        mod._set_replay_tbo_state(
            fb_view,
            enable_two_batch_overlap=True,
            tbo_cuda_graph_enabled=False,
            padded_bs=16,
        )

        self.assertIsNone(fb_view.tbo_split_seq_index)
        self.assertIsNone(fb_view.tbo_children)

    @staticmethod
    def _layout(verify_lens, *, graph_num_tokens=18, cap=None):
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        verify_lens = torch.tensor(verify_lens, dtype=torch.int32)
        qo_indptr = torch.nn.functional.pad(
            torch.cumsum(verify_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        return RaggedVerifyLayout(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            extend_start_loc=qo_indptr[:-1].clone(),
            qo_indptr_device=qo_indptr,
            cap=cap,
        )

    def test_metadata_stages_own_and_promote_private_ragged_layouts(self):
        runner = self._runner()
        captured = self._layout([9, 9])
        runner._captured_ragged_layouts[(None, 18)] = captured
        runner._metadata_stages = [
            {"ragged_layouts": {}},
            {"ragged_layouts": {}},
        ]
        runner._initialize_metadata_stage_ragged_layouts()

        private0 = runner._metadata_stages[0]["ragged_layouts"][(None, 18)]
        private1 = runner._metadata_stages[1]["ragged_layouts"][(None, 18)]
        self.assertIsNot(private0, private1)
        self.assertNotEqual(
            private0.verify_lens.data_ptr(), private1.verify_lens.data_ptr()
        )
        self.assertNotEqual(
            private0.verify_lens.data_ptr(), captured.verify_lens.data_ptr()
        )

        live = self._layout([3, 5], cap=runner.captured_req_width)
        staged, layout_key = runner._stage_private_ragged_verify_layout(
            runner._metadata_stages[0], live, 18
        )

        # Background staging populated only its private snapshot.
        self.assertEqual(captured.verify_lens.tolist(), [9, 9])
        self.assertEqual(staged.verify_lens.tolist(), [3, 5])
        self.assertEqual(staged.extend_start_loc.tolist(), [0, 3])
        self.assertEqual(staged.qo_indptr_device.tolist(), [0, 3, 8])

        runner._promote_private_ragged_verify_layout(staged, layout_key)
        self.assertEqual(captured.verify_lens.tolist(), [3, 5])
        self.assertEqual(captured.extend_start_loc.tolist(), [0, 3])
        self.assertEqual(captured.qo_indptr_device.tolist(), [0, 3, 8])

    def test_ragged_dispatch_snapshot_preserves_token_bucket(self):
        runner = self._runner()
        runner.raw_bs = 2
        runner.raw_num_token = 18
        runner.bs = 4
        runner._ragged_graph_size = 18
        runner._replay_graph_key = mod.ShapeKey(size=18)
        runner._replay_dp_verify_sum_bucket = 144
        staged_layout = object()

        state = runner._snapshot_replay_dispatch_state(
            padded_num_tokens=18,
            ragged_layout_key=(None, 18),
            staged_ragged_layout=staged_layout,
        )
        # The compact token bucket is 18, not padded slots * static width (36).
        self.assertEqual(state.padded_num_tokens, 18)
        self.assertNotEqual(state.padded_num_tokens, state.bs * 9)

        runner.raw_bs = 99
        runner.raw_num_token = 99
        runner.bs = 99
        runner._ragged_graph_size = 99
        runner._replay_graph_key = mod.ShapeKey(size=99)
        runner._replay_dp_verify_sum_bucket = 99
        runner._restore_replay_dispatch_state(state)
        self.assertEqual(runner.raw_bs, 2)
        self.assertEqual(runner.raw_num_token, 18)
        self.assertEqual(runner.bs, 4)
        self.assertEqual(runner._ragged_graph_size, 18)
        self.assertIs(runner._replay_graph_key, state.replay_graph_key)
        self.assertEqual(runner._replay_dp_verify_sum_bucket, 144)
        static_state = runner._snapshot_replay_dispatch_state(
            padded_num_tokens=runner.bs * 9
        )
        self.assertEqual(static_state.padded_num_tokens, 36)

    def test_metadata_stream_waits_for_current_stream_producer(self):
        log = []

        class FakeEvent:
            def __init__(self, name):
                self.name = name

            def record(self, stream):
                log.append(("record", self.name, stream))

        class FakeMetadataStream:
            def wait_event(self, event):
                log.append(("wait", event.name))

        events = iter(
            [
                FakeEvent("producer"),
                FakeEvent("gpu_start"),
                FakeEvent("ready"),
                FakeEvent("producer_1"),
                FakeEvent("gpu_start_1"),
                FakeEvent("ready_1"),
            ]
        )
        producer_stream = object()
        metadata_stream = FakeMetadataStream()
        dispatch = mod._ReplayDispatchState(
            raw_bs=2,
            raw_num_token=18,
            bs=4,
            ragged_graph_size=18,
            replay_graph_key=mod.ShapeKey(size=18),
            replay_dp_verify_sum_bucket=None,
            padded_num_tokens=18,
        )
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner._dspark_metadata_double_buffer = True
        runner._metadata_stage_cursor = 0
        runner._metadata_stream = metadata_stream
        runner._metadata_stages = [
            {
                "registry": object(),
                "consumed": None,
                "producer_ready": None,
                "dispatch_state": None,
                "ragged_layouts": {},
            },
            {
                "registry": object(),
                "consumed": None,
                "producer_ready": None,
                "dispatch_state": None,
                "ragged_layouts": {},
            },
        ]
        runner._staged_batches = {}
        runner._metadata_gpu_samples = []
        runner.device = "cuda"
        runner.load_batch = mock.Mock(
            side_effect=lambda *_args, **_kwargs: (
                log.append(("load_batch",)),
                dispatch,
            )[1]
        )
        runner.record_metadata_timing = mock.Mock()
        runner._drain_metadata_gpu_timings = mock.Mock()
        batch = object()
        batch_1 = object()

        with mock.patch.object(
            torch.cuda, "current_stream", return_value=producer_stream
        ), mock.patch.object(
            torch.cuda, "Event", side_effect=lambda *args, **kwargs: next(events)
        ), mock.patch.object(
            torch.cuda, "stream", return_value=mod.contextlib.nullcontext()
        ):
            runner.stage_batch(batch)
            runner.stage_batch(batch_1)
            with self.assertRaisesRegex(
                RuntimeError, "reused before its prior batch was promoted"
            ):
                runner.stage_batch(object())

        producer_record = log.index(("record", "producer", producer_stream))
        producer_wait = log.index(("wait", "producer"))
        load = log.index(("load_batch",))
        self.assertLess(producer_record, producer_wait)
        self.assertLess(producer_wait, load)
        self.assertEqual(
            runner._metadata_stages[0]["dispatch_state"].padded_num_tokens, 18
        )
        self.assertEqual(
            runner._metadata_stages[1]["dispatch_state"].padded_num_tokens, 18
        )

    def test_discard_staged_batch_releases_slot_after_graph_miss(self):
        ready = object()
        state = mod._ReplayDispatchState(
            raw_bs=2,
            raw_num_token=18,
            bs=4,
            ragged_graph_size=18,
            replay_graph_key=mod.ShapeKey(size=18),
            replay_dp_verify_sum_bucket=None,
            padded_num_tokens=18,
        )
        stage = {"consumed": None, "dispatch_state": state}
        batch = object()
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner._staged_batches = {
            id(batch): {"stage": stage, "ready": ready}
        }

        runner.discard_staged_batch(batch)

        self.assertEqual(runner._staged_batches, {})
        self.assertIsNone(stage["dispatch_state"])
        self.assertIs(stage["consumed"], ready)
        # Repeated cleanup is an intentional no-op on defensive eager paths.
        runner.discard_staged_batch(batch)

    def test_promotion_waits_ready_then_copies_private_layout_and_dispatch(self):
        log = []

        class FakeEvent:
            def __init__(self, name):
                self.name = name

            def record(self, stream):
                log.append(("record", self.name))

        class FakeCurrentStream:
            def wait_event(self, event):
                log.append(("wait", event.name))

        events = iter([FakeEvent("promote_start"), FakeEvent("promote_end")])
        state = mod._ReplayDispatchState(
            raw_bs=2,
            raw_num_token=18,
            bs=4,
            ragged_graph_size=18,
            replay_graph_key=mod.ShapeKey(size=18),
            replay_dp_verify_sum_bucket=None,
            padded_num_tokens=18,
            ragged_layout_key=(None, 18),
            staged_ragged_layout=object(),
        )
        stage = {
            "registry": object(),
            "consumed": None,
            "dispatch_state": state,
        }
        ready = FakeEvent("ready")
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.device = "cuda"
        fb_view = SimpleNamespace(spec_info=object())
        runner.buffer_registry = SimpleNamespace(
            copy_from_registry=lambda source, **kwargs: log.append(
                ("registry", source, kwargs)
            ),
            extract_buffer=lambda **_kwargs: fb_view,
        )
        runner._promote_private_ragged_verify_layout = mock.Mock(
            side_effect=lambda *_args: log.append(("ragged",))
        )
        runner._metadata_gpu_samples = []
        runner._drain_metadata_gpu_timings = mock.Mock()
        runner.raw_bs = runner.raw_num_token = runner.bs = -1
        runner._ragged_graph_size = -1
        runner._replay_graph_key = mod.ShapeKey(size=1)
        runner._replay_dp_verify_sum_bucket = 99
        runner.buffers = SimpleNamespace(num_token_non_padded=None)
        runner.enable_pdmux = False
        runner.tbo_cuda_graph_enabled = False
        runner.enable_two_batch_overlap = False
        runner.capture_forward_mode = mod.ForwardMode.TARGET_VERIFY
        runner.seq_len_fill_value = 1
        backend = SimpleNamespace(
            init_forward_metadata_out_graph=lambda _fb: log.append(("backend",))
        )
        runner._replay_attn_backend = lambda: backend
        runner.record_metadata_timing = mock.Mock()
        runner.model_runner = SimpleNamespace(hisparse_coordinator=None)
        forward_batch = SimpleNamespace(
            spec_info=object(),
            forward_mode=mod.ForwardMode.TARGET_VERIFY,
            seq_lens_sum=42,
        )

        with mock.patch.object(
            torch.cuda, "current_stream", return_value=FakeCurrentStream()
        ), mock.patch.object(
            torch.cuda, "Event", side_effect=lambda *args, **kwargs: next(events)
        ):
            runner._promote_staged_batch(
                {"stage": stage, "ready": ready}, forward_batch
            )

        wait_index = log.index(("wait", "ready"))
        registry_item = next(item for item in log if item[0] == "registry")
        registry_index = log.index(registry_item)
        ragged_index = log.index(("ragged",))
        backend_index = log.index(("backend",))
        self.assertLess(wait_index, registry_index)
        self.assertLess(registry_index, ragged_index)
        self.assertLess(ragged_index, backend_index)
        self.assertEqual(registry_item[2]["padded_bs"], 4)
        self.assertEqual(registry_item[2]["padded_num_tokens"], 18)
        self.assertEqual(runner.raw_bs, 2)
        self.assertEqual(runner.raw_num_token, 18)
        self.assertEqual(runner.bs, 4)
        self.assertIs(runner._replay_graph_key, state.replay_graph_key)
        self.assertIs(stage["consumed"], runner._metadata_gpu_samples[0][2])
        self.assertIsNone(stage["dispatch_state"])

    def test_promotion_rejects_a_different_pdmux_lane(self):
        state = mod._ReplayDispatchState(
            raw_bs=2,
            raw_num_token=18,
            bs=4,
            ragged_graph_size=18,
            replay_graph_key=mod.ShapeKey(size=18, stream_idx=1),
            replay_dp_verify_sum_bucket=None,
            padded_num_tokens=18,
        )
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.enable_pdmux = True
        with mock.patch.object(mod, "get_current_stream_idx", return_value=0):
            with self.assertRaisesRegex(
                RuntimeError, "staged for PDMux lane 1, but replay is on lane 0"
            ):
                runner._promote_staged_batch(
                    {
                        "stage": {"dispatch_state": state},
                        "ready": object(),
                    },
                    SimpleNamespace(),
                )


class TestCaptureOomDegradesToEager(CustomTestCase):
    """An OOM'd capture shape must be recorded as eager-only, not crash the boot.

    The 1M-context boot died outright instead of serving without those graphs.
    Two black-box behaviors are pinned: an OOM is absorbed and the shape is
    recorded in ``_skipped_bs`` (which is what keeps ``can_run_graph``
    from admitting a bucket whose graph was never captured -- replay indexes
    ``self._graphs[shape_key]`` bare, so admitting one is a KeyError at serving
    time, not a slow path), while a non-OOM error still propagates so real bugs
    are not swallowed.
    """

    def _stub(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        stub = SimpleNamespace(
            _skipped_capture_shapes=[],
            _skipped_bs=frozenset(),
        )
        stub._skip_capture_shape_or_raise = MethodType(
            DecodeCudaGraphRunner._skip_capture_shape_or_raise, stub
        )
        return stub

    def test_oom_is_absorbed_and_the_shape_is_recorded(self):
        stub = self._stub()
        stub._skip_capture_shape_or_raise(
            torch.OutOfMemoryError("CUDA out of memory"), 128, 1024, None
        )
        self.assertEqual(stub._skipped_capture_shapes, [(128, 1024, None)])
        self.assertIn(128, stub._skipped_bs)

    def test_plain_runtime_error_naming_oom_is_also_absorbed(self):
        # FlashInfer workspace and driver graph allocations surface as a bare
        # RuntimeError, not torch.OutOfMemoryError.
        stub = self._stub()
        stub._skip_capture_shape_or_raise(
            RuntimeError("CUDA error: out of memory"), 64, 512, None
        )
        self.assertIn(64, stub._skipped_bs)

    def test_a_non_oom_failure_still_raises(self):
        stub = self._stub()
        with self.assertRaises(ValueError):
            stub._skip_capture_shape_or_raise(
                ValueError("a real bug"), 8, 64, None
            )
        self.assertEqual(stub._skipped_capture_shapes, [])
        self.assertEqual(stub._skipped_bs, frozenset())
class TestRaggedVerifyLayoutPerTier(CustomTestCase):
    """Every graph variant of a token tier must replay the staged live layout."""

    def test_replay_stages_layout_of_every_variant(self):
        import torch

        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.max_bs = 8
        runner.captured_req_width = 6
        runner.capture_num_tokens = [6, 12, 24, 48]
        runner.device = "cpu"
        runner.enable_pdmux = False
        runner._captured_ragged_layouts = {}
        with mock.patch.dict(
            os.environ, {"SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE": "0"}
        ):
            captured = {
                variant: runner._capture_ragged_verify_layout(24)
                for variant in ("candidate_unfiltered", "candidate_filtered")
            }

        live = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([6, 1, 6, 3, 6], dtype=torch.int32),
            graph_num_tokens=24,
        )
        expected = live.padded_to_bucket(padded_bs=8, cap=6)
        runner._stage_ragged_verify_layout(live, 24)
        for variant, layout in captured.items():
            with self.subTest(variant=variant):
                torch.testing.assert_close(
                    layout.qo_indptr_device, expected.qo_indptr_device
                )
                torch.testing.assert_close(layout.verify_lens, expected.verify_lens)


if __name__ == "__main__":
    unittest.main()
