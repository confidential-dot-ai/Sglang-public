"""Chat SSE error boundaries and tokenizer-stream cleanup (no model required)."""

import asyncio
import json
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChatStreamErrors(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                enable_cache_report=False,
                stream_response_default_include_usage=False,
            ),
            role="tokenizer",
        )
        self.closed = False
        self.tm = Mock()
        self.tm._config_overrides = {}
        self.tm.server_args.stream_response_default_include_usage = False
        self.tm.server_args.enable_cache_report = False
        self.tm.create_abort_task.return_value = AsyncMock()
        self.chat = OpenAIServingChat.__new__(OpenAIServingChat)
        self.chat.tokenizer_manager = self.tm
        self.chat.chat_encoding_spec = None
        self.chat.reasoning_parser = None
        self.request = ChatCompletionRequest(
            model="m3", messages=[{"role": "user", "content": "Hi"}], stream=True
        )
        self.adapted = GenerateReqInput(text="Hi", stream=True, rid="stream-test")
        self.chat._validate_request = Mock(return_value=None)
        self.chat._convert_to_internal_request = Mock(
            return_value=(self.adapted, self.request)
        )
        self.tm.request_logger.log_requests = False
        self.chat._generate_stream_content = self._content

    async def _content(self, **kwargs):
        if False:
            yield

    def _source(self, error=None, before_first=False, finish_reason=None):
        async def generate(*args):
            try:
                if before_first:
                    raise error
                yield {
                    "text": "",
                    "meta_info": {
                        "id": "stream-test",
                        "prompt_tokens": 1,
                        "finish_reason": finish_reason,
                    },
                }
                if error is not None:
                    raise error
            finally:
                self.closed = True

        self.tm.generate_request = generate

    async def _response(self):
        return await self.chat.handle_request(self.request, None)

    async def _chunks(self):
        response = await self._response()
        chunks = [chunk async for chunk in response.body_iterator]
        self.assertTrue(self.closed)
        return chunks

    def _assert_error(self, chunks, code):
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(chunks.count("data: [DONE]\n\n"), 1)
        payloads = [json.loads(chunk[6:]) for chunk in chunks[:-1]]
        errors = [payload["error"] for payload in payloads if "error" in payload]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["code"], code)
        for payload in payloads:
            for choice in payload.get("choices", []):
                self.assertIsNone(choice.get("finish_reason"))
        return errors[0]

    async def test_parser_exception_emits_error_and_done_and_closes_source(self):
        for exception in (OverflowError, TypeError, RuntimeError):
            with self.subTest(exception=exception):
                self.closed = False
                self._source(finish_reason={"type": "stop"})

                async def broken_content(**kwargs):
                    raise exception("parser failed")
                    yield  # pragma: no cover

                self.chat._generate_stream_content = broken_content
                error = self._assert_error(await self._chunks(), 500)
                self.assertEqual(error["type"], "InternalServerError")
                self.assertIn("parser failed", error["message"])

    async def test_generator_exception_after_first_chunk(self):
        self._source(RuntimeError("generation failed"))
        self._assert_error(await self._chunks(), 500)

    async def test_value_error_after_first_chunk_keeps_400(self):
        self._source(ValueError("invalid input"))
        self._assert_error(await self._chunks(), 400)

    async def test_errors_before_first_chunk_are_http_errors(self):
        for exception, code in ((ValueError, 400), (RuntimeError, 500)):
            with self.subTest(exception=exception):
                self.closed = False
                self._source(exception("early failure"), before_first=True)
                response = await self._response()
                self.assertEqual(response.status_code, code)
                self.assertEqual(json.loads(response.body)["code"], code)
                self.assertTrue(self.closed)

    async def test_explicit_abort_keeps_status_and_closes_source(self):
        self._source(
            finish_reason={
                "type": "abort",
                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                "message": "worker unavailable",
            }
        )
        self._assert_error(await self._chunks(), 503)

    async def test_cancellation_propagates_without_error_or_done(self):
        self._source(asyncio.CancelledError())
        response = await self._response()
        chunks = []
        with self.assertRaises(asyncio.CancelledError):
            async for chunk in response.body_iterator:
                chunks.append(chunk)
        self.assertEqual(len(chunks), 1)  # role only
        self.assertTrue(self.closed)

    async def test_iterator_close_closes_source(self):
        self._source()
        response = await self._response()
        await response.body_iterator.__anext__()  # role chunk
        await response.body_iterator.aclose()
        self.assertTrue(self.closed)

    async def test_client_disconnect_while_waiting_closes_source(self):
        waiting = asyncio.Event()

        async def blocked_content(**kwargs):
            waiting.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover

        self.chat._generate_stream_content = blocked_content
        self._source()
        response = await self._response()
        sent = []

        async def receive():
            await waiting.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await response({"type": "http", "asgi": {"spec_version": "2.0"}}, receive, send)
        self.assertTrue(self.closed)
        self.assertFalse(any(b"[DONE]" in msg.get("body", b"") for msg in sent))
        response.background.assert_awaited_once()

    async def test_success_still_finishes_normally(self):
        self._source(finish_reason={"type": "stop"})
        chunks = await self._chunks()
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(
            json.loads(chunks[-2][6:])["choices"][0]["finish_reason"], "stop"
        )

    async def test_send_failure_closes_prefetched_source(self):
        for failed_message in ("http.response.start", "http.response.body"):
            with self.subTest(failed_message=failed_message):
                self.closed = False
                self._source()
                response = await self._response()

                async def receive():
                    await asyncio.Event().wait()

                async def send(message):
                    if message["type"] == failed_message:
                        raise OSError("client disconnected")

                # ASGI 2.4 reports a disconnect through send(), not receive().
                # The exception remains a transport failure, not an SSE error.
                with self.assertRaises(Exception):
                    await response(
                        {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
                    )
                self.assertTrue(self.closed)


class TestTokenizerStreamAbort(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                enable_cache_report=False,
                stream_response_default_include_usage=False,
            ),
            role="tokenizer",
        )
        self.tm = TokenizerManager.__new__(TokenizerManager)
        self.tm.server_args = SimpleNamespace(
            language_only=False, tokenizer_worker_num=1
        )
        self.tm.auto_create_handle_loop = Mock()
        self.tm._set_default_priority = Mock()
        self.tm.request_logger = Mock()
        self.tm.tokenizer = None
        self.tm.is_pause = False
        self.tm.is_pause_cond = asyncio.Condition()
        self.tm.model_update_lock = SimpleNamespace(reader_lock=asyncio.Lock())
        self.tm._validate_and_resolve_lora = AsyncMock()
        self.tm._tokenize_one_request = AsyncMock()
        self.tm._send_one_request = AsyncMock(side_effect=self._mark_dispatched)
        self.tm.rid_to_state = {}
        self.tm.encoder_dispatch_ready = {}
        self.tm.enable_metrics = False
        self.tm.enable_lora = False
        self.tm._dispatch_to_scheduler = Mock()
        self.tm._init_req_state = self._init_state

    def _init_state(self, obj, request):
        for rid in [obj.rid] if obj.is_single else obj.rid:
            self.tm.rid_to_state[rid] = SimpleNamespace(
                time_stats=SimpleNamespace(api_server_dispatch_time=0.0),
                dispatched=False,
                abort_sent=False,
                lora_released=False,
                obj=SimpleNamespace(lora_path=None),
            )

    def _mark_dispatched(self, *args):
        for state in self.tm.rid_to_state.values():
            state.time_stats.api_server_dispatch_time = 1.0
            state.dispatched = True

    def _obj(self, rids):
        obj = Mock()
        obj.rid = rids
        obj.is_single = isinstance(rids, str)
        obj.return_prompt_token_ids = False
        return obj

    def _assert_aborted(self, rids):
        self.assertEqual(
            [
                call.args[0].rid
                for call in self.tm._dispatch_to_scheduler.call_args_list
            ],
            rids,
        )
        # Current ownership keeps dispatched states until scheduler acknowledgement.
        self.assertEqual(set(self.tm.rid_to_state), set(rids))
        self.assertFalse(self.tm.model_update_lock.reader_lock.locked())

    async def test_close_aborts_before_removing_state(self):
        async def source(*args):
            yield {}

        self.tm._wait_one_response = source
        generator = self.tm.generate_request(self._obj("pending"))
        await generator.__anext__()
        await generator.aclose()
        self._assert_aborted(["pending"])

    async def test_cancel_aborts_before_removing_state(self):
        async def source(*args):
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        self.tm._wait_one_response = source
        with self.assertRaises(asyncio.CancelledError):
            await self.tm.generate_request(self._obj("cancelled")).__anext__()
        self._assert_aborted(["cancelled"])

    async def test_partial_batch_failure_aborts_only_pending_choices(self):
        async def source(*args):
            self._mark_dispatched()
            self.tm.rid_to_state.pop("finished")
            self.tm.rid_to_state[
                "undispatched"
            ].time_stats.api_server_dispatch_time = 0.0
            self.tm.rid_to_state["undispatched"].dispatched = False
            raise RuntimeError("batch failed")
            yield  # pragma: no cover

        self.tm._handle_batch_request = source
        with self.assertRaisesRegex(RuntimeError, "batch failed"):
            await self.tm.generate_request(
                self._obj(["pending-0", "finished", "pending-1", "undispatched"])
            ).__anext__()
        self._assert_aborted(["pending-0", "pending-1"])

    async def test_validation_failure_does_not_abort_undispatched_request(self):
        self.tm._tokenize_one_request.side_effect = ValueError("input too long")
        with self.assertRaisesRegex(ValueError, "input too long"):
            await self.tm.generate_request(self._obj("invalid")).__anext__()
        self._assert_aborted([])

    async def test_abort_send_failure_preserves_original_error_and_cleanup(self):
        async def source(*args):
            raise ValueError("original error")
            yield  # pragma: no cover

        self.tm._wait_one_response = source
        self.tm._dispatch_to_scheduler.side_effect = RuntimeError("channel unavailable")
        with self.assertRaisesRegex(ValueError, "original error"):
            await self.tm.generate_request(self._obj("failed")).__anext__()
        self._assert_aborted(["failed"])

    async def test_completed_request_is_not_aborted_when_closed(self):
        async def source(*args):
            self.tm.rid_to_state.pop("finished")
            yield {}

        self.tm._wait_one_response = source
        generator = self.tm.generate_request(self._obj("finished"))
        await generator.__anext__()
        await generator.aclose()
        self._assert_aborted([])


if __name__ == "__main__":
    unittest.main()
