"""The abort status a scheduler picks must survive to the HTTP response.

`TokenizerManager._handle_abort_finish_reason` raises an HTTPException only
for statuses it explicitly lists. A status the scheduler emits but the list
omits does not degrade to a different code -- it is dropped entirely and the
client receives a silent, empty, 200 completion. That silent path is what
these cases guard: load shedding must reach the caller as 429, which the
provider API contract requires so an overloaded server tells clients to back off.

No server, no model loading -- pure CPU.
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _abort_output(status_code):
    return {
        "meta_info": {
            "finish_reason": {
                "type": "abort",
                "status_code": status_code,
                "message": "The request queue is full.",
            }
        }
    }


class TestAbortStatusPropagation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = SimpleNamespace(obj=SimpleNamespace(rid="r0", lora_path=None))
        self.manager = SimpleNamespace(
            rid_to_state={}, enable_lora=False, lora_registry=None
        )

    async def _handle(self, status_code):
        return await TokenizerManager._handle_abort_finish_reason(
            self.manager, _abort_output(status_code), self.state, is_stream=False
        )

    async def test_load_shedding_reaches_the_client_as_429(self):
        with self.assertRaises(Exception) as caught:
            await self._handle(HTTPStatus.TOO_MANY_REQUESTS)
        self.assertEqual(
            caught.exception.status_code, int(HTTPStatus.TOO_MANY_REQUESTS)
        )

    async def test_service_unavailable_still_propagates(self):
        with self.assertRaises(Exception) as caught:
            await self._handle(HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(
            caught.exception.status_code, int(HTTPStatus.SERVICE_UNAVAILABLE)
        )

    async def test_graceful_abort_without_a_status_is_not_an_error(self):
        # A user-initiated abort carries no status_code and must fall through
        # to the normal flow rather than becoming an HTTP error.
        out = {"meta_info": {"finish_reason": {"type": "abort"}}}
        result = await TokenizerManager._handle_abort_finish_reason(
            self.manager, out, self.state, is_stream=False
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
