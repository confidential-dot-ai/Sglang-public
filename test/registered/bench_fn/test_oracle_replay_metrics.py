"""CPU regression tests for the oracle replay throughput contract."""

import threading
import time

from benchmark.oracle_replay.concurrent_replay import (
    AdmissionController,
    _select_request_url,
)
from benchmark.oracle_replay.metrics import (
    MixBands,
    ScorecardError,
    TokenCounters,
    assert_scorecard,
    summarize_window,
    validate_physical_gpu_count,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestOracleReplayMetrics(CustomTestCase):
    def test_direct_worker_routing_is_conversation_sticky_and_balanced(self):
        workers = [
            "http://127.0.0.1:30010/v1/chat/completions",
            "http://127.0.0.1:30011/v1/chat/completions",
        ]

        self.assertEqual(_select_request_url(workers, 0), workers[0])
        self.assertEqual(_select_request_url(workers, 1), workers[1])
        self.assertEqual(_select_request_url(workers, 2), workers[0])
        self.assertEqual(_select_request_url(workers, 101), workers[1])

    def test_admission_controller_is_fifo(self):
        admission = AdmissionController(capacity=1)
        stop = threading.Event()
        self.assertIsNotNone(admission.acquire(stop))
        observed = []

        def use_slot(name):
            wait_s = admission.acquire(stop)
            if wait_s is not None:
                observed.append(name)
                admission.release()

        first = threading.Thread(target=use_slot, args=("first",))
        second = threading.Thread(target=use_slot, args=("second",))
        first.start()
        for _ in range(100):
            with admission.condition:
                if admission.next_ticket >= 2:
                    break
            time.sleep(0.001)
        second.start()
        for _ in range(100):
            with admission.condition:
                if admission.next_ticket >= 3:
                    break
            time.sleep(0.001)

        admission.release()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(observed, ["first", "second"])
        self.assertEqual(admission.active, 0)

    def test_two_gpu_tpm_uses_physical_gpu_denominator(self):
        result = summarize_window(
            TokenCounters(),
            TokenCounters(prompt_tokens=5_955, completion_tokens=45),
            elapsed_s=60,
            physical_gpus=2,
        )

        self.assertEqual(result["served_tokens"], 6_000)
        self.assertEqual(result["served_tpm_total"], 6_000)
        self.assertEqual(result["served_tpm_per_gpu"], 3_000)
        self.assertNotEqual(
            result["served_tpm_per_gpu"],
            1_500,
            "the old replay driver silently divided every result by four",
        )

    def test_two_visible_gpus_with_four_gpu_denominator_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES exposes 2"):
            validate_physical_gpu_count(4, cuda_visible_devices="6,7", multi_node=False)

    def test_fixed_window_uses_counter_deltas(self):
        warmup_end = TokenCounters(
            completed_requests=20,
            prompt_tokens=100_000,
            completion_tokens=1_000,
            cached_prompt_tokens=95_000,
        )
        measurement_end = TokenCounters(
            completed_requests=30,
            prompt_tokens=179_400,
            completion_tokens=1_600,
            cached_prompt_tokens=170_430,
        )

        result = summarize_window(
            warmup_end, measurement_end, elapsed_s=60, physical_gpus=2
        )

        self.assertEqual(result["completed_requests"], 10)
        self.assertEqual(result["prompt_tokens"], 79_400)
        self.assertEqual(result["completion_tokens"], 600)
        self.assertEqual(result["served_tpm_per_gpu"], 40_000)
        self.assertAlmostEqual(result["cache_hit_rate"], 0.95, places=6)
        self.assertAlmostEqual(result["output_share"], 0.0075, places=6)
        self.assertAlmostEqual(
            result["cached_served_share"]
            + result["uncached_input_share"]
            + result["output_share"],
            1.0,
        )

    def test_scorecard_accepts_exact_deck_mix(self):
        result = summarize_window(
            TokenCounters(),
            TokenCounters(
                completed_requests=10,
                prompt_tokens=79_400,
                completion_tokens=600,
                cached_prompt_tokens=75_430,
            ),
            elapsed_s=60,
            physical_gpus=2,
        )
        assert_scorecard(result)

    def test_scorecard_rejects_mix_and_failures(self):
        result = summarize_window(
            TokenCounters(),
            TokenCounters(
                completed_requests=10,
                failed_requests=1,
                prompt_tokens=59_550,
                completion_tokens=1_000,
                cached_prompt_tokens=58_000,
            ),
            elapsed_s=60,
            physical_gpus=2,
        )

        with self.assertRaisesRegex(
            ScorecardError, "failed_requests=1.*cache_hit_rate=.*output_share="
        ):
            assert_scorecard(result, MixBands())

    def test_rejects_non_monotonic_or_impossible_counters(self):
        with self.assertRaisesRegex(ValueError, "negative deltas"):
            summarize_window(
                TokenCounters(prompt_tokens=10),
                TokenCounters(prompt_tokens=9),
                elapsed_s=1,
                physical_gpus=1,
            )
        with self.assertRaisesRegex(
            ValueError, "cached_prompt_tokens cannot exceed prompt_tokens"
        ):
            summarize_window(
                TokenCounters(),
                TokenCounters(prompt_tokens=9, cached_prompt_tokens=10),
                elapsed_s=1,
                physical_gpus=1,
            )
