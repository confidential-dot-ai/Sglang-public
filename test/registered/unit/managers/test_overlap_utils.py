"""Unit tests for srt/managers/overlap_utils.py's FutureMap."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import (
    FutureMap,
    resolve_forward_inputs,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _make_future_map(num_producers: int, req_pool_size: int = 8) -> FutureMap:
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.empty((req_pool_size, 1), dtype=torch.int64)
    )
    return FutureMap(
        device=torch.device("cuda"),
        spec_algo=SpeculativeAlgorithm.EAGLE,
        req_to_token_pool=req_to_token_pool,
        needs_cpu_seq_lens=True,
        needs_confidence_relay=False,
        num_producers=num_producers,
    )


class _FakeEvent:
    def __init__(self):
        self.complete = False
        self.record_count = 0

    def record(self):
        self.record_count += 1

    def query(self):
        return self.complete


class TestPinnedPrefillInputLease(unittest.TestCase):
    def test_resolve_consumes_deferred_staging_lease(self):
        expected = torch.tensor([11, 12], dtype=torch.int64)

        class _Lease:
            def __init__(self):
                self.devices = []

            def copy_to_device(self, *, device):
                self.devices.append(device)
                return expected

        class _FutureMap:
            def __init__(self):
                self.waited = None

            def _wait_payload_rows(self, rows):
                self.waited = rows

        lease = _Lease()
        rows = torch.tensor([3, 4], dtype=torch.int64)
        batch = SimpleNamespace(
            req_pool_indices_cpu=rows,
            prefill_input_ids_cpu=torch.tensor([11, 12], dtype=torch.int64),
            prefill_input_ids_staging_lease=lease,
            mix_running_indices=None,
            device=torch.device("cuda"),
            input_ids=None,
            enable_overlap=False,
        )
        future_map = _FutureMap()

        resolve_forward_inputs(batch, future_map)

        self.assertIs(batch.input_ids, expected)
        self.assertEqual(lease.devices, [torch.device("cuda")])
        self.assertIs(future_map.waited, rows)
        self.assertIsNone(batch.prefill_input_ids_cpu)
        self.assertIsNone(batch.prefill_input_ids_staging_lease)


class TestFutureMapMultiProducer(CustomTestCase):
    """With more than one producer stream, FutureMap.publish_ready must keep
    one event per producer: a single shared event let a resolve call on one
    producer's tick observe the event armed by the *other* producer's earlier
    publish and proceed without waiting for its own tick's publish."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for FutureMap's Event-based path")

    def test_single_producer_default_matches_pre_fix_shape(self):
        # num_producers defaults to 1: publish_ready must still behave as a
        # single-slot list, not silently grow, and resolve/publish without a
        # producer kwarg must keep touching slot 0.
        fm = _make_future_map(num_producers=1)
        self.assertEqual(len(fm.publish_ready), 1)
        self.assertIsNone(fm.publish_ready[0])

        future_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        new_seq_lens = torch.tensor([5, 7], dtype=torch.int64, device="cuda")
        fm.publish(future_indices, new_seq_lens)
        self.assertIsNotNone(fm.publish_ready[0])

    def test_producers_own_independent_event_slots(self):
        # The confirmed hazard: publishing from one producer must not touch
        # (overwrite, or be mistaken for) another producer's event slot.
        fm = _make_future_map(num_producers=2)
        self.assertEqual(len(fm.publish_ready), 2)

        future_indices = torch.tensor([0], dtype=torch.int64, device="cuda")
        new_seq_lens = torch.tensor([3], dtype=torch.int64, device="cuda")

        future_indices_cpu = future_indices.cpu()
        fm.publish(
            future_indices,
            new_seq_lens,
            producer=0,
            future_indices_cpu=future_indices_cpu,
        )
        self.assertIsNotNone(fm.publish_ready[0])
        self.assertIsNone(fm.publish_ready[1])
        producer0_event = fm.publish_ready[0]

        fm.publish(
            future_indices,
            new_seq_lens,
            producer=1,
            future_indices_cpu=future_indices_cpu,
        )
        self.assertIsNotNone(fm.publish_ready[1])
        # Producer 0's own event object is untouched by producer 1's publish.
        self.assertIs(fm.publish_ready[0], producer0_event)

    def test_resolve_does_not_fail_when_only_one_producer_has_published(self):
        # The exact interleaving that broke the old single-Event design: a
        # resolve call reached before *this* producer's own publish, with
        # only the *other* producer having published so far. The old code's
        # "if self.publish_ready is not None" check couldn't distinguish
        # which producer set it; the fix waits on every producer slot that
        # exists and skips ones that have never published, rather than
        # assuming a single shared slot always means "this tick is ready."
        fm = _make_future_map(num_producers=2)
        future_indices = torch.tensor([0], dtype=torch.int64, device="cuda")
        new_seq_lens = torch.tensor([9], dtype=torch.int64, device="cuda")

        # Only producer 1 (the concurrently-dispatched decode slot) has
        # published; producer 0 never has for this FutureMap instance.
        fm.publish(
            future_indices,
            new_seq_lens,
            producer=1,
            future_indices_cpu=future_indices.cpu(),
        )
        self.assertIsNone(fm.publish_ready[0])
        self.assertIsNotNone(fm.publish_ready[1])

        draft_input = SimpleNamespace(future_indices=future_indices)
        batch = SimpleNamespace(
            spec_info=draft_input,
            seq_lens=None,
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
        )

        # Must not raise (e.g. AttributeError from treating publish_ready as
        # a scalar) and must not hang -- both producer slots are consulted,
        # None ones skipped.
        fm.resolve_seq_lens_cpu(batch)
        self.assertEqual(batch.seq_lens.item(), 9)


if __name__ == "__main__":
    unittest.main()


class TestDeferredSeqLensResolve(CustomTestCase):
    """The DSPARK deferral contract: with defer_spec_cpu_seq_lens armed,
    resolve_seq_lens_cpu must stash the blocking exact-mirror pull on the
    batch and poison seq_lens_cpu/sum to None (an unaudited reader in the
    deferral window must fail loudly, not silently consume the stale
    committed seed); consume_deferred_seq_lens_cpu then materializes the
    exact values once, idempotently. With the flag off, behavior is the
    eager pre-change resolve."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for FutureMap's Event-based path")

    def _resolve_batch(self, fm):

        future_indices = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        new_seq_lens = torch.tensor([11, 23], dtype=torch.int64, device="cuda")
        fm.publish(
            future_indices,
            new_seq_lens,
            future_indices_cpu=future_indices.cpu(),
        )
        spec_info = SimpleNamespace(future_indices=future_indices)
        batch = SimpleNamespace(
            spec_info=spec_info,
            req_pool_indices_cpu=future_indices.cpu(),
            seq_lens=None,
            seq_lens_cpu=torch.tensor([9, 20], dtype=torch.int64),  # stale seed
            seq_lens_sum=29,
            deferred_resolve_seq_lens_cpu=None,
        )
        fm.resolve_seq_lens_cpu(batch)
        return batch

    def test_deferred_resolve_poisons_then_materializes_exact(self):
        from sglang.srt.managers.overlap_utils import consume_deferred_seq_lens_cpu

        fm = _make_future_map(num_producers=1)
        fm.spec_algo = SpeculativeAlgorithm.DSPARK
        fm.defer_spec_cpu_seq_lens = True
        batch = self._resolve_batch(fm)

        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)
        self.assertIsNotNone(batch.deferred_resolve_seq_lens_cpu)
        # Device gather still happened eagerly.
        self.assertEqual(batch.seq_lens.tolist(), [11, 23])

        consume_deferred_seq_lens_cpu(batch)
        self.assertIsNone(batch.deferred_resolve_seq_lens_cpu)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 23])
        self.assertEqual(batch.seq_lens_sum, 34)
        # Idempotent: a second consume is a no-op, not a double resolve.
        consume_deferred_seq_lens_cpu(batch)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 23])

    def test_eager_resolve_unchanged_when_defer_disarmed(self):
        fm = _make_future_map(num_producers=1)
        fm.spec_algo = SpeculativeAlgorithm.DSPARK
        batch = self._resolve_batch(fm)

        self.assertIsNone(batch.deferred_resolve_seq_lens_cpu)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [11, 23])
        self.assertEqual(batch.seq_lens_sum, 34)
