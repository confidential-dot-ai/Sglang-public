"""Unit tests for the compact-draft-cache CCD fix's padded gather/assign
primitives (DFlashWorkerV2._gather_req_to_token_segments_padded,
assign_req_to_token_pool_padded_func) -- see tdx-checkpoints/120.

Background: the ragged version of this gather/assign pair
(_gather_req_to_token_segments + assign_req_to_token_pool_func) uses a
boolean-mask filter (`tensor[bool_mask]`) to pack variable-length
per-request segments into one tightly-packed flat array, which
assign_req_to_token_pool's Triton kernel then unpacks via an on-device
cumulative-offset sum. `tensor[bool_mask]` is a data-dependent-shape
operation regardless of the mask's actual values, so it requires a host
sync every call -- forbidden inside CCD's captured region. Confirmed live:
a first fix attempt addressed two other host syncs in this same code path
but not this one, and crashed a real decode engine mid-capture with "CUDA
error: operation not permitted when stream is capturing".

The padded version sidesteps this by never packing into a ragged array at
all: the gather returns a fixed (bs, max_len) layout (no boolean filter),
and a new Triton kernel (assign_req_to_token_pool_padded) reads each
request's slice at a FIXED stride (pid * max_len) instead of a
data-dependent cumulative offset, re-deriving per-request validity from
the exact `lengths` tensor on-device (never a host/Python value) at both
steps. These tests verify: (1) the padded gather's CPU-testable tensor
logic directly, and (2) end-to-end, that gather+assign together produce
the identical final req_to_token state as a plain-Python reference
implementation of the same semantics -- gated on CUDA, since the assign
kernel is Triton/GPU-only and cannot run in CPU-only CI. Run explicitly
on GPU hardware (also registered in CUDA CI) to validate the kernel
itself; the CPU-only tests below cover the gather in isolation.
"""

import torch

from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_padded_func
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _make_worker(device="cpu"):
    # _gather_req_to_token_segments_padded reads only self.device; a bare,
    # un-__init__-ed instance is a valid, cheap stand-in (same pattern as
    # test_draft_oracle_logprob.py).
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.device = torch.device(device)
    return worker


def _reference_assign(
    req_pool_indices, req_to_token, start_offset, length, out_cache_loc, stride
):
    """Plain-Python reference for what assign_req_to_token_pool_padded's
    Triton kernel does: for each request, copy its first `length[i]`
    (never `stride`) values from out_cache_loc's padded row into
    req_to_token[req_pool_indices[i], start_offset[i] : start_offset[i] + length[i]].
    """
    req_to_token = req_to_token.clone()
    for i in range(req_pool_indices.shape[0]):
        row = int(req_pool_indices[i])
        s = int(start_offset[i])
        n = int(length[i])
        padded_row = out_cache_loc[i * stride : i * stride + n]
        req_to_token[row, s : s + n] = padded_row
    return req_to_token


class TestGatherReqToTokenSegmentsPadded(CustomTestCase):
    def test_returns_padded_flat_shape(self):
        worker = _make_worker()
        req_to_token = torch.arange(40, dtype=torch.int64).reshape(2, 20)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        lengths = torch.tensor([3, 5], dtype=torch.int64)
        start = torch.tensor([2, 10], dtype=torch.int64)

        out = worker._gather_req_to_token_segments_padded(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            start=start,
            lengths=lengths,
            max_len=8,
        )
        self.assertEqual(out.shape, (16,))  # bs=2 * max_len=8, always, regardless of lengths

    def test_valid_positions_match_direct_indexing(self):
        worker = _make_worker()
        req_to_token = torch.arange(40, dtype=torch.int64).reshape(2, 20)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        lengths = torch.tensor([3, 5], dtype=torch.int64)
        start = torch.tensor([2, 10], dtype=torch.int64)
        max_len = 8

        out = worker._gather_req_to_token_segments_padded(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            start=start,
            lengths=lengths,
            max_len=max_len,
        ).reshape(2, max_len)

        self.assertTrue(torch.equal(out[0, :3], req_to_token[0, 2:5]))
        self.assertTrue(torch.equal(out[1, :5], req_to_token[1, 10:15]))

    def test_empty_batch_returns_empty(self):
        worker = _make_worker()
        req_to_token = torch.arange(40, dtype=torch.int64).reshape(2, 20)
        out = worker._gather_req_to_token_segments_padded(
            req_to_token=req_to_token,
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            start=torch.empty(0, dtype=torch.int64),
            lengths=torch.empty(0, dtype=torch.int64),
            max_len=8,
        )
        self.assertEqual(out.numel(), 0)

    def test_start_none_uses_bare_offsets(self):
        worker = _make_worker()
        req_to_token = torch.arange(40, dtype=torch.int64).reshape(2, 20)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        lengths = torch.tensor([4, 4], dtype=torch.int64)
        max_len = 6

        out = worker._gather_req_to_token_segments_padded(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            start=None,
            lengths=lengths,
            max_len=max_len,
        ).reshape(2, max_len)
        self.assertTrue(torch.equal(out[0, :4], req_to_token[0, :4]))
        self.assertTrue(torch.equal(out[1, :4], req_to_token[1, :4]))

    def test_never_reads_out_of_bounds_for_short_requests(self):
        """A request near the end of req_to_token's width with a small
        real length: padding columns must redirect to an in-bounds
        position (0), not run past the table's edge."""
        worker = _make_worker()
        req_to_token = torch.arange(20, dtype=torch.int64).reshape(1, 20)
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        lengths = torch.tensor([2], dtype=torch.int64)
        start = torch.tensor([18], dtype=torch.int64)  # only 2 valid columns left
        max_len = 8  # padded width far exceeds what's left in the row

        out = worker._gather_req_to_token_segments_padded(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            start=start,
            lengths=lengths,
            max_len=max_len,
        )
        self.assertEqual(out.shape, (8,))
        self.assertTrue(torch.equal(out[:2], req_to_token[0, 18:20]))

    def test_boundary_lengths_wide_window(self):
        """max_len > 32 with lengths at the block boundaries the Triton
        assign kernel cares about (0, 32, 33, max_len): the gather must
        produce exact valid prefixes at every one of them."""
        worker = _make_worker()
        max_len = 96
        req_to_token = (
            torch.arange(4 * 200, dtype=torch.int64).reshape(4, 200) + 1
        )
        req_pool_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        lengths = torch.tensor([0, 32, 33, 96], dtype=torch.int64)
        start = torch.tensor([5, 0, 100, 50], dtype=torch.int64)

        out = worker._gather_req_to_token_segments_padded(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            start=start,
            lengths=lengths,
            max_len=max_len,
        ).reshape(4, max_len)

        for i, (s, n) in enumerate([(5, 0), (0, 32), (100, 33), (50, 96)]):
            self.assertTrue(
                torch.equal(out[i, :n], req_to_token[i, s : s + n]),
                f"row {i} valid prefix mismatch",
            )


class TestPaddedAssignReferenceSemantics(CustomTestCase):
    """CPU-only: verifies the reference implementation's own semantics are
    self-consistent (a sanity check on the test helper, not the kernel)."""

    def test_reference_writes_only_valid_length_per_request(self):
        req_to_token = torch.zeros(2, 10, dtype=torch.int64)
        req_pool_indices = torch.tensor([0, 1])
        start_offset = torch.tensor([0, 0])
        length = torch.tensor([3, 5])
        stride = 6
        out_cache_loc = torch.arange(2 * stride, dtype=torch.int64) + 100

        result = _reference_assign(
            req_pool_indices, req_to_token, start_offset, length, out_cache_loc, stride
        )
        self.assertTrue(torch.equal(result[0, :3], torch.tensor([100, 101, 102])))
        self.assertTrue(torch.equal(result[0, 3:], torch.zeros(7, dtype=torch.int64)))
        self.assertTrue(
            torch.equal(result[1, :5], torch.tensor([106, 107, 108, 109, 110]))
        )
        self.assertTrue(torch.equal(result[1, 5:], torch.zeros(5, dtype=torch.int64)))


class TestPaddedAssignKernelMatchesReference(CustomTestCase):
    """GPU-only: exercises the real Triton kernel. Not part of CPU CI
    (Triton requires CUDA) -- skips cleanly when no GPU is available; run
    explicitly on GPU hardware to validate the kernel before trusting it
    live (see tdx-checkpoints/120)."""

    def test_kernel_matches_python_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA (Triton kernel)")
        device = "cuda"
        pool_len = 32
        req_to_token = torch.zeros(3, pool_len, dtype=torch.int64, device=device)
        req_pool_indices = torch.tensor([2, 0, 1], dtype=torch.int64, device=device)
        start_offset = torch.tensor([5, 0, 10], dtype=torch.int64, device=device)
        length = torch.tensor([4, 7, 2], dtype=torch.int64, device=device)
        stride = 8
        bs = 3
        out_cache_loc = (
            torch.arange(bs * stride, dtype=torch.int64, device=device) + 1000
        )

        expected = _reference_assign(
            req_pool_indices.cpu(),
            req_to_token.cpu(),
            start_offset.cpu(),
            length.cpu(),
            out_cache_loc.cpu(),
            stride,
        )

        assign_req_to_token_pool_padded_func(
            req_pool_indices,
            req_to_token,
            start_offset,
            length,
            out_cache_loc,
            bs,
            stride,
        )
        self.assertTrue(torch.equal(req_to_token.cpu(), expected))

    def test_kernel_composes_with_padded_gather_round_trip(self):
        """End-to-end: gather a compact window from one table, assign it
        into another (mirroring the real suffix-copy call site), and
        confirm the destination matches the source segment exactly."""
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA (Triton kernel)")
        device = "cuda"
        worker = _make_worker(device=device)

        src_table = torch.arange(2 * 40, dtype=torch.int64, device=device).reshape(
            2, 40
        )
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
        lengths = torch.tensor([5, 8], dtype=torch.int64, device=device)
        start = torch.tensor([3, 12], dtype=torch.int64, device=device)
        max_len = 10

        gathered = worker._gather_req_to_token_segments_padded(
            req_to_token=src_table,
            req_pool_indices=req_pool_indices,
            start=start,
            lengths=lengths,
            max_len=max_len,
        )

        dst_table = torch.zeros(2, 40, dtype=torch.int64, device=device)
        assign_req_to_token_pool_padded_func(
            req_pool_indices,
            dst_table,
            torch.zeros_like(lengths),
            lengths,
            gathered,
            2,
            max_len,
        )

        self.assertTrue(torch.equal(dst_table[0, :5], src_table[0, 3:8]))
        self.assertTrue(torch.equal(dst_table[1, :8], src_table[1, 12:20]))
        self.assertTrue(torch.equal(dst_table[0, 5:], torch.zeros(35, dtype=torch.int64, device=device)))

    def test_kernel_multi_iteration_stride_and_boundary_lengths(self):
        """stride > BLOCK_SIZE (32): the kernel's inner loop must advance
        save_offset/load_offset across multiple iterations. Production
        stride is draft_window_size + page_size (typically 256+), so the
        multi-iteration path is the ONLY path production uses -- the other
        tests' single-iteration strides never execute the loop carry.
        Lengths chosen to straddle every boundary: 0 (empty row), 17
        (mid-first-block), 32 (exact block edge), 33 (one past the edge),
        96 (== stride, zero padding columns)."""
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA (Triton kernel)")
        device = "cuda"
        stride = 96  # 3 kernel loop iterations at BLOCK_SIZE=32
        bs = 5
        pool_len = 128
        req_to_token = torch.full(
            (bs, pool_len), -1, dtype=torch.int64, device=device
        )
        req_pool_indices = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64, device=device)
        start_offset = torch.tensor([0, 7, 20, 1, 32], dtype=torch.int64, device=device)
        length = torch.tensor([0, 17, 32, 33, 96], dtype=torch.int64, device=device)
        out_cache_loc = (
            torch.arange(bs * stride, dtype=torch.int64, device=device) + 5000
        )

        expected = _reference_assign(
            req_pool_indices.cpu(),
            req_to_token.cpu(),
            start_offset.cpu(),
            length.cpu(),
            out_cache_loc.cpu(),
            stride,
        )

        assign_req_to_token_pool_padded_func(
            req_pool_indices,
            req_to_token,
            start_offset,
            length,
            out_cache_loc,
            bs,
            stride,
        )
        self.assertTrue(torch.equal(req_to_token.cpu(), expected))


class TestCompactDraftSeqLensHostDeviceAgreement(CustomTestCase):
    """CCD's compact branch must publish EXACT per-request compact lengths in
    seq_lens_cpu, because flashinfer's fast_prefill_plan consumes it as exact
    geometry (kv_indptr_host/kv_lens_host) while the device side builds
    kv_indptr/kv_indices from the real lengths. The original code filled it
    with the static bound compact_max_len instead, so the two described
    different layouts and the kernel dereferenced never-written kv_indices
    entries as KV token slots -> illegal memory access on the first
    CCD-eligible batch (tdx-checkpoints/132, root cause in 134).
    """

    def _worker(self, *, draft_window_size, page_size, device="cpu"):
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker.device = torch.device(device)
        worker.draft_window_size = draft_window_size
        worker.page_size = page_size
        return worker

    def test_host_matches_device_formula_across_alignments(self):
        # page_size=64/window=16 is this deployment's config; the alignment
        # slack (visible_start % page_size) is what makes the real lengths
        # vary per request -- exactly what a static fill_() erases.
        worker = self._worker(draft_window_size=16, page_size=64)
        seq_lens = torch.tensor(
            [16, 17, 63, 64, 65, 100, 127, 128, 129, 1000, 4096, 4097],
            dtype=torch.int32,
        )
        device_side = worker._compute_compact_draft_seq_lens(seq_lens)
        host_side = worker._compute_compact_draft_seq_lens_cpu(seq_lens)
        self.assertTrue(torch.equal(device_side.cpu(), host_side))

    def test_lengths_are_not_uniform_so_a_static_fill_would_lie(self):
        # Guards the premise of the fix: if these were all equal, filling
        # with a constant would have been harmless and this test is the
        # canary that the formula changed.
        worker = self._worker(draft_window_size=16, page_size=64)
        seq_lens = torch.tensor([64, 70, 100, 127], dtype=torch.int32)
        lens = worker._compute_compact_draft_seq_lens_cpu(seq_lens)
        self.assertGreater(len(set(lens.tolist())), 1)

    def test_never_exceeds_the_padded_stride(self):
        # compact_max_len (the gather/assign stride) must remain a valid
        # STATIC upper bound: real length <= window + page_size - 1 < stride.
        for window, page in ((16, 64), (8, 32), (16, 128), (64, 64)):
            worker = self._worker(draft_window_size=window, page_size=page)
            compact_max_len = window + (page if page > 1 else 0)
            seq_lens = torch.arange(1, 4096, 7, dtype=torch.int32)
            lens = worker._compute_compact_draft_seq_lens_cpu(seq_lens)
            self.assertLessEqual(
                int(lens.max()),
                compact_max_len,
                f"window={window} page={page}: length exceeds padded stride",
            )
            self.assertGreaterEqual(int(lens.min()), 1)

    def test_page_size_one_is_plain_clamp(self):
        worker = self._worker(draft_window_size=16, page_size=1)
        seq_lens = torch.tensor([1, 5, 16, 17, 500], dtype=torch.int32)
        lens = worker._compute_compact_draft_seq_lens_cpu(seq_lens)
        self.assertEqual(lens.tolist(), [1, 5, 16, 16, 16])


if __name__ == "__main__":
    import unittest

    unittest.main()
