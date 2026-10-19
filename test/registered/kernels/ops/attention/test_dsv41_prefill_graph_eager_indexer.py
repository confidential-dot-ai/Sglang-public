"""The eager index layers of the DeepSeek-V4.1 breakable prefill graph: contexts past
the 16K window the captured indexer scores, against the captured full top-k."""

import copy
import unittest
from functools import partial
from types import SimpleNamespace
from typing import NamedTuple, Optional
from unittest.mock import patch

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    gather_fp4_index_k,
    quantize_fp4_indexer_tensor,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    expand_index_page_table,
)
from sglang.srt.layers.attention.dsv4.v41_indexer import (
    CapturedPrefillInputs,
    PrefillInputs,
    make_candidate_indexer,
    make_full_topk_indexer,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.memory_budget import (
    INDEX_LOGITS_BUDGET_CAP_BYTES,
    eager_indexer_peak_bytes,
    set_planned_index_logits_budget,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
    DeepGEMMPrefillData,
    get_deep_gemm_prefill_data,
    get_prefill_queries,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS, DIM, ROPE, TOPK = 32, 128, 64, 512
KV_PAGE = 256  # the KV page size of the ratio-1 layers
PAGE = 128  # the index-K pool page size under DeepGEMM's paged sparse logits
TOPK_BLOCKS, BLOCK = 2048, 8  # the 16384-position candidate window
GRAPH_BUCKET = 128
# The source's top-k is a deterministic dense pass. A consumer scores sparsely with
# bf16 weights and its tie-breaks at the selection floor move between identical
# runs, so its picks match to a few ulps of the floor, not bit for bit.
MIN_SOURCE_OVERLAP = 0.99
MIN_CONSUMER_OVERLAP = 0.95
# Past 64K the floor's ties move more: a consumer's own eager reruns matched as low
# as 0.955 per row at 256K (mean 0.988), so long contexts bound the mean instead.
MIN_LONG_CONSUMER_ROW_OVERLAP = 0.9


class Selection(NamedTuple):
    """An indexer's output buffers: index-K pool slots and the same picks as
    request-local compressed positions, [rows, TOPK] int32, -1 padded."""

    page_indices: Optional[torch.Tensor]
    raw_indices: torch.Tensor


class IndexPool:
    """The ratio-1 index-K pool in the fp4 page layout
    [pages, PAGE * 64 payload | PAGE * 4 scale]."""

    def __init__(self, num_slots: int):
        k = torch.randn(num_slots, DIM, device="cuda", dtype=torch.bfloat16)
        k_fp4, k_sf = quantize_fp4_indexer_tensor(k, rne=True)
        num_pages = num_slots // PAGE
        self.buf = torch.cat(
            [
                k_fp4.view(torch.uint8).reshape(num_pages, PAGE * 64),
                k_sf.view(torch.uint8).reshape(num_pages, PAGE * 4),
            ],
            1,
        )

    def get_low_ratio_index_k_fp4(self, layer_id, slots):
        slots = slots.to(torch.int64)
        page, off = (slots // PAGE).unsqueeze(-1), slots % PAGE
        dev = self.buf.device
        payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=dev)
        scale_cols = (PAGE * 64 + off * 4).unsqueeze(-1) + torch.arange(4, device=dev)
        payload = self.buf[page, payload_cols].view(torch.int8)
        scales = self.buf[page, scale_cols].contiguous().view(torch.int32).squeeze(-1)
        return payload, scales

    def get_index_k_page_size(self, compress_ratio):
        return PAGE

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.buf


class Indexer:
    """DeepseekV41Indexer's own queries and head weights over lookup projections:
    wq_b and weights_proj return the rows the row ids in q_lora and x carry, so
    the scorer's fused projection and the prefill break's see the same operands."""

    n_heads = n_local_heads = HEADS
    index_head_dim, rope_head_dim = DIM, ROPE
    head_weight_scale = DIM**-0.5 * HEADS**-0.5
    index_topk = TOPK
    is_candidate_source = False
    uses_candidates = False
    queries = DeepseekV41Indexer.queries
    head_weights = DeepseekV41Indexer.head_weights

    def __init__(self, rows: int):
        self.q_raw = torch.randn(rows, HEADS * DIM, device="cuda").to(torch.bfloat16)
        self.w_raw = torch.rand(rows, HEADS, device="cuda").to(torch.bfloat16)
        self.ids = torch.arange(rows, device="cuda")[:, None]
        # The base batch's roped queries and head weights, set by make_batch.
        self.q = self.w = None

    def wq_b(self, q_lora):
        return self.q_raw[q_lora[:, 0]], None

    def head_weights_raw(self, x):
        return self.w_raw[x[:, 0]]

    def as_layer(self, *, source: bool):
        """The same projections as the candidate source or as a consumer."""
        layer = copy.copy(self)
        layer.is_candidate_source, layer.uses_candidates = source, not source
        return layer


class Batch(NamedTuple):
    pool: IndexPool
    indexer: Indexer
    inputs: PrefillInputs
    req_to_token: torch.Tensor  # [requests, stride] int32
    page_table: torch.Tensor  # [rows, KV pages] int32


def make_batch(contexts, rows_per_request, seed, like=None) -> Batch:
    """Requests whose last `rows_per_request` tokens are the extend, with the ratio-1
    index K of every position in the pool; `like` shares another batch's pool."""
    torch.manual_seed(seed)
    dev = "cuda"
    stride = (
        like.req_to_token.shape[1] if like else -(-max(contexts) // KV_PAGE) * KV_PAGE
    )
    request_base = stride * torch.arange(len(contexts), device=dev, dtype=torch.int32)
    req_to_token = (
        torch.arange(stride, device=dev, dtype=torch.int32).expand(len(contexts), -1)
        + request_base[:, None]
    )
    pool = like.pool if like else IndexPool(stride * len(contexts))
    rows = sum(rows_per_request)
    indexer = like.indexer if like else Indexer(rows)
    positions = torch.cat(
        [
            torch.arange(ctx - n, ctx, device=dev)
            for ctx, n in zip(contexts, rows_per_request)
        ]
    )
    # A shared indexer also shares the RoPE table its layers were built with.
    if like:
        freqs_cis = like.inputs.freqs_cis
    else:
        angles = torch.rand(stride, ROPE // 2, device=dev) * 2 * torch.pi
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        indexer.q = indexer.queries(indexer.ids[:rows], freqs_cis[positions])
        indexer.w = indexer.head_weights(indexer.ids[:rows])
    rows_device = torch.tensor(rows_per_request, dtype=torch.int32, device=dev)
    req_rows = torch.repeat_interleave(
        torch.arange(len(contexts), device=dev), rows_device.to(torch.int64)
    )
    page_table = (req_to_token[:, ::KV_PAGE] // KV_PAGE)[req_rows].contiguous()
    inputs = PrefillInputs(
        indexer=indexer,
        layer_id=0,
        compress_ratio=1,
        freqs_cis=freqs_cis,
        x=indexer.ids[:rows],
        q_lora=indexer.ids[:rows],
        positions=positions,
        req_rows=req_rows,
        req_pool_indices=torch.arange(len(contexts), device=dev),
        kv_page_table=page_table,
        seq_lens_cpu=list(contexts),
        rows_per_request=list(rows_per_request),
        rows_per_request_device=rows_device,
        # Placeholders: each layer below writes into its own Selection.
        out_raw_indices=torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev),
        out_page_indices=torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev),
    )
    return Batch(pool, indexer, inputs, req_to_token, page_table)


def with_projection(batch: Batch) -> PrefillInputs:
    """The same inputs with the projection the prefill graph's break precomputed."""
    inputs = batch.inputs
    return msgspec.structs.replace(
        inputs,
        q=batch.indexer.queries(inputs.q_lora, inputs.freqs_cis[inputs.positions]),
        weights=batch.indexer.head_weights(inputs.x),
    )


def new_selection(rows: int) -> Selection:
    return Selection(
        page_indices=torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda"),
        raw_indices=torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda"),
    )


def into(inputs: PrefillInputs, selection: Selection) -> PrefillInputs:
    """The same inputs writing their selection into `selection`."""
    return msgspec.structs.replace(
        inputs,
        out_raw_indices=selection.raw_indices,
        out_page_indices=selection.page_indices,
    )


def eager_layers(batch: Batch, inputs: PrefillInputs):
    """(the candidate source's selection, a consumer's selection), as the eager
    break runs them."""
    prefill, _ = make_candidate_indexer(
        token_to_kv_pool=batch.pool,
        req_to_token=batch.req_to_token,
        page_size=KV_PAGE,
        candidate_topk_blocks=TOPK_BLOCKS,
        candidate_block_size=BLOCK,
    )
    rows = inputs.positions.shape[0]
    source, consumer = new_selection(rows), new_selection(rows)
    published = prefill.publish_prefill(into(inputs, source))
    prefill.consume_prefill(into(inputs, consumer), published)
    return source, consumer


def captured_layer(batch: Batch) -> Selection:
    """The captured prefill graph's full top-k over the whole static width."""
    inputs = batch.inputs
    page_table = expand_index_page_table(
        batch.page_table,
        full_page_size=KV_PAGE,
        compress_ratio=1,
        index_page_size=PAGE,
    )
    rows = inputs.positions.shape[0]
    metadata = PagedIndexerMetadata(
        page_size=KV_PAGE,
        compressed_page_size=PAGE,
        page_table=page_table,
        compressed_seq_lens=(inputs.positions + 1).to(torch.int32),
        use_topk_v2=False,
        use_prefill_cuda_graph=True,
        compress_ratio=1,
        row_chunk=2048 if rows > 2048 else 0,
    )
    out = new_selection(rows)
    full_topk = make_full_topk_indexer(
        token_to_kv_pool=batch.pool, req_to_token=batch.req_to_token
    )
    full_topk.topk_prefill_captured(
        CapturedPrefillInputs(
            indexer=batch.indexer,
            layer_id=0,
            q=batch.indexer.q,
            weights=batch.indexer.w,
            paged_metadata=metadata,
            out_page_indices=out.page_indices,
            out_raw_indices=out.raw_indices,
        ),
    )
    return out


def fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """Two e2m1 codes per byte as values: -0 (code 8) folds into +0, which the
    fused projection and quantize_index_q produce for the same zero element."""
    codes = packed.view(torch.uint8)
    codes = torch.stack((codes & 15, codes >> 4), dim=-1)
    return torch.where(codes == 8, 0, codes)


def picks(selection: Selection, row: int) -> set:
    return set(selection.raw_indices[row].tolist()) - {-1}


def assert_overlap(
    test, got: Selection, want: Selection, minimum: float, row_minimum=None
):
    """Every row overlaps by `minimum`; with `row_minimum`, every row by that and
    the rows on average by `minimum`."""
    shares = []
    for row in range(got.raw_indices.shape[0]):
        g, w = picks(got, row), picks(want, row)
        test.assertEqual(len(g), len(w), row)
        test.assertGreaterEqual(
            len(g & w), (minimum if row_minimum is None else row_minimum) * len(w), row
        )
        shares.append(len(g & w) / max(len(w), 1))
    if row_minimum is not None:
        test.assertGreaterEqual(sum(shares) / len(shares), minimum)


class StaticBatch:
    """The static buffers a prefill graph replays from, refreshed per batch the way
    the runner refreshes the capture metadata."""

    def __init__(self, batch: Batch):
        dev = "cuda"
        rows = GRAPH_BUCKET
        self.ids = torch.zeros(rows, 1, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(rows, dtype=torch.int64, device=dev)
        self.req = torch.zeros(rows, dtype=torch.int64, device=dev)
        self.page_table = torch.zeros(
            rows, batch.page_table.shape[1], dtype=torch.int32, device=dev
        )
        self.page_indices = torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev)
        self.raw_indices = torch.full((rows, TOPK), -1, dtype=torch.int32, device=dev)
        self.projection = {
            "q": torch.zeros(rows, HEADS, DIM, dtype=torch.bfloat16, device=dev),
            "w": torch.zeros(rows, HEADS, dtype=torch.bfloat16, device=dev),
        }
        self.metadata = SimpleNamespace(
            low_ratio_req_indices=self.req,
            low_ratio_pos_i64=self.pos,
            core_metadata=SimpleNamespace(
                sparse_page_indices=lambda ratio: self.page_indices,
                sparse_raw_indices=lambda ratio: self.raw_indices,
                page_table=self.page_table,
            ),
            late_layer_tail=None,
            candidate_metadata=None,
        )

    def load(self, batch: Batch) -> SimpleNamespace:
        """Refresh the buffers in place and return the live forward batch."""
        inputs = batch.inputs
        rows = inputs.positions.shape[0]
        for buf in (self.ids, self.pos, self.req):
            buf.zero_()
        self.page_indices.fill_(-1)
        self.raw_indices.fill_(-1)
        self.ids[:rows] = inputs.x
        self.pos[:rows] = inputs.positions
        self.req[:rows] = inputs.req_rows
        self.page_table[:rows] = batch.page_table
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            encoder_swa_replay=False,
            global_num_token_non_padded_cpu=rows,
            req_pool_indices=inputs.req_pool_indices,
            seq_lens_cpu=torch.tensor(inputs.seq_lens_cpu),
            extend_seq_lens=inputs.rows_per_request_device,
            extend_seq_lens_cpu=inputs.rows_per_request,
        )


def make_backend(batch: Batch, static: StaticBatch, eager_indexer: bool):
    """The attention backend's low-ratio source methods over the fake pool."""
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    prefill, _ = make_candidate_indexer(
        token_to_kv_pool=batch.pool,
        req_to_token=batch.req_to_token,
        page_size=KV_PAGE,
        candidate_topk_blocks=TOPK_BLOCKS,
        candidate_block_size=BLOCK,
    )
    backend = SimpleNamespace(
        prefill_graph_eager_indexer=eager_indexer,
        forward_metadata=static.metadata,
        tail_forward_metadata=None,
        tail_source_kv_layer_id=None,
        prefill_candidates=prefill,
        _low_ratio_in_prefill_graph=lambda: True,
        _source_projection_buffers=lambda rows, ratio: {
            name: buf[:rows] for name, buf in static.projection.items()
        },
        # Paged attention reads the page indices too, so fill both.
        _low_ratio_prefill_reads_page_indices=lambda forward_batch: True,
    )
    for name in (
        "_low_ratio_index_topk",
        "_make_low_ratio_prefill_indexer_inputs",
        "_publish_candidate_metadata",
    ):
        setattr(backend, name, partial(getattr(DeepseekV4AttnBackend, name), backend))
    return backend


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM's paged sparse MQA logits need SM100",
)
class TestPrefillGraphEagerIndexer(CustomTestCase):
    def setUp(self):
        self.enterContext(get_parallel().override(attn_cp_size=1))

    @torch.inference_mode()
    def test_precomputed_projection_is_the_indexers_own(self):
        """The projection break's queries and head weights reach the scorer as the
        operands the indexer would project itself, so the selections agree, for a
        source and a consumer across two requests past the candidate window."""
        batch = make_batch([32768, 20000], [48, 30], seed=1)
        inputs = with_projection(batch)
        projected = get_deep_gemm_prefill_data(batch.inputs, batch.req_to_token)
        handed_over = get_deep_gemm_prefill_data(inputs, batch.req_to_token)
        for name in DeepGEMMPrefillData.__struct_fields__:
            got, want = getattr(handed_over, name), getattr(projected, name)
            if name == "q_fp4":
                self.assertTrue(torch.equal(fp4_values(got), fp4_values(want)), name)
            elif isinstance(want, torch.Tensor):
                self.assertTrue(torch.equal(got, want), name)
            else:
                self.assertEqual(got, want, name)
        source, consumer = eager_layers(batch, inputs)
        want_source, want_consumer = eager_layers(batch, batch.inputs)
        assert_overlap(self, source, want_source, MIN_SOURCE_OVERLAP)
        assert_overlap(self, consumer, want_consumer, MIN_CONSUMER_OVERLAP)
        with self.assertRaises(AssertionError):
            get_prefill_queries(SimpleNamespace(q=inputs.q, weights=None))

    @torch.inference_mode()
    def test_source_layer_beyond_the_window_is_the_captured_full_top_k(self):
        """At 32K, past the 16384 positions the captured indexer may score, the
        eager source layer selects what the captured full top-k selects."""
        batch = make_batch([32768, 20000], [48, 30], seed=2)
        source, _ = eager_layers(batch, with_projection(batch))
        assert_overlap(self, source, captured_layer(batch), MIN_SOURCE_OVERLAP)

    @torch.inference_mode()
    def test_consumer_layer_inside_the_window_is_the_captured_full_top_k(self):
        """Inside the window every block is a candidate, so the eager consumer
        selects what the captured full top-k selects."""
        batch = make_batch([12000, 9000], [48, 30], seed=3)
        _, consumer = eager_layers(batch, with_projection(batch))
        assert_overlap(self, consumer, captured_layer(batch), MIN_CONSUMER_OVERLAP)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM's paged sparse MQA logits need SM100",
)
class TestEagerIndexerPeakMemory(CustomTestCase):
    """The candidate source, the largest eager index layer, stays within the
    startup plan's peak model at long context, and smaller planned tiles select
    what the full-budget tiles select."""

    def setUp(self):
        self.enterContext(get_parallel().override(attn_cp_size=1))
        self.addCleanup(set_planned_index_logits_budget, None)
        # The pool's own gather, not the test pool's index-tensor gather, whose
        # [slots, 64] int64 offsets would dominate the peak.
        self.enterContext(
            patch.object(
                IndexPool,
                "get_low_ratio_index_k_fp4",
                lambda pool, layer_id, slots: gather_fp4_index_k(
                    pool.buf, slots, page_size=PAGE
                ),
            )
        )

    def _publish(self, batch: Batch, budget_bytes: int):
        set_planned_index_logits_budget(None)
        set_planned_index_logits_budget(budget_bytes)
        prefill, _ = make_candidate_indexer(
            token_to_kv_pool=batch.pool,
            req_to_token=batch.req_to_token,
            page_size=KV_PAGE,
            candidate_topk_blocks=TOPK_BLOCKS,
            candidate_block_size=BLOCK,
        )
        rows = batch.inputs.positions.shape[0]
        source = new_selection(rows)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        published = prefill.publish_prefill(into(batch.inputs, source))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del published
        return source, peak

    @torch.inference_mode()
    def test_long_context_publish_fits_the_plan(self):
        context, rows = 1 << 18, 4096
        batch = make_batch([context], [rows], seed=5)
        want, _ = self._publish(batch, INDEX_LOGITS_BUDGET_CAP_BYTES)
        for budget in (256 << 20, 64 << 20):
            got, peak = self._publish(batch, budget)
            self.assertLessEqual(
                peak,
                eager_indexer_peak_bytes(
                    budget_bytes=budget, rows=rows, context_len=context
                ),
            )
            assert_overlap(self, got, want, MIN_SOURCE_OVERLAP)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "DeepGEMM's paged sparse MQA logits need SM100",
)
class TestPrefillGraphIndexerBreak(CustomTestCase):
    def setUp(self):
        self.enterContext(get_parallel().override(attn_cp_size=1))

    @torch.inference_mode()
    def test_replay_selects_what_the_eager_path_selects(self):
        """A prefill graph captured once replays other batches past the window with
        the live rows scored by the eager source and consumer layers, the candidate
        blocks carried between the breaks, and the padding rows left unselected."""
        self._assert_replay_matches_eager([32768, 20000], [30000, 24576])

    @torch.inference_mode()
    def test_replay_past_the_old_64k_cap_selects_what_the_eager_path_selects(self):
        """The same past the 64K cap the recipe used to set: the graph replays
        256K- and 100K-token contexts, so a long conversation's chunks keep it."""
        self._assert_replay_matches_eager(
            [262144, 100000],
            [196608, 70000],
            consumer_row_minimum=MIN_LONG_CONSUMER_ROW_OVERLAP,
        )

    def _assert_replay_matches_eager(
        self, contexts, other_contexts, consumer_row_minimum=None
    ):
        from sglang.srt.layers.attention import deepseek_v4_backend as module
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            set_tc_piecewise_forward_context,
        )

        base = make_batch(contexts, [48, 30], seed=4)
        batches = [base, make_batch(other_contexts, [20, 15], seed=4, like=base)]
        static = StaticBatch(base)
        backend = make_backend(base, static, eager_indexer=True)
        layers = [
            SimpleNamespace(
                layer_id=i,
                compress_ratio=1,
                compressor=None,
                indexer=base.indexer.as_layer(source=i == 0),
                freqs_cis=base.inputs.freqs_cis,
            )
            for i in range(2)
        ]
        # What each layer's attention reads in the graph segment after its break.
        after_layer = [torch.zeros_like(static.raw_indices) for _ in layers]

        def context(forward_batch):
            return set_tc_piecewise_forward_context(
                forward_batch,
                attention_layers=[],
                quant_config=None,
                moe_layers=[],
                moe_fusions=[],
            )

        def run_layers(forward_batch):
            for layer, snapshot in zip(layers, after_layer):
                module.DeepseekV4AttnBackend.forward_low_ratio_sources(
                    backend,
                    layer=layer,
                    x=static.ids,
                    q_lora=static.ids,
                    positions=static.pos,
                    forward_batch=forward_batch,
                    run_compressor=False,
                )
                snapshot.copy_(static.raw_indices)

        graph = BreakableCUDAGraph()
        capture_batch = static.load(base)
        with (
            patch.object(module, "dsa_use_prefill_cp", return_value=False),
            context(capture_batch),
            BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()),
        ):
            run_layers(capture_batch)

        for batch in batches:
            forward_batch = static.load(batch)
            with context(forward_batch):
                graph.replay()
            torch.cuda.synchronize()
            rows = batch.inputs.positions.shape[0]

            reference = make_backend(batch, static, eager_indexer=False)
            wants = []
            for layer in layers:
                static.raw_indices.fill_(-1)
                reference._low_ratio_index_topk(
                    layer,
                    batch.inputs.x,
                    batch.inputs.q_lora,
                    batch.inputs.req_rows,
                    batch.inputs.positions,
                    forward_batch,
                )
                wants.append(Selection(None, static.raw_indices[:rows].clone()))
            for snapshot, want, tolerance, row_minimum in zip(
                after_layer,
                wants,
                (MIN_SOURCE_OVERLAP, MIN_CONSUMER_OVERLAP),
                (None, consumer_row_minimum),
            ):
                assert_overlap(
                    self, Selection(None, snapshot[:rows]), want, tolerance, row_minimum
                )
                self.assertTrue((snapshot[rows:] == -1).all(), "padding rows selected")


if __name__ == "__main__":
    unittest.main()
