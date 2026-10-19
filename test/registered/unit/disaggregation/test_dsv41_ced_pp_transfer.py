"""CED pipeline stages transfer source-owned KV and pending-pair state by layer."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def pool_for_stage(start, end):
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool._stage_start, pool._stage_end = start, end
    pool._unified_kv = False
    pool.swa_kv_pool = SimpleNamespace(kv_buffer=[object()] * (end - start))
    sources = [i for i in (2, 8, 14, 20) if start <= i < end]
    pool.sources_by_ratio = {2: sources}
    pool.kv_pools = {2: SimpleNamespace(kv_buffer=[object()] * len(sources))}
    pool.index_pools = {
        2: SimpleNamespace(
            contiguous_page_row_buffers=lambda: [object()] * len(sources)
        )
    }
    pool.compress_state_pools = [
        SimpleNamespace(ratio=2, request_scoped=True) if i in sources else None
        for i in range(43)
    ]
    pool.indexer_compress_state_pools = [None] * 43
    return pool


class TestCEDPPTransfer(unittest.TestCase):
    def test_both_stages_cover_decode_encoder_entries_once(self):
        """KV and index buffers must land at their global source layers across PP."""
        decode = pool_for_stage(0, 40)
        dst_ids = build_kv_layer_ids(
            token_to_kv_pool=decode,
            draft_token_to_kv_pool=None,
            num_draft_entries=0,
            num_hidden_layers=40,
        )
        self.assertEqual(dst_ids, [2, 8, 14, 20, 2, 8, 14, 20])
        covered = []
        for start, end in ((0, 8), (8, 21)):
            stage = pool_for_stage(start, end)
            src_ids = build_kv_layer_ids(
                token_to_kv_pool=stage,
                draft_token_to_kv_pool=None,
                num_draft_entries=0,
                num_hidden_layers=40,
            )
            pairs = build_transfer_entry_pairs(
                src_ids, dst_ids, len(src_ids), len(dst_ids)
            )
            covered.extend(j for _, j in pairs)
            swa_ids = stage.get_state_layer_ids()
            self.assertEqual(swa_ids, list(range(start, end)))
        self.assertEqual(sorted(covered), list(range(8)))

    def test_full_prefill_tail_stages_cover_all_target_state(self):
        decode = pool_for_stage(0, 40)
        covered_swa, covered_pairs = [], []
        for start, end in ((0, 14), (14, 40)):
            stage = pool_for_stage(start, end)
            for source, dest, covered in (
                (
                    stage.get_state_layer_ids(),
                    decode.get_state_layer_ids(),
                    covered_swa,
                ),
                (
                    stage.get_request_state_layer_ids(),
                    decode.get_request_state_layer_ids(),
                    covered_pairs,
                ),
            ):
                covered.extend(
                    j
                    for _, j in build_transfer_entry_pairs(
                        source, dest, len(source), len(dest)
                    )
                )
        self.assertEqual(sorted(covered_swa), list(range(40)))
        self.assertEqual(sorted(covered_pairs), list(range(4)))

    def test_split_fp4_indexer_buffers_repeat_layer_ids_per_kind(self):
        """The split FP4 indexer registers every payload, then every scale."""
        pool = pool_for_stage(0, 40)
        pool.index_pools[2] = SimpleNamespace(
            contiguous_page_row_buffers=lambda: [object()] * 8
        )
        self.assertEqual(
            pool.get_kv_layer_ids(),
            [2, 8, 14, 20] + [2, 8, 14, 20] + [2, 8, 14, 20],
        )

    def test_pending_pair_transfer_uses_source_layer_ids(self):
        """Ratio-2 state must not be sliced using the ratio-128 layer count."""
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        stage = pool_for_stage(8, 21)
        decode = pool_for_stage(0, 40)
        src_ids = stage.get_request_state_layer_ids()
        dst_ids = decode.get_request_state_layer_ids()
        blocks = []
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = False
        manager.max_transfer_batch_indices = 0
        manager.pp_size = 2
        # TP1 prefill stages feeding a TP2 decode, as in the CED PP2 topology.
        manager.attn_tp_size = 1
        manager._transfer_data = lambda session, entries: blocks.extend(entries) or 0
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.DSV4_REQUEST_STATE],
            state_data_ptrs=[[100, 200, 300]],
            state_item_lens=[[16] * 3],
            state_dim_per_tensor=[],
            state_layer_ids=[src_ids],
            mla_compression_ratios=[0, 0] + [2] * 38,
            prefill_start_layer=8,
            prefill_end_layer=21,
        )
        registration = SimpleNamespace(
            dst_state_data_ptrs=[[1000, 2000, 3000, 4000]],
            dst_state_item_lens=[[16] * 4],
            dst_state_dim_per_tensor=[],
            dst_state_layer_ids=[dst_ids],
            dst_attn_tp_size=2,
        )
        req = SimpleNamespace(mooncake_session_id="test", dst_state_indices=[[0]])
        with envs.SGLANG_OPT_DSV41_PREFILL_TAIL.override(True):
            rc = manager.maybe_send_extra(
                req,
                [[0]],
                executor=None,
                target_rank_registration_info=registration,
            )
            # The prefill tail validates each matched state entry's byte width.
            registration.dst_state_item_lens[0][1] = 32
            with self.assertRaisesRegex(RuntimeError, "state item size mismatch"):
                manager.maybe_send_extra(
                    req,
                    [[0]],
                    executor=None,
                    target_rank_registration_info=registration,
                )
        self.assertEqual(rc, 0)
        self.assertEqual(blocks, [(100, 2000, 16), (200, 3000, 16), (300, 4000, 16)])


if __name__ == "__main__":
    unittest.main()
