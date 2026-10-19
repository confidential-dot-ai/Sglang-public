from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

logger = logging.getLogger(__name__)
_is_npu = is_npu()


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash.

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it.
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_req: int = -1

    ragged_verify_layout: Optional[RaggedVerifyLayout] = None
    # Per-DP-rank packed graph-token buckets for compact SUM_LEN capture.
    # None for static/MAX_LEN and for a rank-consistent graph miss.
    dp_graph_num_tokens: Optional[list[int]] = None
    # Committed/live lengths before the verify caller temporarily expands
    # batch.seq_lens_cpu to the target-attention KV lengths.
    live_seq_lens_cpu: Optional[torch.Tensor] = None
    # Conservative request-lifetime bound for candidate graph dispatch.
    candidate_max_seq_len_upper_bound: Optional[int] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)
        self.num_tokens_for_logprob_per_req = int(self.draft_token_num)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
        *,
        force_eager_decode: bool = False,
    ) -> tuple[ForwardBatch, bool]:
        """Prepare a DFLASH verify forward batch for overlap scheduling.

        The caller computes and stores `batch.out_cache_loc` before this
        method is called. On GPU, this helper packages the verify forward and
        pre-initializes CUDA-graph replay or eager attention metadata so the
        actual forward can run with `skip_attn_backend_init=True`. NPU leaves
        attention/graph metadata initialization to ModelRunner because DP/EP
        padding can still change the compressor's runtime shapes.

        force_eager_decode: run the verify forward eagerly; see
        ForwardBatch.force_eager_decode.
        """
        from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify

        batch.input_ids = self.draft_token
        batch.spec_info = self
        if _is_npu and not batch.forward_mode.is_idle():
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                maybe_build_dsv4_verify_bundle,
            )

            batch.out_cache_loc_dsv4 = maybe_build_dsv4_verify_bundle(
                batch,
                self.draft_token_num,
                live_seq_lens_cpu=self.live_seq_lens_cpu,
            )

        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        if self.ragged_verify_layout is not None:
            from sglang.srt.speculative.dspark_components.dspark_planner import (
                compact_dp_sum_graph_token_buckets,
            )

            self.dp_graph_num_tokens = compact_dp_sum_graph_token_buckets(
                model_runner=target_worker.model_runner,
                global_tier_num_tokens=getattr(
                    batch, "global_spec_verify_tier_num_tokens", None
                ),
                global_num_reqs=getattr(batch, "global_num_tokens", None),
            )
        else:
            self.dp_graph_num_tokens = None
        if not batch.forward_mode.is_idle():
            # Rebuild mamba track indices (lazy: gather the positions planned
            # by mamba_lazy_spec_prepare) and clear the stale extend-time mask
            # before init_new snapshots them into the verify ForwardBatch.
            # Same hook eagle/ngram/dspark run before TARGET_VERIFY.
            prepare_mamba_track_for_verify(batch)
        init_started_ns = time.perf_counter_ns()
        if batch.spec_algorithm.is_dspark():
            verify_forward_batch = ForwardBatch.init_dspark_target_verify(
                batch,
                target_worker.model_runner,
                capture_hidden_mode=self.capture_hidden_mode,
                return_hidden_states_before_norm=False,
                force_eager_decode=force_eager_decode,
            )
        else:
            verify_forward_batch = ForwardBatch.init_new(
                batch,
                target_worker.model_runner,
                capture_hidden_mode=self.capture_hidden_mode,
                return_hidden_states_before_norm=False,
                force_eager_decode=force_eager_decode,
            )
        runner = target_worker.model_runner.decode_cuda_graph_runner
        if runner is not None:
            runner.record_metadata_timing(
                "forward_batch_init_cpu_ms",
                (time.perf_counter_ns() - init_started_ns) / 1e6,
            )

        can_run_cuda_graph = bool(
            runner and runner.can_run_graph(verify_forward_batch)
        )
        if _is_npu:
            # Do not pre-plan target verify on NPU. DP/EP padding can change
            # the compressor's logical batch without changing ForwardBatch's
            # stale-plan shape fields. Let ModelRunner select graph/eager and
            # initialize metadata after final batch preparation.
            return verify_forward_batch, can_run_cuda_graph

        if can_run_cuda_graph:
            load_started_ns = time.perf_counter_ns()
            runner.stage_batch(verify_forward_batch)
            runner.record_metadata_timing(
                "preplan_dispatch_cpu_ms",
                (time.perf_counter_ns() - load_started_ns) / 1e6,
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        layout = self.ragged_verify_layout
        if layout is not None:
            # Graph replay pads the batch to the captured request slots. Keep
            # each synthetic row within the statically captured verify width;
            # an unassignable remainder in the token tier is a ghost tail and
            # must not enlarge a request's causal Q/K geometry.
            layout = layout.padded_to_bucket(
                padded_bs=bs, cap=self.draft_token_num
            )

        if layout is None:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            verify_lens = self.draft_token_num
            kv_indices_extra = self.draft_token_num * bs
        else:
            qo_indptr = layout.qo_indptr_device
            verify_lens = layout.verify_lens
            # Capped/device-only layouts intentionally do not carry a host
            # total. The graph token tier is a safe allocation bound; the
            # device indptrs below delimit the exact initialized prefix.
            kv_indices_extra = layout.graph_num_tokens

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + verify_lens
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if kv_indices_buf is not None:
            # Sync-free fast-plan path: write straight into the attention
            # backend's cuda-graph kv_indices buffer (the captured kernels read
            # it), skipping both the fresh allocation and the wrapper plan()'s
            # device-to-device refresh copy.
            kv_indices = kv_indices_buf
        else:
            kv_indices = torch.empty(
                paged_kernel_lens_sum + kv_indices_extra,
                dtype=torch.int32,
                device=device,
            )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask
