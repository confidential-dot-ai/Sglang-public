import itertools
import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
    compute_position,
)
from sglang.srt.model_loader.weight_utils import get_pp_stage_load_group
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    maybe_build_draft_sampler,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    alloc_verify_window,
    compact_dp_sum_capture_enabled,
    compact_dp_sum_graph_token_buckets,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_sampling_mask import (
    SpeculativeSamplingMaskCapture,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_pp_context,
    draft_tp_context,
    lens_to_device,
    prepare_mamba_track_for_verify,
)
from sglang.srt.utils import (
    is_cuda,
    is_cuda_alike,
    is_hip,
    is_npu,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_npu = is_npu()

_verify_dump_ct = itertools.count()


def _maybe_dump_verify_forward(
    *,
    can_run_cuda_graph: bool,
    verify_ids_2d: torch.Tensor,
    next_token_logits: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    bs: int,
    graph_positions: Optional[torch.Tensor] = None,
    graph_input_ids: Optional[torch.Tensor] = None,
    kv_probe_slots: Optional[torch.Tensor] = None,
    kv_pool=None,
    graph_seq_lens: Optional[torch.Tensor] = None,
    graph_out_cache_loc: Optional[torch.Tensor] = None,
    shadow_logits: Optional[torch.Tensor] = None,
    plan_info_main: Optional[dict] = None,
    plan_info_shadow: Optional[dict] = None,
) -> None:
    """Record the target verify forward's inputs and its top prediction,
    tagged by whether a captured cuda graph served it.

    Eager and graph runs use identical accept code, so if the graph-served
    calls predict garbage while the eager-served ones predict sensible
    continuations *within the same process*, the fault is the captured graph
    rather than anything downstream. Tagging by can_run_cuda_graph gives that
    comparison without needing two servers with different random trajectories.
    """
    path = envs.SGLANG_DSPARK_DEBUG_ACCEPT_DUMP_PATH.get()
    if not path or next_token_logits is None:
        return
    ct = next(_verify_dump_ct)
    if ct > 400:
        return
    stride = verify_ids_2d.shape[1]
    row_logits = next_token_logits.view(bs, stride, -1)[0].float()
    top = row_logits.topk(3, dim=-1)
    # Per-verify-position breakdown (stride == 1 anchor/bonus token + verify_w
    # draft tokens). custom_mask is always None for DSpark (no tree structure;
    # linear window), so causal visibility for positions 1..verify_w is
    # derived implicitly by each attention backend from positions/seqlen
    # alone -- unlike position 0 (a plain single-token case, same shape as
    # ordinary decode). If corruption is isolated to positions >= 1, that
    # points at the multi-query causal alignment specifically, not a
    # position-independent fault (KV, workspace, collective, ...) that would
    # damp every position uniformly.
    absmax_per_pos = row_logits.abs().max(dim=-1).values
    # What the captured graph actually consumed this tick: the runner's static
    # input buffers (refreshed per replay via copy_ in load_batch). Recorded
    # for eager ticks too (tagged cuda_graph=False; buffers are then stale
    # leftovers, ignore offline). positions must be exactly
    # arange(prefix, prefix + stride) and input_ids must equal verify_ids --
    # any divergence under cuda_graph=True is input-side corruption upstream
    # of the model forward.
    graph_positions_row0 = (
        [int(v) for v in graph_positions[:stride]]
        if graph_positions is not None
        else None
    )
    graph_input_ids_row0 = (
        [int(v) for v in graph_input_ids[:stride]]
        if graph_input_ids is not None
        else None
    )
    topk_candidates = None
    kv_probe = None
    if kv_probe_slots is not None and kv_pool is not None:
        probe_layer = len(kv_pool.k_buffer) // 2
        k_rows = kv_pool.k_buffer[probe_layer][kv_probe_slots]
        ks_rows = kv_pool.k_scale_buffer[probe_layer][kv_probe_slots]
        kv_probe = {
            "layer": probe_layer,
            "slots": [int(s) for s in kv_probe_slots],
            "k_zero_frac": round(float((k_rows == 0).float().mean()), 4),
            "k_scale_zero_frac": round(float((ks_rows == 0).float().mean()), 4),
            # Per-slot content fingerprint (sum of packed bytes). Consecutive
            # verify windows overlap in absolute slots and every tick rewrites
            # its whole window, so an overlapping slot whose fingerprint stays
            # frozen across ticks means the captured graph's KV write did not
            # land -- regardless of whether the slot holds recycled nonzero
            # bytes (which defeats the zero-frac check above).
            "k_row_sums": [int(v) for v in k_rows.long().sum(dim=(1, 2))],
        }
    shadow_diff = None
    if shadow_logits is not None:
        shadow_rows = shadow_logits.view(bs, stride, -1)[0].float()
        shadow_top = shadow_rows.argmax(dim=-1)
        graph_top = row_logits.argmax(dim=-1)
        shadow_diff = {
            "max_absdiff_per_pos": [
                round(float(v), 3)
                for v in (shadow_rows - row_logits).abs().max(dim=-1).values
            ],
            "shadow_absmax_per_pos": [
                round(float(v), 3) for v in shadow_rows.abs().max(dim=-1).values
            ],
            "top1_agree_per_pos": [bool(a == b) for a, b in zip(graph_top, shadow_top)],
            "shadow_top1_row0": [int(t) for t in shadow_top],
        }
    with open(path, "a") as f:
        f.write(
            json.dumps(
                {
                    "kind": "verify_fwd",
                    "ct": ct,
                    "cuda_graph": bool(can_run_cuda_graph),
                    "bs": int(bs),
                    "seq_len0": int(seq_lens[0]),
                    "verify_ids_row0": [int(t) for t in verify_ids_2d[0]],
                    "top1_row0": [int(t) for t in top.indices[:, 0]],
                    "top1_logit_row0": [round(float(v), 3) for v in top.values[:, 0]],
                    "has_nan": bool(torch.isnan(row_logits).any()),
                    "logit_absmax": round(float(row_logits.abs().max()), 3),
                    "logit_absmax_per_pos": [
                        round(float(v), 3) for v in absmax_per_pos
                    ],
                    "graph_positions_row0": graph_positions_row0,
                    "graph_input_ids_row0": graph_input_ids_row0,
                    "topk_candidates": topk_candidates,
                    "kv_probe": kv_probe,
                    "shadow_diff": shadow_diff,
                    # The value the captured sparse metadata actually reads.
                    # Must equal seq_len0 after the seq_lens-refresh fix; was
                    # the capture fill value (1) before it.
                    "graph_seq_len0": (
                        int(graph_seq_lens[0]) if graph_seq_lens is not None else None
                    ),
                    # The KV-write locations the captured scatter actually
                    # reads (graph buffer) vs where the page table says this
                    # tick's window KV must land (kv_probe_slots). A mismatch
                    # means every replay writes the window's K/V somewhere
                    # else (slot 0 = the padded-token dumping ground when the
                    # buffer was never refreshed), leaving the real slots
                    # zero/stale -- the all-families-zero signature in the
                    # replay_view prev_kv records.
                    "graph_out_cache_loc": (
                        [int(v) for v in graph_out_cache_loc[:stride]]
                        if graph_out_cache_loc is not None
                        else None
                    ),
                    "expected_write_slots": (
                        [int(v) for v in kv_probe_slots]
                        if kv_probe_slots is not None
                        else None
                    ),
                    # FlashInfer PrefillPlanInfo scalars for the graph-served
                    # plan() call (plan_info_main) vs the shadow eager
                    # verify's own plan() call (plan_info_shadow), same
                    # (bs, window) shape. A mismatch here (cta_tile_q,
                    # padded_batch_size, split_kv) means the two forwards
                    # aren't even running the same kernel schedule, which
                    # would explain a graph-only-vs-eager-only divergence
                    # that survived forcing split_kv off and disabling
                    # fast_prefill_plan individually.
                    "plan_info_main": plan_info_main,
                    "plan_info_shadow": plan_info_shadow,
                }
            )
            + "\n"
        )


@runtime_checkable
class _SupportsDSparkTargetHiddenProjection(Protocol):
    def set_dspark_target_hidden_projector(
        self,
        projector: Callable[[torch.Tensor], torch.Tensor],
        *,
        num_context_features: int,
    ) -> bool: ...

    def should_project_dspark_target_hidden(
        self,
        *,
        forward_mode: ForwardMode,
        capture_hidden_mode: CaptureHiddenMode,
    ) -> bool: ...


def _configure_target_hidden_projection(
    *, target_model, draft_model, is_deepseek_v4_draft: bool
) -> bool:
    """Install the optional token-major projection before the target SP gather."""
    if is_deepseek_v4_draft:
        return False
    if not draft_model.supports_pre_gather_target_hidden_projection:
        return False
    if not isinstance(target_model, _SupportsDSparkTargetHiddenProjection):
        return False
    return bool(
        target_model.set_dspark_target_hidden_projector(
            draft_model.project_target_hidden,
            num_context_features=int(draft_model.num_context_features),
        )
    )


class DSparkWorkerV2(BaseSpecWorker):
    """Non-last PP stages run only the target; draft state belongs to the last stage."""

    def weight_update_runners(self) -> list[tuple[str, "ModelRunner"]]:
        return [("draft", self.draft_model_runner)] if self._hosts_draft else []

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        draft_worker_cls: type[TpModelWorker] = TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = get_schedule().page_size
        self.device = target_worker.device
        self._draft_worker = None
        self.enable_dp_spec_prefill_coordination = (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        )
        self._hosts_draft = get_parallel().pp_group.is_last_rank
        if not self._hosts_draft:
            return

        self._draft_is_moe = draft_is_deepseek_v4()
        self._draft_dp_context_enabled = (
            get_parallel().attn_dp_enabled and not self._draft_is_moe
        )
        self._is_pd_prefill = get_disagg().disaggregation_mode == "prefill"
        self._decode_graph_allowed = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self._is_pd_prefill
        )
        if (
            get_parallel().attn_dp_enabled
            and self._draft_is_moe
            and get_parallel().attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        # Inside the draft scope the context answers the draft's narrowed rank.
        self._target_tp_rank = get_parallel().tp_rank
        load_group = get_pp_stage_load_group()
        with draft_pp_context(), self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                attention_backend_override=(
                    DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                ),
                draft_worker_cls=draft_worker_cls,
                random_seed=target_worker.random_seed,
                load_group=load_group,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None

        # The mask token is input-only (it is embedded, never sampled), so its
        # bound is the embedding-table row count: the PADDED vocab when the
        # target pads its embedding (e.g. Inkling true vocab 200058, padded
        # 201024, mask 200064), else the plain vocab size.
        target_model_config = self.target_worker.model_runner.model_config
        target_embed_rows = (
            getattr(target_model_config.hf_text_config, "padded_vocab_size", None)
            or target_model_config.vocab_size
        )
        # muP targets declare logits_mup_width_multiplier; the draft was
        # trained against the folded head, so compute_base_logits divides.
        self.draft_model.logits_mup_width_multiplier = getattr(
            target_model_config.hf_text_config, "logits_mup_width_multiplier", None
        )
        self._target_is_mambaish = mambaish_config(target_model_config) is not None
        runtime_config = resolve_runtime_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=int(target_embed_rows),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.sample_from_anchor = bool(self.draft_model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id

        parallel = get_parallel()
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        self._draft_graph_group = (
            parallel.attn_tp_group
            if self._draft_dp_context_enabled
            else parallel.tp_group
        )

        # --- CC conditional decode (CCD) worker contract --------------------
        # scheduler.py's CCD branches gate on `spec_algorithm.is_dflash_family()`,
        # which includes DSPARK, so dflash_cc_conditional_decode.py's
        # check_ccd_eligible / DflashCCDChunk read these attributes off THIS
        # worker too. Without explicit values every read would fall into
        # __getattr__'s target-worker delegation and raise AttributeError.
        #
        # block_size: the per-tick token stride of this worker's decode
        # results. DflashCCDChunk views next_token_ids as [bs, block_size]
        # and hands block_size to the finish kernel /
        # build_synthetic_result(speculative_num_draft_tokens=...); DSpark's
        # decode result is out_tokens [bs, verify_num_draft_tokens] flattened,
        # and DFlashDraftInputV2.prepare_for_decode reserves chunk KV in units
        # of server_args.speculative_num_draft_tokens (== gamma + 1 ==
        # verify_num_draft_tokens, enforced by _handle_dspark) -- so the
        # verify window IS DSpark's block, on both the result and the KV axis.
        self.block_size = int(self.verify_num_draft_tokens)
        # DSpark has no windowed/compact draft-KV rebuild (the DFLASH-only
        # --speculative-draft-window-size feature tdx-checkpoints/120-125
        # hardened for capture): the draft-visible KV is written once per
        # tick by TargetHiddenKvInjector at the verify-window slots, with no
        # per-tick host-side remapping. These two are genuinely False/None,
        # not merely defaults.
        self.use_compact_draft_cache = False
        self.draft_window_size = None
        self._need_mamba_verify_commit = False  # set in init_attention_backends

        if get_parallel().tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, query_token_num=%s, "
                "sample_from_anchor=%s, mask_token_id=%s, markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self.query_token_num,
                self.sample_from_anchor,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.query_token_num), device=self.device
        )

        if getattr(self.draft_model, "uses_own_vocab_modules", False):
            if get_parallel().tp_rank == 0:
                logger.info(
                    "DSpark draft uses its checkpoint-local embedding and LM head."
                )
        else:
            target_model = self.target_worker.model_runner.model
            lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError(
                    "DSpark requires the target model to expose `lm_head` with `weight`."
                )
            self.draft_model.attach_shared_modules(
                embed_tokens=unwrap_lora_layer(
                    self._resolve_target_embed_tokens(target_model)
                ),
                lm_head=lm_head,
            )
        self._target_hidden_projection_enabled = False

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=get_parallel().tp_rank,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_sync=self._tp_sync,
        )
        if (
            get_parallel().attn_dp_enabled
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under attention DP does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
        )
        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        # The step's simulated correct length, read by the eager and graph accepts.
        self._simulated_correct_len = (
            torch.zeros((1,), dtype=torch.int64, device=self.device)
            if self._simulate_acc_len > 0
            else None
        )
        self._greedy_draft = envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get()
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            tp_sync=self._tp_sync,
            random_seed=target_worker.random_seed,
            greedy_draft=self._greedy_draft,
            dp_moe_sync=self._draft_is_moe and get_parallel().attn_dp_enabled,
        )
        self._verify_epilogue = None
        target_is_dsv41 = (
            getattr(
                self.target_worker.model_runner.model_config.hf_text_config,
                "model_type",
                None,
            )
            == "deepseek_v41"
        )
        static_epilogue_supported = (
            target_is_dsv41
            and self._verify_planner.mode_value == "static"
            and self._draft_is_moe
            and not get_parallel().attn_dp_enabled
            and get_parallel().pp_size == 1
        )
        # ROCm (V4.1 target only): inside a HIP graph the accept-site TP broadcasts
        # need the group's pynccl communicator
        if (
            (self._verify_planner.is_compact_mode or static_epilogue_supported)
            and self._decode_graph_allowed
            and (is_cuda() or (is_cuda_alike() and target_is_dsv41))
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                tp_sync=self._tp_sync,
                fused_argmax=target_is_dsv41,
                simulated_correct_len=self._simulated_correct_len,
                simulate_fold=_is_hip and envs.SGLANG_DSPARK_SIMULATE_FOLD.get(),
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                    kv_injector=self._kv_injector,
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_greedy = (
            _is_hip
            and self._simulate_acc_len > 0
            and envs.SGLANG_SIMULATE_ACC_GREEDY.get()
        )
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            tp_sync=self._tp_sync,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
            simulated_correct_len=self._simulated_correct_len,
            block_verification=get_spec().speculative_use_block_verification,
        )

        self._forced_budget_frac: Optional[float] = None
        self._need_mamba_verify_commit = False

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=get_parallel().tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

        if self._is_pd_prefill and not self._draft_is_moe:
            self.draft_model.prune_to_ctx_kv_injection()

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._hosts_draft and self._verify_planner.carries_confidence

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self._hosts_draft:
            return super().spec_v2_attn_backends
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _draft_context(self):
        return draft_tp_context(self._draft_dp_context_enabled)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if not self._hosts_draft:
            return
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if not self._hosts_draft:
            return
        with draft_pp_context(), self._draft_context():
            self._draft_worker.init_attention_backends()
        self._target_hidden_projection_enabled = _configure_target_hidden_projection(
            target_model=self.target_worker.model_runner.model,
            draft_model=self.draft_model,
            is_deepseek_v4_draft=self._draft_is_moe,
        )
        if self._target_hidden_projection_enabled and get_parallel().tp_rank == 0:
            logger.info(
                "DSpark prefill target-hidden projection runs before "
                "sequence-parallel gather."
            )
        # Same condition DFlashWorkerV2 computes here: CCD's gate must know
        # whether the target model would need a mamba/linear-attention state
        # commit after verify (which the captured chunk cannot do).
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def ccd_lm_head_requires_collective(self) -> bool:
        """CCD gate hook (see DFlashWorkerV2's namesake): would the LM-head
        paths CCD captures fall through to an NCCL collective recorded
        DIRECTLY in the WHILE body (illegal event nodes -->
        cudaErrorInvalidValue at instantiation, tdx-checkpoints/122)?

        DSpark's exposure is the draft base-logits path: with the
        graph-folded draft sampler, models/dspark.py's
        compute_base_logits/gather_and_crop_vocab all-gather runs INSIDE the
        pre-captured draft cuda graph (legal as an embedded child node, same
        as the target verify graph's own logits all-gather at TP>1); without
        it, DraftBlockProposer.propose's eager fallback would record that
        all-gather straight into the body. Unlike DFlash there is no
        replicated-weight CCD fallback here. The sampler-is-None case is
        already fully gated by ccd_capture_incompatible_reason (with a more
        precise reason), so this only needs to answer for the remaining
        configs: folded sampler present -> no in-body collective.
        """
        if self._draft_sampler is not None:
            return False
        return int(get_parallel().tp_size) > 1

    def ccd_capture_incompatible_reason(self) -> Optional[str]:
        """Worker-capability hook read by check_ccd_eligible /
        partition_ccd_eligible: a static-config reason this worker's
        forward_batch_generation cannot run inside CCD's captured WHILE
        body, or None when it can. Every reason names the real blocker.
        """
        if self._verify_planner.schedules_verify_budget:
            # Non-static ragged verify schedules per-tick verify lengths on
            # the host: verify_lens D2H + .tolist() in schedule_layout, an
            # attn-TP broadcast of verify_lens inside the body
            # (dspark_planner._schedule_verify_lens), and per-tick
            # confidence-relay publishes CCD's chunk never issues -- host
            # syncs and collectives inside the captured region.
            return (
                "CCD with DSpark requires SGLANG_RAGGED_VERIFY_MODE=static "
                f"(got {self._verify_planner.mode_value!r}: per-tick verify "
                "lens are scheduled on the host and broadcast across TP "
                "inside the captured body)"
            )
        if self._observers.has_capture_unsafe_sink:
            # Segment timing records CUDA events per tick (event nodes are
            # illegal in a conditional body), REQS staging issues a per-tick
            # D2H, block-accept/STS recorders .tolist() on flush.
            return (
                "CCD with DSpark requires all step observers disabled "
                "(SGLANG_DSPARK_DEBUG_DUMP / SGLANG_DSPARK_ENABLE_SPS_RECORD "
                "/ SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH / "
                "SGLANG_DSPARK_STS_COLLECT_PATH / "
                "SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS): active sinks record "
                "CUDA events and issue D2H copies every tick"
            )
        if self._draft_sampler is None:
            # Eager proposal path: compute_base_logits runs OUTSIDE the
            # pre-captured draft graph -- under TP>1 that records an NCCL
            # all-gather of the shared LM head directly in the body
            # (tdx-checkpoints/122 crash class), and it only ever runs when
            # the draft forward itself could not replay its captured graph
            # (eager attention-metadata host ops in the body).
            return (
                "CCD with DSpark requires the graph-folded draft sampler "
                "(cuda graphs enabled, markov-head draft): the eager "
                "proposal fallback computes draft base logits outside the "
                "captured draft graph, all-gathering the shared LM head "
                "across TP inside the body under TP>1"
            )
        return None

    def log_ccd_capability(self) -> None:
        """Report, once at startup, whether CCD can actually run here.

        CCD is strictly opt-in and its eligibility is evaluated per batch, so
        a misconfigured deployment silently serves with CCD off and looks
        identical to one where it is on. ccd_capture_incompatible_reason
        already names the real blocker; surfacing it turns "is CCD running?"
        into something an operator reads off the log instead of inferring
        from throughput.
        """
        from sglang.srt.speculative.dflash_cc_conditional_decode import (
            SGLANG_CC_CONDITIONAL_DECODE,
        )

        if not SGLANG_CC_CONDITIONAL_DECODE:
            logger.info("[CCD] disabled (SGLANG_CC_CONDITIONAL_DECODE unset)")
            return
        reason = self.ccd_capture_incompatible_reason()
        if reason is None:
            logger.info(
                "[CCD] enabled; this DSpark worker is capture-eligible "
                "(per-batch gates in check_ccd_eligible still apply)"
            )
        else:
            logger.warning("[CCD] enabled but INELIGIBLE: %s", reason)

    def init_cuda_graphs(self):
        if not self._hosts_draft:
            return
        capture_decode_cuda_graph = self._decode_graph_allowed
        available_mem = self._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM,
            self.device,
            self.gpu_id,
            group=self._draft_graph_group,
        )
        if is_cuda_alike() and capture_decode_cuda_graph:
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with draft_pp_context(), self._draft_context():
            if capture_decode_cuda_graph:
                # Keep the draft model graph enabled when folded proposal is
                # disabled, but do not capture the proposal head as a tail
                # hook. The proposer will compute base logits and the Markov
                # block eagerly from the graph's hidden states instead. Apart
                # from being the intended precision fallback, skipping the
                # unused hook avoids paying for two proposal computations.
                if envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get():
                    self._draft_sampler = self._maybe_build_draft_sampler()
                    if self._draft_sampler is not None:
                        self.draft_model_runner.capture_tail_hooks.append(
                            make_draft_sampler_capture_hook(self._draft_sampler)
                        )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            with self._ring_mapping_capture_safe():
                self._draft_worker.init_cuda_graphs(
                    capture_decode_cuda_graph=capture_decode_cuda_graph
                )
        # After the draft sampler is built -- ccd_capture_incompatible_reason
        # reads self._draft_sampler, which is assigned above, so calling this
        # any earlier reports a spurious 'eager proposal fallback'.
        self.log_ccd_capability()

    @contextmanager
    def _ring_mapping_capture_safe(self):
        """Make the draft's full->SWA ring mapping readable during capture.

        The mapping is initialised to -1 everywhere on purpose, so that a loc
        no forward has published faults loudly instead of silently aliasing
        ring slot 0. Graph capture then walks straight into that: its warmup
        forwards write at synthetic out_cache_loc values that no injector has
        published, translate_loc_from_full_to_swa hands back -1, and
        store_kvcache dies on `index >= 0 && index < size_limit` -- a
        device-side assert from inside BatchPrefillWithPagedKVCache, tens of
        minutes into a boot.

        Capture only needs the indices to be IN RANGE; the values are dummy
        either way, no request is live, and replay reads the mapping tensor
        fresh each tick, so what is captured here is the load, not the value.
        Point every entry at slot 0 for the duration and restore the sentinel
        afterwards, so the loud-fault property still holds at runtime.
        """
        pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        mapping = getattr(pool, "full_to_swa_index_mapping", None)
        if getattr(pool, "draft_swa_ring_stride", None) is None or mapping is None:
            yield
            return
        mapping.fill_(0)
        try:
            yield
        finally:
            mapping.fill_(-1)

    def _maybe_build_draft_sampler(self):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self._target_tp_rank,
            noise=self._proposer.noise,
            greedy_draft=self._greedy_draft,
            tp_sync=self._tp_sync,
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        pass

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        if not self._hosts_draft:
            return
        self._forced_budget_frac = frac
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        if not self._hosts_draft:
            return None
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        if not self._hosts_draft:
            return
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        if not self._hosts_draft:
            return None
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if not self._hosts_draft:
            return
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def _append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """DFlash-named entry point for the PD hidden-state fast path.

        disaggregation/decode.py's _materialize_dflash_draft_kv_for_prebuilt
        calls this name on model_worker when prefill shipped real hidden
        states. DSpark performs the same injection via _kv_injector, and the
        two signatures match exactly, so delegate rather than duplicate.

        Without this, __getattr__ forwards the lookup to target_worker (which
        also lacks it) and the decode scheduler dies with AttributeError.
        """
        self._kv_injector.inject_target_hidden(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
        force_eager_decode: bool = False,
    ) -> GenerationBatchResult:
        """force_eager_decode: run this step's forwards eagerly -- see
        ForwardBatch.force_eager_decode. Only meaningful for the decode
        path."""
        if not self._hosts_draft:
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if batch.is_extend_in_batch and self.enable_dp_spec_prefill_coordination:
                plan = DPSpecPrefillCoordinationPlan(
                    *batch.dp_spec_prefill_coordination_metadata,
                    draft_width=self._proposer.query_token_num,
                    verify_width=self.verify_num_draft_tokens,
                )
                if plan.heterogeneous:
                    return self._forward_dp_spec_prefill_coordination(
                        batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
                    )
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(
                batch,
                on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=capture_hidden_mode,
            )

        return self._forward_decode(
            batch,
            on_publish,
            grammar_barrier,
            force_eager_decode=force_eager_decode,
        )

    def _forward_dp_spec_prefill_coordination(
        self, batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
    ):
        """Pair local prefill with peer verification after the shared draft pass."""
        if not batch.forward_mode.is_extend():
            return self._forward_decode(
                batch, on_publish, grammar_barrier, coordination_plan=plan
            )

        rank = get_parallel().attn_dp_rank
        target_local_only = len(batch.global_num_tokens) == 1
        plan.apply(
            batch,
            "draft",
            rank,
            local_only=target_local_only or self._draft_dp_context_enabled,
        )
        with self._draft_context():
            self._proposer.run_idle_participation(batch)
        plan.apply(batch, "target", rank, local_only=target_local_only)
        self._verify_planner.note_non_decode_step()
        self._observers.note_prefill_step()
        return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

    def _forward_prefill(
        self,
        batch: ScheduleBatch,
        on_publish,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if get_parallel().attn_dp_enabled:
                self.target_worker.forward_batch_generation(
                    batch, capture_hidden_mode=CaptureHiddenMode.FULL
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_kv_loc_plan=True,
        )
        # BCG replay skips model-side Python, so re-evaluate the same pure predicate.
        target_hidden_is_projected = (
            self._target_hidden_projection_enabled
            and self.target_worker.model_runner.model.should_project_dspark_target_hidden(
                forward_mode=batch.forward_mode,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        new_seq_lens = batch.seq_lens
        batch_output.new_seq_lens = new_seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = next_token_ids.device
        ctx_lens = lens_to_device(lens=batch.extend_lens, device=device)
        draft_seq_lens = lens_to_device(lens=batch.prefix_lens, device=device)
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        # A SWA ring is keyed by (draft req slot, position); thread the per-token
        # state_slot + the req's final position so the injector keeps only the
        # last SWA window (older prefill tokens share a ring slot and would race).
        state_slot = final_pos = None
        # A ring-pooled draft needs the same (req slot, final position) pair to
        # publish its mapping, so compute it whenever either consumer is active.
        # The DRAFT's pool, not the target's: draft_swa_ring_stride is set on
        # the ring-pooled draft pool (kv_cache_configurator's
        # _draft_swa_ring_active branch), and TargetHiddenKvInjector reads that
        # same draft pool. Gating on self.model_runner (which is the TARGET
        # worker's runner, assigned in __init__) always found the attribute
        # missing, so state_slot stayed None and _maybe_publish_ring_mapping
        # raised for exactly the configuration the ring exists to serve.
        pool = self.draft_model_runner.token_to_kv_pool
        needs_ring_keys = self._kv_injector.swa_ring_size is not None or (
            getattr(pool, "draft_swa_ring_stride", None) is not None
        )
        if needs_ring_keys:
            repeats = ctx_lens.to(torch.int64)
            num_tokens = sum(batch.extend_lens)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64),
                repeats,
                output_size=num_tokens,
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64),
                repeats,
                output_size=num_tokens,
            )
        draft_pool = self._kv_injector.draft_model_runner.token_to_kv_pool
        if (
            getattr(draft_pool, "swa_valid_from", None) is not None
            and batch.encoder_swa_reset is not None
        ):
            # A newly placed request's ring holds nothing below its cached prefix.
            placed = [i for i, reset in enumerate(batch.encoder_swa_reset) if reset]
            if placed:
                draft_pool.set_swa_valid_from(
                    batch.req_pool_indices[placed],
                    torch.tensor(
                        [batch.prefix_lens[i] for i in placed],
                        dtype=torch.int64,
                        device=device,
                    ),
                )
        # The draft KV goes to the slots the target prefill just wrote.
        cache_loc = self._kv_injector.ids_for(batch_output.kv_loc_plan)
        batch_output.kv_loc_plan = None
        token_indices = logits_output.hidden_states_token_indices
        if token_indices is not None:
            cache_loc = cache_loc[token_indices]
            positions = positions[token_indices]
            if state_slot is not None:
                state_slot = state_slot[token_indices]
                final_pos = final_pos[token_indices]
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=cache_loc,
            positions=positions,
            state_slot=state_slot,
            final_pos=final_pos,
            target_hidden_is_projected=target_hidden_is_projected,
        )
        # Normally nulled to keep large hidden-state buffers out of
        # overlap scheduling's D2H path. Under PD hidden-state transfer the
        # scheduler reads this tensor immediately after we return (see
        # scheduler.py's write_target_hidden_states call), so nulling it
        # would make the transfer deliver nothing and silently fall back to
        # _recompute_dflash_draft_kv_inplace -- the exact work this feature
        # removes. Keep it only when the caller asked to capture.
        if capture_hidden_mode is None:
            logits_output.hidden_states = None
            logits_output.hidden_states_token_indices = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=new_seq_lens,
        )
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        sum_graph_buckets = compact_dp_sum_graph_token_buckets(
            model_runner=self.model_runner,
            global_tier_num_tokens=getattr(
                batch, "global_spec_verify_tier_num_tokens", None
            ),
            global_num_reqs=batch.global_num_tokens,
        )
        if compact_dp_sum_capture_enabled(self.model_runner):
            if sum_graph_buckets is None:
                # Keep the busy and idle ranks on the same static/eager
                # fallback when the gathered SUM token vector is unavailable
                # or outside the captured grid.  Building the legacy DP-max
                # compact dummy here would make its physical row count exceed
                # the zero local contribution used by eager DP padding.
                return None
            # SUM capture permits a different local graph on every rank.  An
            # idle rank still executes one dummy row and contributes the
            # smallest captured token bucket to the collective vector; it no
            # longer fabricates `global_bs` rows merely to match MAX_LEN's
            # cross-rank graph key.
            return idle_ragged_layout(
                tier_num_reqs=1,
                dp_tier_num_tokens=sum_graph_buckets[get_parallel().attn_dp_rank],
                device=self.device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                model_runner=self.model_runner,
            )
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and get_parallel().attn_dp_enabled
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        sum_graph_buckets = compact_dp_sum_graph_token_buckets(
            model_runner=self.model_runner,
            global_tier_num_tokens=getattr(
                batch, "global_spec_verify_tier_num_tokens", None
            ),
            global_num_reqs=batch.global_num_tokens,
        )
        if compact_dp_sum_capture_enabled(self.model_runner):
            # A missing/sentinel/out-of-grid gathered vector is a shared graph
            # miss.  Do not fall through to MAX_LEN's scalar max here: in
            # compact SUM mode the planner interprets a non-None value as this
            # rank's graph-token bucket, and an out-of-grid max can reach
            # round_up_grid before admission gets a chance to reject it.
            return (
                None
                if sum_graph_buckets is None
                else sum_graph_buckets[get_parallel().attn_dp_rank]
            )
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=getattr(
                batch, "global_spec_verify_tier_num_tokens", None
            )
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(
        self,
        batch: ScheduleBatch,
        on_publish,
        grammar_barrier=None,
        coordination_plan=None,
        *,
        force_eager_decode: bool = False,
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and get_parallel().attn_dp_enabled
            and batch.global_num_tokens is not None
            else None
        )
        if coordination_plan is not None:
            rank = get_parallel().attn_dp_rank
            target_local_only = len(batch.global_num_tokens) == 1
            coordination_plan.apply(
                batch,
                "draft",
                rank,
                local_only=target_local_only or self._draft_dp_context_enabled,
            )

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            if get_parallel().attn_dp_enabled:
                if self._draft_is_moe:
                    self._proposer.run_idle_participation(
                        batch, force_eager_decode=force_eager_decode
                    )
                if coordination_plan is not None:
                    coordination_plan.apply(
                        batch, "target", rank, local_only=target_local_only
                    )
                self._verify_executor.run_idle_participation(
                    batch=batch,
                    idle_layout=self._idle_verify_ragged_layout(batch),
                    force_eager_decode=force_eager_decode,
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        from sglang.srt.oracle_decode.forced_token_pool import (
            compute_oracle_commit_fixed_len,
            gather_oracle_window,
            resolve_ccd_forced_lookup,
        )

        oracle_lookup = resolve_ccd_forced_lookup(batch, device)
        oracle_commit = envs.SGLANG_ORACLE_FIXED_COMMIT_LEN.get()
        if oracle_lookup is not None and (
            oracle_commit <= 0
            or oracle_commit > self.verify_num_draft_tokens
            or self._verify_planner.mode_value != "static"
            or batch.has_grammar
            or batch.return_logprob
        ):
            raise ValueError(
                "DSpark oracle requires static verification, a valid fixed "
                "commit length, and no grammar or logprobs"
            )

        self._observers.begin_step()

        target_model = self.target_worker.model_runner.model
        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
        )

        sampling_info = batch.sampling_info
        # Simulated acceptance overrides correct_len, so only the bonus token needs the
        # request's temperature: draft + accept run greedy, the bonus is still sampled.
        simulate_bonus_sampling_info = None
        if (
            self._simulate_acc_greedy
            and sampling_info is not None
            and not sampling_info.is_all_greedy
            and not batch.has_grammar
            and verify_logits_adjustments_are_noop(sampling_info)
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            simulate_bonus_sampling_info = sampling_info
            sampling_info = None
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
                force_eager_decode=force_eager_decode,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens
        if coordination_plan is not None:
            coordination_plan.apply(batch, "target", rank, local_only=target_local_only)

        confidence = proposal.confidence

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        # Draft computation still runs in full. Teacher-force target inputs so
        # committed KV and Engram history describe the emitted oracle sequence.
        if oracle_lookup is not None:
            oracle_verify, oracle_valid = gather_oracle_window(
                prefix_lens, self.verify_num_draft_tokens, oracle_lookup
            )
            verify_ids_2d = torch.where(oracle_valid, oracle_verify, verify_ids_2d)

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        # A live grammar forces the eager path: the folded epilogue accepts inside
        # the cuda graph off its own buffers, where the mask below never lands.
        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            # The epilogue's in-graph accept is greedy (AcceptGreedy);
            # sampling batches must take the eager accept path even when the
            # draft proposal itself folded.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            # The graph accept is greedy: a sampled simulated bonus stays in the graph
            # only when the epilogue draws it there (SGLANG_DSPARK_SIMULATE_FOLD).
            and (
                simulate_bonus_sampling_info is None
                or self._verify_executor.verify_epilogue.simulate_fold
            )
            and oracle_lookup is None
            and not batch.has_grammar
        )
        self._verify_executor.begin_simulated_step()
        if fold_eligible and self._verify_executor.verify_epilogue.simulate_fold:
            self._verify_executor.verify_epilogue.begin_sim_bonus(
                bs,
                (
                    None
                    if simulate_bonus_sampling_info is None
                    else simulate_bonus_sampling_info.temperatures[:bs]
                ),
            )
        prepare_mamba_track_for_verify(batch)
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    verify_window=verify_window,
                    inject_gate=fold_eligible,
                    force_eager_decode=force_eager_decode,
                )
            else:
                if (
                    self._verify_epilogue is not None
                    and self._verify_planner.mode_value == "static"
                ):
                    self._verify_epilogue.begin_static_step(bs, fold_eligible)
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                    force_eager_decode=force_eager_decode,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph
        shadow_logits = None
        plan_info_main = None
        plan_info_shadow = None
        if (
            envs.SGLANG_DSPARK_DEBUG_SHADOW_EAGER_VERIFY.get()
            and can_run_cuda_graph
            and not run_compact
        ):
            from sglang.srt.layers.attention.flashinfer_backend import (
                get_last_dspark_verify_plan_info,
            )

            plan_info_main = get_last_dspark_verify_plan_info()
            shadow = self._verify_executor.run_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=verify_ids_2d,
                verify_window=verify_window,
                sampling_info=sampling_info,
                force_eager_decode=True,
            )
            shadow_logits = shadow.logits_output.next_token_logits
            plan_info_shadow = get_last_dspark_verify_plan_info()
        target_graph_runner = self.model_runner.decode_cuda_graph_runner
        kv_probe_slots = None
        kv_pool = None
        _maybe_dump_verify_forward(
            can_run_cuda_graph=can_run_cuda_graph,
            verify_ids_2d=verify_ids_2d,
            next_token_logits=logits_output.next_token_logits,
            seq_lens=batch.seq_lens,
            bs=bs,
            graph_positions=(
                target_graph_runner.buffers.positions
                if target_graph_runner is not None
                else None
            ),
            graph_input_ids=(
                target_graph_runner.buffers.input_ids
                if target_graph_runner is not None
                else None
            ),
            kv_probe_slots=kv_probe_slots,
            kv_pool=kv_pool,
            graph_seq_lens=(
                target_graph_runner.buffers.seq_lens
                if target_graph_runner is not None
                else None
            ),
            graph_out_cache_loc=(
                target_graph_runner.buffers.out_cache_loc
                if target_graph_runner is not None
                else None
            ),
            shadow_logits=shadow_logits,
            plan_info_main=plan_info_main,
            plan_info_shadow=plan_info_shadow,
        )

        if batch.has_grammar:
            # run_compact scatters its rows back to (bs * chain_len), so the mask
            # lines up with the logits on both verify paths.
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
            if grammar_mask is not None:
                grammar_mask.apply(logits_output.next_token_logits)

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = (
            fold_eligible
            and can_run_cuda_graph
            and (run_compact or self._verify_planner.mode_value == "static")
        )
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            simulate_bonus_sampling_info=simulate_bonus_sampling_info,
        )
        if oracle_lookup is not None:
            forced_tokens, forced_lens, override = compute_oracle_commit_fixed_len(
                prefix_lens + 1,
                self.verify_num_draft_tokens,
                oracle_commit,
                oracle_lookup,
            )
            accept.out_tokens = torch.where(
                override[:, None], forced_tokens, accept.out_tokens
            )
            accept.commit_lens = torch.where(override, forced_lens, accept.commit_lens)
            last_forced = forced_tokens.gather(
                1, (forced_lens.long() - 1).clamp(min=0)[:, None]
            ).squeeze(1)
            accept.bonus = torch.where(override, last_forced, accept.bonus)
            accept.correct_len = accept.commit_lens - 1
            accept.new_seq_lens = prefix_lens + accept.commit_lens.to(prefix_lens.dtype)

        self.model_runner.ngram_embedding_manager.update_after_verify(
            verify_ids_2d=verify_ids_2d,
            req_pool_indices=batch.req_pool_indices,
            commit_lens=accept.commit_lens,
        )
        # The oracle override above already rewrote these accept fields.
        out_tokens = accept.out_tokens
        commit_lens = accept.commit_lens
        bonus = accept.bonus
        new_seq_lens = accept.new_seq_lens

        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(
            sampling_info,
            next_token_logits=logits_output.next_token_logits,
            draft_input=draft_input,
            draft_token_num=self.verify_num_draft_tokens,
            bs=bs,
            greedy_mask=draft_block.greedy_mask,
        )
        if sampling_mask_capture is not None:
            logits_output.sampling_mask_output = sampling_mask_capture.build_output(
                out_tokens=out_tokens,
                commit_lens=commit_lens,
            )
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=self.verify_num_draft_tokens,
            )

        if on_publish is not None:
            if confidence is not None:
                on_publish(new_seq_lens, confidence=confidence)
            else:
                on_publish(new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        folded_commit = folded_accept and epilogue.folds_commit
        # Consume in this step: every decode graph size shares one aux output,
        # which the next target forward overwrites (resolve_aux_hidden_states_width).
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal.folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            confidence_raw=proposal.confidence_raw,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=bonus,
            commit_lens=commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            can_run_cuda_graph=can_run_cuda_graph,
            folded_accept=folded_accept,
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=new_seq_lens,
        )

    def _commit_target_mamba_states_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit the last accepted verify step's KDA/mamba state (chain
        layout: step index = commit_lens - 1) into the persistent caches."""
        if not self._need_mamba_verify_commit:
            return
        # Chain layout only: step index = commit_lens - 1. A tree (topk > 1)
        # layout would need the accept-index mapping the shared spec_utils
        # commit helper does.
        assert get_spec().speculative_eagle_topk in (None, 1)
        attn_backend = self.target_worker.model_runner.attn_backend

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None
        mamba_track_indices = batch.mamba_track_indices

        if mamba_track_indices is not None:
            mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
            seq_lens_cpu = batch.seq_lens_cpu
            if (
                _is_npu
                and seq_lens_cpu is not None
                and seq_lens_cpu.device.type == "cpu"
                and seq_lens_cpu.ndim == 1
                and seq_lens_cpu.numel() == seq_lens_pre_verify.numel()
                and seq_lens_cpu.dtype in (torch.int32, torch.int64)
            ):
                # Verify restores the CPU prefix lengths before the forward.
                # Acceptance can commit at most this many tokens, so this
                # check needs no device readback. Passing None also avoids
                # the NPU backend's conv-state self-copy for untracked rows.
                if all(
                    seq_len >= 0
                    and seq_len // mamba_track_interval
                    == (seq_len + self.verify_num_draft_tokens) // mamba_track_interval
                    for seq_len in seq_lens_cpu.tolist()
                ):
                    mamba_track_indices = None

        if mamba_track_indices is not None:
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        if not self._hosts_draft:
            return None
        return self._verify_planner.confidence_budget_prepare()
