from __future__ import annotations

import bisect
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.environ import envs


class RaggedVerifyMode(str, Enum):
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"


def read_ragged_verify_mode() -> RaggedVerifyMode:
    value = envs.SGLANG_RAGGED_VERIFY_MODE.get()
    for mode in RaggedVerifyMode:
        if value == mode.value:
            return mode
    raise ValueError(
        f"invalid SGLANG_RAGGED_VERIFY_MODE={value!r}; expected one of "
        f"{', '.join(repr(m.value) for m in RaggedVerifyMode)}"
    )


def ragged_verify_compact_enabled() -> bool:
    return read_ragged_verify_mode() == RaggedVerifyMode.COMPACT


def round_up_grid(total: int, grid: Sequence[int]) -> int:
    if not grid:
        raise ValueError("round_up_grid requires a non-empty grid")
    if total > grid[-1]:
        raise ValueError(
            f"total {total} exceeds max grid tier {grid[-1]}; "
            "the caller must reject this batch before selecting a graph tier"
        )
    index = bisect.bisect_left(grid, total)
    return grid[index]


class RaggedVerifyLayout(msgspec.Struct, frozen=True):
    verify_lens: torch.Tensor
    graph_num_tokens: int
    extend_start_loc: torch.Tensor
    qo_indptr_device: torch.Tensor
    verify_lens_cpu: Optional[list[int]] = None
    total_verify_tokens: Optional[int] = None
    qo_indptr_host: Optional[torch.Tensor] = None
    kv_indptr_host: Optional[torch.Tensor] = None
    kv_lens_host: Optional[torch.Tensor] = None
    max_q_len: Optional[int] = None
    max_kv_len: Optional[int] = None
    # Per-row upper bound (capped padded variant); rows never exceed it, so
    # dense [bs, cap] consumers stay in bounds. None = full-coverage variant.
    cap: Optional[int] = None

    def __post_init__(self) -> None:
        if self.verify_lens_cpu is None:
            return
        if not self.verify_lens_cpu:
            raise ValueError("RaggedVerifyLayout requires at least one request")
        if min(self.verify_lens_cpu) < 1:
            raise ValueError(
                f"every request must verify the anchor (verify_len >= 1), got "
                f"{self.verify_lens_cpu}"
            )
        if self.cap is not None and max(self.verify_lens_cpu) > self.cap:
            raise ValueError(
                f"capped layout has a row exceeding cap={self.cap}: "
                f"{self.verify_lens_cpu}"
            )
        if self.total_verify_tokens != sum(self.verify_lens_cpu):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} != "
                f"sum(verify_lens_cpu) {sum(self.verify_lens_cpu)}"
            )
        if not (self.total_verify_tokens <= self.graph_num_tokens):
            raise ValueError(
                f"total_verify_tokens {self.total_verify_tokens} exceeds "
                f"graph_num_tokens {self.graph_num_tokens}"
            )

    @property
    def bs(self) -> int:
        return int(self.verify_lens.shape[0])

    def request_token_offset(self, request_index: int) -> int:
        """Return the packed-token offset at a request boundary.

        TBO has to slice compact verify tensors at request boundaries rather
        than at ``request_index * draft_token_num``. Prefer host metadata so
        the eager split does not synchronize the device. Device-only layouts
        are still supported outside CUDA capture; their one-scalar D2H read is
        unavoidable because Python must construct concrete tensor views.
        """
        if not 0 <= request_index <= self.bs:
            raise IndexError(
                f"request boundary {request_index} outside layout bs={self.bs}"
            )
        if self.verify_lens_cpu is not None:
            return sum(int(v) for v in self.verify_lens_cpu[:request_index])
        if self.qo_indptr_host is not None:
            return int(self.qo_indptr_host[request_index].item())
        if self.qo_indptr_device.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "compact ragged TBO cannot resolve a live request boundary "
                "during CUDA graph capture; capture a non-TBO graph instead"
            )
        return int(self.qo_indptr_device[request_index].item())

    def slice_requests(
        self,
        *,
        start_request: int,
        end_request: int,
        start_token: Optional[int] = None,
        end_token: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        """Build a child layout for a contiguous request range.

        ``qo_indptr`` and ``extend_start_loc`` are rebased to zero for the
        child's sliced query tensor. ``end_token`` may extend past the last
        real query row for the final child, preserving a compact graph/eager
        bucket's ghost tail in ``graph_num_tokens`` without assigning those
        rows to a request.

        The returned layout owns every newly-created rebased tensor. Callers
        that capture its pointers must retain the layout for the graph's full
        lifetime; the current decode graph path intentionally captures compact
        batches without TBO because its live split boundary is dynamic.
        """
        if not 0 <= start_request <= end_request <= self.bs:
            raise ValueError(
                "invalid ragged request slice "
                f"[{start_request}, {end_request}) for bs={self.bs}"
            )

        real_start = self.request_token_offset(start_request)
        real_end = self.request_token_offset(end_request)
        start_token = real_start if start_token is None else int(start_token)
        end_token = real_end if end_token is None else int(end_token)
        if start_token != real_start:
            raise ValueError(
                f"ragged child starts at token {start_token}, but request "
                f"{start_request} starts at {real_start}"
            )
        if end_token < real_end:
            raise ValueError(
                f"ragged child ends at token {end_token}, before request "
                f"{end_request} ends at {real_end}"
            )

        verify_lens = self.verify_lens[start_request:end_request]
        qo_indptr_device = (
            self.qo_indptr_device[start_request : end_request + 1] - real_start
        )
        extend_start_loc = (
            self.extend_start_loc[start_request:end_request] - real_start
        )
        verify_lens_cpu = (
            [int(v) for v in self.verify_lens_cpu[start_request:end_request]]
            if self.verify_lens_cpu is not None
            else None
        )
        # A one-request TBO batch splits as 0 + 1. Empty children never run
        # attention, and the base layout intentionally rejects an empty host
        # request list, so represent that child with device geometry only.
        if start_request == end_request:
            verify_lens_cpu = None

        qo_indptr_host = None
        if self.qo_indptr_host is not None:
            qo_indptr_host = (
                self.qo_indptr_host[start_request : end_request + 1] - real_start
            )
        kv_indptr_host = None
        if self.kv_indptr_host is not None:
            kv_base = self.kv_indptr_host[start_request]
            kv_indptr_host = (
                self.kv_indptr_host[start_request : end_request + 1] - kv_base
            )
        kv_lens_host = (
            self.kv_lens_host[start_request:end_request]
            if self.kv_lens_host is not None
            else None
        )

        max_q_len = (
            max(verify_lens_cpu, default=0)
            if verify_lens_cpu is not None
            else self.max_q_len
        )
        max_kv_len = (
            int(kv_lens_host.max().item())
            if kv_lens_host is not None and kv_lens_host.numel() > 0
            else self.max_kv_len
        )
        return RaggedVerifyLayout(
            verify_lens=verify_lens,
            graph_num_tokens=end_token - start_token,
            extend_start_loc=extend_start_loc,
            qo_indptr_device=qo_indptr_device,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=real_end - real_start,
            qo_indptr_host=qo_indptr_host,
            kv_indptr_host=kv_indptr_host,
            kv_lens_host=kv_lens_host,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            cap=self.cap,
        )

    @classmethod
    def _assemble_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
        verify_lens_cpu: Optional[list[int]] = None,
        total_verify_tokens: Optional[int] = None,
        cap: Optional[int] = None,
    ) -> RaggedVerifyLayout:
        from sglang.kernels.ops.speculative.ragged_verify_kernels import (
            BuildQoIndptr,
        )

        verify_lens = verify_lens.to(torch.int32)
        indptr = BuildQoIndptr.execute(verify_lens=verify_lens)
        return cls(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            extend_start_loc=indptr.extend_start_loc,
            qo_indptr_device=indptr.qo_indptr,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
            cap=cap,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        verify_lens_cpu: list[int],
        total_verify_tokens: int,
        graph_num_tokens: int,
        device: torch.device,
    ) -> RaggedVerifyLayout:
        verify_lens = torch.tensor(verify_lens_cpu, dtype=torch.int32, device=device)
        return cls._assemble_device(
            verify_lens=verify_lens,
            graph_num_tokens=graph_num_tokens,
            verify_lens_cpu=verify_lens_cpu,
            total_verify_tokens=total_verify_tokens,
        )

    @classmethod
    def from_verify_lens_device(
        cls,
        *,
        verify_lens: torch.Tensor,
        graph_num_tokens: int,
    ) -> RaggedVerifyLayout:
        return cls._assemble_device(
            verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
        )

    @classmethod
    def from_verify_lens(
        cls,
        *,
        verify_lens_cpu: Sequence[int],
        device: torch.device,
        grid: Sequence[int],
        graph_num_tokens_floor: int = 0,
    ) -> RaggedVerifyLayout:
        verify_lens_list = [int(v) for v in verify_lens_cpu]
        total_verify_tokens = sum(verify_lens_list)
        bucket_input = max(total_verify_tokens, graph_num_tokens_floor)
        graph_num_tokens = round_up_grid(total=bucket_input, grid=grid)

        return cls._assemble(
            verify_lens_cpu=verify_lens_list,
            total_verify_tokens=total_verify_tokens,
            graph_num_tokens=graph_num_tokens,
            device=device,
        )

    def padded_to_bucket(
        self, *, padded_bs: int, cap: Optional[int] = None
    ) -> RaggedVerifyLayout:
        """Pad to the captured slot count. The full-coverage variant (cap=None)
        keeps sum(verify_lens) == graph_num_tokens, so padding rows (or, with
        no padding rows, the last real row) can exceed the per-request verify
        window. Dense-layout consumers (mamba/KDA) must pass cap=N: rows are
        clamped to N and the layout no longer covers the tier's leftover
        tokens -- those collapse into the consumer's ghost slot. Real tokens
        map identically under both variants."""
        from sglang.kernels.ops.speculative.ragged_verify_kernels import (
            PaddedToBucket,
        )

        padded = PaddedToBucket.execute(
            verify_lens=self.verify_lens,
            graph_num_tokens=self.graph_num_tokens,
            bs=self.bs,
            padded_bs=padded_bs,
        )
        if cap is not None:
            padded = torch.clamp(padded, max=cap)

        return RaggedVerifyLayout._assemble_device(
            verify_lens=padded,
            graph_num_tokens=self.graph_num_tokens,
            total_verify_tokens=None if cap is not None else self.graph_num_tokens,
            cap=cap,
        )


def build_capture_verify_lens(
    *,
    num_tokens: int,
    num_slots: int,
    num_draft_tokens: int,
) -> list[int]:
    if num_slots < 1 or num_tokens < num_slots:
        raise ValueError(
            f"capture layout needs 1 <= num_slots <= num_tokens, got "
            f"num_slots={num_slots}, num_tokens={num_tokens}"
        )
    if num_tokens > num_slots * num_draft_tokens:
        raise ValueError(
            f"capture layout cannot pack num_tokens={num_tokens} into "
            f"{num_slots} rows of at most {num_draft_tokens} tokens"
        )
    base = num_tokens // num_slots
    rem = num_tokens - base * num_slots
    return [base + 1] * rem + [base] * (num_slots - rem)


def resolve_ragged_verify_layout(forward_batch) -> Optional[RaggedVerifyLayout]:
    """Layout riding the batch's spec input, or None. Tolerates the runner's
    ad-hoc replay batch views, which may not carry spec_info at all."""
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is None:
        return None
    return spec_info.ragged_verify_layout


class RaggedTargetVerifyGeometry(msgspec.Struct):
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: Optional[int]


def build_ragged_target_verify_geometry(
    *,
    seq_lens: torch.Tensor,
    layout: RaggedVerifyLayout,
) -> RaggedTargetVerifyGeometry:
    cache_seqlens_int32 = (seq_lens + layout.verify_lens).to(torch.int32)
    cu_seqlens_q = layout.qo_indptr_device.to(torch.int32)
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    max_seq_len_q = (
        max(layout.verify_lens_cpu) if layout.verify_lens_cpu is not None else None
    )
    return RaggedTargetVerifyGeometry(
        cache_seqlens_int32=cache_seqlens_int32,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seq_len_q,
    )


def compute_target_verify_graph_key(
    *,
    bs: int,
    num_draft_tokens: int,
    ragged_layout: Optional[RaggedVerifyLayout],
) -> Tuple[int, int]:
    num_tokens_full_block = num_draft_tokens * bs
    if ragged_layout is None:
        return bs, num_tokens_full_block
    graph_num_tokens = ragged_layout.graph_num_tokens
    assert graph_num_tokens <= num_tokens_full_block, (
        f"ragged verify graph_num_tokens={graph_num_tokens} exceeds full block "
        f"num_draft*bs={num_tokens_full_block}"
    )
    total_verify_tokens = ragged_layout.total_verify_tokens
    if total_verify_tokens is not None:
        assert total_verify_tokens <= graph_num_tokens, (
            f"ragged verify total_verify_tokens={total_verify_tokens} exceeds the "
            f"round-up bucket graph_num_tokens={graph_num_tokens}"
        )
    return graph_num_tokens, graph_num_tokens


class VerifyExtendLengths(msgspec.Struct, frozen=True):
    seq_lens_extended: torch.Tensor
    seq_lens_cpu_extended: List[int]
    extend_seq_lens_cpu: List[int]
    num_tokens: int
    extend_start_loc: Optional[torch.Tensor]


def compute_uniform_extend_lengths(
    *,
    seq_lens: torch.Tensor,
    seq_lens_cpu: List[int],
    extend_len: int,
) -> VerifyExtendLengths:
    batch_size = len(seq_lens_cpu)
    seq_lens_extended = seq_lens + extend_len
    seq_lens_cpu_extended = [x + extend_len for x in seq_lens_cpu]
    extend_seq_lens_cpu = [extend_len] * batch_size
    num_tokens = extend_len * batch_size
    return VerifyExtendLengths(
        seq_lens_extended=seq_lens_extended,
        seq_lens_cpu_extended=seq_lens_cpu_extended,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        num_tokens=num_tokens,
        extend_start_loc=None,
    )


def compute_ragged_extend_lengths(
    *,
    seq_lens: torch.Tensor,
    seq_lens_cpu: List[int],
    ragged_layout: RaggedVerifyLayout,
) -> VerifyExtendLengths:
    extend_seq_lens_cpu = list(ragged_layout.verify_lens_cpu)
    seq_lens_extended = seq_lens + ragged_layout.verify_lens
    seq_lens_cpu_extended = [
        raw + length for raw, length in zip(seq_lens_cpu, extend_seq_lens_cpu)
    ]
    num_tokens = ragged_layout.total_verify_tokens
    extend_start_loc = ragged_layout.extend_start_loc
    return VerifyExtendLengths(
        seq_lens_extended=seq_lens_extended,
        seq_lens_cpu_extended=seq_lens_cpu_extended,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        num_tokens=num_tokens,
        extend_start_loc=extend_start_loc,
    )
