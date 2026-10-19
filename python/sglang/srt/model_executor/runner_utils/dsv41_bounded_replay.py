"""Static tail buckets and live transitions for V4.1 breakable prefill graphs."""

import msgspec

from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def tail_buckets(
    chunk_bucket: int, max_requests: int, window: int = 128
) -> tuple[int, ...]:
    limit = min(chunk_bucket, max_requests * window)
    buckets = []
    size = min(window, limit)
    while size < limit:
        buckets.append(size)
        size *= 2
    return (*buckets, limit)


def select_tail_bucket(
    *, chunk_bucket: int, max_requests: int, extend_lens, window: int = 128
) -> int:
    live = sum(min(n, window) for n in extend_lens)
    return next(
        n for n in tail_buckets(chunk_bucket, max_requests, window) if n >= live
    )


def pad_rows(tensor, size: int, value=0):
    if tensor.shape[0] > size:
        raise ValueError(f"Tail has {tensor.shape[0]} rows, exceeding bucket {size}")
    if tensor.shape[0] == size:
        return tensor
    output = tensor.new_full((size, *tensor.shape[1:]), value)
    output[: tensor.shape[0]].copy_(tensor)
    return output


@eager_on_graph(True)
def gather_tail(*tensors):
    tail = get_attn_backend().tail_forward_metadata.late_layer_tail
    return tuple(None if t is None else tail.rows(t) for t in tensors)


@eager_on_graph(True)
def enter_tail(*tensors):
    backend = get_attn_backend()
    batch = get_tc_piecewise_forward_context().forward_batch
    state = backend.bcg_tail_state
    state.saved = backend.enter_late_layer_tail(batch)
    state.full_live_rows = batch.global_num_token_non_padded_cpu
    batch.global_num_token_non_padded_cpu = sum(
        state.metadata.late_layer_tail.extend_seq_lens_cpu
    )
    tail = state.metadata.late_layer_tail
    return (tail.positions, *(None if t is None else tail.rows(t) for t in tensors))


@eager_on_graph(True)
def exit_tail():
    backend = get_attn_backend()
    batch = get_tc_piecewise_forward_context().forward_batch
    state = backend.bcg_tail_state
    backend.exit_late_layer_tail(state.saved, batch)
    batch.global_num_token_non_padded_cpu = state.full_live_rows
    state.saved = None


@eager_on_graph(True)
def scatter_tail(rows, num_tokens: int):
    tail = get_attn_backend().tail_forward_metadata.late_layer_tail
    output = rows.new_zeros((num_tokens, *rows.shape[1:]))
    output[tail.token_indices] = rows[: tail.token_indices.numel()]
    return output


class BoundedReplayGraphState:
    def __init__(self, *, backend, batch, bucket: int):
        self.bucket = bucket
        self.metadata = backend._build_late_layer_tail_metadata(
            batch, graph_bucket=bucket
        )
        self.saved = None
        self.full_live_rows = None

    def prepare(self, *, backend, batch):
        fresh = backend._build_late_layer_tail_metadata(batch, graph_bucket=self.bucket)
        old_loc = self.metadata.late_layer_tail.swa_out_cache_loc
        old_loc.copy_(fresh.late_layer_tail.swa_out_cache_loc)
        self.metadata.refresh_for_breakable_cuda_graph_replay_(fresh)
        # Cache stores in captured segments retain this address across requests.
        self.metadata.late_layer_tail = msgspec.structs.replace(
            fresh.late_layer_tail, swa_out_cache_loc=old_loc
        )
        self.metadata.core_attn_metadata.swa_out_cache_loc = old_loc
        backend.tail_forward_metadata = self.metadata
        backend.bcg_tail_state = self
