"""Rank-consistent CED graph-check decisions."""


def agree_check_failure(failed):
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return failed
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    value = torch.tensor([failed], dtype=torch.int32, device="cpu")
    if group.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group.cpu_group)
    return bool(value.item())


def bitwise_equal(actual, reference):
    import torch

    return (
        actual.shape == reference.shape
        and actual.dtype == reference.dtype
        and torch.equal(
            actual.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        )
    )
