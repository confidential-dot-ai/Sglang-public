"""Bound graph ownership between completed CED forwards."""

import gc

import torch

_released_load_buffers = False


def release_retired(model):
    retired = getattr(model, "_ced_retired_graphs", None)
    if retired:
        # No graph is executing on the host at this forward boundary. Finish
        # previous device consumers before releasing captured tensor storage.
        torch.cuda.synchronize()
        retired.clear()
        gc.collect()


def capture_has_room():
    from sglang.srt.model_executor.runner.ced_graph_check import agree_check_failure

    # Release loader scratch before the first CED capture. Once graph pools
    # exist, keep their allocator storage untouched throughout serving.
    global _released_load_buffers
    if not _released_load_buffers:
        torch.cuda.empty_cache()
        _released_load_buffers = True
    # A capture cannot rely on the allocator's ordinary OOM recovery.
    free, _ = torch.cuda.mem_get_info()
    return not agree_check_failure(free < 20 * 1024**3)
