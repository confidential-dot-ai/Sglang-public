"""Bound graph ownership between completed CED forwards."""

import gc
import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_released_load_buffers = False
_logged_no_room = False
_frozen = False
_logged_frozen = False


def freeze_captures():
    """Refuse every later CED capture; set identically on all TP ranks."""
    global _frozen
    _frozen = True


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

    global _logged_frozen, _logged_no_room
    if _frozen:
        if not _logged_frozen:
            logger.info(
                "CED graph capture refused while serving: a tail size not "
                "captured at startup runs eagerly"
            )
            _logged_frozen = True
        return False
    # Release loader scratch before the first CED capture. Once graph pools
    # exist, keep their allocator storage untouched throughout serving.
    global _released_load_buffers
    if not _released_load_buffers:
        torch.cuda.empty_cache()
        _released_load_buffers = True
    # A capture cannot rely on the allocator's ordinary OOM recovery.
    free, _ = torch.cuda.mem_get_info()
    min_free = envs.SGLANG_DSV41_CED_GRAPH_MIN_FREE_GB.get() * 1024**3
    if free < min_free:
        # Long-context tails leave large freed blocks (indexer tiles) cached by
        # the allocator, which the driver counts as used. Release them before
        # declining; captured graph pools keep their own storage.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
    if agree_check_failure(free < min_free):
        if not _logged_no_room:
            logger.warning(
                "CED graph capture skipped: %.1f GiB free on this rank, a rank is "
                "below SGLANG_DSV41_CED_GRAPH_MIN_FREE_GB=%.1f; tails run eagerly",
                free / 1024**3,
                min_free / 1024**3,
            )
            _logged_no_room = True
        return False
    return True
