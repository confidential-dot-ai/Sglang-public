"""Explicit all-rank development reload of the two opt-in CED graph modules."""

import importlib
import sys

from sglang.srt.environ import envs


def reload_if_requested(module_name, loaded_revision, model, cache_names):
    if envs.SGLANG_DSV41_CED_TAIL_GRAPH_REVISION.get() == loaded_revision:
        return None
    # Keep retired graphs alive: previous consumers can still hold graph output
    # views. This is a development control, not eviction during serving.
    retired = getattr(model, "_ced_retired_graphs", None)
    if retired is None:
        retired = model._ced_retired_graphs = []
    for name in cache_names:
        cache = getattr(model, name, None)
        if cache is not None:
            retired.append(cache)
            delattr(model, name)
    return importlib.reload(sys.modules[module_name])
