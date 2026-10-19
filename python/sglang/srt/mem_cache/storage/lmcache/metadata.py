"""Tree-owned LMCache state, independent of the selected transport package."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LMCacheExternalFlow:
    key: Any
    lookup: Any
    total_hit: int | None = None
    local_hit_tokens: int | None = None
    load: Any = None
    load_completed: bool = False
    cancelled: bool = False
    retire_requested: bool = False
    prefix_published: bool = False
    loaded_skip_tokens: int = 0
    released_skip_tokens: int = 0
    anchor_node: Any = None
    anchor_lock: Any = None
    mamba_value: Any = None
    request_mamba_value: Any = None
    allocated_request_mamba_for_load: bool = False
    load_req: Any = None
    free_mamba_after_load: bool = False


@dataclass
class LMCachePendingStore:
    operation: Any
    node_id: Any
    lock_params: Any
