# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape across all runners.

    size: the per-phase capture size — what the runner iterates over.
        - prefill: num_tokens
        - decode:  bs
    stream_idx:   pdmux stream index, or None for single-stream runners.
    variant_label: optional execution variant (for example, "lora",
        "nolora", or "chunked_prefix"), or None for runners that don't
        record per-variant graphs.
    attention_variant: optional attention-graph variant label (for example,
        the DSA dual-graph "dense" / "sparse" pair dispatched via
        `attention_graph_variants`), or None when the runner has no
        attention-graph variants. Also carries DeepSeek-V4.1 bounded-prefill
        tail buckets ("tail_<tokens>"). Composes with variant_label so LoRA and
        attention variants are captured independently.
    sum_bucket: bucketed cross-DP-rank total token count this shape was
        captured against (DSPARK TARGET_VERIFY's SUM_LEN capture window,
        SGLANG_ENABLE_DSPARK_DP_VERIFY_SUM_CAPTURE), or None for every
        runner/backend that doesn't capture DP-attention verify shapes this
        way — the default keeps all of them keying on the fields above
        exactly as before this field was added.
    """

    # Tokens for prefill/ragged verify; requests for ordinary decode.
    size: int
    # PDMux stream, or None for a single stream.
    stream_idx: Optional[int] = None
    # LoRA or prefill-prefix variant; None selects the default.
    variant_label: Optional[str] = None
    # Independent attention variant (DSA dense/sparse, candidate_*); None is default.
    attention_variant: Optional[str] = None
    sum_bucket: Optional[int] = None
