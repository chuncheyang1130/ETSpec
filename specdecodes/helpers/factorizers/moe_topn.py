"""Build-time factorizer: swap MoE blocks for PackedTopNMoeBlock (no SVD).

Step 1: this factorizer installs the *structure* (a `PackedTopNMoeBlock`
with random-initialized packed tensors) at build time. The actual fill —
copying kept-expert weights from the target — happens at generate-time
via `materialize_kept_from_target`, after the picker decides which
experts to keep based on tracked usage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.qwen3_moe_topn import apply_packed_topn_structure


class MoETopNFactorizer:
    """Replace each Qwen3-MoE block in the draft with a `PackedTopNMoeBlock`."""

    @classmethod
    def factorize_model(
        cls,
        model: nn.Module,
        svd_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        # `svd_config` is the channel name used by `BaseRecipe.apply_svd`. On
        # this no-SVD path we still go through the same hook so the build
        # pipeline runs structure-swap before quantization / offloading.
        if not svd_config:
            return 0
        return apply_packed_topn_structure(
            model=model,
            top_n=int(svd_config.get("top_n", 32)),
            device=device,
            dtype=compute_dtype,
        )
