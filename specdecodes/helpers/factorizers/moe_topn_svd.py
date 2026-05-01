"""Build-time factorizer: swap MoE blocks for PackedTopNSvdMoeBlock.

Step 1: this factorizer installs the *structure* (a PackedTopNSvdMoeBlock
with random-initialized packed tensors) at build time. The actual SVD-fill —
deriving `vh_shared` / `u_packed` / `down_vh_packed` / `down_u_shared` from
the target's expert weights, optionally guided by usage statistics — is a
later step.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.qwen3_moe_topn_svd import apply_packed_topn_svd_structure


class MoETopNSVDFactorizer:
    """Replace each Qwen3-MoE block in the draft with a PackedTopNSvdMoeBlock."""

    @classmethod
    def factorize_model(
        cls,
        model: nn.Module,
        svd_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        if not svd_config:
            return 0

        rank_down_cfg = svd_config.get("rank_down")
        return apply_packed_topn_svd_structure(
            model=model,
            top_n=int(svd_config.get("top_n", 32)),
            rank=int(svd_config.get("rank", 256)),
            rank_down=int(rank_down_cfg) if rank_down_cfg is not None else None,
            device=device,
            dtype=compute_dtype,
        )
