"""Build-time factorizer: swap MoE blocks for PackedTopNSoftFitMoeBlock.

Drop-in equivalent of `MoETopNFactorizer` (the no-SVD base) but creates
the SoftFit variant — same packed full-rank kept-expert layout, only the
redirect-builder and (at generate time, via the matching tracker) the
expert selection differ.

Pairs with `recipes/moe/moe_topn_softfit_no_offload.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.qwen3_moe_topn_softfit import (
    apply_packed_topn_softfit_structure,
)

from .moe_topn import MoETopNFactorizer


class MoETopNSoftFitFactorizer(MoETopNFactorizer):
    """Replace each Qwen3-MoE block in the draft with a `PackedTopNSoftFitMoeBlock`."""

    @classmethod
    def factorize_model(
        cls,
        model: nn.Module,
        svd_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        # `svd_config` is the channel name used by `BaseRecipe.apply_svd`.
        # On this no-SVD path we still go through it so the build pipeline
        # runs structure-swap before any later quantization / offloading.
        if not svd_config:
            return 0
        return apply_packed_topn_softfit_structure(
            model=model,
            top_n=int(svd_config.get("top_n", 32)),
            redirect_topk=int(svd_config.get("redirect_topk", 4)),
            device=device,
            dtype=compute_dtype,
        )
