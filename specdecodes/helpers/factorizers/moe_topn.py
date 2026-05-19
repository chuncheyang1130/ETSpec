"""Build-time factorizers: swap Qwen3-MoE blocks for packed top-N blocks.

Two flavors, selected by the recipe (not by `compute_dtype` — that's the
*model's* dtype, which is bf16/fp16 for both):

  * `MoETopNFactorizer`     -> bf16 `PackedTopNMoeBlock`
  * `MoETopNFP8Factorizer`  -> FP8 `PackedTopNFP8MoeBlock`

Both flavors share the same routing path (mass-weighted tracker + soft
top-K weight-space redirect + per-expert sig cache); only the expert
storage and `_expert_forward` differ. The FP8 flavor additionally swaps
in a sparse `_routing_weights` override to save a few kernel launches.

Step 1: the factorizer installs the *structure* (a packed block with
random / empty packed tensors) at build time. Step 2 — the actual fill
from kept-expert weights — happens at generate time via
`materialize_kept_from_target`, after the picker decides which experts
to keep based on tracked usage mass.

Pairs with `recipes/moe/moe_topn_no_offload.py` (bf16) and
`recipes/moe/moe_topn_fp8.py` (FP8).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
import torch

from specdecodes.models.utils.moe.qwen3_moe_topn import apply_packed_topn_structure
from specdecodes.models.utils.moe.qwen3_moe_topn_fp8 import apply_packed_topn_fp8_structure


class MoETopNFactorizer:
    """Replace each Qwen3-MoE block in the draft with a bf16 `PackedTopNMoeBlock`."""

    @classmethod
    def factorize_model(
        cls,
        model: nn.Module,
        svd_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        # `svd_config` is the channel name used by `BaseRecipe.apply_svd`. On
        # this no-SVD path we still go through it so the build pipeline runs
        # structure-swap before any later quantization / offloading.
        if not svd_config:
            return 0
        
        if compute_dtype in (torch.bfloat16, torch.float16):
            return apply_packed_topn_structure(
                model=model,
                top_n=int(svd_config.get("top_n", 32)),
                redirect_topk=int(svd_config.get("redirect_topk", 4)),
                device=device,
                dtype=compute_dtype,
            )
            
        elif compute_dtype == torch.float8_e4m3fn:
            return apply_packed_topn_fp8_structure(
                model=model,
                top_n=int(svd_config.get("top_n", 32)),
                redirect_topk=int(svd_config.get("redirect_topk", 4)),
                device=device,
                dtype=compute_dtype,
            )
