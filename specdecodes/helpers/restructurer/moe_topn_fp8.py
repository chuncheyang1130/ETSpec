"""Build-time restructurers: swap Qwen3-MoE blocks for packed top-N blocks.

Two flavors, selected by **which recipe is used** (not by `compute_dtype`):

  * `MoETopNRestructurer`     -> bf16 `PackedTopNMoeBlock`
  * `MoETopNFP8Restructurer`  -> FP8 `PackedTopNFP8MoeBlock`

Both flavors share the same routing path (mass-weighted tracker + soft
top-K weight-space redirect + per-expert sig cache); only the expert
storage and `_expert_forward` differ. The FP8 flavor additionally swaps
in a sparse `_routing_weights` override to save a few kernel launches.

The `dtype` parameter is the *compute* dtype (bf16/fp16) for both
flavors. FP8 storage is hard-coded inside the FP8 block's
`_init_expert_weights` — the recipe doesn't pick it.

Step 1: the restructurer installs the *structure* (a packed block with
random / empty packed tensors) at build time via `apply_structure`. Step
2 — the actual fill from kept-expert weights — happens at generate time
via `materialize_kept_from_target`, after the picker decides which
experts to keep based on tracked usage mass.

Pairs with `recipes/moe/moe_topn_no_offload.py` (bf16) and
`recipes/moe/moe_topn_fp8.py` (FP8).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.qwen3_moe_topn_fp8 import apply_packed_topn_fp8_structure

class MoETopNFP8Restructurer:
    """Replace each Qwen3-MoE block in the draft with an FP8 `PackedTopNFP8MoeBlock`.

    `compute_dtype` is the bf16/fp16 tail/intermediate dtype; FP8 storage
    (e4m3) is hard-coded inside the block.
    """

    @classmethod
    def restructure_model(
        cls,
        model: nn.Module,
        structure_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        if not structure_config:
            return 0

        return apply_packed_topn_fp8_structure(
            model=model,
            top_n=int(structure_config.get("top_n", 32)),
            redirect_topk=int(structure_config.get("redirect_topk", 4)),
            device=device,
            dtype=compute_dtype,
        )
