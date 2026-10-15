"""Build-time restructurer for the all-INT2 (W2A16) stacked MoE target.

Swaps HF MoE -> `Qwen3MoeStackedInt2Block` (every expert HQQ-INT2 quantized,
routing to top-M). Target-only; pairs with `recipes/moe/moe_int2_vanilla.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_int2_target


class MoEStackedInt2TargetRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the target with an all-INT2 block (kept=M)."""

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
        return apply_stacked_int2_target(
            model=model,
            kept=int(structure_config.get("top_m", 128)),
            group_size=int(structure_config.get("group_size", 64)),
            device=device,
            dtype=compute_dtype,
        )
