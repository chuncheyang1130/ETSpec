"""Build-time restructurers for the all-INT4 stacked MoE family.

Both the reduced target and the draft use the **same** block class
(`Qwen3MoeStackedInt4Block`): all experts are HQQ-INT4 quantized once and
kept resident; the only difference is how many experts each routes to (target M,
draft N, M > N). The draft aliases the target's INT4 store (no copy) at bind time.

  * `MoEStackedInt4TargetRestructurer` — swaps HF MoE -> all-INT4 target block
    (quantizes every expert, owns the store), routing to top-M.
  * `MoEStackedInt4DraftRestructurer` — swaps HF MoE -> empty draft block
    (owns no store; aliases the target's store in `bind_target`), routing to top-N.

Disambiguated by `structure_config["kind"]` ("target_stacked_int4" vs
"draft_stacked_int4"). Pairs with `recipes/moe/moe_int4_sd.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from specdecodes.models.utils.moe.base.apply_stacked_moe import (
    apply_stacked_int4_target,
    apply_stacked_int4_draft,
)


class MoEStackedInt4TargetRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the target with an all-INT4 block (kept=M)."""

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
        return apply_stacked_int4_target(
            model=model,
            kept=int(structure_config.get("top_m", 96)),
            redirect_topk=int(structure_config.get("redirect_topk", 4)),
            group_size=int(structure_config.get("group_size", 128)),
            device=device,
            dtype=compute_dtype,
        )


class MoEStackedInt4DraftRestructurer:
    """Replace each HF `Qwen3MoeSparseMoeBlock` in the draft with an empty INT4 block (kept=N)."""

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
        return apply_stacked_int4_draft(
            model=model,
            kept=int(structure_config.get("top_n", 32)),
            redirect_topk=int(structure_config.get("redirect_topk", 4)),
            group_size=int(structure_config.get("group_size", 128)),
            device=device,
            dtype=compute_dtype,
        )
