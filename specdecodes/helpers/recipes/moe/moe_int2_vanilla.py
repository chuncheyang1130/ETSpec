"""Recipe: vanilla all-INT2 Qwen3-MoE (W2A16, no draft / no speculative decoding).

The W2A16 counterpart of `moe_int4_vanilla`: swaps every HF
`Qwen3MoeSparseMoeBlock` for `Qwen3MoeStackedInt2Block` with all experts
retained. Only expert weights are quantized; attention, router, embeddings and
`lm_head` stay bf16.
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_int2 import MoEStackedInt2TargetRestructurer


class Recipe(BaseRecipe):
    """Vanilla all-INT2 MoE recipe with all experts retained."""

    def __init__(self, group_size: int = 64):
        super().__init__()
        self.restructurer = MoEStackedInt2TargetRestructurer
        self.group_size = int(group_size)

    def apply_structure(self, model, structure_config, dtype, device):
        if not structure_config:
            return
        kind = structure_config.get("kind", "target_stacked_int2")
        if kind != "target_stacked_int2":
            raise ValueError(f"Unknown structure kind: {kind!r}")
        self.restructurer.restructure_model(model, structure_config, dtype, device)

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", {})
        target_cfg: Dict[str, Any] = {
            "kind": "target_stacked_int2",
            "top_m": int(target_model.config.num_experts),
            "group_size": self.group_size,
        }
        return {"structure_config": target_cfg}, {"structure_config": {}}
