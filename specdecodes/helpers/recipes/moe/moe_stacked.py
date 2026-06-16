"""Recipe: swap Qwen3-MoE blocks for the full stacked block (no SD).

Replaces every HF `Qwen3MoeSparseMoeBlock` with a full `Qwen3MoeStackedBlock`
(stacked `[E, ...]` expert weights + grouped-matmul forward) and runs under the
NaiveGenerator (`method: vanilla`). The pure full-precision stacked baseline —
no expert subsetting, no quantization. Pairs with `configs/methods/moe_stacked.yaml`.
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_stacked import MoEStackedRestructurer


class Recipe(BaseRecipe):
    """TopN-subset MoE recipe (shared-weight draft, stacked-weight target).

    Only the draft restructurer differs from the FP8/INT4 siblings (shared
    weights / ids-only instead of packed-and-quantized storage). The target
    swap is the same: HF MoE -> `Qwen3MoeStackedBlock`.
    """

    def __init__(self):
        super().__init__()
        self.restructurer = MoEStackedRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        """Dispatch on `kind` so one recipe can apply different swaps to target vs draft."""
        if not structure_config:
            return
        kind = structure_config.get("kind", "draft_stacked")
        if kind == "target_stacked":
            self.restructurer.restructure_model(
                model, structure_config, dtype, device
            )
        else:
            raise ValueError(f"Unknown structure kind: {kind!r}")

    def _build_target_config(
        self, target_model, max_length, cpu_offload_gb, dtype, device
    ) -> Dict[str, Any]:
        """Hook for subclasses that need a custom target device_map (offload)."""
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        draft_cfg = {}

        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        # Swap the target's HF MoE blocks for the stacked-weight variant —
        # the indexed bf16 kernels read its stacked expert tensors directly.
        target_cfg: Dict[str, Any] = {"kind": "target_stacked"}

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
