"""Recipe: LRU-cached (offload) contiguous MoE target for Qwen3-MoE.

Offload counterpart of `moe_contiguous.py`. Target-only (no draft): swaps the
HF MoE blocks for `Qwen3MoeCachedMoeBlock` (CPU expert master + fixed-capacity
GPU LRU pool) so the same base GMM kernels run over a resident slot pool. Use
it as a `vanilla` baseline to A/B correctness, speed, and resident-memory vs
`moe_contiguous` before wiring it into the speculative-decoding offload path.

`capacity` / `pin_memory` are hardcoded here (mirroring how `top_n` is fixed in
`moe_topn_shared`); edit them to sweep the cache budget.
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_cached import MoECachedRestructurer


class Recipe(BaseRecipe):
    """Target-only LRU-cached MoE recipe (offload-friendly)."""

    # Experts kept resident on the GPU per layer (< num_experts to save VRAM).
    CAPACITY = 32
    PIN_MEMORY = True

    def __init__(self):
        super().__init__()
        self.target_restructurer = MoECachedRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        if not structure_config:
            return
        kind = structure_config.get("kind")
        if kind == "target_cached":
            self.target_restructurer.restructure_model(
                model, structure_config, dtype, device
            )
        else:
            raise ValueError(f"Unknown structure kind: {kind!r}")

    def _build_target_config(
        self, target_model, max_length, cpu_offload_gb, dtype, device
    ) -> Dict[str, Any]:
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        target_cfg: Dict[str, Any] = {
            "kind": "target_cached",
            "capacity": self.CAPACITY,
            "pin_memory": self.PIN_MEMORY,
        }
        draft_cfg: Dict[str, Any] = {}

        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
