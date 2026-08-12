"""Recipe: TopN-Expert subset draft for Qwen3-MoE — shared-weight, original dtype.

The bf16 member of the single shared-weight SD structure (the INT4 member is
`moe_int4_sd.py`): target and draft are both `Qwen3MoeStackedBlock`,
differing only in `selected_expert_ids`. The draft restructurer installs shared-
weight draft blocks (`owns_store=False`) that copy no expert weights — they keep
only the kept expert ids and alias the target's stacked weights, which
the bf16 grouped-matmul kernels read by global expert id. Compute stays in the
original weight dtype (no quantization).

Target swap: HF MoE -> full `Qwen3MoeStackedBlock` (the draft aliases its
stacked `[E, ...]` expert tensors).
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_stacked import MoEStackedRestructurer, MoEStackedDraftRestructurer


class Recipe(BaseRecipe):
    """TopN-subset MoE recipe (shared-weight draft, stacked-weight target).

    Only the draft restructurer differs from the FP8/INT4 siblings (shared
    weights / ids-only instead of packed-and-quantized storage). The target
    swap is the same: HF MoE -> `Qwen3MoeStackedBlock`.
    """

    def __init__(self):
        super().__init__()
        self.draft_restructurer = MoEStackedDraftRestructurer
        self.target_restructurer = MoEStackedRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        """Dispatch on `kind` so one recipe can apply different swaps to target vs draft."""
        if not structure_config:
            return
        kind = structure_config.get("kind", "draft_stacked")
        if kind == "target_stacked":
            self.target_restructurer.restructure_model(
                model, structure_config, dtype, device
            )
        elif kind == "draft_stacked":
            self.draft_restructurer.restructure_model(
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
        draft_cfg = {
            "kind": "draft_stacked",
            "top_n": 32,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        # Swap the target's HF MoE blocks for the stacked-weight variant —
        # the indexed bf16 kernels read its stacked expert tensors directly.
        target_cfg: Dict[str, Any] = {"kind": "target_stacked"}

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
