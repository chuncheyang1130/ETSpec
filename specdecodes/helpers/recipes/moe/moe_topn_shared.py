"""Recipe: TopN-Expert subset draft for Qwen3-MoE — shared-weight, original dtype.

Sibling of `moe_topn_int4.py` / `moe_topn_fp8.py`. Same config dict /
generate_configurations / `topn_subset_config` plumbing; the draft restructurer
installs `SharedTopNMoeBlock` instances. Those blocks copy no expert weights —
they keep only the kept expert ids and alias the target's contiguous stacked
weights, which the indexed bf16 Triton kernels read directly. Compute stays in
the original weight dtype (no quantization).

Target swap is the same as the FP8/INT4 recipes (HF MoE ->
`Qwen3MoeContiguousMoeBlock`), which the indexed kernels require to index into
stacked `[E, ...]` expert tensors.
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_contiguous import MoEContiguousRestructurer
from ...restructurer.moe_topn_shared import MoETopNSharedRestructurer


class Recipe(BaseRecipe):
    """TopN-subset MoE recipe (shared-weight draft, contiguous-weight target).

    Only the draft restructurer differs from the FP8/INT4 siblings (shared
    weights / ids-only instead of packed-and-quantized storage). The target
    swap is the same: HF MoE -> `Qwen3MoeContiguousMoeBlock`.
    """

    def __init__(self):
        super().__init__()
        self.draft_restructurer = MoETopNSharedRestructurer
        self.target_restructurer = MoEContiguousRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        """Dispatch on `kind` so one recipe can apply different swaps to target vs draft."""
        if not structure_config:
            return
        kind = structure_config.get("kind", "draft_shared_topn")
        if kind == "target_contiguous":
            self.target_restructurer.restructure_model(
                model, structure_config, dtype, device
            )
        elif kind == "draft_shared_topn":
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
            "kind": "draft_shared_topn",
            "top_n": 32,
            # How many kept experts each dropped expert distributes its routing
            # mass onto (soft top-K redirect in expert-weight-footprint space).
            "redirect_topk": 4,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        # Swap the target's HF MoE blocks for the contiguous-weight variant —
        # the indexed bf16 kernels read its stacked expert tensors directly.
        target_cfg: Dict[str, Any] = {"kind": "target_contiguous"}

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
