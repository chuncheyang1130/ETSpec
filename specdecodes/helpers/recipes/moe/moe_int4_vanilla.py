"""Recipe: vanilla all-INT4 Qwen3-MoE (no draft / no speculative decoding).

Target-only baseline for the INT4 family: swaps every HF `Qwen3MoeSparseMoeBlock`
for `Qwen3MoeStackedInt4Block` with `kept = num_experts` and `redirect_topk = 1`.
With all experts kept the redirect matrix is the identity, so routing reduces to
standard top_k over the full (INT4) expert set — i.e. a plain INT4 MoE forward,
no subset, no redirect. Compute runs the INT4 grouped-matmul kernels.

Pairs with `configs/methods/vanilla_int4_hqq.yaml` (method `vanilla` →
`NaiveGenerator`, no draft). Use it as the INT4 counterpart of `moe_stacked`
(the bf16 vanilla MoE baseline).
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_int4 import MoEStackedInt4TargetRestructurer


class Recipe(BaseRecipe):
    """Vanilla all-INT4 MoE recipe (target-only; all experts, identity redirect)."""

    # redirect_topk = 1 + kept = num_experts => identity redirect => standard top_k routing.
    REDIRECT_TOPK = 1
    GROUP_SIZE = 128

    def __init__(self):
        super().__init__()
        self.restructurer = MoEStackedInt4TargetRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        if not structure_config:
            return
        kind = structure_config.get("kind", "target_stacked_int4")
        if kind == "target_stacked_int4":
            self.restructurer.restructure_model(model, structure_config, dtype, device)
        else:
            raise ValueError(f"Unknown structure kind: {kind!r}")

    def _build_target_config(self, target_model, max_length, cpu_offload_gb, dtype, device) -> Dict[str, Any]:
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        draft_cfg: Dict[str, Any] = {}
        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        # All experts kept (no subset). `kept = num_experts` + redirect_topk=1 => identity.
        num_experts = int(target_model.config.num_experts)
        target_cfg: Dict[str, Any] = {
            "kind": "target_stacked_int4",
            "top_m": num_experts,
            "redirect_topk": self.REDIRECT_TOPK,
            "group_size": self.GROUP_SIZE,
        }

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
