"""Recipe: TopN-Expert subset draft for Qwen3-MoE — bf16, no SVD, no offload.

Pairs with `ExpSpecSDGenerator` / `ExpSpecSDDraftModel`. The FP8 sibling
lives in `moe_topn_fp8.py`.

This is the **base** recipe of the MoE top-N family. It stores its
config dict on the draft model under the attribute `topn_subset_config`;
the matching generator reads it back.

Config is fixed in `generate_configurations` — to tune `top_n` /
`redirect_topk` either edit it here or subclass the recipe.
"""

from typing import Any, Dict

from ..base_recipe import BaseRecipe
from ...factorizers.moe_topn import MoETopNFactorizer


class Recipe(BaseRecipe):
    """Base TopN-subset MoE recipe (bf16/fp16, no SVD, no offload)."""

    def __init__(self):
        super().__init__()
        self.factorizer = MoETopNFactorizer

    def _build_target_config(
        self, target_model, max_length, cpu_offload_gb, dtype, device
    ) -> Dict[str, Any]:
        """Hook for subclasses that need a custom target device_map (offload)."""
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        cfg = {
            "top_n": 32,
            # How many kept experts each dropped expert distributes its
            # routing mass onto. K=1 reduces to the pre-merge "argmax to
            # the single most-similar kept expert" behavior, but in
            # weight-footprint space rather than router-row space.
            "redirect_topk": 4,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", cfg)

        target_config: Dict[str, Any] = {}

        return target_config, {"svd_config": cfg}
