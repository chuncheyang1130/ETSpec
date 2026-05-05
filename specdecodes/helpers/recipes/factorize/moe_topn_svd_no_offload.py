"""Recipe: TopN-Expert SVD draft for Qwen3-MoE (no offload)."""

from typing import Any, Dict

from ..base_recipe import BaseRecipe
from ...factorizers.moe_topn_svd import MoETopNSVDFactorizer


_DEFAULTS: Dict[str, Any] = {
    "top_n": 16,
    "rank": 1560,
    "rank_down": None,
    "svd_device": "cuda:0",
}


class Recipe(BaseRecipe):
    def __init__(self, topn_svd_config: Dict[str, Any] | None = None):
        super().__init__()
        self.factorizer = MoETopNSVDFactorizer
        self.offloader = None

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        svd_config: Dict[str, Any] = {**_DEFAULTS}
        if draft_model is not None:
            setattr(draft_model, "topn_svd_config", svd_config)
        return {}, {"svd_config": svd_config}
