"""Recipe: TopN-Expert SVD draft for Qwen3-MoE (no offload).

Inherits the picker/tracker/draft scaffolding from the no-SVD base recipe
(`recipes/moe/moe_topn_no_offload.py`); swaps in the SVD-flavored
factorizer and overrides `generate_configurations` so the `svd_config`
returned to the build pipeline carries the SVD-specific keys (`rank`,
`rank_down`, `svd_device`) that `MoETopNSVDFactorizer` reads.

Constructed with no arguments — fixed config is hardcoded inline.
"""

from typing import Any, Dict

from ...factorizers.moe_topn_svd import MoETopNSVDFactorizer
from ..moe.moe_topn_no_offload import Recipe as MoeTopNRecipe


class Recipe(MoeTopNRecipe):
    def __init__(self):
        super().__init__()
        self.factorizer = MoETopNSVDFactorizer

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        cfg = {
            "top_n": 32,
            "rank": 1560,
            "rank_down": 1560,
            "svd_device": "cuda:0",
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_svd_config", cfg)

        target_config = {}

        return target_config, {"svd_config": cfg}
