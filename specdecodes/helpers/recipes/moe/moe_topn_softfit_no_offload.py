"""Recipe: TopN-SoftFit MoE draft for Qwen3-MoE — no SVD, no offload.

Pairs with `MoeTopNSoftFitSDGenerator` and `MoeTopNSoftFitSDDraftModel`.

Inherits the build-pipeline scaffolding from the no-SVD TopN base recipe
(`recipes/moe/moe_topn_no_offload.py`); swaps in the SoftFit factorizer
and overrides `generate_configurations` so the config dict carries the
SoftFit-specific keys (`redirect_topk`) and lands on the draft under
`topn_softfit_config` (the matching generator reads from that attribute).
"""

from typing import Any, Dict

from ...factorizers.moe_topn_softfit import MoETopNSoftFitFactorizer
from .moe_topn_no_offload import Recipe as MoeTopNRecipe


class Recipe(MoeTopNRecipe):
    def __init__(self):
        super().__init__()
        self.factorizer = MoETopNSoftFitFactorizer

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        cfg = {
            "top_n": 32,
            # How many kept experts each dropped expert distributes its
            # routing mass onto. K=1 reduces to the base block's argmax
            # behavior (but in weight space, not router-row space).
            "redirect_topk": 4,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_softfit_config", cfg)

        target_config: Dict[str, Any] = {}

        return target_config, {"svd_config": cfg}
