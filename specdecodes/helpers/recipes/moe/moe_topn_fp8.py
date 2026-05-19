"""Recipe: TopN-Expert subset draft for Qwen3-MoE — FP8 expert storage.

FP8 sibling of `moe_topn_no_offload.py`. Inherits its config dict /
generate_configurations / `topn_subset_config` plumbing; only swaps the
restructurer to install `PackedTopNFP8MoeBlock` instances at build time.

Compute path stays bf16/fp16 (router, silu/mul tail, output tail). FP8
storage (e4m3) is hard-coded inside the block. Pairs with the
`expspec_sd_opt` preset which also wraps the draft in CUDA-graph
capture.
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_topn_fp8 import MoETopNFP8Restructurer


class Recipe(BaseRecipe):
    """Base TopN-subset MoE recipe (bf16/fp16, no SVD, no offload)."""

    def __init__(self):
        super().__init__()
        self.restructurer = MoETopNFP8Restructurer

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

        return target_config, {"structure_config": cfg}