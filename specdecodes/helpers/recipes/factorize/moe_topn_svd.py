"""Recipe: TopN-Expert SVD draft for Qwen3-MoE.

The actual SVD replacement happens at generate-time once expert usage has
been observed during prefill, so this recipe only stores the configuration.
The generator (`MoESvdSDGenerator`) reads `topn_svd_config` from the draft
model and invokes the factorizer.
"""

from typing import Any, Dict

from ..base_recipe import BaseRecipe
from ...factorizers.moe_topn_svd import MoETopNSVDFactorizer


class Recipe(BaseRecipe):
    DEFAULT_TOPN_SVD_CONFIG: Dict[str, Any] = {
        "top_n": 16,
        # `rank` is the SVD rank for the gate/up factorization.
        "rank": 1560,
        # `rank_down` is the SVD rank for the down factorization. If left
        # unset (None), it defaults to `rank` inside the factorizer.
        "rank_down": None,
        "svd_device": "cuda:0",
        # Round-to-round tracking driver:
        #   "tree"     - count routing on every position in the verification
        #                tree (all candidates, including rejected branches).
        #   "verified" - count routing only on accepted tokens, summed over
        #                a sliding window of the last `verified_window`
        #                accepted tokens.
        "tracking_mode": "verified",
        "verified_window": 16,
    }

    def __init__(self, topn_svd_config: Dict[str, Any] | None = None):
        super().__init__()
        self.factorizer = MoETopNSVDFactorizer
        self.topn_svd_config: Dict[str, Any] = {
            **self.DEFAULT_TOPN_SVD_CONFIG,
            **(topn_svd_config or {}),
        }

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        # Stash the config on the draft so generator-side code can read it
        # (currently informational; real SVD-fill uses these values in step 2).
        if draft_model is not None:
            setattr(draft_model, "topn_svd_config", dict(self.topn_svd_config))

        # Drive the build-time replacement: the builder calls `apply_svd`,
        # which calls `MoETopNSVDFactorizer.factorize_model(svd_config=...)`,
        # which swaps each Qwen3MoE block for a PackedTopNSvdMoeBlock with
        # random-initialized packed tensors.
        target_config: Dict[str, Any] = {}
        draft_config: Dict[str, Any] = {"svd_config": dict(self.topn_svd_config)}
        return target_config, draft_config
