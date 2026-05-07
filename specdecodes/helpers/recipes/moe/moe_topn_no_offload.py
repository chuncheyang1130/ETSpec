"""Recipe: TopN-Expert subset draft for Qwen3-MoE — full rank, no SVD, no offload.

Pairs with `MoeTopNSubsetSDGenerator` and `MoeTopNSubsetSDDraftModel`.

This is the **base** recipe of the MoE top-N family. The SVD variants in
`recipes/factorize/moe_topn_svd_*.py` inherit from it; subclasses override
the `factorizer_cls` / `_DEFAULTS` / `_config_attr` class attributes (and,
for offload, `_build_target_config`) to add SVD compression and
CPU-expert offloading.

The recipe stores its config dict on the draft model under the attribute
named by `_config_attr` (`topn_subset_config` here, `topn_svd_config` in
the SVD subclass). The matching generator reads from the same attribute.

The config itself is fixed in `_DEFAULTS` — recipes are constructed with
no arguments. Tune by editing `_DEFAULTS` (or by subclassing and
overriding it).
"""

from typing import Any, Dict

from ..base_recipe import BaseRecipe
from ...factorizers.moe_topn import MoETopNFactorizer


_DEFAULTS: Dict[str, Any] = {
    # How many experts to keep per MoE layer (out of `num_experts` in the
    # target). The block always runs all `top_n` experts in parallel; the
    # router decides how to weight them per token.
    "top_n": 16,
    # Per-verification expert-usage logging (off by default).
    "log_expert_usage": False,
    "expert_usage_log_path": None,
}


class Recipe(BaseRecipe):
    """Base TopN-subset MoE recipe (no SVD, no offload).

    Subclasses override:
      * `factorizer_cls` — the build-time block-swap factorizer.
      * `_DEFAULTS` — the per-method fixed config dict.
      * `_config_attr` — name of the attribute set on the draft model
        (and read back by the matching generator).
      * `_build_target_config` — to add a target device_map for offloading.
    """

    def __init__(self):
        super().__init__()
        self.factorizer = MoETopNFactorizer

    def _build_config(self) -> Dict[str, Any]:
        return dict(self._DEFAULTS)

    def _build_target_config(
        self, target_model, max_length, cpu_offload_gb, dtype, device
    ) -> Dict[str, Any]:
        """Hook for subclasses that need a custom target device_map (offload)."""
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        cfg = {
            "top_n": 16,  # Default value
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }
        
        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", cfg)
            
        target_config = {}
        
        return target_config, {"svd_config": cfg}
