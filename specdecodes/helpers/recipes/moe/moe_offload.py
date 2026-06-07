"""Recipe: offload speculative decoding — two-tier cached target + shared draft.

Offload sibling of `moe_topn_shared`. The draft is the same `SharedTopNMoeBlock`
(ids-only, original dtype, no copy), but the **target** is swapped for the
two-tier `Qwen3MoeCachedMoeBlock` instead of the all-resident contiguous block:

  * target pool capacity = ``TOP_N + COLD_CAPACITY`` slots per layer;
  * the hot tier (``num_pinned = TOP_N`` slots) holds the draft's kept experts —
    the shared draft aliases these slots and is load-free / graph-stable;
  * the cold tier (``COLD_CAPACITY`` slots) streams the experts only the
    verification pass needs, LRU.

The generator (`ExpSpecSDOffloadGenerator`) reuses the standard ExpSpec flow:
its per-round `materialize_from_target` both refreshes the draft's redirect and
calls the cached target's `set_hot_tier` to pin the kept experts. So
``TOP_N`` (draft) and ``num_pinned`` (target hot tier) MUST match — they do here.

Edit the class constants to sweep the cache budget.
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_cached import MoECachedRestructurer
from ...restructurer.moe_topn_shared import MoETopNSharedRestructurer


class Recipe(BaseRecipe):
    """Two-tier cached target + shared-weight top-N draft (offload)."""

    TOP_N = 16            # draft kept experts == target hot-tier (pinned) slots
    REDIRECT_TOPK = 8     # soft redirect fan-out for dropped experts
    COLD_CAPACITY = 24    # streaming cold-tier slots; pool capacity = TOP_N + COLD_CAPACITY
    PIN_MEMORY = True

    def __init__(self):
        super().__init__()
        self.draft_restructurer = MoETopNSharedRestructurer
        self.target_restructurer = MoECachedRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        """Dispatch on `kind` (target_cached vs draft_shared_topn)."""
        if not structure_config:
            return
        kind = structure_config.get("kind", "draft_shared_topn")
        if kind == "target_cached":
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
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        draft_cfg = {
            "kind": "draft_shared_topn",
            "top_n": self.TOP_N,
            "redirect_topk": self.REDIRECT_TOPK,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }
        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        target_cfg: Dict[str, Any] = {
            "kind": "target_cached",
            "capacity": self.TOP_N + self.COLD_CAPACITY,
            "num_pinned": self.TOP_N,          # hot tier == draft kept set
            "pin_memory": self.PIN_MEMORY,
        }

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
