"""Recipe: all-INT4 stacked MoE — draft routes to N experts, target to M (M > N).

Both models share the **same** block (`Qwen3MoeStackedInt4Block`): every
expert is HQQ-INT4 quantized once and kept resident, and the draft aliases the
target's INT4 store (no copy). The only difference is the redirecting router's
kept set — draft N, target M. Both kept sets are re-picked after each round from
the target's tracked routing mass (handled by the matching generator).

Differs from `moe_topn_int4.py` (per-subset packed draft + full bf16 target):
here the **target is also reduced** (to M, INT4) and there is no per-round
re-quantization — changing the kept set is a pure routing change.

  draft  -> `MoEStackedInt4DraftRestructurer`  (kind="draft_stacked_int4", kept=N)
  target -> `MoEStackedInt4TargetRestructurer` (kind="target_stacked_int4", kept=M)
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_int4 import (
    MoEStackedInt4TargetRestructurer,
    MoEStackedInt4DraftRestructurer,
)


class Recipe(BaseRecipe):
    """All-INT4 stacked recipe (shared INT4 store; draft N experts, target M experts)."""

    # Draft kept experts (N), target verification experts (M, > N), redirect fan-out,
    # and the HQQ group size along the contraction dim (must divide hidden & intermediate).
    TOP_N = 32
    TOP_M = 96
    REDIRECT_TOPK = 8
    GROUP_SIZE = 128
    # Accumulate prefill + accepted tokens only into the usage tracker (default),
    # vs. every processed tree token (rejected branches included). See the
    # generator's accept-aware tracking notes.
    TRACK_ACCEPTED_ONLY = False

    def __init__(self):
        super().__init__()
        self.draft_restructurer = MoEStackedInt4DraftRestructurer
        self.target_restructurer = MoEStackedInt4TargetRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        """Dispatch on `kind` so one recipe applies different swaps to target vs draft."""
        if not structure_config:
            return
        kind = structure_config.get("kind", "draft_stacked_int4")
        if kind == "target_stacked_int4":
            self.target_restructurer.restructure_model(model, structure_config, dtype, device)
        elif kind == "draft_stacked_int4":
            self.draft_restructurer.restructure_model(model, structure_config, dtype, device)
        else:
            raise ValueError(f"Unknown structure kind: {kind!r}")

    def _build_target_config(self, target_model, max_length, cpu_offload_gb, dtype, device) -> Dict[str, Any]:
        """Hook for subclasses that need a custom target device_map (offload)."""
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        # The draft config carries BOTH kept sizes: the generator reads `top_n`/`top_m`
        # off the draft model to pick the draft (N) and target (M) sets each round.
        draft_cfg = {
            "kind": "draft_stacked_int4",
            "top_n": self.TOP_N,
            "top_m": self.TOP_M,
            "redirect_topk": self.REDIRECT_TOPK,
            "group_size": self.GROUP_SIZE,
            "track_accepted_only": self.TRACK_ACCEPTED_ONLY,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }
        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        target_cfg: Dict[str, Any] = {
            "kind": "target_stacked_int4",
            "top_m": self.TOP_M,
            "redirect_topk": self.REDIRECT_TOPK,
            "group_size": self.GROUP_SIZE,
        }

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
