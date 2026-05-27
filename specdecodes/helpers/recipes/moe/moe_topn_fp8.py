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
from ...restructurer.moe_contiguous import MoEContiguousRestructurer
from ...restructurer.moe_topn_fp8 import MoETopNFP8Restructurer


class Recipe(BaseRecipe):
    """Base TopN-subset MoE recipe (fp8 draft, contiguous-weight target).

    Same structure as the bf16 sibling — only the draft restructurer differs
    (FP8 packed storage instead of bf16). The target swap is the same: HF
    MoE -> `Qwen3MoeContiguousMoeBlock`.
    """

    def __init__(self):
        super().__init__()
        self.draft_restructurer = MoETopNFP8Restructurer
        self.target_restructurer = MoEContiguousRestructurer

    def apply_structure(self, model, structure_config, dtype, device):
        """Dispatch on `kind` so one recipe can apply different swaps to target vs draft."""
        if not structure_config:
            return
        kind = structure_config.get("kind", "draft_packed_topn")
        if kind == "target_contiguous":
            self.target_restructurer.restructure_model(
                model, structure_config, dtype, device
            )
        elif kind == "draft_packed_topn":
            self.draft_restructurer.restructure_model(
                model, structure_config, dtype, device
            )
        else:
            raise ValueError(f"Unknown structure kind: {kind!r}")

    def _build_target_config(
        self, target_model, max_length, cpu_offload_gb, dtype, device
    ) -> Dict[str, Any]:
        """Hook for subclasses that need a custom target device_map (offload)."""
        return {}

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        draft_cfg = {
            "kind": "draft_packed_topn",
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
            setattr(draft_model, "topn_subset_config", draft_cfg)

        # Swap the target's HF MoE blocks for the contiguous-weight variant
        # (drives the GMM Triton kernels instead of HF's per-expert dispatch).
        target_cfg: Dict[str, Any] = {"kind": "target_contiguous"}

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}