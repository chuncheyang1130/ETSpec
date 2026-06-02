"""Recipe: TopN-Expert subset draft for Qwen3-MoE — INT4 (W4A16, HQQ) expert storage.

INT4 sibling of `moe_topn_fp8.py`. Inherits the same config dict /
generate_configurations / `topn_subset_config` plumbing; only swaps the
draft restructurer to install `PackedTopNINT4MoeBlock` instances at build
time.

Compute path stays bf16/fp16 (router, silu/mul tail, activations). Weights are
4-bit (HQQ quant, group-size 128 along the contraction dim) packed 2 codes per
byte; dequant happens in-register inside the two W4A16 Triton kernels. Pairs
with the same target swap as the FP8 recipe (HF MoE → `Qwen3MoeContiguousMoeBlock`).
"""

from typing import Any, Dict

from specdecodes.helpers.recipes.base_recipe import BaseRecipe
from ...restructurer.moe_contiguous import MoEContiguousRestructurer
from ...restructurer.moe_topn_int4 import MoETopNINT4Restructurer


class Recipe(BaseRecipe):
    """Base TopN-subset MoE recipe (INT4/HQQ draft, contiguous-weight target).

    Same structure as the FP8 sibling — only the draft restructurer differs
    (HQQ-INT4 packed storage instead of FP8 e4m3). The target swap is the
    same: HF MoE -> `Qwen3MoeContiguousMoeBlock`.
    """

    def __init__(self):
        super().__init__()
        self.draft_restructurer = MoETopNINT4Restructurer
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
            # HQQ group size along the contraction (input) dim. Must divide
            # both hidden_size and moe_intermediate_size, and must be even
            # (2 int4 codes per byte). 128 matches the BMM kernels' tile.
            "group_size": 128,
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_subset_config", draft_cfg)

        # Swap the target's HF MoE blocks for the contiguous-weight variant
        # (drives the GMM Triton kernels instead of HF's per-expert dispatch).
        target_cfg: Dict[str, Any] = {"kind": "target_contiguous"}

        return {"structure_config": target_cfg}, {"structure_config": draft_cfg}
