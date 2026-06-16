"""
Recipe: manual expert-subset MoE — you supply the kept experts (NO calibration).

The manual sibling of `moe_calibrated_subset` (which auto-selects by calibrating on
benchmark prompts). Here you provide the selection directly, two ways:

  * ``kept_path`` — a per-layer JSON `{layer_name: [global expert ids]}` (the format
    written by `tools/calibrate_experts.py` / `expert_calibration.save_selection`).
    Each layer gets exactly its listed experts. THIS is the per-layer path.
  * ``kept_ids``  — a single global list of expert ids applied to EVERY MoE layer
    (convenience for a uniform subset).

No model forward, no tokenizer, no benchmark — just read ids and hand them to the
restructurer, which swaps each MoE block for a reduced "expert pool + redirect" block
at the chosen precision (bf16 | int4) and installs the kept set via `set_kept`
(dropped experts' routing mass redirected onto the kept ones). Runs under the NAIVE
generator (`method: vanilla`). Set the inputs via the recipe `init_args` in the yaml.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ..base_recipe import BaseRecipe
from .expert_calibration import load_selection
from ...restructurer.moe_calibrated_subset import MoECalibratedSubsetRestructurer


class Recipe(BaseRecipe):
    """Apply a user-supplied expert subset (per-layer file or global list) — no calibration."""

    def __init__(
        self,
        precision: str = "int4",            # bf16 (exact) | int4 (fast) | fp8 (todo)
        redirect_topk: int = 8,             # redirect fan-out for dropped experts
        group_size: int = 128,              # HQQ group size (int4 only)
        kept_path: Optional[str] = None,    # per-layer JSON {layer_name: [global ids]}
        kept_ids: Optional[list] = None,    # global list applied to EVERY MoE layer
    ):
        super().__init__()
        self.precision = str(precision).lower()
        self.redirect_topk = int(redirect_topk)
        self.group_size = int(group_size)
        self.kept_path = kept_path
        self.kept_ids = [int(i) for i in kept_ids] if kept_ids else None
        self.restructurer = MoECalibratedSubsetRestructurer
        if not self.kept_path and not self.kept_ids:
            raise ValueError(
                "moe_subset recipe needs either `kept_path` (per-layer JSON) or "
                "`kept_ids` (global list of expert ids) in its init_args."
            )

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        # target_model is FULL + raw HF here (no calibration, no swap yet). Build the
        # per-layer kept map, then let the restructurer reduce each block from HF.
        E = int(target_model.config.num_experts)

        if self.kept_path:
            if not os.path.exists(self.kept_path):
                raise FileNotFoundError(f"moe_subset: kept_path not found: {self.kept_path}")
            kept, meta = load_selection(self.kept_path)
            if meta.get("num_experts") not in (None, E):
                raise ValueError(
                    f"selection {self.kept_path} was built for num_experts={meta.get('num_experts')} "
                    f"but this model has {E}."
                )
            logging.info("[MoESubset] loaded per-layer selection from %s (%d layers)",
                         self.kept_path, len(kept))
        else:
            ids = sorted(set(self.kept_ids))
            if not (0 < len(ids) <= E) or ids[0] < 0 or ids[-1] >= E:
                raise ValueError(f"kept_ids must be 0 < count <= {E} valid expert ids; got {self.kept_ids}")
            # Enumerate the (still raw HF) MoE layer names; apply the same ids to each.
            from specdecodes.models.utils.moe.base.qwen3_moe_stacked import _is_qwen3_moe_block
            names = [n for n, m in target_model.named_modules() if _is_qwen3_moe_block(m)]
            kept = {n: list(ids) for n in names}
            logging.info("[MoESubset] manual kept_ids: %d experts on each of %d layers", len(ids), len(names))

        target_cfg: Dict[str, Any] = {
            "kind": "subset",
            "precision": self.precision,
            "source": "hf",                 # raw HF -> restructurer builds the reduced block from_huggingface
            "kept_per_layer": kept,
            "redirect_topk": self.redirect_topk,
            "group_size": self.group_size,
        }
        return {"structure_config": target_cfg}, {"structure_config": {}}
