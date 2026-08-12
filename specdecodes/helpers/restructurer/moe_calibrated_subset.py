"""
Restructurer for the calibrated / manual expert-subset MoE pipeline (precision-pluggable).

For each MoE block it builds a **compacted** subset block that physically stacks ONLY
that layer's kept experts (chosen by `expert_calibration.select_kept_by_coverage`, or
supplied manually) — a genuinely smaller model, not the full store with restricted
routing. The kept set varies per layer (adaptive budget), so each block carries its own
`[M, ...]` store. The full router is kept, but routing selects the original model
top-`k` directly among the retained experts. A layer whose kept set is the full set
stays a full block (no compaction).

Two source paths (set by the recipe via ``structure_config["source"]``):
  * ``"stacked"`` — the model already holds full `Qwen3MoeStackedBlock`s (built so
    calibration ran on the fast GMM forward). Compact by slicing them (`from_stacked`);
    the HF sources are already gone.
  * ``"hf"`` — the model still holds raw HF blocks (manual selection or a calibration
    cache hit). Compact straight from HF (`from_huggingface`).

Precision backends: ``bf16`` (`Qwen3MoeSubsetBlock`, exact Triton GMM), ``int4``
(`Qwen3MoeSubsetInt4Block`, INT4 Triton GMM — quantizes only the kept experts), ``fp8``
(todo). Pairs with `recipes/moe/moe_calibrated_subset.py` and `recipes/moe/moe_subset.py`.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import (
    Qwen3MoeStackedBlock,
    _is_qwen3_moe_block as _is_hf_moe_block,
)
from specdecodes.models.utils.moe.base.expert_usage_tracker import _set_module_by_name


class MoECalibratedSubsetRestructurer:
    """Convert MoE blocks (HF or full-stacked) to compact retained-expert blocks."""

    @classmethod
    def restructure_model(
        cls,
        model: nn.Module,
        structure_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        if not structure_config:
            return 0
        precision = str(structure_config.get("precision", "bf16")).lower()
        source = str(structure_config.get("source", "hf")).lower()
        kept_per_layer: Dict[str, List[int]] = structure_config.get("kept_per_layer", {}) or {}
        group_size = int(structure_config.get("group_size", 128))

        if source == "stacked":
            # full stacked blocks (exact type — subclasses are not yet present)
            blocks = [(n, m) for n, m in list(model.named_modules())
                      if type(m) is Qwen3MoeStackedBlock]
        else:
            blocks = [(n, m) for n, m in list(model.named_modules()) if _is_hf_moe_block(m)]

        if precision in ("bf16", "bfloat16", "fp16", "float16"):
            cls._apply_bf16(model, blocks, kept_per_layer, source)
        elif precision == "int4":
            cls._apply_int4(model, blocks, kept_per_layer, group_size,
                            source, device, compute_dtype)
        elif precision == "fp8":
            raise NotImplementedError(
                "precision='fp8' has no stacked subset block yet — add an FP8 pool block "
                "(mirror Qwen3MoeStackedInt4Block with from_huggingface/from_stacked) "
                "and an `_apply_fp8` branch here."
            )
        else:
            raise ValueError(f"unknown precision {precision!r} (bf16|int4|fp8)")

        n_reduced = sum(1 for n, _ in blocks if cls._kept_for(kept_per_layer, n, 10 ** 9) is not None)
        logging.info(
            "[CalibratedSubset] converted %d MoE blocks (precision=%s, source=%s); "
            "%d reduced, %d kept full.",
            len(blocks), precision, source, n_reduced, len(blocks) - n_reduced,
        )
        return len(blocks)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _kept_for(kept_per_layer: Dict[str, List[int]], name: str, E: int) -> Optional[List[int]]:
        """This layer's kept ids, or None to keep the layer full (no reduction)."""
        ids = kept_per_layer.get(name)
        if ids is None:
            return None
        ids = [int(i) for i in ids]
        return ids if 0 < len(ids) < E else None

    # ------------------------------------------------------------------ bf16
    @classmethod
    def _apply_bf16(cls, model, blocks, kept_per_layer, source) -> None:
        from specdecodes.models.utils.moe.base.qwen3_moe_subset import Qwen3MoeSubsetBlock
        for i, (name, blk) in enumerate(blocks):
            ids = cls._kept_for(kept_per_layer, name, int(blk.num_experts))
            if ids is None and source == "stacked":
                # Full layer that's already a full stacked block — leave it live.
                continue
            if ids is None:
                new = Qwen3MoeStackedBlock.from_huggingface(blk)
            elif source == "stacked":
                new = Qwen3MoeSubsetBlock.from_stacked(blk, kept_ids=ids)
            else:
                new = Qwen3MoeSubsetBlock.from_huggingface(blk, kept_ids=ids)
            _set_module_by_name(model, name, new)
            # Release the original: the model ref is gone (swapped), but `blocks` still
            # pins it — drop that too so the old [E, ...] weights free now, not at return.
            blocks[i] = (name, None)
            del blk
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); gc.collect()

    # ------------------------------------------------------------------ int4
    @classmethod
    def _apply_int4(cls, model, blocks, kept_per_layer, group_size,
                    source, device, compute_dtype) -> None:
        from specdecodes.models.utils.moe.hqq.qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block
        from specdecodes.models.utils.moe.hqq.qwen3_moe_subset_int4 import Qwen3MoeSubsetInt4Block
        for i, (name, blk) in enumerate(blocks):
            E = int(blk.num_experts)
            ids = cls._kept_for(kept_per_layer, name, E)
            if source == "stacked":
                dev = device if device is not None else blk.router_weights.device
                if ids is None:                                  # full layer: full INT4 store
                    new = Qwen3MoeStackedInt4Block.from_stacked(
                        blk, kept=E, group_size=group_size,
                        device=dev, compute_dtype=compute_dtype)
                else:                                            # reduced: quantize only the kept M
                    new = Qwen3MoeSubsetInt4Block.from_stacked(
                        blk, kept_ids=ids, group_size=group_size,
                        device=dev, compute_dtype=compute_dtype)
            else:
                dev = device if device is not None else blk.experts[0].gate_proj.weight.device
                if ids is None:
                    new = Qwen3MoeStackedInt4Block.from_huggingface(
                        blk, kept=E, group_size=group_size,
                        device=dev, compute_dtype=compute_dtype)
                else:
                    new = Qwen3MoeSubsetInt4Block.from_huggingface(
                        blk, kept_ids=ids, group_size=group_size,
                        device=dev, compute_dtype=compute_dtype)
            _set_module_by_name(model, name, new)
            blocks[i] = (name, None)            # drop the list's pin so the original frees now
            del blk
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); gc.collect()
