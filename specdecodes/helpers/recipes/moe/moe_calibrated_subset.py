"""
Recipe: calibrated per-layer expert-subset MoE for pure (naive) inference.

Pipeline (all at build time, on the FULL model before any swap):
  1. CALIBRATE  — run benchmark prompts through the full model and accumulate
     per-layer routing mass (prefill + optional decode).
  2. SELECT     — per layer, keep the fewest experts covering a fraction `tau`
     of routing mass (adaptive budget: diffuse layers keep more, concentrated
     layers fewer), then take those experts' global ids.
  3. RESTRUCTURE — swap each MoE block for a reduced "expert pool + redirect"
     block at the chosen precision (bf16 | int4 | fp8), routing the dropped
     experts' mass onto the kept ones via the weight-footprint redirect.

Precision-agnostic: the calibration + selection produce a precision-independent
`{layer: kept_ids}`; the restructurer picks the block backend. Runs under the
NAIVE generator (target-only, no speculative decoding) — set `method: vanilla`.

Tune via the recipe's `init_args` in the method yaml (precision, coverage_tau,
calibration dataset / size / gen tokens, redirect_topk, budget clamps).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import os

from ..base_recipe import BaseRecipe
from .expert_calibration import calibrate, save_selection, load_selection, summarize_budgets
from ...restructurer.moe_calibrated_subset import MoECalibratedSubsetRestructurer


class Recipe(BaseRecipe):
    """Calibrate -> select per-layer experts by coverage -> reduced pool+redirect blocks."""

    def __init__(
        self,
        precision: str = "int4",          # bf16 (exact, slow) | int4 (fast) | fp8 (todo)
        coverage_tau: float = 0.92,        # per-layer: keep fewest experts covering this mass
        redirect_topk: int = 8,            # redirect fan-out for dropped experts
        group_size: int = 128,             # HQQ group size (int4 only)
        calib_dataset: str = "gsm8k",      # gsm8k | humaneval | mbpp | math500
        calib_n: int = 8,                  # number of calibration prompts
        calib_gen_tokens: int = 64,        # tokens to generate per prompt (0 = prefill only)
        query_version: str = "qwen",
        min_budget: int = 8,               # floor on per-layer kept experts (>= top_k)
        max_budget: Optional[int] = None,  # cap (default num_experts -> full allowed)
        llm_path: Optional[str] = None,    # tokenizer source (else target_model.config._name_or_path)
        kept_path: Optional[str] = None,   # cache: load selection if present (skip calibration), else save here
        recalibrate: bool = False,         # force re-run calibration even if kept_path exists
    ):
        super().__init__()
        self.precision = str(precision).lower()
        self.coverage_tau = float(coverage_tau)
        self.redirect_topk = int(redirect_topk)
        self.group_size = int(group_size)
        self.calib_dataset = str(calib_dataset)
        self.calib_n = int(calib_n)
        self.calib_gen_tokens = int(calib_gen_tokens)
        self.query_version = str(query_version)
        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget) if max_budget else None
        self.llm_path = llm_path
        self.kept_path = kept_path
        self.recalibrate = bool(recalibrate)
        self.restructurer = MoECalibratedSubsetRestructurer

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        # target_model is FULL + unmodified here. Either LOAD a precomputed selection
        # (calibration done offline -> skip it) or CALIBRATE now and (optionally) save.
        E = int(target_model.config.num_experts)

        if self.kept_path and os.path.exists(self.kept_path) and not self.recalibrate:
            # Cache hit: skip calibration. Model stays raw HF -> restructurer builds
            # the reduced block straight from HF.
            kept, meta = load_selection(self.kept_path)
            if meta.get("num_experts") not in (None, E):
                raise ValueError(
                    f"selection {self.kept_path} was built for num_experts={meta.get('num_experts')} "
                    f"but this model has {E}; recalibrate (set recalibrate: true)."
                )
            logging.info("[CalibratedSubset] loaded selection from %s (%d layers, meta=%s) — "
                         "skipping calibration", self.kept_path, len(kept), meta)
            source = "hf"
        else:
            # Cache miss: build the FULL stacked block (fast GMM) and calibrate on
            # IT (not the slow raw-HF forward). The restructurer then converts these
            # same weights in-place into the reduced block (`source="stacked"`).
            from transformers import AutoTokenizer
            from specdecodes.models.utils.moe.base.apply_stacked_moe import (
                apply_stacked_target,
            )
            n_full = apply_stacked_target(target_model)
            logging.info("[CalibratedSubset] built %d full stacked blocks for calibration", n_full)
            source = "stacked"

            path = self.llm_path or getattr(target_model.config, "_name_or_path", None)
            if path is None:
                raise ValueError("cannot locate tokenizer; set recipe init_arg `llm_path`.")
            tokenizer = AutoTokenizer.from_pretrained(path)

            kept, budgets = calibrate(
                target_model, tokenizer,
                dataset=self.calib_dataset, n=self.calib_n, gen_tokens=self.calib_gen_tokens,
                tau=self.coverage_tau, min_budget=self.min_budget, max_budget=self.max_budget,
                query_version=self.query_version, device=device,
            )
            logging.info("[CalibratedSubset] %s", summarize_budgets(budgets, E))
            if self.kept_path:
                save_selection(self.kept_path, kept, budgets, {
                    "num_experts": E, "coverage_tau": self.coverage_tau,
                    "dataset": self.calib_dataset, "calib_n": self.calib_n,
                    "calib_gen_tokens": self.calib_gen_tokens,
                })
                logging.info("[CalibratedSubset] saved selection -> %s", self.kept_path)

        target_cfg: Dict[str, Any] = {
            "kind": "calibrated_subset",
            "precision": self.precision,
            "source": source,
            "kept_per_layer": kept,
            "redirect_topk": self.redirect_topk,
            "group_size": self.group_size,
        }
        return {"structure_config": target_cfg}, {"structure_config": {}}
