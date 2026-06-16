"""
Calibration + per-layer expert selection for the calibrated-subset MoE pipeline.

Precision-agnostic core (no INT4/FP8/bf16 specifics): run benchmark prompts
through the *full* model, accumulate per-layer routing mass, then pick each
layer's kept experts by a coverage target (the diffuse layers keep more experts,
the concentrated layers fewer). The resulting `{layer_name: kept_ids}` is handed
to a precision-specific restructurer that swaps in the reduced "expert pool +
redirect" blocks.

Used by `recipes/moe/moe_calibrated_subset.py`. Reuses the routing-mass tracker
(`install_expert_usage_tracker` / `get_expert_usage`) which reruns the full
router, so it works on the raw HF MoE blocks before any swap.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from specdecodes.models.utils.moe.base.expert_usage_tracker import (
    install_expert_usage_tracker,
    remove_expert_usage_tracker,
    reset_expert_usage,
    get_expert_usage,
)


# ---------------------------------------------------------------------------
# Calibration prompts
# ---------------------------------------------------------------------------
def load_calibration_prompts(dataset: str, n: int, query_version: str = "qwen") -> List[str]:
    """Return up to `n` prompt strings from a benchmark loader (or a tiny builtin set)."""
    if dataset == "gsm8k":
        from run.pipelines.benchmarks.loader.gsm8k import load_gsm8k_dataset
        return [d["query"] for d in load_gsm8k_dataset(query_version=query_version)[:n]]
    if dataset == "humaneval":
        from run.pipelines.benchmarks.loader.humaneval import load_humaneval_dataset
        return [d["query"] for d in load_humaneval_dataset(query_version=query_version)[:n]]
    if dataset == "mbpp":
        from run.pipelines.benchmarks.loader.mbpp import load_mbpp_dataset
        return [d["query"] for d in load_mbpp_dataset(query_version=query_version)[:n]]
    if dataset == "math500":
        from run.pipelines.benchmarks.loader.math500 import load_math500_dataset
        return [d["query"] for d in load_math500_dataset(query_version=query_version)[:n]]
    # Fallback: a couple of generic prompts so calibration always has *something*.
    logging.warning("[calib] unknown dataset %r; using builtin probes", dataset)
    return [
        "Explain step by step how to compute the area of a triangle.",
        "Write a Python function that returns the n-th Fibonacci number.",
    ][:n]


# ---------------------------------------------------------------------------
# Calibration: accumulate per-layer routing mass over the full model
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_calibration(
    model,
    tokenizer,
    prompts: List[str],
    gen_tokens: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Accumulate per-layer routing mass over `prompts` (prefill + optional decode).

    `gen_tokens > 0` runs autoregressive generation so the calibration captures
    *decode-phase* usage (the kept set keeps drifting for a few hundred tokens, so
    prefill alone undercaptures); `gen_tokens == 0` is prefill-only. Returns the
    pooled per-layer mass dict from `get_expert_usage` (keyed by module name).
    """
    install_expert_usage_tracker(model)
    reset_expert_usage(model)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    for i, q in enumerate(prompts):
        msgs = [{"role": "user", "content": q}]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        if gen_tokens and gen_tokens > 0:
            model.generate(ids, max_new_tokens=int(gen_tokens), do_sample=False,
                           pad_token_id=eos)
        else:
            model(ids)
        logging.info("[calib] prompt %d/%d done", i + 1, len(prompts))
    mass = {k: v.detach().clone() for k, v in get_expert_usage(model).items()}
    remove_expert_usage_tracker(model)
    return mass


# ---------------------------------------------------------------------------
# Selection: per-layer kept experts by coverage target
# ---------------------------------------------------------------------------
@torch.no_grad()
def select_kept_by_coverage(
    mass: Dict[str, torch.Tensor],
    tau: float,
    min_budget: int,
    max_budget: Optional[int],
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """Per layer, keep the fewest experts whose mass covers a fraction `tau`.

    Concentrated layers reach `tau` with few experts; diffuse layers need many —
    so the budget is adaptive per layer. Clamped to `[min_budget, max_budget]`.

    Returns `(kept_ids, budgets)`: kept_ids[name] = sorted global expert ids (a
    python list, serializable into the structure config); budgets[name] = the M.
    """
    kept: Dict[str, List[int]] = {}
    budgets: Dict[str, int] = {}
    for name, m in mass.items():
        m = m.float()
        E = int(m.numel())
        if E == 0:
            continue
        total = m.sum().clamp_min(1e-12)
        vals, idx = torch.sort(m / total, descending=True)
        csum = torch.cumsum(vals, dim=0)
        # smallest M with cumulative coverage >= tau
        M = int((csum < float(tau)).sum().item()) + 1
        hi = int(max_budget) if max_budget else E
        M = max(int(min_budget), min(M, hi, E))
        sel, _ = torch.sort(idx[:M])
        kept[name] = sel.to(torch.long).tolist()
        budgets[name] = M
    return kept, budgets


@torch.no_grad()
def calibrate(
    model,
    tokenizer,
    *,
    dataset: str,
    n: int,
    gen_tokens: int,
    tau: float,
    min_budget: int = 8,
    max_budget: Optional[int] = None,
    query_version: str = "qwen",
    device: str = "cuda:0",
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """One-shot calibration -> per-layer kept experts. THE standalone entry point.

    Runs `n` `dataset` prompts through the (full) model, accumulates per-layer
    routing mass, and returns `(kept_per_layer, budgets)` where kept_per_layer is
    `{layer_name: [global expert ids]}` (each layer keeps the fewest experts
    covering `tau` of its mass). Pure read of the model — no structural change.
    """
    prompts = load_calibration_prompts(dataset, n, query_version)
    logging.info("[calib] %d %s prompts (gen_tokens=%d, tau=%.3f) ...", len(prompts), dataset, gen_tokens, tau)
    mass = run_calibration(model, tokenizer, prompts, gen_tokens, device)
    return select_kept_by_coverage(mass, tau, min_budget, max_budget)


def save_selection(path: str, kept: Dict[str, List[int]], budgets: Dict[str, int],
                   meta: Dict[str, Any]) -> None:
    """Persist the per-layer kept experts (+ budgets + metadata) to a JSON file.

    Lets calibration be a one-time *offline* step: calibrate once -> save here ->
    inference loads it and skips calibration entirely.
    """
    obj = {
        "kept_per_layer": {k: [int(i) for i in v] for k, v in kept.items()},
        "budgets": {k: int(v) for k, v in budgets.items()},
        "meta": meta,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_selection(path: str) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Load a saved selection -> (kept_per_layer, meta). Inverse of `save_selection`."""
    with open(path) as f:
        obj = json.load(f)
    kept = {k: [int(i) for i in v] for k, v in obj["kept_per_layer"].items()}
    return kept, obj.get("meta", {})


def summarize_budgets(budgets: Dict[str, int], num_experts: int) -> str:
    """One-line summary of the per-layer budget distribution (for logging)."""
    if not budgets:
        return "no layers"
    vals = sorted(budgets.values())
    n = len(vals)
    mean = sum(vals) / n
    return (f"{n} layers, budget min={vals[0]} max={vals[-1]} mean={mean:.1f} "
            f"median={vals[n // 2]} (of {num_experts}); total={sum(vals)} "
            f"(uniform-equivalent M={sum(vals) / n:.1f})")
