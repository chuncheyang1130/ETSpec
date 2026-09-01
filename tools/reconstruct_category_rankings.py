#!/usr/bin/env python
"""Reconstruct category rankings and a uniform selection from saved MoE moments.

This is an offline operation: it does not load the model or regenerate calibration
sequences. By default it reconstructs REAP's conditional score for every category,
then averages those category scores equally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from specdecodes.helpers.recipes.moe.expert_calibration import save_mass, save_selection
from specdecodes.models.utils.moe.base.activation_svd import balance_category_signal


def _load_moments(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def _ranking_payload(scores: Mapping[str, torch.Tensor], top_m: int):
    payload = {}
    for name, values in scores.items():
        order = torch.argsort(values.float(), descending=True, stable=True)
        ordered = [int(e) for e in order.tolist()]
        payload[name] = {
            "ordered_experts": ordered,
            "ordered_scores": [float(values[e].item()) for e in ordered],
            f"top_{top_m}_experts": ordered[:top_m],
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments", required=True, help="saved activation moment cache")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument(
        "--selection-out",
        default=None,
        help="loader-compatible selection path (default: <out-prefix>.kept_uniform<M>.json)",
    )
    parser.add_argument(
        "--selection-category",
        default="balanced",
        help="ranking used for the keep-set: 'balanced' or one recorded category",
    )
    parser.add_argument("--top-m", type=int, default=64)
    args = parser.parse_args()

    if args.top_m <= 0:
        parser.error("--top-m must be positive")

    artifact = _load_moments(args.moments)
    snapshots = artifact.get("category_snapshots")
    if not snapshots:
        raise ValueError("moment cache has no category_snapshots")
    tracked_tokens = artifact.get("tracked_tokens")
    if not tracked_tokens:
        raise ValueError("moment cache has no tracked_tokens")

    scores, per_category = balance_category_signal(
        snapshots,
        tracked_tokens,
        signal="importance",
        normalization="conditional",
    )
    num_experts = {int(values.numel()) for values in scores.values()}
    if len(num_experts) != 1:
        raise ValueError(f"inconsistent expert counts across layers: {sorted(num_experts)}")
    num_experts_value = next(iter(num_experts))
    if args.top_m > num_experts_value:
        parser.error(
            f"--top-m {args.top_m} exceeds the number of experts {num_experts_value}"
        )

    categories = list(snapshots)
    metadata = {
        "model": artifact.get("meta", {}).get("model"),
        "method": "category_balanced_conditional_reap",
        "signal": "conditional_reap_importance",
        "categories": categories,
        "tracked_tokens": {
            category: int(tracked_tokens[category]) for category in categories
        },
        "per_category_samples": artifact.get("meta", {}).get("per_category"),
        "formula": "mean_category(sum_active(g*||f_e||)/routing_count_expert_category)",
        "zero_count_policy": "score_zero",
        "enable_thinking": bool(artifact.get("meta", {}).get("enable_thinking", False)),
        "moments": args.moments,
        "top_m": int(args.top_m),
    }

    scores_path = args.out_prefix + ".scores.json"
    rankings_path = args.out_prefix + ".rankings.json"
    selection_path = (
        args.selection_out
        or args.out_prefix + f".kept_uniform{args.top_m}.json"
    )
    save_mass(scores_path, scores, metadata)

    rankings = {
        "meta": metadata,
        "categories": {
            category: _ranking_payload(values, args.top_m)
            for category, values in per_category.items()
        },
        "balanced": _ranking_payload(scores, args.top_m),
    }
    os.makedirs(os.path.dirname(rankings_path) or ".", exist_ok=True)
    with open(rankings_path, "w") as handle:
        json.dump(rankings, handle, indent=2)

    selection_category = str(args.selection_category)
    if selection_category == "balanced":
        selection_rankings = rankings["balanced"]
        selection_method = "category_balanced_conditional_reap"
        selection_formula = metadata["formula"]
    else:
        if selection_category not in rankings["categories"]:
            parser.error(
                f"--selection-category {selection_category!r} is not present in "
                f"the moment cache; choose from {categories} or 'balanced'"
            )
        selection_rankings = rankings["categories"][selection_category]
        selection_method = "category_conditional_reap"
        selection_formula = (
            "sum_active(g*||f_e||)/routing_count_expert_category"
        )
    kept = {
        name: sorted(int(e) for e in payload[f"top_{args.top_m}_experts"])
        for name, payload in selection_rankings.items()
    }
    budgets = {name: int(args.top_m) for name in kept}
    save_selection(
        selection_path,
        kept,
        budgets,
        {
            **metadata,
            "method": selection_method,
            "num_experts": num_experts_value,
            "selection_mode": "uniform",
            "selection_category": selection_category,
            "selection_formula": selection_formula,
            "uniform_budget": int(args.top_m),
            "ranking_source": rankings_path,
            "selection_rule": (
                f"first {args.top_m} experts from each {selection_category} "
                "conditional-REAP ranking; IDs stored ascending"
            ),
        },
    )

    print(f"saved conditional REAP scores -> {scores_path}")
    print(f"saved four-category and balanced rankings -> {rankings_path}")
    print(
        f"saved uniform-{args.top_m} {selection_category} selection -> "
        f"{selection_path}"
    )


if __name__ == "__main__":
    main()
