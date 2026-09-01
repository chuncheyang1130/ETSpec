#!/usr/bin/env python
"""Build REAP expert-pruning plans in routed-functional plan format.

REAP (Router-weighted Expert Activation Pruning) drops whole experts and keeps
the survivors dense, so every plan this emits uses only the ``removed`` and
``full`` actions.  Saliency is the router-weighted expert activation norm
already recorded per layer by ``plan_routed_functional_compression.py``:

    S_j = mean_{t : j in top-k(x_t)} g_j(x_t) * || E_j(x_t) ||_2

which is ``expert_usage[j]["reap_mean"]``.  Reading it from an existing plan
keeps the baseline on exactly the same calibration tokens as the heterogeneous
plans it is compared against.
"""

from __future__ import annotations

import argparse
import json
import os


def _keep_count(num_experts: int, budget: float, top_k: int) -> int:
    keep = int(round(num_experts * budget))
    return max(top_k, min(num_experts, keep))


def build_plan(source: dict, budget: float, saliency: str, top_k: int) -> dict:
    layers = {}
    for name, layer in source["layers"].items():
        usage = layer["expert_usage"]
        num_experts = len(usage)
        keep = _keep_count(num_experts, budget, top_k)
        # Descending saliency, expert id breaking ties so never-routed experts
        # (saliency 0.0) are dropped in a reproducible order.
        ordered = sorted(usage, key=lambda e: (-float(usage[e][saliency]), int(e)))
        full = sorted(int(e) for e in ordered[:keep])
        removed = sorted(int(e) for e in ordered[keep:])
        layers[name] = {
            "expert_actions": {
                str(e): ("full" if e in set(full) else "removed")
                for e in range(num_experts)
            },
            "by_action": {"removed": removed, "full": full},
            "expert_usage": usage,
        }
    kept = sum(len(layer["by_action"]["full"]) for layer in layers.values())
    total = sum(len(layer["expert_actions"]) for layer in layers.values())
    source_meta = source.get("meta", {})
    full_parameters = int(source_meta["full_expert_parameters"])
    meta = {
        "objective": "REAP router-weighted expert activation norm",
        "planner": "reap",
        "action_names": ["removed", "full"],
        "model": source_meta.get("model"),
        "saliency": saliency,
        "saliency_definition": (
            "mean over calibration tokens routed to the expert of "
            "router_gate * ||expert_output||_2"
        ),
        "pruning_granularity": "uniform_per_layer",
        "full_expert_parameters": full_parameters,
        "retained_parameter_ratio": kept / total,
        "planned_parameters": round(full_parameters * kept / total),
        "requested_budget": budget,
        "experts_per_layer": total // len(layers),
        "kept_experts_per_layer": kept // len(layers),
        "top_k": top_k,
        "basis_category": source_meta.get("basis_category"),
        "categories": source_meta.get("categories"),
        "category_ranges": source_meta.get("category_ranges"),
        "requested_tokens_per_category": source_meta.get("requested_tokens_per_category"),
        "router_candidates": source_meta.get("router_candidates"),
        "routing_reference": "exact top-k within the retained expert pool",
        "moments": source_meta.get("moments"),
        "source_plan": source_meta.get("__source_plan__"),
    }
    return {"meta": meta, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="expert_activation/routed_functional_coding256_budget45_claude.json",
        help="plan JSON supplying per-layer expert_usage saliency statistics",
    )
    parser.add_argument(
        "--budget",
        type=int,
        nargs="+",
        required=True,
        help="retained expert-parameter percentages (compression rate = 100 - budget)",
    )
    parser.add_argument("--saliency", choices=["reap_mean", "reap_sum"], default="reap_mean")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--out-template",
        default="expert_activation/reap_coding256_budget{budget}.json",
    )
    args = parser.parse_args()

    with open(args.source) as handle:
        source = json.load(handle)
    source.setdefault("meta", {})["__source_plan__"] = args.source

    for budget in args.budget:
        plan = build_plan(source, budget / 100.0, args.saliency, args.top_k)
        out = args.out_template.format(budget=budget)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as handle:
            json.dump(plan, handle, indent=2)
            handle.write("\n")
        meta = plan["meta"]
        print(
            f"wrote {out}: {meta['kept_experts_per_layer']}/{meta['experts_per_layer']} "
            f"experts per layer, retained ratio {meta['retained_parameter_ratio']:.4f}, "
            f"compression {100 * (1 - meta['retained_parameter_ratio']):.1f}%"
        )


if __name__ == "__main__":
    main()
