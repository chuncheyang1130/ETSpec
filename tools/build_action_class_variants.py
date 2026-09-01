#!/usr/bin/env python
"""M2: derive matched-budget single-action-class plans from a heterogeneous plan.

M1 measures one plan and gets one number -- the aggregate correction the additive
model needs.  That is one equation in two unknowns, because a heterogeneous plan
mixes removal damage and low-rank damage and the measurement cannot say which of
them the additive model got wrong.

This tool splits the equation.  It rebuilds the source plan twice, holding each
layer's parameter budget exactly where the source plan put it and changing only
which actions are allowed:

  svd_only      actions in {full, gu*_down*}   -- no expert is ever removed
  removal_only  actions in {full, removed}     -- no expert is ever approximated

Measuring each variant with ``tools/measure_plan_additive_gap.py`` yields
``lambda_svd`` and ``lambda_rm`` separately.  Their ratio is the quantity the
break-even analysis could only bracket.

Allocation within a layer is the convex-hull greedy on the plan's own recorded
``single_action_relative_distortion``: take the downgrade with the smallest
additive damage per parameter saved until the layer's budget is met.  So each
variant is the best plan its own action class can build under the additive
objective -- which is the fair comparison, since that objective is what we are
auditing.

Holding the per-layer budget fixed is deliberate: it isolates the action class
from the cross-layer allocator, which the uniform-budget ablation already showed
is worth +6.3 HumanEval points and is not the part under suspicion.

Usage:
  python tools/build_action_class_variants.py \
      --plan expert_activation/routed_functional_coding256_budget50_claude.json \
      --out-prefix expert_activation/m2_budget50
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format="%(message)s")

_ACTION = re.compile(r"^gu(?P<gate_up>\d+)_down(?P<down>\d+)$")


def action_cost(action: str, hidden: int, intermediate: int) -> int:
    """Parameters an action keeps, in the planner's cost model (_action_spec)."""

    if action == "removed":
        return 0
    if action == "full":
        return 3 * hidden * intermediate
    match = _ACTION.fullmatch(action)
    if not match:
        raise ValueError(f"unrecognized action {action!r}")
    gate_up = int(match.group("gate_up"))
    down = int(match.group("down"))
    return gate_up * (hidden + 2 * intermediate) + down * (hidden + intermediate)


def hull_steps(damages, costs, ladder):
    """Downgrade steps along one expert's ladder, projected onto the lower hull.

    ``ladder`` is ordered most to least expensive.  Returns
    ``[(ratio, to_action, saved, extra_damage), ...]`` with non-decreasing ratio,
    so a global greedy over the steps of every expert is optimal for the
    continuous relaxation and near-optimal for the discrete problem.
    """

    points = [(costs[a], damages[a], a) for a in ladder]
    steps = []
    current_cost, current_damage, _ = points[0]
    index = 1
    while index < len(points):
        best = None
        for candidate in range(index, len(points)):
            cost, damage, action = points[candidate]
            saved = current_cost - cost
            if saved <= 0:
                continue
            ratio = (damage - current_damage) / saved
            if best is None or ratio < best[0]:
                best = (ratio, candidate, action, saved, damage - current_damage)
        if best is None:
            break
        ratio, candidate, action, saved, extra = best
        steps.append((ratio, action, saved, extra))
        current_cost, current_damage, _ = points[candidate]
        index = candidate + 1
    return steps


def plan_layer(
    layer: dict,
    ladder: list[str],
    budget: int,
    *,
    hidden: int,
    intermediate: int,
    category: str,
    min_retained_experts: int,
) -> tuple[dict, float, int]:
    """Greedy matched-budget assignment over one restricted action ladder."""

    scores = layer["single_action_relative_distortion"]
    costs = {action: action_cost(action, hidden, intermediate) for action in ladder}
    experts = sorted(scores, key=lambda value: int(value))
    full_cost = costs["full"]

    def damage(expert: str, action: str) -> float:
        value = scores[expert][action]
        return float(value[category] if isinstance(value, dict) else value)

    assignment = {expert: "full" for expert in experts}
    total = full_cost * len(experts)
    needed = total - budget
    if needed <= 0:
        return assignment, 0.0, total

    pending = []
    for expert in experts:
        damages = {action: damage(expert, action) for action in ladder}
        for order, step in enumerate(hull_steps(damages, costs, ladder)):
            pending.append((step[0], expert, order, step[1], step[2], step[3]))
    pending.sort(key=lambda item: (item[0], item[1], item[2]))

    taken = {expert: -1 for expert in experts}
    removable = len(experts) - int(min_retained_experts)
    removed = 0
    saved_total = 0
    accumulated = 0.0
    for ratio, expert, order, action, saved, extra in pending:
        if saved_total >= needed:
            break
        if taken[expert] != order - 1:
            continue                      # its earlier step was skipped
        if action == "removed" and removed >= removable:
            continue
        assignment[expert] = action
        taken[expert] = order
        saved_total += saved
        accumulated += extra
        if action == "removed":
            removed += 1

    achieved = total - saved_total
    return assignment, accumulated, achieved


def build_variant(plan: dict, ladder: list[str], *, hidden: int, intermediate: int) -> dict:
    meta = plan.get("meta", {})
    category = meta.get("basis_category", "coding")
    categories = list(meta.get("categories") or [category])
    min_retained = int(meta.get("min_retained_experts", 0) or 0)

    variant = {key: value for key, value in plan.items() if key != "layers"}
    variant["meta"] = dict(meta)
    variant["meta"]["derived_from"] = meta.get("plan_path") or "source plan"
    variant["meta"]["action_ladder"] = list(ladder)
    variant["layers"] = OrderedDict()

    shortfall = 0
    for name, layer in plan["layers"].items():
        source_budget = sum(
            action_cost(action, hidden, intermediate)
            for action in layer["expert_actions"].values()
        )
        assignment, additive, achieved = plan_layer(
            layer, ladder, source_budget,
            hidden=hidden, intermediate=intermediate, category=category,
            min_retained_experts=min_retained,
        )
        shortfall += achieved - source_budget

        by_action: dict[str, list[int]] = {}
        for expert, action in assignment.items():
            by_action.setdefault(action, []).append(int(expert))
        for values in by_action.values():
            values.sort()

        diagnostics = dict(layer.get("diagnostics", {}))
        diagnostics["raw_additive_category_relative_distortion"] = {
            category: additive
        }
        diagnostics["distortion_source"] = "action_class_variant_additive"
        diagnostics["retained_parameter_ratio"] = achieved / (
            3 * hidden * intermediate * len(assignment)
        )
        diagnostics["retained_experts"] = sum(
            1 for action in assignment.values() if action != "removed"
        )
        variant["layers"][name] = {
            "expert_actions": assignment,
            "by_action": by_action,
            "expert_usage": layer.get("expert_usage", {}),
            "single_action_relative_distortion": layer["single_action_relative_distortion"],
            "diagnostics": diagnostics,
        }
        if len(categories) > 1:
            logging.warning("multi-category plan: variant priced on %s only", category)
    variant["meta"]["budget_shortfall_parameters"] = int(shortfall)
    return variant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=768)
    args = ap.parse_args()

    with open(args.plan) as handle:
        plan = json.load(handle)

    tiers = sorted(
        {action for layer in plan["layers"].values()
         for action in layer["single_action_relative_distortion"][
             next(iter(layer["single_action_relative_distortion"]))
         ] if _ACTION.fullmatch(action)},
        key=lambda action: action_cost(action, args.hidden, args.intermediate),
        reverse=True,
    )
    ladders = {
        "svd_only": ["full"] + tiers,
        "removal_only": ["full", "removed"],
    }

    for label, ladder in ladders.items():
        variant = build_variant(
            plan, ladder, hidden=args.hidden, intermediate=args.intermediate
        )
        path = f"{args.out_prefix}_{label}.json"
        with open(path, "w") as handle:
            json.dump(variant, handle)
        source_total = sum(
            action_cost(action, args.hidden, args.intermediate)
            for layer in plan["layers"].values()
            for action in layer["expert_actions"].values()
        )
        variant_total = sum(
            action_cost(action, args.hidden, args.intermediate)
            for layer in variant["layers"].values()
            for action in layer["expert_actions"].values()
        )
        additive = [
            layer["diagnostics"]["raw_additive_category_relative_distortion"][
                plan["meta"].get("basis_category", "coding")
            ]
            for layer in variant["layers"].values()
        ]
        logging.info(
            "%-13s ladder=%-34s params=%.6f of source  mean additive=%.6g  -> %s",
            label, "/".join(ladder), variant_total / source_total,
            sum(additive) / len(additive), path,
        )


if __name__ == "__main__":
    main()
