#!/usr/bin/env python3
"""Re-assign a plan's action tiers by REAP saliency, keeping its cost exactly.

The source plan's per-layer histogram of actions (how many experts get each
tier) is preserved verbatim, so the parameter cost is identical layer by layer.
Only *which* expert lands in which tier changes: experts are sorted by REAP
saliency and the cheapest tiers go to the least salient experts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--saliency", default="reap_mean")
    ap.add_argument(
        "--keep-removed",
        action="store_true",
        help="leave the source plan's removed set untouched and reorder only the "
        "retained experts. Reassigning removals by saliency drops high-traffic "
        "experts and cost 20 human-eval points at b30, so this isolates the "
        "question of WHICH retained experts stay exact.",
    )
    args = ap.parse_args()

    plan = json.load(open(args.source))
    tier_order = list(plan.get("meta", {}).get("action_names", ()))
    if not tier_order or tier_order[0] != "removed" or tier_order[-1] != "full":
        raise ValueError("source plan meta.action_names must run from removed to full")
    for layer in plan["layers"].values():
        usage = layer["expert_usage"]
        actions = layer["expert_actions"]
        current = list(actions.values()) if isinstance(actions, dict) else list(actions)
        n = len(current)
        histogram = Counter(current)

        # least salient first, so the cheapest tiers are handed out first
        order = sorted(range(n), key=lambda e: (float(usage[str(e)][args.saliency]), e))
        if args.keep_removed:
            removed = [e for e in range(n) if current[e] == "removed"]
            order = removed + [e for e in order if current[e] != "removed"]

        assigned: dict[int, str] = {}
        cursor = 0
        unknown = set(histogram) - set(tier_order)
        if unknown:
            raise ValueError(f"source plan contains unknown actions: {sorted(unknown)}")
        for tier in tier_order:
            for _ in range(histogram.get(tier, 0)):
                assigned[order[cursor]] = tier
                cursor += 1
        assert cursor == n, f"assigned {cursor} of {n}"

        layer["expert_actions"] = {str(e): assigned[e] for e in range(n)}
        by_action: dict[str, list[int]] = {}
        for expert, tier in assigned.items():
            by_action.setdefault(tier, []).append(expert)
        layer["by_action"] = {k: sorted(v) for k, v in by_action.items()}

    plan["meta"]["planner"] = "planalloc_reaprank_tiers"
    plan["meta"]["derived_from"] = [args.source]
    plan["meta"]["saliency"] = args.saliency
    plan["meta"]["reaprank_keep_removed"] = bool(args.keep_removed)
    json.dump(plan, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
