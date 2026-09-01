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

TIER_ORDER = ["removed", "gu256_down128", "gu384_down256", "full"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--saliency", default="reap_mean")
    args = ap.parse_args()

    plan = json.load(open(args.source))
    for layer in plan["layers"].values():
        usage = layer["expert_usage"]
        actions = layer["expert_actions"]
        current = list(actions.values()) if isinstance(actions, dict) else list(actions)
        n = len(current)
        histogram = Counter(current)

        # least salient first, so the cheapest tiers are handed out first
        order = sorted(range(n), key=lambda e: (float(usage[str(e)][args.saliency]), e))

        assigned: dict[int, str] = {}
        cursor = 0
        for tier in TIER_ORDER:
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
    json.dump(plan, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
