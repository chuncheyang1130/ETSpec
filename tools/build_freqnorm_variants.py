#!/usr/bin/env python
"""Rank experts by per-routed-token damage instead of corpus-average damage.

The 2x2 showed the planner's ranking, not its allocation, is what loses to REAP.
The diagnostic showed why: ``single_action_relative_distortion`` is

    sum_t ||y_full - y_plan||^2  /  sum_t ||y_full||^2

summed over *all* calibration tokens, so an expert that only fires on 1% of them
looks 100x cheaper to delete than an equally damaging expert that fires
everywhere -- even when the tokens that do route to it are wrecked.  REAP's
``reap_mean`` averages over *routed* tokens only and is therefore undiluted,
which is the whole of the 0.744 -> 0.954 jump in rank agreement once the
planner's damage is divided by routing frequency.

This re-ranks with

    damage(e) / max(topk_count(e), 1) ** alpha

alpha=1 is the fully undiluted per-routed-token rate; alpha=0 reproduces the
shipped ranking.  Allocation is copied from a reference plan so the only thing
that changes against the measured corners is the ranking.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--alloc-plan", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--category", default="coding")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    source = json.load(open(args.source))
    alloc = json.load(open(args.alloc_plan))
    layers = {}
    total = 0
    for name, layer in source["layers"].items():
        sad = layer["single_action_relative_distortion"]
        usage = layer["expert_usage"]
        n = len(usage)
        count = int(len(alloc["layers"][name]["by_action"].get("removed", [])))

        def rate(expert: int) -> tuple[float, int]:
            damage = float(sad[str(expert)]["removed"][args.category])
            freq = max(float(usage[str(expert)]["topk_count"]), 1.0)
            return (damage / freq ** args.alpha, expert)

        drop = set(sorted(range(n), key=rate)[:count])
        layers[name] = {
            "expert_actions": {str(e): ("removed" if e in drop else "full") for e in range(n)},
            "by_action": {
                "removed": sorted(drop),
                "full": [e for e in range(n) if e not in drop],
            },
            "expert_usage": usage,
        }
        total += len(drop)
    meta = dict(source["meta"])
    meta.update({
        "planner": args.name,
        "action_names": ["removed", "full"],
        "ranking": f"single_action_relative_distortion / topk_count**{args.alpha}",
        "derived_from": {"source": args.source, "allocation": args.alloc_plan},
    })
    json.dump({"meta": meta, "layers": layers}, open(args.out, "w"))
    print(f"{args.name:<24} alpha={args.alpha} removed={total} "
          f"({total / len(layers):.1f}/layer) -> {args.out}")


if __name__ == "__main__":
    main()
