#!/usr/bin/env python
"""Separate the planner's two design choices: allocation vs ranking.

M3 showed the planner's removal-only plan loses to REAP at matched budget even
though they use the same action class, so the defect is in the local objective.
But "the objective" is two decisions:

  allocation  how many experts each layer removes (the cross-layer budget split)
  ranking     which experts a layer removes once its count is fixed

removal_only is (planner alloc, planner rank); reap_b50 is (uniform alloc, REAP
rank).  They differ on both axes at once.  This writes the other two corners so
the 2x2 attributes the loss.

Both rankings are read out of the same source plan, so nothing is recomputed and
the only thing that varies between corners is the decision under test.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="planner plan carrying both rankings")
    parser.add_argument("--alloc-plan", required=True, help="plan supplying per-layer removed counts")
    parser.add_argument("--category", default="coding")
    parser.add_argument("--uniform-keep", type=int, default=64)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    source = json.load(open(args.source))
    alloc = json.load(open(args.alloc_plan))

    def emit(name: str, counts: dict[str, int], key) -> None:
        layers = {}
        total_removed = 0
        for layer_name, layer in source["layers"].items():
            n = int(layer["num_experts"]) if "num_experts" in layer else len(layer["expert_usage"])
            order = sorted(range(n), key=lambda e: key(layer, e))  # least valuable first
            drop = set(order[: counts[layer_name]])
            actions = ["removed" if e in drop else "full" for e in range(n)]
            layers[layer_name] = {
                "expert_actions": actions,
                "by_action": {
                    "removed": sorted(drop),
                    "full": [e for e in range(n) if e not in drop],
                },
                "expert_usage": layer["expert_usage"],
            }
            total_removed += len(drop)
        meta = dict(source["meta"])
        meta["planner"] = name
        meta["action_names"] = ["removed", "full"]
        meta["derived_from"] = {"source": args.source, "allocation": args.alloc_plan}
        path = f"{args.out_prefix}_{name}.json"
        with open(path, "w") as handle:
            json.dump({"meta": meta, "layers": layers}, handle)
        n_layers = len(layers)
        print(f"{name:<20} removed={total_removed} ({total_removed / n_layers:.1f}/layer) -> {path}")

    planner_counts = {
        name: len(layer["by_action"].get("removed", []))
        for name, layer in alloc["layers"].items()
    }
    uniform_counts = {
        name: len(lay["expert_usage"]) - args.uniform_keep
        for name, lay in source["layers"].items()
    }

    def reap_key(layer, expert):
        return (float(layer["expert_usage"][str(expert)]["reap_mean"]), expert)

    def planner_key(layer, expert):
        return (
            float(layer["single_action_relative_distortion"][str(expert)]["removed"][args.category]),
            expert,
        )

    # Corner 3 and corner 4; the diagonal corners are already measured.
    emit("planalloc_reaprank", planner_counts, reap_key)
    emit("unialloc_planrank", uniform_counts, planner_key)


if __name__ == "__main__":
    main()
