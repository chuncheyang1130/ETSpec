#!/usr/bin/env python
"""Build configurable full / SVD / removed tier assignments per MoE layer."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--category", default="coding")
    parser.add_argument("--joint-report", default=None)
    parser.add_argument(
        "--promotion-policy", choices=["importance", "reconstruction"], default="importance"
    )
    parser.add_argument("--error-key", default="gu256_down256")
    parser.add_argument("--base-full", type=int, default=32)
    parser.add_argument("--promote", type=int, default=16)
    parser.add_argument("--svd", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rankings = json.load(open(args.rankings))["categories"][args.category]
    report = json.load(open(args.joint_report)) if args.joint_report else None
    if args.promotion_policy == "reconstruction" and report is None:
        parser.error("--promotion-policy reconstruction requires --joint-report")
    full_per_layer, svd_per_layer, removed_per_layer = {}, {}, {}
    for layer, payload in rankings.items():
        ordered = [int(value) for value in payload["ordered_experts"]]
        base = ordered[: args.base_full]
        candidates = ordered[args.base_full : args.base_full + args.promote + args.svd]
        if args.promotion_policy == "importance":
            promoted = candidates[: args.promote]
        else:
            experts = report["layers"][layer]["experts"]
            scored = []
            for expert_id in candidates:
                item = experts.get(str(expert_id))
                # Never-routed experts have no measured avoided error and should not
                # consume a full slot under the compression-aware policy.
                score = 0.0 if item is None else (
                    float(item["output_energy"])
                    * float(item["relative_errors"][args.error_key])
                )
                scored.append((score, expert_id))
            promoted = [expert for _, expert in sorted(scored, reverse=True)[: args.promote]]
        promoted_set = set(promoted)
        svd = [expert for expert in candidates if expert not in promoted_set]
        full = base + promoted
        removed = [expert for expert in ordered if expert not in set(full) | set(svd)]
        if not (len(full) == args.base_full + args.promote and len(svd) == args.svd):
            raise RuntimeError(f"invalid tier counts for {layer}")
        full_per_layer[layer] = full
        svd_per_layer[layer] = svd
        removed_per_layer[layer] = removed

    payload = {
        "full_per_layer": full_per_layer,
        "svd_per_layer": svd_per_layer,
        "removed_per_layer": removed_per_layer,
        "meta": {
            "category": args.category,
            "promotion_policy": args.promotion_policy,
            "error_key": args.error_key if report is not None else None,
            "base_full": args.base_full,
            "promoted_full": args.promote,
            "full_experts": args.base_full + args.promote,
            "svd_experts": args.svd,
            "removed_experts": len(next(iter(removed_per_layer.values()))),
            "rankings": args.rankings,
            "joint_report": args.joint_report,
        },
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}: {len(full_per_layer)} layers")


if __name__ == "__main__":
    main()
