#!/usr/bin/env python
"""Clone a compression plan with a different SVD factor calibration.

A fixed-plan factor test: every expert keeps its action, so the allocation is
identical and only how the tiered experts' factors are calibrated changes. The
restructurer reads the calibration from ``meta``, so deploying the clone builds
the new factors on the fly from ``hidden_cache_dir``.

The clone is not a priced plan: its distortion tables still describe the source
plan's factors. ``meta.factor_calibration_swap`` records the source, so a result
cannot be mistaken for a replanned arm.

Usage:
  python tools/swap_plan_factor_calibration.py \
      --plan expert_activation/nonuni_ctrl_r4_b40.json \
      --config configs/methods/moe_nonuni_ctrl_r4_b40.yaml \
      --factor-calibration topk_g2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    FACTOR_CALIBRATIONS,
)
from tools.plan_routed_functional_compression import _factor_calibration_scope

_TAGS = {"shortlist": "sl", "topk_g2": "asvd", "topk_g1": "asvd1", "topk_uniform": "asvdu"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument(
        "--factor-calibration", required=True, choices=FACTOR_CALIBRATIONS
    )
    ap.add_argument(
        "--factor-metric", choices=("diagonal", "gram"), default=None,
        help="also swap the SVD metric (default: keep the source plan's)",
    )
    ap.add_argument(
        "--gram-shrink-target", choices=("diagonal", "layer", "layer_only"),
        default=None,
    )
    ap.add_argument(
        "--config", default=None,
        help="the source plan's method config; a copy pointing at the clone is written",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    source = Path(args.plan)
    plan = json.loads(source.read_text())
    meta = plan["meta"]
    if not meta.get("hidden_cache_dir"):
        raise SystemExit(
            f"{source} records no hidden_cache_dir; swapped factors are built from it"
        )
    before = {
        key: meta.get(key)
        for key in (
            "factor_calibration", "factor_metric", "gram_shrink_target",
            "factor_calibration_scope", "factor_artifact_dir",
        )
    }
    meta["factor_calibration"] = args.factor_calibration
    if args.factor_metric is not None:
        meta["factor_metric"] = args.factor_metric
    if args.gram_shrink_target is not None:
        meta["gram_shrink_target"] = args.gram_shrink_target
    metric = str(meta.get("factor_metric", "diagonal"))
    shrink = str(meta.get("gram_shrink_target", "diagonal"))
    meta["factor_calibration_scope"] = _factor_calibration_scope(
        str(meta.get("trace_mode", "streaming")), metric, shrink,
        bool(meta.get("down_compensation", False)),
        factor_calibration=args.factor_calibration,
        # The gate/up output metric is part of the scope; dropping it here would
        # hand the clone a scope string that no longer describes its factors.
        factor_output_metric=str(meta.get("factor_output_metric") or "none"),
    )
    # Saved factors belong to the source plan, and would fail its hash check.
    meta["factor_artifact_dir"] = None
    meta["factor_calibration_swap"] = {
        "source_plan": str(source),
        "source_meta": before,
        "note": "allocation and distortion tables are the source plan's; "
                "only the deployed factors differ",
    }

    tag = _TAGS[args.factor_calibration]
    if metric == "gram":
        tag += "gram" + ("L" if shrink != "diagonal" else "")
    out = Path(args.out) if args.out else source.with_name(
        f"{source.stem}_fix{tag}.json"
    )
    if out.resolve() == source.resolve():
        raise SystemExit("refusing to overwrite the source plan")
    with open(out, "w") as handle:
        json.dump(plan, handle)
    print(f"plan   -> {out}  ({before['factor_calibration'] or 'shortlist'} -> "
          f"{args.factor_calibration}, metric {metric})")

    if args.config:
        text = Path(args.config).read_text()
        plan_ref = out.as_posix() if out.is_absolute() else f"./{out.as_posix()}"
        text, count = re.subn(
            r"(?m)^(\s*plan_path:\s*).*$", lambda m: m.group(1) + plan_ref, text
        )
        if count != 1:
            raise SystemExit(f"{args.config}: expected one plan_path line, found {count}")
        if not re.search(r"(?m)^\s*hidden_cache_dir:\s*\S", text):
            raise SystemExit(
                f"{args.config} has no hidden_cache_dir; the swapped factors are built from it"
            )
        text = re.sub(r"(?m)^\s*factors_dir:.*\n", "", text)
        config_out = Path(args.config).with_name(f"moe_{out.stem}.yaml")
        config_out.write_text(text)
        print(f"config -> {config_out}")


if __name__ == "__main__":
    main()
