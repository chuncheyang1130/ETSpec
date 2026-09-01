"""Rewrite the clipped propagation factors from the raw gains already recorded.

``estimate_moe_propagation.py`` median-normalizes its per-layer gains and then
clamps them into ``[--factor-clip-min, --factor-clip-max]``, defaulting to
``[0.25, 4.0]``. That guard was sized for a calibration whose gains were flat
because they were measuring a bfloat16 noise floor rather than the model. Once
the floor is subtracted the real spread is far wider, and on Qwen3-30B-A3B the
clip pins six layers at the ceiling -- layer 2 at 4.0 when its measured gain is
149x the median, a separation of 54 standard errors. The allocator therefore
sees the model's most damaging layers as merely four times worse than average
and strips them to the per-layer memory floor.

The raw gains survive in the artifact, so re-deriving the factors under a wider
bound costs no GPU time. This is a stopgap for calibrations already on disk; new
runs should pass the wider bounds to the estimator directly.
"""

import argparse
import json
import math
import statistics


def _factors(raw: dict[int, dict[str, float]], categories, clip_min, clip_max):
    """Median-normalize in log space per category, then clamp, as the estimator does."""
    out = {}
    for category in categories:
        values = [raw[layer][category] for layer in sorted(raw)]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"raw gains for {category!r} must be finite and positive")
        center = math.exp(statistics.median(math.log(v) for v in values))
        for layer in sorted(raw):
            scaled = raw[layer][category] / center
            out.setdefault(layer, {})[category] = min(max(scaled, clip_min), clip_max)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propagation-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor-clip-min", type=float, default=0.1)
    parser.add_argument("--factor-clip-max", type=float, default=200.0)
    args = parser.parse_args()

    if not (0 < args.factor_clip_min <= args.factor_clip_max):
        parser.error("factor clipping bounds must satisfy 0 < min <= max")

    with open(args.propagation_path) as handle:
        payload = json.load(handle)
    categories = list(payload["category_names"])
    names = list(payload["layers"])

    for key, source in (("svd_factors", "raw_svd_gain"), ("remove_factors", "raw_remove_gain")):
        raw = {i: payload["layers"][n][source] for i, n in enumerate(names)}
        factors = _factors(raw, categories, args.factor_clip_min, args.factor_clip_max)
        for i, name in enumerate(names):
            payload["layers"][name][key] = factors[i]
            if key == "svd_factors":
                # "factors" is the legacy single-gain field older loaders read.
                payload["layers"][name]["factors"] = dict(factors[i])
        for category in categories:
            values = [factors[i][category] for i in range(len(names))]
            at_bound = sum(
                1 for v in values
                if v <= args.factor_clip_min or v >= args.factor_clip_max
            )
            print(
                f"{key}[{category}]: {min(values):.3f}..{max(values):.1f} "
                f"({at_bound} of {len(values)} at a clip bound)"
            )

    payload["factor_clip"] = [float(args.factor_clip_min), float(args.factor_clip_max)]
    payload["reclipped_from"] = args.propagation_path
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
