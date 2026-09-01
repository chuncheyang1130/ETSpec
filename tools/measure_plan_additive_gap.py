#!/usr/bin/env python
"""M1: measure how wrong the planner's additive damage model is.

The planner scores an action by the routed-output error it causes *in
isolation* -- expert ``e`` takes action ``a`` while every other expert stays
full -- and then adds those per-expert numbers up.  A deployed plan turns
thousands of them on at once, so the truth is not the sum.  This tool assembles
the plan for real and measures what the sum should have been.

For each layer it replays the same calibration hidden states the plan was built
from, runs them through the full block and through the assembled compressed
block, and reports

    measured = sum_t ||y_full(t) - y_plan(t)||^2 / sum_t ||y_full(t)||^2

which is exactly the ``relative`` quantity the planner's rerouting-conditioned
scorer computes (``tools/plan_routed_functional_compression.py``, the
``base_sum / reference_sum`` in ``_build_layer_rerouting_conditioned_scores``).
The plan stores the additive model's estimate of the same number as
``diagnostics.raw_additive_category_relative_distortion``, so their ratio

    lambda_layer = measured / additive

is the per-layer correction factor the objective is missing.  ``lambda ~= 1``
would mean the additive model is sound and the accuracy gap lies elsewhere.

Usage (GPU, one layer at a time -- the model stays on CPU):

  python tools/measure_plan_additive_gap.py \
      --plan expert_activation/routed_functional_coding256_budget50_claude.json \
      --factors-dir expert_activation/routed_functional_coding256_budget50_claude.factors \
      --out expert_activation/additive_gap_budget50.json

Add ``--layers model.layers.0.mlp model.layers.24.mlp --max-tokens 8192`` for a
smoke run before committing to the full sweep.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
    _action_tiers,
    _sampled_rows,
)
from specdecodes.models.utils.moe.base.expert_usage_tracker import _set_module_by_name
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import (
    Qwen3MoeStackedBlock,
    _is_qwen3_moe_block,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _category_ids(meta: dict, row_ids: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Map selected cache rows onto their calibration category."""

    ranges = meta.get("category_ranges") or {}
    categories = list(meta.get("categories") or ranges)
    if not ranges:
        return torch.zeros(row_ids.numel(), dtype=torch.long), ["all"]
    ids = torch.full((row_ids.numel(),), -1, dtype=torch.long)
    for index, category in enumerate(categories):
        info = ranges[category]
        offset = int(info["offset"])
        stop = offset + int(info["tokens"])
        ids[(row_ids >= offset) & (row_ids < stop)] = index
    if int((ids < 0).sum()):
        raise ValueError("calibration rows fall outside every category range")
    return ids, categories


def _load_hidden(hidden_metadata: dict, name: str) -> torch.Tensor:
    rows = int(hidden_metadata["rows"][name])
    hidden = int(hidden_metadata["hidden_sizes"][name])
    path = hidden_metadata["paths"][name]
    expected = rows * hidden * 2
    if os.path.getsize(path) != expected:
        raise ValueError(f"hidden cache size mismatch for {name}")
    raw = torch.from_file(path, shared=False, size=rows * hidden, dtype=torch.uint16)
    return raw.view(torch.bfloat16).reshape(rows, hidden)


def _category_sum(values: torch.Tensor, category_ids: torch.Tensor, categories: int):
    out = torch.zeros(categories, dtype=torch.float64)
    out.index_add_(0, category_ids, values.double().cpu())
    return out


@torch.no_grad()
def measure_layer(
    stacked: Qwen3MoeStackedBlock,
    compact: Qwen3MoeRoutedFunctionalSVDSubsetBlock,
    hidden: torch.Tensor,
    row_ids: torch.Tensor,
    category_ids: torch.Tensor,
    categories: int,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (reference_energy_sum, squared_error_sum) per category."""

    dtype = stacked.gate_proj_stacked.dtype
    reference_sum = torch.zeros(categories, dtype=torch.float64)
    error_sum = torch.zeros(categories, dtype=torch.float64)
    total = int(row_ids.numel())
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        rows = row_ids[start:stop]
        batch = hidden.index_select(0, rows).to(device=device, dtype=dtype)
        reference = stacked(batch.unsqueeze(0)).reshape(stop - start, -1).float()
        approximate = compact(batch.unsqueeze(0)).reshape(stop - start, -1).float()
        chunk = category_ids[start:stop]
        reference_sum += _category_sum(reference.square().sum(1), chunk, categories)
        error_sum += _category_sum(
            (reference - approximate).square().sum(1), chunk, categories
        )
        del batch, reference, approximate
    return reference_sum, error_sum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--factors-dir", default=None,
                    help="prebuilt factor artifact; defaults to <plan>.factors if present")
    ap.add_argument("--moments-path", default=None,
                    help="build factors on the fly from this moments cache when no "
                         "factor artifact exists (M2 variant plans); uses the same "
                         "build_factor_state path the shipped artifacts came from")
    ap.add_argument("--category", default=None,
                    help="moments snapshot category (default: the plan's basis_category)")
    ap.add_argument("--router-candidates", type=int, default=None)
    ap.add_argument("--hidden-cache", default=None,
                    help="hidden-state cache dir; defaults to the plan's meta.hidden_cache_dir")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--layers", nargs="*", default=None,
                    help="restrict to these layer names (default: every planned layer)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="subsample this many calibration tokens per layer (0 = all)")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.plan) as handle:
        plan = json.load(handle)
    meta = plan.get("meta", {})

    factors_dir = args.factors_dir
    if factors_dir is None:
        guess = args.plan[:-5] + ".factors" if args.plan.endswith(".json") else None
        if guess and os.path.isfile(os.path.join(guess, "metadata.json")):
            factors_dir = guess
    factor_metadata = None
    gate_moments = up_moments = down_moments = None
    if factors_dir is not None:
        with open(os.path.join(factors_dir, "metadata.json")) as handle:
            factor_metadata = json.load(handle)
    elif args.moments_path:
        moments = torch.load(args.moments_path, map_location="cpu", weights_only=False)
        category = args.category or meta.get("basis_category", "coding")
        snapshot = moments["category_snapshots"][category]
        gate_moments = snapshot["sum_gate_input_metric"]
        up_moments = snapshot["sum_up_input_metric"]
        down_moments = snapshot["sum_g2_z2"]
    else:
        raise SystemExit(
            "pass --factors-dir to measure shipped factors, or --moments-path to "
            "build them on the fly for a variant plan"
        )

    hidden_cache = args.hidden_cache or meta.get("hidden_cache_dir")
    if not hidden_cache:
        raise SystemExit("no hidden cache: pass --hidden-cache")
    with open(os.path.join(hidden_cache, "metadata.json")) as handle:
        hidden_metadata = json.load(handle)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required: both block forwards dispatch to Triton grouped GEMMs"
        )

    logging.info("Loading %s onto CPU (one layer is moved to %s at a time)...",
                 args.model, device)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    planned = list(plan["layers"])
    names = [
        name for name, module in model.named_modules()
        if (_is_qwen3_moe_block(module) or type(module) is Qwen3MoeStackedBlock)
        and name in plan["layers"]
    ]
    if set(names) != set(planned):
        raise SystemExit(f"plan layers missing from model: {sorted(set(planned) - set(names))}")
    if args.layers:
        unknown = sorted(set(args.layers) - set(names))
        if unknown:
            raise SystemExit(f"unknown layer names: {unknown}")
        selected_names = [name for name in names if name in set(args.layers)]
    else:
        selected_names = names

    results = {}
    for order, name in enumerate(names):
        if name not in set(selected_names):
            # Still free the weights we will not touch, so CPU RAM stays flat.
            _set_module_by_name(model, name, torch.nn.Identity())
            gc.collect()
            continue

        layer = plan["layers"][name]
        source = model.get_submodule(name)
        stacked = (
            Qwen3MoeStackedBlock.from_huggingface(source)
            if _is_qwen3_moe_block(source) else source
        )
        del source
        stacked = stacked.to(device)

        full_ids, tier_specs = _action_tiers(layer)
        hidden = _load_hidden(hidden_metadata, name)
        row_ids = _sampled_rows(meta, int(hidden.shape[0]), args.seed)

        if factor_metadata is not None:
            filename = factor_metadata["layers"].get(name)
            if filename is None:
                raise SystemExit(f"factor artifact has no entry for {name}")
            factor_state = torch.load(
                os.path.join(factors_dir, filename), map_location="cpu",
                weights_only=False,
            )
            compact = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_factor_state(
                stacked, full_ids=full_ids, tier_specs=tier_specs,
                factor_state=factor_state,
            ).to(device)
            del factor_state
        else:
            # Same seeding the restructurer uses, so on-the-fly factors match the
            # ones a deployed run of this plan would build.
            layer_seed = args.seed + order * 1_000_003
            rng_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(layer_seed)
                compact = Qwen3MoeRoutedFunctionalSVDSubsetBlock.from_stacked(
                    stacked,
                    full_ids=full_ids,
                    tier_specs=tier_specs,
                    gate_moments=gate_moments[name].to(device),
                    up_moments=up_moments[name].to(device),
                    down_moments=down_moments[name].to(device),
                    calibration_hidden=hidden.index_select(0, row_ids),
                    router_candidates=int(
                        args.router_candidates
                        if args.router_candidates is not None
                        else meta.get("router_candidates", 32)
                    ),
                    routing_batch_size=args.batch_size,
                    randomized=True,
                    layer_seed=layer_seed,
                ).to(device)
        if args.max_tokens and args.max_tokens < int(row_ids.numel()):
            generator = torch.Generator().manual_seed(args.seed + 7919)
            keep = torch.randperm(int(row_ids.numel()), generator=generator)[: args.max_tokens]
            row_ids = row_ids.index_select(0, keep.sort().values)
        category_ids, categories = _category_ids(meta, row_ids)

        reference_sum, error_sum = measure_layer(
            stacked, compact, hidden, row_ids, category_ids, len(categories),
            device=device, batch_size=args.batch_size,
        )
        measured = {
            category: float(error_sum[index] / reference_sum[index].clamp_min(1e-30))
            for index, category in enumerate(categories)
        }
        additive = layer.get("diagnostics", {}).get(
            "raw_additive_category_relative_distortion", {}
        )
        ratios = {
            category: (float(measured[category] / additive[category])
                       if additive.get(category) else None)
            for category in categories
        }
        results[name] = {
            "measured_relative_distortion": measured,
            "additive_relative_distortion": {
                category: float(additive[category]) for category in categories
                if category in additive
            },
            "lambda": ratios,
            "tokens": int(row_ids.numel()),
            "retained_parameter_ratio": layer.get("diagnostics", {}).get(
                "retained_parameter_ratio"
            ),
        }
        primary = categories[0]
        logging.info(
            "[%2d/%2d] %-24s measured=%.6g additive=%.6g lambda=%s",
            order + 1, len(names), name, measured[primary],
            additive.get(primary, float("nan")),
            f"{ratios[primary]:.3f}" if ratios[primary] else "n/a",
        )

        _set_module_by_name(model, name, torch.nn.Identity())
        del stacked, compact, hidden, row_ids, category_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The objective reduces over layers with the mean (meta.layer_reduction), so
    # the aggregate lambda the planner actually experiences is the ratio of means.
    summary = {}
    categories = list(meta.get("categories") or ["coding"])
    for category in categories:
        measured = [r["measured_relative_distortion"][category] for r in results.values()
                    if category in r["measured_relative_distortion"]]
        additive = [r["additive_relative_distortion"][category] for r in results.values()
                    if category in r["additive_relative_distortion"]]
        per_layer = sorted(
            r["lambda"][category] for r in results.values()
            if r["lambda"].get(category) is not None
        )
        if not measured or not additive:
            continue
        mean_measured = sum(measured) / len(measured)
        mean_additive = sum(additive) / len(additive)
        summary[category] = {
            "layers": len(measured),
            "mean_measured": mean_measured,
            "mean_additive": mean_additive,
            "lambda_of_means": mean_measured / mean_additive if mean_additive else None,
            "lambda_median": per_layer[len(per_layer) // 2] if per_layer else None,
            "lambda_min": per_layer[0] if per_layer else None,
            "lambda_max": per_layer[-1] if per_layer else None,
        }
        logging.info(
            "\n%s: mean measured=%.6g mean additive=%.6g  ->  lambda=%.3f "
            "(per-layer median %.3f, range %.3f-%.3f over %d layers)",
            category, mean_measured, mean_additive,
            summary[category]["lambda_of_means"],
            summary[category]["lambda_median"],
            summary[category]["lambda_min"], summary[category]["lambda_max"],
            len(measured),
        )

    artifact = {
        "format": "moe_plan_additive_gap",
        "plan_path": args.plan,
        "factors_dir": factors_dir,
        "hidden_cache_dir": hidden_cache,
        "model": args.model,
        "max_tokens": int(args.max_tokens),
        "layers": results,
        "summary": summary,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(artifact, handle, indent=2)
        logging.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
