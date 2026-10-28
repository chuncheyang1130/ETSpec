"""Memory-safe surgery for routed-functional heterogeneous SVD plans."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from specdecodes.models.utils.moe.base.expert_usage_tracker import _set_module_by_name
from specdecodes.models.utils.moe.base.gemma4_adapter import prepare_moe_blocks
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeHybridSVDBlock,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import (
    Qwen3MoeStackedBlock,
    _is_qwen3_moe_block,
)


_ACTION = re.compile(r"^gu(?P<gate_up>\d+)_down(?P<down>\d+)$")


def _action_tiers(layer: dict) -> tuple[list[int], list[tuple[list[int], int, int]]]:
    by_action = layer.get("by_action")
    if not isinstance(by_action, dict):
        raise ValueError("compression layer is missing by_action")
    full = [int(value) for value in by_action.get("full", [])]
    tiers = []
    for action, values in by_action.items():
        match = _ACTION.fullmatch(str(action))
        if match and values:
            tiers.append((
                [int(value) for value in values],
                int(match.group("gate_up")),
                int(match.group("down")),
            ))
    # Stable rank ordering fixes compact local slots across repeated loads.
    tiers.sort(key=lambda item: (item[1], item[2], item[0]))
    return full, tiers


def _routing_penalties(layer: dict, scale: float) -> Dict[int, float]:
    """Per-expert router-logit discounts for one layer's tiered experts.

    Must reproduce what the planner priced, or the model runs a routing the plan
    never scored. The planner's rule is

        lam = scale * sqrt(action_damage / routed_output_energy)

    per expert and action. Both terms are recorded relative to the layer
    reference energy, so the reference cancels here and never has to be shipped.
    ``scale`` scales a quantity that is already dimensionless -- the expert's own
    relative damage amplitude -- so it needs no layer-wide statistic and stays
    comparable across layers, budgets and models.

    Per-expert rather than per-tier is not a refinement: measured on the b40
    plan, distortion spreads 460x WITHIN the largest tier while tier medians
    span 2.4x and RISE with rank, because the planner spends high ranks on the
    experts hardest to approximate. A per-tier discount carries the wrong sign.
    """

    actions = layer.get("expert_actions") or {}
    distortion = layer.get("single_action_relative_distortion") or {}
    usage = layer.get("expert_usage") or {}
    tiered = [
        expert for expert, action in actions.items()
        if action not in ("full", "removed")
    ]
    if not tiered or not float(scale):
        return {}

    # A calibrated plan carries the threshold it actually fitted, so there is no
    # formula to re-evaluate and no way for the two sides to drift. scale shrinks
    # the fitted value; the plan was priced at its own scale, so overriding it here
    # is off-plan and says so upstream.
    fitted = {
        int(expert): float(
            usage.get(expert, {}).get(f"routing_penalty_{actions[expert]}", 0.0)
        )
        for expert in tiered
    }
    if any(value > 0.0 for value in fitted.values()):
        return {
            expert: float(scale) * max(value, 0.0)
            for expert, value in fitted.items()
        }

    energies = {
        expert: float(usage.get(expert, {}).get("routed_output_energy", 0.0))
        for expert in tiered
    }
    exact = any(value > 0.0 for value in energies.values())
    if not exact:
        # A plan built before the planner recorded routed_output_energy cannot
        # reproduce the priced lam. Such a plan was also priced with NO discount,
        # so deploying one discounted is off-plan either way; the layer-median
        # form below keeps the same per-expert ordering and lets an existing plan
        # be probed, but a plan meant to be deployed discounted must be replanned
        # with --routing-penalty-scale.
        amplitudes = {}
        total = sum(
            float(stats.get("routing_square_mass", 0.0)) for stats in usage.values()
        )
        for expert in tiered:
            entry = (distortion.get(expert) or {}).get(actions[expert]) or {}
            energy = sum(float(value) for value in entry.values())
            share = float(usage.get(expert, {}).get("routing_square_mass", 0.0))
            if energy > 0.0 and share > 0.0 and total > 0.0:
                amplitudes[expert] = (energy / (share / total)) ** 0.5
        if not amplitudes:
            return {}
        median = sorted(amplitudes.values())[len(amplitudes) // 2]
        if median <= 0:
            return {}
        return {
            int(expert): float(scale) * amplitudes.get(expert, median) / median
            for expert in tiered
        }

    penalties: Dict[int, float] = {}
    for expert in tiered:
        entry = (distortion.get(expert) or {}).get(actions[expert]) or {}
        damage = sum(float(value) for value in entry.values())
        energy = energies[expert]
        # An expert the calibration never routed to has no measured damage and
        # no measured energy; it is priced at zero damage everywhere else in the
        # plan, so it is discounted by zero here for the same reason.
        penalties[int(expert)] = (
            float(scale) * (damage / energy) ** 0.5
            if damage > 0.0 and energy > 0.0 else 0.0
        )
    return penalties


def _sampled_rows(meta: dict, total_rows: int, seed: int) -> torch.Tensor:
    ranges = meta.get("category_ranges") or {}
    categories = meta.get("categories") or list(ranges)
    if not ranges:
        return torch.arange(total_rows, dtype=torch.long)
    selected = []
    for category_id, category in enumerate(categories):
        info = ranges[category]
        offset = int(info["offset"])
        count = int(info["tokens"])
        take = int(info["sampled"])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + category_id * 1_000_003)
        selected.append(
            torch.randperm(count, generator=generator)[:take].sort().values + offset
        )
    rows = torch.cat(selected)
    if int(rows.max().item()) >= total_rows:
        raise ValueError("plan calibration rows exceed the hidden-state cache")
    return rows


class MoERoutedFunctionalSVDRestructurer:
    @classmethod
    @torch.no_grad()
    def restructure_model(
        cls,
        model: nn.Module,
        structure_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> int:
        if not structure_config:
            return 0
        with open(structure_config["plan_path"]) as handle:
            plan = json.load(handle)
        factors_dir = structure_config.get("factors_dir")
        factor_metadata = None
        if factors_dir:
            with open(os.path.join(factors_dir, "metadata.json")) as handle:
                factor_metadata = json.load(handle)
            with open(structure_config["plan_path"], "rb") as handle:
                actual_hash = hashlib.sha256(handle.read()).hexdigest()
            if factor_metadata.get("plan_sha256") != actual_hash:
                raise ValueError("factor artifact was built for a different plan JSON")
            moments = gate_moments = up_moments = down_moments = None
        else:
            moments = torch.load(
                structure_config["moments_path"], map_location="cpu", weights_only=False
            )
            category = str(structure_config.get("category", plan["meta"].get("basis_category", "coding")))
            snapshot = moments["category_snapshots"][category]
            gate_moments = snapshot["sum_gate_input_metric"]
            up_moments = snapshot["sum_up_input_metric"]
            down_moments = snapshot["sum_g2_z2"]
        hidden_cache_dir = structure_config.get("hidden_cache_dir")
        hidden_metadata = None
        if hidden_cache_dir and factor_metadata is None:
            with open(os.path.join(hidden_cache_dir, "metadata.json")) as handle:
                hidden_metadata = json.load(handle)
        router_candidates = int(structure_config.get(
            "router_candidates", plan.get("meta", {}).get("router_candidates", 32)
        ))
        routing_batch_size = int(structure_config.get("routing_batch_size", 2048))
        scale_floor = float(structure_config.get("scale_floor", 1e-12))
        randomized = bool(structure_config.get("randomized", True))
        # The plan records the metric its factors were priced in. Reading it from
        # the plan rather than the method config means a Gram plan cannot be
        # deployed with diagonal factors, which would silently ship weights the
        # planner never evaluated.
        gram_metric = str(
            plan.get("meta", {}).get("factor_metric", "diagonal")
        ) == "gram"
        # The shrinkage target changes the factors, so like the metric itself it is
        # read from the PLAN, never the method config: a plan priced against the
        # layer Gram must not be deployed with diagonal-shrunk factors.
        gram_shrink_mode = str(
            plan.get("meta", {}).get("gram_shrink_target", "diagonal")
        )
        # Same rule: the compensated down target is a different matrix, so a plan
        # that priced compensated factors must not be deployed without them.
        down_compensation = bool(
            plan.get("meta", {}).get("down_compensation", False)
        )
        # Same rule: which tokens calibrate the factors, and how each is weighted,
        # changes every tiered expert's weights. Plans predating the option were
        # all calibrated on the router shortlist.
        factor_calibration = str(
            plan.get("meta", {}).get("factor_calibration", "shortlist")
        )
        # Plans written before the gate/up output metric existed have no key,
        # and "none" reproduces their factors exactly.
        factor_output_metric = str(
            plan.get("meta", {}).get("factor_output_metric") or "none"
        )
        seed = int(structure_config.get("seed", 0))
        # A deployment knob, deliberately NOT read from the plan: the discount
        # changes no factor, only how the router prices them, so one factor
        # artifact serves an entire scale sweep. scale = 0 is bit-identical to the
        # undiscounted block.
        # The plan's own priced value by default, so a plan priced with a
        # discount deploys with it. An explicit config value overrides, which is
        # how an scale sweep reuses one factor artifact -- off-plan by definition,
        # so it says so.
        planned_scale = float(plan.get("meta", {}).get("routing_penalty_scale", 0.0))
        override = structure_config.get("routing_penalty_scale")
        routing_penalty_scale = planned_scale if override is None else float(override)
        if override is not None and float(override) != planned_scale:
            logging.warning(
                "routing_penalty_scale=%s overrides the plan's priced %s; the "
                "allocation was scored under the plan's value",
                float(override), planned_scale,
            )

        prepare_moe_blocks(model)    # Gemma 4: give each layer an MoE block to replace
        names = [
            name for name, module in model.named_modules()
            if (_is_qwen3_moe_block(module) or type(module) is Qwen3MoeStackedBlock)
            and name in plan["layers"]
        ]
        if set(names) != set(plan["layers"]):
            missing = sorted(set(plan["layers"]) - set(names))
            raise ValueError(f"compression plan layers not found in model: {missing}")

        for layer_index, name in enumerate(names):
            source = model.get_submodule(name)
            if _is_qwen3_moe_block(source):
                stacked = Qwen3MoeStackedBlock.from_huggingface(source)
                del source
            else:
                stacked = source
            full_ids, tier_specs = _action_tiers(plan["layers"][name])
            covered = full_ids + [value for ids, _, _ in tier_specs for value in ids]
            removed = [int(value) for value in plan["layers"][name]["by_action"].get("removed", [])]
            if sorted(covered + removed) != list(range(int(stacked.num_experts))):
                raise ValueError(f"{name} does not assign every original expert exactly once")

            calibration_hidden = None
            if hidden_metadata is not None:
                rows = int(hidden_metadata["rows"][name])
                hidden = int(hidden_metadata["hidden_sizes"][name])
                path = hidden_metadata["paths"][name]
                expected = rows * hidden * 2
                if os.path.getsize(path) != expected:
                    raise ValueError(f"hidden cache size mismatch for {name}")
                raw = torch.from_file(
                    path, shared=False, size=rows * hidden, dtype=torch.uint16
                )
                all_hidden = raw.view(torch.bfloat16).reshape(rows, hidden)
                row_ids = _sampled_rows(plan.get("meta", {}), rows, seed)
                calibration_hidden = all_hidden.index_select(0, row_ids)
                del all_hidden, raw

            layer_seed = seed + layer_index * 1_000_003
            stacked_device = stacked.gate_proj_stacked.device
            rng_devices = (
                [stacked_device.index if stacked_device.index is not None else torch.cuda.current_device()]
                if stacked_device.type == "cuda" else []
            )
            if factor_metadata is not None:
                filename = factor_metadata["layers"].get(name)
                if filename is None:
                    raise ValueError(f"factor artifact has no entry for {name}")
                factor_state = torch.load(
                    os.path.join(factors_dir, filename),
                    map_location="cpu",
                    weights_only=False,
                )
                compact = Qwen3MoeHybridSVDBlock.from_factor_state(
                    stacked,
                    full_ids=full_ids,
                    tier_specs=tier_specs,
                    factor_state=factor_state,
                )
                del factor_state
            else:
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(layer_seed)
                    compact = Qwen3MoeHybridSVDBlock.from_stacked(
                        stacked,
                        full_ids=full_ids,
                        tier_specs=tier_specs,
                        gate_moments=gate_moments[name],
                        up_moments=up_moments[name],
                        down_moments=down_moments[name],
                        calibration_hidden=calibration_hidden,
                        router_candidates=router_candidates,
                        routing_batch_size=routing_batch_size,
                        scale_floor=scale_floor,
                        randomized=randomized,
                        layer_seed=layer_seed,
                        gram_metric=gram_metric,
                        gram_shrink_mode=gram_shrink_mode,
                        down_compensation=down_compensation,
                        factor_calibration=factor_calibration,
                        factor_output_metric=factor_output_metric,
                    )
            if str(structure_config.get("tier_output_ablation", "none")) == "zero":
                # Every low-rank expert's output factor lives in this one store,
                # so zeroing it makes each tiered expert return exactly 0 while
                # the router still selects it at its original weight.
                compact.down_output_store.data.zero_()
                if layer_index == 0:
                    logging.warning(
                        "tier_output_ablation=zero: tiered experts keep their "
                        "router slot but output 0 (ablation, not a deployment)"
                    )
            tier_penalty = structure_config.get("tier_routing_penalty")
            if tier_penalty:
                layer_actions = plan["layers"][name]["expert_actions"]
                tier_names = {a for a in layer_actions.values() if a not in ("full", "removed")}
                unknown = set(tier_penalty) - set(plan["meta"]["action_names"])
                if unknown:
                    raise ValueError(f"tier_routing_penalty names unknown actions {sorted(unknown)}")
                compact.set_routing_penalty({
                    int(expert): float(tier_penalty.get(action, 0.0))
                    for expert, action in layer_actions.items()
                    if action in tier_names and float(tier_penalty.get(action, 0.0))
                })
                if layer_index == 0:
                    logging.warning("tier_routing_penalty=%s (off-plan routing)", tier_penalty)
            removed_policy = str(structure_config.get("removed_policy", "reroute"))
            if removed_policy != "reroute":
                compact.set_removed_policy(removed_policy)
                if layer_index == 0:
                    logging.warning("removed_policy=%s (removed experts' slots are dropped, not rerouted)",
                                    removed_policy)
            if routing_penalty_scale:
                compact.set_routing_penalty(
                    _routing_penalties(plan["layers"][name], routing_penalty_scale)
                )
            _set_module_by_name(model, name, compact)
            tier_summary = ", ".join(
                f"{tier.experts}x({tier.gate_up_rank}/{tier.down_rank})"
                for tier in compact.svd_tiers
            )
            logging.info(
                "[RoutedFunctionalSVD] %s (%d/%d): full=%d tiers=[%s] "
                "removed=%d kept=%d ratio=%.6f",
                name,
                layer_index + 1,
                len(names),
                compact.full_experts,
                tier_summary,
                len(removed),
                compact.kept,
                compact.expert_parameter_ratio(),
            )
            if compact.zero_moment_gate_up_experts or compact.zero_moment_down_experts:
                logging.warning(
                    "[RoutedFunctionalSVD] %s final weight-only fallbacks: gate/up=%d down=%d",
                    name,
                    compact.zero_moment_gate_up_experts,
                    compact.zero_moment_down_experts,
                )
            del stacked, compact, calibration_hidden
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return len(names)
