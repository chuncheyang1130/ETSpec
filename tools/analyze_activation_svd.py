#!/usr/bin/env python
"""Category-aware activation-SVD experiments for Qwen3-MoE experts.

This tool does not modify the model. It first collects category-separated expert
importance and routed diagonal activation moments for every expert. It then builds
the category-balanced ranking, chooses full/SVD/pruned tiers from that new ranking,
and compares weight-only with activation-scaled SVD at matched ranks.

Example::

  python tools/analyze_activation_svd.py \
    --model Qwen/Qwen3-30B-A3B \
    --per-category 64 --gen-tokens 128 \
    --full-experts 39 --svd-experts 38 \
    --ranks 64 128 256 \
    --out expert_activation/activation_svd_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm.auto import tqdm

from specdecodes.helpers.recipes.moe.expert_calibration import (
    CALIB_CATEGORIES,
    load_mass,
    load_selection,
    save_mass,
)
from specdecodes.models.utils.moe.base.activation_svd import (
    RoutedIntermediateMomentCollector,
    balance_category_moments,
    balance_category_signal,
    compare_shared_gate_up_svd,
    compare_weight_and_activation_svd,
    expert_down_weight,
    expert_projection_weight,
)
from specdecodes.models.utils.moe.base.expert_usage_tracker import _is_qwen3_moe_block


logging.basicConfig(level=logging.INFO, format="%(message)s")


def _select_tiers(
    scores: Mapping[str, torch.Tensor],
    full_experts: int,
    svd_experts: int,
    max_layers: int | None = None,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    full: Dict[str, List[int]] = {}
    svd: Dict[str, List[int]] = {}
    pruned: Dict[str, List[int]] = {}
    items = list(scores.items())
    if max_layers is not None:
        items = items[: int(max_layers)]
    for name, value in items:
        E = int(value.numel())
        if full_experts < 0 or svd_experts <= 0 or full_experts + svd_experts > E:
            raise ValueError(
                f"invalid tier sizes for {name}: full={full_experts}, "
                f"svd={svd_experts}, experts={E}"
            )
        order = torch.argsort(value.float(), descending=True).tolist()
        full[name] = sorted(int(e) for e in order[:full_experts])
        svd[name] = sorted(int(e) for e in order[full_experts:full_experts + svd_experts])
        pruned[name] = sorted(int(e) for e in order[full_experts + svd_experts:])
    return full, svd, pruned


def _encode(tokenizer, prompt: str) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )


@torch.no_grad()
def _generate_category_sequences(
    model,
    tokenizer,
    prompts_by_category: Mapping[str, Sequence[str]],
    *,
    device: str,
    gen_tokens: int,
    max_cache_len: int,
    chunk_size: int,
):
    from specdecodes.models.generators.naive import NaiveGenerator
    from specdecodes.models.utils.cache_utils import create_kv_cache

    generator = NaiveGenerator(
        target_model=model,
        tokenizer=tokenizer,
        draft_model=None,
        draft_params=None,
        cache_implementation="static",
        profiling=False,
        profiling_verbose=False,
        generator_kwargs={"prefill_chunk_size": int(chunk_size)},
    )
    cache = create_kv_cache(
        "static",
        max_cache_len=int(max_cache_len),
        max_batch_size=1,
        config=model.config,
        device=device,
        dtype=model.dtype,
    )
    sequences: Dict[str, List[torch.Tensor]] = defaultdict(list)
    skipped: Dict[str, int] = defaultdict(int)
    for category, prompts in prompts_by_category.items():
        for prompt in tqdm(prompts, desc=f"Generating {category}"):
            ids = _encode(tokenizer, prompt)
            if ids.shape[1] + int(gen_tokens) + 1 > int(max_cache_len):
                skipped[category] += 1
                continue
            generated = generator.generate(
                ids.to(device),
                max_new_tokens=int(gen_tokens),
                do_sample=False,
                past_key_values=cache,
            )
            sequences[category].append(generated.detach().cpu())
            cache.reset()
        if not sequences[category]:
            raise RuntimeError(
                f"no usable {category} prompts fit max_cache_len={max_cache_len}"
            )
        logging.info(
            "[%s] generated %d sequence(s), skipped %d oversized prompt(s)",
            category,
            len(sequences[category]),
            skipped[category],
        )
    del cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generator, dict(sequences), dict(skipped)


@torch.no_grad()
def _collect_category_moments(
    model,
    generator,
    sequences_by_category: Mapping[str, Sequence[torch.Tensor]],
    svd_experts: Mapping[str, Sequence[int]],
    *,
    device: str,
    chunk_size: int,
    track_gate_up_inputs: bool = False,
):
    from specdecodes.models.utils.cache_utils import create_kv_cache

    snapshots = {}
    tracked_tokens = {}
    with RoutedIntermediateMomentCollector(
        model,
        svd_experts,
        track_gate_up_inputs=track_gate_up_inputs,
    ) as collector:
        for category, sequences in sequences_by_category.items():
            collector.reset()
            tokens = 0
            for sequence in tqdm(sequences, desc=f"Replaying {category}"):
                replay_ids = sequence[:, :-1]
                if replay_ids.numel() == 0:
                    continue
                tokens += int(replay_ids.numel())
                replay_cache = create_kv_cache("dynamic")
                outputs = generator._chunked_prefill_forward(
                    replay_ids.to(device),
                    replay_cache,
                    prefill_chunk_size=int(chunk_size),
                    use_position_ids=True,
                )
                del outputs, replay_cache
            if tokens <= 0:
                raise RuntimeError(f"category {category!r} produced no replay tokens")
            snapshots[category] = collector.snapshot()
            tracked_tokens[category] = tokens
            logging.info("[%s] collected routed moments from %d tokens", category, tokens)
    return snapshots, tracked_tokens


def _load_moments(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def _ranking_payload(scores: Mapping[str, torch.Tensor]):
    payload = {}
    for name, values in scores.items():
        order = torch.argsort(values.float(), descending=True)
        payload[name] = {
            "ordered_experts": [int(e) for e in order.tolist()],
            "ordered_scores": [float(values[e].item()) for e in order.tolist()],
        }
    return payload


def _select_analysis_views(moment_artifact, analysis_category: str):
    """Select ranking scores and activation moments for one analysis domain.

    Category-specific ranking uses conditional REAP importance, matching the
    coding keep files produced by ``reconstruct_category_rankings.py``. The SVD
    metric uses the category's unconditional per-token ``g^2 z^2`` expectation,
    so aggregation still reflects how often each expert is routed.
    """
    if analysis_category == "balanced":
        return (
            {
                name: value.float()
                for name, value in moment_artifact["balanced_importance"].items()
            },
            {
                name: value.float()
                for name, value in moment_artifact["balanced_second_moment"].items()
            },
            moment_artifact["category_importance"],
            "category_balanced_importance_per_token",
            "mean_category(sum(g*||f_e||)/tracked_tokens_category)",
        )

    snapshots = moment_artifact.get("category_snapshots") or {}
    if analysis_category not in snapshots:
        raise ValueError(
            f"analysis category {analysis_category!r} is unavailable; choose "
            f"from {sorted(snapshots)} or 'balanced'"
        )
    tracked_tokens = moment_artifact.get("tracked_tokens") or {}
    tokens = int(tracked_tokens.get(analysis_category, 0))
    if tokens <= 0:
        raise ValueError(f"analysis category {analysis_category!r} has no tracked tokens")

    _, conditional_by_category = balance_category_signal(
        snapshots,
        tracked_tokens,
        signal="importance",
        normalization="conditional",
    )
    scores = {
        name: value.float()
        for name, value in conditional_by_category[analysis_category].items()
    }
    moments = {
        name: value.float() / float(tokens)
        for name, value in snapshots[analysis_category]["sum_g2_z2"].items()
    }
    return (
        scores,
        moments,
        conditional_by_category,
        f"{analysis_category}_conditional_reap_importance",
        "sum_active(g*||f_e||)/routing_count_expert_category",
    )


def _select_projection_moments(
    moment_artifact, analysis_category: str, projection: str
):
    """Select the calibrated input metric for one expert projection."""
    projection = str(projection)
    if projection == "down":
        return _select_analysis_views(moment_artifact, analysis_category)[1]
    signal = {
        "gate": "sum_gate_input_metric",
        "up": "sum_up_input_metric",
    }.get(projection)
    if signal is None:
        raise ValueError(f"unknown expert projection {projection!r}")

    snapshots = moment_artifact.get("category_snapshots") or {}
    if analysis_category == "balanced":
        return balance_category_moments(
            snapshots,
            moment_artifact.get("tracked_tokens") or {},
            signal=signal,
        )
    if analysis_category not in snapshots:
        raise ValueError(
            f"analysis category {analysis_category!r} is unavailable; choose "
            f"from {sorted(snapshots)} or 'balanced'"
        )
    tokens = int((moment_artifact.get("tracked_tokens") or {}).get(analysis_category, 0))
    if tokens <= 0:
        raise ValueError(f"analysis category {analysis_category!r} has no tracked tokens")
    values = snapshots[analysis_category].get(signal)
    if not values:
        raise ValueError(
            f"moment cache has no {projection}_proj calibration metric; recollect "
            "with --projections gate up down"
        )
    return {name: value.float() / float(tokens) for name, value in values.items()}


def _analyze(
    model,
    scores: Mapping[str, torch.Tensor],
    full: Mapping[str, Sequence[int]],
    svd: Mapping[str, Sequence[int]],
    pruned: Mapping[str, Sequence[int]],
    balanced_moments: Mapping[str, torch.Tensor],
    *,
    ranks: Sequence[int],
    svd_device: str,
    exact_svd: bool,
    seed: int,
    target_relative_error: float,
    projection: str = "down",
):
    torch.manual_seed(int(seed))
    layers = {}
    aggregate = {
        str(rank): {
            "weight_only_error_sum": 0.0,
            "activation_scaled_error_sum": 0.0,
            "importance_weighted_weight_error": 0.0,
            "importance_weighted_activation_error": 0.0,
            "importance_sum": 0.0,
            "absolute_weight_error": 0.0,
            "absolute_activation_error": 0.0,
            "weighted_energy": 0.0,
            "hybrid_parameters": 0,
            "dense_parameters": 0,
            "experts": 0,
        }
        for rank in ranks
    }
    plan_counts = {"pruned": 0, "full": 0, **{f"rank_{rank}": 0 for rank in ranks}}
    plan_parameters = 0
    dense_plan_parameters = 0

    for name in tqdm(list(svd), desc=f"Analyzing {projection}_proj SVD layers"):
        module = model.get_submodule(name)
        layer_result = {
            "full_experts": list(full[name]),
            "svd_experts": list(svd[name]),
            "pruned_experts": list(pruned[name]),
            "experts": {},
            "rank_plan": {
                **{str(e): "full" for e in full[name]},
                **{str(e): "pruned" for e in pruned[name]},
            },
        }
        moment_matrix = balanced_moments[name]
        sample_weight = expert_projection_weight(module, svd[name][0], projection)
        output_features, input_features = map(int, sample_weight.shape)
        matrix_parameters = output_features * input_features
        E = len(full[name]) + len(svd[name]) + len(pruned[name])
        plan_counts["full"] += len(full[name])
        plan_counts["pruned"] += len(pruned[name])
        plan_parameters += len(full[name]) * 3 * matrix_parameters
        dense_plan_parameters += E * 3 * matrix_parameters
        for rank in ranks:
            aggregate[str(rank)]["hybrid_parameters"] += (
                len(full[name]) * 3 * matrix_parameters
                + len(svd[name])
                * (
                    2 * matrix_parameters
                    + int(rank) * (output_features + input_features)
                )
            )
            aggregate[str(rank)]["dense_parameters"] += E * 3 * matrix_parameters
        for expert_id in svd[name]:
            metrics = compare_weight_and_activation_svd(
                expert_projection_weight(module, expert_id, projection),
                moment_matrix[expert_id],
                ranks,
                device=svd_device,
                randomized=not exact_svd,
            )
            importance = max(0.0, float(scores[name][expert_id].item()))
            rank_result = {}
            selected_rank = None
            for rank in ranks:
                weight_error = metrics.weight_only[int(rank)]
                activation_error = metrics.activation_scaled[int(rank)]
                factor_ratio = float(
                    int(rank) * (output_features + input_features) / matrix_parameters
                )
                rank_result[str(rank)] = {
                    "weight_only_relative_error": weight_error,
                    "activation_scaled_relative_error": activation_error,
                    f"{projection}_proj_parameter_ratio": factor_ratio,
                }
                agg = aggregate[str(rank)]
                agg["weight_only_error_sum"] += weight_error
                agg["activation_scaled_error_sum"] += activation_error
                agg["importance_weighted_weight_error"] += importance * weight_error
                agg["importance_weighted_activation_error"] += importance * activation_error
                agg["importance_sum"] += importance
                agg["absolute_weight_error"] += metrics.weighted_energy * weight_error
                agg["absolute_activation_error"] += metrics.weighted_energy * activation_error
                agg["weighted_energy"] += metrics.weighted_energy
                agg["experts"] += 1
                if selected_rank is None and activation_error <= float(target_relative_error):
                    selected_rank = int(rank)
            if selected_rank is None:
                selected_action = "full"
                plan_counts["full"] += 1
                plan_parameters += 3 * matrix_parameters
            else:
                selected_action = f"rank_{selected_rank}"
                plan_counts[selected_action] += 1
                plan_parameters += (
                    2 * matrix_parameters
                    + selected_rank * (output_features + input_features)
                )
            layer_result["rank_plan"][str(expert_id)] = selected_action
            layer_result["experts"][str(expert_id)] = {
                "ranking_score": importance,
                "weighted_energy": metrics.weighted_energy,
                "active_dimensions": metrics.active_dimensions,
                "ranks": rank_result,
                "selected_action": selected_action,
            }
        layers[name] = layer_result

    summary = {}
    for rank, agg in aggregate.items():
        n = max(int(agg["experts"]), 1)
        importance_sum = max(float(agg["importance_sum"]), 1e-30)
        energy = max(float(agg["weighted_energy"]), 1e-30)
        dense_parameters = max(int(agg["dense_parameters"]), 1)
        summary[rank] = {
            "mean_weight_only_relative_error": agg["weight_only_error_sum"] / n,
            "mean_activation_scaled_relative_error": agg["activation_scaled_error_sum"] / n,
            "importance_weighted_weight_only_error": (
                agg["importance_weighted_weight_error"] / importance_sum
            ),
            "importance_weighted_activation_scaled_error": (
                agg["importance_weighted_activation_error"] / importance_sum
            ),
            "global_weight_only_relative_error": agg["absolute_weight_error"] / energy,
            "global_activation_scaled_relative_error": (
                agg["absolute_activation_error"] / energy
            ),
            "hybrid_expert_parameter_ratio": (
                int(agg["hybrid_parameters"]) / dense_parameters
            ),
            "hybrid_expert_parameters": int(agg["hybrid_parameters"]),
            "dense_expert_parameters": int(agg["dense_parameters"]),
            "experts": int(agg["experts"]),
        }
    plan_summary = {
        "projection": str(projection),
        "target_relative_error": float(target_relative_error),
        "action_counts": plan_counts,
        "selected_expert_parameter_ratio": (
            plan_parameters / max(dense_plan_parameters, 1)
        ),
        "selected_expert_parameters": int(plan_parameters),
        "dense_expert_parameters": int(dense_plan_parameters),
    }
    return layers, summary, plan_summary


def _projection_configuration_memory(
    model,
    full: Mapping[str, Sequence[int]],
    svd: Mapping[str, Sequence[int]],
    pruned: Mapping[str, Sequence[int]],
    ranks: Sequence[int],
):
    """Parameter ratios for compressing one, two, or all three projections."""
    totals = {
        str(count): {str(rank): 0 for rank in ranks}
        for count in (1, 2, 3)
    }
    dense = 0
    for name in svd:
        module = model.get_submodule(name)
        sample = expert_projection_weight(module, svd[name][0], "down")
        output_features, input_features = map(int, sample.shape)
        matrix_parameters = output_features * input_features
        experts = len(full[name]) + len(svd[name]) + len(pruned[name])
        dense += experts * 3 * matrix_parameters
        for count in (1, 2, 3):
            for rank in ranks:
                totals[str(count)][str(rank)] += (
                    len(full[name]) * 3 * matrix_parameters
                    + len(svd[name])
                    * (
                        (3 - count) * matrix_parameters
                        + count * int(rank) * (output_features + input_features)
                    )
                )
    return {
        "dense_expert_parameters": int(dense),
        "ratios": {
            count: {
                rank: value / max(dense, 1)
                for rank, value in by_rank.items()
            }
            for count, by_rank in totals.items()
        },
    }


def _shared_gate_up_configuration_memory(
    model,
    full: Mapping[str, Sequence[int]],
    svd: Mapping[str, Sequence[int]],
    pruned: Mapping[str, Sequence[int]],
    ranks: Sequence[int],
):
    """Memory ratios for Cat and SkipCat on the middle expert tier.

    Gate/up share their input-side factor. Down remains independently
    factorized. ``skipcat_all_three`` also applies block skipping to down.
    Ratios cover all original expert weights, including the full and pruned
    tiers, but exclude non-expert model parameters.
    """

    totals = {
        method: {str(rank): 0 for rank in ranks}
        for method in (
            "independent_all_three",
            "cat_gate_up_independent_down",
            "skipcat_all_three",
        )
    }
    dense = 0
    for name in svd:
        module = model.get_submodule(name)
        gate = expert_projection_weight(module, svd[name][0], "gate")
        output_features, input_features = map(int, gate.shape)
        matrix_parameters = output_features * input_features
        experts = len(full[name]) + len(svd[name]) + len(pruned[name])
        dense += experts * 3 * matrix_parameters
        full_parameters = len(full[name]) * 3 * matrix_parameters
        for rank in ranks:
            rank = int(rank)
            independent_middle = 3 * rank * (input_features + output_features)
            cat_gate_up = rank * (input_features + 2 * output_features)
            independent_down = rank * (input_features + output_features)
            skipped_gate_up = rank * (
                input_features + 2 * output_features - rank
            )
            skipped_down = rank * (input_features + output_features - rank)
            totals["independent_all_three"][str(rank)] += (
                full_parameters + len(svd[name]) * independent_middle
            )
            totals["cat_gate_up_independent_down"][str(rank)] += (
                full_parameters
                + len(svd[name]) * (cat_gate_up + independent_down)
            )
            totals["skipcat_all_three"][str(rank)] += (
                full_parameters + len(svd[name]) * (skipped_gate_up + skipped_down)
            )
    return {
        "dense_expert_parameters": int(dense),
        "parameters": {
            method: {rank: int(value) for rank, value in by_rank.items()}
            for method, by_rank in totals.items()
        },
        "ratios": {
            method: {
                rank: value / max(dense, 1) for rank, value in by_rank.items()
            }
            for method, by_rank in totals.items()
        },
    }


def _analyze_shared_gate_up(
    model,
    scores: Mapping[str, torch.Tensor],
    full: Mapping[str, Sequence[int]],
    svd: Mapping[str, Sequence[int]],
    pruned: Mapping[str, Sequence[int]],
    gate_moments: Mapping[str, torch.Tensor],
    up_moments: Mapping[str, torch.Tensor],
    *,
    ranks: Sequence[int],
    svd_device: str,
    exact_svd: bool,
    seed: int,
):
    """Analyze a calibrated gate/up pair with one reusable input basis."""

    torch.manual_seed(int(seed))
    fields = (
        "joint_weight_only",
        "joint_activation_scaled",
        "gate_weight_only",
        "gate_activation_scaled",
        "up_weight_only",
        "up_activation_scaled",
    )
    aggregate = {
        str(rank): {
            **{f"{field}_sum": 0.0 for field in fields},
            **{f"importance_weighted_{field}": 0.0 for field in fields},
            **{f"absolute_{field}": 0.0 for field in fields},
            "importance_sum": 0.0,
            "joint_energy": 0.0,
            "gate_energy": 0.0,
            "up_energy": 0.0,
            "experts": 0,
        }
        for rank in ranks
    }
    layers = {}
    energy_for = {
        "joint_weight_only": "joint_weighted_energy",
        "joint_activation_scaled": "joint_weighted_energy",
        "gate_weight_only": "gate_weighted_energy",
        "gate_activation_scaled": "gate_weighted_energy",
        "up_weight_only": "up_weighted_energy",
        "up_activation_scaled": "up_weighted_energy",
    }
    for name in tqdm(list(svd), desc="Analyzing shared gate/up SVD layers"):
        module = model.get_submodule(name)
        layer_result = {
            "full_experts": list(full[name]),
            "svd_experts": list(svd[name]),
            "pruned_experts": list(pruned[name]),
            "experts": {},
        }
        for expert_id in svd[name]:
            metrics = compare_shared_gate_up_svd(
                expert_projection_weight(module, expert_id, "gate"),
                expert_projection_weight(module, expert_id, "up"),
                gate_moments[name][expert_id],
                up_moments[name][expert_id],
                ranks,
                device=svd_device,
                randomized=not exact_svd,
            )
            importance = max(0.0, float(scores[name][expert_id].item()))
            rank_result = {}
            for rank in ranks:
                rank = int(rank)
                values = {field: float(getattr(metrics, field)[rank]) for field in fields}
                rank_result[str(rank)] = {
                    f"{field}_relative_error": value
                    for field, value in values.items()
                }
                agg = aggregate[str(rank)]
                agg["importance_sum"] += importance
                agg["joint_energy"] += metrics.joint_weighted_energy
                agg["gate_energy"] += metrics.gate_weighted_energy
                agg["up_energy"] += metrics.up_weighted_energy
                agg["experts"] += 1
                for field, value in values.items():
                    agg[f"{field}_sum"] += value
                    agg[f"importance_weighted_{field}"] += importance * value
                    agg[f"absolute_{field}"] += (
                        value * float(getattr(metrics, energy_for[field]))
                    )
            layer_result["experts"][str(expert_id)] = {
                "ranking_score": importance,
                "joint_weighted_energy": metrics.joint_weighted_energy,
                "gate_weighted_energy": metrics.gate_weighted_energy,
                "up_weighted_energy": metrics.up_weighted_energy,
                "active_dimensions": metrics.active_dimensions,
                "ranks": rank_result,
            }
        layers[name] = layer_result

    summary = {}
    for rank, agg in aggregate.items():
        n = max(int(agg["experts"]), 1)
        importance_sum = max(float(agg["importance_sum"]), 1e-30)
        values = {"experts": int(agg["experts"])}
        for field in fields:
            branch = field.split("_", 1)[0]
            energy_key = "joint_energy" if branch == "joint" else f"{branch}_energy"
            values[f"mean_{field}_relative_error"] = agg[f"{field}_sum"] / n
            values[f"importance_weighted_{field}_relative_error"] = (
                agg[f"importance_weighted_{field}"] / importance_sum
            )
            values[f"global_{field}_relative_error"] = (
                agg[f"absolute_{field}"] / max(float(agg[energy_key]), 1e-30)
            )
        summary[rank] = values
    return layers, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--scores", default=None,
                        help="optional previous score file recorded for comparison only; "
                             "tier selection uses the newly balanced importance")
    parser.add_argument("--out", required=True, help="JSON reconstruction report")
    parser.add_argument("--moments-out", default=None, help="moment cache (default: <out>.moments.pt)")
    parser.add_argument(
        "--sequences-out",
        default=None,
        help=(
            "optional generated-sequence cache for routed-functional planning; "
            "only valid while collecting new moments"
        ),
    )
    parser.add_argument("--balanced-scores-out", default=None,
                        help="category-balanced importance JSON (default: <out>.balanced_importance.json)")
    parser.add_argument("--rankings-out", default=None,
                        help="per-category and balanced ordered rankings (default: <out>.rankings.json)")
    parser.add_argument("--from-moments", default=None, help="reuse a previously saved moment cache")
    parser.add_argument(
        "--ranking-moments",
        default=None,
        help="optional original-model moment cache used only to lock tier ranking",
    )
    parser.add_argument(
        "--routing-kept-path",
        default=None,
        help="selection JSON used to restrict routing during generation/replay",
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--per-category", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--categories", nargs="+", default=list(CALIB_CATEGORIES))
    parser.add_argument(
        "--analysis-category",
        default="balanced",
        help="ranking/moment domain: 'balanced' or one category saved in the cache",
    )
    parser.add_argument("--full-experts", type=int, default=39)
    parser.add_argument("--svd-experts", type=int, default=38)
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=["gate", "up", "down"],
        default=["down"],
        help="expert projections to calibrate and compare",
    )
    parser.add_argument(
        "--shared-gate-up",
        action="store_true",
        help=(
            "jointly decompose gate/up with one activation-calibrated input-side "
            "basis and report Cat/SkipCat memory"
        ),
    )
    parser.add_argument(
        "--shared-gate-up-only",
        action="store_true",
        help="run only the joint gate/up analysis and skip repeated independent SVDs",
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-cache-len", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--no-stacked", action="store_true")
    parser.add_argument("--svd-device", default="cpu")
    parser.add_argument("--exact-svd", action="store_true")
    parser.add_argument("--target-relative-error", type=float, default=0.05,
                        help="smallest candidate rank with error <= this is selected; "
                             "otherwise retain that expert at full rank")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=None,
                        help="debugging only: analyze/collect the first N scored layers")
    args = parser.parse_args()

    if args.chunk_size <= 0 or args.gen_tokens <= 0 or args.per_category <= 0:
        parser.error("--chunk-size, --gen-tokens, and --per-category must be positive")
    if args.from_moments and args.sequences_out:
        parser.error(
            "--sequences-out cannot be used with --from-moments because no "
            "sequences are generated"
        )
    if not args.ranks or min(args.ranks) <= 0:
        parser.error("--ranks must contain positive integers")
    args.projections = list(dict.fromkeys(args.projections))
    if args.shared_gate_up_only:
        args.shared_gate_up = True
        args.projections = []
    torch.set_num_threads(max(1, int(args.cpu_threads)))

    if args.target_relative_error < 0:
        parser.error("--target-relative-error must be non-negative")
    reference_meta = None
    if args.scores:
        _, reference_meta = load_mass(args.scores)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    logging.info("Loading %s directly on %s (%s)", args.model, args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()

    if not args.no_stacked:
        from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_target
        logging.info("Converted %d MoE blocks to memory-safe stacked form", apply_stacked_target(model))

    routing_kept = None
    if args.routing_kept_path:
        if args.no_stacked:
            raise ValueError("--routing-kept-path requires stacked MoE blocks")
        routing_kept, routing_meta = load_selection(args.routing_kept_path)
        modules = dict(model.named_modules())
        missing = sorted(set(routing_kept) - set(modules))
        if missing:
            raise KeyError(f"routing selection layers not found in model: {missing[:4]}")
        applied = 0
        for name, ids in routing_kept.items():
            module = modules[name]
            set_kept = getattr(module, "set_kept", None)
            if not callable(set_kept):
                raise TypeError(f"{name} does not support retained-pool routing")
            set_kept(torch.as_tensor(ids, dtype=torch.long))
            applied += 1
        logging.info(
            "Restricted %d MoE routers using %s (%s)",
            applied,
            args.routing_kept_path,
            routing_meta.get("selection_category", "unknown category"),
        )

    # Collect every expert before making any tier decision. Restricting to the
    # first N layers remains available only as an explicit smoke-test option.
    all_experts = {
        name: (
            list(routing_kept[name])
            if routing_kept is not None and name in routing_kept
            else list(range(int(module.num_experts)))
        )
        for name, module in model.named_modules()
        if _is_qwen3_moe_block(module)
    }
    if args.max_layers is not None:
        all_experts = dict(list(all_experts.items())[: int(args.max_layers)])
    if not all_experts:
        raise RuntimeError("no supported Qwen3 MoE layers found")

    moments_path = args.moments_out or os.path.splitext(args.out)[0] + ".moments.pt"
    if args.from_moments:
        moment_artifact = _load_moments(args.from_moments)
        logging.info("Loaded activation moments <- %s", args.from_moments)
    else:
        from run.pipelines.benchmarks.loader.openhermes import load_openhermes_dataset

        rows = load_openhermes_dataset(
            mode="stratified",
            per_category=args.per_category,
            seed=args.seed,
            categories=list(args.categories),
            scan_cap=300_000,
        )
        prompts_by_category = {
            category: [row["query"] for row in rows if row["category"] == category]
            for category in args.categories
        }
        missing = [category for category, prompts in prompts_by_category.items() if not prompts]
        if missing:
            raise RuntimeError(f"OpenHermes returned no prompts for categories: {missing}")

        generator, sequences, skipped = _generate_category_sequences(
            model,
            tokenizer,
            prompts_by_category,
            device=args.device,
            gen_tokens=args.gen_tokens,
            max_cache_len=args.max_cache_len,
            chunk_size=args.chunk_size,
        )
        if args.sequences_out:
            os.makedirs(os.path.dirname(args.sequences_out) or ".", exist_ok=True)
            torch.save(
                {
                    "sequences": sequences,
                    "meta": {
                        "model": args.model,
                        "categories": list(args.categories),
                        "per_category": int(args.per_category),
                        "gen_tokens": int(args.gen_tokens),
                        "max_cache_len": int(args.max_cache_len),
                        "chunk_size": int(args.chunk_size),
                        "enable_thinking": False,
                        "routing_kept_path": args.routing_kept_path,
                        "skipped": skipped,
                        "seed": int(args.seed),
                    },
                },
                args.sequences_out,
            )
            logging.info(
                "Saved generated calibration sequences -> %s", args.sequences_out
            )
        snapshots, tracked_tokens = _collect_category_moments(
            model,
            generator,
            sequences,
            all_experts,
            device=args.device,
            chunk_size=args.chunk_size,
            track_gate_up_inputs=any(
                projection in ("gate", "up") for projection in args.projections
            ) or bool(args.shared_gate_up),
        )
        balanced = balance_category_moments(snapshots, tracked_tokens)
        balanced_importance, category_importance = balance_category_signal(
            snapshots, tracked_tokens, signal="importance"
        )
        moment_artifact = {
            "balanced_second_moment": balanced,
            "balanced_importance": balanced_importance,
            "category_importance": category_importance,
            "category_snapshots": snapshots,
            "tracked_tokens": tracked_tokens,
            "collected_experts": all_experts,
            "meta": {
                "model": args.model,
                "scores": args.scores,
                "reference_score_meta": reference_meta,
                "categories": list(args.categories),
                "per_category": int(args.per_category),
                "gen_tokens": int(args.gen_tokens),
                "max_cache_len": int(args.max_cache_len),
                "chunk_size": int(args.chunk_size),
                "enable_thinking": False,
                "routing_kept_path": args.routing_kept_path,
                "projections": list(args.projections),
                "skipped_prompts": skipped,
            },
        }
        os.makedirs(os.path.dirname(moments_path) or ".", exist_ok=True)
        torch.save(moment_artifact, moments_path)
        logging.info("Saved category-balanced activation moments -> %s", moments_path)

    if "balanced_importance" not in moment_artifact:
        raise ValueError(
            "moment cache predates all-expert category-balanced collection; recollect moments"
        )

    ranking_artifact = (
        _load_moments(args.ranking_moments)
        if args.ranking_moments
        else moment_artifact
    )
    (
        scores,
        _,
        category_importance,
        score_signal,
        score_formula,
    ) = _select_analysis_views(ranking_artifact, str(args.analysis_category))
    base = os.path.splitext(args.out)[0]
    balanced_scores_path = args.balanced_scores_out or base + ".balanced_importance.json"
    rankings_path = args.rankings_out or base + ".rankings.json"
    balanced_meta = {
        "model": args.model,
        "signal": score_signal,
        "analysis_category": str(args.analysis_category),
        "categories": list(moment_artifact["meta"]["categories"]),
        "tracked_tokens": {
            category: int(tokens)
            for category, tokens in moment_artifact["tracked_tokens"].items()
        },
        "formula": score_formula,
        "enable_thinking": False,
        "moments": args.from_moments or moments_path,
        "ranking_moments": args.ranking_moments,
        "routing_kept_path": args.routing_kept_path,
    }
    save_mass(balanced_scores_path, scores, balanced_meta)
    ranking_artifact = {
        "meta": balanced_meta,
        "categories": {
            category: _ranking_payload(values)
            for category, values in category_importance.items()
        },
        "balanced": _ranking_payload(scores),
    }
    os.makedirs(os.path.dirname(rankings_path) or ".", exist_ok=True)
    with open(rankings_path, "w") as handle:
        json.dump(ranking_artifact, handle, indent=2)
    logging.info("Saved balanced importance -> %s", balanced_scores_path)
    logging.info("Saved per-category and balanced rankings -> %s", rankings_path)

    if args.collect_only:
        return

    # Tier and SVD-rank decisions happen only after the all-expert balanced
    # ranking has been materialized. ``--collect-only`` deliberately stops before
    # this point so ranking can be inspected without making any selection.
    full, svd, pruned = _select_tiers(
        scores, args.full_experts, args.svd_experts, args.max_layers
    )
    requested_ranks = sorted(set(args.ranks))
    projection_reports = {}
    for projection in args.projections:
        projection_moments = _select_projection_moments(
            moment_artifact,
            str(args.analysis_category),
            projection,
        )
        layers, summary, rank_plan = _analyze(
            model,
            scores,
            full,
            svd,
            pruned,
            projection_moments,
            ranks=requested_ranks,
            svd_device=args.svd_device,
            exact_svd=args.exact_svd,
            seed=args.seed,
            target_relative_error=args.target_relative_error,
            projection=projection,
        )
        projection_reports[projection] = {
            "summary": summary,
            "selected_rank_plan": rank_plan,
            "layers": layers,
        }
    shared_gate_up_report = None
    shared_gate_up_memory = None
    if args.shared_gate_up:
        gate_moments = _select_projection_moments(
            moment_artifact,
            str(args.analysis_category),
            "gate",
        )
        up_moments = _select_projection_moments(
            moment_artifact,
            str(args.analysis_category),
            "up",
        )
        shared_layers, shared_summary = _analyze_shared_gate_up(
            model,
            scores,
            full,
            svd,
            pruned,
            gate_moments,
            up_moments,
            ranks=requested_ranks,
            svd_device=args.svd_device,
            exact_svd=args.exact_svd,
            seed=args.seed,
        )
        shared_gate_up_report = {
            "summary": shared_summary,
            "layers": shared_layers,
            "common_input_metric": "sum_gate_input_metric + sum_up_input_metric",
            "factorization": "cat([W_gate, W_up], dim=0) with one shared right/input factor",
        }
        shared_gate_up_memory = _shared_gate_up_configuration_memory(
            model, full, svd, pruned, requested_ranks
        )
    configuration_memory = _projection_configuration_memory(
        model, full, svd, pruned, requested_ranks
    )
    report = {
        "meta": {
            "model": args.model,
            "reference_scores": args.scores,
            "reference_score_meta": reference_meta,
            "balanced_scores": balanced_scores_path,
            "rankings": rankings_path,
            "moments": args.from_moments or moments_path,
            "ranking_moments": args.ranking_moments,
            "routing_kept_path": args.routing_kept_path,
            "analysis_category": str(args.analysis_category),
            "full_experts": int(args.full_experts),
            "svd_experts": int(args.svd_experts),
            "pruned_experts": int(len(next(iter(pruned.values())))) if pruned else 0,
            "ranks": requested_ranks,
            "projections": list(args.projections),
            "shared_gate_up": bool(args.shared_gate_up),
            "svd_device": args.svd_device,
            "exact_svd": bool(args.exact_svd),
            "objective": (
                "coding routed diagonal input metrics; down uses g^2*z^2, "
                "gate/up use downstream-Jacobian-energy-weighted g^2*h^2"
            ),
        },
        "projection_reports": projection_reports,
        "configuration_memory": configuration_memory,
    }
    if shared_gate_up_report is not None:
        report["shared_gate_up_report"] = shared_gate_up_report
        report["shared_gate_up_memory"] = shared_gate_up_memory
    if args.projections == ["down"]:
        # Preserve the original report shape for existing analysis consumers.
        report.update(projection_reports["down"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    logging.info("Saved activation-SVD comparison -> %s", args.out)
    for projection, projection_report in projection_reports.items():
        for rank, values in projection_report["summary"].items():
            logging.info(
                "%s rank %s: weight-only error=%.6f, activation-scaled error=%.6f",
                projection,
                rank,
                values["global_weight_only_relative_error"],
                values["global_activation_scaled_relative_error"],
            )
    if shared_gate_up_report is not None:
        for rank, values in shared_gate_up_report["summary"].items():
            logging.info(
                "shared gate/up rank %s: weight-only joint error=%.6f, "
                "calibrated joint error=%.6f, gate error=%.6f, up error=%.6f, "
                "SkipCat expert-memory ratio=%.6f",
                rank,
                values["global_joint_weight_only_relative_error"],
                values["global_joint_activation_scaled_relative_error"],
                values["global_gate_activation_scaled_relative_error"],
                values["global_up_activation_scaled_relative_error"],
                shared_gate_up_memory["ratios"]["skipcat_all_three"][rank],
            )


if __name__ == "__main__":
    main()
