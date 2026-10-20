#!/usr/bin/env python
"""Plan full / calibrated-low-rank / removed MoE experts by routed distortion.

This is the experimental, interaction-aware allocator.  It differs from the
older rank-grid analysis in three ways:

1. it compares the *aggregate MoE output*, preserving cancellation/correlation;
2. removal recomputes top-k and router normalization; and
3. category distortions are reduced with an equal-domain robust objective.

To bound calibration cost, expert outputs are materialized only for each
token's router top-L shortlist (``--router-candidates``).  The evaluator refuses
plans with fewer than top-k retained candidates for any token.

Streaming mode instead consumes every cached token and immediately reduces
each expert/action to category-separated single-action error statistics. It
stores no token outputs or token-level Gram matrices; joint action errors are
therefore additive estimates, while every individual SVD or removal score uses
the complete calibration cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from specdecodes.helpers.recipes.moe.expert_calibration import load_selection
from specdecodes.models.utils.moe.base.activation_svd import expert_projection_weight
from specdecodes.models.utils.moe.base.expert_usage_tracker import _is_qwen3_moe_block
from specdecodes.models.utils.moe.base.routed_functional_compression import (
    DistortionReport,
    FunctionalCompressionPlan,
    LayerFunctionalTrace,
    LayerIndependentActionTrace,
    DAMAGE_UNITS,
    build_distortion_risk_traces,
    build_factorized_risk_traces,
    build_importance_compressibility_traces,
    build_mean_loss_traces,
    build_posterior_distortion_traces,
    build_propagated_risk_traces,
    cached_greedy_global_downgrade,
    gaussian_output_projection,
    robust_objective,
)
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
    apply_deployed_expert_factors,
    FACTOR_CALIBRATIONS,
    build_deployed_expert_factors,
    FACTOR_OUTPUT_METRICS,
    factor_calibration_selection,
)
from tools.analyze_activation_svd import (
    _generate_category_sequences,
    _load_moments,
    _select_projection_moments,
)
from tools.evaluate_joint_svd_rank_grid import (
    _apply_factors,
    _capture_hidden_states,
    _decompose_calibrated,
    _load_hidden_cache,
    _load_hidden_metadata,
    _make_generator,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_rank_pair(value: str) -> tuple[int, int]:
    try:
        gate_up, down = value.split(":", 1)
        pair = int(gate_up), int(down)
    except Exception as error:
        raise argparse.ArgumentTypeError("rank pairs must look like 384:256") from error
    if min(pair) <= 0:
        raise argparse.ArgumentTypeError("ranks must be positive")
    return pair


def _rescaled_propagation_factors(
    record: Mapping, action: str, scale: float
) -> dict[str, float] | None:
    """Re-evaluate one layer's stored damage curve at a different scale.

    ``tools/estimate_moe_propagation.py`` records ``damage(s) = c * s**gamma``
    per layer and category, so the gain per unit relative local damage at any
    scale is ``c * s**(gamma - 2)``. Recovering it from the curve avoids
    re-running the model just to move the evaluation point.
    """

    curve = record.get("scale_curve")
    if curve is None:
        return None
    rescaled = {}
    for category, coefficients in curve.items():
        exponent = float(coefficients["exponent"])
        coefficient = float(coefficients["coefficient"])
        rescaled[str(category)] = coefficient * float(scale) ** (exponent - 2.0)
    return rescaled


def _load_propagation_factors(
    path: str,
    traces: Sequence[LayerIndependentActionTrace],
    *,
    svd_scale: float | None = None,
    removal_scale: float | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load and category-align end-to-end layer propagation factors."""

    with open(path) as handle:
        artifact = json.load(handle)
    if artifact.get("format") != "moe_propagation_damage":
        raise ValueError(
            f"{path}: expected a moe_propagation_damage calibration artifact"
        )
    artifact_categories = tuple(str(value) for value in artifact["category_names"])
    layers = artifact.get("layers") or {}
    requested_scale = {"svd": svd_scale, "remove": removal_scale}
    result = {}
    for trace in traces:
        if trace.name not in layers:
            raise ValueError(f"{path}: missing propagation calibration for {trace.name}")
        record = layers[trace.name]
        result[trace.name] = {}
        for action in ("svd", "remove"):
            values = None
            scale = requested_scale[action]
            if scale is not None:
                values = _rescaled_propagation_factors(record, action, scale)
                if values is None:
                    raise ValueError(
                        f"{path}: {trace.name} has no scale curve, so its "
                        f"{action} gain cannot be re-evaluated at scale {scale}; "
                        "recalibrate with the current propagation tool"
                    )
            if values is None:
                values = record.get(f"{action}_factors")
            if values is None:
                values = record.get("factors")
            if values is None:
                raise ValueError(
                    f"{path}: {trace.name} lacks {action} propagation factors"
                )
            if isinstance(values, Mapping):
                missing = [name for name in trace.category_names if name not in values]
                if missing:
                    raise ValueError(
                        f"{path}: {trace.name} {action} factors lack categories "
                        f"{missing}"
                    )
                aligned = [values[name] for name in trace.category_names]
            else:
                tensor = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
                if int(tensor.numel()) != len(artifact_categories):
                    raise ValueError(
                        f"{path}: {trace.name} {action} factor count does not "
                        "match category_names"
                    )
                by_category = dict(zip(artifact_categories, tensor.tolist()))
                missing = [
                    name for name in trace.category_names if name not in by_category
                ]
                if missing:
                    raise ValueError(
                        f"{path}: {trace.name} {action} factors lack categories "
                        f"{missing}"
                    )
                aligned = [by_category[name] for name in trace.category_names]
            result[trace.name][action] = torch.as_tensor(
                aligned, dtype=torch.float64
            )
    return result


def _action_spec(
    rank_pairs: Sequence[tuple[int, int]], hidden: int, intermediate: int
) -> tuple[list[tuple[int, int]], tuple[str, ...], torch.Tensor]:
    full_cost = 3 * int(hidden) * int(intermediate)
    unique = sorted(
        set((int(gu), int(down)) for gu, down in rank_pairs),
        key=lambda pair: pair[0] * (hidden + 2 * intermediate)
        + pair[1] * (hidden + intermediate),
    )
    costs = [0]
    kept_pairs = []
    for gate_up, down in unique:
        if gate_up > min(hidden, 2 * intermediate):
            raise ValueError(f"gate/up rank {gate_up} exceeds its matrix rank")
        if down > min(hidden, intermediate):
            raise ValueError(f"down rank {down} exceeds its matrix rank")
        cost = gate_up * (hidden + 2 * intermediate) + down * (hidden + intermediate)
        if cost >= full_cost:
            logging.warning(
                "Skipping rank pair %d:%d because factor cost %d >= full cost %d",
                gate_up,
                down,
                cost,
                full_cost,
            )
            continue
        if cost <= costs[-1]:
            raise ValueError("rank-pair costs must be strictly increasing")
        kept_pairs.append((gate_up, down))
        costs.append(cost)
    if not kept_pairs:
        raise ValueError("no low-rank action is cheaper than a full expert")
    costs.append(full_cost)
    names = ("removed",) + tuple(
        f"gu{gate_up}_down{down}" for gate_up, down in kept_pairs
    ) + ("full",)
    return kept_pairs, names, torch.tensor(costs, dtype=torch.long)


def _selected_calibration_rows(
    sequences: Mapping[str, Sequence[torch.Tensor]],
    *,
    total_rows: int,
    tokens_per_category: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
    dict,
]:
    row_ids = []
    category_ids = []
    selected_prompt_ids = []
    prompt_category_ids = []
    ranges = {}
    offset = 0
    prompt_offset = 0
    names = tuple(str(name) for name in sequences)
    for category_id, name in enumerate(names):
        prompt_lengths = torch.tensor(
            [
                int(sequence[:, :-1].numel())
                for sequence in sequences[name]
            ],
            dtype=torch.long,
        )
        count = int(prompt_lengths.sum().item())
        if count <= 0:
            raise ValueError(f"category {name!r} has no replay tokens")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + category_id * 1_000_003)
        take = min(int(tokens_per_category), count)
        chosen = torch.randperm(count, generator=generator)[:take].sort().values + offset
        row_ids.append(chosen)
        category_ids.append(torch.full((take,), category_id, dtype=torch.long))
        local_prompt_ids = torch.repeat_interleave(
            torch.arange(int(prompt_lengths.numel()), dtype=torch.long),
            prompt_lengths,
        )
        selected_prompt_ids.append(
            local_prompt_ids.index_select(0, chosen - offset) + prompt_offset
        )
        prompt_category_ids.append(
            torch.full(
                (int(prompt_lengths.numel()),), category_id, dtype=torch.long
            )
        )
        ranges[name] = {
            "offset": offset,
            "tokens": count,
            "sampled": take,
            "prompt_offset": prompt_offset,
            "prompts": int(prompt_lengths.numel()),
        }
        offset += count
        prompt_offset += int(prompt_lengths.numel())
    if offset != int(total_rows):
        raise ValueError(
            f"sequence cache describes {offset} tokens but hidden cache has {total_rows}"
        )
    return (
        torch.cat(row_ids),
        torch.cat(category_ids),
        torch.cat(selected_prompt_ids),
        torch.cat(prompt_category_ids),
        names,
        ranges,
    )


def _select_hidden_rows_without_full_copy(
    all_hidden: torch.Tensor, row_ids: torch.Tensor
) -> torch.Tensor:
    """Return the complete cache by alias when row_ids is the identity."""

    rows = int(all_hidden.shape[0])
    ids = torch.as_tensor(row_ids, dtype=torch.long).reshape(-1)
    if int(ids.numel()) == rows and torch.equal(ids, torch.arange(rows)):
        return all_hidden
    return all_hidden.index_select(0, ids)


def _validate_full_expert_coverage(
    module_names: Sequence[str],
    modules: Mapping[str, object],
    moment_artifact: Mapping,
    gate_moments: Mapping[str, torch.Tensor],
    up_moments: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate and construct an all-expert routing pool for every layer."""

    collected_by_layer = moment_artifact.get("collected_experts") or {}
    routing_ids = {}
    for name in module_names:
        experts = int(modules[name].num_experts)
        expected = set(range(experts))
        for projection, moments in (("gate", gate_moments), ("up", up_moments)):
            if name not in moments or moments[name].ndim != 2:
                raise ValueError(f"{name} has no {projection}_proj moment matrix")
            if int(moments[name].shape[0]) != experts:
                raise ValueError(
                    f"{name} {projection}_proj moments cover {moments[name].shape[0]} "
                    f"rows, but the model has {experts} experts"
                )
        collected = collected_by_layer.get(name)
        if collected is None:
            raise ValueError(
                f"{name}: the moment artifact does not record collected_experts; "
                "full routing requires an explicitly full-expert moment artifact"
            )
        missing = sorted(expected - {int(value) for value in collected})
        if missing:
            raise ValueError(
                f"{name}: full routing requested, but calibration moments are "
                f"missing {len(missing)} expert(s), for example {missing[:8]}. "
                "Recollect gate/up moments without --routing-kept-path."
            )
        routing_ids[name] = torch.arange(experts, dtype=torch.long)
    return routing_ids


def _validate_hidden_cache_routing(
    metadata: Mapping,
    expected_ids: Mapping[str, torch.Tensor],
    *,
    full_routing: bool,
    routing_kept_path: str | None,
    sequence_meta: Mapping,
) -> None:
    """Prevent replaying hidden states produced by a different routing pool."""

    recorded = metadata.get("routing_pool_ids")
    if recorded is not None:
        for name, ids in expected_ids.items():
            if name not in recorded or [int(value) for value in ids.tolist()] != [
                int(value) for value in recorded[name]
            ]:
                raise ValueError(
                    f"hidden cache routing pool does not match current pool for {name}"
                )
        return
    if full_routing:
        raise ValueError(
            "the reusable hidden cache has no full-routing provenance. Recreate it "
            "without --reuse-hidden-cache before running the full-128 experiment"
        )
    # Backward compatibility for the existing legacy 96-expert cache: its
    # sequence artifact records the exact selection used during replay.
    if sequence_meta.get("routing_kept_path") != routing_kept_path:
        raise ValueError(
            "legacy hidden cache has no routing provenance and the sequence-cache "
            "routing selection does not match --routing-kept-path"
        )


@torch.no_grad()
def _pool_router_logits(module, hidden: torch.Tensor, *, device: str, batch_size: int):
    selected = module.selected_expert_ids.detach().long()
    router = module.router_weights.index_select(0, selected)
    values = []
    for start in range(0, hidden.shape[0], int(batch_size)):
        x = hidden[start : start + int(batch_size)].to(
            device=device, dtype=router.dtype
        )
        values.append(F.linear(x, router).float().cpu())
    return torch.cat(values), selected.cpu()


def _factor_calibration_scope(
    trace_mode: str,
    factor_metric: str,
    factor_shrink: str = "diagonal",
    down_compensation: bool = False,
    factor_calibration: str = "shortlist",
    factor_output_metric: str = "none",
) -> str | None:
    """Identity of the factor calibration, used to gate trace reuse.

    The metric is part of it: reusing a diagonal-calibrated trace under a Gram
    plan would price factors that the deployment will not build.  Only the
    streaming builder calibrates factors, so other modes have no scope.
    """

    if str(trace_mode) != "streaming":
        return None
    if str(factor_metric) == "gram":
        # v3: the Gram is now shrunk toward its diagonal by N_eff (see
        # ordinary_svd.gram_shrinkage_coefficient). Raw-Gram traces price factors the
        # deployment no longer builds, so they must not be reusable.
        # v4: the shrinkage TARGET changes the factors, so a v3 trace
        # (shrunk toward the diagonal) must not be reused for a layer-
        # shrunk plan. The mode is folded into the scope for that reason.
        scope = f"router_top_l_all_tokens_gram_{factor_shrink}_v4"
    else:
        scope = "router_top_l_all_tokens_v1"
    if str(factor_calibration) != "shortlist":
        # The scopes above name the top-L shortlist. The top-k modes calibrate
        # on other tokens with other weights and ignore the recorded moments, so
        # they get their own name, while every existing shortlist scope keeps its
        # exact string and its traces stay reusable.
        scope = scope.replace(
            "router_top_l_all_tokens", f"router_{factor_calibration}_tokens"
        )
    if str(factor_output_metric) != "none":
        # The gate/up output metric changes the factors, so a trace priced
        # without it must not be reused for a plan that deploys it.
        scope = f"{scope}_out{factor_output_metric}"
    # The compensated down target is a different matrix, so a trace priced
    # without it cannot be reused for a run that deploys it, or vice versa.
    return f"{scope}_downcomp" if down_compensation else scope


def _calibration_moment_or_fallback(
    recorded: torch.Tensor,
    router_conditioned: torch.Tensor,
    *,
    input_features: int,
) -> tuple[torch.Tensor, str]:
    """Choose a finite activation metric, using weight-only scaling last."""

    recorded = torch.nan_to_num(recorded.detach().float(), nan=0.0).clamp_min(0)
    if int(recorded.numel()) == int(input_features) and float(recorded.max().item()) > 0:
        return recorded, "recorded"
    conditioned = torch.nan_to_num(
        router_conditioned.detach().float(), nan=0.0
    ).clamp_min(0)
    if int(conditioned.numel()) == int(input_features) and float(conditioned.max().item()) > 0:
        return conditioned, "router_conditioned"
    return torch.ones(
        int(input_features), device=recorded.device, dtype=torch.float32
    ), "weight_only"


def _category_energy_sum(
    values: torch.Tensor,
    category_ids: torch.Tensor,
    categories: int,
) -> torch.Tensor:
    """Accumulate one scalar value per token into FP64 category totals."""

    return torch.bincount(
        category_ids.long().cpu(),
        weights=values.detach().double().cpu(),
        minlength=int(categories),
    )


def _log_damage_moments(
    energy: torch.Tensor,
    token_categories: torch.Tensor,
    reference_mean_energy: torch.Tensor,
    categories: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce one action's active-token damages to category log moments.

    Dividing each token's routed squared error by the category mean MoE output
    energy makes it a relative damage, so its log moments describe the *shape*
    of that expert's damage distribution independently of the layer's output
    scale. Only the first two log moments and a count are kept, which is all the
    distortion-risk planner needs to recover a conditional mean and dispersion
    without retaining any per-token tensor.

    Tokens with exactly zero damage carry no log information and are counted
    out. They sit at the bottom of the distribution, so excluding them can only
    make the estimated upper tail more conservative, never less.
    """

    relative = energy.detach().double().cpu()
    ids = token_categories.long().cpu()
    relative = relative / reference_mean_energy.index_select(0, ids)
    positive = relative > 0
    if not bool(positive.any()):
        empty = torch.zeros(int(categories), dtype=torch.float64)
        return empty, empty.clone(), torch.zeros(int(categories), dtype=torch.long)
    selected = ids[positive]
    logs = relative[positive].log()
    return (
        torch.bincount(selected, weights=logs, minlength=int(categories)),
        torch.bincount(selected, weights=logs.square(), minlength=int(categories)),
        torch.bincount(selected, minlength=int(categories)).long(),
    )


def _factorized_frobenius_error(
    weight: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> tuple[float, float]:
    """Return exact ``(||W-LR||_F^2, ||W||_F^2)`` without forming ``LR``."""

    weight = weight.detach().float()
    left = left.detach().float()
    right = right.detach().float()
    weight_energy = float(weight.square().sum().item())
    left_gram = left.t() @ left
    right_gram = right @ right.t()
    approximation_energy = float((left_gram * right_gram).sum().item())
    cross = float(((left.t() @ weight) * right).sum().item())
    error = max(0.0, weight_energy + approximation_energy - 2.0 * cross)
    return error, weight_energy


def _deployed_factor_structural_error(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    factors: Mapping[str, torch.Tensor],
    down_target: torch.Tensor | None = None,
) -> float:
    """Relative projection-weight error of one exactly deployed SVD action.

    ``down_target`` is the matrix the down factors were actually fitted to. Under
    down compensation that is not ``down_weight``, and scoring against
    ``down_weight`` would inflate the structural error of exactly those actions
    whose true damage fell -- an anti-correlated prior for the sparse experts
    that depend on it most.
    """

    gate_error, gate_energy = _factorized_frobenius_error(
        gate_weight,
        factors["gate_output_factor"],
        factors["gu_input_factor"],
    )
    up_error, up_energy = _factorized_frobenius_error(
        up_weight,
        factors["up_output_factor"],
        factors["gu_input_factor"],
    )
    down_error, down_energy = _factorized_frobenius_error(
        down_weight if down_target is None else down_target,
        factors["down_output_factor"],
        factors["down_input_factor"],
    )
    return float(
        (gate_error + up_error + down_error)
        / max(gate_energy + up_energy + down_energy, 1e-30)
    )


@torch.no_grad()
def _full_expert_output(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    deployed = inputs.to(dtype=storage_dtype)
    gate = F.linear(deployed, gate_weight.to(storage_dtype))
    up = F.linear(deployed, up_weight.to(storage_dtype))
    middle = F.silu(gate) * up
    return F.linear(middle, down_weight.to(storage_dtype))


@torch.no_grad()
def fit_routing_penalty_threshold(
    margin: torch.Tensor,
    keep_cost: torch.Tensor,
    drop_cost: torch.Tensor,
    keep_price: float = 1.0,
) -> tuple[float, float]:
    """Best router-logit discount for one expert, and the damage it removes.

    ``keep_cost`` is what each token costs if the expert serves it, ``drop_cost``
    what it costs if the expert loses the slot to the top-(k+1) expert. The ideal
    policy is per-token -- displace wherever dropping is cheaper -- but inference
    can evaluate neither term, so a margin threshold is the deployable surrogate.

    Sorting by margin turns every candidate threshold into a prefix, so the total
    cost of each is two cumulative sums and the best is their argmin: exact, and
    O(n log n) with no extra forward pass.

    Returns ``(threshold, gain)``, gain being the fraction of this action's
    damage the threshold removes. A gain of zero means no threshold beats serving
    every token -- the honest per-expert verdict, readable without a benchmark.
    Both branches grow with margin (a wider margin means both a worse replacement
    AND a heavier routing weight), so a threshold only pays where the reroute
    cost grows faster.

    ``keep_price`` is the class exchange rate, and this is the right place for it.
    ``tier_price`` currently multiplies tier RISK at the very end of the pipeline,
    which can only reorder the final ranking. But the claim it encodes -- that a
    unit of truncation error costs more downstream than a unit of removal error,
    because removal is absorbed by router renormalisation -- applies equally to
    the keep/drop comparison made HERE. If it is true, this threshold is
    mispriced too, and ``lam`` itself is wrong. Priced here it changes WHICH
    tokens get displaced, not just how the action is ranked afterwards.
    ``1.0`` is an exact no-op and reproduces the untouched fit.
    """

    if float(keep_price) <= 0:
        raise ValueError(f"keep_price must be positive, got {keep_price}")

    tokens = int(margin.numel())
    if tokens == 0:
        return 0.0, 0.0
    order = torch.argsort(margin)
    ordered = margin[order]
    keep = keep_cost[order].double() * float(keep_price)
    drop = drop_cost[order].double()
    zero = torch.zeros(1, dtype=torch.float64, device=keep.device)
    cost = torch.cat((zero, drop.cumsum(0))) + (
        keep.sum() - torch.cat((zero, keep.cumsum(0)))
    )
    best = int(cost.argmin().item())
    baseline = float(cost[0].item())
    # ``best`` counts displaced tokens, and displacing them needs a threshold
    # ABOVE their margins: lam = ordered[best] displaces exactly the first
    # ``best``. Displacing all of them has no such next margin to stop at, so
    # step just past the largest one.
    if best == 0:
        threshold = 0.0
    elif best < tokens:
        threshold = float(ordered[best])
    else:
        threshold = math.nextafter(float(ordered[-1]), math.inf)
    # A negative threshold would mean PREFERRING the compressed expert to an
    # exact one, which this term does not model.
    threshold = max(threshold, 0.0)
    gain = (baseline - float(cost[best].item())) / max(baseline, 1e-30)
    return threshold, gain


@torch.no_grad()
def _build_layer_streaming_trace(
    module,
    name: str,
    hidden: torch.Tensor,
    category_ids: torch.Tensor,
    category_names: Sequence[str],
    gate_moments: torch.Tensor,
    up_moments: torch.Tensor,
    rank_pairs: Sequence[tuple[int, int]],
    action_names: Sequence[str],
    action_cost: torch.Tensor,
    *,
    prompt_ids: torch.Tensor | None = None,
    prompt_category_ids: torch.Tensor | None = None,
    device: str,
    routing_batch_size: int,
    router_candidates: int,
    randomized: bool,
    scale_floor: float,
    occupancy_alpha: float,
    removal_fallback_depths: int = 1,
    gram_metric: bool = False,
    gram_shrink_target_mode: str = "diagonal",
    down_compensation: bool = False,
    factor_calibration: str = "shortlist",
    factor_output_metric: str = "none",
    damage_unit: str = "energy",
    routing_penalty_scale: float = 0.0,
    routing_penalty_calibrate: bool = False,
    seed: int,
):
    """Build compact single-action statistics from every cached hidden state.

    The reference MoE output is evaluated with the stacked module in chunks.
    SVD errors use every token that currently activates the expert. Candidate
    factors are calibrated on the token set ``factor_calibration`` selects (the
    router top-L by default), exactly as when the final factor artifact is
    saved, so the factors scored by planning are the factors loaded for
    inference. Removal errors exactly recompute top-k
    replacement and normalization for removing that expert alone. Multiple
    selected actions are combined additively by the allocator; no per-token
    output or Gram tensor is retained.
    """

    logits, global_expert_ids = _pool_router_logits(
        module, hidden, device=device, batch_size=routing_batch_size
    )
    tokens, pool_size = map(int, logits.shape)
    if prompt_ids is None and prompt_category_ids is None:
        # Backward-compatible direct API behavior: independent one-token
        # clusters. The CLI always supplies true sequence/prompt membership.
        prompt_ids = torch.arange(tokens, dtype=torch.long)
        prompt_category_ids = category_ids.clone()
    elif prompt_ids is None or prompt_category_ids is None:
        raise ValueError(
            "prompt_ids and prompt_category_ids must be provided together"
        )
    prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)
    prompt_category_ids = torch.as_tensor(
        prompt_category_ids, dtype=torch.long
    ).reshape(-1)
    if int(prompt_ids.numel()) != tokens:
        raise ValueError("prompt_ids must identify every streaming calibration token")
    prompts = int(prompt_category_ids.numel())
    if prompts <= 0 or bool(((prompt_ids < 0) | (prompt_ids >= prompts)).any()):
        raise ValueError("prompt_ids contains an invalid prompt index")
    top_k = int(module.top_k)
    if pool_size <= top_k:
        raise ValueError("streaming removal scoring requires at least top_k + 1 experts")
    depths = int(removal_fallback_depths)
    if depths < 1:
        raise ValueError("removal_fallback_depths must be at least 1")
    if pool_size < top_k + depths:
        raise ValueError(
            f"{name}: {depths} removal fallback depths need at least "
            f"{top_k + depths} routable experts, but the pool has {pool_size}"
        )
    shortlist = min(max(top_k + depths, int(router_candidates)), pool_size)
    shortlist_values, shortlist_ids = torch.topk(logits, k=shortlist, dim=-1)
    # This is deliberately the same top-L normalization used by
    # Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state.  Active top-k
    # probabilities below still follow the model's actual routing semantics.
    shortlist_probability = F.softmax(shortlist_values, dim=-1)
    use_recorded_moments = str(factor_calibration) == "shortlist"
    if use_recorded_moments:
        # Kept inline rather than through factor_calibration_selection: the
        # planner widens its shortlist to top_k + depths, and existing traces
        # must stay bit-identical.
        factor_ids = shortlist_ids
        factor_weight_table = shortlist_probability.square()
    else:
        # No plan exists while traces are built, so a top-k mode is priced on the
        # full router. Deployment routes over the plan's retained experts; the two
        # differ only for tokens whose original top-k held a removed expert.
        factor_ids, factor_weight_table = factor_calibration_selection(
            logits,
            factor_calibration=factor_calibration,
            top_k=top_k,
            norm_topk_prob=bool(module.norm_topk_prob),
            router_candidates=router_candidates,
        )
    topk_ids = shortlist_ids[:, :top_k]
    topk_values = shortlist_values[:, :top_k]
    # One column per candidate fallback depth: depth d is the top-(k+1+d)
    # expert, i.e. the expert that absorbs the traffic once the d nearer
    # candidates have themselves been removed.
    replacement_ids = shortlist_ids[:, top_k : top_k + depths]
    replacement_values = shortlist_values[:, top_k : top_k + depths]
    if bool(module.norm_topk_prob):
        topk_weights = F.softmax(topk_values, dim=-1)
        replacement_probability = torch.exp(
            replacement_values - torch.logsumexp(topk_values, dim=-1).unsqueeze(1)
        )
    else:
        all_probabilities = F.softmax(logits, dim=-1)
        topk_weights = all_probabilities.gather(1, topk_ids)
        replacement_probability = all_probabilities.gather(1, replacement_ids)
        del all_probabilities

    topk_count = torch.bincount(topk_ids.reshape(-1), minlength=pool_size).long()
    shortlist_count = torch.bincount(
        shortlist_ids.reshape(-1), minlength=pool_size
    ).long()
    standby_count = torch.bincount(
        shortlist_ids[:, top_k:].reshape(-1), minlength=pool_size
    ).long()
    fallback_count = torch.bincount(
        replacement_ids.reshape(-1), minlength=pool_size
    ).long()
    del logits, shortlist_values

    categories = len(category_names)
    if not (0 <= float(occupancy_alpha) <= 1):
        raise ValueError("occupancy_alpha must be in [0, 1]")
    storage_dtype = expert_projection_weight(
        module, int(global_expert_ids[0]), "gate"
    ).dtype
    if str(damage_unit) not in DAMAGE_UNITS:
        raise ValueError(
            f"damage_unit must be one of {DAMAGE_UNITS}, got {damage_unit!r}"
        )
    amplitude = str(damage_unit) == "amplitude"
    if amplitude and (
        routing_penalty_calibrate
        or float(routing_penalty_scale) > 0.0
        or float(occupancy_alpha) != 1.0
    ):
        raise ValueError(
            "the pair Gram, routing-penalty pricing and occupancy_alpha != 1 are "
            "defined on squared error; build this trace with damage_unit='energy'"
        )

    def per_token_damage(energy: torch.Tensor) -> torch.Tensor:
        """Squared norms in, the trace's damage unit out (identity for energy)."""
        return energy.clamp_min(0).sqrt() if amplitude else energy

    hidden_size = int(hidden.shape[1])
    reference_output = torch.empty(
        tokens, hidden_size, dtype=storage_dtype, device="cpu"
    )
    for start in tqdm(
        range(0, tokens, int(routing_batch_size)),
        desc=f"Full-token reference {name}",
        leave=False,
    ):
        stop = min(tokens, start + int(routing_batch_size))
        batch = hidden[start:stop].to(device=device, dtype=storage_dtype)
        reference_output[start:stop] = (
            module(batch.unsqueeze(0)).reshape(-1, hidden_size).cpu()
        )
        del batch

    reference_energy = _category_energy_sum(
        per_token_damage(reference_output.float().square().sum(dim=1)),
        category_ids,
        categories,
    )
    # Per-token damages are normalized by this so their log moments describe the
    # damage distribution's shape rather than the layer's output scale.
    reference_mean_energy = (
        reference_energy
        / torch.bincount(category_ids.long().cpu(), minlength=categories)
        .double()
        .clamp_min(1.0)
    ).clamp_min(1e-300)
    prompt_reference_energy = torch.bincount(
        prompt_ids,
        weights=per_token_damage(
            reference_output.float().square().sum(dim=1)
        ).double(),
        minlength=prompts,
    )
    # One replacement aggregate per depth. Each expert is evaluated once for the
    # union of (token, depth) slots it fills, so the extra depths cost extra
    # token evaluations but no extra weight loads.
    replacement_output = torch.empty(
        depths, tokens, hidden_size, dtype=storage_dtype, device="cpu"
    )
    for local_id in tqdm(
        range(pool_size), desc=f"Full-token standby {name}", leave=False
    ):
        slots = (replacement_ids == local_id).nonzero(as_tuple=False)
        if not int(slots.shape[0]):
            continue
        expert_id = int(global_expert_ids[local_id].item())
        gate_weight = expert_projection_weight(module, expert_id, "gate")
        up_weight = expert_projection_weight(module, expert_id, "up")
        down_weight = expert_projection_weight(module, expert_id, "down")
        unique_tokens = torch.unique(slots[:, 0])
        for start in range(0, int(unique_tokens.numel()), int(routing_batch_size)):
            ids = unique_tokens[start : start + int(routing_batch_size)]
            x = hidden.index_select(0, ids).to(device=device, dtype=torch.float32)
            value = _full_expert_output(
                x,
                gate_weight,
                up_weight,
                down_weight,
                storage_dtype=storage_dtype,
            ).cpu()
            position = torch.searchsorted(ids, slots[:, 0])
            inside = (position < int(ids.numel())) & (
                ids.index_select(0, position.clamp_max(int(ids.numel()) - 1))
                == slots[:, 0]
            )
            for depth in range(depths):
                chosen = inside & (slots[:, 1] == depth)
                if not bool(chosen.any()):
                    continue
                replacement_output[depth].index_copy_(
                    0,
                    slots[chosen, 0],
                    value.index_select(0, position[chosen]),
                )
            del x, value

    action_numerator = torch.zeros(
        pool_size, len(action_names), categories, dtype=torch.float64
    )
    routing_square_mass_by_category = torch.zeros(
        pool_size, categories, dtype=torch.float64
    )
    expert_output_energy = torch.zeros(
        pool_size, categories, dtype=torch.float64
    )
    linear_routing_output_energy = torch.zeros(
        pool_size, categories, dtype=torch.float64
    )
    conditional_importance = torch.zeros(
        pool_size, categories, dtype=torch.float64
    )
    active_token_count = torch.zeros(
        pool_size, categories, dtype=torch.long
    )
    structural_action_error = torch.ones(
        pool_size, len(action_names), dtype=torch.float64
    )
    structural_action_error[:, -1] = 0.0
    damage_log_sum = torch.zeros(
        pool_size, len(action_names), categories, dtype=torch.float64
    )
    damage_log_square_sum = torch.zeros_like(damage_log_sum)
    damage_positive_count = torch.zeros(
        pool_size, len(action_names), categories, dtype=torch.long
    )
    removal_depth_numerator = torch.zeros(
        pool_size, depths, categories, dtype=torch.float64
    )
    removal_depth_log_sum = torch.zeros_like(removal_depth_numerator)
    removal_depth_log_square_sum = torch.zeros_like(removal_depth_numerator)
    removal_depth_positive_count = torch.zeros(
        pool_size, depths, categories, dtype=torch.long
    )
    removal_depth_prompt_numerator = torch.zeros(
        pool_size, depths, prompts, dtype=torch.float64
    )
    prompt_action_numerator = torch.zeros(
        pool_size, len(action_names), prompts, dtype=torch.float64
    )
    routing_penalty = torch.zeros(
        pool_size, len(action_names), dtype=torch.float64
    )
    routing_penalty_gain = torch.zeros(
        pool_size, len(action_names), dtype=torch.float64
    )
    prompt_expert_output_energy = torch.zeros(
        pool_size, prompts, dtype=torch.float64
    )
    prompt_linear_routing_output_energy = torch.zeros(
        pool_size, prompts, dtype=torch.float64
    )
    prompt_active_token_count = torch.zeros(
        pool_size, prompts, dtype=torch.long
    )
    prompt_active_mask = torch.zeros(pool_size, prompts, dtype=torch.bool)
    usage_stats = {
        "topk_count": topk_count,
        "shortlist_count": shortlist_count,
        "standby_count": standby_count,
        "fallback_count": fallback_count,
        "routing_mass": torch.zeros(pool_size, dtype=torch.float64),
        "routing_square_mass": torch.zeros(pool_size, dtype=torch.float64),
        "reap_sum": torch.zeros(pool_size, dtype=torch.float64),
        "reap_mean": torch.zeros(pool_size, dtype=torch.float64),
    }
    fallback_counts = {
        "gate_up_router_conditioned": 0,
        "gate_up_weight_only": 0,
        "down_router_conditioned": 0,
        "down_weight_only": 0,
    }

    # Must match Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state exactly:
    # the plan prices the factors the deployment will build, so the shrinkage
    # target has to be identical on both sides.
    layer_gram_target = None
    layer_shrink_alpha = (
        1.0 if str(gram_shrink_target_mode) == "layer_only" else "auto"
    )
    if gram_metric and str(gram_shrink_target_mode) in ("layer", "layer_only"):
        acc = torch.zeros(
            (int(hidden.shape[1]),) * 2, device=device, dtype=torch.float32
        )
        for start in range(0, int(hidden.shape[0]), int(routing_batch_size)):
            batch = hidden[start : start + int(routing_batch_size)].to(
                device=device, dtype=torch.float32
            )
            acc += batch.t() @ batch
            del batch
        layer_gram_target = acc.cpu()
        del acc

    for local_id in tqdm(
        range(pool_size), desc=f"Full-token actions {name}", leave=False
    ):
        matches = (topk_ids == local_id).nonzero(as_tuple=False)
        active = bool(int(matches.shape[0]))
        token_ids = matches[:, 0] if active else torch.empty(0, dtype=torch.long)
        slots = matches[:, 1] if active else torch.empty(0, dtype=torch.long)
        expert_id = int(global_expert_ids[local_id].item())
        factor_token_ids, factor_slots = (factor_ids == local_id).nonzero(
            as_tuple=True
        )
        factor_x = hidden.index_select(0, factor_token_ids).to(
            device=device, dtype=torch.float32
        )
        factor_weights = factor_weight_table[
            factor_token_ids, factor_slots
        ].to(device=device, dtype=torch.float32)
        gate_weight = expert_projection_weight(module, expert_id, "gate")
        up_weight = expert_projection_weight(module, expert_id, "up")
        down_weight = expert_projection_weight(module, expert_id, "down")
        if active:
            weights = topk_weights[token_ids, slots].float()
            x = hidden.index_select(0, token_ids).to(
                device=device, dtype=torch.float32
            )
            original_y = _full_expert_output(
                x,
                gate_weight,
                up_weight,
                down_weight,
                storage_dtype=storage_dtype,
            ).float()
            token_categories = category_ids.index_select(0, token_ids)
            token_prompts = prompt_ids.index_select(0, token_ids)
            prompt_active_mask[local_id, token_prompts.unique()] = True
            routed_norm = weights.to(device) * torch.linalg.vector_norm(
                original_y, dim=1
            )
            counts = torch.bincount(
                token_categories.long(), minlength=categories
            ).long()
            importance_sum = _category_energy_sum(
                routed_norm, token_categories, categories
            )
            active_token_count[local_id] = counts
            conditional_importance[local_id] = importance_sum / counts.clamp_min(1)
            expert_output_energy[local_id] = _category_energy_sum(
                routed_norm if amplitude else routed_norm.square(),
                token_categories,
                categories,
            )
            linear_routed_energy = weights.to(device) * original_y.square().sum(dim=1)
            linear_routing_output_energy[local_id] = _category_energy_sum(
                linear_routed_energy, token_categories, categories
            )
            prompt_expert_output_energy[local_id] = torch.bincount(
                token_prompts,
                weights=(
                    routed_norm if amplitude else routed_norm.square()
                ).detach().double().cpu(),
                minlength=prompts,
            )
            prompt_linear_routing_output_energy[local_id] = torch.bincount(
                token_prompts,
                weights=linear_routed_energy.detach().double().cpu(),
                minlength=prompts,
            )
            prompt_active_token_count[local_id] = torch.bincount(
                token_prompts, minlength=prompts
            )
            usage_stats["routing_mass"][local_id] = weights.double().sum()
            square_mass = _category_energy_sum(
                weights.square(), token_categories, categories
            )
            routing_square_mass_by_category[local_id] = square_mass
            usage_stats["routing_square_mass"][local_id] = square_mass.sum()
            usage_stats["reap_sum"][local_id] = routed_norm.double().sum().cpu()
            usage_stats["reap_mean"][local_id] = routed_norm.double().mean().cpu()

            reference = reference_output.index_select(0, token_ids).to(
                device=device, dtype=torch.float32
            )
            route = weights.to(device)
            for depth in range(depths):
                replacement = replacement_output[depth].index_select(
                    0, token_ids
                ).to(device=device, dtype=torch.float32)
                replacement_route = replacement_probability[token_ids, depth].to(device)
                if bool(module.norm_topk_prob):
                    normalizer = (1.0 - route + replacement_route).clamp_min(1e-12)
                    removal_delta = (
                        (route - replacement_route).unsqueeze(1) * reference
                        - route.unsqueeze(1) * original_y
                        + replacement_route.unsqueeze(1) * replacement
                    ) / normalizer.unsqueeze(1)
                else:
                    normalizer = (1.0 - route).clamp_min(1e-12)
                    removal_delta = (
                        route.unsqueeze(1) * (reference - original_y)
                        + replacement_route.unsqueeze(1) * replacement
                    ) / normalizer.unsqueeze(1)
                removal_energy = removal_delta.square().sum(dim=1)
                removal_damage = per_token_damage(removal_energy)
                removal_depth_numerator[local_id, depth] = _category_energy_sum(
                    removal_damage, token_categories, categories
                )
                removal_depth_prompt_numerator[local_id, depth] = torch.bincount(
                    token_prompts,
                    weights=removal_damage.detach().double().cpu(),
                    minlength=prompts,
                )
                (
                    removal_depth_log_sum[local_id, depth],
                    removal_depth_log_square_sum[local_id, depth],
                    removal_depth_positive_count[local_id, depth],
                ) = _log_damage_moments(
                    removal_damage,
                    token_categories,
                    reference_mean_energy,
                    categories,
                )
                if depth == 0:
                    # Retained for routing-penalty pricing below: a token whose
                    # penalised expert loses its slot incurs exactly the damage
                    # measured here, because losing the slot to the top-(k+1)
                    # expert IS removal as far as that token is concerned.
                    displaced_energy = removal_energy.clone()
                del replacement, removal_delta, removal_energy
            # Action zero keeps the depth-zero measurement, which is what a plan
            # whose nearest fallback survives actually incurs. The retention-aware
            # planner mixes the deeper entries in later.
            action_numerator[local_id, 0] = removal_depth_numerator[local_id, 0]
            prompt_action_numerator[local_id, 0] = removal_depth_prompt_numerator[
                local_id, 0
            ]
            damage_log_sum[local_id, 0] = removal_depth_log_sum[local_id, 0]
            damage_log_square_sum[local_id, 0] = removal_depth_log_square_sum[
                local_id, 0
            ]
            damage_positive_count[local_id, 0] = removal_depth_positive_count[
                local_id, 0
            ]

        gate_fallback_recorded = False
        for action, (gate_up_rank, down_rank) in enumerate(rank_pairs, start=1):
            factors, fallbacks = build_deployed_expert_factors(
                gate_weight,
                up_weight,
                down_weight,
                gate_moments[expert_id],
                up_moments[expert_id],
                torch.zeros(int(gate_weight.shape[0]), dtype=torch.float32),
                expert_id=expert_id,
                gate_up_rank=gate_up_rank,
                down_rank=down_rank,
                layer_seed=seed,
                candidate_inputs=factor_x,
                candidate_weights=factor_weights,
                scale_floor=scale_floor,
                randomized=randomized,
                storage_dtype=storage_dtype,
                gram_metric=gram_metric,
                allow_uncalibrated=True,
                # Must match the deployment's shrinkage target exactly: the plan
                # prices the factors the restructurer will build, and pricing them
                # against a different target is the plan/deployment divergence that
                # made the first gram run ship uncalibrated factors.
                gram_shrink_target=layer_gram_target,
                gram_shrink_alpha=layer_shrink_alpha,
                down_compensation=down_compensation,
                use_recorded_moments=use_recorded_moments,
                output_metric=factor_output_metric,
            )
            structural_action_error[local_id, action] = (
                _deployed_factor_structural_error(
                    gate_weight,
                    up_weight,
                    down_weight,
                    factors,
                    down_target=fallbacks.get("down_target"),
                )
            )
            if active:
                compressed_y, _ = apply_deployed_expert_factors(x, factors)
                delta = route.unsqueeze(1) * (compressed_y.float() - original_y)
                delta_energy = delta.square().sum(dim=1)
                if routing_penalty_calibrate:
                    margin = (
                        topk_values[token_ids, slots].to(device)
                        - replacement_values[token_ids, 0].to(device)
                    )
                    (
                        routing_penalty[local_id, action],
                        routing_penalty_gain[local_id, action],
                    ) = fit_routing_penalty_threshold(
                        margin, delta_energy, displaced_energy
                    )
                if float(routing_penalty_scale) > 0.0:
                    # Deployment discounts this expert's router logit by lam, so
                    # it keeps a slot only where its margin over the top-(k+1)
                    # expert survives the discount. Pricing the action as if it
                    # still served every token would price a plan the model will
                    # never run. Tokens it loses are priced as removal, which is
                    # what they actually incur.
                    #
                    # lam is self-normalising -- the expert's own relative damage
                    # amplitude -- so it needs no layer-wide statistic and can be
                    # recomputed at deployment from quantities the plan records.
                    if routing_penalty_calibrate:
                        # scale shrinks the FITTED threshold toward zero. The fit
                        # is a per-expert argmin over calibration tokens, so it
                        # is the term most exposed to overfitting them; scale = 1
                        # trusts it fully, below 1 is the conservative reading.
                        lam = float(routing_penalty_scale) * float(
                            routing_penalty[local_id, action]
                        )
                    else:
                        relative = (
                            delta_energy.sum()
                            / routed_norm.square().sum().clamp_min(1e-30)
                        )
                        lam = float(
                            routing_penalty_scale
                            * float(relative.clamp_min(0).sqrt())
                        )
                        routing_penalty[local_id, action] = lam
                    margin = (
                        topk_values[token_ids, slots].to(device)
                        - replacement_values[token_ids, 0].to(device)
                    )
                    # Each expert is priced as the only one discounted, matching
                    # the additive treatment removal already gets in this trace.
                    delta_energy = torch.where(
                        margin < lam, displaced_energy, delta_energy
                    )
                delta_damage = per_token_damage(delta_energy)
                action_numerator[local_id, action] = _category_energy_sum(
                    delta_damage, token_categories, categories
                )
                prompt_action_numerator[local_id, action] = torch.bincount(
                    token_prompts,
                    weights=delta_damage.detach().double().cpu(),
                    minlength=prompts,
                )
                (
                    damage_log_sum[local_id, action],
                    damage_log_square_sum[local_id, action],
                    damage_positive_count[local_id, action],
                ) = _log_damage_moments(
                    delta_damage,
                    token_categories,
                    reference_mean_energy,
                    categories,
                )
            if not gate_fallback_recorded:
                fallback_counts["gate_up_router_conditioned"] += int(
                    fallbacks["gate_up_router_conditioned"]
                )
                fallback_counts["gate_up_weight_only"] += int(
                    fallbacks["gate_up_weight_only"]
                )
                if fallbacks["gate_up_router_conditioned"]:
                    logging.debug(
                        "%s expert %d has a zero recorded gate/up moment; "
                        "router-conditioned fallback uses %d top-%d token states "
                        "(%d active top-%d states)",
                        name,
                        expert_id,
                        int(factor_token_ids.numel()),
                        shortlist,
                        int(token_ids.numel()),
                        top_k,
                    )
                gate_fallback_recorded = True
            fallback_counts["down_weight_only"] += int(
                fallbacks["down_weight_only"]
            )
            del factors
            if active:
                del compressed_y, delta, delta_energy
        del factor_x, factor_weights
        if active:
            del x, original_y, reference

    # The measured single-action error already corresponds to alpha=1 because
    # it is summed over active tokens.  Re-express it as
    # p_e**alpha * conditional_error by multiplying with p_e**(alpha-1),
    # where p_e is the category-local share of squared routing mass.  Experts
    # with no mass also have zero measured action error and remain zero.
    raw_action_numerator = action_numerator.clone()
    if float(occupancy_alpha) != 1.0:
        total_mass = routing_square_mass_by_category.sum(dim=0).clamp_min(1e-30)
        occupancy = routing_square_mass_by_category / total_mass.unsqueeze(0)
        scale = torch.zeros_like(occupancy)
        observed = occupancy > 0
        scale[observed] = occupancy[observed].pow(float(occupancy_alpha) - 1.0)
        action_numerator.mul_(scale.unsqueeze(1))

    # The deployment recomputes lam from the plan alone, so it needs the same
    # denominator this trace divided by. Recorded relative to the layer reference
    # energy, exactly as single_action_relative_distortion is, so the reference
    # cancels when the two are divided and neither has to be shipped.
    usage_stats["routed_output_energy"] = (
        expert_output_energy.sum(dim=1).double()
        / float(reference_energy.sum().clamp_min(1e-30))
    )
    if routing_penalty_calibrate or float(routing_penalty_scale) > 0.0:
        for action_id, action_label in enumerate(action_names):
            if action_label in ("removed", "full"):
                continue
            usage_stats[f"routing_penalty_{action_label}"] = routing_penalty[
                :, action_id
            ]
            if routing_penalty_calibrate:
                # The fraction of this action's damage the best threshold
                # removes. Zero means no threshold beats keeping every token,
                # which is the honest per-expert verdict on the whole idea and
                # is readable without running a benchmark.
                usage_stats[f"routing_penalty_gain_{action_label}"] = (
                    routing_penalty_gain[:, action_id]
                )

    trace = LayerIndependentActionTrace(
        name=name,
        action_numerator=action_numerator,
        raw_action_numerator=raw_action_numerator,
        expert_output_energy=expert_output_energy,
        linear_routing_output_energy=linear_routing_output_energy,
        conditional_importance=conditional_importance,
        active_token_count=active_token_count,
        structural_action_error=structural_action_error,
        prompt_action_numerator=prompt_action_numerator,
        prompt_expert_output_energy=prompt_expert_output_energy,
        prompt_linear_routing_output_energy=prompt_linear_routing_output_energy,
        prompt_active_token_count=prompt_active_token_count,
        prompt_reference_energy=prompt_reference_energy,
        prompt_category_ids=prompt_category_ids,
        prompt_active_mask=prompt_active_mask,
        damage_log_sum=damage_log_sum,
        damage_log_square_sum=damage_log_square_sum,
        damage_positive_count=damage_positive_count,
        removal_depth_numerator=removal_depth_numerator,
        removal_depth_log_sum=removal_depth_log_sum,
        removal_depth_log_square_sum=removal_depth_log_square_sum,
        removal_depth_positive_count=removal_depth_positive_count,
        removal_depth_prompt_numerator=removal_depth_prompt_numerator,
        reference_energy=reference_energy,
        category_names=category_names,
        action_costs=action_cost.unsqueeze(0).expand(pool_size, -1).clone(),
        action_names=action_names,
        top_k=top_k,
        damage_unit=str(damage_unit),
    )
    if any(fallback_counts.values()):
        logging.debug("%s calibration fallbacks: %s", name, fallback_counts)
    return trace, global_expert_ids, fallback_counts, usage_stats


@dataclass
class _ConditionedLayerScores:
    """Exact current-plan report and adjacent one-expert alternatives."""

    base: DistortionReport
    candidates: dict[tuple[int, int], DistortionReport]


def _ordered_retained_routes(
    routing_order: torch.Tensor,
    retained: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select retained experts without re-resolving router-logit ties."""

    available = retained[routing_order]
    if int(available.sum(dim=1).min().item()) < int(count):
        raise ValueError("retained routing pool has fewer requested experts")
    positions = torch.arange(
        int(routing_order.shape[1]), dtype=torch.long
    ).unsqueeze(0).expand_as(routing_order)
    positions = positions.masked_fill(~available, int(routing_order.shape[1]))
    selected = torch.topk(positions, k=int(count), largest=False, sorted=True).values
    return routing_order.gather(1, selected), selected


@torch.no_grad()
def _build_conditioned_factor_bank(
    module,
    hidden: torch.Tensor,
    calibration_ids: torch.Tensor,
    calibration_weights: torch.Tensor,
    global_expert_ids: torch.Tensor,
    actions: torch.Tensor,
    gate_moments: torch.Tensor,
    up_moments: torch.Tensor,
    down_moments: torch.Tensor,
    rank_pairs: Sequence[tuple[int, int]],
    *,
    device: str,
    randomized: bool,
    gram_metric: bool = False,
    use_recorded_moments: bool = True,
    factor_output_metric: str = "none",
    scale_floor: float,
    seed: int,
) -> tuple[dict[tuple[int, int], dict[str, torch.Tensor]], dict[str, int]]:
    """Materialize only low-rank factors needed by adjacent refinement moves.

    ``calibration_weights`` is already the per-token weight (see
    ``factor_calibration_selection``), not a probability still to be squared.
    """

    full_action = len(rank_pairs) + 1
    factor_bank: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    fallbacks = {
        "gate_up_router_conditioned": 0,
        "gate_up_weight_only": 0,
        "down_weight_only": 0,
    }
    for local_id in range(int(actions.numel())):
        current = int(actions[local_id].item())
        needed = {
            value
            for value in (current - 1, current, current + 1)
            if 0 < value < full_action
        }
        if not needed:
            continue
        token_ids, positions = (calibration_ids == local_id).nonzero(as_tuple=True)
        candidate_inputs = candidate_weights = None
        if int(token_ids.numel()):
            candidate_inputs = hidden.index_select(0, token_ids).to(
                device=device, dtype=torch.float32
            )
            candidate_weights = calibration_weights[
                token_ids, positions
            ].to(device=device, dtype=torch.float32)
        expert_id = int(global_expert_ids[local_id].item())
        gate_weight = expert_projection_weight(module, expert_id, "gate")
        up_weight = expert_projection_weight(module, expert_id, "up")
        down_weight = expert_projection_weight(module, expert_id, "down")
        for action in sorted(needed):
            gate_up_rank, down_rank = rank_pairs[action - 1]
            factors, used = build_deployed_expert_factors(
                gate_weight,
                up_weight,
                down_weight,
                gate_moments[expert_id],
                up_moments[expert_id],
                down_moments[expert_id],
                expert_id=expert_id,
                gate_up_rank=int(gate_up_rank),
                down_rank=int(down_rank),
                layer_seed=int(seed),
                candidate_inputs=candidate_inputs,
                candidate_weights=candidate_weights,
                scale_floor=scale_floor,
                randomized=randomized,
                storage_dtype=gate_weight.dtype,
                gram_metric=gram_metric,
                allow_uncalibrated=True,
                use_recorded_moments=use_recorded_moments,
                output_metric=factor_output_metric,
            )
            factor_bank[(local_id, action)] = factors
            for key in fallbacks:
                fallbacks[key] += int(used[key])
        del candidate_inputs, candidate_weights
    return factor_bank, fallbacks


@torch.no_grad()
def _conditioned_action_output(
    module,
    global_expert_ids: torch.Tensor,
    factor_bank: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
    rank_pairs: Sequence[tuple[int, int]],
    local_id: int,
    action: int,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply one expert representation exactly as deployed."""

    full_action = len(rank_pairs) + 1
    if int(action) == full_action:
        expert_id = int(global_expert_ids[int(local_id)].item())
        return _full_expert_output(
            inputs,
            expert_projection_weight(module, expert_id, "gate"),
            expert_projection_weight(module, expert_id, "up"),
            expert_projection_weight(module, expert_id, "down"),
            storage_dtype=expert_projection_weight(module, expert_id, "gate").dtype,
        ).float()
    if not (0 < int(action) < full_action):
        raise ValueError("conditioned expert output requires a retained action")
    output, _ = apply_deployed_expert_factors(
        inputs, dict(factor_bank[(int(local_id), int(action))])
    )
    return output.float()


@torch.no_grad()
def _build_layer_rerouting_conditioned_scores(
    module,
    name: str,
    hidden: torch.Tensor,
    category_ids: torch.Tensor,
    category_names: Sequence[str],
    gate_moments: torch.Tensor,
    up_moments: torch.Tensor,
    down_moments: torch.Tensor,
    rank_pairs: Sequence[tuple[int, int]],
    actions: torch.Tensor,
    *,
    device: str,
    routing_batch_size: int,
    router_candidates: int,
    refinement_chunk_size: int,
    randomized: bool,
    gram_metric: bool = False,
    factor_calibration: str = "shortlist",
    factor_output_metric: str = "none",
    scale_floor: float,
    seed: int,
) -> tuple[_ConditionedLayerScores, torch.Tensor, dict[str, int]]:
    """Measure adjacent actions under routing induced by the current plan.

    This evaluator consumes every cached token in chunks. It stores router
    ordering and one layer's factor bank, but no token-by-expert outputs or
    token Gram matrices. Qwen3's normalized top-k routing permits exact
    one-expert removal/addition updates from the baseline aggregate.
    """

    if not bool(module.norm_topk_prob):
        raise NotImplementedError(
            "rerouting-conditioned refinement currently requires normalized top-k routing"
        )
    logits, global_expert_ids = _pool_router_logits(
        module, hidden, device=device, batch_size=routing_batch_size
    )
    tokens, pool_size = map(int, logits.shape)
    actions = torch.as_tensor(actions, dtype=torch.long).reshape(-1)
    if int(actions.numel()) != pool_size:
        raise ValueError(f"{name}: action count does not match routing pool")
    top_k = int(module.top_k)
    retained = actions != 0
    if int(retained.sum().item()) < top_k:
        raise ValueError(f"{name}: current plan retains fewer than top-k experts")

    # A complete, once-resolved ordering makes additions/removals deterministic
    # even when BF16 router logits contain exact ties.
    routing_order = torch.topk(logits, k=pool_size, dim=1).indices
    rank_position = torch.empty_like(routing_order)
    rank_position.scatter_(
        1,
        routing_order,
        torch.arange(pool_size).unsqueeze(0).expand(tokens, -1),
    )
    route_count = top_k + int(int(retained.sum().item()) > top_k)
    current_ordered, current_positions = _ordered_retained_routes(
        routing_order, retained, route_count
    )
    current_ids = current_ordered[:, :top_k]
    current_position = current_positions[:, :top_k]
    current_values = logits.gather(1, current_ids)
    current_weights = F.softmax(current_values, dim=1)
    current_log_normalizer = torch.logsumexp(current_values, dim=1)
    standby_ids = (
        current_ordered[:, top_k]
        if route_count > top_k
        else torch.full((tokens,), -1, dtype=torch.long)
    )
    standby_values = (
        logits.gather(1, standby_ids.unsqueeze(1)).squeeze(1)
        if route_count > top_k
        else torch.full((tokens,), -torch.inf)
    )
    reference_ids = routing_order[:, :top_k]

    shortlist = min(max(int(router_candidates), top_k), pool_size)
    # Match the selected-factor artifact builder exactly: its calibration pool
    # is produced by a direct top-L call, independently of the full ordering
    # retained for conditioned rerouting.
    shortlist_values, shortlist_ids = torch.topk(logits, k=shortlist, dim=1)
    shortlist_probability = F.softmax(shortlist_values, dim=1)
    if str(factor_calibration) == "shortlist":
        calibration_ids = shortlist_ids
        calibration_weights = shortlist_probability.square()
    else:
        calibration_ids, calibration_weights = factor_calibration_selection(
            logits,
            factor_calibration=factor_calibration,
            top_k=top_k,
            norm_topk_prob=bool(module.norm_topk_prob),
            router_candidates=router_candidates,
            # The current plan's routing, as the restructurer would deploy it.
            retained=retained,
        )
    factor_bank, fallback_counts = _build_conditioned_factor_bank(
        module,
        hidden,
        calibration_ids,
        calibration_weights,
        global_expert_ids,
        actions,
        gate_moments,
        up_moments,
        down_moments,
        rank_pairs,
        device=device,
        randomized=randomized,
        gram_metric=gram_metric,
        use_recorded_moments=str(factor_calibration) == "shortlist",
        factor_output_metric=factor_output_metric,
        scale_floor=scale_floor,
        seed=seed,
    )

    categories = len(category_names)
    reference_sum = torch.zeros(categories, dtype=torch.float64)
    base_sum = torch.zeros(categories, dtype=torch.float64)
    base_route_changes = 0.0
    valid_candidates = {
        (expert, new_action)
        for expert in range(pool_size)
        for new_action in (
            int(actions[expert].item()) - 1,
            int(actions[expert].item()) + 1,
        )
        if 0 <= new_action <= len(rank_pairs) + 1
        and new_action != int(actions[expert].item())
        and not (
            new_action == 0 and int(retained.sum().item()) <= top_k
        )
    }
    candidate_sum = {
        key: torch.zeros(categories, dtype=torch.float64)
        for key in valid_candidates
    }
    candidate_route_changes = {key: 0.0 for key in valid_candidates}
    storage_dtype = expert_projection_weight(
        module, int(global_expert_ids[0].item()), "gate"
    ).dtype

    for start in range(0, tokens, int(refinement_chunk_size)):
        stop = min(tokens, start + int(refinement_chunk_size))
        ids = torch.arange(start, stop, dtype=torch.long)
        batch = hidden[start:stop].to(device=device, dtype=torch.float32)
        count = stop - start
        selected = current_ids[start:stop]
        selected_weights = current_weights[start:stop].to(device)
        baseline = torch.zeros(
            count, int(hidden.shape[1]), device=device, dtype=torch.float32
        )
        bottom_output = torch.empty_like(baseline)
        for local_id in torch.unique(selected).tolist():
            rows, slots = (selected == int(local_id)).nonzero(as_tuple=True)
            device_rows = rows.to(device)
            device_slots = slots.to(device)
            value = _conditioned_action_output(
                module,
                global_expert_ids,
                factor_bank,
                rank_pairs,
                int(local_id),
                int(actions[int(local_id)].item()),
                batch.index_select(0, device_rows),
            )
            baseline.index_add_(
                0,
                device_rows,
                selected_weights[device_rows, device_slots].unsqueeze(1) * value,
            )
            bottom = slots == top_k - 1
            if bool(bottom.any()):
                device_bottom = bottom.to(device)
                bottom_output.index_copy_(
                    0, rows[bottom].to(device), value[device_bottom]
                )

        standby_output = None
        if route_count > top_k:
            standby = standby_ids[start:stop]
            standby_output = torch.empty_like(baseline)
            for local_id in torch.unique(standby).tolist():
                rows = (standby == int(local_id)).nonzero(as_tuple=True)[0]
                device_rows = rows.to(device)
                value = _conditioned_action_output(
                    module,
                    global_expert_ids,
                    factor_bank,
                    rank_pairs,
                    int(local_id),
                    int(actions[int(local_id)].item()),
                    batch.index_select(0, device_rows),
                )
                standby_output.index_copy_(0, device_rows, value)

        reference = module(batch.to(storage_dtype).unsqueeze(0)).reshape(
            count, -1
        ).float()
        reference_energy = reference.square().sum(1)
        base_error = (reference - baseline).square().sum(1)
        chunk_categories = category_ids.index_select(0, ids)
        reference_sum += _category_energy_sum(
            reference_energy, chunk_categories, categories
        )
        base_by_category = _category_energy_sum(
            base_error, chunk_categories, categories
        )
        base_sum += base_by_category
        current_changed = (
            torch.sort(selected, dim=1).values
            != torch.sort(reference_ids[start:stop], dim=1).values
        ).any(dim=1)
        base_changed_count = float(current_changed.sum().item())
        base_route_changes += base_changed_count

        for key in valid_candidates:
            local_id, new_action = key
            old_action = int(actions[local_id].item())
            candidate_sum[key] += base_by_category
            candidate_route_changes[key] += base_changed_count
            if old_action > 0 and new_action > 0:
                rows, slots = (selected == local_id).nonzero(as_tuple=True)
                if not int(rows.numel()):
                    continue
                device_rows = rows.to(device)
                device_slots = slots.to(device)
                inputs = batch.index_select(0, device_rows)
                old_output = _conditioned_action_output(
                    module, global_expert_ids, factor_bank, rank_pairs,
                    local_id, old_action, inputs,
                )
                new_output = _conditioned_action_output(
                    module, global_expert_ids, factor_bank, rank_pairs,
                    local_id, new_action, inputs,
                )
                candidate = baseline.index_select(0, device_rows) + selected_weights[
                    device_rows, device_slots
                ].unsqueeze(1) * (new_output - old_output)
                new_error = (
                    reference.index_select(0, device_rows) - candidate
                ).square().sum(1)
                delta = new_error - base_error.index_select(0, device_rows)
                candidate_sum[key] += _category_energy_sum(
                    delta, chunk_categories.index_select(0, rows), categories
                )
                continue

            if new_action == 0:
                if standby_output is None:
                    continue
                rows, slots = (selected == local_id).nonzero(as_tuple=True)
                if not int(rows.numel()):
                    continue
                device_rows = rows.to(device)
                device_slots = slots.to(device)
                inputs = batch.index_select(0, device_rows)
                old_output = _conditioned_action_output(
                    module, global_expert_ids, factor_bank, rank_pairs,
                    local_id, old_action, inputs,
                )
                route = selected_weights[device_rows, device_slots]
                replacement_ratio = torch.exp(
                    standby_values[start:stop].index_select(0, rows).to(device)
                    - current_log_normalizer[start:stop]
                    .index_select(0, rows).to(device)
                )
                normalizer = (1.0 - route + replacement_ratio).clamp_min(1e-12)
                candidate = (
                    baseline.index_select(0, device_rows)
                    - route.unsqueeze(1) * old_output
                    + replacement_ratio.unsqueeze(1)
                    * standby_output.index_select(0, device_rows)
                ) / normalizer.unsqueeze(1)
                new_error = (
                    reference.index_select(0, device_rows) - candidate
                ).square().sum(1)
                delta = new_error - base_error.index_select(0, device_rows)
                candidate_sum[key] += _category_energy_sum(
                    delta, chunk_categories.index_select(0, rows), categories
                )
                candidate_ids = selected.index_select(0, rows).clone()
                candidate_ids[torch.arange(int(rows.numel())), slots] = (
                    standby_ids[start:stop].index_select(0, rows)
                )
            else:
                # The expert is currently removed; only tokens where it outranks
                # the current kth retained expert change their routed output.
                enters = rank_position[start:stop, local_id] < current_position[
                    start:stop, -1
                ]
                rows = enters.nonzero(as_tuple=True)[0]
                if not int(rows.numel()):
                    continue
                device_rows = rows.to(device)
                new_output = _conditioned_action_output(
                    module,
                    global_expert_ids,
                    factor_bank,
                    rank_pairs,
                    local_id,
                    new_action,
                    batch.index_select(0, device_rows),
                )
                bottom_weight = selected_weights[device_rows, -1]
                added_ratio = torch.exp(
                    logits[start:stop, local_id].index_select(0, rows).to(device)
                    - current_log_normalizer[start:stop]
                    .index_select(0, rows).to(device)
                )
                normalizer = (
                    1.0 - bottom_weight + added_ratio
                ).clamp_min(1e-12)
                candidate = (
                    baseline.index_select(0, device_rows)
                    - bottom_weight.unsqueeze(1)
                    * bottom_output.index_select(0, device_rows)
                    + added_ratio.unsqueeze(1) * new_output
                ) / normalizer.unsqueeze(1)
                new_error = (
                    reference.index_select(0, device_rows) - candidate
                ).square().sum(1)
                delta = new_error - base_error.index_select(0, device_rows)
                candidate_sum[key] += _category_energy_sum(
                    delta, chunk_categories.index_select(0, rows), categories
                )
                candidate_ids = selected.index_select(0, rows).clone()
                candidate_ids[:, -1] = int(local_id)

            reference_changed = current_changed.index_select(0, rows)
            new_changed = (
                torch.sort(candidate_ids, dim=1).values
                != torch.sort(
                    reference_ids[start:stop].index_select(0, rows), dim=1
                ).values
            ).any(dim=1)
            candidate_route_changes[key] += float(
                new_changed.sum().item() - reference_changed.sum().item()
            )
        del baseline, bottom_output, standby_output, reference, batch

    base = DistortionReport(
        numerator=base_sum,
        denominator=reference_sum,
        relative=base_sum / reference_sum.clamp_min(1e-30),
        route_change_rate=float(base_route_changes / max(tokens, 1)),
    )
    candidates = {
        key: DistortionReport(
            numerator=value.clamp_min(0),
            denominator=reference_sum.clone(),
            relative=value.clamp_min(0) / reference_sum.clamp_min(1e-30),
            route_change_rate=float(candidate_route_changes[key] / max(tokens, 1)),
        )
        for key, value in candidate_sum.items()
    }
    del factor_bank
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _ConditionedLayerScores(base=base, candidates=candidates), global_expert_ids, fallback_counts


def _conditioned_component(
    report: DistortionReport, layer_reduction: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_reduction == "energy":
        return report.numerator.double(), report.denominator.double()
    if layer_reduction == "mean":
        return report.relative.double(), torch.ones_like(report.relative).double()
    raise ValueError("layer_reduction must be energy or mean")


def _reassign_plan_tiers_by_saliency(
    plan: FunctionalCompressionPlan,
    traces: Sequence[LayerIndependentActionTrace],
    usage_stats: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    saliency_key: str,
    objective: str,
    cvar_fraction: float,
    layer_reduction: str,
) -> FunctionalCompressionPlan:
    """Preserve every layer's action histogram but rank experts by saliency."""

    reassigned: dict[str, torch.Tensor] = {}
    numerator = denominator = None
    category_names = tuple(traces[0].category_names)
    memory = 0
    for trace in traces:
        current = plan.actions[trace.name].long()
        try:
            saliency = torch.as_tensor(usage_stats[trace.name][saliency_key]).double()
        except KeyError as error:
            raise ValueError(
                f"{trace.name}: tier assignment needs usage statistic {saliency_key!r}"
            ) from error
        order = sorted(
            range(trace.experts), key=lambda expert: (float(saliency[expert]), expert)
        )
        actions = torch.empty_like(current)
        cursor = 0
        for action in range(trace.actions):
            count = int((current == action).sum().item())
            if count:
                actions[torch.tensor(order[cursor : cursor + count])] = action
                cursor += count
        if cursor != trace.experts:
            raise RuntimeError(f"{trace.name}: failed to reassign every expert")
        reassigned[trace.name] = actions
        memory += trace.memory(actions)
        report = trace.evaluate(actions)
        if layer_reduction == "energy":
            layer_numerator, layer_denominator = report.numerator, report.denominator
        elif layer_reduction == "mean":
            layer_numerator = report.relative
            layer_denominator = torch.ones_like(report.relative)
        else:
            raise ValueError("layer_reduction must be energy or mean")
        numerator = layer_numerator if numerator is None else numerator + layer_numerator
        denominator = (
            layer_denominator if denominator is None else denominator + layer_denominator
        )
    if memory != plan.memory:
        raise RuntimeError("saliency tier reassignment changed plan memory")
    score, relative = robust_objective(
        numerator, denominator, method=objective, cvar_fraction=cvar_fraction
    )
    history = list(plan.history)
    history.append(
        {
            "operation": "reassign_tiers_by_saliency",
            "saliency": saliency_key,
            "memory": int(memory),
            "objective": float(score),
        }
    )
    return FunctionalCompressionPlan(
        actions=reassigned,
        memory=int(memory),
        objective=float(score),
        category_relative_distortion={
            category: float(relative[index].item())
            for index, category in enumerate(category_names)
        },
        history=history,
    )


@torch.no_grad()
def _rerouting_conditioned_refinement(
    initial_plan: FunctionalCompressionPlan,
    traces: Sequence[LayerIndependentActionTrace],
    modules: Mapping[str, object],
    hidden_metadata: Mapping,
    row_ids: torch.Tensor,
    category_ids: torch.Tensor,
    category_names: Sequence[str],
    gate_moments: Mapping[str, torch.Tensor],
    up_moments: Mapping[str, torch.Tensor],
    down_moments: Mapping[str, torch.Tensor],
    rank_pairs: Sequence[tuple[int, int]],
    *,
    objective: str,
    cvar_fraction: float,
    layer_reduction: str,
    min_retained_experts: int,
    min_layer_memory_ratio: float,
    max_layer_relative_distortion: float | None,
    rounds: int,
    swaps_per_round: int,
    candidates_per_boundary: int,
    device: str,
    routing_batch_size: int,
    router_candidates: int,
    refinement_chunk_size: int,
    randomized: bool,
    gram_metric: bool = False,
    factor_calibration: str = "shortlist",
    factor_output_metric: str = "none",
    scale_floor: float,
    seed: int,
) -> tuple[FunctionalCompressionPlan, dict[str, DistortionReport], dict]:
    """Refine a plan with exact current-routing, memory-neutral swaps.

    Each accepted exchange upgrades one expert and downgrades another across
    the same adjacent action boundary. Their parameter changes cancel exactly.
    A layer is touched at most once per round, so both conditioned reports in a
    swap remain exact. The next round replays the resulting routing pools.
    """

    traces = list(traces)
    trace_by_name = {trace.name: trace for trace in traces}
    actions = {
        name: value.detach().long().cpu().clone()
        for name, value in initial_plan.actions.items()
    }
    initial_memory = sum(
        trace.memory(actions[trace.name]) for trace in traces
    )
    history = list(initial_plan.history)
    reports: dict[str, DistortionReport] = {}
    refinement_fallbacks: dict[str, dict[str, int]] = {}
    round_summaries = []

    for round_index in range(int(rounds)):
        scores_by_layer: dict[str, _ConditionedLayerScores] = {}
        for layer_index, trace in enumerate(traces):
            name = trace.name
            logging.info(
                "Rerouting-conditioned refinement round %d: %s",
                round_index + 1,
                name,
            )
            all_hidden = _load_hidden_cache(hidden_metadata, name)
            hidden = _select_hidden_rows_without_full_copy(all_hidden, row_ids)
            del all_hidden
            scores, expert_ids, fallbacks = _build_layer_rerouting_conditioned_scores(
                modules[name],
                name,
                hidden,
                category_ids,
                category_names,
                gate_moments[name],
                up_moments[name],
                down_moments[name],
                rank_pairs,
                actions[name],
                device=device,
                routing_batch_size=routing_batch_size,
                router_candidates=router_candidates,
                refinement_chunk_size=refinement_chunk_size,
                randomized=randomized,
                gram_metric=gram_metric,
                factor_calibration=factor_calibration,
                factor_output_metric=factor_output_metric,
                scale_floor=scale_floor,
                seed=int(seed) + layer_index * 1_000_003,
            )
            expected = torch.arange(trace.experts, dtype=torch.long)
            if not torch.equal(expert_ids.cpu(), expected):
                raise ValueError(
                    f"{name}: rerouting refinement currently requires the full expert pool"
                )
            scores_by_layer[name] = scores
            reports[name] = scores.base
            refinement_fallbacks[name] = fallbacks
            del hidden
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        components = {
            name: _conditioned_component(report, layer_reduction)
            for name, report in reports.items()
        }
        global_numerator = sum(
            (value[0] for value in components.values()),
            torch.zeros(len(category_names), dtype=torch.float64),
        )
        global_denominator = sum(
            (value[1] for value in components.values()),
            torch.zeros(len(category_names), dtype=torch.float64),
        )
        current_objective, current_relative = robust_objective(
            global_numerator,
            global_denominator,
            method=objective,
            cvar_fraction=cvar_fraction,
        )
        round_start_objective = current_objective
        used_layers: set[str] = set()
        accepted = 0

        while accepted < int(swaps_per_round):
            upgrades: dict[int, list[dict]] = {}
            downgrades: dict[int, list[dict]] = {}
            for trace in traces:
                name = trace.name
                if name in used_layers:
                    continue
                current = actions[name]
                current_memory = trace.memory(current)
                full_memory = trace.memory(
                    torch.full((trace.experts,), trace.actions - 1, dtype=torch.long)
                )
                memory_floor = ceil(float(min_layer_memory_ratio) * full_memory)
                retained = int((current != 0).sum().item())
                retained_floor = max(trace.top_k, int(min_retained_experts))
                for (expert, new_action), report in scores_by_layer[
                    name
                ].candidates.items():
                    old_action = int(current[expert].item())
                    if abs(int(new_action) - old_action) != 1:
                        continue
                    old_cost = int(trace.action_costs[expert, old_action].item())
                    new_cost = int(trace.action_costs[expert, new_action].item())
                    delta_cost = new_cost - old_cost
                    if delta_cost == 0:
                        continue
                    if new_action == 0 and retained <= retained_floor:
                        continue
                    if current_memory + delta_cost < memory_floor:
                        continue
                    if max_layer_relative_distortion is not None and float(
                        report.relative.max().item()
                    ) > float(max_layer_relative_distortion):
                        continue
                    candidate_component = _conditioned_component(
                        report, layer_reduction
                    )
                    candidate_numerator = (
                        global_numerator
                        - components[name][0]
                        + candidate_component[0]
                    )
                    candidate_denominator = (
                        global_denominator
                        - components[name][1]
                        + candidate_component[1]
                    )
                    standalone, _ = robust_objective(
                        candidate_numerator,
                        candidate_denominator,
                        method=objective,
                        cvar_fraction=cvar_fraction,
                    )
                    entry = {
                        "layer": name,
                        "expert": int(expert),
                        "old_action": old_action,
                        "new_action": int(new_action),
                        "cost": abs(delta_cost),
                        "report": report,
                        "component": candidate_component,
                        "standalone": standalone,
                    }
                    boundary = min(old_action, int(new_action))
                    target = upgrades if delta_cost > 0 else downgrades
                    target.setdefault(boundary, []).append(entry)

            for values in upgrades.values():
                values.sort(key=lambda value: value["standalone"])
                del values[int(candidates_per_boundary) :]
            for values in downgrades.values():
                values.sort(key=lambda value: value["standalone"])
                del values[int(candidates_per_boundary) :]

            best = None
            for boundary in sorted(set(upgrades) & set(downgrades)):
                for upgrade in upgrades[boundary]:
                    for downgrade in downgrades[boundary]:
                        if upgrade["layer"] == downgrade["layer"]:
                            continue
                        if upgrade["cost"] != downgrade["cost"]:
                            continue
                        pair_numerator = (
                            global_numerator
                            - components[upgrade["layer"]][0]
                            - components[downgrade["layer"]][0]
                            + upgrade["component"][0]
                            + downgrade["component"][0]
                        )
                        pair_denominator = (
                            global_denominator
                            - components[upgrade["layer"]][1]
                            - components[downgrade["layer"]][1]
                            + upgrade["component"][1]
                            + downgrade["component"][1]
                        )
                        pair_objective, pair_relative = robust_objective(
                            pair_numerator,
                            pair_denominator,
                            method=objective,
                            cvar_fraction=cvar_fraction,
                        )
                        if pair_objective >= current_objective - 1e-12:
                            continue
                        comparison = (
                            pair_objective,
                            upgrade["layer"],
                            upgrade["expert"],
                            downgrade["layer"],
                            downgrade["expert"],
                        )
                        if best is None or comparison < best[0]:
                            best = (
                                comparison,
                                upgrade,
                                downgrade,
                                pair_numerator,
                                pair_denominator,
                                pair_relative,
                            )
            if best is None:
                break

            _, upgrade, downgrade, global_numerator, global_denominator, current_relative = best
            previous_objective = current_objective
            current_objective = float(best[0][0])
            for change in (upgrade, downgrade):
                name = change["layer"]
                actions[name][change["expert"]] = change["new_action"]
                reports[name] = change["report"]
                components[name] = change["component"]
                used_layers.add(name)
            accepted += 1
            history.append(
                {
                    "kind": "rerouting_conditioned_memory_neutral_swap",
                    "round": round_index + 1,
                    "upgrade": {
                        "layer": upgrade["layer"],
                        "expert": upgrade["expert"],
                        "from": trace_by_name[upgrade["layer"]].action_names[
                            upgrade["old_action"]
                        ],
                        "to": trace_by_name[upgrade["layer"]].action_names[
                            upgrade["new_action"]
                        ],
                    },
                    "downgrade": {
                        "layer": downgrade["layer"],
                        "expert": downgrade["expert"],
                        "from": trace_by_name[downgrade["layer"]].action_names[
                            downgrade["old_action"]
                        ],
                        "to": trace_by_name[downgrade["layer"]].action_names[
                            downgrade["new_action"]
                        ],
                    },
                    "parameters_exchanged": int(upgrade["cost"]),
                    "objective_before": float(previous_objective),
                    "objective_after": float(current_objective),
                }
            )

        round_summaries.append(
            {
                "round": round_index + 1,
                "objective_before": float(round_start_objective),
                "objective_after": float(current_objective),
                "accepted_swaps": int(accepted),
            }
        )
        logging.info(
            "Rerouting-conditioned round %d accepted %d swaps: %.6g -> %.6g",
            round_index + 1,
            accepted,
            round_start_objective,
            current_objective,
        )
        if not accepted:
            break

    final_memory = sum(trace.memory(actions[trace.name]) for trace in traces)
    if final_memory != initial_memory:
        raise RuntimeError(
            "rerouting-conditioned refinement changed the global memory budget"
        )
    final_plan = FunctionalCompressionPlan(
        actions={name: value.clone() for name, value in actions.items()},
        memory=int(final_memory),
        objective=float(current_objective),
        category_relative_distortion={
            category: float(current_relative[index].item())
            for index, category in enumerate(category_names)
        },
        history=history,
    )
    return final_plan, reports, {
        "rounds": round_summaries,
        "fallbacks_by_layer": refinement_fallbacks,
        "initial_memory": int(initial_memory),
        "final_memory": int(final_memory),
    }


@torch.no_grad()
def _build_layer_trace(
    module,
    name: str,
    hidden: torch.Tensor,
    category_ids: torch.Tensor,
    category_names: Sequence[str],
    gate_moments: torch.Tensor,
    up_moments: torch.Tensor,
    rank_pairs: Sequence[tuple[int, int]],
    action_names: Sequence[str],
    action_cost: torch.Tensor,
    *,
    device: str,
    routing_batch_size: int,
    router_candidates: int,
    sketch_dim: int,
    randomized: bool,
    scale_floor: float,
    seed: int,
) -> tuple[LayerFunctionalTrace, torch.Tensor, dict[str, int], dict[str, torch.Tensor]]:
    logits, global_expert_ids = _pool_router_logits(
        module, hidden, device=device, batch_size=routing_batch_size
    )
    tokens, pool_size = map(int, logits.shape)
    shortlist = min(int(router_candidates), pool_size)
    if shortlist < int(module.top_k):
        raise ValueError("router shortlist must contain at least top_k experts")
    shortlist_ids = torch.topk(logits, k=shortlist, dim=-1).indices
    shortlist_mask = torch.zeros(tokens, pool_size, dtype=torch.bool)
    shortlist_mask.scatter_(1, shortlist_ids, True)
    # Reuse this one recorded ordering everywhere. A second topk call can
    # resolve exact BF16 ties differently even on identical logits.
    topk_ids = shortlist_ids[:, : int(module.top_k)]
    topk_values = logits.gather(1, topk_ids)
    if bool(module.norm_topk_prob):
        topk_weights = F.softmax(topk_values, dim=-1)
    else:
        topk_weights = F.softmax(logits, dim=-1).gather(1, topk_ids)
    topk_weight_by_expert = torch.zeros(tokens, pool_size, dtype=torch.float32)
    topk_weight_by_expert.scatter_(1, topk_ids, topk_weights.float())

    sample_gate = expert_projection_weight(module, int(global_expert_ids[0]), "gate")
    intermediate, hidden_size = map(int, sample_gate.shape)
    projection = gaussian_output_projection(hidden_size, sketch_dim, seed=seed)
    projection_device = projection.to(device=device, dtype=torch.float32)
    actions = len(action_names)
    outputs = torch.zeros(
        tokens, pool_size, actions, sketch_dim, dtype=torch.float16
    )
    max_gu_rank = max(pair[0] for pair in rank_pairs)
    ranks_by_gu = {
        gate_up: sorted({down for gu, down in rank_pairs if gu == gate_up})
        for gate_up in sorted({pair[0] for pair in rank_pairs})
    }
    # Use the original router geometry to calibrate replacement candidates as
    # well as currently active experts.  Squared shortlist probability matches
    # the squared output-distortion objective.
    shortlist_probability = F.softmax(
        logits.masked_fill(~shortlist_mask, -torch.inf), dim=-1
    )
    fallback_counts = {
        "gate_up_router_conditioned": 0,
        "gate_up_weight_only": 0,
        "down_router_conditioned": 0,
        "down_weight_only": 0,
    }
    usage_stats = {
        "topk_count": torch.zeros(pool_size, dtype=torch.long),
        "shortlist_count": shortlist_mask.sum(dim=0).long(),
        "standby_count": torch.zeros(pool_size, dtype=torch.long),
        "routing_mass": torch.zeros(pool_size, dtype=torch.float64),
        "reap_sum": torch.zeros(pool_size, dtype=torch.float64),
        "reap_mean": torch.zeros(pool_size, dtype=torch.float64),
    }

    for local_id in tqdm(range(pool_size), desc=f"Functional actions {name}", leave=False):
        token_ids = shortlist_mask[:, local_id].nonzero(as_tuple=True)[0]
        if not int(token_ids.numel()):
            continue
        expert_id = int(global_expert_ids[local_id].item())
        x = hidden.index_select(0, token_ids).to(device=device, dtype=torch.float32)
        gate_weight = expert_projection_weight(module, expert_id, "gate").float()
        up_weight = expert_projection_weight(module, expert_id, "up").float()
        down_weight = expert_projection_weight(module, expert_id, "down").float()
        original_z = F.silu(x @ gate_weight.t()) * (x @ up_weight.t())
        original_y = original_z @ down_weight.t()
        outputs[token_ids, local_id, -1] = (
            original_y @ projection_device
        ).to(device="cpu", dtype=torch.float16)
        active_weights = topk_weight_by_expert[token_ids, local_id].to(device)
        active = active_weights > 0
        active_count = int(active.sum().item())
        usage_stats["topk_count"][local_id] = active_count
        usage_stats["standby_count"][local_id] = int(token_ids.numel()) - active_count
        if active_count:
            active_weights = active_weights[active]
            activation_norm = torch.linalg.vector_norm(original_y[active], dim=1)
            routed_activation = active_weights * activation_norm
            usage_stats["routing_mass"][local_id] = active_weights.double().sum().cpu()
            usage_stats["reap_sum"][local_id] = routed_activation.double().sum().cpu()
            usage_stats["reap_mean"][local_id] = (
                routed_activation.double().mean().cpu()
            )

        stacked = torch.cat((gate_weight, up_weight), dim=0)
        candidate_weight = shortlist_probability[token_ids, local_id].to(device).square()
        shortlist_input_moment = (
            candidate_weight.unsqueeze(1) * x.square()
        ).sum(dim=0)
        joint_moment, gate_up_source = _calibration_moment_or_fallback(
            gate_moments[expert_id] + up_moments[expert_id],
            shortlist_input_moment,
            input_features=hidden_size,
        )
        if gate_up_source != "recorded":
            fallback_counts[f"gate_up_{gate_up_source}"] += 1
        u_gu, s_gu, v_gu, scale_gu = _decompose_calibrated(
            stacked,
            joint_moment,
            max_gu_rank,
            randomized=randomized,
            scale_floor=scale_floor,
        )
        for gate_up_rank, down_ranks in ranks_by_gu.items():
            joined = _apply_factors(
                x, u_gu, s_gu, v_gu, scale_gu, gate_up_rank
            )
            compressed_z = F.silu(joined[:, :intermediate]) * joined[:, intermediate:]
            conditional_down_moment = (
                candidate_weight.unsqueeze(1) * compressed_z.square()
            ).sum(dim=0)
            down_moment, down_source = _calibration_moment_or_fallback(
                conditional_down_moment,
                compressed_z.square().sum(dim=0),
                input_features=intermediate,
            )
            if down_source != "recorded":
                fallback_counts[f"down_{down_source}"] += 1
            u_down, s_down, v_down, scale_down = _decompose_calibrated(
                down_weight,
                down_moment,
                max(down_ranks),
                randomized=randomized,
                scale_floor=scale_floor,
            )
            for down_rank in down_ranks:
                action = 1 + list(rank_pairs).index((gate_up_rank, down_rank))
                compressed_y = _apply_factors(
                    compressed_z,
                    u_down,
                    s_down,
                    v_down,
                    scale_down,
                    down_rank,
                )
                outputs[token_ids, local_id, action] = (
                    compressed_y @ projection_device
                ).to(device="cpu", dtype=torch.float16)
            del joined, compressed_z, u_down, s_down, v_down, scale_down
        del (
            x,
            gate_weight,
            up_weight,
            down_weight,
            original_z,
            original_y,
            stacked,
            u_gu,
            s_gu,
            v_gu,
            scale_gu,
            joint_moment,
            shortlist_input_moment,
        )

    token_ids = torch.arange(tokens).unsqueeze(1)
    reference = (
        topk_weights.unsqueeze(-1)
        * outputs[token_ids, topk_ids, -1].float()
    ).sum(dim=1)
    costs = action_cost.unsqueeze(0).expand(pool_size, -1).clone()
    trace = LayerFunctionalTrace(
        name=name,
        # Router logits are small compared with expert-output traces. Keeping
        # them in FP32 prevents near-tied experts from changing order merely
        # because the trace was quantized. The reference stays FP32 as well so
        # the all-full plan is an exact zero-distortion baseline.
        router_logits=logits,
        reference_output=reference,
        candidate_outputs=outputs,
        category_ids=category_ids,
        category_names=category_names,
        action_costs=costs,
        action_names=action_names,
        top_k=int(module.top_k),
        norm_topk_prob=bool(module.norm_topk_prob),
        reference_topk_ids=topk_ids,
        routing_candidate_mask=shortlist_mask,
        routing_order=shortlist_ids,
    )
    if any(fallback_counts.values()):
        logging.warning("%s calibration fallbacks: %s", name, fallback_counts)
    return trace, global_expert_ids, fallback_counts, usage_stats


@torch.no_grad()
def _build_layer_gram_trace(
    module,
    name: str,
    hidden: torch.Tensor,
    category_ids: torch.Tensor,
    category_names: Sequence[str],
    gate_moments: torch.Tensor,
    up_moments: torch.Tensor,
    rank_pairs: Sequence[tuple[int, int]],
    action_names: Sequence[str],
    action_cost: torch.Tensor,
    *,
    device: str,
    routing_batch_size: int,
    router_candidates: int,
    randomized: bool,
    scale_floor: float,
    seed: int,
) -> tuple[LayerFunctionalTrace, torch.Tensor, dict[str, int], dict[str, torch.Tensor]]:
    """Build an exact full-output Gram trace from deployed BF16 factors."""

    logits, global_expert_ids = _pool_router_logits(
        module, hidden, device=device, batch_size=routing_batch_size
    )
    tokens, pool_size = map(int, logits.shape)
    shortlist = min(int(router_candidates), pool_size)
    if shortlist < int(module.top_k):
        raise ValueError("router shortlist must contain at least top_k experts")
    shortlist_values, shortlist_ids = torch.topk(logits, k=shortlist, dim=-1)
    shortlist_mask = torch.zeros(tokens, pool_size, dtype=torch.bool)
    shortlist_mask.scatter_(1, shortlist_ids, True)
    topk_ids = shortlist_ids[:, : int(module.top_k)]
    topk_values = shortlist_values[:, : int(module.top_k)]
    if bool(module.norm_topk_prob):
        topk_weights = F.softmax(topk_values, dim=-1)
    else:
        topk_weights = F.softmax(logits, dim=-1).gather(1, topk_ids)
    topk_weight_by_expert = torch.zeros(tokens, pool_size, dtype=torch.float32)
    topk_weight_by_expert.scatter_(1, topk_ids, topk_weights.float())
    shortlist_probability = F.softmax(shortlist_values, dim=-1)

    sample_gate = expert_projection_weight(module, int(global_expert_ids[0]), "gate")
    intermediate, hidden_size = map(int, sample_gate.shape)
    retained_actions = len(action_names) - 1
    # Sparse shortlist layout: [token, shortlist position, non-removed action, H].
    candidate_vectors = torch.zeros(
        tokens,
        shortlist,
        retained_actions,
        hidden_size,
        dtype=sample_gate.dtype,
        device="cpu",
    )
    shortlist_positions = torch.full((tokens, pool_size), -1, dtype=torch.long)
    shortlist_positions.scatter_(
        1,
        shortlist_ids,
        torch.arange(shortlist).unsqueeze(0).expand(tokens, -1),
    )
    fallback_counts = {
        "gate_up_router_conditioned": 0,
        "gate_up_weight_only": 0,
        "down_router_conditioned": 0,
        "down_weight_only": 0,
    }
    usage_stats = {
        "topk_count": torch.zeros(pool_size, dtype=torch.long),
        "shortlist_count": shortlist_mask.sum(dim=0).long(),
        "standby_count": torch.zeros(pool_size, dtype=torch.long),
        "routing_mass": torch.zeros(pool_size, dtype=torch.float64),
        "reap_sum": torch.zeros(pool_size, dtype=torch.float64),
        "reap_mean": torch.zeros(pool_size, dtype=torch.float64),
    }
    storage_dtype = sample_gate.dtype
    zero_down_moment = torch.zeros(intermediate, dtype=torch.float32)

    for local_id in tqdm(
        range(pool_size), desc=f"Exact Gram actions {name}", leave=False
    ):
        token_ids = shortlist_mask[:, local_id].nonzero(as_tuple=True)[0]
        if not int(token_ids.numel()):
            continue
        positions = shortlist_positions[token_ids, local_id]
        expert_id = int(global_expert_ids[local_id].item())
        x = hidden.index_select(0, token_ids).to(device=device, dtype=torch.float32)
        gate_weight = expert_projection_weight(module, expert_id, "gate")
        up_weight = expert_projection_weight(module, expert_id, "up")
        down_weight = expert_projection_weight(module, expert_id, "down")
        deployed_x = x.to(storage_dtype)
        original_gate = F.linear(deployed_x, gate_weight.to(storage_dtype))
        original_up = F.linear(deployed_x, up_weight.to(storage_dtype))
        original_z = F.silu(original_gate) * original_up
        original_y = F.linear(original_z, down_weight.to(storage_dtype))
        candidate_vectors[token_ids, positions, -1] = original_y.cpu()

        active_weights = topk_weight_by_expert[token_ids, local_id].to(device)
        active = active_weights > 0
        active_count = int(active.sum().item())
        usage_stats["topk_count"][local_id] = active_count
        usage_stats["standby_count"][local_id] = int(token_ids.numel()) - active_count
        if active_count:
            activation_norm = torch.linalg.vector_norm(
                original_y[active].float(), dim=1
            )
            routed_activation = active_weights[active] * activation_norm
            usage_stats["routing_mass"][local_id] = (
                active_weights[active].double().sum().cpu()
            )
            usage_stats["reap_sum"][local_id] = routed_activation.double().sum().cpu()
            usage_stats["reap_mean"][local_id] = routed_activation.double().mean().cpu()

        candidate_weight = shortlist_probability[token_ids, positions].to(
            device=device, dtype=torch.float32
        ).square()
        expert_gate_fallback_recorded = False
        for action_offset, (gate_up_rank, down_rank) in enumerate(rank_pairs):
            factors, fallbacks = build_deployed_expert_factors(
                gate_weight,
                up_weight,
                down_weight,
                gate_moments[expert_id],
                up_moments[expert_id],
                zero_down_moment,
                expert_id=expert_id,
                gate_up_rank=gate_up_rank,
                down_rank=down_rank,
                layer_seed=seed,
                candidate_inputs=x,
                candidate_weights=candidate_weight,
                scale_floor=scale_floor,
                randomized=randomized,
                storage_dtype=storage_dtype,
            )
            compressed_y, _ = apply_deployed_expert_factors(x, factors)
            candidate_vectors[token_ids, positions, action_offset] = compressed_y.cpu()
            if not expert_gate_fallback_recorded:
                fallback_counts["gate_up_router_conditioned"] += int(
                    fallbacks["gate_up_router_conditioned"]
                )
                fallback_counts["gate_up_weight_only"] += int(
                    fallbacks["gate_up_weight_only"]
                )
                expert_gate_fallback_recorded = True
            fallback_counts["down_weight_only"] += int(
                fallbacks["down_weight_only"]
            )
            del factors, compressed_y
        del x, original_gate, original_up, original_z, original_y

    reference = (
        topk_weights.float().unsqueeze(2)
        * candidate_vectors[:, : int(module.top_k), -1].float()
    ).sum(dim=1)
    vectors = candidate_vectors.reshape(
        tokens, shortlist * retained_actions, hidden_size
    )
    gram = torch.empty(
        tokens, shortlist * retained_actions, shortlist * retained_actions,
        dtype=torch.float32,
    )
    cross = torch.empty(tokens, shortlist * retained_actions, dtype=torch.float32)
    energy = reference.float().square().sum(1)
    gram_batch = max(1, min(16, tokens))
    for start in range(0, tokens, gram_batch):
        stop = min(tokens, start + gram_batch)
        value = vectors[start:stop].to(device=device, dtype=torch.float32)
        ref = reference[start:stop].to(device=device, dtype=torch.float32)
        gram[start:stop] = torch.bmm(value, value.transpose(1, 2)).cpu()
        cross[start:stop] = torch.bmm(value, ref.unsqueeze(2)).squeeze(2).cpu()
        del value, ref
    trace = LayerFunctionalTrace(
        name=name,
        router_logits=logits,
        candidate_gram=gram,
        reference_cross=cross,
        reference_energy=energy,
        category_ids=category_ids,
        category_names=category_names,
        action_costs=action_cost.unsqueeze(0).expand(pool_size, -1).clone(),
        action_names=action_names,
        top_k=int(module.top_k),
        norm_topk_prob=bool(module.norm_topk_prob),
        reference_topk_ids=topk_ids,
        routing_candidate_mask=shortlist_mask,
        routing_order=shortlist_ids,
    )
    if any(fallback_counts.values()):
        logging.warning("%s calibration fallbacks: %s", name, fallback_counts)
    return trace, global_expert_ids, fallback_counts, usage_stats


def _json_plan(
    plan,
    traces,
    global_ids,
    usage_stats,
    *,
    action_names,
    full_memory,
    budget,
    token_reduction,
    token_energy_floor_ratio,
    conditioned_reports: Mapping[str, DistortionReport] | None = None,
    importance_compressibility: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    posterior_distortion: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    factorized_risk: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    propagated_risk: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    distortion_risk: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    mean_loss: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
):
    layers = {}
    for trace in traces:
        actions = plan.actions[trace.name]
        ids = global_ids[trace.name]
        by_action = {}
        for action_id, action_name in enumerate(action_names):
            chosen = ids[actions == action_id].tolist()
            by_action[action_name] = [int(value) for value in chosen]
        report = (
            conditioned_reports[trace.name]
            if conditioned_reports is not None
            else trace.evaluate(
                actions,
                token_reduction=token_reduction,
                token_energy_floor_ratio=token_energy_floor_ratio,
            )
        )
        full_actions = torch.full(
            (trace.experts,), trace.actions - 1, dtype=torch.long
        )
        full_layer_memory = trace.memory(full_actions)
        allocation_diagnostics = (
            None
            if importance_compressibility is None
            else importance_compressibility[trace.name]
        )
        posterior_diagnostics = (
            None
            if posterior_distortion is None
            else posterior_distortion[trace.name]
        )
        factorized_diagnostics = (
            None if factorized_risk is None else factorized_risk[trace.name]
        )
        propagated_diagnostics = (
            None if propagated_risk is None else propagated_risk[trace.name]
        )
        distortion_diagnostics = (
            None if distortion_risk is None else distortion_risk[trace.name]
        )
        mean_loss_diagnostics = (
            None if mean_loss is None else mean_loss[trace.name]
        )
        layers[trace.name] = {
            "expert_actions": {
                str(int(expert_id)): action_names[int(action_id)]
                for expert_id, action_id in zip(ids.tolist(), actions.tolist())
            },
            "by_action": by_action,
            "expert_usage": {
                str(int(expert_id)): {
                    metric: (
                        int(values[local_id].item())
                        if values.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
                        else float(values[local_id].item())
                    )
                    for metric, values in usage_stats[trace.name].items()
                }
                for local_id, expert_id in enumerate(ids)
            },
            "single_action_relative_distortion": (
                {
                    str(int(expert_id)): {
                        action_name: {
                            category: float(
                                trace.raw_action_numerator[
                                    local_id, action_id, category_id
                                ].item()
                                / trace.reference_energy[category_id]
                                .clamp_min(1e-30)
                                .item()
                            )
                            for category_id, category in enumerate(
                                trace.category_names
                            )
                        }
                        for action_id, action_name in enumerate(action_names)
                    }
                    for local_id, expert_id in enumerate(ids)
                }
                if isinstance(trace, LayerIndependentActionTrace)
                else None
            ),
            "allocation_action_relative_score": (
                {
                    str(int(expert_id)): {
                        action_name: {
                            category: float(
                                trace.action_numerator[
                                    local_id, action_id, category_id
                                ].item()
                                / trace.reference_energy[category_id]
                                .clamp_min(1e-30)
                                .item()
                            )
                            for category_id, category in enumerate(
                                trace.category_names
                            )
                        }
                        for action_id, action_name in enumerate(action_names)
                    }
                    for local_id, expert_id in enumerate(ids)
                }
                if isinstance(trace, LayerIndependentActionTrace)
                else None
            ),
            "importance_compressibility": (
                None
                if allocation_diagnostics is None
                else {
                    "importance_prior": {
                        category: float(
                            allocation_diagnostics["importance_prior"][category_id].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "experts": {
                        str(int(expert_id)): {
                            "active_token_count": {
                                category: int(
                                    allocation_diagnostics["active_token_count"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "observed_conditional_importance": {
                                category: float(
                                    allocation_diagnostics[
                                        "observed_conditional_importance"
                                    ][local_id, category_id].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "shrunken_conditional_importance": {
                                category: float(
                                    allocation_diagnostics[
                                        "shrunken_conditional_importance"
                                    ][local_id, category_id].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "importance_confidence": {
                                category: float(
                                    allocation_diagnostics["importance_confidence"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "actions": {
                                action_name: {
                                    "structural_relative_error": float(
                                        allocation_diagnostics[
                                            "structural_relative_action_error"
                                        ][local_id, action_id].item()
                                    ),
                                    "activation_relative_error": {
                                        category: float(
                                            allocation_diagnostics[
                                                "activation_relative_action_error"
                                            ][local_id, action_id, category_id].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "shrunken_relative_error": {
                                        category: float(
                                            allocation_diagnostics[
                                                "shrunken_relative_action_error"
                                            ][local_id, action_id, category_id].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "risk": {
                                        category: float(
                                            allocation_diagnostics["action_risk"][
                                                local_id, action_id, category_id
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                }
                                for action_id, action_name in enumerate(action_names)
                            },
                        }
                        for local_id, expert_id in enumerate(ids)
                    },
                }
            ),
            "posterior_distortion": (
                None
                if posterior_diagnostics is None
                else {
                    "category_prompt_count": {
                        category: int(
                            posterior_diagnostics["category_prompt_count"][
                                category_id
                            ].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "structural_to_functional_scale": {
                        action_name: {
                            category: float(
                                posterior_diagnostics[
                                    "structural_to_functional_scale"
                                ][action_id, category_id].item()
                            )
                            for category_id, category in enumerate(
                                trace.category_names
                            )
                        }
                        for action_id, action_name in enumerate(action_names)
                    },
                    "between_expert_log_variance": {
                        action_name: {
                            category: float(
                                posterior_diagnostics[
                                    "between_expert_log_variance"
                                ][action_id, category_id].item()
                            )
                            for category_id, category in enumerate(
                                trace.category_names
                            )
                        }
                        for action_id, action_name in enumerate(action_names)
                    },
                    "experts": {
                        str(int(expert_id)): {
                            "active_prompt_count": {
                                category: int(
                                    posterior_diagnostics["active_prompt_count"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "actions": {
                                action_name: {
                                    "observed_functional_risk": {
                                        category: float(
                                            posterior_diagnostics[
                                                "observed_functional_risk"
                                            ][
                                                local_id,
                                                action_id,
                                                category_id,
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "structural_prior_risk": {
                                        category: float(
                                            posterior_diagnostics[
                                                "calibrated_structural_prior"
                                            ][
                                                local_id,
                                                action_id,
                                                category_id,
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "posterior_confidence": {
                                        category: float(
                                            posterior_diagnostics[
                                                "posterior_confidence"
                                            ][
                                                local_id,
                                                action_id,
                                                category_id,
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "posterior_risk": {
                                        category: float(
                                            posterior_diagnostics[
                                                "posterior_action_risk"
                                            ][
                                                local_id,
                                                action_id,
                                                category_id,
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                }
                                for action_id, action_name in enumerate(action_names)
                            },
                        }
                        for local_id, expert_id in enumerate(ids)
                    },
                }
            ),
            "factorized_risk": (
                None
                if factorized_diagnostics is None
                else {
                    "exposure_prior": {
                        category: float(
                            factorized_diagnostics["exposure_prior"][
                                category_id
                            ].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "frequency_alpha": float(
                        factorized_diagnostics["frequency_alpha"].item()
                    ),
                    "experts": {
                        str(int(expert_id)): {
                            "active_prompt_count": {
                                category: int(
                                    factorized_diagnostics["active_prompt_count"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "observed_exposure": {
                                category: float(
                                    factorized_diagnostics["observed_exposure"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "activation_frequency": {
                                category: float(
                                    factorized_diagnostics[
                                        "activation_frequency"
                                    ][local_id, category_id].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "observed_importance": {
                                category: float(
                                    factorized_diagnostics["observed_importance"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "posterior_exposure": {
                                category: float(
                                    factorized_diagnostics["posterior_exposure"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "conservative_importance": {
                                category: float(
                                    factorized_diagnostics[
                                        "conservative_exposure"
                                    ][local_id, category_id].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "rerouting_removal_upper_risk": {
                                category: float(
                                    factorized_diagnostics[
                                        "rerouting_removal_upper_risk"
                                    ][local_id, category_id].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "exposure_confidence": {
                                category: float(
                                    factorized_diagnostics["exposure_confidence"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "actions": {
                                action_name: {
                                    "observed_conditional_compressibility": {
                                        category: float(
                                            factorized_diagnostics[
                                                "observed_conditional_compressibility"
                                            ][local_id, action_id, category_id].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "posterior_conditional_compressibility": {
                                        category: float(
                                            factorized_diagnostics[
                                                "posterior_conditional_compressibility"
                                            ][local_id, action_id, category_id].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "conservative_conditional_degradation": {
                                        category: float(
                                            factorized_diagnostics[
                                                "conservative_conditional_compressibility"
                                            ][
                                                local_id,
                                                action_id,
                                                category_id,
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                    "risk": {
                                        category: float(
                                            factorized_diagnostics["action_risk"][
                                                local_id, action_id, category_id
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    },
                                }
                                for action_id, action_name in enumerate(action_names)
                            },
                        }
                        for local_id, expert_id in enumerate(ids)
                    },
                }
            ),
            "propagated_risk": (
                None
                if propagated_diagnostics is None
                else {
                    "frequency_alpha": float(
                        propagated_diagnostics["frequency_alpha"].item()
                    ),
                    "svd_propagation_factor": {
                        category: float(
                            propagated_diagnostics["svd_propagation_factor"][
                                category_id
                            ].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "removal_propagation_factor": {
                        category: float(
                            propagated_diagnostics["removal_propagation_factor"][
                                category_id
                            ].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "experts": {
                        str(int(expert_id)): {
                            "active_prompt_count": {
                                category: int(
                                    propagated_diagnostics["active_prompt_count"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "activation_frequency": {
                                category: float(
                                    propagated_diagnostics["activation_frequency"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "observed_importance": {
                                category: float(
                                    propagated_diagnostics["observed_importance"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "conservative_importance": {
                                category: float(
                                    propagated_diagnostics[
                                        "conservative_importance"
                                    ][local_id, category_id].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "actions": {
                                action_name: {
                                    category: float(
                                        propagated_diagnostics["action_risk"][
                                            local_id, action_id, category_id
                                        ].item()
                                    )
                                    for category_id, category in enumerate(
                                        trace.category_names
                                    )
                                }
                                for action_id, action_name in enumerate(action_names)
                            },
                        }
                        for local_id, expert_id in enumerate(ids)
                    },
                }
            ),
            "distortion_risk": (
                None
                if distortion_diagnostics is None
                else {
                    "tail_fraction": float(
                        distortion_diagnostics["tail_fraction"].item()
                    ),
                    "tail_weight": float(
                        distortion_diagnostics["tail_weight"].item()
                    ),
                    "svd_propagation_factor": {
                        category: float(
                            distortion_diagnostics["svd_propagation_factor"][
                                category_id
                            ].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "removal_propagation_factor": {
                        category: float(
                            distortion_diagnostics["removal_propagation_factor"][
                                category_id
                            ].item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    },
                    "experts": {
                        str(int(expert_id)): {
                            **{
                                field: {
                                    category: float(
                                        distortion_diagnostics[field][
                                            local_id, category_id
                                        ].item()
                                    )
                                    for category_id, category in enumerate(
                                        trace.category_names
                                    )
                                }
                                for field in (
                                    "activation_frequency",
                                    "conditional_expert_energy",
                                )
                            },
                            "active_prompt_count": {
                                category: int(
                                    distortion_diagnostics["active_prompt_count"][
                                        local_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            },
                            "actions": {
                                action_name: {
                                    field: {
                                        category: float(
                                            distortion_diagnostics[field][
                                                local_id, action_id, category_id
                                            ].item()
                                        )
                                        for category_id, category in enumerate(
                                            trace.category_names
                                        )
                                    }
                                    for field in (
                                        "observed_conditional_damage",
                                        "posterior_conditional_damage",
                                        "posterior_log_dispersion",
                                        "tail_factor",
                                        "spectral_weight",
                                        "action_risk",
                                    )
                                }
                                for action_id, action_name in enumerate(action_names)
                            },
                        }
                        for local_id, expert_id in enumerate(ids)
                    },
                }
            ),
            "mean_loss": (
                None
                if mean_loss_diagnostics is None
                else {
                    "damage_unit": trace.damage_unit,
                    **{
                        field: {
                            category: float(
                                mean_loss_diagnostics[field][category_id].item()
                            )
                            for category_id, category in enumerate(
                                trace.category_names
                            )
                        }
                        for field in (
                            "reference_mean_output",
                            "svd_propagation_factor",
                            "removal_propagation_factor",
                        )
                    },
                    "experts": {
                        str(int(expert_id)): {
                            action_name: {
                                category: float(
                                    mean_loss_diagnostics["mean_loss"][
                                        local_id, action_id, category_id
                                    ].item()
                                )
                                for category_id, category in enumerate(
                                    trace.category_names
                                )
                            }
                            for action_id, action_name in enumerate(action_names)
                        }
                        for local_id, expert_id in enumerate(ids)
                    },
                }
            ),
            "diagnostics": {
                "category_relative_distortion": {
                    category: float(report.relative[index].item())
                    for index, category in enumerate(trace.category_names)
                },
                "worst_category_relative_distortion": float(
                    report.relative.max().item()
                ),
                "route_change_rate": (
                    None
                    if isinstance(trace, LayerIndependentActionTrace)
                    and conditioned_reports is None
                    else float(report.route_change_rate)
                ),
                "distortion_source": (
                    "rerouting_conditioned_exact_current_plan"
                    if conditioned_reports is not None
                    else "initial_planner_trace"
                ),
                "retained_parameter_ratio": float(
                    trace.memory(actions) / full_layer_memory
                ),
                "retained_experts": int((actions != 0).sum().item()),
                "raw_additive_category_relative_distortion": (
                    {
                        category: float(
                            trace.raw_action_numerator[
                                torch.arange(trace.experts), actions, category_id
                            ].sum().item()
                            / trace.reference_energy[category_id]
                            .clamp_min(1e-30)
                            .item()
                        )
                        for category_id, category in enumerate(trace.category_names)
                    }
                    if isinstance(trace, LayerIndependentActionTrace)
                    else None
                ),
            },
        }
    return {
        "meta": {
            "objective": "category-robust routed aggregate output distortion",
            "full_expert_parameters": int(full_memory),
            "budget_parameters": int(budget),
            "retained_parameter_ratio": float(plan.memory / full_memory),
            "planned_parameters": int(plan.memory),
            "action_names": list(action_names),
        },
        "objective": float(plan.objective),
        "category_relative_distortion": plan.category_relative_distortion,
        "layers": layers,
        "history": plan.history,
    }


def _layer_retained_fraction(
    plan: FunctionalCompressionPlan,
    traces: Sequence[LayerIndependentActionTrace],
) -> dict[str, float]:
    """Fraction of each layer's experts the plan keeps in any retained action."""

    fractions = {}
    for trace in traces:
        actions = plan.actions[trace.name]
        fractions[trace.name] = max(
            float((actions != 0).sum().item()) / max(trace.experts, 1), 1e-6
        )
    return fractions


@torch.no_grad()
def _retention_aware_replanning(
    plan: FunctionalCompressionPlan,
    traces: Sequence[LayerIndependentActionTrace],
    *,
    builder_kwargs: Mapping,
    allocator_kwargs: Mapping,
    rounds: int,
    damping: float,
    tolerance: float = 0.01,
) -> tuple[FunctionalCompressionPlan, dict, list[dict]]:
    """Iterate removal risk and allocation to a self-consistent retention level.

    Round zero scores removal against the nearest shortlist fallback, which is
    only correct for a plan that removes almost nothing. Each round below feeds
    the previous plan's retained fraction back into the removal-depth mixture and
    reallocates, so the removal risk a layer is charged matches the amount of
    removal that layer actually got.

    Convergence requires *both* that the allocation stopped moving and that the
    assumed retention matches the achieved retention. Testing only the former
    would declare success while the fixed point is still far away, since a small
    risk change often flips no decision at all.

    Allocations are discrete, so an exact fixed point need not exist: the
    retention can legitimately alternate between two adjacent plans. Damping
    contracts that cycle -- the feedback is negative, since deeper fallbacks raise
    removal risk, which retains more experts, which shallows the fallback -- but
    it cannot shrink it below one expert's worth of retention, which is why the
    tolerance is floored at two experts. When the loop still fails to settle it
    returns the most self-consistent round it saw rather than whichever round it
    happened to stop on, and says so.

    Measured caveat: on Qwen3-30B-A3B this loop converges immediately and barely
    moves the plan, because removal damage turns out to be flat in fallback
    depth. See :func:`fallback_depth_weights` for why.
    """

    assumed = {trace.name: 1.0 for trace in traces}
    diagnostics = None
    history = []
    previous_actions = {
        name: value.clone() for name, value in plan.actions.items()
    }
    # One expert of retention is the finest step any layer can take, so a
    # tolerance below that can never be met by a discrete allocator.
    smallest_layer = min(trace.experts for trace in traces)
    tolerance = max(float(tolerance), 2.0 / max(smallest_layer, 1))
    best = None
    for round_index in range(int(rounds)):
        observed = _layer_retained_fraction(plan, traces)
        step = float(damping)
        assumed = {
            name: min(
                1.0,
                max(
                    (1.0 - step) * assumed[name] + step * observed[name],
                    1e-6,
                ),
            )
            for name in observed
        }
        allocation_traces, diagnostics = build_distortion_risk_traces(
            traces, retained_fraction=assumed, **dict(builder_kwargs)
        )
        plan = cached_greedy_global_downgrade(
            allocation_traces, **dict(allocator_kwargs)
        )
        changed = sum(
            int((plan.actions[name] != previous_actions[name]).sum().item())
            for name in previous_actions
        )
        achieved = _layer_retained_fraction(plan, traces)
        gap = max(abs(assumed[name] - achieved[name]) for name in achieved)
        removed = sum(
            int((plan.actions[name] == 0).sum().item()) for name in plan.actions
        )
        experts = sum(trace.experts for trace in traces)
        record = {
            "round": round_index + 1,
            "mean_assumed_retention": sum(assumed.values()) / max(len(assumed), 1),
            "mean_achieved_retention": sum(achieved.values()) / max(len(achieved), 1),
            "max_retention_gap": gap,
            "removed_experts": removed,
            "total_experts": experts,
            "actions_changed": changed,
            "objective": float(plan.objective),
        }
        history.append(record)
        logging.info(
            "Retention round %d: assumed %.4f, achieved %.4f (gap %.4f) -> "
            "removed %d/%d experts, %d action changes, objective %.6g",
            record["round"],
            record["mean_assumed_retention"],
            record["mean_achieved_retention"],
            gap,
            removed,
            experts,
            changed,
            record["objective"],
        )
        previous_actions = {
            name: value.clone() for name, value in plan.actions.items()
        }
        if best is None or gap < best[0]:
            best = (
                gap,
                FunctionalCompressionPlan(
                    actions={
                        name: value.clone() for name, value in plan.actions.items()
                    },
                    memory=plan.memory,
                    objective=plan.objective,
                    category_relative_distortion=dict(
                        plan.category_relative_distortion
                    ),
                    history=list(plan.history),
                ),
                diagnostics,
                record["round"],
            )
        if changed == 0 and gap <= tolerance:
            logging.info(
                "Retention iteration converged after %d rounds", round_index + 1
            )
            return plan, diagnostics, history
    if best is not None:
        logging.warning(
            "Retention iteration did not settle in %d rounds; returning round %d, "
            "the most self-consistent one seen (retention gap %.4f)",
            int(rounds),
            best[3],
            best[0],
        )
        for record in history:
            record["converged"] = False
        return best[1], best[2], history
    return plan, diagnostics, history


def _safe_layer_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


@torch.no_grad()
def _save_selected_factor_artifact(
    *,
    output_dir: str,
    plan_path: str,
    plan,
    traces: Sequence[LayerFunctionalTrace],
    global_ids: Mapping[str, torch.Tensor],
    modules: Mapping[str, object],
    hidden_metadata: Mapping,
    row_ids: torch.Tensor,
    gate_moments: Mapping[str, torch.Tensor],
    up_moments: Mapping[str, torch.Tensor],
    down_moments: Mapping[str, torch.Tensor],
    rank_pairs: Sequence[tuple[int, int]],
    router_candidates: int,
    routing_batch_size: int,
    scale_floor: float,
    randomized: bool,
    seed: int,
    gram_metric: bool,
    gram_shrink_mode: str,
    down_compensation: bool,
    factor_calibration: str = "shortlist",
    factor_output_metric: str = "none",
) -> None:
    """Save only factors selected by the final plan, one layer per file."""

    os.makedirs(output_dir, exist_ok=True)
    layer_files = {}
    total_parameters = 0
    for layer_index, trace in enumerate(traces):
        name = trace.name
        actions = plan.actions[name]
        ids = global_ids[name]
        tier_specs = []
        for action_index, (gate_up_rank, down_rank) in enumerate(rank_pairs, start=1):
            chosen = ids[actions == action_index].tolist()
            if chosen:
                tier_specs.append((
                    [int(value) for value in chosen],
                    int(gate_up_rank),
                    int(down_rank),
                ))
        all_hidden = _load_hidden_cache(hidden_metadata, name)
        calibration_hidden = _select_hidden_rows_without_full_copy(
            all_hidden, row_ids
        )
        del all_hidden
        state = Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
            modules[name],
            tier_specs=tier_specs,
            gate_moments=gate_moments[name],
            up_moments=up_moments[name],
            down_moments=down_moments[name],
            calibration_hidden=calibration_hidden,
            router_candidates=router_candidates,
            routing_batch_size=routing_batch_size,
            scale_floor=scale_floor,
            randomized=randomized,
            layer_seed=int(seed) + layer_index * 1_000_003,
            # These change the factors. Omitting them silently saved
            # diagonal-metric factors for a Gram plan, which is the plan/
            # deployment divergence the trace scope exists to prevent.
            gram_metric=gram_metric,
            gram_shrink_mode=gram_shrink_mode,
            down_compensation=down_compensation,
            factor_calibration=factor_calibration,
            factor_output_metric=factor_output_metric,
            # The same retained set the restructurer's from_stacked routes over,
            # so saved factors equal the ones built on the fly from this plan.
            retained_ids=ids[actions != 0].tolist(),
        )
        cpu_tiers = []
        for tier in state["tiers"]:
            cpu_tier = {}
            for key, value in tier.items():
                cpu_tier[key] = value.detach().cpu() if torch.is_tensor(value) else value
                if key.endswith("_factor"):
                    total_parameters += int(value.numel())
            cpu_tiers.append(cpu_tier)
        state["tiers"] = cpu_tiers
        filename = f"{_safe_layer_name(name)}.pt"
        torch.save(state, os.path.join(output_dir, filename))
        layer_files[name] = filename
        logging.info("Saved selected deployed factors -> %s", filename)
        del state, calibration_hidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open(plan_path, "rb") as handle:
        plan_sha256 = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "format": "routed_functional_svd_selected_factors",
        "format_version": 1,
        "plan_path": os.path.abspath(plan_path),
        "plan_sha256": plan_sha256,
        "storage_dtype": str(modules[traces[0].name].gate_proj_stacked.dtype),
        "factor_parameters": int(total_parameters),
        "layers": layer_files,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    logging.info(
        "Saved factor artifact metadata -> %s (%d parameters)",
        os.path.join(output_dir, "metadata.json"),
        total_parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group_inputs = parser.add_argument_group("inputs", "required artifacts and what the plan is priced against")
    group_inputs.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    group_inputs.add_argument("--moments", required=True)
    group_inputs.add_argument("--basis-category", default="balanced")
    group_inputs.add_argument(
        "--routing-kept-path",
        default=None,
        help=(
            "optional expert-selection JSON; omit it to route over and jointly "
            "plan all original experts"
        ),
    )
    group_inputs.add_argument("--sequence-cache", required=True)
    group_inputs.add_argument("--hidden-cache-dir", required=True)
    group_inputs.add_argument(
        "--propagation-path",
        default=None,
        help=(
            "moe_propagation_damage JSON produced by "
            "tools/estimate_moe_propagation.py; required by propagated_risk and "
            "optional for distortion_risk"
        ),
    )
    group_budget = parser.add_argument_group("budget", "what the plan is allowed to spend")
    group_budget.add_argument("--rank-pairs", nargs="+", type=_parse_rank_pair, default=[(256, 128), (384, 256)])
    group_budget.add_argument("--memory-ratio", type=float, default=0.50)
    group_budget.add_argument(
        "--min-retained-experts",
        type=int,
        default=0,
        help="hard minimum number of non-removed experts in every layer",
    )
    group_budget.add_argument(
        "--min-layer-memory-ratio",
        type=float,
        default=0.0,
        help="hard per-layer floor relative to that layer's full expert parameters",
    )
    group_budget.add_argument(
        "--layer-min-memory-json",
        default=None,
        help="greedy only: JSON {layer name or index: floor} overriding "
        "--min-layer-memory-ratio for those layers (1.0 keeps a layer "
        "uncompressed), e.g. to protect the layers a propagation probe finds "
        "most sensitive",
    )
    group_budget.add_argument(
        "--min-layer-routed-mass",
        type=float,
        default=0.0,
        help="coverage constraint: never remove an expert if the layer would keep "
        "less than this fraction of its routed mass on retained experts (tiers keep "
        "their mass). Generation collapses near 54-65%% retained mass on "
        "Qwen3-30B-A3B. 0 disables it. Greedy allocator only.",
    )
    group_budget.add_argument(
        "--max-layer-relative-distortion",
        type=float,
        default=None,
        help="optional hard cap on every layer/category relative distortion",
    )
    group_ranking = parser.add_argument_group("ranking", "how experts are scored and ordered against each other")
    group_ranking.add_argument(
        "--routing-penalty-calibrate",
        action="store_true",
        help=(
            "fit each expert's discount instead of scaling it: choose the "
            "margin threshold that minimises measured damage, exactly, from the "
            "per-token cost of keeping the expert versus losing the slot. "
            "--routing-penalty-scale then shrinks the fitted value toward zero "
            "(1.0 trusts the fit). Safe to pass with scale 0: the plan is priced "
            "exactly as an undiscounted one but records every fitted threshold "
            "and the damage fraction it would remove"
        ),
    )
    group_ranking.add_argument(
        "--routing-penalty-scale",
        type=float,
        default=0.0,
        help=(
            "RETIRED -- any nonzero value is rejected. It discounted a "
            "compressed expert's router logit by scale times its own relative "
            "damage amplitude and repriced displaced tokens as removal. The "
            "discount is equivalent to scaling the expert's output by gamma < 1, "
            "but the SVD fit's own optimality condition already places the "
            "least-squares optimum at gamma = 1, so the term can only add error; "
            "the calibrated fit confirmed it, returning a zero threshold on all "
            "18,432 cells. Only 0 is accepted"
        ),
    )
    group_ranking.add_argument(
        "--confidence-kappa",
        type=float,
        default=64.0,
        help=(
            "pseudo-count controlling shrinkage toward structural and layer "
            "importance priors in importance-compressibility planning"
        ),
    )
    group_ranking.add_argument(
        "--importance-power",
        type=float,
        default=2.0,
        help="power applied to normalized conditional REAP importance",
    )
    group_ranking.add_argument(
        "--importance-prior",
        choices=["median", "mean"],
        default="median",
        help="layer/category prior used for sparsely observed expert importance",
    )
    group_ranking.add_argument("--objective", choices=["mean", "worst", "cvar"], default="worst")
    group_ranking.add_argument("--cvar-fraction", type=float, default=0.5)
    group_ranking.add_argument(
        "--token-reduction",
        choices=["energy", "relative"],
        default="energy",
        help=(
            "aggregate raw residual/reference energy ('energy') or average "
            "per-token relative distortion within each category ('relative'); "
            "use 'energy' with --layer-reduction mean for MoE-output-relative "
            "layer normalization"
        ),
    )
    group_ranking.add_argument(
        "--token-energy-floor-ratio",
        type=float,
        default=0.05,
        help=(
            "for token-relative distortion, floor each reference energy at "
            "this fraction of its category median"
        ),
    )
    group_ranking.add_argument(
        "--layer-reduction",
        choices=["energy", "mean"],
        default="mean",
        help=(
            "weight layers by reference energy ('energy') or give every layer "
            "equal weight after within-layer normalization ('mean')"
        ),
    )
    group_ranking.add_argument(
        "--occupancy-alpha",
        type=float,
        default=1.0,
        help=(
            "streaming exponent for activation frequency: 1 keeps expected "
            "global frequency, 0 uses conditional contribution only, and 0.5 "
            "tempers frequency to protect rare high-contribution experts"
        ),
    )
    group_ranking.add_argument(
        "--removal-tiebreak",
        choices=["none", "reap_mean", "reap_sum"],
        default="reap_mean",
        help=(
            "REAP statistic used only to break near-ties between direct removal "
            "candidates; 'none' is the functional-only ablation"
        ),
    )
    group_ranking.add_argument(
        "--removal-tiebreak-rtol",
        type=float,
        default=0.01,
        help="relative functional-score tolerance for the REAP removal near-tie",
    )
    group_ranking.add_argument(
        "--tier-price-scale-json",
        default=None,
        help=(
            "mean_loss only: JSON {\"layers\": {layer: [scale per SVD tier, cheapest "
            "first]}} multiplying tier prices on top of --tier-price, e.g. the "
            "measured damage ratio of refined to traced factors "
            "(tools/refine_tier_factors.py). Layers not listed keep scale 1."
        ),
    )
    group_ranking.add_argument(
        "--tier-price",
        type=float,
        default=1.0,
        help=(
            "multiply the risk of the SVD-tier actions relative to removal. 1.0 is "
            "an exact no-op. The propagation gains that otherwise set this exchange "
            "rate come from a random-direction probe and price the two action "
            "classes by error magnitude alone; measured on Qwen3-30B-A3B, shifting "
            "routing mass from removal into tiers costs 0.038 human-eval points per "
            "1%% of mass. Raising this drives tier usage toward zero, which is the "
            "right behaviour above the ~0.35-budget crossover where removal-only "
            "(REAP) wins; below it the ladder is worth 43-49 points and the price "
            "should return to 1.0."
        ),
    )
    group_ranking.add_argument(
        "--action-interaction-rho",
        type=float,
        default=0.0,
        help=(
            "mean correlation between the error vectors of co-routed compressed "
            "experts. The objective otherwise sums per-expert damage, which is "
            "exact only if those errors are orthogonal; they are not, so the true "
            "layer error carries cross terms the sum cannot see (~1.7-2x on this "
            "model). A token whose top-k holds m perturbed experts is scored "
            "(1 + rho*(m-1)) times its additive damage. This is what prices buying "
            "MANY cheap tiers: a more faithful factorisation lowers each expert's "
            "error, and a purely additive objective answers by buying more of them. "
            "0 (default) reproduces the additive objective exactly."
        ),
    )
    group_ranking.add_argument(
        "--tier-assignment",
        choices=("risk", "reap_mean", "reap_sum"),
        default="risk",
        help=(
            "after allocating each layer's action counts, optionally assign "
            "cheaper tiers to experts with lower REAP saliency while preserving "
            "memory and every per-layer action histogram"
        ),
    )
    group_factors = parser.add_argument_group("factors", "how a retained expert is factorized")
    group_factors.add_argument("--router-candidates", type=int, default=16)
    group_factors.add_argument("--sketch-dim", type=int, default=16)
    group_factors.add_argument(
        "--gram-shrink-target",
        choices=("diagonal", "layer", "layer_only"),
        default="diagonal",
        help=(
            "what a starved per-expert Gram is shrunk toward under --factor-metric "
            "gram. 'diagonal' keeps only per-coordinate variance and discards the "
            "correlation structure whitening exists for; 'layer' shrinks toward the "
            "LAYER Gram pooled over every calibration token, which is the per-layer "
            "covariance SVD-LLM-style whitening actually assumes. Ignored for the "
            "diagonal metric."
        ),
    )
    group_factors.add_argument(
        "--down-compensation",
        action="store_true",
        help=(
            "fit the down projection to reproduce the expert's ORIGINAL output "
            "from the compressed intermediate, instead of approximating W_down "
            "as a map in the compressed intermediate's geometry. Measured on "
            "Qwen3-30B-A3B coding256, ~72%% of the down-stage error is inherited "
            "from the gate/up truncation rather than caused by down's own rank "
            "limit; targeting the original output lets down absorb 2.2-2.6x of "
            "it at identical rank, bytes and kernels. Requires calibration "
            "tokens (--hidden-cache-dir); a no-op for experts that have none."
        ),
    )
    group_factors.add_argument(
        "--factor-metric",
        choices=("diagonal", "gram"),
        default="diagonal",
        help=(
            "input metric the SVD is calibrated in. 'diagonal' is the shipped "
            "per-coordinate second moment. 'gram' uses the full Z^T diag(w) Z, "
            "which is the correct metric when input coordinates are correlated; "
            "it is built from the routed calibration tokens, so a diagonal "
            "moments artifact is simply unused rather than an error. Changing "
            "this changes the deployed weights and REQUIRES its own --trace-dir."
        ),
    )
    group_factors.add_argument(
        "--factor-output-metric",
        choices=FACTOR_OUTPUT_METRICS,
        default="none",
        help=(
            "how the gate/up factorization weights its 768 output channels. "
            "'none' is the plain Frobenius norm (every channel equal); "
            "'jacobian' weights channel j by how much an error there reaches "
            "the expert output through the SiLU product and W_down. Needs "
            "calibration tokens, and gets its own factor-calibration scope"
        ),
    )
    group_factors.add_argument(
        "--factor-calibration",
        choices=FACTOR_CALIBRATIONS,
        default=None,
        help=(
            "which tokens calibrate the SVD and how each is weighted; default "
            "'topk_g2' for --trace-mode streaming, 'shortlist' otherwise. "
            "'shortlist' (legacy) uses every token whose router top-L "
            "(--router-candidates) contains the expert, weighted by the squared "
            "top-L softmax, with gate/up taken from the recorded sensitivity "
            "moments. 'topk_g2' is ASVD-style: only the tokens routed to the "
            "expert (router top-k over the experts the plan retains, so a "
            "removed expert's tokens go to the next retained one), weighted by "
            "the squared routing weight, for gate/up and down alike; recorded "
            "moments are ignored. Traces are priced before any removal. "
            "'topk_uniform' uses the same tokens unweighted. Streaming traces "
            "only; changes the deployed weights and REQUIRES its own --trace-dir."
        ),
    )
    group_factors.add_argument(
        "--damage-unit",
        choices=DAMAGE_UNITS,
        default=None,
        help=(
            "unit of every per-token damage the planner prices: 'amplitude' "
            "g||f - fhat|| (L2 norm, REAP's unit; the propagation probe measures "
            "logit KL growing like |delta|^1.25, much nearer amplitude than "
            "energy) or 'energy' g^2||f - fhat||^2 (squared L2, the original). "
            "Default 'amplitude' for --planner distortion_risk on streaming "
            "traces, 'energy' otherwise. SVD factors are still fitted by least "
            "squares. Changes every trace number and REQUIRES its own --trace-dir."
        ),
    )
    group_factors.add_argument(
        "--damage-reference",
        choices=("layer_mean", "none"),
        default="layer_mean",
        help=(
            "what distortion_risk and mean_loss divide per-token damage by: "
            "'layer_mean' (the "
            "layer's mean MoE output, the original) or 'none' (absolute damage). "
            "Within a layer this is one constant; it only changes how layers "
            "trade off. The propagation factors are fitted per relative error, "
            "so 'none' mixes units. Applied at pricing time; traces are reusable."
        ),
    )
    group_factors.add_argument("--scale-floor", type=float, default=1e-12)
    group_factors.add_argument("--exact-svd", action="store_true")
    group_refinement = parser.add_argument_group("refinement", "iterative improvement passes over a finished allocation")
    group_refinement.add_argument(
        "--mean-loss-frequency-power",
        type=float,
        default=0.0,
        help=(
            "mean_loss only: alpha in price *= (activations / layer mean)**alpha. "
            "0 = mean per activation (frequency-blind, the default); 1 = total "
            "damage, so a busy expert outprices an equally-lossy quiet one. "
            "Pricing-time only, so existing traces are reusable"
        ),
    )
    group_refinement.add_argument(
        "--planner",
        choices=[
            "distortion",
            "importance_compressibility",
            "posterior_distortion",
            "factorized_risk",
            "propagated_risk",
            "distortion_risk",
            "mean_loss",
        ],
        default="distortion",
        help=(
            "initial allocation objective: routed additive distortion or "
            "conditional REAP importance times confidence-shrunk compressibility, "
            "prompt-clustered empirical-Bayes functional distortion, or "
            "posterior routed exposure times conditional compressibility, or "
            "linear-routing importance times degradation and propagated damage, "
            "or the mean-CVaR distortion risk of the measured per-token damage"
        ),
    )
    group_ranking.add_argument(
        "--propagation-svd-scale",
        type=float,
        default=None,
        help=(
            "re-evaluate the stored propagation damage curve for low-rank "
            "actions at this relative perturbation scale instead of using the "
            "factors the calibration tool saved"
        ),
    )
    group_ranking.add_argument(
        "--propagation-removal-scale",
        type=float,
        default=None,
        help=(
            "re-evaluate the stored propagation damage curve for removal at "
            "this relative perturbation scale"
        ),
    )
    group_ranking.add_argument(
        "--tail-fraction",
        type=float,
        default=0.01,
        help=(
            "distortion_risk beta: the worst fraction of corpus tokens the risk "
            "measure averages over. An expert routed less often than beta is "
            "amplified by 1/beta relative to its expected damage, so beta must "
            "sit well below the mean routing frequency top_k/experts or every "
            "expert saturates and the tail term stops discriminating"
        ),
    )
    group_ranking.add_argument(
        "--tail-weight",
        type=float,
        default=0.5,
        help=(
            "distortion_risk lambda mixing expected damage with the tail term; "
            "0 scores the exact expected relative damage and 1 uses CVaR alone"
        ),
    )
    group_ranking.add_argument(
        "--frequency-floor-tokens",
        type=float,
        default=1.0,
        help=(
            "distortion_risk floor on activation frequency, expressed in "
            "calibration tokens, so an expert never routed in the corpus is not "
            "scored as free to remove"
        ),
    )
    group_ranking.add_argument(
        "--frequency-temper",
        type=float,
        default=0.0,
        help=(
            "distortion_risk exponent dividing routing frequency out of the risk. "
            "0 keeps the exposure-weighted measure, which ranks experts almost "
            "exactly as REAP's reap_sum does and therefore deletes rare "
            "specialists in favour of frequent substitutable experts; 1 scores an "
            "expert purely on damage per token it is routed to, the quantity "
            "REAP's reap_mean measures"
        ),
    )
    group_ranking.add_argument(
        "--margin-proxy-weight",
        type=float,
        default=0.0,
        help=(
            "distortion_risk weight for a first-order residual-amplitude proxy "
            "to logit-margin damage. It is normalized into energy-risk units "
            "per action/category; 0 reproduces the previous equation exactly"
        ),
    )
    group_ranking.add_argument(
        "--risk-uncertainty-z",
        type=float,
        default=1.0,
        help=(
            "one-sided posterior uncertainty multiplier used by risk planners' "
            "importance, SVD degradation, and rerouting-removal guards"
        ),
    )
    group_ranking.add_argument(
        "--disable-rerouting-removal-guard",
        action="store_true",
        help=(
            "use importance alone for factorized/propagated removal risk instead of the "
            "maximum of importance and measured rerouting-aware removal risk"
        ),
    )
    group_refinement.add_argument(
        "--removal-fallback-depths",
        type=int,
        default=1,
        help=(
            "how many router-shortlist fallback depths to measure removal damage "
            "against. Depth zero is the top-(k+1) expert; deeper entries are what "
            "a plan that also removed the nearer candidates incurs"
        ),
    )
    group_refinement.add_argument(
        "--removal-depth-rounds",
        type=int,
        default=0,
        help=(
            "retention-aware replanning rounds for distortion_risk: each feeds the "
            "previous plan's retained fraction back into the removal-depth mixture "
            "and reallocates. Zero keeps the nearest-fallback removal risk"
        ),
    )
    group_refinement.add_argument(
        "--removal-depth-damping",
        type=float,
        default=0.5,
        help=(
            "damping on the retention fixed point; the feedback is negative, so "
            "one can oscillate and 0.5 is a safe default"
        ),
    )
    group_refinement.add_argument(
        "--posterior-min-active-prompts",
        type=int,
        default=2,
        help=(
            "minimum independent prompts activating an expert before its "
            "measured functional risk can override the structural prior"
        ),
    )
    group_refinement.add_argument(
        "--posterior-epsilon",
        type=float,
        default=1e-12,
        help="numerical floor used by the log-risk posterior planner",
    )
    group_refinement.add_argument(
        "--rerouting-refinement-rounds",
        type=int,
        default=0,
        help=(
            "full-token closed-loop refinement rounds after streaming allocation; "
            "zero disables rerouting-conditioned refinement"
        ),
    )
    group_refinement.add_argument(
        "--rerouting-swaps-per-round",
        type=int,
        default=16,
        help="maximum memory-neutral adjacent-action swaps accepted per round",
    )
    group_refinement.add_argument(
        "--rerouting-candidates-per-boundary",
        type=int,
        default=64,
        help="best upgrades and downgrades retained per action boundary",
    )
    group_refinement.add_argument(
        "--rerouting-chunk-size",
        type=int,
        default=256,
        help="cached token states evaluated at once during conditioned replay",
    )
    group_runtime = parser.add_argument_group("runtime", "I/O, devices and caches; none of these change the plan")
    group_runtime.add_argument("--reuse-hidden-cache", action="store_true")
    group_runtime.add_argument(
        "--tokens-per-category",
        type=int,
        default=256,
        help=(
            "random token states per category for projected/gram traces; "
            "streaming mode always consumes every cached token"
        ),
    )
    group_runtime.add_argument(
        "--trace-mode",
        choices=["projected", "gram", "streaming"],
        default="projected",
        help=(
            "'gram' evaluates exact full-output squared error with no random "
            "projection; 'projected' retains the legacy sketch trace; "
            "'streaming' consumes every cached token and stores compact exact "
            "single-action statistics"
        ),
    )
    group_runtime.add_argument("--routing-batch-size", type=int, default=2048)
    group_runtime.add_argument("--chunk-size", type=int, default=128)
    group_runtime.add_argument("--max-layers", type=int, default=None)
    group_runtime.add_argument("--trace-dir", default=None)
    group_runtime.add_argument(
        "--traces-only",
        action="store_true",
        help="build and save every layer trace to --trace-dir, then exit without "
        "allocating; plan later on CPU with --reuse-traces",
    )
    group_runtime.add_argument(
        "--reuse-traces",
        action="store_true",
        help="load existing per-layer trace files from --trace-dir when available",
    )
    group_runtime.add_argument("--out", required=True)
    group_runtime.add_argument(
        "--factors-out-dir",
        default=None,
        help=(
            "optional directory for the exact selected BF16 factors evaluated "
            "and loaded by inference"
        ),
    )
    group_runtime.add_argument("--device", default="cuda:0")
    group_runtime.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    group_runtime.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.factor_calibration is None:
        # The ASVD-style calibration is the default wherever it exists. The
        # shortlist (top-L tokens, recorded sens_t gate/up moments) remains for
        # the legacy trace modes and for reproducing plans built with it.
        args.factor_calibration = (
            "topk_g2" if args.trace_mode == "streaming" else "shortlist"
        )
    if args.damage_unit is None:
        # Amplitude is implemented for the streaming distortion-risk path only;
        # every other planner keeps the squared error it was built around.
        args.damage_unit = (
            "amplitude"
            if args.planner in ("distortion_risk", "mean_loss")
            and args.trace_mode == "streaming"
            else "energy"
        )
    if not (0 < args.memory_ratio <= 1):
        parser.error("--memory-ratio must be in (0, 1]")
    if args.confidence_kappa < 0:
        parser.error("--confidence-kappa must be nonnegative")
    if args.importance_power <= 0:
        parser.error("--importance-power must be positive")
    if args.posterior_min_active_prompts < 2:
        parser.error("--posterior-min-active-prompts must be at least 2")
    if args.posterior_epsilon <= 0:
        parser.error("--posterior-epsilon must be positive")
    if min(args.tokens_per_category, args.router_candidates, args.sketch_dim) <= 0:
        parser.error("token, shortlist, and sketch sizes must be positive")
    if args.reuse_traces and not args.trace_dir:
        parser.error("--reuse-traces requires --trace-dir")
    if args.traces_only and not args.trace_dir:
        parser.error("--traces-only requires --trace-dir")
    if args.factors_out_dir and args.trace_mode not in {"gram", "streaming"}:
        parser.error("--factors-out-dir requires --trace-mode gram or streaming")
    if args.removal_tiebreak_rtol < 0:
        parser.error("--removal-tiebreak-rtol must be nonnegative")
    if args.risk_uncertainty_z < 0:
        parser.error("--risk-uncertainty-z must be nonnegative")
    if args.token_energy_floor_ratio < 0:
        parser.error("--token-energy-floor-ratio must be nonnegative")
    if args.min_retained_experts < 0:
        parser.error("--min-retained-experts must be nonnegative")
    if not (0 <= args.min_layer_memory_ratio <= 1):
        parser.error("--min-layer-memory-ratio must be in [0, 1]")
    if (
        args.max_layer_relative_distortion is not None
        and args.max_layer_relative_distortion < 0
    ):
        parser.error("--max-layer-relative-distortion must be nonnegative")
    if not (0 <= args.occupancy_alpha <= 1):
        parser.error("--occupancy-alpha must be in [0, 1]")
    if args.factor_calibration != "shortlist" and args.trace_mode != "streaming":
        parser.error(
            "--factor-calibration other than 'shortlist' needs --trace-mode "
            "streaming; the projected and gram trace builders calibrate on the "
            "shortlist only"
        )
    if args.damage_unit == "amplitude":
        conflicts = [
            flag
            for flag, active in (
                ("--trace-mode " + args.trace_mode, args.trace_mode != "streaming"),
                (
                    "--planner " + args.planner,
                    args.planner not in ("distortion_risk", "mean_loss"),
                ),
                ("--margin-proxy-weight", args.margin_proxy_weight > 0),
                ("--action-interaction-rho", float(args.action_interaction_rho) != 0),
                ("--occupancy-alpha", float(args.occupancy_alpha) != 1.0),
                ("--routing-penalty-calibrate", bool(args.routing_penalty_calibrate)),
            )
            if active
        ]
        if conflicts:
            parser.error(
                "--damage-unit amplitude is implemented for the streaming "
                "distortion_risk and mean_loss planners only, and these are "
                f"defined on squared error: {', '.join(conflicts)}. Pass "
                "--damage-unit energy."
            )
    if args.damage_reference == "none" and args.planner not in (
        "distortion_risk", "mean_loss"
    ):
        parser.error(
            "--damage-reference none is implemented for --planner "
            "distortion_risk and mean_loss only"
        )
    if args.mean_loss_frequency_power and args.planner != "mean_loss":
        parser.error("--mean-loss-frequency-power applies to --planner mean_loss")
    if args.tier_price_scale_json and args.planner != "mean_loss":
        parser.error("--tier-price-scale-json applies to --planner mean_loss")
    if args.planner == "mean_loss":
        if args.trace_mode != "streaming":
            parser.error("--planner mean_loss needs --trace-mode streaming")
        if args.removal_depth_rounds:
            parser.error(
                "--removal-depth-rounds re-prices removal inside distortion_risk; "
                "mean_loss does not use it"
            )
    if args.planner == "propagated_risk" and args.propagation_path is None:
        parser.error("--planner propagated_risk requires --propagation-path")
    for flag, value in (
        ("--propagation-svd-scale", args.propagation_svd_scale),
        ("--propagation-removal-scale", args.propagation_removal_scale),
    ):
        if value is not None and value <= 0:
            parser.error(f"{flag} must be positive")
    if args.removal_fallback_depths < 1:
        parser.error("--removal-fallback-depths must be at least 1")
    if args.removal_depth_rounds < 0:
        parser.error("--removal-depth-rounds must be nonnegative")
    if not (0 < args.removal_depth_damping <= 1):
        parser.error("--removal-depth-damping must be in (0, 1]")
    if not (0 < args.tail_fraction <= 1):
        parser.error("--tail-fraction must be in (0, 1]")
    if not (0 <= args.tail_weight <= 1):
        parser.error("--tail-weight must be in [0, 1]")
    if not (0 <= args.frequency_temper <= 1):
        parser.error("--frequency-temper must be in [0, 1]")
    if args.margin_proxy_weight < 0:
        parser.error("--margin-proxy-weight must be nonnegative")
    if args.routing_penalty_scale:
        # Retired. Discounting a compressed expert's router logit is the same
        # operation as scaling its output by gamma < 1, and the least-squares
        # optimum is gamma* = 1 + <rho, fhat> / ||fhat||^2 with rho = f - fhat.
        # But <rho, fhat> = 0 is precisely the first-order condition the SVD fit
        # already imposes on the calibration distribution, so gamma* = 1: the
        # discount moves AWAY from the optimum rather than toward it. The
        # measurement agreed -- the calibrated fit returned a zero threshold on
        # all 18,432 cells, dropping a token never beating serving it truncated
        # (minimum ratio 2.03x). Removal is the only action that may move
        # routing, because it is the only one that takes the expert away.
        parser.error(
            "--routing-penalty-scale is retired: discounting an SVD expert's "
            "router logit sets gamma < 1 when the SVD fit's own optimality "
            "condition already puts the optimum at gamma = 1, so it can only "
            "increase error. The calibrated fit measured this directly, "
            "returning a zero threshold on all 18,432 cells. Pass "
            "--routing-penalty-calibrate alone (scale 0) to record the fitted "
            "thresholds as a diagnostic without repricing any action."
        )
    if args.frequency_floor_tokens < 0:
        parser.error("--frequency-floor-tokens must be nonnegative")
    if args.planner == "distortion_risk":
        if args.trace_mode != "streaming":
            parser.error("--planner distortion_risk requires --trace-mode streaming")
        if args.occupancy_alpha != 1.0:
            # The tail measure replaces frequency tempering, and this planner
            # reads the untempered numerators, so a non-unit alpha would only
            # look like it were doing something.
            logging.warning(
                "--occupancy-alpha %.3g is ignored by --planner distortion_risk; "
                "control the frequency/tail trade-off with --tail-weight and "
                "--tail-fraction instead",
                args.occupancy_alpha,
            )
    if min(
        args.rerouting_swaps_per_round,
        args.rerouting_candidates_per_boundary,
        args.rerouting_chunk_size,
    ) <= 0:
        parser.error("rerouting swap, candidate, and chunk counts must be positive")
    if args.rerouting_refinement_rounds < 0:
        parser.error("--rerouting-refinement-rounds must be nonnegative")
    if args.rerouting_refinement_rounds:
        if args.trace_mode != "streaming":
            parser.error("rerouting-conditioned refinement requires --trace-mode streaming")
        if args.token_reduction != "energy":
            parser.error(
                "rerouting-conditioned refinement currently requires "
                "--token-reduction energy"
            )
        if args.routing_kept_path is not None:
            parser.error(
                "rerouting-conditioned refinement currently requires full-pool routing"
            )
    if args.planner in {
        "importance_compressibility",
        "posterior_distortion",
        "factorized_risk",
        "propagated_risk",
    }:
        if args.trace_mode != "streaming":
            parser.error(
                f"{args.planner.replace('_', '-')} planning requires "
                "--trace-mode streaming"
            )
        if args.max_layer_relative_distortion is not None:
            parser.error(
                "--max-layer-relative-distortion is not defined for the "
                f"{args.planner.replace('_', '-')} risk objective"
            )

    artifact = _load_moments(args.moments)
    gate_moments = _select_projection_moments(artifact, args.basis_category, "gate")
    up_moments = _select_projection_moments(artifact, args.basis_category, "up")
    sequence_artifact = _load_moments(args.sequence_cache)
    sequences = sequence_artifact["sequences"]
    full_routing = args.routing_kept_path is None
    if full_routing:
        module_names = list(gate_moments)
        routing_kept = {
            name: torch.arange(int(gate_moments[name].shape[0]), dtype=torch.long)
            for name in module_names
        }
        routing_meta = {"selection_category": "all_original_experts"}
    else:
        routing_kept, routing_meta = load_selection(args.routing_kept_path)
        module_names = [name for name in routing_kept if name in gate_moments]
    if args.max_layers is not None:
        module_names = module_names[: int(args.max_layers)]
    if not module_names:
        raise RuntimeError("no common supported MoE layers in the moment artifact")
    metadata_path = Path(args.hidden_cache_dir) / "metadata.json"
    trace_paths = {
        name: (
            None
            if not args.trace_dir
            else Path(args.trace_dir) / f"{name.replace('.', '_')}.pt"
        )
        for name in module_names
    }
    complete_trace_reuse = bool(args.reuse_traces) and all(
        path is not None and path.exists() for path in trace_paths.values()
    )
    model_required_after_trace_reuse = bool(
        args.rerouting_refinement_rounds or args.factors_out_dir
    )
    model = None
    modules = {}
    if complete_trace_reuse and not model_required_after_trace_reuse:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"hidden-cache metadata required to validate cached traces: {metadata_path}"
            )
        hidden_metadata = _load_hidden_metadata(args.hidden_cache_dir)
        _validate_hidden_cache_routing(
            hidden_metadata,
            routing_kept,
            full_routing=full_routing,
            routing_kept_path=args.routing_kept_path,
            sequence_meta=sequence_artifact.get("meta") or {},
        )
        down_moments = _select_projection_moments(
            artifact, args.basis_category, "down"
        )
        hidden_size = int(gate_moments[module_names[0]].shape[1])
        intermediate = int(down_moments[module_names[0]].shape[1])
        if any(
            int(gate_moments[name].shape[1]) != hidden_size
            or int(down_moments[name].shape[1]) != intermediate
            for name in module_names
        ):
            raise ValueError("cached layers do not share one expert projection shape")
        logging.info(
            "Reusing complete functional trace set without loading the model"
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, args.dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logging.info("Loading %s directly on %s", args.model, args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": args.device},
        ).eval()
        from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_target

        logging.info("Converted %d MoE blocks to stacked form", apply_stacked_target(model))
        modules = dict(model.named_modules())
        if full_routing:
            module_names = [
                name
                for name, module in modules.items()
                if name in gate_moments and _is_qwen3_moe_block(module)
            ]
            if args.max_layers is not None:
                module_names = module_names[: int(args.max_layers)]
            routing_kept = _validate_full_expert_coverage(
                module_names,
                modules,
                artifact,
                gate_moments,
                up_moments,
            )
        for name in module_names:
            module = modules[name]
            if not _is_qwen3_moe_block(module):
                raise TypeError(f"{name} is not a supported stacked Qwen3 MoE block")
            if full_routing:
                module.set_full()
            else:
                module.set_kept(torch.as_tensor(routing_kept[name], dtype=torch.long))
        if (args.reuse_hidden_cache or complete_trace_reuse) and metadata_path.exists():
            hidden_metadata = _load_hidden_metadata(args.hidden_cache_dir)
            _validate_hidden_cache_routing(
                hidden_metadata,
                routing_kept,
                full_routing=full_routing,
                routing_kept_path=args.routing_kept_path,
                sequence_meta=sequence_artifact.get("meta") or {},
            )
            logging.info("Reusing hidden cache <- %s", metadata_path)
        else:
            generator = _make_generator(model, tokenizer, args.chunk_size)
            hidden_metadata = _capture_hidden_states(
                model,
                generator,
                sequences,
                module_names,
                cache_dir=args.hidden_cache_dir,
                device=args.device,
                chunk_size=args.chunk_size,
            )
            hidden_metadata["routing_pool_mode"] = "full" if full_routing else "restricted"
            hidden_metadata["routing_kept_path"] = args.routing_kept_path
            hidden_metadata["routing_pool_ids"] = {
                name: [int(value) for value in torch.as_tensor(ids).tolist()]
                for name, ids in routing_kept.items()
                if name in module_names
            }
            with open(metadata_path, "w") as handle:
                json.dump(hidden_metadata, handle, indent=2)
            del generator
        sample = modules[module_names[0]]
        sample_gate = expert_projection_weight(sample, 0, "gate")
        intermediate, hidden_size = map(int, sample_gate.shape)

    if full_routing:
        logging.info("Analysis routing uses all original experts in every layer")
    else:
        logging.info(
            "Restricted analysis routing with %s (%s)",
            args.routing_kept_path,
            routing_meta.get("selection_category", "unknown"),
        )
    if complete_trace_reuse:
        logging.info("Reusing hidden cache <- %s", metadata_path)

    total_rows = int(hidden_metadata["rows"][module_names[0]])
    selected_token_limit = (
        total_rows if args.trace_mode == "streaming" else args.tokens_per_category
    )
    (
        row_ids,
        category_ids,
        prompt_ids,
        prompt_category_ids,
        category_names,
        category_ranges,
    ) = _selected_calibration_rows(
        sequences,
        total_rows=total_rows,
        tokens_per_category=selected_token_limit,
        seed=args.seed,
    )
    rank_pairs, action_names, action_cost = _action_spec(
        args.rank_pairs, hidden_size, intermediate
    )
    logging.info("Actions: %s", ", ".join(action_names))

    traces = []
    global_ids = {}
    calibration_fallbacks = {}
    usage_stats = {}
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)
    for layer_index, name in enumerate(module_names):
        trace_path = trace_paths[name]
        if args.reuse_traces and trace_path is not None and trace_path.exists():
            saved = _load_moments(str(trace_path))
            trace_kind = saved.get("trace_kind", "functional")
            expected_scope = _factor_calibration_scope(
                args.trace_mode, args.factor_metric, args.gram_shrink_target,
                args.down_compensation,
                factor_calibration=args.factor_calibration,
                factor_output_metric=args.factor_output_metric,
            )
            if args.trace_mode == "streaming" and saved.get(
                "factor_calibration_scope"
            ) != expected_scope:
                raise ValueError(
                    f"cached trace {trace_path} was calibrated as "
                    f"{saved.get('factor_calibration_scope')!r} but this run needs "
                    f"{expected_scope!r} (--factor-metric {args.factor_metric}, "
                    f"--factor-calibration {args.factor_calibration}); "
                    "use a different --trace-dir, rerun without --reuse-traces, "
                    "or pass --factor-calibration shortlist for traces built "
                    "before that option existed"
                )
            cached_usage = saved.get("usage_stats") or {}
            penalty_mismatch = float(
                saved.get("routing_penalty_scale", 0.0)
            ) != float(args.routing_penalty_scale) or bool(
                saved.get("routing_penalty_calibrate", False)
            ) != bool(args.routing_penalty_calibrate)
            if penalty_mismatch and float(args.routing_penalty_scale) == 0.0:
                # The penalty is retired, so every run now prices at scale 0. A
                # trace cached under a nonzero scale is still identical to an
                # unpriced one whenever every fitted threshold came out zero: a
                # token is displaced only where margin < lam, and lam == 0 against
                # a nonnegative margin displaces nothing. That is what the flag
                # did in every measured run, so reuse those traces instead of
                # forcing a rebuild that cannot change a number.
                penalty_keys = [
                    key
                    for key in cached_usage
                    if key.startswith("routing_penalty_")
                ]
                if penalty_keys and not any(
                    bool(torch.as_tensor(cached_usage[key]).any())
                    for key in penalty_keys
                ):
                    penalty_mismatch = False
            if args.trace_mode == "streaming" and penalty_mismatch:
                raise ValueError(
                    f"cached trace {trace_path} was priced at routing penalty "
                    f"scale={saved.get('routing_penalty_scale', 0.0)} "
                    f"calibrate={saved.get('routing_penalty_calibrate', False)}, "
                    f"but this run needs scale={args.routing_penalty_scale} "
                    f"calibrate={args.routing_penalty_calibrate}. The penalty "
                    "reprices every action, so use a different --trace-dir or "
                    "rerun without --reuse-traces."
                )
            if (
                args.trace_mode == "streaming"
                and float(saved.get("occupancy_alpha", 1.0))
                != float(args.occupancy_alpha)
            ):
                raise ValueError(
                    f"cached trace {trace_path} uses occupancy alpha "
                    f"{saved.get('occupancy_alpha', 1.0)}, requested "
                    f"{args.occupancy_alpha}"
                )
            saved_unit = str(saved.get("damage_unit", "energy"))
            if args.trace_mode == "streaming" and saved_unit != args.damage_unit:
                raise ValueError(
                    f"cached trace {trace_path} prices damage as {saved_unit!r}, "
                    f"but this run needs {args.damage_unit!r}; use a different "
                    "--trace-dir, rerun without --reuse-traces, or pass "
                    f"--damage-unit {saved_unit}"
                )
            trace = (
                LayerIndependentActionTrace.from_state_dict(saved["trace"])
                if trace_kind == "independent_action"
                else LayerFunctionalTrace.from_state_dict(saved["trace"])
            )
            expert_ids = torch.as_tensor(saved["global_expert_ids"], dtype=torch.long)
            layer_fallbacks = dict(saved.get("calibration_fallbacks") or {})
            layer_usage = saved.get("usage_stats")
            lacks_prompt_statistics = (
                args.planner == "posterior_distortion"
                and isinstance(trace, LayerIndependentActionTrace)
                and trace.prompt_action_numerator is None
            )
            lacks_factorized_statistics = (
                args.planner in {"factorized_risk", "propagated_risk"}
                and isinstance(trace, LayerIndependentActionTrace)
                and trace.prompt_expert_output_energy is None
            )
            lacks_propagated_statistics = (
                args.planner == "propagated_risk"
                and isinstance(trace, LayerIndependentActionTrace)
                and (
                    trace.linear_routing_output_energy is None
                    or trace.prompt_linear_routing_output_energy is None
                    or trace.prompt_active_token_count is None
                )
            )
            lacks_log_damage_statistics = (
                args.planner == "distortion_risk"
                and isinstance(trace, LayerIndependentActionTrace)
                and (
                    trace.damage_log_sum is None
                    or trace.prompt_active_token_count is None
                )
            )
            if trace.name != name or trace.action_names != tuple(action_names):
                raise ValueError(
                    f"cached trace {trace_path} is incompatible with current actions"
                )
            expected_kind = (
                "independent_action"
                if args.trace_mode == "streaming"
                else "functional"
            )
            if trace_kind != expected_kind or (
                args.trace_mode != "streaming"
                and trace.uses_gram != (args.trace_mode == "gram")
            ):
                raise ValueError(
                    f"cached trace {trace_path} does not use {args.trace_mode} mode"
                )
            expected_ids = torch.as_tensor(routing_kept[name], dtype=torch.long)
            if not torch.equal(expert_ids, expected_ids):
                raise ValueError(
                    f"cached trace {trace_path} uses a different routing pool"
                )
            if (
                lacks_prompt_statistics
                or lacks_factorized_statistics
                or lacks_propagated_statistics
                or lacks_log_damage_statistics
            ):
                if complete_trace_reuse:
                    raise ValueError(
                        f"cached trace {trace_path} has no prompt-clustered "
                        "statistics; rerun without --reuse-traces to rebuild it"
                    )
                logging.info(
                    "Rebuilding legacy trace without prompt statistics: %s",
                    trace_path,
                )
            elif layer_usage is None:
                if complete_trace_reuse:
                    raise ValueError(
                        f"cached trace {trace_path} has no REAP usage statistics; "
                        "rerun without --reuse-traces to rebuild it"
                    )
                logging.info(
                    "Rebuilding legacy trace without REAP usage statistics: %s",
                    trace_path,
                )
            else:
                traces.append(trace)
                global_ids[name] = expert_ids
                calibration_fallbacks[name] = layer_fallbacks
                usage_stats[name] = {
                    metric: torch.as_tensor(values).cpu()
                    for metric, values in layer_usage.items()
                }
                logging.info("Reused routed functional trace <- %s", trace_path)
                continue
        logging.info("Building routed functional trace for %s", name)
        all_hidden = _load_hidden_cache(hidden_metadata, name)
        hidden = _select_hidden_rows_without_full_copy(all_hidden, row_ids)
        del all_hidden
        builder = {
            "projected": _build_layer_trace,
            "gram": _build_layer_gram_trace,
            "streaming": _build_layer_streaming_trace,
        }[args.trace_mode]
        builder_kwargs = dict(
            device=args.device,
            routing_batch_size=args.routing_batch_size,
            router_candidates=args.router_candidates,
            randomized=not args.exact_svd,
            scale_floor=args.scale_floor,
            seed=args.seed + layer_index * 1_000_003,
        )
        if args.trace_mode == "streaming":
            builder_kwargs["occupancy_alpha"] = args.occupancy_alpha
            builder_kwargs["gram_shrink_target_mode"] = args.gram_shrink_target
            builder_kwargs["down_compensation"] = args.down_compensation
            builder_kwargs["removal_fallback_depths"] = args.removal_fallback_depths
            builder_kwargs["prompt_ids"] = prompt_ids
            builder_kwargs["prompt_category_ids"] = prompt_category_ids
            builder_kwargs["gram_metric"] = args.factor_metric == "gram"
            builder_kwargs["factor_calibration"] = args.factor_calibration
            builder_kwargs["factor_output_metric"] = args.factor_output_metric
            builder_kwargs["damage_unit"] = args.damage_unit
            builder_kwargs["routing_penalty_scale"] = args.routing_penalty_scale
            builder_kwargs["routing_penalty_calibrate"] = (
                args.routing_penalty_calibrate
            )
        if args.trace_mode == "projected":
            builder_kwargs["sketch_dim"] = args.sketch_dim
        if args.routing_penalty_scale and args.removal_fallback_depths < 1:
            raise ValueError(
                "--routing-penalty-scale prices a displaced token as removal, "
                "which needs at least one removal fallback depth"
            )
        if args.routing_penalty_scale and args.trace_mode != "streaming":
            raise ValueError(
                "--routing-penalty-scale prices actions from per-token routing "
                f"margins, which only the streaming trace retains; got "
                f"--trace-mode {args.trace_mode}"
            )
        trace, expert_ids, layer_fallbacks, layer_usage = builder(
            modules[name],
            name,
            hidden,
            category_ids,
            category_names,
            gate_moments[name],
            up_moments[name],
            rank_pairs,
            action_names,
            action_cost,
            **builder_kwargs,
        )
        traces.append(trace)
        global_ids[name] = expert_ids
        calibration_fallbacks[name] = layer_fallbacks
        usage_stats[name] = layer_usage
        if trace_path is not None:
            torch.save(
                {
                    "trace_kind": (
                        "independent_action"
                        if args.trace_mode == "streaming"
                        else "functional"
                    ),
                    # The penalty REPRICES action_numerator (tokens whose margin
                    # does not survive lam are scored as removal), so a trace is
                    # only valid for the penalty it was built with. Recorded here
                    # and checked on reuse, exactly like occupancy_alpha.
                    "routing_penalty_scale": float(args.routing_penalty_scale),
                    "routing_penalty_calibrate": bool(
                        args.routing_penalty_calibrate
                    ),
                    "trace": trace.state_dict(),
                    "global_expert_ids": expert_ids,
                    "calibration_fallbacks": layer_fallbacks,
                    "usage_stats": layer_usage,
                    "occupancy_alpha": (
                        float(args.occupancy_alpha)
                        if args.trace_mode == "streaming"
                        else None
                    ),
                    "factor_calibration_scope": _factor_calibration_scope(
                        args.trace_mode, args.factor_metric,
                        args.gram_shrink_target, args.down_compensation,
                        factor_calibration=args.factor_calibration,
                        factor_output_metric=args.factor_output_metric,
                    ),
                    "damage_unit": str(args.damage_unit),
                    "importance_compressibility_statistics": (
                        "conditional_reap_output_energy_structural_v1"
                        if args.trace_mode == "streaming"
                        else None
                    ),
                    "posterior_distortion_statistics": (
                        "prompt_clustered_functional_structural_v1"
                        if args.trace_mode == "streaming"
                        else None
                    ),
                    "factorized_risk_statistics": (
                        "prompt_clustered_exposure_compressibility_v1"
                        if args.trace_mode == "streaming"
                        else None
                    ),
                    "propagated_risk_statistics": (
                        "linear_routing_energy_prompt_clustered_v1"
                        if args.trace_mode == "streaming"
                        else None
                    ),
                    "distortion_risk_statistics": (
                        "per_token_log_damage_moments_v1"
                        if args.trace_mode == "streaming"
                        else None
                    ),
                },
                trace_path,
            )
        del hidden
        gc.collect()
        torch.cuda.empty_cache()

    if args.traces_only:
        logging.info("--traces-only: %d layer traces in %s; skipping allocation",
                     len(traces), args.trace_dir)
        return

    full_memory = sum(
        trace.memory(torch.full((trace.experts,), trace.actions - 1))
        for trace in traces
    )
    budget = int(args.memory_ratio * full_memory)
    logging.info(
        "Planning to %.2f%% of pool expert parameters (%d / %d)",
        100 * args.memory_ratio,
        budget,
        full_memory,
    )
    allocation_traces = traces
    importance_compressibility = None
    posterior_distortion = None
    factorized_risk = None
    propagated_risk = None
    distortion_risk = None
    mean_loss = None
    if args.planner == "importance_compressibility":
        allocation_traces, importance_compressibility = (
            build_importance_compressibility_traces(
                traces,
                confidence_kappa=args.confidence_kappa,
                importance_power=args.importance_power,
                importance_prior=args.importance_prior,
            )
        )
        logging.info(
            "Initial allocation uses conditional importance x compressibility "
            "(kappa=%.3g, power=%.3g, prior=%s)",
            args.confidence_kappa,
            args.importance_power,
            args.importance_prior,
        )
    elif args.planner == "posterior_distortion":
        allocation_traces, posterior_distortion = (
            build_posterior_distortion_traces(
                traces,
                minimum_active_prompts=args.posterior_min_active_prompts,
                epsilon=args.posterior_epsilon,
            )
        )
        logging.info(
            "Initial allocation uses prompt-clustered posterior functional risk "
            "(minimum active prompts=%d)",
            args.posterior_min_active_prompts,
        )
    elif args.planner == "factorized_risk":
        allocation_traces, factorized_risk = build_factorized_risk_traces(
            traces,
            minimum_active_prompts=args.posterior_min_active_prompts,
            epsilon=args.posterior_epsilon,
            frequency_alpha=args.occupancy_alpha,
            uncertainty_z=args.risk_uncertainty_z,
            guard_removal_with_rerouting=(
                not args.disable_rerouting_removal_guard
            ),
        )
        logging.info(
            "Initial allocation uses posterior routed exposure x conditional "
            "compressibility (minimum active prompts=%d, frequency alpha=%.3g, "
            "uncertainty z=%.3g, rerouting removal guard=%s)",
            args.posterior_min_active_prompts,
            args.occupancy_alpha,
            args.risk_uncertainty_z,
            not args.disable_rerouting_removal_guard,
        )
    elif args.planner == "propagated_risk":
        propagation_factors = _load_propagation_factors(
            args.propagation_path,
            traces,
            svd_scale=args.propagation_svd_scale,
            removal_scale=args.propagation_removal_scale,
        )
        allocation_traces, propagated_risk = build_propagated_risk_traces(
            traces,
            propagation_factors,
            minimum_active_prompts=args.posterior_min_active_prompts,
            epsilon=args.posterior_epsilon,
            frequency_alpha=args.occupancy_alpha,
            uncertainty_z=args.risk_uncertainty_z,
            guard_removal_with_rerouting=(
                not args.disable_rerouting_removal_guard
            ),
        )
        logging.info(
            "Initial allocation uses q^alpha E[g||f||^2|active] importance x "
            "conditional SVD degradation x calibrated propagation damage "
            "(minimum active prompts=%d, frequency alpha=%.3g, "
            "uncertainty z=%.3g, rerouting removal guard=%s)",
            args.posterior_min_active_prompts,
            args.occupancy_alpha,
            args.risk_uncertainty_z,
            not args.disable_rerouting_removal_guard,
        )
    elif args.planner == "distortion_risk":
        propagation_factors = (
            None
            if args.propagation_path is None
            else _load_propagation_factors(
                args.propagation_path,
                traces,
                svd_scale=args.propagation_svd_scale,
                removal_scale=args.propagation_removal_scale,
            )
        )
        distortion_builder_kwargs = dict(
            propagation_factors=propagation_factors,
            tail_fraction=args.tail_fraction,
            tail_weight=args.tail_weight,
            minimum_active_prompts=args.posterior_min_active_prompts,
            uncertainty_z=args.risk_uncertainty_z,
            frequency_floor_tokens=args.frequency_floor_tokens,
            frequency_temper=args.frequency_temper,
            tier_price=args.tier_price,
            damage_reference=args.damage_reference,
            margin_proxy_weight=args.margin_proxy_weight,
            epsilon=args.posterior_epsilon,
        )
        allocation_traces, distortion_risk = build_distortion_risk_traces(
            traces, **distortion_builder_kwargs
        )
        logging.info(
            "Initial allocation uses the mean-CVaR distortion risk of measured "
            "per-token damage (tail fraction=%.4g, tail weight=%.3g, minimum "
            "active prompts=%d, uncertainty z=%.3g, margin proxy=%.3g, "
            "propagation=%s)",
            args.tail_fraction,
            args.tail_weight,
            args.posterior_min_active_prompts,
            args.risk_uncertainty_z,
            args.margin_proxy_weight,
            args.propagation_path or "none",
        )
    elif args.planner == "mean_loss":
        propagation_factors = (
            None
            if args.propagation_path is None
            else _load_propagation_factors(
                args.propagation_path,
                traces,
                svd_scale=args.propagation_svd_scale,
                removal_scale=args.propagation_removal_scale,
            )
        )
        if args.propagation_path is not None and (
            args.propagation_svd_scale is None
            or args.propagation_svd_scale != args.propagation_removal_scale
        ):
            logging.warning(
                "mean_loss with class-specific propagation scales prices tiers "
                "against removal by a random-noise artefact; pass one shared "
                "--propagation-svd-scale/--propagation-removal-scale"
            )
        allocation_traces, mean_loss = build_mean_loss_traces(
            traces,
            propagation_factors=propagation_factors,
            damage_reference=args.damage_reference,
            frequency_power=args.mean_loss_frequency_power,
            epsilon=args.posterior_epsilon,
            tier_price=args.tier_price,
            tier_price_scale=(
                json.loads(Path(args.tier_price_scale_json).read_text())["layers"]
                if args.tier_price_scale_json else None
            ),
        )
        logging.info(
            "Initial allocation prices every action by its mean %s loss per "
            "activation (reference=%s, propagation=%s, scales svd=%s removal=%s)",
            args.damage_unit,
            args.damage_reference,
            args.propagation_path or "none",
            args.propagation_svd_scale,
            args.propagation_removal_scale,
        )
    removal_saliency = (
        None
        if args.removal_tiebreak == "none"
        or args.planner
        in {
            "importance_compressibility",
            "posterior_distortion",
            "factorized_risk",
            "propagated_risk",
            "distortion_risk",
            "mean_loss",
        }
        else {
            name: values[args.removal_tiebreak]
            for name, values in usage_stats.items()
        }
    )
    # Experts the calibration never routed to -- absent from every token's top-k
    # AND from every shortlist -- are measured at zero damage for every action,
    # removal included. The greedy scores moves by damage-per-byte, so it cannot
    # distinguish removing such an expert from parking it on a cheap tier: both
    # score the same, and it stops wherever the budget runs out, leaving the
    # expert holding parameters that can never be used at inference. Start them
    # removed so the budget goes to experts that are actually routed to. This
    # also keeps them out of the SVD tiers, where the gram metric has no
    # calibration tokens to build a factor from.
    forced_initial_actions: dict[str, torch.Tensor] = {}
    forced_total = 0
    for trace in allocation_traces:
        stats = usage_stats.get(trace.name)
        if not stats or "topk_count" not in stats or "shortlist_count" not in stats:
            continue
        topk = torch.as_tensor(stats["topk_count"]).reshape(-1)
        shortlist = torch.as_tensor(stats["shortlist_count"]).reshape(-1)
        if int(topk.numel()) != int(trace.experts):
            continue
        # mean_loss prices only the tokens an expert serves, so an expert no
        # token routes to has no evidence at all, whatever its shortlist rank.
        dead = (
            topk <= 0
            if args.planner == "mean_loss"
            else (topk <= 0) & (shortlist <= 0)
        )
        # Never start below the retention floor: the greedy enforces it for its
        # own moves, so handing it an already-violating plan would be silent.
        budgeted = int(trace.experts) - int(args.min_retained_experts)
        if int(dead.sum().item()) > max(budgeted, 0):
            keep = dead.nonzero().flatten()[: max(budgeted, 0)]
            dead = torch.zeros_like(dead)
            dead[keep] = True
        if not bool(dead.any()):
            continue
        actions = torch.full(
            (int(trace.experts),), int(trace.actions) - 1, dtype=torch.long
        )
        actions[dead] = 0
        forced_initial_actions[trace.name] = actions
        forced_total += int(dead.sum().item())
    if forced_total:
        logging.info(
            "forced removal of %d never-routed experts across %d layers",
            forced_total,
            len(forced_initial_actions),
        )

    if float(args.action_interaction_rho):
        enabled = 0
        for trace in allocation_traces:
            if not hasattr(trace, "set_interaction_model"):
                continue
            occupancy = None
            stats = usage_stats.get(trace.name)
            if stats and "topk_count" in stats:
                counts = torch.as_tensor(stats["topk_count"]).double().reshape(-1, 1)
                tokens = float(counts.sum().item()) / float(trace.top_k)
                occupancy = counts / max(tokens, 1.0)
            trace.set_interaction_model(
                float(args.action_interaction_rho), occupancy=occupancy
            )
            enabled += 1
        logging.info(
            "co-perturbation interaction model rho=%.3f enabled on %d/%d layers",
            float(args.action_interaction_rho), enabled, len(allocation_traces),
        )

    allocator_kwargs = dict(
        memory_budget=budget,
        initial_actions=forced_initial_actions or None,
        objective=args.objective,
        cvar_fraction=args.cvar_fraction,
        removal_saliency=removal_saliency,
        removal_tiebreak_rtol=args.removal_tiebreak_rtol,
        token_reduction=args.token_reduction,
        token_energy_floor_ratio=args.token_energy_floor_ratio,
        layer_reduction=args.layer_reduction,
        min_retained_experts=args.min_retained_experts,
        min_layer_memory_ratio=args.min_layer_memory_ratio,
        max_layer_relative_distortion=args.max_layer_relative_distortion,
    )
    if args.layer_min_memory_json:
        raw_floors = json.loads(Path(args.layer_min_memory_json).read_text())
        names_by_index = {name.split(".")[2]: name for name in (t.name for t in allocation_traces)}
        allocator_kwargs["layer_min_memory_ratio"] = {
            names_by_index.get(str(key), str(key)): float(value)
            for key, value in raw_floors.items()
        }
    if float(args.min_layer_routed_mass) > 0.0:
        allocator_kwargs["min_layer_routed_mass"] = float(args.min_layer_routed_mass)
        allocator_kwargs["expert_routed_mass"] = {
            name: torch.as_tensor(stats["routing_mass"], dtype=torch.float64)
            for name, stats in usage_stats.items()
        }
    plan = cached_greedy_global_downgrade(allocation_traces, **allocator_kwargs)
    retention_history = []
    if args.planner == "distortion_risk" and args.removal_depth_rounds:
        plan, distortion_risk, retention_history = _retention_aware_replanning(
            plan,
            traces,
            builder_kwargs=distortion_builder_kwargs,
            allocator_kwargs=allocator_kwargs,
            rounds=args.removal_depth_rounds,
            damping=args.removal_depth_damping,
        )
    if args.tier_assignment != "risk":
        before = float(plan.objective)
        plan = _reassign_plan_tiers_by_saliency(
            plan,
            allocation_traces,
            usage_stats,
            saliency_key=args.tier_assignment,
            objective=args.objective,
            cvar_fraction=args.cvar_fraction,
            layer_reduction=args.layer_reduction,
        )
        logging.info(
            "reassigned fixed per-layer tier histograms by %s saliency "
            "(local proxy objective %.6g -> %.6g; memory unchanged)",
            args.tier_assignment,
            before,
            plan.objective,
        )
    conditioned_reports = None
    refinement_metadata = None
    down_moments = None
    if args.rerouting_refinement_rounds:
        down_moments = _select_projection_moments(
            artifact, args.basis_category, "down"
        )
        plan, conditioned_reports, refinement_metadata = (
            _rerouting_conditioned_refinement(
                plan,
                traces,
                modules,
                hidden_metadata,
                row_ids,
                category_ids,
                category_names,
                gate_moments,
                up_moments,
                down_moments,
                rank_pairs,
                objective=args.objective,
                cvar_fraction=args.cvar_fraction,
                layer_reduction=args.layer_reduction,
                min_retained_experts=args.min_retained_experts,
                min_layer_memory_ratio=args.min_layer_memory_ratio,
                max_layer_relative_distortion=args.max_layer_relative_distortion,
                rounds=args.rerouting_refinement_rounds,
                swaps_per_round=args.rerouting_swaps_per_round,
                candidates_per_boundary=args.rerouting_candidates_per_boundary,
                device=args.device,
                routing_batch_size=args.routing_batch_size,
                router_candidates=args.router_candidates,
                refinement_chunk_size=args.rerouting_chunk_size,
                randomized=not args.exact_svd,
                gram_metric=args.factor_metric == "gram",
                factor_calibration=args.factor_calibration,
                factor_output_metric=args.factor_output_metric,
                scale_floor=args.scale_floor,
                seed=args.seed,
            )
        )
    payload = _json_plan(
        plan,
        traces,
        global_ids,
        usage_stats,
        action_names=action_names,
        full_memory=full_memory,
        budget=budget,
        token_reduction=args.token_reduction,
        token_energy_floor_ratio=args.token_energy_floor_ratio,
        conditioned_reports=conditioned_reports,
        importance_compressibility=importance_compressibility,
        posterior_distortion=posterior_distortion,
        factorized_risk=factorized_risk,
        propagated_risk=propagated_risk,
        distortion_risk=distortion_risk,
        mean_loss=mean_loss,
    )
    if args.planner in {
        "importance_compressibility",
        "posterior_distortion",
        "factorized_risk",
        "propagated_risk",
        "distortion_risk",
        "mean_loss",
    }:
        payload["meta"]["objective"] = (
            "category-robust conditional importance-compressibility risk"
            if args.planner == "importance_compressibility"
            else "category-robust prompt-clustered posterior functional risk"
            if args.planner == "posterior_distortion"
            else "category-robust propagated importance-degradation risk"
            if args.planner == "propagated_risk"
            else "category-robust mean-CVaR per-token damage distortion risk"
            if args.planner == "distortion_risk"
            else "category-robust mean contribution loss per activation"
            if args.planner == "mean_loss"
            else "category-robust conservative importance-degradation risk"
        )
        payload["category_allocation_objective"] = dict(
            plan.category_relative_distortion
        )
    payload["meta"].update(
        {
            "model": args.model,
            "planner": args.planner,
            "confidence_kappa": (
                float(args.confidence_kappa)
                if args.planner == "importance_compressibility"
                else None
            ),
            "importance_power": (
                float(args.importance_power)
                if args.planner == "importance_compressibility"
                else None
            ),
            "importance_prior": (
                args.importance_prior
                if args.planner == "importance_compressibility"
                else None
            ),
            "posterior_min_active_prompts": (
                int(args.posterior_min_active_prompts)
                if args.planner
                in {
                    "posterior_distortion",
                    "factorized_risk",
                    "propagated_risk",
                    "distortion_risk",
                }
                else None
            ),
            "posterior_epsilon": (
                float(args.posterior_epsilon)
                if args.planner
                in {
                    "posterior_distortion",
                    "factorized_risk",
                    "propagated_risk",
                    "distortion_risk",
                }
                else None
            ),
            "distortion_risk_tail_fraction": (
                float(args.tail_fraction)
                if args.planner == "distortion_risk"
                else None
            ),
            "distortion_risk_tail_weight": (
                float(args.tail_weight)
                if args.planner == "distortion_risk"
                else None
            ),
            "distortion_risk_frequency_temper": float(args.frequency_temper),
                "distortion_risk_tier_price": float(args.tier_price),
                "gram_shrink_target": str(args.gram_shrink_target),
            # Recorded because each of these changes the plan or the factors.
            # Leaving them out made a shipped plan unreproducible, and the
            # propagation scales in particular ARE the SVD-vs-removal exchange
            # rate -- the single most consequential knob the planner exposes.
            "seed": int(args.seed),
            "scale_floor": float(args.scale_floor),
            "exact_svd": bool(args.exact_svd),
            "propagation_svd_scale": (
                None if args.propagation_svd_scale is None
                else float(args.propagation_svd_scale)
            ),
            "propagation_removal_scale": (
                None if args.propagation_removal_scale is None
                else float(args.propagation_removal_scale)
            ),
            "removal_depth_damping": float(args.removal_depth_damping),
            "down_compensation": bool(args.down_compensation),
            "routing_penalty_scale": float(args.routing_penalty_scale),
            # Names the formula, so a deployment cannot discount by a different
            # rule than the plan was priced under. Absent means an older plan
            # that was priced with no discount at all.
            "routing_penalty_calibrated": bool(args.routing_penalty_calibrate),
            "routing_penalty_definition": (
                "scale * fitted margin threshold, per expert and action"
                if args.routing_penalty_calibrate else
                "scale * sqrt(action_damage / routed_output_energy), per expert "
                "and action, both relative to the layer reference energy"
            ),
            "distortion_risk_margin_proxy_weight": (
                float(args.margin_proxy_weight)
                if args.planner == "distortion_risk"
                else None
            ),
            "tier_assignment": str(args.tier_assignment),
            "distortion_risk_frequency_floor_tokens": (
                float(args.frequency_floor_tokens)
                if args.planner == "distortion_risk"
                else None
            ),
            "distortion_risk_removal_fallback_depths": (
                int(args.removal_fallback_depths)
                if args.trace_mode == "streaming"
                else None
            ),
            "distortion_risk_retention_rounds": (
                int(args.removal_depth_rounds)
                if args.planner == "distortion_risk"
                else None
            ),
            "distortion_risk_retention_history": (
                retention_history if args.planner == "distortion_risk" else None
            ),
            "distortion_risk_uncertainty_z": (
                float(args.risk_uncertainty_z)
                if args.planner == "distortion_risk"
                else None
            ),
            "factorized_frequency_alpha": (
                float(args.occupancy_alpha)
                if args.planner in {"factorized_risk", "propagated_risk"}
                else None
            ),
            "factorized_uncertainty_z": (
                float(args.risk_uncertainty_z)
                if args.planner in {"factorized_risk", "propagated_risk"}
                else None
            ),
            "factorized_rerouting_removal_guard": (
                not args.disable_rerouting_removal_guard
                if args.planner in {"factorized_risk", "propagated_risk"}
                else None
            ),
            "propagation_path": (
                args.propagation_path
                if args.planner in {"propagated_risk", "distortion_risk"}
                else None
            ),
            "moments": args.moments,
            "basis_category": args.basis_category,
            "routing_kept_path": args.routing_kept_path,
            "routing_pool_mode": "full" if full_routing else "restricted",
            "routing_pool_size_by_layer": {
                name: int(torch.as_tensor(routing_kept[name]).numel())
                for name in module_names
            },
            "sequence_cache": args.sequence_cache,
            "hidden_cache_dir": args.hidden_cache_dir,
            "categories": list(category_names),
            "category_ranges": category_ranges,
            "tokens_per_category": (
                None
                if args.trace_mode == "streaming"
                else int(args.tokens_per_category)
            ),
            "requested_tokens_per_category": int(args.tokens_per_category),
            "calibration_token_selection": (
                "all_cached_tokens"
                if args.trace_mode == "streaming"
                else "uniform_random_without_replacement"
            ),
            "router_candidates": int(args.router_candidates),
            "sketch_dim": (
                None if args.trace_mode == "gram" else int(args.sketch_dim)
            ),
            "trace_mode": args.trace_mode,
            "action_interaction_model": (
                "iterative_rerouting_conditioned_exact_adjacent_marginals"
                if args.rerouting_refinement_rounds
                else "separable_conditional_importance_compressibility_risk"
                if args.planner == "importance_compressibility"
                else "separable_prompt_clustered_posterior_functional_risk"
                if args.planner == "posterior_distortion"
                else "separable_conservative_importance_times_conditional_degradation"
                if args.planner == "factorized_risk"
                else "separable_propagated_importance_times_conditional_degradation"
                if args.planner == "propagated_risk"
                else (
                    "additive_exact_single_action_full_token_statistics"
                    if not float(args.action_interaction_rho)
                    else "co_perturbation_corrected_rho"
                    f"{float(args.action_interaction_rho):g}"
                )
                if args.trace_mode == "streaming"
                else "joint_per_token"
            ),
            "occupancy_alpha": (
                float(args.occupancy_alpha)
                if args.trace_mode == "streaming"
                else None
            ),
            "factor_calibration_scope": _factor_calibration_scope(
                args.trace_mode, args.factor_metric, args.gram_shrink_target,
                args.down_compensation,
                factor_calibration=args.factor_calibration,
                factor_output_metric=args.factor_output_metric,
            ),
            "factor_metric": str(args.factor_metric),
            # Read back by the restructurer: it selects the deployed factors.
            "factor_calibration": str(args.factor_calibration),
            "factor_output_metric": str(args.factor_output_metric),
            "mean_loss_frequency_power": float(args.mean_loss_frequency_power),
            "mean_loss_tier_price": float(args.tier_price),
            "mean_loss_tier_price_scale_json": args.tier_price_scale_json,
            "min_layer_routed_mass": float(args.min_layer_routed_mass),
            # Units of every priced damage and of single_action_relative_distortion.
            "damage_unit": str(args.damage_unit),
            "damage_reference": str(args.damage_reference),
            "rank_pairs": [list(pair) for pair in rank_pairs],
            "calibration_fallbacks_by_layer": calibration_fallbacks,
            "objective_reducer": args.objective,
            "distortion_definition": (
                "normalized_conditional_reap_importance_power_times_"
                "confidence_shrunk_relative_action_error"
                if args.planner == "importance_compressibility"
                else "prompt_clustered_empirical_bayes_posterior_of_"
                "layer_moe_output_relative_single_action_squared_error"
                if args.planner == "posterior_distortion"
                else (
                    "frequency_tempered_conservative_routed_importance_times_"
                    "conservative_conditional_svd_degradation_with_removal_"
                    + (
                        "equal_to_max_importance_and_rerouting_aware_error"
                        if not args.disable_rerouting_removal_guard
                        else "equal_to_conservative_importance"
                    )
                )
                if args.planner == "factorized_risk"
                else (
                    "frequency_tempered_linear_routing_energy_importance_times_"
                    "g2_weighted_conditional_svd_degradation_times_calibrated_"
                    "downstream_propagation_damage_with_rerouting_removal_guard"
                )
                if args.planner == "propagated_risk"
                else (
                    "mean_over_layers(sum_token_routed_output_squared_error / "
                    "sum_token_reference_routed_output_energy)"
                    if args.token_reduction == "energy"
                    and args.layer_reduction == "mean"
                    else "configured_token_and_layer_reduction"
                )
            ),
            "token_reduction": args.token_reduction,
            "token_energy_floor_ratio": float(args.token_energy_floor_ratio),
            "layer_reduction": args.layer_reduction,
            "min_retained_experts": int(args.min_retained_experts),
            "min_layer_memory_ratio": float(args.min_layer_memory_ratio),
            "layer_min_memory": (
                json.loads(Path(args.layer_min_memory_json).read_text())
                if args.layer_min_memory_json else None
            ),
            "max_layer_relative_distortion": args.max_layer_relative_distortion,
            "rerouting_conditioned_refinement": refinement_metadata,
            "rerouting_refinement_rounds": int(
                args.rerouting_refinement_rounds
            ),
            "rerouting_swaps_per_round": int(args.rerouting_swaps_per_round),
            "rerouting_candidates_per_boundary": int(
                args.rerouting_candidates_per_boundary
            ),
            "rerouting_chunk_size": int(args.rerouting_chunk_size),
            "removal_tiebreak": (
                "none"
                if args.planner
                in {
                    "importance_compressibility",
                    "posterior_distortion",
                    "factorized_risk",
                    "propagated_risk",
                }
                else args.removal_tiebreak
            ),
            "requested_removal_tiebreak": args.removal_tiebreak,
            "removal_tiebreak_rtol": float(args.removal_tiebreak_rtol),
            "usage_statistics": {
                "topk_count": "number of calibration tokens selecting the expert",
                "shortlist_count": "number of tokens ranking expert in router top-L",
                "standby_count": "top-L appearances outside top-k",
                "routing_mass": "sum of normalized active top-k router weights",
                "routing_square_mass": "sum of squared active top-k router weights",
                "reap_sum": "sum_active(g * ||f_e(h)||_2)",
                "reap_mean": "mean_active(g * ||f_e(h)||_2)",
            },
            "cvar_fraction": float(args.cvar_fraction),
            "routing_reference": "exact top-k within routing-kept pool",
            "rerouting_scope": (
                "exact current retained pool over all original experts"
                if args.rerouting_refinement_rounds
                else "exact top-k within each token's recorded router shortlist"
            ),
            "factor_artifact_dir": args.factors_out_dir,
        }
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Saved routed functional compression plan -> %s", args.out)
    if args.factors_out_dir:
        if model is None:
            raise RuntimeError(
                "--factors-out-dir requires the model to be loaded; enable "
                "rerouting refinement or omit --reuse-traces"
            )
        if down_moments is None:
            down_moments = _select_projection_moments(
                artifact, args.basis_category, "down"
            )
        _save_selected_factor_artifact(
            output_dir=args.factors_out_dir,
            plan_path=args.out,
            plan=plan,
            traces=traces,
            global_ids=global_ids,
            modules=modules,
            hidden_metadata=hidden_metadata,
            row_ids=row_ids,
            gate_moments=gate_moments,
            up_moments=up_moments,
            down_moments=down_moments,
            rank_pairs=rank_pairs,
            router_candidates=args.router_candidates,
            routing_batch_size=args.routing_batch_size,
            scale_floor=args.scale_floor,
            randomized=not args.exact_svd,
            seed=args.seed,
            gram_metric=args.factor_metric == "gram",
            gram_shrink_mode=args.gram_shrink_target,
            down_compensation=args.down_compensation,
            factor_calibration=args.factor_calibration,
            factor_output_metric=args.factor_output_metric,
        )
    logging.info(
        "Planned ratio %.4f, objective %.6g, category objective %s",
        plan.memory / full_memory,
        plan.objective,
        plan.category_relative_distortion,
    )


if __name__ == "__main__":
    main()
