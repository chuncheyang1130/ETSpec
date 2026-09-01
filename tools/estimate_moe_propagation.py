#!/usr/bin/env python
"""Calibrate downstream propagation damage for every MoE layer.

For a small prompt sample this tool perturbs one MoE layer's output by a known
relative energy and measures how much damage reaches the model's prediction,
divided by the relative damage injected locally. The resulting per-layer gain
tells the planner how much a unit of local compression error at that depth
actually costs.

Three properties of the probe decide whether that gain is meaningful:

*Metric.* ``logit_kl`` measures ``KL(p_full || p_perturbed)`` on the language
model head, which is the quantity model quality and speculative-decoding
acceptance actually depend on. The legacy ``hidden_energy`` metric measures
relative final-hidden squared error instead; the final RMSNorm divides out
magnitude, so it undercounts an early perturbation that the residual stream
partly absorbs and overcounts a late one that lands on the stream untouched.

*Direction.* ``random`` perturbs along an energy-matched isotropic direction,
which is what an SVD residual or a removal delta looks like. The legacy
``attenuate`` mode scales the output toward zero, a perturbation parallel to
the output and the gentlest one available: the residual stream absorbs a pure
gain change gracefully, while a rotation propagates and flips downstream
routers.

*Scale.* A single small perturbation only measures the linear regime. Damage
grows as ``s**gamma``; a linear response gives ``gamma = 2``, and because the
injected local damage is exactly ``s**2`` the gain scales as ``s**(gamma - 2)``.
A layer whose error compounds through downstream nonlinearities has
``gamma > 2`` and its gain is therefore *understated* by a small-``s`` probe.
Sweeping several scales recovers ``(gamma, coefficient)`` per layer so the gain
can be evaluated at the perturbation magnitude a plan actually inflicts --
separately for low-rank actions and for removal, which are orders of magnitude
apart.

The tool also records, per layer and scale, the fraction of downstream MoE
router top-k decisions that the perturbation changes. That count falls with
depth, opposite to the measured energy gain, and is the channel a pure energy
metric cannot see.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from specdecodes.helpers.recipes.moe.expert_calibration import load_selection
from specdecodes.models.utils.moe.base.expert_usage_tracker import _is_qwen3_moe_block
from tools.analyze_activation_svd import _load_moments


logging.basicConfig(level=logging.INFO, format="%(message)s")


def _replace_first_tensor(output, value: torch.Tensor):
    if torch.is_tensor(output):
        return value
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (value, *output[1:])
    if isinstance(output, list) and output and torch.is_tensor(output[0]):
        return [value, *output[1:]]
    raise TypeError("MoE propagation probe expected a tensor output")


def _first_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError("model output does not expose a final hidden-state tensor")


def _normalize_propagation_gains(
    raw: torch.Tensor,
    *,
    clip_min: float,
    clip_max: float,
    epsilon: float = 1e-30,
) -> torch.Tensor:
    """Median-normalize positive ``[layers, categories]`` gains."""

    values = torch.as_tensor(raw, dtype=torch.float64)
    if values.ndim != 2 or not int(values.numel()):
        raise ValueError("raw propagation gains must have shape [layers, categories]")
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("raw propagation gains must be finite and positive")
    if not (0 < float(clip_min) <= float(clip_max)):
        raise ValueError("propagation clipping bounds must satisfy 0 < min <= max")
    center = torch.quantile(values.clamp_min(epsilon).log(), 0.5, dim=0).exp()
    return (values / center.unsqueeze(0)).clamp(
        min=float(clip_min), max=float(clip_max)
    )


# Damage that survives noise-floor subtraction by less than this fraction of the
# floor is recorded at this fraction rather than at zero: the probe resolved it
# no better than that, and pretending otherwise would let one unresolved sample
# dominate a geometric mean.
_RESOLUTION_FRACTION = 0.05


def _fit_scale_exponent(
    scales: Sequence[float], damages: Sequence[float], *, epsilon: float = 1e-30
) -> tuple[float, float]:
    """Least-squares fit of ``log damage = gamma * log s + log coefficient``.

    Returns ``(gamma, coefficient)``. A single scale cannot identify a slope, so
    it falls back to the linear-response exponent ``gamma = 2``, which makes the
    gain scale-free and reproduces the legacy single-scale behavior exactly.
    """

    x = torch.as_tensor(list(scales), dtype=torch.float64).log()
    y = torch.as_tensor(list(damages), dtype=torch.float64).clamp_min(epsilon).log()
    if int(x.numel()) != int(y.numel()) or not int(x.numel()):
        raise ValueError("scale sweep requires one damage per scale")
    if int(x.numel()) == 1:
        gamma = 2.0
        return gamma, float((y[0] - gamma * x[0]).exp().item())
    centered_x = x - x.mean()
    denominator = float(centered_x.square().sum().item())
    if denominator <= 0:
        raise ValueError("scale sweep requires at least two distinct scales")
    gamma = float((centered_x * (y - y.mean())).sum().item() / denominator)
    coefficient = float((y.mean() - gamma * x.mean()).exp().item())
    return gamma, coefficient


def _gain_at_scale(gamma: float, coefficient: float, scale: float) -> float:
    """Evaluate ``damage(s) / s**2``, the gain per unit relative local damage."""

    if float(scale) <= 0:
        raise ValueError("evaluation scale must be positive")
    return float(coefficient) * float(scale) ** (float(gamma) - 2.0)


def _selected_sequences(
    sequences: dict[str, Sequence[torch.Tensor]],
    *,
    prompts_per_category: int,
    max_tokens: int,
    seed: int,
) -> tuple[tuple[str, ...], list[tuple[int, str, torch.Tensor]]]:
    categories = tuple(str(name) for name in sequences)
    selected = []
    for category_id, category in enumerate(categories):
        candidates = list(sequences[category])
        if not candidates:
            raise ValueError(f"category {category!r} contains no sequences")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + category_id * 1_000_003)
        order = torch.randperm(len(candidates), generator=generator)
        for index in order[: min(int(prompts_per_category), len(candidates))]:
            input_ids = torch.as_tensor(candidates[int(index)]).long()
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
                raise ValueError("cached calibration sequences must have shape [1, tokens]")
            # Match next-token replay and cap compute without padding.
            replay = input_ids[:, :-1]
            if int(replay.shape[1]) > int(max_tokens):
                replay = replay[:, : int(max_tokens)]
            if int(replay.numel()):
                selected.append((category_id, category, replay.contiguous()))
    return categories, selected


def _noise_seed(seed: int, layer_index: int, scale: float) -> int:
    """Derive a per-layer, per-scale noise seed.

    Seeding inside each hook rather than once per forward pass makes a layer's
    perturbation depend only on that layer and scale. A marginal measurement can
    then subtract two forward passes that differ solely in the layer under test,
    instead of also differing in every other layer's random draw.
    """

    return (
        int(seed)
        + int(layer_index) * 7919
        + int(round(float(scale) * 1_000_000)) * 104_729
    ) % (2**31 - 1)


def _perturb(
    value: torch.Tensor, scale: float, mode: str, generator: torch.Generator | None
) -> torch.Tensor:
    """Return an output perturbed by exactly ``scale**2`` relative energy.

    ``attenuate`` scales every token toward zero, so the perturbation is
    parallel to the output. ``random`` draws an isotropic direction per token and
    renormalizes it to that token's norm, which both matches the relative energy
    exactly and, in high dimension, is close to orthogonal to the output -- the
    geometry an SVD residual or a removal delta actually has.
    """

    if mode == "attenuate":
        return value * (1.0 - float(scale))
    if mode != "random":
        raise ValueError("perturbation mode must be attenuate or random")
    flat = value.reshape(-1, value.shape[-1])
    noise = torch.randn(
        flat.shape, dtype=torch.float32, device=flat.device, generator=generator
    )
    noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
    delta = float(scale) * torch.linalg.vector_norm(
        flat.float(), dim=-1, keepdim=True
    ) * noise
    return (flat.float() + delta).reshape(value.shape).to(value.dtype)


@torch.inference_mode()
def _forward(
    base_model,
    lm_head,
    input_ids: torch.Tensor,
    *,
    positions: torch.Tensor,
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(final_hidden, measurement)`` for one forward pass.

    ``measurement`` is per-position log-probabilities under ``logit_kl`` and the
    sampled final hidden states under ``hidden_energy``. Restricting the head to
    ``positions`` keeps the vocabulary-sized tensors small enough to hold both a
    baseline and a perturbed copy.
    """

    output = base_model(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden = (
        output.last_hidden_state
        if hasattr(output, "last_hidden_state")
        else _first_tensor(output)
    )
    sampled = hidden.reshape(-1, hidden.shape[-1]).index_select(0, positions)
    if metric == "hidden_energy":
        return hidden, sampled.float()
    if metric != "logit_kl":
        raise ValueError("gain metric must be logit_kl or hidden_energy")
    return hidden, F.log_softmax(lm_head(sampled).float(), dim=-1)


def _relative_damage(
    baseline_hidden: torch.Tensor,
    baseline_measurement: torch.Tensor,
    hidden: torch.Tensor,
    measurement: torch.Tensor,
    *,
    metric: str,
) -> float:
    """Return the damage reaching the model prediction.

    ``logit_kl`` returns the mean per-position ``KL(p_full || p_perturbed)``,
    which is dimensionless but not an energy ratio, so a gain measured with it
    turns the planner's relative-damage risk into an estimate of total KL.
    ``hidden_energy`` returns relative final-hidden squared error.
    """

    if metric == "hidden_energy":
        return float(
            (
                (measurement - baseline_measurement).square().sum()
                / baseline_measurement.square().sum().clamp_min(1e-30)
            ).item()
        )
    divergence = (
        baseline_measurement.exp() * (baseline_measurement - measurement)
    ).sum(dim=-1)
    return float(divergence.clamp_min(0).mean().item())


class _RouterRecorder:
    """Capture each MoE layer's router top-k decision during a forward pass."""

    def __init__(self, modules, names: Sequence[str]) -> None:
        self._modules = modules
        self._names = list(names)
        self._handles = []
        self.selection: dict[str, torch.Tensor] = {}

    def _hook(self, name: str):
        def record(module, inputs, _output):
            hidden = inputs[0]
            flat = hidden.reshape(-1, hidden.shape[-1])
            logits = F.linear(flat, module.router_weights.to(flat.dtype))
            top_k = min(int(module.top_k), int(logits.shape[1]))
            self.selection[name] = (
                torch.topk(logits.float(), k=top_k, dim=-1).indices.sort(dim=-1).values
            )

        return record

    def __enter__(self) -> "_RouterRecorder":
        for name in self._names:
            self._handles.append(
                self._modules[name].register_forward_hook(self._hook(name))
            )
        return self

    def __exit__(self, *_exception) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def flip_rate(self, other: "_RouterRecorder", names: Sequence[str]) -> float:
        """Fraction of (layer, token) router decisions that differ from ``other``."""

        changed = 0
        total = 0
        for name in names:
            if name not in self.selection or name not in other.selection:
                continue
            mine = self.selection[name]
            theirs = other.selection[name]
            if mine.shape != theirs.shape:
                continue
            changed += int((mine != theirs).any(dim=-1).sum().item())
            total += int(mine.shape[0])
        return float(changed) / float(max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--sequence-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--routing-kept-path", default=None)
    parser.add_argument("--prompts-per-category", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--gain-metric",
        choices=["logit_kl", "hidden_energy"],
        default="logit_kl",
        help=(
            "measure damage as head KL divergence, which is what quality and "
            "speculative acceptance depend on, or as legacy relative "
            "final-hidden squared error"
        ),
    )
    parser.add_argument(
        "--perturbation",
        choices=["random", "attenuate"],
        default="random",
        help=(
            "energy-matched isotropic direction, matching the geometry of an SVD "
            "residual or removal delta, or the legacy output-parallel attenuation"
        ),
    )
    parser.add_argument(
        "--perturbation-scales",
        nargs="+",
        type=float,
        default=[0.05, 0.2, 0.5],
        help=(
            "relative perturbation magnitudes to sweep; at least two distinct "
            "scales are needed to identify the damage growth exponent"
        ),
    )
    parser.add_argument(
        "--svd-scale",
        type=float,
        default=0.1,
        help="scale at which the low-rank action gain is evaluated",
    )
    parser.add_argument(
        "--removal-scale",
        type=float,
        default=0.5,
        help=(
            "scale at which the removal gain is evaluated; removal is a far "
            "larger perturbation than a low-rank approximation"
        ),
    )
    parser.add_argument(
        "--joint-baseline-scale",
        type=float,
        default=None,
        help=(
            "if set, perturb every MoE layer at this scale and measure each "
            "layer's marginal damage on top of it, which is the correct "
            "linearization point for a plan that compresses all layers"
        ),
    )
    parser.add_argument(
        "--noise-floor-scale",
        type=float,
        default=1.0 / 256.0,
        help=(
            "reference perturbation whose own downstream damage is negligible, "
            "measured per layer and subtracted from every sweep scale. A "
            "48-layer bfloat16 forward carries an additive head-KL floor of "
            "order 1e-3 that does not shrink with the perturbation, so without "
            "this subtraction the small-scale end of the sweep measures the "
            "floor instead of the layer and the fitted growth exponent "
            "collapses toward zero. Set to zero to disable"
        ),
    )
    parser.add_argument(
        "--kl-positions",
        type=int,
        default=128,
        help="token positions sampled per prompt for the head measurement",
    )
    parser.add_argument("--uncertainty-z", type=float, default=1.0)
    parser.add_argument("--factor-clip-min", type=float, default=0.25)
    parser.add_argument("--factor-clip-max", type=float, default=4.0)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.prompts_per_category, args.max_tokens, args.kl_positions) <= 0:
        parser.error("prompt, token, and position counts must be positive")
    scales = sorted({float(value) for value in args.perturbation_scales})
    if not scales or min(scales) <= 0 or max(scales) >= 1:
        parser.error("every --perturbation-scales entry must be in (0, 1)")
    if len(scales) == 1:
        logging.warning(
            "A single perturbation scale cannot identify the damage growth "
            "exponent; the gain falls back to the linear-response assumption "
            "gamma=2 and is therefore scale-free"
        )
    if min(args.svd_scale, args.removal_scale) <= 0:
        parser.error("--svd-scale and --removal-scale must be positive")
    if args.joint_baseline_scale is not None and not (
        0 < args.joint_baseline_scale < 1
    ):
        parser.error("--joint-baseline-scale must be in (0, 1)")
    if args.noise_floor_scale < 0:
        parser.error("--noise-floor-scale must be nonnegative")
    if args.noise_floor_scale >= min(scales):
        parser.error(
            "--noise-floor-scale must be smaller than every sweep scale, so "
            "that the reference carries floor rather than signal"
        )
    if args.uncertainty_z < 0:
        parser.error("--uncertainty-z must be nonnegative")
    if not (0 < args.factor_clip_min <= args.factor_clip_max):
        parser.error("factor clipping bounds must satisfy 0 < min <= max")

    sequence_artifact = _load_moments(args.sequence_cache)
    categories, selected = _selected_sequences(
        sequence_artifact["sequences"],
        prompts_per_category=args.prompts_per_category,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("no usable calibration sequences")

    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
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
    if args.routing_kept_path is not None:
        routing_kept, _ = load_selection(args.routing_kept_path)
        for name, ids in routing_kept.items():
            if name in modules:
                modules[name].set_kept(torch.as_tensor(ids, dtype=torch.long))
    module_names = [
        name for name, module in modules.items() if _is_qwen3_moe_block(module)
    ]
    if args.max_layers is not None:
        module_names = module_names[: int(args.max_layers)]
    if not module_names:
        raise RuntimeError("the model contains no supported stacked MoE layers")
    downstream = {
        name: module_names[index + 1 :] for index, name in enumerate(module_names)
    }

    base_model = getattr(model, "model", model)
    lm_head = getattr(model, "lm_head", None)
    if args.gain_metric == "logit_kl" and lm_head is None:
        raise RuntimeError("--gain-metric logit_kl requires the model to expose lm_head")

    layers, categories_count = len(module_names), len(categories)
    gain_samples = {
        name: [[[] for _ in categories] for _ in scales] for name in module_names
    }
    local_samples = {
        name: [[[] for _ in categories] for _ in scales] for name in module_names
    }
    flip_samples = {
        name: [[[] for _ in categories] for _ in scales] for name in module_names
    }
    generator = torch.Generator(device=args.device)

    def perturbation_hook(scale: float, layer_index: int):
        record: dict[str, float] = {}

        def inject(_module, _inputs, output):
            value = _first_tensor(output)
            generator.manual_seed(_noise_seed(args.seed, layer_index, scale))
            perturbed = _perturb(value, scale, args.perturbation, generator)
            record["reference_energy"] = value.float().square().sum().item()
            record["delta_energy"] = (
                perturbed.float() - value.float()
            ).square().sum().item()
            return _replace_first_tensor(output, perturbed)

        return inject, record

    def measure(input_ids, positions, perturbations):
        """Run one forward pass with the given ``{layer index: scale}`` map."""

        handles = []
        records = {}
        for layer_index, scale in perturbations.items():
            hook, record = perturbation_hook(scale, layer_index)
            records[layer_index] = record
            handles.append(
                modules[module_names[layer_index]].register_forward_hook(hook)
            )
        try:
            with _RouterRecorder(modules, module_names) as routes:
                hidden, measurement = _forward(
                    base_model,
                    lm_head,
                    input_ids,
                    positions=positions,
                    metric=args.gain_metric,
                )
        finally:
            for handle in handles:
                handle.remove()
        return hidden, measurement, routes, records

    for category_id, category, cached_ids in tqdm(
        selected, desc="Propagation prompts"
    ):
        input_ids = cached_ids.to(args.device)
        tokens = int(input_ids.numel())
        position_generator = torch.Generator(device="cpu")
        position_generator.manual_seed(int(args.seed) + tokens)
        take = min(int(args.kl_positions), tokens)
        positions = (
            torch.randperm(tokens, generator=position_generator)[:take]
            .sort()
            .values.to(args.device)
        )
        baseline_hidden, baseline_measurement, baseline_routes, _ = measure(
            input_ids, positions, {}
        )

        joint = args.joint_baseline_scale
        floor_scale = float(args.noise_floor_scale)
        for layer_index, name in enumerate(module_names):
            # Without a joint baseline the reference is the exact model. With
            # one, it is every *other* layer already carrying the plan-level
            # perturbation while this layer stays exact, so the difference below
            # isolates this layer's marginal contribution and stays nonnegative
            # whatever sweep scale is used.
            reference_damage = 0.0
            reference_routes = baseline_routes
            others = (
                {}
                if joint is None
                else {
                    other: joint
                    for other in range(len(module_names))
                    if other != layer_index
                }
            )
            # The reference holds every other layer in whatever state the joint
            # baseline puts it in, and this layer at a perturbation small enough
            # that its own downstream damage is negligible. Subtracting it
            # removes the joint baseline's damage and, more importantly, the
            # additive bfloat16 noise floor of a 48-layer forward: that floor is
            # the same size whatever the perturbation, so it swamps the signal
            # at exactly the small scales a low-rank action operates at, and
            # flattens the fitted growth exponent toward zero.
            if joint is not None or floor_scale > 0:
                probe = dict(others)
                if floor_scale > 0:
                    probe[layer_index] = floor_scale
                reference_hidden, reference_measurement, reference_routes, _ = measure(
                    input_ids, positions, probe
                )
                reference_damage = _relative_damage(
                    baseline_hidden,
                    baseline_measurement,
                    reference_hidden,
                    reference_measurement,
                    metric=args.gain_metric,
                )
                del reference_hidden, reference_measurement

            for scale_id, scale in enumerate(scales):
                hidden, measurement, routes, records = measure(
                    input_ids, positions, {**others, layer_index: scale}
                )
                damage = _relative_damage(
                    baseline_hidden,
                    baseline_measurement,
                    hidden,
                    measurement,
                    metric=args.gain_metric,
                )
                record = records[layer_index]
                local = record["delta_energy"] / max(record["reference_energy"], 1e-30)
                # The reference is one noisy realization of the floor, so a
                # layer whose true damage sits below the floor subtracts to
                # something near zero or negative. Flooring that at zero would
                # send it to 1e-30 and let a single unresolved sample dominate
                # the log-space average; flooring it at a small fraction of the
                # reference instead records it as "below the resolution of this
                # probe", which is what was actually observed.
                resolved = damage - reference_damage
                gain_samples[name][scale_id][category_id].append(
                    max(resolved, _RESOLUTION_FRACTION * reference_damage)
                    / max(local, 1e-30)
                )
                local_samples[name][scale_id][category_id].append(local)
                flip_samples[name][scale_id][category_id].append(
                    routes.flip_rate(reference_routes, downstream[name])
                )
                del hidden, measurement
        del baseline_hidden, baseline_measurement

    # Aggregate every (layer, scale, category) in log space, then fit the damage
    # growth exponent so the gain can be evaluated where a plan operates.
    observed = torch.zeros(layers, len(scales), categories_count, dtype=torch.float64)
    conservative = torch.zeros_like(observed)
    log_standard_error = torch.zeros_like(observed)
    flip_gain = torch.zeros_like(observed)
    local_damage = torch.zeros_like(observed)
    counts = torch.zeros_like(observed, dtype=torch.long)
    for layer_id, name in enumerate(module_names):
        for scale_id in range(len(scales)):
            for category_id in range(categories_count):
                values = torch.tensor(
                    gain_samples[name][scale_id][category_id], dtype=torch.float64
                )
                counts[layer_id, scale_id, category_id] = int(values.numel())
                if not int(values.numel()):
                    raise RuntimeError(f"{name}: no propagation samples collected")
                logs = values.clamp_min(1e-30).log()
                observed[layer_id, scale_id, category_id] = logs.mean().exp()
                if int(values.numel()) > 1:
                    log_standard_error[layer_id, scale_id, category_id] = (
                        logs.std(unbiased=True) / float(values.numel()) ** 0.5
                    )
                conservative[layer_id, scale_id, category_id] = (
                    logs.mean()
                    + float(args.uncertainty_z)
                    * log_standard_error[layer_id, scale_id, category_id]
                ).exp()
                flips = torch.tensor(
                    flip_samples[name][scale_id][category_id], dtype=torch.float64
                )
                locals_ = torch.tensor(
                    local_samples[name][scale_id][category_id], dtype=torch.float64
                )
                local_damage[layer_id, scale_id, category_id] = locals_.mean()
                flip_gain[layer_id, scale_id, category_id] = flips.mean() / max(
                    float(locals_.mean().item()), 1e-30
                )

    exponent = torch.zeros(layers, categories_count, dtype=torch.float64)
    coefficient = torch.zeros_like(exponent)
    svd_raw = torch.zeros_like(exponent)
    removal_raw = torch.zeros_like(exponent)
    for layer_id, name in enumerate(module_names):
        for category_id in range(categories_count):
            # The conservative gain at scale s equals damage(s)/s**2, so the fit
            # is performed on the implied damage curve.
            damages = [
                float(conservative[layer_id, scale_id, category_id].item())
                * float(scale) ** 2
                for scale_id, scale in enumerate(scales)
            ]
            gamma, factor = _fit_scale_exponent(scales, damages)
            exponent[layer_id, category_id] = gamma
            coefficient[layer_id, category_id] = factor
            svd_raw[layer_id, category_id] = max(
                _gain_at_scale(gamma, factor, args.svd_scale), 1e-30
            )
            removal_raw[layer_id, category_id] = max(
                _gain_at_scale(gamma, factor, args.removal_scale), 1e-30
            )

    svd_factors = _normalize_propagation_gains(
        svd_raw, clip_min=args.factor_clip_min, clip_max=args.factor_clip_max
    )
    removal_factors = _normalize_propagation_gains(
        removal_raw, clip_min=args.factor_clip_min, clip_max=args.factor_clip_max
    )
    for label, raw, factors in (
        ("svd", svd_raw, svd_factors),
        ("remove", removal_raw, removal_factors),
    ):
        clipped = int(
            (
                (factors <= float(args.factor_clip_min))
                | (factors >= float(args.factor_clip_max))
            )
            .sum()
            .item()
        )
        logging.info(
            "%s gain spread before clipping: %.4g..%.4g (%d of %d entries at a "
            "clip bound)",
            label,
            float(raw.min().item()),
            float(raw.max().item()),
            clipped,
            int(factors.numel()),
        )
    logging.info(
        "Damage growth exponent across layers: %.3f..%.3f (median %.3f); a value "
        "above 2 means a small-scale probe understates that layer's gain",
        float(exponent.min().item()),
        float(exponent.max().item()),
        float(exponent.median().item()),
    )

    def by_category(values: torch.Tensor, layer_id: int) -> dict[str, float]:
        return {
            category: float(values[layer_id, category_id].item())
            for category_id, category in enumerate(categories)
        }

    layers_payload = {}
    for layer_id, name in enumerate(module_names):
        layers_payload[name] = {
            # The planner reads these three; "factors" keeps older loaders that
            # predate the separate low-rank and removal gains working.
            "factors": by_category(svd_factors, layer_id),
            "svd_factors": by_category(svd_factors, layer_id),
            "remove_factors": by_category(removal_factors, layer_id),
            "scale_curve": {
                category: {
                    "exponent": float(exponent[layer_id, category_id].item()),
                    "coefficient": float(coefficient[layer_id, category_id].item()),
                }
                for category_id, category in enumerate(categories)
            },
            "raw_svd_gain": by_category(svd_raw, layer_id),
            "raw_remove_gain": by_category(removal_raw, layer_id),
            "route_flip_gain": {
                category: float(flip_gain[layer_id, :, category_id].mean().item())
                for category_id, category in enumerate(categories)
            },
            "by_scale": {
                str(scale): {
                    "observed_gain": by_category(observed[:, scale_id, :], layer_id),
                    "conservative_gain": by_category(
                        conservative[:, scale_id, :], layer_id
                    ),
                    "log_standard_error": by_category(
                        log_standard_error[:, scale_id, :], layer_id
                    ),
                    "relative_local_damage": by_category(
                        local_damage[:, scale_id, :], layer_id
                    ),
                    "downstream_route_flip_rate": {
                        category: float(
                            flip_gain[layer_id, scale_id, category_id].item()
                        )
                        * float(local_damage[layer_id, scale_id, category_id].item())
                        for category_id, category in enumerate(categories)
                    },
                    "prompt_count": {
                        category: int(counts[layer_id, scale_id, category_id].item())
                        for category_id, category in enumerate(categories)
                    },
                }
                for scale_id, scale in enumerate(scales)
            },
        }

    payload = {
        "format": "moe_propagation_damage",
        "format_version": 2,
        "model": args.model,
        "sequence_cache": os.path.abspath(args.sequence_cache),
        "category_names": list(categories),
        "gain_metric": args.gain_metric,
        "probe": f"{args.perturbation}_moe_output_perturbation",
        "perturbation_scales": [float(value) for value in scales],
        "svd_scale": float(args.svd_scale),
        "removal_scale": float(args.removal_scale),
        "joint_baseline_scale": (
            None
            if args.joint_baseline_scale is None
            else float(args.joint_baseline_scale)
        ),
        "noise_floor_scale": float(args.noise_floor_scale),
        "gain_definition": (
            "head_kl/relative_local_moe_error"
            if args.gain_metric == "logit_kl"
            else "relative_final_hidden_error/relative_local_moe_error"
        ),
        "scale_curve_definition": "damage(s) = coefficient * s**exponent",
        "normalization": "category_median_across_layers",
        "kl_positions": int(args.kl_positions),
        "uncertainty_z": float(args.uncertainty_z),
        "factor_clip": [float(args.factor_clip_min), float(args.factor_clip_max)],
        "layers": layers_payload,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Saved MoE propagation calibration -> %s", args.out)


if __name__ == "__main__":
    main()
