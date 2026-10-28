"""Activation-conditioned SVD utilities for routed Qwen3-MoE experts.

The collector records the diagonal second moment of each selected expert's
SwiGLU intermediate activation, weighted by the square of its routing value::

    sum_t 1[e routed] * g[t, e]^2 * z[t, e]^2

where ``z = silu(W_gate h) * (W_up h)``.  Dividing each category's sum by its
tracked-token count and averaging categories yields a small, category-balanced
input metric for ``down_proj``.  No hidden states or full covariance matrices are
stored.

The analysis helper compares ordinary truncated SVD of ``W_down`` against SVD of
``W_down @ diag(sqrt(E[g^2 z^2]))`` under the same activation-weighted error.
It deliberately reports reconstruction quality only; a separate hybrid MoE block
is required to realize parameter or runtime savings from the factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .expert_usage_tracker import _compute_router_logits, _is_qwen3_moe_block
from .moe_arch import (
    activation_derivative, activation_fn, block_activation, compute_routing,
    describe_moe_block,
)


@dataclass
class ActivationSVDMetrics:
    """Activation-weighted relative reconstruction error at each requested rank."""

    weight_only: Dict[int, float]
    activation_scaled: Dict[int, float]
    weighted_energy: float
    active_dimensions: int


@dataclass
class SharedGateUpSVDMetrics:
    """Errors for a gate/up pair constrained to one input-side basis.

    ``joint_*`` uses the common input metric supplied to the joint
    decomposition.  The branch errors are evaluated with the original gate and
    up metrics, so they remain comparable to the independent-projection report.
    """

    joint_weight_only: Dict[int, float]
    joint_activation_scaled: Dict[int, float]
    gate_weight_only: Dict[int, float]
    gate_activation_scaled: Dict[int, float]
    up_weight_only: Dict[int, float]
    up_activation_scaled: Dict[int, float]
    joint_weighted_energy: float
    gate_weighted_energy: float
    up_weighted_energy: float
    active_dimensions: int


def _expert_gate_up(
    module: nn.Module, expert_id: int, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one expert's pre-activation gate and linear up branches."""
    gate_stacked = getattr(module, "gate_proj_stacked", None)
    up_stacked = getattr(module, "up_proj_stacked", None)
    if gate_stacked is not None and up_stacked is not None:
        return x @ gate_stacked[expert_id].t(), x @ up_stacked[expert_id].t()

    experts = getattr(module, "experts", None)
    if experts is None:
        raise TypeError(
            f"{module.__class__.__name__} exposes neither stacked gate/up weights nor experts"
        )
    expert = experts[expert_id]
    return expert.gate_proj(x), expert.up_proj(x)


def _expert_intermediate(module: nn.Module, expert_id: int, x: torch.Tensor) -> torch.Tensor:
    """Evaluate one expert's gated intermediate ``act(gate) * up`` without its down projection."""
    gate, up = _expert_gate_up(module, expert_id, x)
    return activation_fn(gate, block_activation(module)) * up


def expert_down_weight(module: nn.Module, expert_id: int) -> torch.Tensor:
    """Return ``W_down`` as ``[hidden, intermediate]`` for a global expert id."""
    down_stacked = getattr(module, "down_proj_stacked", None)
    if down_stacked is not None:
        return down_stacked[expert_id]
    experts = getattr(module, "experts", None)
    if experts is None:
        raise TypeError(
            f"{module.__class__.__name__} exposes neither stacked down weights nor experts"
        )
    return experts[expert_id].down_proj.weight


def expert_projection_weight(
    module: nn.Module, expert_id: int, projection: str
) -> torch.Tensor:
    """Return one projection as ``[output_features, input_features]``."""
    projection = str(projection)
    if projection == "down":
        return expert_down_weight(module, expert_id)
    if projection not in ("gate", "up"):
        raise ValueError(f"unknown expert projection {projection!r}")
    stacked = getattr(module, f"{projection}_proj_stacked", None)
    if stacked is not None:
        return stacked[expert_id]
    experts = getattr(module, "experts", None)
    if experts is None:
        raise TypeError(
            f"{module.__class__.__name__} exposes neither stacked weights nor experts"
        )
    return getattr(experts[expert_id], f"{projection}_proj").weight


class RoutedIntermediateMomentCollector:
    """Collect routed ``g^2 z^2`` moments for selected experts using pre-hooks.

    ``selected_per_layer`` is keyed by the model's MoE module paths and contains
    global expert ids. Accumulators live on the same device as each block while a
    category is replayed and are copied to CPU only when ``snapshot`` is called.
    """

    def __init__(
        self,
        model: nn.Module,
        selected_per_layer: Mapping[str, Sequence[int]],
        *,
        track_gate_up_inputs: bool = False,
    ) -> None:
        self.model = model
        self.selected_per_layer = {
            str(name): sorted({int(e) for e in ids})
            for name, ids in selected_per_layer.items()
        }
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, torch.Tensor] = {}
        self.mass: Dict[str, torch.Tensor] = {}
        self.importance: Dict[str, torch.Tensor] = {}
        self.track_gate_up_inputs = bool(track_gate_up_inputs)
        self.gate_input_sums: Dict[str, torch.Tensor] = {}
        self.up_input_sums: Dict[str, torch.Tensor] = {}
        self.down_column_energy: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        modules = dict(model.named_modules())
        missing = sorted(set(self.selected_per_layer) - set(modules))
        if missing:
            raise KeyError(f"SVD score layers not found in model: {missing[:4]}")

        for name, ids in self.selected_per_layer.items():
            module = modules[name]
            if not _is_qwen3_moe_block(module):
                raise TypeError(f"{name} is not a supported Qwen3 MoE block")
            E = int(module.num_experts)
            if any(e < 0 or e >= E for e in ids):
                raise ValueError(f"{name}: selected expert ids must be in [0, {E - 1}]")
            if not ids:
                continue
            intermediate = int(expert_down_weight(module, ids[0]).shape[1])
            device = expert_down_weight(module, ids[0]).device
            self.sums[name] = torch.zeros(E, intermediate, dtype=torch.float32, device=device)
            self.counts[name] = torch.zeros(E, dtype=torch.float32, device=device)
            self.mass[name] = torch.zeros(E, dtype=torch.float32, device=device)
            self.importance[name] = torch.zeros(E, dtype=torch.float32, device=device)
            if self.track_gate_up_inputs:
                hidden = int(expert_projection_weight(module, ids[0], "gate").shape[1])
                self.gate_input_sums[name] = torch.zeros(
                    E, hidden, dtype=torch.float32, device=device
                )
                self.up_input_sums[name] = torch.zeros(
                    E, hidden, dtype=torch.float32, device=device
                )
                self.down_column_energy[name] = torch.zeros(
                    E, intermediate, dtype=torch.float32, device=device
                )
                for expert_id in ids:
                    down = expert_down_weight(module, expert_id)
                    self.down_column_energy[name][expert_id].copy_(
                        down.float().square().sum(dim=0)
                    )
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name, ids)))

    def _make_hook(self, name: str, selected_ids: Sequence[int]):
        selected = set(int(e) for e in selected_ids)

        @torch.no_grad()
        def hook(module: nn.Module, inputs) -> None:
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            flat = hidden.reshape(-1, hidden.shape[-1]) if hidden.dim() == 3 else hidden
            routing = getattr(module, "_routing_weights", None)
            if callable(routing):
                # Stacked blocks may restrict routing to a retained pool. Use the
                # block's actual routing path so post-pruning calibration does not
                # silently measure top-k over all original router rows.
                topk_vals, topk_idx = routing(flat)
                topk_vals = topk_vals.float()
            else:
                topk_idx, topk_vals = compute_routing(module, flat)
                topk_vals = topk_vals.float()

            act = block_activation(module)
            active_selected = selected.intersection(int(e) for e in topk_idx.unique().tolist())
            for expert_id in active_selected:
                token_idx, slot_idx = (topk_idx == expert_id).nonzero(as_tuple=True)
                if token_idx.numel() == 0:
                    continue
                x_e = flat.index_select(0, token_idx)
                gate_e, up_e = _expert_gate_up(module, expert_id, x_e)
                gate_e = gate_e.float()
                up_e = up_e.float()
                gate_act = activation_fn(gate_e, act)
                z_e = gate_act * up_e
                g_e = topk_vals[token_idx, slot_idx].float().unsqueeze(1)
                self.sums[name][expert_id].add_((g_e.square() * z_e.square()).sum(dim=0))
                if self.track_gate_up_inputs:
                    column_energy = self.down_column_energy[name][expert_id]
                    act_prime = activation_derivative(gate_e, act)
                    up_sensitivity = (
                        gate_act.square() * column_energy.unsqueeze(0)
                    ).sum(dim=1, keepdim=True)
                    gate_sensitivity = (
                        up_e.square()
                        * act_prime.square()
                        * column_energy.unsqueeze(0)
                    ).sum(dim=1, keepdim=True)
                    weighted_hidden = g_e.square() * x_e.float().square()
                    self.up_input_sums[name][expert_id].add_(
                        (weighted_hidden * up_sensitivity).sum(dim=0)
                    )
                    self.gate_input_sums[name][expert_id].add_(
                        (weighted_hidden * gate_sensitivity).sum(dim=0)
                    )
                self.counts[name][expert_id].add_(float(token_idx.numel()))
                self.mass[name][expert_id].add_(g_e.sum())
                down = expert_down_weight(module, expert_id)
                output_norm = (z_e.to(down.dtype) @ down.t()).float().norm(dim=-1)
                self.importance[name][expert_id].add_(
                    (g_e.squeeze(1) * output_norm).sum()
                )

        return hook

    @torch.no_grad()
    def reset(self) -> None:
        for tensor in self.sums.values():
            tensor.zero_()
        for tensor in self.counts.values():
            tensor.zero_()
        for tensor in self.mass.values():
            tensor.zero_()
        for tensor in self.importance.values():
            tensor.zero_()
        for tensor in self.gate_input_sums.values():
            tensor.zero_()
        for tensor in self.up_input_sums.values():
            tensor.zero_()

    @torch.no_grad()
    def snapshot(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "sum_g2_z2": {name: value.detach().cpu().clone() for name, value in self.sums.items()},
            "routing_count": {name: value.detach().cpu().clone() for name, value in self.counts.items()},
            "routing_mass": {name: value.detach().cpu().clone() for name, value in self.mass.items()},
            "importance": {name: value.detach().cpu().clone() for name, value in self.importance.items()},
            "sum_gate_input_metric": {
                name: value.detach().cpu().clone()
                for name, value in self.gate_input_sums.items()
            },
            "sum_up_input_metric": {
                name: value.detach().cpu().clone()
                for name, value in self.up_input_sums.items()
            },
        }

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.remove()


def balance_category_moments(
    category_snapshots: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    tracked_tokens: Mapping[str, int],
    signal: str = "sum_g2_z2",
) -> Dict[str, torch.Tensor]:
    """Average one category per-token moment signal with equal category weight."""
    categories = list(category_snapshots)
    if not categories:
        raise ValueError("no category snapshots were provided")
    output: Dict[str, torch.Tensor] = {}
    for category in categories:
        tokens = int(tracked_tokens.get(category, 0))
        if tokens <= 0:
            raise ValueError(f"category {category!r} has no tracked tokens")
        sums = category_snapshots[category].get(signal)
        if sums is None:
            raise KeyError(f"category {category!r} has no moment signal {signal!r}")
        for name, value in sums.items():
            contribution = value.float() / float(tokens)
            output[name] = output.get(name, torch.zeros_like(contribution)) + contribution
    scale = 1.0 / float(len(categories))
    return {name: value.mul(scale) for name, value in output.items()}


def balance_category_signal(
    category_snapshots: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    tracked_tokens: Mapping[str, int],
    signal: str = "importance",
    normalization: str = "per_token",
) -> tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """Normalize a signal within category, then average categories equally.

    ``normalization="per_token"`` divides by every tracked token in the category,
    retaining expert activation frequency. ``normalization="conditional"`` divides
    by that expert's routed-token count, matching REAP's frequency-agnostic mean over
    tokens on which the expert is active. Experts with no routed tokens receive zero.

    Returns ``(balanced, per_category)``. The second mapping is retained so callers
    can report each category's expert ranking instead of only the final aggregate.
    """
    if normalization not in ("per_token", "conditional"):
        raise ValueError(
            f"unknown signal normalization {normalization!r}; "
            "expected 'per_token' or 'conditional'"
        )
    categories = list(category_snapshots)
    if not categories:
        raise ValueError("no category snapshots were provided")
    per_category: Dict[str, Dict[str, torch.Tensor]] = {}
    balanced: Dict[str, torch.Tensor] = {}
    for category in categories:
        tokens = int(tracked_tokens.get(category, 0))
        if tokens <= 0:
            raise ValueError(f"category {category!r} has no tracked tokens")
        values = category_snapshots[category].get(signal)
        if values is None:
            raise KeyError(f"category {category!r} has no signal {signal!r}")
        if normalization == "per_token":
            normalized = {
                name: value.float() / float(tokens) for name, value in values.items()
            }
        else:
            counts = category_snapshots[category].get("routing_count")
            if counts is None:
                raise KeyError(
                    f"category {category!r} has no signal 'routing_count'"
                )
            normalized = {}
            for name, value in values.items():
                if name not in counts:
                    raise KeyError(
                        f"category {category!r} routing_count has no layer {name!r}"
                    )
                denominator = counts[name].float()
                if denominator.shape != value.shape:
                    raise ValueError(
                        f"category {category!r} layer {name!r}: signal/count shape "
                        f"mismatch {tuple(value.shape)} != {tuple(denominator.shape)}"
                    )
                normalized[name] = torch.where(
                    denominator > 0,
                    value.float() / denominator.clamp_min(1),
                    torch.zeros_like(value, dtype=torch.float32),
                )
        per_category[category] = normalized
        for name, value in per_category[category].items():
            balanced[name] = balanced.get(name, torch.zeros_like(value)) + value
    scale = 1.0 / float(len(categories))
    return ({name: value.mul(scale) for name, value in balanced.items()}, per_category)


@torch.no_grad()
def compare_weight_and_activation_svd(
    weight: torch.Tensor,
    second_moment: torch.Tensor,
    ranks: Iterable[int],
    *,
    device: torch.device | str = "cpu",
    randomized: bool = True,
    oversample: int = 8,
    niter: int = 2,
) -> ActivationSVDMetrics:
    """Compare two SVD bases under activation-weighted down-projection error.

    The ordinary basis is fitted to ``weight``. The calibrated basis is fitted to
    ``weight * sqrt(second_moment)``. Both are evaluated as relative squared error
    under that same diagonal activation metric, making the comparison matched.
    """
    w = weight.detach().to(device=device, dtype=torch.float32)
    moment = second_moment.detach().to(device=device, dtype=torch.float32).clamp_min(0)
    if w.dim() != 2 or moment.dim() != 1 or w.shape[1] != moment.numel():
        raise ValueError(
            f"expected weight [H, I] and moment [I], got {tuple(w.shape)} and {tuple(moment.shape)}"
        )
    requested = sorted({int(rank) for rank in ranks})
    if not requested or requested[0] <= 0:
        raise ValueError("ranks must contain positive integers")
    limit = min(w.shape)
    if requested[-1] > limit:
        raise ValueError(f"rank {requested[-1]} exceeds matrix rank limit {limit}")

    scale = moment.sqrt()
    weighted = w * scale.unsqueeze(0)
    total = float(weighted.square().sum().item())
    if total <= 0:
        zeros = {rank: 0.0 for rank in requested}
        return ActivationSVDMetrics(zeros, zeros.copy(), 0.0, int((scale > 0).sum().item()))

    max_rank = requested[-1]

    def decompose(matrix: torch.Tensor):
        if randomized and max_rank < min(matrix.shape):
            q = min(min(matrix.shape), max_rank + max(0, int(oversample)))
            u, s, v = torch.svd_lowrank(matrix, q=q, niter=int(niter))
            order = torch.argsort(s, descending=True)
            return u[:, order], s[order], v[:, order].t()
        return torch.linalg.svd(matrix, full_matrices=False)

    uw, sw, vhw = decompose(w)
    ua, sa, vha = decompose(weighted)
    weight_errors: Dict[int, float] = {}
    activation_errors: Dict[int, float] = {}
    for rank in requested:
        approx_w = (uw[:, :rank] * sw[:rank]) @ vhw[:rank]
        residual_w = (w - approx_w) * scale.unsqueeze(0)
        weight_errors[rank] = float(residual_w.square().sum().item() / total)

        approx_weighted = (ua[:, :rank] * sa[:rank]) @ vha[:rank]
        residual_a = weighted - approx_weighted
        activation_errors[rank] = float(residual_a.square().sum().item() / total)

    return ActivationSVDMetrics(
        weight_only=weight_errors,
        activation_scaled=activation_errors,
        weighted_energy=total,
        active_dimensions=int((scale > 0).sum().item()),
    )


@torch.no_grad()
def compare_shared_gate_up_svd(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    gate_second_moment: torch.Tensor,
    up_second_moment: torch.Tensor,
    ranks: Iterable[int],
    *,
    device: torch.device | str = "cpu",
    randomized: bool = True,
    oversample: int = 8,
    niter: int = 2,
) -> SharedGateUpSVDMetrics:
    """Compare weight-only and calibrated gate/up shared-input bases.

    Gate and up must reuse one input projection at inference, so their separate
    sensitivity metrics cannot be applied as two different right-hand scalings.
    We form one joint diagonal metric by adding their non-negative per-input
    sensitivities, concatenate the two weights along the output dimension, and
    decompose the jointly scaled matrix::

        cat([W_gate, W_up], dim=0) @ diag(sqrt(D_gate + D_up)).

    The resulting right factor is shared, while the left factor is sliced into
    separate gate and up reconstructions.  Joint errors use ``D_gate + D_up``;
    branch errors are additionally evaluated under their original metrics.
    """

    gate = gate_weight.detach().to(device=device, dtype=torch.float32)
    up = up_weight.detach().to(device=device, dtype=torch.float32)
    gate_moment = (
        gate_second_moment.detach().to(device=device, dtype=torch.float32).clamp_min(0)
    )
    up_moment = (
        up_second_moment.detach().to(device=device, dtype=torch.float32).clamp_min(0)
    )
    if gate.dim() != 2 or up.dim() != 2 or gate.shape != up.shape:
        raise ValueError(
            "gate/up weights must be equal-shaped [output, input] matrices, got "
            f"{tuple(gate.shape)} and {tuple(up.shape)}"
        )
    if (
        gate_moment.dim() != 1
        or up_moment.dim() != 1
        or gate_moment.numel() != gate.shape[1]
        or up_moment.numel() != up.shape[1]
    ):
        raise ValueError(
            "gate/up moments must match the input dimension, got "
            f"{tuple(gate_moment.shape)}, {tuple(up_moment.shape)}, "
            f"input={gate.shape[1]}"
        )

    requested = sorted({int(rank) for rank in ranks})
    if not requested or requested[0] <= 0:
        raise ValueError("ranks must contain positive integers")
    stacked = torch.cat((gate, up), dim=0)
    limit = min(stacked.shape)
    if requested[-1] > limit:
        raise ValueError(f"rank {requested[-1]} exceeds joint matrix rank limit {limit}")

    joint_moment = gate_moment + up_moment
    joint_scale = joint_moment.sqrt()
    weighted = stacked * joint_scale.unsqueeze(0)
    gate_energy = float((gate * gate_moment.sqrt().unsqueeze(0)).square().sum().item())
    up_energy = float((up * up_moment.sqrt().unsqueeze(0)).square().sum().item())
    joint_energy = float(weighted.square().sum().item())
    if joint_energy <= 0:
        zeros = {rank: 0.0 for rank in requested}
        return SharedGateUpSVDMetrics(
            joint_weight_only=zeros.copy(),
            joint_activation_scaled=zeros.copy(),
            gate_weight_only=zeros.copy(),
            gate_activation_scaled=zeros.copy(),
            up_weight_only=zeros.copy(),
            up_activation_scaled=zeros.copy(),
            joint_weighted_energy=0.0,
            gate_weighted_energy=gate_energy,
            up_weighted_energy=up_energy,
            active_dimensions=int((joint_scale > 0).sum().item()),
        )

    max_rank = requested[-1]

    def decompose(matrix: torch.Tensor):
        if randomized and max_rank < min(matrix.shape):
            q = min(min(matrix.shape), max_rank + max(0, int(oversample)))
            u, s, v = torch.svd_lowrank(matrix, q=q, niter=int(niter))
            order = torch.argsort(s, descending=True)
            return u[:, order], s[order], v[:, order].t()
        return torch.linalg.svd(matrix, full_matrices=False)

    uw, sw, vhw = decompose(stacked)
    ua, sa, vha = decompose(weighted)
    output_features = gate.shape[0]
    gate_scale = gate_moment.sqrt()
    up_scale = up_moment.sqrt()
    # Convert a residual in the common scaled coordinates to either branch's
    # original metric without explicitly dividing a factor by tiny scales.
    gate_from_joint = torch.where(
        joint_moment > 0,
        (gate_moment / joint_moment.clamp_min(torch.finfo(torch.float32).tiny)).sqrt(),
        torch.zeros_like(joint_moment),
    )
    up_from_joint = torch.where(
        joint_moment > 0,
        (up_moment / joint_moment.clamp_min(torch.finfo(torch.float32).tiny)).sqrt(),
        torch.zeros_like(joint_moment),
    )

    joint_weight_errors: Dict[int, float] = {}
    joint_activation_errors: Dict[int, float] = {}
    gate_weight_errors: Dict[int, float] = {}
    gate_activation_errors: Dict[int, float] = {}
    up_weight_errors: Dict[int, float] = {}
    up_activation_errors: Dict[int, float] = {}
    for rank in requested:
        approx_weight = (uw[:, :rank] * sw[:rank]) @ vhw[:rank]
        raw_residual = stacked - approx_weight
        weighted_residual = raw_residual * joint_scale.unsqueeze(0)
        joint_weight_errors[rank] = float(
            weighted_residual.square().sum().item() / joint_energy
        )
        gate_weight_errors[rank] = float(
            (raw_residual[:output_features] * gate_scale.unsqueeze(0))
            .square()
            .sum()
            .item()
            / max(gate_energy, 1e-30)
        )
        up_weight_errors[rank] = float(
            (raw_residual[output_features:] * up_scale.unsqueeze(0))
            .square()
            .sum()
            .item()
            / max(up_energy, 1e-30)
        )

        approx_weighted = (ua[:, :rank] * sa[:rank]) @ vha[:rank]
        calibrated_residual = weighted - approx_weighted
        joint_activation_errors[rank] = float(
            calibrated_residual.square().sum().item() / joint_energy
        )
        gate_activation_errors[rank] = float(
            (calibrated_residual[:output_features] * gate_from_joint.unsqueeze(0))
            .square()
            .sum()
            .item()
            / max(gate_energy, 1e-30)
        )
        up_activation_errors[rank] = float(
            (calibrated_residual[output_features:] * up_from_joint.unsqueeze(0))
            .square()
            .sum()
            .item()
            / max(up_energy, 1e-30)
        )

    return SharedGateUpSVDMetrics(
        joint_weight_only=joint_weight_errors,
        joint_activation_scaled=joint_activation_errors,
        gate_weight_only=gate_weight_errors,
        gate_activation_scaled=gate_activation_errors,
        up_weight_only=up_weight_errors,
        up_activation_scaled=up_activation_errors,
        joint_weighted_energy=joint_energy,
        gate_weighted_energy=gate_energy,
        up_weighted_energy=up_energy,
        active_dimensions=int((joint_scale > 0).sum().item()),
    )
