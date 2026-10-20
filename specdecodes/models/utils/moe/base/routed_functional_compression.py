"""Routing-aware functional objectives and budget allocation for MoE compression.

The planner treats every expert representation as a discrete action. Action zero
is removal; subsequent actions are increasingly expensive approximations, with
the final action conventionally representing the full expert. Unlike independent
expert-error objectives, every candidate plan is evaluated after recomputing
top-k routing and aggregating all selected expert outputs.

Legacy traces store exact hidden vectors or fixed random projections. Exact
Gram traces instead store only the inner products needed to evaluate routed
squared error, avoiding both full output storage and projection randomness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil, pi, sqrt
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistortionReport:
    """Category-separated routed aggregate distortion for one layer."""

    numerator: torch.Tensor
    denominator: torch.Tensor
    relative: torch.Tensor
    route_change_rate: float


@dataclass(frozen=True)
class BatchedDistortionReport:
    """Vectorized reports for several plans of the same layer."""

    numerator: torch.Tensor
    denominator: torch.Tensor
    relative: torch.Tensor
    route_change_rate: torch.Tensor


class LayerFunctionalTrace:
    """In-memory functional trace for one MoE layer.

    Shapes:

    * ``router_logits``: ``[tokens, experts]`` for the candidate routing pool.
    * output mode: ``reference_output`` and ``candidate_outputs``;
    * Gram mode: ``candidate_gram``, ``reference_cross``, and
      ``reference_energy`` over non-removed actions in ``routing_order``.
    * ``action_costs``: ``[experts, actions]`` in bytes or parameters.

    Candidate outputs at action zero are ignored because that action removes the
    expert. All tensors are deliberately ordinary PyTorch tensors so traces can
    be kept on CPU and moved to an accelerator layer by layer if desired.
    """

    def __init__(
        self,
        *,
        name: str,
        router_logits: torch.Tensor,
        reference_output: torch.Tensor | None = None,
        candidate_outputs: torch.Tensor | None = None,
        candidate_gram: torch.Tensor | None = None,
        reference_cross: torch.Tensor | None = None,
        reference_energy: torch.Tensor | None = None,
        category_ids: torch.Tensor,
        category_names: Sequence[str],
        action_costs: torch.Tensor,
        action_names: Sequence[str],
        top_k: int,
        norm_topk_prob: bool,
        reference_topk_ids: torch.Tensor | None = None,
        routing_candidate_mask: torch.Tensor | None = None,
        routing_order: torch.Tensor | None = None,
    ) -> None:
        self.name = str(name)
        self.router_logits = router_logits.detach()
        self.reference_output = (
            None if reference_output is None else reference_output.detach()
        )
        self.candidate_outputs = (
            None if candidate_outputs is None else candidate_outputs.detach()
        )
        self.candidate_gram = (
            None if candidate_gram is None else candidate_gram.detach()
        )
        self.reference_cross = (
            None if reference_cross is None else reference_cross.detach()
        )
        self.reference_energy = (
            None if reference_energy is None else reference_energy.detach()
        )
        self.category_ids = category_ids.detach().long()
        self.category_names = tuple(str(value) for value in category_names)
        self.action_costs = action_costs.detach().long()
        self.action_names = tuple(str(value) for value in action_names)
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.reference_topk_ids = (
            None if reference_topk_ids is None else reference_topk_ids.detach().long()
        )
        self.routing_candidate_mask = (
            None
            if routing_candidate_mask is None
            else routing_candidate_mask.detach().bool()
        )
        self.routing_order = (
            None if routing_order is None else routing_order.detach().long()
        )
        self._validate()

    @property
    def tokens(self) -> int:
        return int(self.router_logits.shape[0])

    @property
    def experts(self) -> int:
        return int(self.router_logits.shape[1])

    @property
    def actions(self) -> int:
        return len(self.action_names)

    @property
    def uses_gram(self) -> bool:
        return self.candidate_gram is not None

    def _validate(self) -> None:
        if self.router_logits.ndim != 2:
            raise ValueError("router_logits must have shape [tokens, experts]")
        tokens, experts = map(int, self.router_logits.shape)
        output_mode = self.reference_output is not None or self.candidate_outputs is not None
        gram_mode = any(
            value is not None
            for value in (
                self.candidate_gram,
                self.reference_cross,
                self.reference_energy,
            )
        )
        if output_mode == gram_mode:
            raise ValueError("trace must contain exactly one of output or Gram data")
        actions = len(self.action_names)
        if output_mode:
            if self.reference_output is None or self.candidate_outputs is None:
                raise ValueError("output traces require reference and candidate outputs")
            if self.reference_output.ndim != 2 or int(self.reference_output.shape[0]) != tokens:
                raise ValueError("reference_output must have shape [tokens, sketch]")
            if self.candidate_outputs.ndim != 4:
                raise ValueError(
                    "candidate_outputs must have shape [tokens, experts, actions, sketch]"
                )
            out_tokens, out_experts, out_actions, sketch = map(
                int, self.candidate_outputs.shape
            )
            if (out_tokens, out_experts, out_actions, sketch) != (
                tokens,
                experts,
                actions,
                int(self.reference_output.shape[1]),
            ):
                raise ValueError("candidate output dimensions do not match the trace")
        else:
            if any(
                value is None
                for value in (
                    self.candidate_gram,
                    self.reference_cross,
                    self.reference_energy,
                )
            ):
                raise ValueError("Gram traces require gram, cross, and energy tensors")
            if self.routing_order is None:
                raise ValueError("Gram traces require routing_order")
            vectors = int(self.routing_order.shape[1]) * (actions - 1)
            if tuple(self.candidate_gram.shape) != (tokens, vectors, vectors):
                raise ValueError("candidate_gram has incompatible dimensions")
            if tuple(self.reference_cross.shape) != (tokens, vectors):
                raise ValueError("reference_cross has incompatible dimensions")
            if tuple(self.reference_energy.shape) != (tokens,):
                raise ValueError("reference_energy must have shape [tokens]")
        if actions < 2:
            raise ValueError("at least removed and one retained action are required")
        if self.action_names[0] != "removed":
            raise ValueError("action zero must be named 'removed'")
        if tuple(self.action_costs.shape) != (experts, actions):
            raise ValueError("action_costs must have shape [experts, actions]")
        if bool((self.action_costs[:, 0] != 0).any()):
            raise ValueError("removed actions must have zero cost")
        if bool((self.action_costs[:, 1:] < self.action_costs[:, :-1]).any()):
            raise ValueError("expert action costs must be nondecreasing")
        if not (0 < self.top_k <= experts):
            raise ValueError("top_k must be in [1, experts]")
        if self.category_ids.shape != (tokens,):
            raise ValueError("category_ids must have shape [tokens]")
        if not self.category_names:
            raise ValueError("at least one category is required")
        if tokens and bool(
            ((self.category_ids < 0) | (self.category_ids >= len(self.category_names))).any()
        ):
            raise ValueError("category id is outside category_names")
        if self.reference_topk_ids is not None and tuple(
            self.reference_topk_ids.shape
        ) != (tokens, self.top_k):
            raise ValueError("reference_topk_ids must have shape [tokens, top_k]")
        if self.routing_candidate_mask is not None:
            if tuple(self.routing_candidate_mask.shape) != (tokens, experts):
                raise ValueError(
                    "routing_candidate_mask must have shape [tokens, experts]"
                )
            if tokens and bool(
                (self.routing_candidate_mask.sum(dim=1) < self.top_k).any()
            ):
                raise ValueError(
                    "routing_candidate_mask must expose at least top_k experts per token"
                )
        if self.routing_order is not None:
            if self.routing_order.ndim != 2 or int(self.routing_order.shape[0]) != tokens:
                raise ValueError("routing_order must have shape [tokens, candidates]")
            if int(self.routing_order.shape[1]) < self.top_k:
                raise ValueError("routing_order must contain at least top_k candidates")
            if bool(((self.routing_order < 0) | (self.routing_order >= experts)).any()):
                raise ValueError("routing_order contains an invalid expert index")
            if any(
                torch.unique(row).numel() != row.numel()
                for row in self.routing_order
            ):
                raise ValueError("routing_order contains duplicate experts")

    def memory(self, expert_actions: torch.Tensor) -> int:
        actions = self._validate_plan(expert_actions)
        expert_ids = torch.arange(self.experts, device=actions.device)
        costs = self.action_costs.to(actions.device)[expert_ids, actions]
        return int(costs.sum().item())

    def _validate_plan(self, expert_actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(
            expert_actions, dtype=torch.long, device=self.router_logits.device
        ).reshape(-1)
        if int(actions.numel()) != self.experts:
            raise ValueError(f"{self.name}: plan must contain one action per expert")
        if bool(((actions < 0) | (actions >= self.actions)).any()):
            raise ValueError(f"{self.name}: invalid action index")
        if int((actions != 0).sum().item()) < self.top_k:
            raise ValueError(f"{self.name}: plan retains fewer than top_k experts")
        return actions

    def _aggregate_token_distortion(
        self,
        residual_energy: torch.Tensor,
        reference_energy: torch.Tensor,
        *,
        token_reduction: str,
        token_energy_floor_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate token errors by category without mixing incompatible scales.

        ``energy`` preserves the legacy ratio-of-sums objective. ``relative``
        first normalizes every token by its reference-output energy and then
        averages the normalized errors. A category-local median-energy floor
        prevents nearly-zero reference outputs from producing unbounded ratios.

        ``residual_energy`` may have shape ``[tokens]`` or ``[plans, tokens]``.
        Returned numerator/denominator/relative tensors always have a leading
        plan dimension; scalar evaluation removes it at the call site.
        """

        token_reduction = str(token_reduction)
        if token_reduction not in {"energy", "relative"}:
            raise ValueError("token_reduction must be 'energy' or 'relative'")
        if float(token_energy_floor_ratio) < 0:
            raise ValueError("token_energy_floor_ratio must be nonnegative")
        residual = residual_energy.float()
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)
        if residual.ndim != 2 or int(residual.shape[1]) != self.tokens:
            raise ValueError("residual energy must have shape [tokens] or [plans, tokens]")
        reference = reference_energy.float().reshape(-1)
        if int(reference.numel()) != self.tokens:
            raise ValueError("reference energy must have one value per token")

        numerators = []
        denominators = []
        for category in range(len(self.category_names)):
            mask = self.category_ids.to(reference.device) == category
            category_reference = reference[mask]
            category_residual = residual[:, mask]
            if token_reduction == "energy":
                numerators.append(category_residual.sum(dim=1))
                denominators.append(category_reference.sum().expand(residual.shape[0]))
                continue

            # Keep the floor tied to the typical activation scale, while still
            # behaving sensibly when a category contains zero-energy tokens.
            median = category_reference.median()
            fallback_scale = category_reference.mean() * 1e-6
            scale = torch.maximum(median, fallback_scale).clamp_min(1e-12)
            floor = float(token_energy_floor_ratio) * scale
            normalized = category_residual / category_reference.clamp_min(floor)
            numerators.append(normalized.sum(dim=1))
            denominators.append(
                torch.full(
                    (residual.shape[0],),
                    int(mask.sum().item()),
                    device=residual.device,
                    dtype=residual.dtype,
                )
            )

        numerator = torch.stack(numerators, dim=1).double().cpu()
        denominator = torch.stack(denominators, dim=1).double().cpu()
        relative = numerator / denominator.clamp_min(1e-30)
        return numerator, denominator, relative

    @torch.no_grad()
    def evaluate(
        self,
        expert_actions: torch.Tensor,
        *,
        token_reduction: str = "energy",
        token_energy_floor_ratio: float = 0.05,
    ) -> DistortionReport:
        """Evaluate a complete plan after exact retained-pool top-k routing."""

        actions = self._validate_plan(expert_actions)
        removed = actions == 0
        logits = self.router_logits.float().masked_fill(removed.unsqueeze(0), -torch.inf)
        if self.routing_candidate_mask is not None:
            logits = logits.masked_fill(
                ~self.routing_candidate_mask.to(logits.device), -torch.inf
            )
        if self.routing_order is not None:
            order = self.routing_order.to(actions.device)
            retained_in_order = ~removed[order]
            available = retained_in_order.sum(dim=1)
        else:
            order = None
            retained_in_order = None
            available = torch.isfinite(logits).sum(dim=1)
        if bool((available < self.top_k).any()):
            categories = len(self.category_names)
            infinity = torch.full((categories,), torch.inf, dtype=torch.float64)
            return DistortionReport(
                numerator=infinity,
                denominator=torch.ones(categories, dtype=torch.float64),
                relative=infinity,
                route_change_rate=1.0,
            )
        if order is None:
            topk_values, topk_ids = torch.topk(logits, k=self.top_k, dim=-1)
        else:
            # Keep the original top-L ordering so exact BF16 router ties cannot
            # change merely because topk is invoked on a smaller masked tensor.
            positions = torch.arange(
                order.shape[1], device=order.device, dtype=torch.long
            ).unsqueeze(0).expand_as(order)
            positions = positions.masked_fill(~retained_in_order, order.shape[1])
            selected_positions = torch.topk(
                -positions, k=self.top_k, dim=1
            ).indices
            topk_ids = order.gather(1, selected_positions)
            topk_values = self.router_logits.float().gather(1, topk_ids)
        if self.norm_topk_prob:
            weights = F.softmax(topk_values, dim=-1)
        else:
            probabilities = F.softmax(logits, dim=-1)
            weights = probabilities.gather(1, topk_ids)

        token_ids = torch.arange(self.tokens, device=topk_ids.device).unsqueeze(1)
        selected_actions = actions[topk_ids]
        if self.uses_gram:
            if order is None:
                raise RuntimeError("Gram trace lost routing-order positions")
            vector_ids = selected_positions * (self.actions - 1) + selected_actions - 1
            cross = self.reference_cross.float().gather(1, vector_ids)
            rows = self.candidate_gram.float().gather(
                1,
                vector_ids.unsqueeze(2).expand(-1, -1, self.candidate_gram.shape[2]),
            )
            gram = rows.gather(
                2, vector_ids.unsqueeze(1).expand(-1, self.top_k, -1)
            )
            linear = (weights * cross).sum(dim=1)
            quadratic = torch.einsum("ti,tij,tj->t", weights, gram, weights)
            reference_energy = self.reference_energy.float()
            residual_energy = (reference_energy - 2 * linear + quadratic).clamp_min(0)
        else:
            outputs = self.candidate_outputs[
                token_ids,
                topk_ids,
                selected_actions,
            ].float()
            approximation = (weights.unsqueeze(-1) * outputs).sum(dim=1)
            residual_energy = (
                self.reference_output.float() - approximation
            ).square().sum(dim=-1)
            reference_energy = self.reference_output.float().square().sum(dim=-1)

        numerator, denominator, relative = self._aggregate_token_distortion(
            residual_energy,
            reference_energy,
            token_reduction=token_reduction,
            token_energy_floor_ratio=token_energy_floor_ratio,
        )
        numerator = numerator[0]
        denominator = denominator[0]
        relative = relative[0]

        route_change = 0.0
        if self.reference_topk_ids is not None and self.tokens:
            reference = torch.sort(self.reference_topk_ids.to(topk_ids.device), dim=1).values
            current = torch.sort(topk_ids, dim=1).values
            route_change = float((reference != current).any(dim=1).float().mean().item())
        return DistortionReport(
            numerator=numerator,
            denominator=denominator,
            relative=relative,
            route_change_rate=route_change,
        )

    @torch.no_grad()
    def evaluate_many(
        self,
        expert_actions: torch.Tensor,
        *,
        token_reduction: str = "energy",
        token_energy_floor_ratio: float = 0.05,
    ) -> BatchedDistortionReport:
        """Vectorized equivalent of :meth:`evaluate` for ``[plans, experts]``.

        This is used by the full-model allocator to score every one-step action
        in an affected layer with one tensor program instead of one Python call
        per expert.
        """

        plans = torch.as_tensor(
            expert_actions, dtype=torch.long, device=self.router_logits.device
        )
        if plans.ndim == 1:
            plans = plans.unsqueeze(0)
        if plans.ndim != 2 or int(plans.shape[1]) != self.experts:
            raise ValueError("plans must have shape [plans, experts]")
        if bool(((plans < 0) | (plans >= self.actions)).any()):
            raise ValueError(f"{self.name}: invalid action index")
        if bool(((plans != 0).sum(dim=1) < self.top_k).any()):
            raise ValueError(f"{self.name}: a plan retains fewer than top_k experts")

        count = int(plans.shape[0])
        removed = plans == 0
        if self.routing_order is not None:
            order = self.routing_order.to(plans.device)
            expanded_order = order.unsqueeze(0).expand(count, -1, -1)
            plan_grid = plans.unsqueeze(1).expand(-1, self.tokens, -1)
            retained = plan_grid.gather(2, expanded_order) != 0
            available = retained.sum(dim=2)
            positions = torch.arange(
                order.shape[1], device=plans.device, dtype=torch.long
            ).view(1, 1, -1).expand_as(expanded_order)
            positions = positions.masked_fill(~retained, order.shape[1])
            selected_positions = torch.topk(
                -positions, k=self.top_k, dim=2
            ).indices
            topk_ids = expanded_order.gather(2, selected_positions)
            logits_grid = self.router_logits.float().unsqueeze(0).expand(count, -1, -1)
            topk_values = logits_grid.gather(2, topk_ids)
        else:
            logits_grid = self.router_logits.float().unsqueeze(0).expand(count, -1, -1)
            logits_grid = logits_grid.masked_fill(removed.unsqueeze(1), -torch.inf)
            if self.routing_candidate_mask is not None:
                logits_grid = logits_grid.masked_fill(
                    ~self.routing_candidate_mask.to(plans.device).unsqueeze(0),
                    -torch.inf,
                )
            available = torch.isfinite(logits_grid).sum(dim=2)
            topk_values, topk_ids = torch.topk(logits_grid, k=self.top_k, dim=2)
        invalid = (available < self.top_k).any(dim=1)

        if self.norm_topk_prob:
            weights = F.softmax(topk_values, dim=2)
        else:
            if self.routing_order is not None:
                logits_grid = self.router_logits.float().unsqueeze(0).expand(count, -1, -1)
                logits_grid = logits_grid.masked_fill(removed.unsqueeze(1), -torch.inf)
                if self.routing_candidate_mask is not None:
                    logits_grid = logits_grid.masked_fill(
                        ~self.routing_candidate_mask.to(plans.device).unsqueeze(0),
                        -torch.inf,
                    )
            weights = F.softmax(logits_grid, dim=2).gather(2, topk_ids)

        plan_grid = plans.unsqueeze(1).expand(-1, self.tokens, -1)
        selected_actions = plan_grid.gather(2, topk_ids)
        if self.uses_gram:
            if self.routing_order is None:
                raise RuntimeError("Gram trace lost routing-order positions")
            vector_ids = (
                selected_positions * (self.actions - 1) + selected_actions - 1
            )
            token_grid = torch.arange(self.tokens, device=plans.device).view(1, -1, 1)
            token_grid = token_grid.expand(count, -1, self.top_k)
            cross = self.reference_cross.float()[token_grid, vector_ids]
            pair_tokens = token_grid.unsqueeze(3).expand(
                -1, -1, self.top_k, self.top_k
            )
            pair_rows = vector_ids.unsqueeze(3).expand_as(pair_tokens)
            pair_cols = vector_ids.unsqueeze(2).expand_as(pair_tokens)
            gram = self.candidate_gram.float()[pair_tokens, pair_rows, pair_cols]
            linear = (weights * cross).sum(dim=2)
            quadratic = torch.einsum("pti,ptij,ptj->pt", weights, gram, weights)
            reference_energy = self.reference_energy.float()
            residual_energy = (
                reference_energy.unsqueeze(0) - 2 * linear + quadratic
            ).clamp_min(0)
        else:
            token_ids = torch.arange(self.tokens, device=plans.device).view(1, -1, 1)
            token_ids = token_ids.expand(count, -1, self.top_k)
            outputs = self.candidate_outputs[
                token_ids,
                topk_ids,
                selected_actions,
            ].float()
            approximation = (weights.unsqueeze(-1) * outputs).sum(dim=2)
            residual_energy = (
                self.reference_output.float().unsqueeze(0) - approximation
            ).square().sum(dim=-1)
            reference_energy = self.reference_output.float().square().sum(dim=-1)

        numerator, denominator, relative = self._aggregate_token_distortion(
            residual_energy,
            reference_energy,
            token_reduction=token_reduction,
            token_energy_floor_ratio=token_energy_floor_ratio,
        )

        route_change = torch.zeros(count, dtype=torch.float32)
        if self.reference_topk_ids is not None and self.tokens:
            reference = torch.sort(
                self.reference_topk_ids.to(topk_ids.device), dim=1
            ).values
            current = torch.sort(topk_ids, dim=2).values
            route_change = (
                (current != reference.unsqueeze(0)).any(dim=2).float().mean(dim=1).cpu()
            )
        if bool(invalid.any()):
            numerator[invalid.cpu()] = torch.inf
            relative[invalid.cpu()] = torch.inf
            route_change[invalid.cpu()] = 1
        return BatchedDistortionReport(
            numerator=numerator,
            denominator=denominator,
            relative=relative,
            route_change_rate=route_change,
        )

    def state_dict(self) -> dict:
        """Return a portable CPU payload suitable for ``torch.save``."""

        return {
            "name": self.name,
            "router_logits": self.router_logits.cpu(),
            "reference_output": (
                None if self.reference_output is None else self.reference_output.cpu()
            ),
            "candidate_outputs": (
                None if self.candidate_outputs is None else self.candidate_outputs.cpu()
            ),
            "candidate_gram": (
                None if self.candidate_gram is None else self.candidate_gram.cpu()
            ),
            "reference_cross": (
                None if self.reference_cross is None else self.reference_cross.cpu()
            ),
            "reference_energy": (
                None if self.reference_energy is None else self.reference_energy.cpu()
            ),
            "category_ids": self.category_ids.cpu(),
            "category_names": self.category_names,
            "action_costs": self.action_costs.cpu(),
            "action_names": self.action_names,
            "top_k": self.top_k,
            "norm_topk_prob": self.norm_topk_prob,
            "reference_topk_ids": (
                None
                if self.reference_topk_ids is None
                else self.reference_topk_ids.cpu()
            ),
            "routing_candidate_mask": (
                None
                if self.routing_candidate_mask is None
                else self.routing_candidate_mask.cpu()
            ),
            "routing_order": (
                None if self.routing_order is None else self.routing_order.cpu()
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "LayerFunctionalTrace":
        return cls(**dict(state))


# Units a streaming trace can price per-token damage in. "energy" is the squared
# L2 norm of the routed output error; "amplitude" is its L2 norm.
DAMAGE_UNITS = ("energy", "amplitude")


class LayerIndependentActionTrace:
    """Compact full-token statistics for additive action planning.

    ``action_numerator[e, a, c]`` stores the full-calibration routed output
    squared error caused by assigning action ``a`` to expert ``e`` while all
    other experts remain full.  SVD entries are exact single-action errors;
    removal entries include exact top-k replacement and router
    renormalization for a single removed expert.  Combining several entries is
    an additive approximation, deliberately trading cross-action terms for the
    ability to use every calibration token without a per-token Gram trace.

    ``damage_unit`` says what "error" means for every per-token quantity here:
    ``energy`` sums squared L2 norms (the original), ``amplitude`` sums L2 norms.
    Fields named ``*_energy`` hold whichever unit the trace was built in.
    """

    def __init__(
        self,
        *,
        name: str,
        action_numerator: torch.Tensor,
        raw_action_numerator: torch.Tensor | None = None,
        expert_output_energy: torch.Tensor | None = None,
        linear_routing_output_energy: torch.Tensor | None = None,
        conditional_importance: torch.Tensor | None = None,
        active_token_count: torch.Tensor | None = None,
        structural_action_error: torch.Tensor | None = None,
        prompt_action_numerator: torch.Tensor | None = None,
        prompt_expert_output_energy: torch.Tensor | None = None,
        prompt_linear_routing_output_energy: torch.Tensor | None = None,
        prompt_active_token_count: torch.Tensor | None = None,
        prompt_reference_energy: torch.Tensor | None = None,
        prompt_category_ids: torch.Tensor | None = None,
        prompt_active_mask: torch.Tensor | None = None,
        damage_log_sum: torch.Tensor | None = None,
        damage_log_square_sum: torch.Tensor | None = None,
        damage_positive_count: torch.Tensor | None = None,
        removal_depth_numerator: torch.Tensor | None = None,
        removal_depth_log_sum: torch.Tensor | None = None,
        removal_depth_log_square_sum: torch.Tensor | None = None,
        removal_depth_positive_count: torch.Tensor | None = None,
        removal_depth_prompt_numerator: torch.Tensor | None = None,
        reference_energy: torch.Tensor,
        category_names: Sequence[str],
        action_costs: torch.Tensor,
        action_names: Sequence[str],
        top_k: int,
        damage_unit: str = "energy",
    ) -> None:
        self.name = str(name)
        self.action_numerator = action_numerator.detach().double().cpu()
        self.raw_action_numerator = (
            self.action_numerator.clone()
            if raw_action_numerator is None
            else raw_action_numerator.detach().double().cpu()
        )
        self.expert_output_energy = (
            None
            if expert_output_energy is None
            else expert_output_energy.detach().double().cpu()
        )
        self.linear_routing_output_energy = (
            None
            if linear_routing_output_energy is None
            else linear_routing_output_energy.detach().double().cpu()
        )
        self.conditional_importance = (
            None
            if conditional_importance is None
            else conditional_importance.detach().double().cpu()
        )
        self.active_token_count = (
            None
            if active_token_count is None
            else active_token_count.detach().long().cpu()
        )
        self.structural_action_error = (
            None
            if structural_action_error is None
            else structural_action_error.detach().double().cpu()
        )
        self.prompt_action_numerator = (
            None
            if prompt_action_numerator is None
            else prompt_action_numerator.detach().double().cpu()
        )
        self.prompt_expert_output_energy = (
            None
            if prompt_expert_output_energy is None
            else prompt_expert_output_energy.detach().double().cpu()
        )
        self.prompt_linear_routing_output_energy = (
            None
            if prompt_linear_routing_output_energy is None
            else prompt_linear_routing_output_energy.detach().double().cpu()
        )
        self.prompt_active_token_count = (
            None
            if prompt_active_token_count is None
            else prompt_active_token_count.detach().long().cpu()
        )
        self.prompt_reference_energy = (
            None
            if prompt_reference_energy is None
            else prompt_reference_energy.detach().double().cpu().reshape(-1)
        )
        self.prompt_category_ids = (
            None
            if prompt_category_ids is None
            else prompt_category_ids.detach().long().cpu().reshape(-1)
        )
        self.prompt_active_mask = (
            None
            if prompt_active_mask is None
            else prompt_active_mask.detach().bool().cpu()
        )
        self.damage_log_sum = (
            None if damage_log_sum is None else damage_log_sum.detach().double().cpu()
        )
        self.damage_log_square_sum = (
            None
            if damage_log_square_sum is None
            else damage_log_square_sum.detach().double().cpu()
        )
        self.damage_positive_count = (
            None
            if damage_positive_count is None
            else damage_positive_count.detach().long().cpu()
        )
        # Removal damage resolved by how deep in the router shortlist the
        # surviving fallback expert sits. Depth zero is the top-(k+1) candidate
        # and duplicates the action-zero statistics above; deeper entries are
        # what a plan that also removed the nearer candidates actually incurs.
        self.removal_depth_numerator = (
            None
            if removal_depth_numerator is None
            else removal_depth_numerator.detach().double().cpu()
        )
        self.removal_depth_log_sum = (
            None
            if removal_depth_log_sum is None
            else removal_depth_log_sum.detach().double().cpu()
        )
        self.removal_depth_log_square_sum = (
            None
            if removal_depth_log_square_sum is None
            else removal_depth_log_square_sum.detach().double().cpu()
        )
        self.removal_depth_positive_count = (
            None
            if removal_depth_positive_count is None
            else removal_depth_positive_count.detach().long().cpu()
        )
        self.removal_depth_prompt_numerator = (
            None
            if removal_depth_prompt_numerator is None
            else removal_depth_prompt_numerator.detach().double().cpu()
        )
        self.reference_energy = reference_energy.detach().double().cpu().reshape(-1)
        self.category_names = tuple(str(value) for value in category_names)
        self.action_costs = action_costs.detach().long().cpu()
        self.action_names = tuple(str(value) for value in action_names)
        self.top_k = int(top_k)
        self.damage_unit = str(damage_unit)
        if self.damage_unit not in DAMAGE_UNITS:
            raise ValueError(
                f"damage_unit must be one of {DAMAGE_UNITS}, got {self.damage_unit!r}"
            )
        # Interaction correction state. Occupancy is derived from statistics the
        # trace already holds, so no new measurement is needed; rho defaults to 0,
        # which reproduces the additive objective exactly.
        self.interaction_rho = 0.0
        self._expert_occupancy: torch.Tensor | None = None
        self._validate()

    @property
    def experts(self) -> int:
        return int(self.action_numerator.shape[0])

    @property
    def actions(self) -> int:
        return int(self.action_numerator.shape[1])

    @property
    def tokens(self) -> int:
        # Compact traces intentionally do not retain per-token tensors.
        return 0

    @property
    def uses_gram(self) -> bool:
        return False

    def _validate(self) -> None:
        if self.action_numerator.ndim != 3:
            raise ValueError("action_numerator must have shape [experts, actions, categories]")
        if tuple(self.raw_action_numerator.shape) != tuple(self.action_numerator.shape):
            raise ValueError("raw_action_numerator must match action_numerator")
        experts, actions, categories = map(int, self.action_numerator.shape)
        for field_name in (
            "expert_output_energy",
            "linear_routing_output_energy",
            "conditional_importance",
            "active_token_count",
        ):
            value = getattr(self, field_name)
            if value is not None and tuple(value.shape) != (experts, categories):
                raise ValueError(
                    f"{field_name} must have shape [experts, categories]"
                )
        if self.structural_action_error is not None and tuple(
            self.structural_action_error.shape
        ) != (experts, actions):
            raise ValueError(
                "structural_action_error must have shape [experts, actions]"
            )
        prompt_fields = (
            self.prompt_action_numerator,
            self.prompt_reference_energy,
            self.prompt_category_ids,
            self.prompt_active_mask,
        )
        if any(value is not None for value in prompt_fields):
            if any(value is None for value in prompt_fields):
                raise ValueError("prompt-level trace fields must be provided together")
            prompts = int(self.prompt_reference_energy.numel())
            if tuple(self.prompt_action_numerator.shape) != (
                experts,
                actions,
                prompts,
            ):
                raise ValueError(
                    "prompt_action_numerator must have shape "
                    "[experts, actions, prompts]"
                )
            if self.prompt_expert_output_energy is not None and tuple(
                self.prompt_expert_output_energy.shape
            ) != (experts, prompts):
                raise ValueError(
                    "prompt_expert_output_energy must have shape "
                    "[experts, prompts]"
                )
            if self.prompt_linear_routing_output_energy is not None and tuple(
                self.prompt_linear_routing_output_energy.shape
            ) != (experts, prompts):
                raise ValueError(
                    "prompt_linear_routing_output_energy must have shape "
                    "[experts, prompts]"
                )
            if self.prompt_active_token_count is not None and tuple(
                self.prompt_active_token_count.shape
            ) != (experts, prompts):
                raise ValueError(
                    "prompt_active_token_count must have shape [experts, prompts]"
                )
            if tuple(self.prompt_category_ids.shape) != (prompts,):
                raise ValueError("prompt_category_ids must have one value per prompt")
            if tuple(self.prompt_active_mask.shape) != (experts, prompts):
                raise ValueError(
                    "prompt_active_mask must have shape [experts, prompts]"
                )
            if prompts <= 0:
                raise ValueError("prompt-level trace must contain at least one prompt")
            if bool(
                ((self.prompt_category_ids < 0)
                | (self.prompt_category_ids >= categories)).any()
            ):
                raise ValueError("prompt_category_ids contains an invalid category")
        elif any(
            value is not None
            for value in (
                self.prompt_expert_output_energy,
                self.prompt_linear_routing_output_energy,
                self.prompt_active_token_count,
            )
        ):
            raise ValueError(
                "prompt expert statistics require the prompt-level trace fields"
            )
        damage_fields = (
            "damage_log_sum",
            "damage_log_square_sum",
            "damage_positive_count",
        )
        damage_values = [getattr(self, field_name) for field_name in damage_fields]
        if any(value is not None for value in damage_values):
            if any(value is None for value in damage_values):
                raise ValueError("log-damage trace fields must be provided together")
            for field_name, value in zip(damage_fields, damage_values):
                if tuple(value.shape) != (experts, actions, categories):
                    raise ValueError(
                        f"{field_name} must have shape [experts, actions, categories]"
                    )
            # Log damages are signed, so only their second moment and counts are
            # sign constrained. Both sums must still be finite: a non-finite
            # entry means a zero or overflowing damage leaked into the log.
            if not bool(torch.isfinite(self.damage_log_sum).all()) or not bool(
                torch.isfinite(self.damage_log_square_sum).all()
            ):
                raise ValueError("log-damage moments must be finite")
            if bool((self.damage_log_square_sum < 0).any()):
                raise ValueError("damage_log_square_sum must be nonnegative")
            if bool((self.damage_positive_count < 0).any()):
                raise ValueError("damage_positive_count must be nonnegative")
        depth_fields = (
            "removal_depth_numerator",
            "removal_depth_log_sum",
            "removal_depth_log_square_sum",
            "removal_depth_positive_count",
            "removal_depth_prompt_numerator",
        )
        depth_values = [getattr(self, field_name) for field_name in depth_fields]
        if any(value is not None for value in depth_values):
            if any(value is None for value in depth_values):
                raise ValueError(
                    "removal-depth trace fields must be provided together"
                )
            if self.removal_depth_numerator.ndim != 3:
                raise ValueError(
                    "removal_depth_numerator must have shape "
                    "[experts, depths, categories]"
                )
            depths = int(self.removal_depth_numerator.shape[1])
            if depths < 1:
                raise ValueError("at least one removal fallback depth is required")
            for field_name in depth_fields[:4]:
                value = getattr(self, field_name)
                if tuple(value.shape) != (experts, depths, categories):
                    raise ValueError(
                        f"{field_name} must have shape [experts, depths, categories]"
                    )
            if self.prompt_reference_energy is None:
                raise ValueError(
                    "removal-depth statistics require the prompt-level trace fields"
                )
            prompts = int(self.prompt_reference_energy.numel())
            if tuple(self.removal_depth_prompt_numerator.shape) != (
                experts,
                depths,
                prompts,
            ):
                raise ValueError(
                    "removal_depth_prompt_numerator must have shape "
                    "[experts, depths, prompts]"
                )
            for field_name in (
                "removal_depth_numerator",
                "removal_depth_log_square_sum",
                "removal_depth_positive_count",
                "removal_depth_prompt_numerator",
            ):
                if bool((getattr(self, field_name) < 0).any()):
                    raise ValueError(f"{field_name} must be nonnegative")
            if not bool(torch.isfinite(self.removal_depth_log_sum).all()):
                raise ValueError("removal_depth_log_sum must be finite")
        if actions != len(self.action_names):
            raise ValueError("action names do not match action_numerator")
        if categories != len(self.category_names):
            raise ValueError("category names do not match action_numerator")
        if tuple(self.reference_energy.shape) != (categories,):
            raise ValueError("reference_energy must have one value per category")
        if tuple(self.action_costs.shape) != (experts, actions):
            raise ValueError("action_costs must have shape [experts, actions]")
        if actions < 2 or self.action_names[0] != "removed":
            raise ValueError("action zero must be the removed action")
        if bool((self.action_costs[:, 0] != 0).any()):
            raise ValueError("removed actions must have zero cost")
        if bool((self.action_costs[:, 1:] < self.action_costs[:, :-1]).any()):
            raise ValueError("expert action costs must be nondecreasing")
        if not (0 < self.top_k <= experts):
            raise ValueError("top_k must be in [1, experts]")
        if not bool(torch.isfinite(self.action_numerator).all()):
            raise ValueError("action_numerator must be finite")
        if not bool(torch.isfinite(self.raw_action_numerator).all()):
            raise ValueError("raw_action_numerator must be finite")
        for field_name in (
            "expert_output_energy",
            "linear_routing_output_energy",
            "conditional_importance",
            "structural_action_error",
            "prompt_action_numerator",
            "prompt_expert_output_energy",
            "prompt_linear_routing_output_energy",
            "prompt_reference_energy",
        ):
            value = getattr(self, field_name)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{field_name} must be finite")
            if value is not None and bool((value < 0).any()):
                raise ValueError(f"{field_name} must be nonnegative")
        if self.active_token_count is not None and bool(
            (self.active_token_count < 0).any()
        ):
            raise ValueError("active_token_count must be nonnegative")
        if self.prompt_active_token_count is not None and bool(
            (self.prompt_active_token_count < 0).any()
        ):
            raise ValueError("prompt_active_token_count must be nonnegative")
        if bool((self.reference_energy < 0).any()):
            raise ValueError("reference_energy must be nonnegative")

    def _validate_plan(self, expert_actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(expert_actions, dtype=torch.long).reshape(-1)
        if int(actions.numel()) != self.experts:
            raise ValueError(f"{self.name}: plan must contain one action per expert")
        if bool(((actions < 0) | (actions >= self.actions)).any()):
            raise ValueError(f"{self.name}: invalid action index")
        if int((actions != 0).sum().item()) < self.top_k:
            raise ValueError(f"{self.name}: plan retains fewer than top_k experts")
        return actions

    def memory(self, expert_actions: torch.Tensor) -> int:
        actions = self._validate_plan(expert_actions)
        expert_ids = torch.arange(self.experts)
        return int(self.action_costs[expert_ids, actions].sum().item())

    def set_interaction_model(
        self, rho: float, occupancy: torch.Tensor | None = None
    ) -> None:
        """Enable the co-perturbation correction with correlation ``rho``.

        ``occupancy`` is the share of tokens whose top-k contains each expert,
        shaped ``[experts, categories]``. Supply it when the trace is a derived
        one that does not carry ``active_token_count`` -- the risk traces the
        allocator actually receives are rebuilt and drop it. ``rho = 0`` clears
        the correction and restores the exact additive objective.
        """

        self.interaction_rho = float(rho)
        if not self.interaction_rho:
            self._expert_occupancy = None
            return
        if occupancy is None:
            if self.active_token_count is None:
                raise ValueError(
                    f"{self.name}: the interaction model needs occupancy or "
                    "active_token_count"
                )
            counts = self.active_token_count.double()
            tokens = (counts.sum(dim=0) / float(self.top_k)).clamp_min(1.0)
            occupancy = counts / tokens.unsqueeze(0)
        occupancy = torch.as_tensor(occupancy).double().reshape(int(self.experts), -1)
        if int(occupancy.shape[1]) == 1 and int(self.reference_energy.numel()) > 1:
            occupancy = occupancy.expand(-1, int(self.reference_energy.numel()))
        self._expert_occupancy = occupancy.contiguous()

    def _interaction_scale(self, plans: torch.Tensor) -> torch.Tensor | None:
        """Inflate additive damage by the co-perturbation a token actually sees.

        Summing ``|r_e dy_e|^2`` over experts is exact only if the per-expert
        error vectors are orthogonal. A token routes to ``top_k`` experts at
        once, and when several are compressed their errors can carry cross
        terms the sum cannot see.

        Model the mean pairwise correlation of co-routed error vectors as ``rho``.
        A token whose top-k holds ``m`` perturbed experts then has
        ``true ~ additive * (1 + rho * (m - 1))`` -- one perturbed expert has no
        partner to interfere with, hence ``m - 1``. Aggregating over tokens
        caps ``m`` by the expected occupancy of perturbed experts. It also
        estimates an effective multiplicity from the selected action damages,
        ``m_error = (sum sqrt(d_e))^2 / sum d_e``. This prevents many negligible
        perturbations from receiving the same interaction penalty as several
        equally large ones.

        This is the term that punishes buying *many* cheap tiers. A metric that
        makes each tier more faithful lowers every ``dy_e``, and a purely additive
        objective answers by buying more of them until the cross terms it never
        priced dominate the error.
        """

        if not self.interaction_rho or self._expert_occupancy is None:
            return None
        perturbed = (plans != (self.actions - 1)).double()
        occupancy_multiplicity = perturbed @ self._expert_occupancy
        expert_ids = torch.arange(self.experts).unsqueeze(0)
        selected_damage = self.action_numerator[expert_ids, plans]
        root_sum = selected_damage.clamp_min(0).sqrt().sum(dim=1)
        damage_sum = selected_damage.clamp_min(0).sum(dim=1)
        error_multiplicity = root_sum.square() / damage_sum.clamp_min(1e-30)
        effective = torch.minimum(occupancy_multiplicity, error_multiplicity)
        return 1.0 + float(self.interaction_rho) * (effective - 1.0).clamp_min(0.0)

    def evaluate(self, expert_actions: torch.Tensor, **_ignored) -> DistortionReport:
        actions = self._validate_plan(expert_actions)
        expert_ids = torch.arange(self.experts)
        numerator = self.action_numerator[expert_ids, actions].sum(dim=0)
        scale = self._interaction_scale(actions.unsqueeze(0))
        if scale is not None:
            numerator = numerator * scale.reshape(-1)
        denominator = self.reference_energy.clone()
        relative = numerator / denominator.clamp_min(1e-30)
        return DistortionReport(
            numerator=numerator,
            denominator=denominator,
            relative=relative,
            route_change_rate=0.0,
        )

    def evaluate_many(
        self, expert_actions: torch.Tensor, **_ignored
    ) -> BatchedDistortionReport:
        plans = torch.as_tensor(expert_actions, dtype=torch.long)
        if plans.ndim == 1:
            plans = plans.unsqueeze(0)
        if plans.ndim != 2 or int(plans.shape[1]) != self.experts:
            raise ValueError("plans must have shape [plans, experts]")
        if bool(((plans < 0) | (plans >= self.actions)).any()):
            raise ValueError(f"{self.name}: invalid action index")
        if bool(((plans != 0).sum(dim=1) < self.top_k).any()):
            raise ValueError(f"{self.name}: a plan retains fewer than top_k experts")
        expert_ids = torch.arange(self.experts).unsqueeze(0)
        numerator = self.action_numerator[expert_ids, plans].sum(dim=1)
        scale = self._interaction_scale(plans)
        if scale is not None:
            numerator = numerator * scale
        denominator = self.reference_energy.unsqueeze(0).expand_as(numerator)
        relative = numerator / denominator.clamp_min(1e-30)
        return BatchedDistortionReport(
            numerator=numerator,
            denominator=denominator,
            relative=relative,
            route_change_rate=torch.zeros(int(plans.shape[0])),
        )

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "action_numerator": self.action_numerator,
            "raw_action_numerator": self.raw_action_numerator,
            "expert_output_energy": self.expert_output_energy,
            "linear_routing_output_energy": self.linear_routing_output_energy,
            "conditional_importance": self.conditional_importance,
            "active_token_count": self.active_token_count,
            "structural_action_error": self.structural_action_error,
            "prompt_action_numerator": self.prompt_action_numerator,
            "prompt_expert_output_energy": self.prompt_expert_output_energy,
            "prompt_linear_routing_output_energy": (
                self.prompt_linear_routing_output_energy
            ),
            "prompt_active_token_count": self.prompt_active_token_count,
            "prompt_reference_energy": self.prompt_reference_energy,
            "prompt_category_ids": self.prompt_category_ids,
            "prompt_active_mask": self.prompt_active_mask,
            "damage_log_sum": self.damage_log_sum,
            "damage_log_square_sum": self.damage_log_square_sum,
            "damage_positive_count": self.damage_positive_count,
            "removal_depth_numerator": self.removal_depth_numerator,
            "removal_depth_log_sum": self.removal_depth_log_sum,
            "removal_depth_log_square_sum": self.removal_depth_log_square_sum,
            "removal_depth_positive_count": self.removal_depth_positive_count,
            "removal_depth_prompt_numerator": self.removal_depth_prompt_numerator,
            "reference_energy": self.reference_energy,
            "category_names": self.category_names,
            "action_costs": self.action_costs,
            "action_names": self.action_names,
            "top_k": self.top_k,
            "damage_unit": self.damage_unit,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "LayerIndependentActionTrace":
        return cls(**dict(state))


# Largest log-risk that still exponentiates to a finite float64, with headroom
# for the products the risk builders form afterwards.
_MAX_LOG_RISK = float(
    torch.log(torch.tensor(torch.finfo(torch.float64).max, dtype=torch.float64))
) - 2.0


def _geometric_median(values: torch.Tensor, *, default: float = 1.0) -> torch.Tensor:
    """Return the multiplicative center of the positive entries of ``values``.

    A geometric median is robust and preserves multiplicative scale, which is
    the natural summary for the relative quantities every risk builder shrinks
    toward. ``default`` is returned when nothing positive was observed, where a
    neutral scale is safer than an arbitrary one.
    """

    positive = values.double().reshape(-1)
    positive = positive[positive > 0]
    if not int(positive.numel()):
        return torch.tensor(float(default), dtype=torch.float64)
    return torch.quantile(positive.log(), 0.5).exp()


def _empirical_bayes_weights(
    values: torch.Tensor,
    sampling_variance: torch.Tensor,
    evidence: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(confidence, posterior_variance, between_variance)`` per expert.

    The shrinkage weight is ``tau^2 / (tau^2 + v_e)``, where ``v_e`` is that
    expert's sampling variance and ``tau^2`` is the between-expert variance left
    after subtracting the mean sampling variance. Confidence therefore comes
    from the data rather than from a fixed pseudo-count, and an expert with no
    usable evidence gets weight zero so it falls back entirely on its prior.

    Both arguments must already be expressed on whatever scale the caller means
    to shrink on -- log scale for a positive multiplicative quantity, linear
    scale for something like a dispersion.
    """

    eps = float(epsilon)
    reliable = evidence & torch.isfinite(sampling_variance)
    observed = values[reliable]
    if int(observed.numel()) >= 2:
        between_variance = (
            observed.var(unbiased=True) - sampling_variance[reliable].mean()
        ).clamp_min(0.0)
    else:
        between_variance = torch.tensor(0.0, dtype=torch.float64)
    if float(between_variance.item()) <= 0:
        zeros = torch.zeros_like(values)
        return zeros, zeros.clone(), between_variance
    confidence = between_variance / (between_variance + sampling_variance)
    confidence = torch.where(
        torch.isfinite(confidence), confidence, torch.zeros_like(confidence)
    )
    confidence = torch.where(evidence, confidence, torch.zeros_like(confidence))
    posterior_variance = torch.where(
        torch.isfinite(sampling_variance),
        between_variance
        * sampling_variance
        / (between_variance + sampling_variance).clamp_min(eps),
        between_variance,
    )
    return confidence, posterior_variance, between_variance


def _log_space_posterior(
    observed: torch.Tensor,
    log_variance: torch.Tensor,
    prior: torch.Tensor,
    evidence: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shrink one per-expert observation vector toward a prior in log space.

    Returns ``(posterior, confidence, posterior_log_variance, between_variance)``.
    """

    eps = float(epsilon)
    logs = observed.clamp_min(eps).log()
    confidence, posterior_log_variance, between_variance = _empirical_bayes_weights(
        logs, log_variance, evidence, epsilon=eps
    )
    posterior = (
        confidence * logs + (1.0 - confidence) * prior.clamp_min(eps).log()
    ).exp()
    return posterior, confidence, posterior_log_variance, between_variance


def _conservative_upper(
    posterior: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    *,
    uncertainty_z: float,
    epsilon: float,
) -> torch.Tensor:
    """Return a one-sided log-space upper bound on a positive posterior."""

    return torch.exp(
        (
            posterior.clamp_min(float(epsilon)).log()
            + float(uncertainty_z) * posterior_log_variance.clamp_min(0).sqrt()
        ).clamp_max(_MAX_LOG_RISK)
    )


def build_importance_compressibility_traces(
    traces: Sequence[LayerIndependentActionTrace],
    *,
    confidence_kappa: float = 64.0,
    importance_power: float = 2.0,
    importance_prior: str = "median",
) -> tuple[list[LayerIndependentActionTrace], dict[str, dict[str, torch.Tensor]]]:
    """Convert full-token traces into conditional importance/action risks.

    REAP-style conditional routed magnitude anchors expert importance.  The
    action term is the routed expert-output reconstruction error normalized by
    that expert's routed output energy.  Sparse observations are shrunk toward
    a structural projection-weight reconstruction prior using
    ``n / (n + confidence_kappa)``.  Conditional importance itself is shrunk
    toward a layer/category prior for the same reason, protecting specialists
    that receive too few calibration tokens to estimate reliably.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one streaming trace is required")
    if float(confidence_kappa) < 0:
        raise ValueError("confidence_kappa must be nonnegative")
    if float(importance_power) <= 0:
        raise ValueError("importance_power must be positive")
    if importance_prior not in {"median", "mean"}:
        raise ValueError("importance_prior must be median or mean")

    converted = []
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for trace in traces:
        if not isinstance(trace, LayerIndependentActionTrace):
            raise TypeError(
                "importance-compressibility planning requires streaming traces"
            )
        required = {
            "expert_output_energy": trace.expert_output_energy,
            "conditional_importance": trace.conditional_importance,
            "active_token_count": trace.active_token_count,
            "structural_action_error": trace.structural_action_error,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{trace.name}: cached trace lacks {', '.join(missing)}; "
                "rebuild the streaming trace with the new planner"
            )

        counts = trace.active_token_count.double()
        observed_importance = trace.conditional_importance.double()
        structural = trace.structural_action_error.double()
        output_energy = trace.expert_output_energy.double()
        categories = len(trace.category_names)
        prior = torch.empty(categories, dtype=torch.float64)
        for category in range(categories):
            observed = counts[:, category] > 0
            values = observed_importance[observed, category]
            positive = values[values > 0]
            if int(positive.numel()):
                prior[category] = (
                    positive.median()
                    if importance_prior == "median"
                    else positive.mean()
                )
            else:
                # A completely unobserved/zero-output category carries no
                # evidence for relative expert ordering. A unit prior makes
                # every expert equally protected instead of falsely safe.
                prior[category] = 1.0
        prior = prior.clamp_min(1e-30)

        confidence = counts / (counts + float(confidence_kappa))
        if float(confidence_kappa) == 0:
            confidence = torch.ones_like(counts)
        shrunken_importance = (
            confidence * observed_importance
            + (1.0 - confidence) * prior.unsqueeze(0)
        )
        normalized_importance = shrunken_importance / prior.unsqueeze(0)

        activation_error = trace.raw_action_numerator / output_energy.unsqueeze(
            1
        ).clamp_min(1e-30)
        has_output_evidence = output_energy > 1e-30
        action_confidence = confidence * has_output_evidence.double()
        shrunken_error = (
            action_confidence.unsqueeze(1) * activation_error
            + (1.0 - action_confidence.unsqueeze(1))
            * structural.unsqueeze(2)
        ).clamp_min(0.0)
        action_risk = normalized_importance.pow(float(importance_power)).unsqueeze(
            1
        ) * shrunken_error
        # Full experts define zero compression risk exactly. Numerical noise in
        # an optional structural field must never penalize the full action.
        action_risk[:, -1, :] = 0.0

        converted.append(
            LayerIndependentActionTrace(
                name=trace.name,
                action_numerator=action_risk,
                raw_action_numerator=action_risk,
                reference_energy=torch.ones(categories, dtype=torch.float64),
                category_names=trace.category_names,
                action_costs=trace.action_costs,
                action_names=trace.action_names,
                top_k=trace.top_k,
            )
        )
        diagnostics[trace.name] = {
            "active_token_count": counts.long(),
            "importance_prior": prior,
            "importance_confidence": confidence,
            "observed_conditional_importance": observed_importance,
            "shrunken_conditional_importance": shrunken_importance,
            "normalized_importance": normalized_importance,
            "activation_relative_action_error": activation_error,
            "structural_relative_action_error": structural,
            "shrunken_relative_action_error": shrunken_error,
            "action_risk": action_risk,
        }
    return converted, diagnostics


def build_posterior_distortion_traces(
    traces: Sequence[LayerIndependentActionTrace],
    *,
    minimum_active_prompts: int = 2,
    epsilon: float = 1e-12,
) -> tuple[list[LayerIndependentActionTrace], dict[str, dict[str, torch.Tensor]]]:
    """Build prompt-clustered empirical-Bayes compression-risk traces.

    For each expert/action/category, the observed functional risk is the sum
    of routed output error over the sum of the layer MoE output energy. Prompt
    totals are retained as independent clusters, so uncertainty reflects
    variation between calibration prompts instead of treating correlated
    tokens as independent observations.

    Sparse estimates are shrunk in log-risk space toward a structural SVD
    prior.  The structural error is mapped into functional-risk units by a
    robust layer/action/category scale fitted from sufficiently observed
    experts.  The shrinkage weight follows the estimated sampling variance and
    empirical between-expert variance; it therefore needs no prompt-count
    pseudo-constant such as ``kappa``.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one streaming trace is required")
    if int(minimum_active_prompts) < 2:
        raise ValueError("minimum_active_prompts must be at least 2")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")

    converted = []
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    eps = float(epsilon)
    for trace in traces:
        if not isinstance(trace, LayerIndependentActionTrace):
            raise TypeError("posterior-distortion planning requires streaming traces")
        required = {
            "structural_action_error": trace.structural_action_error,
            "prompt_action_numerator": trace.prompt_action_numerator,
            "prompt_reference_energy": trace.prompt_reference_energy,
            "prompt_category_ids": trace.prompt_category_ids,
            "prompt_active_mask": trace.prompt_active_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{trace.name}: cached trace lacks {', '.join(missing)}; "
                "rebuild its streaming trace without --reuse-traces"
            )

        prompt_error = trace.prompt_action_numerator.double()
        prompt_energy = trace.prompt_reference_energy.double()
        prompt_categories = trace.prompt_category_ids.long()
        prompt_active = trace.prompt_active_mask.bool()
        structural = trace.structural_action_error.double()
        experts, actions, _ = map(int, prompt_error.shape)
        categories = len(trace.category_names)

        observed_risk = torch.zeros(
            experts, actions, categories, dtype=torch.float64
        )
        cluster_variance = torch.full_like(observed_risk, float("inf"))
        active_prompt_count = torch.zeros(
            experts, categories, dtype=torch.long
        )
        category_prompt_count = torch.zeros(categories, dtype=torch.long)

        for category in range(categories):
            prompt_ids = (prompt_categories == category).nonzero(as_tuple=True)[0]
            count = int(prompt_ids.numel())
            category_prompt_count[category] = count
            if count <= 0:
                continue
            x = prompt_error.index_select(2, prompt_ids)
            y = prompt_energy.index_select(0, prompt_ids)
            denominator = y.sum().clamp_min(eps)
            risk = x.sum(dim=2) / denominator
            observed_risk[:, :, category] = risk
            active_prompt_count[:, category] = prompt_active.index_select(
                1, prompt_ids
            ).sum(dim=1)
            if count > 1:
                residual = x - risk.unsqueeze(2) * y.reshape(1, 1, -1)
                cluster_variance[:, :, category] = (
                    float(count)
                    / float(count - 1)
                    * residual.square().sum(dim=2)
                    / denominator.square()
                )

        # Delta-method sampling variance in log-risk space.
        log_variance = cluster_variance / observed_risk.clamp_min(eps).square()
        enough_evidence = active_prompt_count >= int(minimum_active_prompts)
        enough_action_evidence = enough_evidence.unsqueeze(1) & (
            observed_risk > eps
        )
        log_variance = torch.where(
            enough_action_evidence,
            log_variance,
            torch.full_like(log_variance, float("inf")),
        )

        prior_scale = torch.ones(actions, categories, dtype=torch.float64)
        between_expert_variance = torch.zeros(
            actions, categories, dtype=torch.float64
        )
        calibrated_prior = torch.zeros_like(observed_risk)
        confidence = torch.zeros_like(observed_risk)
        posterior_risk = torch.zeros_like(observed_risk)

        for action in range(actions - 1):
            structural_action = structural[:, action].clamp_min(eps)
            for category in range(categories):
                evidence = enough_action_evidence[:, action, category]
                variance = log_variance[:, action, category]
                observed = observed_risk[:, action, category]
                reliable = evidence & torch.isfinite(variance)
                # No trustworthy functional observation leaves a neutral scale,
                # so the structural prior alone controls the ordering.
                scale = _geometric_median(
                    observed[reliable] / structural_action[reliable]
                )
                prior_scale[action, category] = scale
                prior = (scale * structural_action).clamp_min(eps)
                calibrated_prior[:, action, category] = prior

                (
                    posterior_risk[:, action, category],
                    confidence[:, action, category],
                    _,
                    between_expert_variance[action, category],
                ) = _log_space_posterior(
                    observed, variance, prior, evidence, epsilon=eps
                )

        # A full expert is the exact zero-risk reference action.
        calibrated_prior[:, -1, :] = 0.0
        confidence[:, -1, :] = 1.0
        posterior_risk[:, -1, :] = 0.0

        converted.append(
            LayerIndependentActionTrace(
                name=trace.name,
                action_numerator=posterior_risk,
                raw_action_numerator=posterior_risk,
                reference_energy=torch.ones(categories, dtype=torch.float64),
                category_names=trace.category_names,
                action_costs=trace.action_costs,
                action_names=trace.action_names,
                top_k=trace.top_k,
            )
        )
        diagnostics[trace.name] = {
            "category_prompt_count": category_prompt_count,
            "active_prompt_count": active_prompt_count,
            "observed_functional_risk": observed_risk,
            "cluster_sampling_variance": cluster_variance,
            "log_sampling_variance": log_variance,
            "structural_relative_action_error": structural,
            "structural_to_functional_scale": prior_scale,
            "calibrated_structural_prior": calibrated_prior,
            "between_expert_log_variance": between_expert_variance,
            "posterior_confidence": confidence,
            "posterior_action_risk": posterior_risk,
        }
    return converted, diagnostics


def build_factorized_risk_traces(
    traces: Sequence[LayerIndependentActionTrace],
    *,
    minimum_active_prompts: int = 2,
    epsilon: float = 1e-12,
    frequency_alpha: float = 0.5,
    uncertainty_z: float = 1.0,
    guard_removal_with_rerouting: bool = True,
) -> tuple[list[LayerIndependentActionTrace], dict[str, dict[str, torch.Tensor]]]:
    """Build conservative importance times action-degradation risk traces.

    The method deliberately separates how much an expert matters from how much
    a particular low-rank action damages it:

    ``importance = q_e^alpha E[g^2 ||f_e(h)||^2 | e active] /
                    E[||MoE(h)||^2]``

    ``compressibility = sum(g^2 ||f_e(h)-fhat_e(h)||^2) /
                         sum(g^2 ||f_e(h)||^2)``

    SVD risk is their product. ``frequency_alpha=1`` recovers expected routed
    energy, while smaller values temper frequency so a rare, high-contribution
    specialist is not overwhelmed by occupancy alone. Prompt-clustered
    empirical-Bayes shrinkage is applied to importance and degradation
    separately. A one-sided log-space uncertainty bound protects noisy
    measurements, and sparse positive observations are never shrunk below
    their measured value. Removal is additionally guarded by the measured
    top-k replacement and router-renormalization error when requested.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one streaming trace is required")
    if int(minimum_active_prompts) < 2:
        raise ValueError("minimum_active_prompts must be at least 2")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 <= float(frequency_alpha) <= 1):
        raise ValueError("frequency_alpha must be in [0, 1]")
    if float(uncertainty_z) < 0:
        raise ValueError("uncertainty_z must be nonnegative")

    eps = float(epsilon)
    converted = []
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for trace in traces:
        if not isinstance(trace, LayerIndependentActionTrace):
            raise TypeError("factorized-risk planning requires streaming traces")
        required = {
            "structural_action_error": trace.structural_action_error,
            "prompt_action_numerator": trace.prompt_action_numerator,
            "prompt_expert_output_energy": trace.prompt_expert_output_energy,
            "prompt_reference_energy": trace.prompt_reference_energy,
            "prompt_category_ids": trace.prompt_category_ids,
            "prompt_active_mask": trace.prompt_active_mask,
            "active_token_count": trace.active_token_count,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{trace.name}: cached trace lacks {', '.join(missing)}; "
                "rebuild its streaming trace without --reuse-traces"
            )

        prompt_error = trace.prompt_action_numerator.double()
        prompt_expert_energy = trace.prompt_expert_output_energy.double()
        prompt_reference_energy = trace.prompt_reference_energy.double()
        prompt_categories = trace.prompt_category_ids.long()
        prompt_active = trace.prompt_active_mask.bool()
        structural = trace.structural_action_error.double()
        experts, actions, _ = map(int, prompt_error.shape)
        categories = len(trace.category_names)

        observed_exposure = torch.zeros(experts, categories, dtype=torch.float64)
        exposure_sampling_variance = torch.full_like(
            observed_exposure, float("inf")
        )
        active_prompt_count = torch.zeros(
            experts, categories, dtype=torch.long
        )
        category_prompt_count = torch.zeros(categories, dtype=torch.long)
        observed_removal_risk = torch.zeros_like(observed_exposure)
        removal_sampling_variance = torch.full_like(
            observed_exposure, float("inf")
        )

        observed_compressibility = torch.zeros(
            experts, actions, categories, dtype=torch.float64
        )
        compressibility_sampling_variance = torch.full_like(
            observed_compressibility, float("inf")
        )

        for category in range(categories):
            prompt_ids = (prompt_categories == category).nonzero(as_tuple=True)[0]
            prompts = int(prompt_ids.numel())
            category_prompt_count[category] = prompts
            if prompts <= 0:
                continue
            z = prompt_expert_energy.index_select(1, prompt_ids)
            y = prompt_reference_energy.index_select(0, prompt_ids)
            x = prompt_error.index_select(2, prompt_ids)
            active = prompt_active.index_select(1, prompt_ids)
            active_counts = active.sum(dim=1)
            active_prompt_count[:, category] = active_counts

            reference_total = y.sum().clamp_min(eps)
            exposure = z.sum(dim=1) / reference_total
            observed_exposure[:, category] = exposure
            if prompts > 1:
                residual = z - exposure.unsqueeze(1) * y.unsqueeze(0)
                exposure_sampling_variance[:, category] = (
                    float(prompts)
                    / float(prompts - 1)
                    * residual.square().sum(dim=1)
                    / reference_total.square()
                )

            removal = x[:, 0, :].sum(dim=1) / reference_total
            observed_removal_risk[:, category] = removal
            if prompts > 1:
                removal_residual = (
                    x[:, 0, :] - removal.unsqueeze(1) * y.unsqueeze(0)
                )
                removal_sampling_variance[:, category] = (
                    float(prompts)
                    / float(prompts - 1)
                    * removal_residual.square().sum(dim=1)
                    / reference_total.square()
                )

            expert_total = z.sum(dim=1)
            compressibility = x.sum(dim=2) / expert_total.reshape(-1, 1).clamp_min(
                eps
            )
            observed_compressibility[:, :, category] = compressibility
            residual = x - compressibility.unsqueeze(2) * z.unsqueeze(1)
            valid_variance = active_counts > 1
            if bool(valid_variance.any()):
                scale = torch.zeros(experts, dtype=torch.float64)
                scale[valid_variance] = (
                    active_counts[valid_variance].double()
                    / (active_counts[valid_variance].double() - 1.0)
                    / expert_total[valid_variance].clamp_min(eps).square()
                )
                compressibility_sampling_variance[:, :, category] = (
                    residual.square().sum(dim=2) * scale.unsqueeze(1)
                )
                compressibility_sampling_variance[
                    ~valid_variance, :, category
                ] = float("inf")

        enough_prompts = active_prompt_count >= int(minimum_active_prompts)

        # Convert expected routed exposure (alpha=1) into
        # q**alpha * conditional second moment. Since
        # exposure = q * E[z^2 | active] / E[reference energy], multiplying by
        # q**(alpha-1) performs the conversion without counting frequency twice.
        active_tokens = trace.active_token_count.double()
        category_tokens = (
            active_tokens.sum(dim=0) / float(trace.top_k)
        ).clamp_min(1.0)
        activation_frequency = active_tokens / category_tokens.unsqueeze(0)
        frequency_scale = torch.zeros_like(activation_frequency)
        observed_frequency = activation_frequency > 0
        frequency_scale[observed_frequency] = activation_frequency[
            observed_frequency
        ].pow(float(frequency_alpha) - 1.0)
        observed_importance = observed_exposure * frequency_scale
        importance_sampling_variance = (
            exposure_sampling_variance * frequency_scale.square()
        )

        # Exposure posterior in log space. Its prior is a robust typical
        # expert exposure within the layer/category; no fixed pseudo-count is
        # needed because confidence comes from between-prompt variance.
        exposure_log_variance = importance_sampling_variance / observed_importance.clamp_min(
            eps
        ).square()
        exposure_evidence = enough_prompts & (observed_importance > eps)
        exposure_log_variance = torch.where(
            exposure_evidence,
            exposure_log_variance,
            torch.full_like(exposure_log_variance, float("inf")),
        )
        exposure_prior = torch.ones(categories, dtype=torch.float64)
        exposure_between_variance = torch.zeros(categories, dtype=torch.float64)
        exposure_confidence = torch.zeros_like(observed_exposure)
        posterior_exposure = torch.zeros_like(observed_exposure)
        exposure_posterior_log_variance = torch.zeros_like(observed_exposure)
        for category in range(categories):
            evidence = exposure_evidence[:, category]
            variance = exposure_log_variance[:, category]
            reliable = evidence & torch.isfinite(variance)
            # With no evidence at all, equal protection is safer than treating
            # every expert as zero exposure.
            prior = _geometric_median(
                observed_importance[reliable, category].clamp_min(eps),
                default=1.0 / max(experts, 1),
            )
            exposure_prior[category] = prior
            (
                posterior_exposure[:, category],
                exposure_confidence[:, category],
                exposure_posterior_log_variance[:, category],
                exposure_between_variance[category],
            ) = _log_space_posterior(
                observed_importance[:, category],
                variance,
                prior.expand(experts),
                evidence,
                epsilon=eps,
            )

        conservative_exposure = _conservative_upper(
            posterior_exposure,
            exposure_posterior_log_variance,
            uncertainty_z=uncertainty_z,
            epsilon=eps,
        )
        # A specialist seen in too few independent prompts has an uncertain
        # posterior, but a genuinely large observed contribution is direct
        # evidence of harm. Never shrink that positive observation downward.
        sparse_importance = (~exposure_evidence) & (observed_importance > eps)
        conservative_exposure = torch.where(
            sparse_importance,
            torch.maximum(conservative_exposure, observed_importance),
            conservative_exposure,
        )

        # Conditional SVD-error posterior. Removal is defined as complete loss
        # (Q=1); full is the exact reference (Q=0).
        compressibility_log_variance = (
            compressibility_sampling_variance
            / observed_compressibility.clamp_min(eps).square()
        )
        compressibility_evidence = enough_prompts.unsqueeze(1) & (
            observed_compressibility > eps
        )
        compressibility_log_variance = torch.where(
            compressibility_evidence,
            compressibility_log_variance,
            torch.full_like(compressibility_log_variance, float("inf")),
        )
        structural_scale = torch.ones(actions, categories, dtype=torch.float64)
        compressibility_between_variance = torch.zeros(
            actions, categories, dtype=torch.float64
        )
        structural_prior = torch.zeros_like(observed_compressibility)
        compressibility_confidence = torch.zeros_like(observed_compressibility)
        posterior_compressibility = torch.zeros_like(observed_compressibility)
        compressibility_posterior_log_variance = torch.zeros_like(
            observed_compressibility
        )
        posterior_compressibility[:, 0, :] = 1.0
        structural_prior[:, 0, :] = 1.0

        for action in range(1, actions - 1):
            structural_action = structural[:, action].clamp_min(eps)
            for category in range(categories):
                evidence = compressibility_evidence[:, action, category]
                variance = compressibility_log_variance[:, action, category]
                observed = observed_compressibility[:, action, category]
                reliable = evidence & torch.isfinite(variance)
                scale = _geometric_median(
                    observed[reliable] / structural_action[reliable]
                )
                structural_scale[action, category] = scale
                prior = (scale * structural_action).clamp_min(eps)
                structural_prior[:, action, category] = prior

                (
                    posterior_compressibility[:, action, category],
                    compressibility_confidence[:, action, category],
                    compressibility_posterior_log_variance[:, action, category],
                    compressibility_between_variance[action, category],
                ) = _log_space_posterior(
                    observed, variance, prior, evidence, epsilon=eps
                )

        structural_prior[:, -1, :] = 0.0
        compressibility_confidence[:, 0, :] = 1.0
        compressibility_confidence[:, -1, :] = 1.0
        posterior_compressibility[:, -1, :] = 0.0
        conservative_compressibility = _conservative_upper(
            posterior_compressibility,
            compressibility_posterior_log_variance,
            uncertainty_z=uncertainty_z,
            epsilon=eps,
        )
        sparse_compressibility = (~compressibility_evidence) & (
            observed_compressibility > eps
        )
        conservative_compressibility = torch.where(
            sparse_compressibility,
            torch.maximum(
                conservative_compressibility, observed_compressibility
            ),
            conservative_compressibility,
        )
        conservative_compressibility[:, 0, :] = 1.0
        conservative_compressibility[:, -1, :] = 0.0
        # A more expensive action that estimates worse than a cheaper action
        # is dominated. Project the posterior onto the monotone action order
        # so the adjacent-downgrade allocator cannot be trapped behind such a
        # noisy, irrational step. Removal has Q=1 and full has Q=0.
        for action in range(1, actions - 1):
            conservative_compressibility[:, action, :] = torch.minimum(
                conservative_compressibility[:, action, :],
                conservative_compressibility[:, action - 1, :],
            )
        action_risk = conservative_exposure.unsqueeze(1) * conservative_compressibility
        removal_upper = observed_removal_risk.clone()
        finite_removal_variance = torch.isfinite(removal_sampling_variance)
        removal_upper[finite_removal_variance] += float(uncertainty_z) * (
            removal_sampling_variance[finite_removal_variance]
            .clamp_min(0)
            .sqrt()
        )
        action_risk[:, 0, :] = conservative_exposure
        if guard_removal_with_rerouting:
            action_risk[:, 0, :] = torch.maximum(
                action_risk[:, 0, :], removal_upper
            )
        action_risk[:, -1, :] = 0.0

        converted.append(
            LayerIndependentActionTrace(
                name=trace.name,
                action_numerator=action_risk,
                raw_action_numerator=action_risk,
                reference_energy=torch.ones(categories, dtype=torch.float64),
                category_names=trace.category_names,
                action_costs=trace.action_costs,
                action_names=trace.action_names,
                top_k=trace.top_k,
            )
        )
        diagnostics[trace.name] = {
            "category_prompt_count": category_prompt_count,
            "active_prompt_count": active_prompt_count,
            "observed_exposure": observed_exposure,
            "activation_frequency": activation_frequency,
            "frequency_alpha": torch.tensor(float(frequency_alpha)),
            "observed_importance": observed_importance,
            "exposure_sampling_variance": exposure_sampling_variance,
            "exposure_log_variance": exposure_log_variance,
            "exposure_prior": exposure_prior,
            "exposure_between_expert_log_variance": exposure_between_variance,
            "exposure_confidence": exposure_confidence,
            "posterior_exposure": posterior_exposure,
            "exposure_posterior_log_variance": exposure_posterior_log_variance,
            "conservative_exposure": conservative_exposure,
            "observed_conditional_compressibility": observed_compressibility,
            "compressibility_sampling_variance": compressibility_sampling_variance,
            "compressibility_log_variance": compressibility_log_variance,
            "structural_relative_action_error": structural,
            "structural_to_compressibility_scale": structural_scale,
            "calibrated_structural_compressibility_prior": structural_prior,
            "compressibility_between_expert_log_variance": (
                compressibility_between_variance
            ),
            "compressibility_confidence": compressibility_confidence,
            "posterior_conditional_compressibility": posterior_compressibility,
            "compressibility_posterior_log_variance": (
                compressibility_posterior_log_variance
            ),
            "conservative_conditional_compressibility": (
                conservative_compressibility
            ),
            "observed_rerouting_removal_risk": observed_removal_risk,
            "rerouting_removal_upper_risk": removal_upper,
            "action_risk": action_risk,
        }
    return converted, diagnostics


def build_propagated_risk_traces(
    traces: Sequence[LayerIndependentActionTrace],
    propagation_factors: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    minimum_active_prompts: int = 2,
    epsilon: float = 1e-12,
    frequency_alpha: float = 0.5,
    uncertainty_z: float = 1.0,
    guard_removal_with_rerouting: bool = True,
) -> tuple[list[LayerIndependentActionTrace], dict[str, dict[str, torch.Tensor]]]:
    """Build the propagation-aware planner risk agreed for MoE compression.

    For expert ``e`` in layer ``l`` and category ``c`` the calibrated
    importance is

    ``I = q**alpha E[g ||f||^2 | active] / E[||MoE||^2]``.

    Low-rank action risk is ``G_svd * I * D_rank``, where ``D_rank`` is the
    prompt-calibrated conditional degradation measured with ``g^2`` routed
    residuals. Removal risk is ``G_remove * max(I, R_reroute)`` when the
    rerouting guard is enabled. Full precision has exactly zero risk.
    ``G`` is supplied by a separate end-to-end perturbation calibration and
    therefore does not conflate local expert importance with downstream layer
    sensitivity.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one streaming trace is required")
    if int(minimum_active_prompts) < 2:
        raise ValueError("minimum_active_prompts must be at least 2")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 <= float(frequency_alpha) <= 1):
        raise ValueError("frequency_alpha must be in [0, 1]")
    if float(uncertainty_z) < 0:
        raise ValueError("uncertainty_z must be nonnegative")

    # Reuse the existing calibrated g^2-weighted SVD degradation and exact
    # single-removal rerouting guard. Only the importance definition changes.
    _, degradation_diagnostics = build_factorized_risk_traces(
        traces,
        minimum_active_prompts=minimum_active_prompts,
        epsilon=epsilon,
        frequency_alpha=frequency_alpha,
        uncertainty_z=uncertainty_z,
        guard_removal_with_rerouting=guard_removal_with_rerouting,
    )

    eps = float(epsilon)
    converted = []
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for trace in traces:
        if not isinstance(trace, LayerIndependentActionTrace):
            raise TypeError("propagated-risk planning requires streaming traces")
        required = {
            "linear_routing_output_energy": trace.linear_routing_output_energy,
            "prompt_linear_routing_output_energy": (
                trace.prompt_linear_routing_output_energy
            ),
            "prompt_active_token_count": trace.prompt_active_token_count,
            "prompt_category_ids": trace.prompt_category_ids,
            "active_token_count": trace.active_token_count,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{trace.name}: cached trace lacks {', '.join(missing)}; "
                "rebuild its streaming trace without --reuse-traces"
            )
        if trace.name not in propagation_factors:
            raise ValueError(f"missing propagation factors for {trace.name}")

        experts = trace.experts
        categories = len(trace.category_names)
        layer_factors = propagation_factors[trace.name]
        try:
            svd_factor = torch.as_tensor(
                layer_factors["svd"], dtype=torch.float64
            ).reshape(-1)
            removal_factor = torch.as_tensor(
                layer_factors["remove"], dtype=torch.float64
            ).reshape(-1)
        except KeyError as error:
            raise ValueError(
                f"{trace.name}: propagation factors require 'svd' and 'remove'"
            ) from error
        for action_name, factor in (
            ("svd", svd_factor),
            ("remove", removal_factor),
        ):
            if tuple(factor.shape) != (categories,):
                raise ValueError(
                    f"{trace.name}: {action_name} propagation factor must have "
                    "one value per category"
                )
            if not bool(torch.isfinite(factor).all()) or bool((factor <= 0).any()):
                raise ValueError(
                    f"{trace.name}: {action_name} propagation factors must be "
                    "finite and positive"
                )

        active_tokens = trace.active_token_count.double()
        category_tokens = (
            active_tokens.sum(dim=0) / float(trace.top_k)
        ).clamp_min(1.0)
        activation_frequency = active_tokens / category_tokens.unsqueeze(0)
        conditional_linear_energy = trace.linear_routing_output_energy.double() / (
            active_tokens.clamp_min(1.0)
        )
        reference_mean_energy = trace.reference_energy.double() / category_tokens
        observed_importance = (
            activation_frequency.pow(float(frequency_alpha))
            * conditional_linear_energy
            / reference_mean_energy.clamp_min(eps).unsqueeze(0)
        )
        observed_importance = torch.where(
            active_tokens > 0,
            observed_importance,
            torch.zeros_like(observed_importance),
        )

        prompt_energy = trace.prompt_linear_routing_output_energy.double()
        prompt_counts = trace.prompt_active_token_count.double()
        prompt_categories = trace.prompt_category_ids.long()
        active_prompt_count = torch.zeros(
            experts, categories, dtype=torch.long
        )
        importance_sampling_variance = torch.full(
            (experts, categories), float("inf"), dtype=torch.float64
        )
        category_prompt_count = torch.zeros(categories, dtype=torch.long)
        for category in range(categories):
            prompt_ids = (prompt_categories == category).nonzero(as_tuple=True)[0]
            category_prompt_count[category] = int(prompt_ids.numel())
            if not int(prompt_ids.numel()):
                continue
            energy = prompt_energy.index_select(1, prompt_ids)
            counts = prompt_counts.index_select(1, prompt_ids)
            active = counts > 0
            active_prompt_count[:, category] = active.sum(dim=1)
            scale = (
                activation_frequency[:, category].pow(float(frequency_alpha))
                / reference_mean_energy[category].clamp_min(eps)
            )
            for expert in range(experts):
                selected_energy = energy[expert, active[expert]]
                selected_counts = counts[expert, active[expert]]
                sample_count = int(selected_counts.numel())
                if sample_count > 1:
                    total_count = selected_counts.sum().clamp_min(1.0)
                    conditional_mean = selected_energy.sum() / total_count
                    residual = (
                        selected_energy - conditional_mean * selected_counts
                    )
                    variance_of_mean = (
                        float(sample_count)
                        / float(sample_count - 1)
                        * residual.square().sum()
                        / total_count.square()
                    )
                    importance_sampling_variance[expert, category] = (
                        scale[expert].square() * variance_of_mean
                    )

        evidence = (
            active_prompt_count >= int(minimum_active_prompts)
        ) & (observed_importance > eps)
        log_variance = (
            importance_sampling_variance
            / observed_importance.clamp_min(eps).square()
        )
        log_variance = torch.where(
            evidence,
            log_variance,
            torch.full_like(log_variance, float("inf")),
        )
        importance_prior = torch.ones(categories, dtype=torch.float64)
        between_expert_variance = torch.zeros(categories, dtype=torch.float64)
        posterior_confidence = torch.zeros_like(observed_importance)
        posterior_importance = torch.zeros_like(observed_importance)
        posterior_log_variance = torch.zeros_like(observed_importance)
        for category in range(categories):
            category_evidence = evidence[:, category]
            variance = log_variance[:, category]
            reliable = category_evidence & torch.isfinite(variance)
            prior = _geometric_median(
                observed_importance[reliable, category].clamp_min(eps),
                default=1.0 / max(experts, 1),
            )
            importance_prior[category] = prior
            (
                posterior_importance[:, category],
                posterior_confidence[:, category],
                posterior_log_variance[:, category],
                between_expert_variance[category],
            ) = _log_space_posterior(
                observed_importance[:, category],
                variance,
                prior.expand(experts),
                category_evidence,
                epsilon=eps,
            )

        conservative_importance = _conservative_upper(
            posterior_importance,
            posterior_log_variance,
            uncertainty_z=uncertainty_z,
            epsilon=eps,
        )
        sparse_positive = (~evidence) & (observed_importance > eps)
        conservative_importance = torch.where(
            sparse_positive,
            torch.maximum(conservative_importance, observed_importance),
            conservative_importance,
        )

        base = degradation_diagnostics[trace.name]
        degradation = base["conservative_conditional_compressibility"]
        rerouting_upper = base["rerouting_removal_upper_risk"]
        action_risk = (
            conservative_importance.unsqueeze(1)
            * degradation
            * svd_factor.reshape(1, 1, -1)
        )
        removal_base = conservative_importance
        if guard_removal_with_rerouting:
            removal_base = torch.maximum(removal_base, rerouting_upper)
        action_risk[:, 0, :] = (
            removal_base * removal_factor.reshape(1, -1)
        )
        action_risk[:, -1, :] = 0.0

        converted.append(
            LayerIndependentActionTrace(
                name=trace.name,
                action_numerator=action_risk,
                raw_action_numerator=action_risk,
                reference_energy=torch.ones(categories, dtype=torch.float64),
                category_names=trace.category_names,
                action_costs=trace.action_costs,
                action_names=trace.action_names,
                top_k=trace.top_k,
            )
        )
        diagnostics[trace.name] = {
            **base,
            "category_prompt_count": category_prompt_count,
            "active_prompt_count": active_prompt_count,
            "activation_frequency": activation_frequency,
            "frequency_alpha": torch.tensor(float(frequency_alpha)),
            "conditional_linear_routing_output_energy": conditional_linear_energy,
            "reference_mean_energy": reference_mean_energy,
            "observed_importance": observed_importance,
            "importance_sampling_variance": importance_sampling_variance,
            "importance_log_variance": log_variance,
            "importance_prior": importance_prior,
            "importance_between_expert_log_variance": between_expert_variance,
            "importance_confidence": posterior_confidence,
            "posterior_importance": posterior_importance,
            "importance_posterior_log_variance": posterior_log_variance,
            "conservative_importance": conservative_importance,
            "svd_propagation_factor": svd_factor,
            "removal_propagation_factor": removal_factor,
            "action_risk": action_risk,
        }
    return converted, diagnostics


def _lognormal_tail_factor(
    frequency: torch.Tensor,
    log_dispersion: torch.Tensor,
    tail_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(CVaR_beta(D) / E[D | active], d/dsigma)`` in closed form.

    ``D`` is per-token damage relative to the layer's mean MoE output energy.
    It is exactly zero on the ``1 - q`` fraction of tokens that do not route to
    the expert, and is modelled as lognormal with log-dispersion ``sigma`` on
    the rest.  Writing ``u = min(beta / q, 1)`` for the share of active tokens
    that fall inside the worst ``beta`` fraction of the corpus,

    ``T = (q / beta) * Phi(sigma - Phi^{-1}(1 - u))``.

    Two limits make the scale interpretable.  When the expert is rarer than the
    tail (``q <= beta``) every active token is in the tail and ``T = q / beta``,
    so a rare expert is amplified by exactly ``1 / beta`` over its expected
    damage.  When damage is deterministic (``sigma = 0``) and the expert is
    always active, ``T = q = 1`` and the risk collapses back to the mean.  The
    derivative is zero in the saturated branch because the tail then contains
    all of the active mass whatever shape it has.
    """

    beta = float(tail_fraction)
    q = frequency.double().clamp(min=0.0, max=1.0)
    sigma = log_dispersion.double().clamp_min(0.0)
    saturated = q <= beta
    share = (beta / q.clamp_min(beta)).clamp(max=1.0)
    residual = (1.0 - share).clamp(min=1e-12, max=1.0 - 1e-12)
    argument = sigma - torch.special.ndtri(residual)
    scale = q / beta
    factor = torch.where(saturated, scale, scale * torch.special.ndtr(argument))
    gradient = torch.where(
        saturated,
        torch.zeros_like(scale),
        scale * torch.exp(-0.5 * argument.square()) / sqrt(2.0 * pi),
    )
    return factor, gradient


def fallback_depth_weights(
    retained_fraction: float, depths: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distribute removal probability over surviving-fallback depths.

    A single-removal measurement assumes the top-(k+1) expert survives to absorb
    the traffic. Under a plan that removes most of a layer it usually does not,
    and the true fallback sits deeper in the router shortlist with a smaller
    logit, so the damage is larger. Modelling survival as independent with
    probability ``rho`` (the layer's retained fraction) gives a geometric law
    over the depth of the first surviving candidate:

    ``pi_d = rho * (1 - rho)**d`` for ``d = 0 .. depths-1``

    with the leftover mass ``(1 - rho)**depths`` assigned to an absorbing state
    in which no measured candidate survives. That state is scored with the
    expert's own routed energy, the no-compensation ceiling of the bracket
    ``measured <= true <= exposure``.

    Returns ``(pi, absorbing_mass)``. ``rho = 1`` puts all mass on depth zero and
    exactly reproduces the single-removal measurement.

    Independence is an approximation in the optimistic direction: removals are
    chosen to be low risk, which correlates with low routing weight, and the
    near-shortlist candidates carry relatively high weight, so they survive more
    often than ``rho`` suggests. The absorbing ceiling is what keeps the estimate
    from running away when it does not hold.

    Measured on Qwen3-30B-A3B (128 experts, top-8), the depth response is flat:
    median damage at depth 5 is 0.98x the damage at depth 0, and only 14% of
    experts are worse at depth 5 than at depth 0. The absorbing ceiling sits
    within 9% of depth zero. The mechanism is visible in the removal delta

    ``[(g_e - g_r) y - g_e f_e + g_r f_r] / (1 - g_e + g_r)``

    with ``g_e`` about 0.12 under top-8 routing. A *closer* fallback has a larger
    ``g_r``, and because specialized experts are close to mutually orthogonal,
    ``+ g_r f_r`` does not cancel the lost ``- g_e f_e`` -- it injects its own
    wrong-direction output. Substituting a nearer expert is therefore no better
    than substituting a further one, and both are close to substituting nothing.

    So on this model retention-aware removal is a no-op and the planner leaves it
    off. Architectures with far larger per-expert routing weight (a top-2-of-8
    model has ``g_e`` near 0.5) should behave differently, which is why the
    machinery and its diagnostic remain available.
    """

    if not (0 < float(retained_fraction) <= 1):
        raise ValueError("retained_fraction must be in (0, 1]")
    if int(depths) < 1:
        raise ValueError("depths must be positive")
    rho = float(retained_fraction)
    index = torch.arange(int(depths), dtype=torch.float64)
    weights = rho * torch.pow(1.0 - rho, index)
    absorbing = torch.tensor((1.0 - rho) ** int(depths), dtype=torch.float64)
    return weights, absorbing


def _mixed_removal_statistics(
    trace: "LayerIndependentActionTrace",
    retained_fraction: float,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend depth-resolved removal statistics for one retained fraction.

    Returns ``(numerator, prompt_numerator, log_sum, log_square_sum, count)``
    shaped like the action-zero slices they replace.

    Numerators mix exactly, including the absorbing state, because the expert's
    routed output energy is already measured in the same units. The log moments
    mix only over measured depths, renormalized: the absorbing state has an exact
    mean but no per-token log moments of its own, and it only shapes the tail of
    a minority of the mass, so inheriting the measured shape is the smaller
    approximation.
    """

    depths = int(trace.removal_depth_numerator.shape[1])
    weights, absorbing = fallback_depth_weights(retained_fraction, depths)
    shaped = weights.reshape(1, -1, 1)

    numerator = (trace.removal_depth_numerator * shaped).sum(dim=1)
    numerator = numerator + absorbing * trace.expert_output_energy
    prompt_numerator = (
        trace.removal_depth_prompt_numerator * shaped
    ).sum(dim=1) + absorbing * trace.prompt_expert_output_energy

    counts = trace.removal_depth_positive_count.double()
    observed = counts > 0
    share = torch.where(observed, weights.reshape(1, -1, 1), torch.zeros_like(counts))
    total_share = share.sum(dim=1, keepdim=True).clamp_min(float(epsilon))
    share = share / total_share
    mean_log = trace.removal_depth_log_sum / counts.clamp_min(1.0)
    mean_log_square = trace.removal_depth_log_square_sum / counts.clamp_min(1.0)
    # Report the mixture on the depth-zero token count so downstream dispersion
    # confidence still reflects how many tokens were actually observed.
    count = trace.removal_depth_positive_count[:, 0, :]
    scale = count.double()
    log_sum = (share * mean_log).sum(dim=1) * scale
    log_square_sum = (share * mean_log_square).sum(dim=1) * scale
    return numerator, prompt_numerator, log_sum, log_square_sum, count


def build_distortion_risk_traces(
    traces: Sequence[LayerIndependentActionTrace],
    *,
    propagation_factors: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    tail_fraction: float = 0.01,
    tail_weight: float = 0.5,
    minimum_active_prompts: int = 2,
    uncertainty_z: float = 1.0,
    frequency_floor_tokens: float = 1.0,
    frequency_temper: float = 0.0,
    margin_proxy_weight: float = 0.0,
    retained_fraction: Mapping[str, float] | float | None = None,
    tier_price: float = 1.0,
    damage_reference: str = "layer_mean",
    epsilon: float = 1e-12,
) -> tuple[list[LayerIndependentActionTrace], dict[str, dict[str, torch.Tensor]]]:
    """Score every expert action with a mean-CVaR distortion risk measure.

    The target quantity is the distortion of the whole compressed model, and
    each action contributes to it through the per-token relative damage

    ``d_t = g_t^2 ||f_e(h_t) - fhat_e(h_t)||^2 / E_t[||MoE(h_t)||^2]``,

    which is measured directly for every calibration token.  A trace built with
    ``damage_unit="amplitude"`` measures ``g_t ||f_e - fhat_e|| / E_t[||MoE||]``
    instead; every step below is unit-agnostic except the structural prior,
    which is converted to match.  ``damage_reference="none"`` drops the division
    by the layer's mean MoE output and prices absolute damage; everything below
    is scale-equivariant within a layer, so that only changes how layers trade
    off against each other.  This planner
    summarises that distribution with

    ``rho = Ebar * [(1 - lam) * q + lam * T_beta(q, sigma)]``

    where ``Ebar`` is the conditional mean damage, ``q`` the activation
    frequency, ``sigma`` the conditional log-dispersion, and ``T_beta`` the
    lognormal tail factor above.  ``lam = 0`` recovers ``q * Ebar``, which is
    exactly the expert's share of relative routed damage, so the risk stays in
    the relative-energy units that a propagation gain is calibrated against.

    ``margin_proxy_weight`` optionally mixes this energy risk with an
    amplitude risk derived from the same lognormal posterior. Since a logit
    margin changes to first order with the residual norm rather than its
    squared norm, that component uses

    ``E[sqrt(d) | active] = sqrt(E[d | active]) * exp(-sigma^2 / 8)``.

    It is robustly rescaled per action and category into the energy-risk units
    before mixing, so the weight changes rankings rather than the overall
    action-class price. Zero preserves the original equation exactly.

    This deliberately replaces the ``q**alpha`` tempering used by the earlier
    planners.  That exponent could only trade a rare specialist against a
    frequent workhorse, because it has a single knob acting on frequency alone.
    Here frequency enters ``rho`` monotonically -- a more frequently routed
    expert of equal conditional importance is always scored as riskier to
    compress -- while rare experts are protected by two separate channels: the
    ``1 / beta`` tail amplification once ``q`` drops below ``beta``, and the
    conditional dispersion ``sigma``, which raises the risk of an expert whose
    damage is catastrophic on a handful of tokens even when its mean damage is
    unremarkable.

    Removal is not special-cased.  It is action zero, scored from its own
    measured per-token damage, which already includes exact top-k replacement
    and router renormalization.  Sparse experts fall back to a structural
    low-rank prior calibrated into damage units, and the mean and tail shape
    receive a single joint log-space uncertainty inflation rather than one
    inflation each.  ``rho`` is positively homogeneous and subadditive, so the
    additive sum the allocator forms over experts and layers is an upper bound
    on the plan's joint risk rather than merely an approximation.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one streaming trace is required")
    if not (0 < float(tail_fraction) <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")
    if not (0 <= float(tail_weight) <= 1):
        raise ValueError("tail_weight must be in [0, 1]")
    if int(minimum_active_prompts) < 2:
        raise ValueError("minimum_active_prompts must be at least 2")
    if float(uncertainty_z) < 0:
        raise ValueError("uncertainty_z must be nonnegative")
    if float(frequency_floor_tokens) < 0:
        raise ValueError("frequency_floor_tokens must be nonnegative")
    if not (0 <= float(frequency_temper) <= 1):
        raise ValueError("frequency_temper must be in [0, 1]")
    if float(margin_proxy_weight) < 0:
        raise ValueError("margin_proxy_weight must be nonnegative")
    if float(epsilon) <= 0:
        raise ValueError("epsilon must be positive")
    if str(damage_reference) not in ("layer_mean", "none"):
        raise ValueError(
            "damage_reference must be 'layer_mean' or 'none', "
            f"got {damage_reference!r}"
        )

    eps = float(epsilon)
    lam = float(tail_weight)
    # Once beta exceeds every expert's routing frequency the tail term saturates
    # for all of them, collapsing to lam * q / beta -- a rescaled expectation
    # that ranks experts exactly as tail_weight=0 would. The mean frequency is
    # top_k / experts, so beta has to sit well below that to discriminate.
    mean_frequency = float(traces[0].top_k) / max(traces[0].experts, 1)
    if lam > 0 and float(tail_fraction) >= mean_frequency:
        logger.warning(
            "tail_fraction=%.4g is at or above the mean routing frequency "
            "%.4g (top_k/experts); most experts saturate the tail term, which "
            "then ranks them the same way tail_weight=0 does. Use a "
            "tail_fraction well below %.4g.",
            float(tail_fraction),
            mean_frequency,
            mean_frequency,
        )
    converted = []
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for trace in traces:
        if not isinstance(trace, LayerIndependentActionTrace):
            raise TypeError("distortion-risk planning requires streaming traces")
        required = {
            "expert_output_energy": trace.expert_output_energy,
            "active_token_count": trace.active_token_count,
            "structural_action_error": trace.structural_action_error,
            "prompt_action_numerator": trace.prompt_action_numerator,
            "prompt_active_token_count": trace.prompt_active_token_count,
            "prompt_category_ids": trace.prompt_category_ids,
            "damage_log_sum": trace.damage_log_sum,
            "damage_log_square_sum": trace.damage_log_square_sum,
            "damage_positive_count": trace.damage_positive_count,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{trace.name}: cached trace lacks {', '.join(missing)}; "
                "rebuild its streaming trace without --reuse-traces"
            )
        if trace.damage_unit != getattr(traces[0], "damage_unit", "energy"):
            raise ValueError(
                "all layers must share one damage_unit; "
                f"{trace.name} is {trace.damage_unit!r}"
            )
        amplitude = trace.damage_unit == "amplitude"
        if amplitude and float(margin_proxy_weight) > 0:
            raise ValueError(
                f"{trace.name}: margin_proxy_weight mixes in sqrt(damage), but "
                "this trace's damage is already an amplitude"
            )

        experts = trace.experts
        actions = trace.actions
        categories = len(trace.category_names)

        svd_factor = torch.ones(categories, dtype=torch.float64)
        removal_factor = torch.ones(categories, dtype=torch.float64)
        # The exchange rate between "delete this expert" and "truncate it" is set
        # here and nowhere else. The propagation gains that normally set it come
        # from a RANDOM-direction probe, so they price the two classes by error
        # MAGNITUDE alone and cannot see that a removal's error is absorbed by
        # router renormalisation while a truncation's is injected at full routing
        # weight. Measured on this model, moving routing mass from removal into
        # tiers costs 0.038 human-eval points per 1% of mass (r = -0.82), and REAP
        # (removal only) beats this planner by 5.18 / 6.71 points at budget
        # 0.40 / 0.35 -- exactly where tier spending peaks. ``tier_price`` is the
        # correction: > 1 makes truncation dearer relative to removal. 1.0 is an
        # exact no-op. The monotone projection below caps tier risk at the removal
        # risk, so a large value saturates into "removal strictly dominates"
        # rather than diverging.
        price = float(tier_price)
        if price <= 0:
            raise ValueError(f"tier_price must be positive, got {price}")
        # This guard was lost, leaving the whole load unreachable behind the
        # raise above: svd_factor/removal_factor stayed 1.0, --propagation-path
        # was silently ignored, and the run still logged "propagation=<path>".
        if propagation_factors is not None:
            if trace.name not in propagation_factors:
                raise ValueError(f"missing propagation factors for {trace.name}")
            layer_factors = propagation_factors[trace.name]
            try:
                svd_factor = torch.as_tensor(
                    layer_factors["svd"], dtype=torch.float64
                ).reshape(-1)
                removal_factor = torch.as_tensor(
                    layer_factors["remove"], dtype=torch.float64
                ).reshape(-1)
            except KeyError as error:
                raise ValueError(
                    f"{trace.name}: propagation factors require 'svd' and 'remove'"
                ) from error
            for action_name, factor in (
                ("svd", svd_factor),
                ("remove", removal_factor),
            ):
                if tuple(factor.shape) != (categories,):
                    raise ValueError(
                        f"{trace.name}: {action_name} propagation factor must have "
                        "one value per category"
                    )
                if not bool(torch.isfinite(factor).all()) or bool((factor <= 0).any()):
                    raise ValueError(
                        f"{trace.name}: {action_name} propagation factors must be "
                        "finite and positive"
                    )

        active_tokens = trace.active_token_count.double()
        category_tokens = (
            active_tokens.sum(dim=0) / float(trace.top_k)
        ).clamp_min(1.0)
        if str(damage_reference) == "none":
            # Absolute damage. Every step below is scale-equivariant within a
            # layer, so this multiplies the layer's risk by its mean MoE output
            # and changes only the exchange rate between layers.
            reference_mean_energy = torch.ones_like(category_tokens)
        else:
            reference_mean_energy = (
                trace.reference_energy.double() / category_tokens
            ).clamp_min(eps)
        # An expert never routed within the calibration corpus is not proven
        # harmless; its frequency is only bounded above by one token in the
        # sample. Flooring q there keeps unobserved experts on the same scale as
        # the rarest observed ones instead of scoring them as free to remove.
        frequency_floor = (
            float(frequency_floor_tokens) / category_tokens
        ).unsqueeze(0).expand(experts, -1)
        frequency = torch.maximum(
            active_tokens / category_tokens.unsqueeze(0), frequency_floor
        ).clamp_max(1.0)

        # Retention-aware removal: replace the action-zero statistics with a
        # fallback-depth mixture before anything else consumes them, so the
        # uncertainty, shrinkage and tail machinery all see the corrected values.
        damage_numerator = trace.raw_action_numerator.double().clone()
        prompt_numerator = trace.prompt_action_numerator.double().clone()
        damage_log_sum = trace.damage_log_sum.double().clone()
        damage_log_square_sum = trace.damage_log_square_sum.double().clone()
        damage_positive_count = trace.damage_positive_count.clone()
        layer_retained_fraction = None
        if retained_fraction is not None:
            layer_retained_fraction = float(
                retained_fraction[trace.name]
                if isinstance(retained_fraction, Mapping)
                else retained_fraction
            )
            if trace.removal_depth_numerator is None:
                raise ValueError(
                    f"{trace.name}: retention-aware removal needs depth-resolved "
                    "removal statistics; rebuild its streaming trace with "
                    "--removal-fallback-depths above 1"
                )
            if trace.prompt_expert_output_energy is None:
                raise ValueError(
                    f"{trace.name}: retention-aware removal needs "
                    "prompt_expert_output_energy"
                )
            (
                damage_numerator[:, 0, :],
                prompt_numerator[:, 0, :],
                damage_log_sum[:, 0, :],
                damage_log_square_sum[:, 0, :],
                damage_positive_count[:, 0, :],
            ) = _mixed_removal_statistics(
                trace, layer_retained_fraction, epsilon=eps
            )

        token_scale = active_tokens.clamp_min(1.0) * reference_mean_energy.unsqueeze(0)
        observed_damage = damage_numerator / token_scale.unsqueeze(1)
        conditional_expert_energy = (
            trace.expert_output_energy.double() / token_scale
        )
        for category in range(categories):
            observed = active_tokens[:, category] > 0
            values = conditional_expert_energy[observed, category]
            values = values[values > 0]
            typical = (
                torch.quantile(values.log(), 0.5).exp()
                if int(values.numel())
                else torch.tensor(1.0, dtype=torch.float64)
            )
            conditional_expert_energy[~observed, category] = typical

        positive_count = damage_positive_count
        positive = positive_count.double().clamp_min(1.0)
        mean_log_damage = damage_log_sum / positive
        observed_dispersion = (
            damage_log_square_sum / positive - mean_log_damage.square()
        ).clamp_min(0.0).sqrt()
        has_dispersion = positive_count >= 2
        observed_dispersion = torch.where(
            has_dispersion, observed_dispersion, torch.zeros_like(observed_dispersion)
        )

        prompt_error = prompt_numerator
        prompt_counts = trace.prompt_active_token_count.double()
        prompt_categories = trace.prompt_category_ids.long()
        active_prompt_count = torch.zeros(experts, categories, dtype=torch.long)
        category_prompt_count = torch.zeros(categories, dtype=torch.long)
        damage_sampling_variance = torch.full(
            (experts, actions, categories), float("inf"), dtype=torch.float64
        )
        for category in range(categories):
            prompt_ids = (prompt_categories == category).nonzero(as_tuple=True)[0]
            prompts = int(prompt_ids.numel())
            category_prompt_count[category] = prompts
            if prompts <= 0:
                continue
            x = prompt_error.index_select(2, prompt_ids)
            counts = prompt_counts.index_select(1, prompt_ids)
            clusters = (counts > 0).sum(dim=1)
            active_prompt_count[:, category] = clusters
            # Ratio-estimator cluster variance: prompts, not tokens, are the
            # independent sampling unit, so correlated tokens within one prompt
            # cannot inflate the apparent evidence.
            denominator = (
                counts.sum(dim=1).clamp_min(1.0) * reference_mean_energy[category]
            )
            residual = x - observed_damage[:, :, category].unsqueeze(2) * (
                reference_mean_energy[category] * counts.unsqueeze(1)
            )
            usable = clusters > 1
            if not bool(usable.any()):
                continue
            scale = torch.zeros(experts, dtype=torch.float64)
            scale[usable] = (
                clusters[usable].double()
                / (clusters[usable].double() - 1.0)
                / denominator[usable].square()
            )
            damage_sampling_variance[:, :, category] = (
                residual.square().sum(dim=2) * scale.unsqueeze(1)
            )
            damage_sampling_variance[~usable, :, category] = float("inf")

        enough_prompts = active_prompt_count >= int(minimum_active_prompts)
        evidence = enough_prompts.unsqueeze(1) & (observed_damage > eps)
        log_variance = (
            damage_sampling_variance / observed_damage.clamp_min(eps).square()
        )
        log_variance = torch.where(
            evidence, log_variance, torch.full_like(log_variance, float("inf"))
        )

        structural = trace.structural_action_error.double()
        if amplitude:
            # A relative squared Frobenius error; the prior stands in for the
            # damage, so it has to be in the damage's units.
            structural = structural.clamp_min(0).sqrt()
        structural_prior = conditional_expert_energy.unsqueeze(
            1
        ) * structural.unsqueeze(2)
        structural_scale = torch.ones(actions, categories, dtype=torch.float64)
        between_expert_variance = torch.zeros(actions, categories, dtype=torch.float64)
        calibrated_prior = torch.zeros_like(observed_damage)
        damage_confidence = torch.zeros_like(observed_damage)
        posterior_damage = torch.zeros_like(observed_damage)
        posterior_log_variance = torch.zeros_like(observed_damage)
        for action in range(actions - 1):
            base = structural_prior[:, action, :].clamp_min(eps)
            for category in range(categories):
                action_evidence = evidence[:, action, category]
                variance = log_variance[:, action, category]
                observed = observed_damage[:, action, category]
                reliable = action_evidence & torch.isfinite(variance)
                # A multiplicative median keeps the prior on the observed damage
                # scale without letting a few extreme experts set it.
                scale = _geometric_median(
                    observed[reliable] / base[reliable, category]
                )
                structural_scale[action, category] = scale
                prior = (scale * base[:, category]).clamp_min(eps)
                calibrated_prior[:, action, category] = prior

                (
                    posterior_damage[:, action, category],
                    damage_confidence[:, action, category],
                    posterior_log_variance[:, action, category],
                    between_expert_variance[action, category],
                ) = _log_space_posterior(
                    observed, variance, prior, action_evidence, epsilon=eps
                )

        # A log-scale estimated from m independent observations has variance
        # sigma^2 / (2m); prompts are again the independent unit.
        clusters = active_prompt_count.double().unsqueeze(1).expand_as(
            observed_dispersion
        )
        dispersion_sampling_variance = torch.where(
            has_dispersion & (clusters > 1),
            observed_dispersion.square() / (2.0 * clusters.clamp_min(2.0)),
            torch.full_like(observed_dispersion, float("inf")),
        )
        dispersion_prior = torch.zeros(actions, categories, dtype=torch.float64)
        dispersion_confidence = torch.zeros_like(observed_dispersion)
        posterior_dispersion = torch.zeros_like(observed_dispersion)
        dispersion_posterior_variance = torch.zeros_like(observed_dispersion)
        for action in range(actions - 1):
            for category in range(categories):
                variance = dispersion_sampling_variance[:, action, category]
                observed = observed_dispersion[:, action, category]
                # A dispersion is a nonnegative scale that can legitimately be
                # zero, so unlike every multiplicative quantity here it is shrunk
                # on the linear scale toward the layer's typical dispersion.
                usable = has_dispersion[:, action, category] & torch.isfinite(variance)
                reliable = observed[usable]
                prior = (
                    reliable.median()
                    if int(reliable.numel())
                    else torch.tensor(0.0, dtype=torch.float64)
                )
                dispersion_prior[action, category] = prior
                (
                    dispersion_confidence[:, action, category],
                    dispersion_posterior_variance[:, action, category],
                    _,
                ) = _empirical_bayes_weights(observed, variance, usable, epsilon=eps)
                weight = dispersion_confidence[:, action, category]
                posterior_dispersion[:, action, category] = (
                    weight * observed + (1.0 - weight) * prior
                )

        expanded_frequency = frequency.unsqueeze(1).expand_as(posterior_damage)
        tail_factor, tail_gradient = _lognormal_tail_factor(
            expanded_frequency, posterior_dispersion, tail_fraction
        )
        spectral_weight = (1.0 - lam) * expanded_frequency + lam * tail_factor
        # Frequency tempering. The untempered weight makes rho a *total exposure*
        # measure -- it ranks experts almost exactly as REAP's `reap_sum` does
        # (Spearman +0.975) rather than as its per-token `reap_mean` (+0.751),
        # so the allocator deletes rare specialists in favour of frequent,
        # substitutable workhorses. Dividing by q**temper interpolates toward the
        # conditional damage: 0 keeps the exposure-weighted measure unchanged,
        # 1 scores an expert purely on how much damage it does per token it is
        # routed to. The tail term is tempered with it so the two channels stay
        # on one scale.
        temper = float(frequency_temper)
        if temper > 0:
            spectral_weight = spectral_weight / expanded_frequency.clamp_min(
                eps
            ).pow(temper)
        risk = posterior_damage * spectral_weight
        amplitude_risk = torch.zeros_like(risk)
        amplitude_to_energy_scale = torch.ones(
            actions, categories, dtype=torch.float64
        )
        margin_weight = float(margin_proxy_weight)
        if margin_weight > 0:
            # If d is lognormal with arithmetic mean mu and log-dispersion
            # sigma, E[sqrt(d)] = sqrt(mu) exp(-sigma^2/8). sqrt(d) is the
            # component that is first-order in a downstream margin change.
            conditional_amplitude = posterior_damage.clamp_min(0).sqrt() * torch.exp(
                -posterior_dispersion.square() / 8.0
            )
            amplitude_tail, _ = _lognormal_tail_factor(
                expanded_frequency, posterior_dispersion / 2.0, tail_fraction
            )
            amplitude_spectral_weight = (
                (1.0 - lam) * expanded_frequency + lam * amplitude_tail
            )
            if temper > 0:
                amplitude_spectral_weight = amplitude_spectral_weight / (
                    expanded_frequency.clamp_min(eps).pow(temper)
                )
            amplitude_risk = conditional_amplitude * amplitude_spectral_weight
            for action in range(actions - 1):
                for category in range(categories):
                    amp = amplitude_risk[:, action, category]
                    energy = risk[:, action, category]
                    usable = (amp > eps) & (energy > eps)
                    scale = _geometric_median(energy[usable] / amp[usable])
                    amplitude_to_energy_scale[action, category] = scale
                    amplitude_risk[:, action, category] *= scale
            risk = (risk + margin_weight * amplitude_risk) / (1.0 + margin_weight)

        # One combined inflation, not one per factor: the conditional mean and
        # the tail shape enter the same scalar, so the delta method gives their
        # joint log-space uncertainty. Inflating them separately would compound
        # for low-rank actions while leaving removal inflated only once, which
        # silently biases the choice between action types.
        dispersion_sensitivity = lam * tail_gradient / spectral_weight.clamp_min(eps)
        risk_log_variance = (
            posterior_log_variance
            + dispersion_sensitivity.square()
            * dispersion_posterior_variance.clamp_min(0)
        )
        # Multiplicative rather than via _conservative_upper: that helper floors
        # its input at epsilon, and an action with exactly zero measured damage
        # must stay exactly zero rather than acquiring a floor's worth of risk.
        conservative_risk = risk * torch.exp(
            (
                float(uncertainty_z) * risk_log_variance.clamp_min(0).sqrt()
            ).clamp_max(_MAX_LOG_RISK)
        )

        # Too few prompts to trust the posterior is not evidence of safety: a
        # large directly measured damage stands on its own.
        observed_tail, _ = _lognormal_tail_factor(
            expanded_frequency, observed_dispersion, tail_fraction
        )
        observed_risk = observed_damage * (
            (1.0 - lam) * expanded_frequency + lam * observed_tail
        )
        sparse_positive = (~evidence) & (observed_damage > eps)
        conservative_risk = torch.where(
            sparse_positive,
            torch.maximum(conservative_risk, observed_risk),
            conservative_risk,
        )

        conservative_risk[:, 0, :] = (
            conservative_risk[:, 0, :] * removal_factor.reshape(1, -1)
        )
        if actions > 2:
            conservative_risk[:, 1:-1, :] = conservative_risk[:, 1:-1, :] * (
                (svd_factor * price).reshape(1, 1, -1)
            )
        # A full expert is the exact zero-risk reference action.
        conservative_risk[:, -1, :] = 0.0
        # A more expensive action that estimates worse than a cheaper one is
        # dominated; projecting onto the monotone action order keeps the
        # adjacent-downgrade allocator from being trapped behind such a step.
        for action in range(1, actions - 1):
            conservative_risk[:, action, :] = torch.minimum(
                conservative_risk[:, action, :],
                conservative_risk[:, action - 1, :],
            )

        converted.append(
            LayerIndependentActionTrace(
                name=trace.name,
                action_numerator=conservative_risk,
                raw_action_numerator=conservative_risk,
                reference_energy=torch.ones(categories, dtype=torch.float64),
                category_names=trace.category_names,
                action_costs=trace.action_costs,
                action_names=trace.action_names,
                top_k=trace.top_k,
            )
        )
        diagnostics[trace.name] = {
            "tail_fraction": torch.tensor(float(tail_fraction)),
            "tail_weight": torch.tensor(lam),
            "frequency_temper": torch.tensor(float(frequency_temper)),
            "margin_proxy_weight": torch.tensor(float(margin_proxy_weight)),
            "retained_fraction": torch.tensor(
                float("nan")
                if layer_retained_fraction is None
                else layer_retained_fraction
            ),
            "category_prompt_count": category_prompt_count,
            "active_prompt_count": active_prompt_count,
            "activation_frequency": frequency,
            "conditional_expert_energy": conditional_expert_energy,
            "observed_conditional_damage": observed_damage,
            "damage_sampling_variance": damage_sampling_variance,
            "damage_log_variance": log_variance,
            "structural_relative_action_error": structural,
            "structural_to_damage_scale": structural_scale,
            "calibrated_structural_prior": calibrated_prior,
            "damage_between_expert_log_variance": between_expert_variance,
            "damage_confidence": damage_confidence,
            "posterior_conditional_damage": posterior_damage,
            "damage_posterior_log_variance": posterior_log_variance,
            "observed_log_dispersion": observed_dispersion,
            "dispersion_sampling_variance": dispersion_sampling_variance,
            "dispersion_prior": dispersion_prior,
            "dispersion_confidence": dispersion_confidence,
            "posterior_log_dispersion": posterior_dispersion,
            "dispersion_posterior_variance": dispersion_posterior_variance,
            "tail_factor": tail_factor,
            "spectral_weight": spectral_weight,
            "amplitude_risk": amplitude_risk,
            "amplitude_to_energy_scale": amplitude_to_energy_scale,
            "risk_log_variance": risk_log_variance,
            "svd_propagation_factor": svd_factor,
            "removal_propagation_factor": removal_factor,
            "action_risk": conservative_risk,
        }
    return converted, diagnostics


def build_mean_loss_traces(
    traces: Sequence[LayerIndependentActionTrace],
    *,
    propagation_factors: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    damage_reference: str = "layer_mean",
    frequency_power: float = 0.0,
    epsilon: float = 1e-12,
    tier_price: float = 1.0,
    tier_price_scale: Mapping[str, Sequence[float]] | None = None,
) -> tuple[list[LayerIndependentActionTrace], dict[str, dict[str, torch.Tensor]]]:
    """Price every expert action by its mean contribution loss per activation.

    For expert ``e``, action ``a`` and category ``c``::

        price = w * (1 / n_ec) * sum_{t routed to e in c} d_t(e, a) / R_c

    ``d_t`` is the per-token damage in the trace's unit (``g ||f - fhat||`` for
    amplitude traces, its square for energy traces); removal damage already
    includes rerouting to the next expert with renormalized weights. ``n_ec`` is
    the expert's active token count, ``R_c`` the layer's mean MoE output in the
    same unit (``damage_reference="layer_mean"``) or 1 (``"none"``), and ``w``
    the layer's propagation factor for the action class.

    ``frequency_power`` (alpha) interpolates between the mean per activation and
    the TOTAL damage the expert does::

        price *= (n_ec / mean_e n_ec) ** alpha

    alpha=0 is the pure mean (an expert that fires 1,292 times and one that fires
    34,060 times are priced alike when their per-activation loss matches);
    alpha=1 is total damage, which prices the busy one 26x higher. The
    normalization by the layer's mean active count keeps the price in the same
    units, so alpha only reweights experts WITHIN a layer. Measured motivation:
    per-activation and total loss rank experts at Spearman 0.657, and 327 experts
    carrying 13.1% of all activations sit on the cheapest tier under alpha=0.
    Applied to every action, including removal -- mixing a per-activation price
    for removal with a total price for tiers would make the two incomparable and
    break the allocator's cost/benefit comparison. Raising alpha risks the
    opposite failure (deleting rare specialists, which cost 9.15 points), so it
    is a sweep, not a default.

    Otherwise deliberately nothing else: no tail term, no shrinkage, no
    uncertainty inflation, no tier price. Frequency weighting was
    measured to delete rare specialists (-9.15 human-eval points at budget
    0.40), whereas an average per activation keeps an expert that loses a lot
    when it fires, however rarely it fires. An expert no token routes to has no
    evidence and prices at zero; the planner starts those removed.

    Each action is capped at the price of the next cheaper one, so the
    adjacent-downgrade allocator never meets a tier that costs more than
    removing the expert outright.

    ``tier_price`` (eta) multiplies every SVD-tier price and nothing else: the
    exchange rate between a unit of truncation damage and a unit of removal
    damage. The per-layer trace measures both on uncompressed inputs, where
    truncation error has not yet compounded through later layers, so eta > 1 is
    the correction for tier damage the trace under-counts. 1.0 is an exact no-op.

    ``tier_price_scale`` maps a layer name to one multiplier per SVD tier
    (cheapest rank first), applied on top of eta. It re-prices tiers whose
    deployed factors differ from the ones the trace measured -- e.g. factors
    refined against the expert's true output, whose measured damage ratio to
    the traced factors is layer- and rank-dependent. Layers absent from the
    mapping keep scale 1.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one streaming trace is required")
    if str(damage_reference) not in ("layer_mean", "none"):
        raise ValueError(
            "damage_reference must be 'layer_mean' or 'none', "
            f"got {damage_reference!r}"
        )
    eps = float(epsilon)
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if not float(tier_price) > 0:
        raise ValueError(f"tier_price must be positive, got {tier_price}")
    unit = getattr(traces[0], "damage_unit", "energy")
    converted = []
    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    for trace in traces:
        if not isinstance(trace, LayerIndependentActionTrace):
            raise TypeError("mean-loss planning requires streaming traces")
        if trace.active_token_count is None:
            raise ValueError(
                f"{trace.name}: cached trace lacks active_token_count; rebuild "
                "its streaming trace without --reuse-traces"
            )
        if trace.damage_unit != unit:
            raise ValueError(
                f"all layers must share one damage_unit; {trace.name} is "
                f"{trace.damage_unit!r}"
            )
        actions = trace.actions
        categories = len(trace.category_names)
        active = trace.active_token_count.double()
        if str(damage_reference) == "none":
            reference = torch.ones(categories, dtype=torch.float64)
        else:
            category_tokens = (
                active.sum(dim=0) / float(trace.top_k)
            ).clamp_min(1.0)
            reference = (
                trace.reference_energy.double() / category_tokens
            ).clamp_min(eps)
        mean_loss = trace.raw_action_numerator.double() / (
            active.clamp_min(1.0) * reference.unsqueeze(0)
        ).unsqueeze(1)
        alpha = float(frequency_power)
        if alpha:
            counts = active.clamp_min(1.0)
            # Per layer and category, so the price keeps its units and alpha is
            # purely a within-layer reweighting.
            scale = (counts / counts.mean(dim=0, keepdim=True)).pow(alpha)
            mean_loss = mean_loss * scale.unsqueeze(1)
        mean_loss[:, -1, :] = 0.0

        svd_factor = torch.ones(categories, dtype=torch.float64)
        removal_factor = torch.ones(categories, dtype=torch.float64)
        if propagation_factors is not None:
            if trace.name not in propagation_factors:
                raise ValueError(f"missing propagation factors for {trace.name}")
            layer_factors = propagation_factors[trace.name]
            if "svd" not in layer_factors or "remove" not in layer_factors:
                raise ValueError(
                    f"{trace.name}: propagation factors require 'svd' and 'remove'"
                )
            svd_factor = torch.as_tensor(
                layer_factors["svd"], dtype=torch.float64
            ).reshape(-1)
            removal_factor = torch.as_tensor(
                layer_factors["remove"], dtype=torch.float64
            ).reshape(-1)
            for label, factor in (("svd", svd_factor), ("remove", removal_factor)):
                if (
                    tuple(factor.shape) != (categories,)
                    or not bool(torch.isfinite(factor).all())
                    or bool((factor <= 0).any())
                ):
                    raise ValueError(
                        f"{trace.name}: {label} propagation factors must be "
                        "finite, positive and one per category"
                    )

        price = mean_loss.clone()
        price[:, 0, :] *= removal_factor.reshape(1, -1)
        if actions > 2:
            price[:, 1:-1, :] *= svd_factor.reshape(1, 1, -1) * float(tier_price)
            if tier_price_scale is not None and trace.name in tier_price_scale:
                scale = torch.as_tensor(
                    [float(v) for v in tier_price_scale[trace.name]], dtype=torch.float64
                )
                if tuple(scale.shape) != (actions - 2,) or not bool(
                    (torch.isfinite(scale) & (scale > 0)).all()
                ):
                    raise ValueError(
                        f"{trace.name}: tier_price_scale needs {actions - 2} positive "
                        f"finite values, got {scale.tolist()}"
                    )
                price[:, 1:-1, :] *= scale.reshape(1, -1, 1)
        for action in range(1, actions - 1):
            price[:, action, :] = torch.minimum(
                price[:, action, :], price[:, action - 1, :]
            )
        converted.append(
            LayerIndependentActionTrace(
                name=trace.name,
                action_numerator=price,
                raw_action_numerator=price,
                reference_energy=torch.ones(categories, dtype=torch.float64),
                category_names=trace.category_names,
                action_costs=trace.action_costs,
                action_names=trace.action_names,
                top_k=trace.top_k,
                damage_unit=trace.damage_unit,
            )
        )
        diagnostics[trace.name] = {
            "mean_loss": mean_loss,
            "reference_mean_output": reference,
            "svd_propagation_factor": svd_factor,
            "removal_propagation_factor": removal_factor,
            "action_price": price,
        }
    return converted, diagnostics


def robust_objective(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    method: str = "worst",
    cvar_fraction: float = 0.5,
) -> tuple[float, torch.Tensor]:
    """Reduce category distortions to a scalar allocation objective."""

    relative = numerator.double() / denominator.double().clamp_min(1e-30)
    method = str(method)
    if method == "mean":
        value = relative.mean()
    elif method == "worst":
        value = relative.max()
    elif method == "cvar":
        if not (0 < float(cvar_fraction) <= 1):
            raise ValueError("cvar_fraction must be in (0, 1]")
        count = max(1, ceil(float(cvar_fraction) * int(relative.numel())))
        value = torch.topk(relative, k=count).values.mean()
    else:
        raise ValueError("objective method must be mean, worst, or cvar")
    return float(value.item()), relative


def _layer_objective_components(
    report: DistortionReport | BatchedDistortionReport,
    *,
    layer_reduction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive objective components for global layer aggregation.

    ``energy`` reproduces the legacy ratio of total residual energy to total
    reference energy. ``mean`` gives every MoE layer one vote by aggregating
    its already-normalized category distortion with a unit denominator.
    """

    if layer_reduction == "energy":
        return report.numerator, report.denominator
    if layer_reduction == "mean":
        return report.relative, torch.ones_like(report.relative)
    raise ValueError("layer_reduction must be 'energy' or 'mean'")


@dataclass
class FunctionalCompressionPlan:
    actions: Dict[str, torch.Tensor]
    memory: int
    objective: float
    category_relative_distortion: Dict[str, float]
    history: List[dict]


def _prefer_candidate(
    candidate: tuple,
    incumbent: tuple | None,
    *,
    removal_tiebreak_rtol: float,
    use_removal_saliency: bool,
) -> bool:
    """Compare allocator candidates with an optional REAP removal near-tie."""

    if incumbent is None:
        return True
    candidate_score, candidate_objective, candidate_saved, candidate_layer, candidate_expert, candidate_action, candidate_saliency = candidate
    incumbent_score, incumbent_objective, incumbent_saved, incumbent_layer, incumbent_expert, incumbent_action, incumbent_saliency = incumbent
    scale = max(abs(candidate_score), abs(incumbent_score), 1e-30)
    near = abs(candidate_score - incumbent_score) <= float(removal_tiebreak_rtol) * scale
    if (
        use_removal_saliency
        and near
        and candidate_action == 0
        and incumbent_action == 0
        and candidate_saliency != incumbent_saliency
    ):
        return candidate_saliency < incumbent_saliency
    return (
        candidate_score,
        candidate_objective,
        -candidate_saved,
        candidate_layer,
        candidate_expert,
        candidate_action,
    ) < (
        incumbent_score,
        incumbent_objective,
        -incumbent_saved,
        incumbent_layer,
        incumbent_expert,
        incumbent_action,
    )


@torch.no_grad()
def greedy_global_downgrade(
    traces: Sequence[LayerFunctionalTrace],
    *,
    memory_budget: int,
    objective: str = "worst",
    cvar_fraction: float = 0.5,
    initial_actions: Mapping[str, torch.Tensor] | None = None,
    removal_saliency: Mapping[str, torch.Tensor] | None = None,
    removal_tiebreak_rtol: float = 0.01,
    token_reduction: str = "energy",
    token_energy_floor_ratio: float = 0.05,
    layer_reduction: str = "energy",
    min_retained_experts: int = 0,
    min_layer_memory_ratio: float = 0.0,
    max_layer_relative_distortion: float | None = None,
) -> FunctionalCompressionPlan:
    """Interaction-aware greedy downgrading under a global memory budget.

    Every candidate downgrade re-evaluates its complete layer plan, so its score
    reflects current router renormalization and cross-expert residual interaction.
    The implementation favors clarity for offline planning; traces can use small
    output sketches to keep the repeated evaluations inexpensive.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one functional trace is required")
    if int(memory_budget) < 0:
        raise ValueError("memory_budget must be nonnegative")
    category_names = traces[0].category_names
    if any(trace.category_names != category_names for trace in traces):
        raise ValueError("all traces must use the same category ordering")
    if len({trace.name for trace in traces}) != len(traces):
        raise ValueError("trace names must be unique")
    if int(min_retained_experts) < 0:
        raise ValueError("min_retained_experts must be nonnegative")
    if not (0 <= float(min_layer_memory_ratio) <= 1):
        raise ValueError("min_layer_memory_ratio must be in [0, 1]")
    if max_layer_relative_distortion is not None and float(max_layer_relative_distortion) < 0:
        raise ValueError("max_layer_relative_distortion must be nonnegative")
    if layer_reduction not in {"energy", "mean"}:
        raise ValueError("layer_reduction must be 'energy' or 'mean'")

    plans: Dict[str, torch.Tensor] = {}
    reports: Dict[str, DistortionReport] = {}
    layer_memory_floors: Dict[str, int] = {}
    for trace in traces:
        if initial_actions is not None and trace.name in initial_actions:
            actions = trace._validate_plan(initial_actions[trace.name]).cpu()
        else:
            actions = torch.full(
                (trace.experts,), trace.actions - 1, dtype=torch.long
            )
        plans[trace.name] = actions
        reports[trace.name] = trace.evaluate(
            actions,
            token_reduction=token_reduction,
            token_energy_floor_ratio=token_energy_floor_ratio,
        )
        full_actions = torch.full((trace.experts,), trace.actions - 1, dtype=torch.long)
        layer_memory_floors[trace.name] = ceil(
            float(min_layer_memory_ratio) * trace.memory(full_actions)
        )

    def totals():
        components = [
            _layer_objective_components(report, layer_reduction=layer_reduction)
            for report in reports.values()
        ]
        numerator = sum((value[0] for value in components), torch.zeros(
            len(category_names), dtype=torch.float64
        ))
        denominator = sum((value[1] for value in components), torch.zeros(
            len(category_names), dtype=torch.float64
        ))
        return numerator, denominator

    current_memory = sum(
        trace.memory(plans[trace.name]) for trace in traces
    )
    numerator, denominator = totals()
    current_objective, current_relative = robust_objective(
        numerator,
        denominator,
        method=objective,
        cvar_fraction=cvar_fraction,
    )
    history: List[dict] = []

    while current_memory > int(memory_budget):
        best = None
        for trace in traces:
            current_plan = plans[trace.name]
            retained = int((current_plan != 0).sum().item())
            retained_floor = max(trace.top_k, int(min_retained_experts))
            current_layer_memory = trace.memory(current_plan)
            current_components = _layer_objective_components(
                reports[trace.name], layer_reduction=layer_reduction
            )
            saliency = (
                torch.zeros(trace.experts)
                if removal_saliency is None
                else torch.as_tensor(removal_saliency[trace.name]).float()
            )
            for expert_id in range(trace.experts):
                old_action = int(current_plan[expert_id].item())
                if old_action <= 0:
                    continue
                for new_action in range(old_action):
                    if new_action == 0 and retained <= retained_floor:
                        continue
                    old_cost = int(trace.action_costs[expert_id, old_action].item())
                    new_cost = int(trace.action_costs[expert_id, new_action].item())
                    saved = old_cost - new_cost
                    if saved <= 0:
                        continue
                    if current_layer_memory - saved < layer_memory_floors[trace.name]:
                        continue
                    candidate_plan = current_plan.clone()
                    candidate_plan[expert_id] = new_action
                    candidate_report = trace.evaluate(
                        candidate_plan,
                        token_reduction=token_reduction,
                        token_energy_floor_ratio=token_energy_floor_ratio,
                    )
                    if (
                        max_layer_relative_distortion is not None
                        and float(candidate_report.relative.max().item())
                        > float(max_layer_relative_distortion)
                    ):
                        continue
                    candidate_components = _layer_objective_components(
                        candidate_report, layer_reduction=layer_reduction
                    )
                    candidate_numerator = (
                        numerator - current_components[0] + candidate_components[0]
                    )
                    candidate_denominator = (
                        denominator - current_components[1] + candidate_components[1]
                    )
                    candidate_objective, candidate_relative = robust_objective(
                        candidate_numerator,
                        candidate_denominator,
                        method=objective,
                        cvar_fraction=cvar_fraction,
                    )
                    if not torch.isfinite(torch.tensor(candidate_objective)):
                        continue
                    effective_saved = min(saved, current_memory - int(memory_budget))
                    score = (candidate_objective - current_objective) / effective_saved
                    comparison = (
                        score,
                        candidate_objective,
                        saved,
                        trace.name,
                        expert_id,
                        new_action,
                        float(saliency[expert_id].item()),
                    )
                    if _prefer_candidate(
                        comparison,
                        None if best is None else best[0],
                        removal_tiebreak_rtol=removal_tiebreak_rtol,
                        use_removal_saliency=removal_saliency is not None,
                    ):
                        best = (
                            comparison,
                            trace,
                            expert_id,
                            old_action,
                            new_action,
                            saved,
                            candidate_plan,
                            candidate_report,
                            candidate_numerator,
                            candidate_denominator,
                            candidate_relative,
                            candidate_objective,
                        )
        if best is None:
            raise RuntimeError(
                f"cannot reach memory budget {memory_budget}; minimum valid plan "
                f"still uses {current_memory}"
            )
        (
            _,
            trace,
            expert_id,
            old_action,
            new_action,
            saved,
            candidate_plan,
            candidate_report,
            numerator,
            denominator,
            current_relative,
            current_objective,
        ) = best
        plans[trace.name] = candidate_plan
        reports[trace.name] = candidate_report
        current_memory -= saved
        history.append(
            {
                "layer": trace.name,
                "expert": int(expert_id),
                "from": trace.action_names[old_action],
                "to": trace.action_names[new_action],
                "cost_saved": int(saved),
                "memory": int(current_memory),
                "objective": float(current_objective),
                "category_relative_distortion": {
                    category: float(current_relative[index].item())
                    for index, category in enumerate(category_names)
                },
                "route_change_rate": float(candidate_report.route_change_rate),
            }
        )

    return FunctionalCompressionPlan(
        actions={name: action.clone() for name, action in plans.items()},
        memory=int(current_memory),
        objective=float(current_objective),
        category_relative_distortion={
            category: float(current_relative[index].item())
            for index, category in enumerate(category_names)
        },
        history=history,
    )


def _batched_robust_objective(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    method: str,
    cvar_fraction: float,
) -> torch.Tensor:
    relative = numerator.double() / denominator.double().clamp_min(1e-30)
    if method == "mean":
        return relative.mean(dim=1)
    if method == "worst":
        return relative.max(dim=1).values
    if method == "cvar":
        if not (0 < float(cvar_fraction) <= 1):
            raise ValueError("cvar_fraction must be in (0, 1]")
        count = max(1, ceil(float(cvar_fraction) * int(relative.shape[1])))
        return torch.topk(relative, k=count, dim=1).values.mean(dim=1)
    raise ValueError("objective method must be mean, worst, or cvar")


@torch.no_grad()
def cached_greedy_global_downgrade(
    traces: Sequence[LayerFunctionalTrace],
    *,
    memory_budget: int,
    objective: str = "worst",
    cvar_fraction: float = 0.5,
    initial_actions: Mapping[str, torch.Tensor] | None = None,
    removal_saliency: Mapping[str, torch.Tensor] | None = None,
    removal_tiebreak_rtol: float = 0.01,
    token_reduction: str = "energy",
    token_energy_floor_ratio: float = 0.05,
    layer_reduction: str = "energy",
    min_retained_experts: int = 0,
    min_layer_memory_ratio: float = 0.0,
    max_layer_relative_distortion: float | None = None,
    min_layer_routed_mass: float = 0.0,
    expert_routed_mass: Mapping[str, torch.Tensor] | None = None,
    layer_min_memory_ratio: Mapping[str, float] | None = None,
) -> FunctionalCompressionPlan:
    """Scalable exact-greedy allocator with cached layer-local marginals.

    A change only invalidates functional marginals in its own MoE layer. Every
    other layer's per-category numerator delta remains valid and can be cheaply
    re-reduced under the current global robust objective. The affected layer is
    rebuilt in one vectorized :meth:`LayerFunctionalTrace.evaluate_many` call.

    ``min_layer_routed_mass`` is a coverage CONSTRAINT: a removal is never
    proposed if it would leave less than this fraction of the layer's routed
    mass (``expert_routed_mass``, per expert) on retained experts. Measured on
    Qwen3-30B-A3B, generation collapses once retained mass falls to ~54-65%
    (REAP b30, mean_loss r4 b30), while above it accuracy does not track mass,
    so coverage belongs in the constraints and fidelity in the objective. Tiers
    keep their expert's mass; only removal can violate the floor. 0 is a no-op.

    ``layer_min_memory_ratio`` overrides ``min_layer_memory_ratio`` for the
    layers it names (1.0 keeps a layer uncompressed). It protects layers whose
    output perturbation the rest of the network amplifies far more than the
    per-layer trace can see (on Qwen3-30B-A3B a random-perturbation probe puts
    layers 2-3 at 100-300x the median layer's KL gain), without letting a
    noisy sensitivity estimate reallocate the whole budget.
    """

    traces = list(traces)
    if not traces:
        raise ValueError("at least one functional trace is required")
    if int(memory_budget) < 0:
        raise ValueError("memory_budget must be nonnegative")
    category_names = traces[0].category_names
    if any(trace.category_names != category_names for trace in traces):
        raise ValueError("all traces must use the same category ordering")
    if len({trace.name for trace in traces}) != len(traces):
        raise ValueError("trace names must be unique")
    if int(min_retained_experts) < 0:
        raise ValueError("min_retained_experts must be nonnegative")
    if not (0 <= float(min_layer_memory_ratio) <= 1):
        raise ValueError("min_layer_memory_ratio must be in [0, 1]")
    if max_layer_relative_distortion is not None and float(max_layer_relative_distortion) < 0:
        raise ValueError("max_layer_relative_distortion must be nonnegative")
    if layer_reduction not in {"energy", "mean"}:
        raise ValueError("layer_reduction must be 'energy' or 'mean'")

    plans: Dict[str, torch.Tensor] = {}
    reports: Dict[str, DistortionReport] = {}
    layer_memory_floors: Dict[str, int] = {}
    for trace in traces:
        if initial_actions is not None and trace.name in initial_actions:
            actions = trace._validate_plan(initial_actions[trace.name]).cpu()
        else:
            actions = torch.full((trace.experts,), trace.actions - 1, dtype=torch.long)
        plans[trace.name] = actions
        reports[trace.name] = trace.evaluate(
            actions,
            token_reduction=token_reduction,
            token_energy_floor_ratio=token_energy_floor_ratio,
        )
        full_actions = torch.full((trace.experts,), trace.actions - 1, dtype=torch.long)
        ratio = float(min_layer_memory_ratio)
        if layer_min_memory_ratio is not None and trace.name in layer_min_memory_ratio:
            ratio = float(layer_min_memory_ratio[trace.name])
            if not (0 <= ratio <= 1):
                raise ValueError(f"{trace.name}: layer_min_memory_ratio must be in [0, 1]")
        layer_memory_floors[trace.name] = ceil(ratio * trace.memory(full_actions))
    if layer_min_memory_ratio is not None:
        unknown = set(layer_min_memory_ratio) - {trace.name for trace in traces}
        if unknown:
            raise ValueError(f"layer_min_memory_ratio names unknown layers {sorted(unknown)}")
    if not (0.0 <= float(min_layer_routed_mass) < 1.0):
        raise ValueError("min_layer_routed_mass must be in [0, 1)")
    layer_mass = {}
    if float(min_layer_routed_mass) > 0.0:
        if expert_routed_mass is None:
            raise ValueError("min_layer_routed_mass needs expert_routed_mass")
        for trace in traces:
            mass = torch.as_tensor(expert_routed_mass[trace.name], dtype=torch.float64)
            if mass.shape != (trace.experts,):
                raise ValueError(f"{trace.name}: expert_routed_mass must be per expert")
            layer_mass[trace.name] = (mass, float(min_layer_routed_mass) * float(mass.sum()))

    components = {
        name: _layer_objective_components(report, layer_reduction=layer_reduction)
        for name, report in reports.items()
    }
    denominator = sum(
        (value[1] for value in components.values()),
        torch.zeros(len(category_names), dtype=torch.float64),
    )
    numerator = sum(
        (value[0] for value in components.values()),
        torch.zeros(len(category_names), dtype=torch.float64),
    )
    current_objective, current_relative = robust_objective(
        numerator,
        denominator,
        method=objective,
        cvar_fraction=cvar_fraction,
    )
    current_memory = sum(trace.memory(plans[trace.name]) for trace in traces)
    history: List[dict] = []

    def build_candidates(trace: LayerFunctionalTrace):
        current = plans[trace.name]
        retained = int((current != 0).sum().item())
        retained_floor = max(trace.top_k, int(min_retained_experts))
        current_layer_memory = trace.memory(current)
        mass_floor = layer_mass.get(trace.name)
        retained_mass = (
            float(mass_floor[0][current != 0].sum()) if mass_floor is not None else 0.0
        )
        expert_ids = []
        old_actions = []
        new_actions = []
        savings = []
        candidate_plans = []
        for expert_id in range(trace.experts):
            old_action = int(current[expert_id].item())
            if old_action <= 0:
                continue
            for new_action in range(old_action):
                if new_action == 0 and retained <= retained_floor:
                    continue
                if (
                    new_action == 0
                    and mass_floor is not None
                    and retained_mass - float(mass_floor[0][expert_id]) < mass_floor[1]
                ):
                    continue
                saved = int(
                    trace.action_costs[expert_id, old_action].item()
                    - trace.action_costs[expert_id, new_action].item()
                )
                if saved <= 0:
                    continue
                if current_layer_memory - saved < layer_memory_floors[trace.name]:
                    continue
                candidate = current.clone()
                candidate[expert_id] = new_action
                expert_ids.append(expert_id)
                old_actions.append(old_action)
                new_actions.append(new_action)
                savings.append(saved)
                candidate_plans.append(candidate)
        if not candidate_plans:
            return None
        batch = trace.evaluate_many(
            torch.stack(candidate_plans),
            token_reduction=token_reduction,
            token_energy_floor_ratio=token_energy_floor_ratio,
        )
        valid = torch.ones(len(candidate_plans), dtype=torch.bool)
        if max_layer_relative_distortion is not None:
            valid &= batch.relative.max(dim=1).values <= float(
                max_layer_relative_distortion
            )
        if not bool(valid.any()):
            return None
        return {
            "expert_ids": torch.tensor(expert_ids, dtype=torch.long)[valid],
            "old_actions": torch.tensor(old_actions, dtype=torch.long)[valid],
            "new_actions": torch.tensor(new_actions, dtype=torch.long)[valid],
            "savings": torch.tensor(savings, dtype=torch.float64)[valid],
            "plans": torch.stack(candidate_plans)[valid],
            "reports": BatchedDistortionReport(
                numerator=batch.numerator[valid],
                denominator=batch.denominator[valid],
                relative=batch.relative[valid],
                route_change_rate=batch.route_change_rate[valid],
            ),
        }

    candidates = {trace.name: build_candidates(trace) for trace in traces}
    # Progress reporting. The loop rescans every layer per step and applies one
    # downgrade, so runtime scales with the step count, not the budget gap: a
    # ranking that prefers cheap tier downgrades over removals needs several
    # times more steps to free the same bytes. Without this the phase is silent
    # for however long it takes and a slow run is indistinguishable from a hang.
    _greedy_steps = 0
    _greedy_start_memory = current_memory
    _greedy_gap = max(current_memory - int(memory_budget), 1)
    while current_memory > int(memory_budget):
        _greedy_steps += 1
        if _greedy_steps % 250 == 0:
            logging.info(
                "  greedy: %d steps, %.1f%% of budget gap closed (%d -> %d)",
                _greedy_steps,
                100.0 * (_greedy_start_memory - current_memory) / _greedy_gap,
                current_memory,
                int(memory_budget),
            )
        best = None
        for trace in traces:
            cache = candidates[trace.name]
            if cache is None:
                continue
            batch = cache["reports"]
            batch_components = _layer_objective_components(
                batch, layer_reduction=layer_reduction
            )
            candidate_numerator = (
                numerator.unsqueeze(0)
                - components[trace.name][0].unsqueeze(0)
                + batch_components[0]
            )
            candidate_denominator = (
                denominator.unsqueeze(0)
                - components[trace.name][1].unsqueeze(0)
                + batch_components[1]
            )
            candidate_objectives = _batched_robust_objective(
                candidate_numerator,
                candidate_denominator,
                method=objective,
                cvar_fraction=cvar_fraction,
            )
            effective_savings = cache["savings"].clamp_max(
                float(current_memory - int(memory_budget))
            )
            scores = (candidate_objectives - current_objective) / effective_savings
            # An aggressive budget can leave a layer with only removals that
            # exhaust at least one token's routing shortlist. Those candidates
            # deliberately evaluate to infinite distortion. Do not pass an
            # infinite pseudo-winner into the near-tie code: ``inf - inf`` is
            # NaN and therefore produces an empty tie set. Other layers can
            # still contain a valid downgrade.
            finite_scores = torch.isfinite(scores)
            if not bool(finite_scores.any()):
                continue
            comparable_scores = scores.masked_fill(~finite_scores, torch.inf)
            index = int(torch.argmin(comparable_scores).item())
            saliency = None
            if removal_saliency is not None:
                saliency = torch.as_tensor(
                    removal_saliency[trace.name], dtype=torch.float64
                )
                if saliency.shape != (trace.experts,):
                    raise ValueError(
                        f"{trace.name}: removal saliency must have one value per expert"
                    )
                # If the raw winner is a removal, choose the lowest-REAP removal
                # whose functional score lies inside the configured near-tie band.
                if int(cache["new_actions"][index].item()) == 0:
                    raw_score = scores[index]
                    scale = torch.maximum(
                        scores.abs(), raw_score.abs().clamp_min(1e-30)
                    )
                    near = torch.isfinite(scores) & (
                        (scores - raw_score).abs()
                        <= float(removal_tiebreak_rtol) * scale
                    )
                    removal_near = near & (cache["new_actions"] == 0)
                    candidate_reap = saliency[cache["expert_ids"]]
                    candidate_reap = candidate_reap.masked_fill(
                        ~torch.isfinite(candidate_reap), torch.inf
                    )
                    candidate_reap = candidate_reap.masked_fill(
                        ~removal_near, torch.inf
                    )
                    best_reap = candidate_reap.min()
                    reap_ties = removal_near & (candidate_reap == best_reap)
                    tied_indices = reap_ties.nonzero(as_tuple=True)[0]
                    # The raw finite removal is always in ``removal_near``.
                    # Retain it if saliency itself offers no finite tie; this
                    # also makes malformed/non-finite auxiliary statistics
                    # harmless to the functional allocator.
                    if int(tied_indices.numel()):
                        index = int(
                            tied_indices[
                                torch.argmin(scores.index_select(0, tied_indices))
                            ].item()
                        )
            removal_value = (
                0.0
                if saliency is None
                else float(saliency[int(cache["expert_ids"][index].item())].item())
            )
            comparison = (
                float(scores[index].item()),
                float(candidate_objectives[index].item()),
                int(cache["savings"][index].item()),
                trace.name,
                int(cache["expert_ids"][index].item()),
                int(cache["new_actions"][index].item()),
                removal_value,
            )
            if _prefer_candidate(
                comparison,
                None if best is None else best[0],
                removal_tiebreak_rtol=removal_tiebreak_rtol,
                use_removal_saliency=removal_saliency is not None,
            ):
                best = (
                    comparison,
                    trace,
                    cache,
                    index,
                    candidate_numerator,
                    candidate_denominator,
                    candidate_objectives,
                )
        if best is None or not torch.isfinite(torch.tensor(best[0][0])):
            raise RuntimeError(
                f"cannot reach memory budget {memory_budget}; minimum valid plan "
                f"still uses {current_memory}"
            )

        (
            _, trace, cache, index, candidate_numerator,
            candidate_denominator, candidate_objectives,
        ) = best
        expert_id = int(cache["expert_ids"][index].item())
        old_action = int(cache["old_actions"][index].item())
        new_action = int(cache["new_actions"][index].item())
        saved = int(cache["savings"][index].item())
        plans[trace.name] = cache["plans"][index].clone()
        batch = cache["reports"]
        reports[trace.name] = DistortionReport(
            numerator=batch.numerator[index].clone(),
            denominator=batch.denominator[index].clone(),
            relative=batch.relative[index].clone(),
            route_change_rate=float(batch.route_change_rate[index].item()),
        )
        components[trace.name] = _layer_objective_components(
            reports[trace.name], layer_reduction=layer_reduction
        )
        numerator = candidate_numerator[index].clone()
        denominator = candidate_denominator[index].clone()
        current_objective = float(candidate_objectives[index].item())
        current_relative = numerator / denominator.clamp_min(1e-30)
        current_memory -= saved
        history.append(
            {
                "layer": trace.name,
                "expert": expert_id,
                "from": trace.action_names[old_action],
                "to": trace.action_names[new_action],
                "cost_saved": saved,
                "memory": int(current_memory),
                "objective": current_objective,
                "category_relative_distortion": {
                    category: float(current_relative[i].item())
                    for i, category in enumerate(category_names)
                },
                "route_change_rate": reports[trace.name].route_change_rate,
                "removal_saliency": (
                    None
                    if new_action != 0 or removal_saliency is None
                    else float(
                        torch.as_tensor(removal_saliency[trace.name])[expert_id].item()
                    )
                ),
            }
        )
        candidates[trace.name] = build_candidates(trace)

    return FunctionalCompressionPlan(
        actions={name: action.clone() for name, action in plans.items()},
        memory=int(current_memory),
        objective=float(current_objective),
        category_relative_distortion={
            category: float(current_relative[index].item())
            for index, category in enumerate(category_names)
        },
        history=history,
    )


def gaussian_output_projection(
    hidden_size: int,
    sketch_size: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct a reproducible Gaussian JL projection for output residuals."""

    hidden_size, sketch_size = int(hidden_size), int(sketch_size)
    if hidden_size <= 0 or sketch_size <= 0:
        raise ValueError("hidden_size and sketch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    projection = torch.randn(hidden_size, sketch_size, generator=generator)
    projection /= float(sketch_size) ** 0.5
    return projection.to(device=device)
