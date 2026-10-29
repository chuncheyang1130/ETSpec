"""Runtime block for heterogeneous routed-functional MoE compression plans.

Each retained expert is either exact or belongs to one ordinary-SVD rank tier.
The router runs once over the union of retained experts.  CUDA dispatch then
invokes one fixed-grid grouped kernel per tier, which keeps the action layout a
model-construction constant and is safe for ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ordinary_svd import (
    calibrated_balanced_low_rank_factors,
    gram_shrinkage_coefficient,
    input_metric_from_samples,
    uninformative_input_metric,
)
from .moe_arch import (
    SILU, activation_derivative, activation_fn, describe_moe_block, retained_topk_routing,
    router_logits,
)
from .qwen3_moe_stacked import Qwen3MoeStackedBlock
from .triton_fused_down_gmm_reduction import triton_fused_down_gmm_reduction
from .triton_fused_gate_up_gmm_silu import activation_code, triton_fused_gate_up_gmm_silu
from .triton_hybrid_svd_gmm import (
    fused_down_gmm_reduction_accumulate,
    fused_gate_up_gmm_silu_store,
    fused_ragged_down_gmm_reduction_accumulate,
    fused_ragged_gate_up_gmm_silu_store,
    fused_ragged_projection_gmm_store,
    hybrid_moe_experts_overlapped,
)


# Floor on the relative two-sided ridge of the compensation's normal equations,
# added to the evidence deficit. Keeps the solve finite for an expert whose
# evidence is ample but whose auto-moment is still singular.
_DOWN_COMPENSATION_RIDGE = 1e-6

# Run the low-rank chain on a second stream so it overlaps the full experts
# (see ``_stream_overlap_forward``). Off by default: it changes only scheduling,
# but it is new, so the shipped path stays the single-stream one until it has run
# a benchmark. ``ETSPEC_MOE_STREAM_OVERLAP=1`` turns it on.
_STREAM_OVERLAP = os.environ.get("ETSPEC_MOE_STREAM_OVERLAP", "0") == "1"


@torch.no_grad()
def _expert_middle(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    storage_dtype: torch.dtype,
    activation: str = SILU,
) -> torch.Tensor:
    """The uncompressed gated intermediate ``act(gate) * up``, in the deployed dtype."""

    deployed = inputs.to(device=gate_weight.device, dtype=storage_dtype)
    gate = F.linear(deployed, gate_weight.to(storage_dtype))
    up = F.linear(deployed, up_weight.to(storage_dtype))
    return activation_fn(gate, activation) * up


def routed_factor_seed(
    layer_seed: int, expert_id: int, gate_up_rank: int, down_rank: int
) -> int:
    """Stable RNG seed for one expert/action factorization."""

    return int(
        int(layer_seed)
        + int(expert_id) * 1_000_003
        + int(gate_up_rank) * 1009
        + int(down_rank) * 9176
    )


# Which tokens calibrate a tiered expert's SVD, and what weight each carries.
#   shortlist     every token whose router top-L contains the expert, weighted by
#                 the squared softmax over that top-L; gate/up use the recorded
#                 sensitivity-weighted moments when present. The original design.
#   topk_g2       ASVD-style: only the tokens the compressed model routes to the
#                 expert -- its top-k over the experts the plan RETAINS, so a token
#                 whose original top-k held a removed expert calibrates the next
#                 retained one instead -- weighted by the squared routing weight g
#                 (renormalized among that retained top-k) that multiplies the
#                 expert's output. Gate/up and down see the same tokens and
#                 weights, and no recorded moment is used.
#   topk_g1       the same tokens weighted by g rather than g^2. The exponent on
#                 g should match the damage unit: a SQUARED objective
#                 (g||f-fhat||)^2 gives g^2, an AMPLITUDE objective g||f-fhat||
#                 gives g. topk_g2 was derived when the planner priced energy;
#                 the planner now prices amplitude, so g^2 over-weights the
#                 high-g tokens relative to what is charged.
#   topk_uniform  the same tokens unweighted: the expert calibrated like a dense
#                 layer on exactly the inputs it serves. Ignores the router
#                 entirely, which is the most robust choice if the deployed
#                 routing differs from the calibration routing.
FACTOR_CALIBRATIONS = ("shortlist", "topk_g2", "topk_g1", "topk_uniform")

# How the gate/up factorization weights its OUTPUT channels (the 768 intermediate
# channels).  "none" is the shipped plain Frobenius norm, which charges every
# channel equally.  "jacobian" weights channel j by how much an error there
# actually reaches the expert output through the SiLU product and W_down:
#
#     c_gate[j] = ||W_down[:,j]||^2 * sum_t w_t * SiLU'(a_tj)^2 * b_tj^2
#     c_up[j]   = ||W_down[:,j]||^2 * sum_t w_t * SiLU(a_tj)^2
#
# The factorization then truncates D @ [W_gate;W_up] @ S instead of
# [W_gate;W_up] @ S and undoes D, which stays a closed-form Eckart-Young
# solution and leaves the stored factor shapes (and the kernels) untouched.
# Measured held-out on 18 experts: -13.9% expert-output error gate/up-only,
# -10.4% on a full gu/down tier, 17/17 experts.
FACTOR_OUTPUT_METRICS = ("none", "jacobian")


def factor_calibration_selection(
    logits: torch.Tensor,
    *,
    factor_calibration: str,
    top_k: int,
    norm_topk_prob: bool,
    router_candidates: int,
    retained: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token calibration experts and the weight of each (token, expert) pair.

    Returns ``(ids, weights)``, both ``[tokens, L]``: token ``t`` calibrates expert
    ``ids[t, j]`` with weight ``weights[t, j]``. The deployment and the planner
    both select through here, so a plan prices the factors inference loads.

    ``retained`` is a boolean mask over experts. The top-k modes route over the
    retained experts only, exactly like the deployed router, so a removed
    expert's traffic lands on the next retained expert. Its weight is the one the
    deployed block applies: renormalized over the top-k when ``norm_topk_prob``,
    otherwise the ORIGINAL softmax probability (removed experts are masked after
    the softmax, see ``moe_arch.retained_topk_routing``). ``None`` means nothing
    is removed. The shortlist mode ignores it.
    """

    mode = str(factor_calibration)
    if mode not in FACTOR_CALIBRATIONS:
        raise ValueError(
            f"factor_calibration must be one of {FACTOR_CALIBRATIONS}, got {mode!r}"
        )
    logits = logits.float()
    full_logits = logits
    experts = int(logits.shape[-1])
    shortlist_width = min(int(router_candidates), experts)
    if mode == "shortlist":
        values, ids = torch.topk(logits, k=shortlist_width, dim=-1)
        return ids, F.softmax(values, dim=-1).square()
    if retained is not None:
        retained = torch.as_tensor(retained, dtype=torch.bool).reshape(-1)
        if int(retained.numel()) != experts:
            raise ValueError(
                f"retained mask has {int(retained.numel())} entries "
                f"for {experts} experts"
            )
        if int(retained.sum().item()) < int(top_k):
            raise ValueError("the plan retains fewer than top_k experts")
        # A removed expert can never win a slot, so each token's top-k is its
        # top-k among retained experts.
        logits = logits.masked_fill(~retained.to(logits.device), -torch.inf)
    values, ids = torch.topk(logits, k=min(int(top_k), experts), dim=-1)
    if mode == "topk_uniform":
        return ids, torch.ones_like(values)
    # The weight the model actually applies to this expert's output.
    if norm_topk_prob:
        routing = F.softmax(values, dim=-1)
    else:
        # Softmax over ALL experts (removed included), as the deployed router.
        routing = F.softmax(full_logits, dim=-1).gather(-1, ids)
    if mode == "topk_g1":
        return ids, routing
    return ids, routing.square()


@torch.no_grad()
def apply_deployed_expert_factors(
    inputs: torch.Tensor, factors: dict[str, torch.Tensor], activation: str = SILU
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same balanced, quantized factors stored by inference."""

    dtype = factors["gu_input_factor"].dtype
    deployed_inputs = inputs.to(
        device=factors["gu_input_factor"].device, dtype=dtype
    )
    latent = F.linear(deployed_inputs, factors["gu_input_factor"])
    gate = F.linear(latent, factors["gate_output_factor"])
    up = F.linear(latent, factors["up_output_factor"])
    middle = activation_fn(gate, activation) * up
    down_latent = F.linear(middle, factors["down_input_factor"])
    output = F.linear(down_latent, factors["down_output_factor"])
    return output, middle


@torch.no_grad()
def gate_up_output_metric(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale_floor: float = 1e-6,
    activation: str = SILU,
) -> torch.Tensor:
    """Per-output-channel sensitivity of the gate/up stage, stacked [gate; up].

    Returns ``sqrt(c)`` for the stacked ``[W_gate; W_up]``, i.e. the diagonal of
    ``D``.  ``scale_floor`` is relative to each branch's own maximum: a channel
    the SiLU gates off has c ~ 0, and ``D^-1`` would otherwise explode on it.
    """

    x = inputs.float()
    w = weights.float().reshape(-1, 1)
    a = F.linear(x, gate_weight.float())
    b = F.linear(x, up_weight.float())
    column_energy = down_weight.float().pow(2).sum(dim=0)
    act_derivative = activation_derivative(a, activation)
    gate_sensitivity = column_energy * (w * act_derivative.square() * b.square()).sum(0)
    up_sensitivity = column_energy * (w * activation_fn(a, activation).square()).sum(0)
    floored = []
    for sensitivity in (gate_sensitivity, up_sensitivity):
        peak = float(sensitivity.max().item())
        if not (peak > 0) or not bool(torch.isfinite(sensitivity).all()):
            sensitivity = torch.ones_like(sensitivity)
            peak = 1.0
        floored.append(sensitivity.clamp_min(scale_floor * peak))
    return torch.cat(floored).sqrt()


def build_deployed_expert_factors(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_moment: torch.Tensor,
    up_moment: torch.Tensor,
    recorded_down_moment: torch.Tensor,
    *,
    expert_id: int,
    gate_up_rank: int,
    down_rank: int,
    layer_seed: int,
    candidate_inputs: torch.Tensor | None,
    candidate_weights: torch.Tensor | None,
    scale_floor: float,
    randomized: bool,
    storage_dtype: torch.dtype,
    gram_metric: bool | None = None,
    allow_uncalibrated: bool = False,
    gram_shrink_target: torch.Tensor | None = None,
    gram_shrink_alpha: float | str = "auto",
    down_compensation: bool = False,
    use_recorded_moments: bool = True,
    output_metric: str = "none",
    activation: str = SILU,
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    """Construct exactly the BF16/FP16 factors evaluated and loaded at runtime.

    ``gram_metric`` selects the input metric the SVD is calibrated in: ``False``
    the diagonal second moment, ``True`` the full Gram ``Z^T diag(w) Z`` (correct
    when the input coordinates are correlated, which hidden states strongly are).
    ``None`` infers it from the recorded moments' rank.  A plan records its choice
    so the deployment rebuilt from it cannot silently use the other metric.

    ``use_recorded_moments=False`` ignores all three recorded moments and
    calibrates both stages on ``candidate_inputs`` alone (the top-k modes of
    ``factor_calibration_selection``). The recorded gate/up moments carry a
    heuristic downstream-sensitivity factor and a different token set, so mixing
    them into a calibration chosen elsewhere would fit gate/up and down to
    different traffic.

    ``output_metric="jacobian"`` additionally weights the gate/up OUTPUT channels
    by their downstream sensitivity (see ``FACTOR_OUTPUT_METRICS``).  It needs
    calibration tokens; without them the stage falls back to ``"none"``.
    """

    if str(output_metric) not in FACTOR_OUTPUT_METRICS:
        raise ValueError(
            f"output_metric must be one of {FACTOR_OUTPUT_METRICS}, "
            f"got {output_metric!r}"
        )

    device = gate_weight.device
    hidden = int(gate_weight.shape[1])
    intermediate = int(gate_weight.shape[0])
    joint_moment = gate_moment.float() + up_moment.float()
    if gram_metric is None:
        gram_metric = bool(joint_moment.ndim == 2)
    gram_metric = bool(gram_metric)
    # Same rule the gram metric enforces below, for the same reason: the plan
    # priced a COMPENSATED down target, and without calibration tokens there is
    # nothing to compensate against, so the deployment would silently ship the
    # uncompensated weight under a plan that scored the other matrix. The gram
    # guard sits inside the moment-fallback branch, which a diagonal-moment plan
    # never enters, so compensation needs its own check up here.
    # ``allow_uncalibrated`` again means "the caller has a source; THIS expert
    # merely never appears in a shortlist", which is a real and benign condition.
    if down_compensation and candidate_inputs is None and not allow_uncalibrated:
        raise ValueError(
            "down_compensation requires calibration tokens, but candidate_inputs "
            "is None and the caller reported no calibration source. Set "
            "hidden_cache_dir in the method config; the recorded moments cannot "
            "substitute, because the compensation needs the per-token middle "
            "activations, not their second moment."
        )
    # A recorded moment of the other kind carries no usable information for this
    # metric, so fall through to computing one from the routed calibration
    # tokens rather than silently factorising in the wrong geometry. That is the
    # designed path under --factor-metric gram against a diagonal artifact, not a
    # calibration failure, so it is reported separately below.
    if (
        not use_recorded_moments
        and candidate_inputs is None
        and not allow_uncalibrated
    ):
        raise ValueError(
            "use_recorded_moments=False calibrates on routed tokens only, but "
            "candidate_inputs is None and the caller reported no calibration "
            "source. Set hidden_cache_dir in the method config."
        )
    recorded_wrong_kind = bool(gram_metric != (joint_moment.ndim == 2))
    if recorded_wrong_kind or not use_recorded_moments:
        joint_moment = torch.zeros((), dtype=torch.float32)
    gate_up_fallback = float(joint_moment.max().item()) <= 0
    if gate_up_fallback and candidate_inputs is not None:
        if candidate_weights is None:
            raise ValueError("candidate inputs require candidate weights")
        joint_moment = input_metric_from_samples(
            candidate_inputs,
            candidate_weights,
            gram=gram_metric,
            # The layer Gram, when supplied, is the shrinkage target: it is the
            # per-layer covariance SVD-LLM-style whitening assumes, pooled over
            # every calibration token instead of this expert's routed subset.
            shrinkage=gram_shrink_alpha,
            shrinkage_target=gram_shrink_target if gram_metric else None,
        ).cpu()
    gate_up_weight_only = float(joint_moment.max().item()) <= 0
    if gate_up_weight_only:
        # Diagonal mode reaches here only for an expert the calibration never
        # routed to, and an ordinary weight SVD is the right answer. Gram mode
        # reaches here whenever the caller supplied no calibration tokens, which
        # is a plumbing failure, not a rare expert: the recorded diagonal moments
        # were discarded as the wrong kind, so falling through would silently
        # deploy UNCALIBRATED factors that the plan priced as calibrated.
        # ``allow_uncalibrated`` says the caller HAS a calibration source and this
        # particular expert simply never appears in any token's shortlist -- a real
        # condition (34 of 6144 experts on this model), handled the same way the
        # diagonal metric handles it: an ordinary weight SVD. Without that
        # assurance a missing metric means the tokens never arrived at all, which
        # would silently deploy uncalibrated factors the plan priced as calibrated.
        if gram_metric and candidate_inputs is None and not allow_uncalibrated:
            raise ValueError(
                "gram factor metric requires calibration tokens, but "
                "candidate_inputs is None and the caller reported no calibration "
                "source. Supply the routed hidden states (set hidden_cache_dir in "
                "the method config); a diagonal moments artifact cannot substitute."
            )
        joint_moment = uninformative_input_metric(hidden, gram=gram_metric)

    rng_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    seed = routed_factor_seed(layer_seed, expert_id, gate_up_rank, down_rank)
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        joined = torch.cat((gate_weight.float(), up_weight.float()), dim=0)
        output_scale = None
        if str(output_metric) == "jacobian" and candidate_inputs is not None:
            if candidate_weights is None:
                raise ValueError("candidate inputs require candidate weights")
            output_scale = gate_up_output_metric(
                gate_weight,
                up_weight,
                down_weight,
                candidate_inputs,
                candidate_weights,
                activation=activation,
            )
            joined = output_scale.unsqueeze(1) * joined
        left, right = calibrated_balanced_low_rank_factors(
            joined,
            joint_moment,
            int(gate_up_rank),
            scale_floor=scale_floor,
            randomized=randomized,
        )
        if output_scale is not None:
            # Undo D on the left factor only: D is diagonal and acts within each
            # 768-row block, so the shared gu_input_factor is untouched.
            left = left / output_scale.unsqueeze(1)
        factors = {
            "gate_output_factor": left[:intermediate].to(storage_dtype),
            "up_output_factor": left[intermediate:].to(storage_dtype),
            "gu_input_factor": right.to(storage_dtype),
        }

        # The compensation below has to build the full intermediate x intermediate
        # auto-moment regardless, so truncating in that same metric instead of its
        # diagonal is free. Measured over 72 cells it takes the compensated error
        # from 0.205 to 0.115 of the uncompressed expert's output energy, against
        # an uncompensated diagonal baseline of 0.365 -- 1.8x -> 3.2x. Only the
        # calibration changes; the stored factors keep their shape and the kernels
        # never see a Gram.
        down_gram = bool(gram_metric or down_compensation)
        selected_down_moment = recorded_down_moment.float()
        if not use_recorded_moments or down_gram != (selected_down_moment.ndim == 2):
            selected_down_moment = torch.zeros((), dtype=torch.float32)
        compressed_middle = None
        if candidate_inputs is not None:
            deployed_inputs = candidate_inputs.to(
                device=device, dtype=storage_dtype
            )
            latent = F.linear(deployed_inputs, factors["gu_input_factor"])
            gate = F.linear(latent, factors["gate_output_factor"])
            up = F.linear(latent, factors["up_output_factor"])
            compressed_middle = activation_fn(gate, activation) * up
            selected_down_moment = input_metric_from_samples(
                compressed_middle, candidate_weights.to(device), gram=down_gram
            ).cpu()
            if float(selected_down_moment.max().item()) <= 0:
                selected_down_moment = input_metric_from_samples(
                    compressed_middle,
                    torch.ones(
                        compressed_middle.shape[0],
                        device=compressed_middle.device,
                    ),
                    gram=down_gram,
                ).cpu()
        down_fallback = float(selected_down_moment.max().item()) <= 0
        if down_fallback:
            selected_down_moment = uninformative_input_metric(
                intermediate, gram=down_gram
            )
        # The down projection is the only stage with upstream damage to absorb.
        # Fitting it to imitate W_down assumes the middle it receives is the
        # middle W_down was trained on; it is not, because gate/up were just
        # truncated. Measured on Qwen3-30B-A3B coding256, ~72% of the down-stage
        # error is inherited from that truncation rather than caused by down's
        # own rank limit, and the inherited part is a ROTATION of the middle, not
        # a rescale (a scalar corrector recovers 1.8% of it, a diagonal one 17%).
        #
        # So target the original output instead of the original matrix:
        #     min_rank-r  sum_t w_t || W_down z_t - What zhat_t ||^2
        # whose unconstrained optimum is the reduced-rank-regression estimator
        # What* = W_down B^T A^{-1}, with A = sum w zhat zhat^T (what down will
        # receive) and B = sum w zhat z^T (how it relates to what down should
        # have received). Folding What* into the weight before truncation costs
        # nothing at inference: same rank, same bytes, same kernels.
        #
        # zhat == z gives B == A and What* == W_down, so this is exactly a
        # generalization of the previous behavior, not a replacement for it.
        down_target = down_weight.float()
        compensated = False
        if down_compensation and compressed_middle is not None:
            clean_middle = _expert_middle(
                candidate_inputs, gate_weight, up_weight, storage_dtype, activation
            )
            weights = candidate_weights.to(device=device, dtype=torch.float32)
            weighted = compressed_middle.float() * weights.unsqueeze(1)
            auto = (weighted.t() @ compressed_middle.float()).double()
            cross = (weighted.t() @ clean_middle.float()).double()
            # A starved expert cannot estimate a full intermediate x intermediate
            # map, so ridge BOTH normal equations by the same amount. Doing it to
            # ``auto`` alone -- the obvious move, and what shrinking the Gram
            # toward its diagonal amounts to -- is not a regularizer at all: it
            # makes the two moments inconsistent estimates and INFLATES the
            # correction exactly where evidence is thinnest. Measured, that turned
            # a 11x win into a 3x loss on the most starved expert sampled.
            #
            # Ridging both sides instead shrinks the correction toward the
            # identity, i.e. toward not correcting, and preserves the defining
            # identity exactly: zhat == z gives auto == cross, hence C == I for
            # any lambda. The scale is intermediate/N_eff, so it is negligible for
            # a well-sampled expert and dominant for one with no evidence.
            #
            # The scale is the EVIDENCE DEFICIT, max(0, intermediate/N_eff - 1),
            # not intermediate/N_eff: an expert with at least as many effective
            # tokens as dimensions can estimate the map, so it gets the bare
            # numerical floor and the undamped estimator, which measured best in
            # this metric (0.409 vs 0.445 of the uncompensated error). Damping
            # only switches on below N_eff == intermediate, where the map is
            # genuinely underdetermined. Median N_eff here is ~7x intermediate.
            #
            # An expert that appears in no token's shortlist arrives here with a
            # ZERO-ROW ``compressed_middle``: the caller passes an empty tensor,
            # not None, so the guard above admits it (34 of 6144 experts on this
            # model). ``auto`` is then exactly zero -- and because the ridge is
            # SCALED by mean(diag(auto)), the evidence deficit is multiplied to
            # nothing rather than dominating, so the solve fails outright instead
            # of damping toward the identity. Candidate weights that are all zero
            # land in the same state. No evidence means the identity IS the right
            # correction, which is what leaving ``down_target`` at ``down_weight``
            # already gives, so skip. ``nan > 0`` is False, so a degenerate N_eff
            # is caught by the same test.
            auto_scale = float(auto.diagonal().mean())
            if auto_scale > 0:
                alpha = min(
                    gram_shrinkage_coefficient(weights, intermediate), 0.999
                )
                deficit = max(0.0, alpha / (1.0 - alpha) - 1.0)
                eye = torch.eye(
                    intermediate, device=auto.device, dtype=auto.dtype
                )
                ridge = auto_scale * (deficit + _DOWN_COMPENSATION_RIDGE)
                correction = torch.linalg.solve(
                    auto + ridge * eye, cross + ridge * eye
                ).t()
                down_target = (down_weight.double() @ correction).float()
                compensated = True
        left_down, right_down = calibrated_balanced_low_rank_factors(
            down_target,
            selected_down_moment,
            int(down_rank),
            scale_floor=scale_floor,
            randomized=randomized,
        )
        factors["down_output_factor"] = left_down.to(storage_dtype)
        factors["down_input_factor"] = right_down.to(storage_dtype)
    return factors, {
        # Only a recorded moment of the RIGHT kind that turned out unusable is a
        # fallback worth counting; a kind mismatch is the intended gram path, and
        # ignoring recorded moments is the intended top-k path.
        "gate_up_router_conditioned": bool(
            gate_up_fallback and not gate_up_weight_only and not recorded_wrong_kind
            and use_recorded_moments
        ),
        "gate_up_weight_only": bool(gate_up_weight_only),
        "down_weight_only": bool(down_fallback),
        # The down factors approximate this matrix, which is W_down only when the
        # compensation is off. A structural (weight-space) error must be measured
        # against it, not against W_down, or a compensated action looks worse
        # structurally exactly when its true damage went down.
        "down_target": down_target,
        "down_compensated": bool(compensated),
    }


class RoutedFunctionalSVDTier(nn.Module):
    """One homogeneous low-rank tier: a named view into the block factor store.

    A tier does not own its factors. The deployed block runs one merged kernel
    over every low-rank expert it retains, whatever tier the expert is in, and
    that kernel needs all the tiers' factors in one buffer -- so the block
    allocates them and each tier exposes its own ``[experts, ...]`` slice.

    The slices are views, not copies. Writing one -- which is how both
    ``build_factor_state`` and the saved-artifact loader fill a tier -- writes
    the store the kernels read, and the store is the only allocation, so the
    factors are never held twice.
    """

    def __init__(
        self,
        expert_ids: torch.Tensor,
        *,
        expert_offset: int,
        hidden_size: int,
        intermediate_size: int,
        gate_up_rank: int,
        down_rank: int,
        spans: Mapping[str, tuple[int, tuple[int, ...]]],
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.expert_offset = int(expert_offset)
        self.experts = int(expert_ids.numel())
        self.gate_up_rank = int(gate_up_rank)
        self.down_rank = int(down_rank)
        self.register_buffer("expert_ids", expert_ids.to(device=device, dtype=torch.long))
        # {attribute: (start, shape)} into the block store of the same name kind.
        self.spans = {str(key): (int(start), tuple(shape))
                      for key, (start, shape) in spans.items()}

    FACTOR_STORES = {
        "gu_input_factor": "gu_input_store",
        "gate_output_factor": "gate_output_store",
        "up_output_factor": "up_output_store",
        "down_input_factor": "down_input_store",
        "down_output_factor": "down_output_store",
    }

    # Factors whose store is rank-major but whose view is [outputs, rank]. The
    # kernels need every tile's row stride to be a compile-time model dimension
    # rather than the expert's own rank (see triton_hybrid_svd_gmm); the builder
    # and the saved artifacts use the [outputs, rank] orientation. So the store
    # is transposed and the view transposes it back -- a view, not a copy, so
    # filling it still fills the store the kernels read.
    TRANSPOSED_VIEWS = frozenset(
        {"gate_output_factor", "up_output_factor", "down_output_factor"}
    )

    def bind_views(self, block: nn.Module) -> None:
        """(Re)build the tier's factor views over the block's current stores."""

        for attribute, store_name in self.FACTOR_STORES.items():
            start, shape = self.spans[attribute]
            count = 1
            for size in shape:
                count *= int(size)
            # `detach()` before slicing: the view shares the store's memory but
            # not its autograd identity, so the in-place writes that fill a tier
            # are not an in-place edit of a parameter's view. These factors are
            # deployed, never trained -- the kernels are custom ops and carry no
            # backward either way.
            store = getattr(block, store_name).detach()
            view = store[start : start + count].view(*shape)
            if attribute in self.TRANSPOSED_VIEWS:
                view = view.transpose(1, 2)
            # A plain attribute, deliberately: the store is the parameter, and a
            # second Parameter over the same memory would double every count.
            object.__setattr__(self, attribute, view)

    def expert_parameter_count(self) -> int:
        total = 0
        for attribute in self.FACTOR_STORES:
            _, shape = self.spans[attribute]
            count = 1
            for size in shape:
                count *= int(size)
            total += count
        return int(total)


class Qwen3MoeHybridSVDBlock(nn.Module):
    """Exact / heterogeneous ordinary-SVD / removed MoE block (Qwen3-MoE, Gemma 4, DeepSeek-V2)."""

    router_fp32 = False     # set from the source block (DeepSeek-V2 computes fp32 router logits)

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        *,
        activation: str = SILU,
        full_ids: torch.Tensor,
        tier_specs: Sequence[tuple[torch.Tensor, int, int]],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        # Expert nonlinearity: SiLU for Qwen3, GeLU-tanh for Gemma 4.
        self.activation = str(activation)
        self._act = activation_code(self.activation)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.full_experts = int(full_ids.numel())
        self.register_buffer("router_weights", torch.empty(
            num_experts, hidden_size, device=device, dtype=dtype
        ))
        self.register_buffer("full_expert_ids", full_ids.to(device=device, dtype=torch.long))
        self.register_buffer(
            "full_active_slots",
            torch.arange(self.full_experts, device=device, dtype=torch.long),
            persistent=False,
        )
        # One flat store per factor kind, holding every tier's experts back to
        # back in local-slot order, with per-expert element offsets. Ranks stay
        # ragged: padding every tier up to the widest rank would hand back a
        # third of what the plan bought, so the merged kernels read each
        # expert's own slab instead. See triton_hybrid_svd_gmm.
        specs = [
            (ids.to(device=device, dtype=torch.long), int(gate_up_rank), int(down_rank))
            for ids, gate_up_rank, down_rank in tier_specs
        ]
        # Rank-major for the three output-side factors -- see
        # RoutedFunctionalSVDTier.TRANSPOSED_VIEWS.
        store_shapes = {
            "gu_input_store": lambda n, gu, dn: (n, gu, hidden_size),
            "gate_output_store": lambda n, gu, dn: (n, gu, intermediate_size),
            "up_output_store": lambda n, gu, dn: (n, gu, intermediate_size),
            "down_input_store": lambda n, gu, dn: (n, dn, intermediate_size),
            "down_output_store": lambda n, gu, dn: (n, dn, hidden_size),
        }
        attribute_of = {
            store: attribute
            for attribute, store in RoutedFunctionalSVDTier.FACTOR_STORES.items()
        }
        cursors = {name: 0 for name in store_shapes}
        tier_spans: list[dict[str, tuple[int, tuple[int, ...]]]] = []
        for ids, gate_up_rank, down_rank in specs:
            count = int(ids.numel())
            spans = {}
            for name, shape_of in store_shapes.items():
                shape = shape_of(count, gate_up_rank, down_rank)
                size = 1
                for extent in shape:
                    size *= int(extent)
                spans[attribute_of[name]] = (cursors[name], shape)
                cursors[name] += size
            tier_spans.append(spans)
        for name, size in cursors.items():
            setattr(self, name, nn.Parameter(
                torch.empty(size, device=device, dtype=dtype)
            ))

        tiers = []
        offset = self.full_experts
        selected = [self.full_expert_ids]
        gate_up_ranks: list[int] = []
        down_ranks: list[int] = []
        for (ids, gate_up_rank, down_rank), spans in zip(specs, tier_spans):
            tier = RoutedFunctionalSVDTier(
                ids,
                expert_offset=offset,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                gate_up_rank=gate_up_rank,
                down_rank=down_rank,
                spans=spans,
                device=device,
            )
            tiers.append(tier)
            selected.append(ids)
            count = int(ids.numel())
            gate_up_ranks.extend([gate_up_rank] * count)
            down_ranks.extend([down_rank] * count)
            offset += count
        self.svd_tiers = nn.ModuleList(tiers)
        self.kept = int(offset)
        self.low_rank_experts = self.kept - self.full_experts
        self.max_gate_up_rank = max(gate_up_ranks) if gate_up_ranks else 0
        self.max_down_rank = max(down_ranks) if down_ranks else 0
        # Per-low-rank-expert rank and store offset, indexed by the expert's
        # position in the low-rank pool (local slot minus `full_experts`).
        table = lambda values: torch.as_tensor(
            values, device=device, dtype=torch.long
        )
        self.register_buffer("gate_up_ranks", table(gate_up_ranks), persistent=False)
        self.register_buffer("down_ranks", table(down_ranks), persistent=False)
        for attribute, buffer_name in (
            ("gu_input_factor", "gu_input_base"),
            ("gate_output_factor", "gu_output_base"),
            ("down_input_factor", "down_input_base"),
            ("down_output_factor", "down_output_base"),
        ):
            bases: list[int] = []
            for tier, spans in zip(tiers, tier_spans):
                start, shape = spans[attribute]
                stride = 1
                for extent in shape[1:]:
                    stride *= int(extent)
                bases.extend(start + slot * stride for slot in range(tier.experts))
            self.register_buffer(buffer_name, table(bases), persistent=False)
        self.bind_tier_views()
        if not (0 < self.top_k <= self.kept <= self.num_experts):
            raise ValueError(
                f"retained pool ({self.kept}) must be between top_k "
                f"({self.top_k}) and num_experts ({self.num_experts})"
            )
        selected_ids = torch.cat(selected)
        if int(torch.unique(selected_ids).numel()) != self.kept:
            raise ValueError("full and low-rank expert ids must be unique and disjoint")
        if bool(((selected_ids < 0) | (selected_ids >= self.num_experts)).any()):
            raise ValueError("expert id outside the original router")
        self.register_buffer("selected_expert_ids", selected_ids)
        # Router logits price the ORIGINAL expert. A tiered expert no longer
        # delivers that function, and the router has no way to know: its logit is
        # a stale quote. This buffer discounts the quote before the softmax, so a
        # damaged expert both wins fewer top-k slots and carries less weight in
        # the ones it does win. Zero for full experts, and all-zero restores the
        # untouched routing bit-for-bit, so the default deployment is unchanged.
        self.register_buffer(
            "routing_penalty",
            torch.zeros(self.kept, device=device, dtype=torch.float32),
        )
        # What a token does when a removed expert was among its ORIGINAL top-k.
        # "reroute" (default, bit-identical to before): top-k over the retained
        # experts, so the next-best retained expert takes the slot. "drop": keep
        # only the retained members of the original top-k and renormalize over
        # them, so no low-affinity substitute is pulled in.
        self.removed_policy = "reroute"
        self._compute_dtype = dtype

    def bind_tier_views(self) -> None:
        """Point every tier's factor views at this block's current stores."""

        for tier in self.svd_tiers:
            tier.bind_views(self)

    def _apply(self, *args, **kwargs):
        # `.to()`, `.cuda()` and friends replace a parameter's storage. The tier
        # views alias the stores, so they have to be rebuilt or they would keep
        # serving the old device's memory while the kernels read the new.
        module = super()._apply(*args, **kwargs)
        module.bind_tier_views()
        return module

    @torch.no_grad()
    def set_routing_penalty(self, penalty_by_expert: Mapping[int, float]) -> None:
        """Discount tiered experts' router logits, keyed by ORIGINAL expert id.

        Applied in logit space, so the discount is scale-free: subtracting
        ``lam`` multiplies the expert's pre-normalization probability by
        ``exp(-lam)`` regardless of how peaked the layer's routing happens to be.
        Experts absent from the mapping keep a zero discount.
        """

        values = torch.zeros_like(self.routing_penalty)
        selected = self.selected_expert_ids.tolist()
        position = {int(expert): slot for slot, expert in enumerate(selected)}
        for expert, penalty in penalty_by_expert.items():
            slot = position.get(int(expert))
            if slot is None:
                # A removed expert has no slot; a penalty for it is a plan/config
                # mismatch, not a no-op worth swallowing silently.
                raise ValueError(
                    f"routing penalty names expert {int(expert)}, which this "
                    "block does not retain"
                )
            values[slot] = float(penalty)
        if bool((values[: self.full_experts] != 0).any()):
            raise ValueError("full experts are exact and must not be discounted")
        self.routing_penalty.copy_(values)

    def set_removed_policy(self, policy: str) -> None:
        if policy not in ("reroute", "drop"):
            raise ValueError(f"removed_policy must be 'reroute' or 'drop', got {policy!r}")
        self.removed_policy = policy

    def _routing_weights(self, inputs: torch.Tensor):
        if not self.norm_topk_prob:
            return self._raw_prob_routing_weights(inputs)
        kept_router = self.router_weights.index_select(0, self.selected_expert_ids)
        logits = F.linear(inputs, kept_router).float() - self.routing_penalty
        values, slots = torch.topk(logits, k=self.top_k, dim=-1)
        if self.removed_policy == "drop":
            return self._drop_removed_weights(inputs, logits, values, slots)
        weights = F.softmax(values, dim=-1).to(inputs.dtype)
        return weights, slots

    def _raw_prob_routing_weights(self, inputs: torch.Tensor):
        """norm_topk_prob=False routers (DeepSeek-V2): the router is untouched.

        Softmax over ALL original experts, removed experts masked after it, top-k
        among the retained, weights = their original probabilities. Re-softmaxing
        over the retained rows (what deleting router rows does) would inflate every
        survivor by 1 / (1 - removed mass) -- see moe_arch.retained_topk_routing.
        """
        if bool((self.routing_penalty != 0).any()):
            raise NotImplementedError("routing_penalty is not defined for norm_topk_prob=False routers")
        full_logits = router_logits(inputs, self.router_weights, fp32=self.router_fp32)
        weights, slots = retained_topk_routing(
            full_logits, self.selected_expert_ids, self.top_k, norm_topk_prob=False
        )
        if self.removed_policy == "drop":
            # Keep only members of the ORIGINAL top-k; a token whose whole original
            # top-k was removed falls back to the rerouted slots.
            original = torch.topk(full_logits, k=self.top_k, dim=-1).indices
            chosen = self.selected_expert_ids.index_select(0, slots.reshape(-1)).reshape(slots.shape)
            member = (chosen.unsqueeze(-1) == original.unsqueeze(1)).any(dim=-1)
            orphan = ~member.any(dim=-1, keepdim=True)
            weights = torch.where(orphan, weights, weights * member)
        return weights.to(inputs.dtype), slots

    def _drop_removed_weights(self, inputs, logits, values, slots):
        """Weights for "drop": zero every slot outside the ORIGINAL top-k.

        Slots keep the rerouted top-k shape so both forwards run unchanged; a
        dropped slot just carries weight 0. A token whose whole original top-k
        was removed has nothing to keep and falls back to rerouting.
        """
        full_logits = F.linear(inputs, self.router_weights).float()
        original = torch.topk(full_logits, k=self.top_k, dim=-1).indices
        chosen = self.selected_expert_ids.index_select(0, slots.reshape(-1)).reshape(slots.shape)
        member = (chosen.unsqueeze(-1) == original.unsqueeze(1)).any(dim=-1)
        orphan = ~member.any(dim=-1, keepdim=True)
        member = member | orphan
        if not self.norm_topk_prob:
            # norm_topk_prob=False routing (incl. "drop") is _raw_prob_routing_weights:
            # a softmax over the kept rows (`logits`) must never produce its weights.
            raise AssertionError("norm_topk_prob=False routing must use _raw_prob_routing_weights")
        weights = F.softmax(values.masked_fill(~member, float("-inf")), dim=-1)
        return weights.to(inputs.dtype), slots

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape
        flat = x.reshape(-1, hidden)
        weights, slots = self._routing_weights(flat)
        output = torch.zeros_like(flat)
        for local_slot in range(self.kept):
            token_ids, positions = (slots == local_slot).nonzero(as_tuple=True)
            expert_input = flat.index_select(0, token_ids)
            if local_slot < self.full_experts:
                gate = F.linear(expert_input, self.full_gate[local_slot])
                up = F.linear(expert_input, self.full_up[local_slot])
                expert_output = F.linear(
                    activation_fn(gate, self.activation) * up, self.full_down[local_slot]
                )
            else:
                expert_output = None
                for tier in self.svd_tiers:
                    if tier.expert_offset <= local_slot < tier.expert_offset + tier.experts:
                        tier_slot = local_slot - tier.expert_offset
                        latent = F.linear(expert_input, tier.gu_input_factor[tier_slot])
                        gate = F.linear(latent, tier.gate_output_factor[tier_slot])
                        up = F.linear(latent, tier.up_output_factor[tier_slot])
                        middle = activation_fn(gate, self.activation) * up
                        down_latent = F.linear(middle, tier.down_input_factor[tier_slot])
                        expert_output = F.linear(
                            down_latent, tier.down_output_factor[tier_slot]
                        )
                if expert_output is None:
                    raise RuntimeError("local expert slot is not assigned to a rank tier")
            output.index_add_(
                0,
                token_ids,
                expert_output * weights[token_ids, positions].unsqueeze(1),
            )
        return output.reshape(batch, sequence, hidden)

    def _cuda_grouped_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape
        flat = x.reshape(-1, hidden)
        tokens = int(flat.shape[0])
        weights, slots = self._routing_weights(flat)
        expert_slots, sorted_token_ids = torch.sort(slots.reshape(-1))
        counts = torch.zeros(
            self.kept, device=slots.device, dtype=torch.long
        ).scatter_add_(0, expert_slots, torch.ones_like(expert_slots))
        token_offsets = torch.cat((
            torch.zeros(1, device=slots.device, dtype=torch.long),
            torch.cumsum(counts, dim=0),
        ))
        assignments = int(sorted_token_ids.numel())

        if (
            _STREAM_OVERLAP
            and self.stream_overlap_supported
            and self.full_experts
            and self.low_rank_experts
        ):
            output = self._stream_overlap_forward(
                flat, weights, token_offsets, sorted_token_ids, tokens, assignments
            )
            return output.to(x.dtype).reshape(batch, sequence, hidden)

        # Two launches per stage, not two per stage per tier. Every retained
        # slot is exact or low-rank, and the assignments are sorted by slot, so
        # the exact kernel and the ragged kernel write *disjoint, contiguous*
        # row ranges that together cover the buffer, and each reads back only
        # its own rows. So the buffer needs no zero-fill and neither producer
        # needs a private copy to add in afterwards.
        intermediate = self._full_gate_up(
            flat, token_offsets, sorted_token_ids, assignments
        )
        if self.low_rank_experts:
            self._low_rank_gate_up(
                intermediate, flat, token_offsets, sorted_token_ids, tokens, assignments
            )

        flat_weights = weights.reshape(-1)
        # One FP32 accumulator for every expert's routed contribution. Both down
        # kernels already atomic_add into FP32 for the arrival-order reason set
        # out in the bf16 down-GMM kernel; sharing it extends that to the
        # exact/low-rank sum, which the per-tier BF16 adds it replaces did not.
        output = self._full_down(
            intermediate, token_offsets, sorted_token_ids, flat_weights, tokens
        )
        if self.low_rank_experts:
            self._low_rank_down(
                output, intermediate, token_offsets, sorted_token_ids,
                flat_weights, assignments,
            )
        return output.to(x.dtype).reshape(batch, sequence, hidden)

    # A subclass whose full-expert steps allocate their own buffers cannot use the
    # two-stream path, which has to own the allocation to order it before the fork.
    stream_overlap_supported = True

    def _stream_overlap_forward(
        self, flat, weights, token_offsets, sorted_token_ids, tokens, assignments
    ) -> torch.Tensor:
        """Both expert stages with the low-rank chain on a second stream.

        One opaque custom op rather than stream contexts inline here: see
        ``hybrid_moe_experts_overlapped`` for why Dynamo rejects the inline form,
        and for the measured per-layer effect. The op also owns the buffer
        allocation, which the single-stream ``_full_gate_up`` / ``_full_down`` do --
        the fork has to be ordered after the allocation, not after the kernel the
        allocating step launches.
        """
        return hybrid_moe_experts_overlapped(
            flat, weights.reshape(-1), token_offsets, sorted_token_ids,
            self.full_gate, self.full_up, self.full_down, self.full_active_slots,
            self.gu_input_store, self.gu_input_base, self.gate_up_ranks,
            self.gate_output_store, self.up_output_store, self.gu_output_base,
            self.down_input_store, self.down_input_base, self.down_ranks,
            self.down_output_store, self.down_output_base,
            self.top_k, self.full_experts, self.hidden_size,
            self.intermediate_size, self.max_gate_up_rank, self.max_down_rank,
            tokens, assignments, self._act,
        )

    # The four storage-dependent steps of the grouped forward. A subclass that
    # stores the experts differently (the HQQ INT4 block) overrides only these.
    def _low_rank_gate_up(
        self, intermediate, flat, token_offsets, sorted_token_ids, tokens, assignments
    ) -> None:
        """Write the low-rank experts' rows of the shared SwiGLU intermediate."""

        latent = flat.new_empty((assignments, self.max_gate_up_rank))
        fused_ragged_projection_gmm_store(
            latent,
            flat,
            self.gu_input_store,
            self.gu_input_base,
            self.gate_up_ranks,
            token_offsets,
            sorted_token_ids,
            self.top_k,
            self.full_experts,
            self.hidden_size,
            False,
        )
        fused_ragged_gate_up_gmm_silu_store(
            intermediate,
            latent,
            self.gate_output_store,
            self.up_output_store,
            self.gu_output_base,
            self.gate_up_ranks,
            token_offsets,
            tokens,
            self.full_experts,
            self.max_gate_up_rank,
            self._act,
        )

    def _low_rank_down(
        self, output, intermediate, token_offsets, sorted_token_ids,
        flat_weights, assignments,
    ) -> None:
        """Atomically add the low-rank experts' routed outputs into ``output``."""

        down_latent = intermediate.new_empty((assignments, self.max_down_rank))
        fused_ragged_projection_gmm_store(
            down_latent,
            intermediate,
            self.down_input_store,
            self.down_input_base,
            self.down_ranks,
            token_offsets,
            sorted_token_ids,
            self.top_k,
            self.full_experts,
            self.intermediate_size,
            True,
        )
        fused_ragged_down_gmm_reduction_accumulate(
            output,
            down_latent,
            self.down_output_store,
            self.down_output_base,
            self.down_ranks,
            token_offsets,
            sorted_token_ids,
            flat_weights,
            self.top_k,
            self.full_experts,
            self.max_down_rank,
        )

    def _full_gate_up(
        self, flat, token_offsets, sorted_token_ids, assignments
    ) -> torch.Tensor:
        """Allocate the [assignments, intermediate] buffer and fill the exact rows."""

        intermediate = flat.new_empty((assignments, self.intermediate_size))
        if self.full_experts:
            fused_gate_up_gmm_silu_store(
                intermediate,
                flat,
                self.full_gate,
                self.full_up,
                self.full_active_slots,
                token_offsets,
                sorted_token_ids,
                self.top_k,
                self._act,
            )
        return intermediate

    def _full_down(
        self, intermediate, token_offsets, sorted_token_ids, flat_weights, tokens
    ) -> torch.Tensor:
        """The FP32 [tokens, hidden] accumulator holding the exact experts' sum."""

        output = torch.zeros(
            (tokens, self.hidden_size), device=intermediate.device, dtype=torch.float32
        )
        if self.full_experts:
            fused_down_gmm_reduction_accumulate(
                output,
                intermediate,
                self.full_down,
                self.full_active_slots,
                token_offsets,
                sorted_token_ids,
                flat_weights,
                self.top_k,
            )
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self._cuda_grouped_forward(x)
        return self._eager_forward(x)

    @classmethod
    @torch.no_grad()
    def build_factor_state(
        cls,
        full_block: Qwen3MoeStackedBlock,
        *,
        tier_specs: Sequence[tuple[Sequence[int], int, int]],
        gate_moments: torch.Tensor,
        up_moments: torch.Tensor,
        down_moments: torch.Tensor,
        calibration_hidden: torch.Tensor | None = None,
        router_candidates: int = 32,
        routing_batch_size: int = 2048,
        scale_floor: float = 1e-12,
        randomized: bool = True,
        layer_seed: int = 0,
        gram_metric: bool | None = None,
        gram_shrink_mode: str = "diagonal",
        down_compensation: bool = False,
        factor_calibration: str = "shortlist",
        factor_output_metric: str = "none",
        retained_ids: Sequence[int] | None = None,
    ) -> dict:
        gate_weights = full_block.gate_proj_stacked
        up_weights = full_block.up_proj_stacked
        down_weights = full_block.down_proj_stacked
        device, dtype = gate_weights.device, gate_weights.dtype
        experts = int(gate_weights.shape[0])

        factor_calibration = str(factor_calibration)
        if factor_calibration not in FACTOR_CALIBRATIONS:
            raise ValueError(
                f"factor_calibration must be one of {FACTOR_CALIBRATIONS}, "
                f"got {factor_calibration!r}"
            )
        use_recorded_moments = factor_calibration == "shortlist"
        if gram_metric and calibration_hidden is None:
            raise ValueError(
                "gram factor metric requires calibration hidden states, but "
                "calibration_hidden is None. Set hidden_cache_dir in the method "
                "config; a diagonal moments artifact cannot substitute."
            )
        if not use_recorded_moments and calibration_hidden is None:
            raise ValueError(
                f"factor_calibration={factor_calibration!r} calibrates on each "
                "expert's routed tokens, but calibration_hidden is None. Set "
                "hidden_cache_dir in the method config; the recorded moments "
                "cannot substitute."
            )
        calibration_ids = calibration_weights = None
        if calibration_hidden is not None:
            logits = []
            for start in range(0, int(calibration_hidden.shape[0]), int(routing_batch_size)):
                batch = calibration_hidden[start : start + int(routing_batch_size)].to(
                    device=device, dtype=dtype
                )
                logits.append(router_logits(
                    batch, full_block.router_weights,
                    fp32=bool(getattr(full_block, "router_fp32", False)),
                ).cpu())
            logits = torch.cat(logits)
            retained = None
            if retained_ids is not None:
                # The plan's full and tiered experts. Tiered experts are then
                # calibrated on the traffic the compressed layer actually sends
                # them, including tokens rerouted from removed experts.
                retained = torch.zeros(experts, dtype=torch.bool)
                retained[torch.as_tensor(
                    [int(value) for value in retained_ids], dtype=torch.long
                )] = True
                tiered = [int(value) for ids, _, _ in tier_specs for value in ids]
                if tiered and not bool(retained[torch.as_tensor(tiered)].all()):
                    raise ValueError("every tiered expert must be retained")
            calibration_ids, calibration_weights = factor_calibration_selection(
                logits,
                factor_calibration=factor_calibration,
                top_k=int(full_block.top_k),
                norm_topk_prob=bool(full_block.norm_topk_prob),
                router_candidates=int(router_candidates),
                retained=retained,
            )

        # One Gram for the whole layer, pooled over every calibration token. This
        # is the estimand SVD-LLM-style whitening assumes -- a per-layer input
        # covariance from the full calibration set -- whereas the per-expert Gram
        # below sees only that expert's routed tokens. Used as the shrinkage
        # target it repairs the starved per-expert estimates without discarding
        # the off-diagonal structure that whitening exists to exploit. Computed
        # once per layer, so it costs one extra matmul, not one per expert.
        layer_gram = None
        # "layer_only" is SVD-LLM's granularity: ONE covariance per layer, used by
        # every expert, with no per-expert conditioning at all. alpha = 1 makes the
        # blend return the target unchanged.
        shrink_alpha = 1.0 if str(gram_shrink_mode) == "layer_only" else "auto"
        if gram_metric and calibration_hidden is not None and (
            str(gram_shrink_mode) in ("layer", "layer_only")
        ):
            accumulator = torch.zeros(
                (int(calibration_hidden.shape[1]),) * 2,
                device=device,
                dtype=torch.float32,
            )
            for start in range(
                0, int(calibration_hidden.shape[0]), int(routing_batch_size)
            ):
                batch = calibration_hidden[
                    start : start + int(routing_batch_size)
                ].to(device=device, dtype=torch.float32)
                accumulator += batch.t() @ batch
                del batch
            layer_gram = accumulator.cpu()
            del accumulator

        zero_gate_up = zero_down = 0
        state_tiers = []
        for ids, gate_up_rank, down_rank in tier_specs:
            expert_ids = torch.as_tensor(ids, device=device, dtype=torch.long).reshape(-1)
            factor_lists = {
                "gate_output_factor": [],
                "up_output_factor": [],
                "gu_input_factor": [],
                "down_output_factor": [],
                "down_input_factor": [],
            }
            for expert_tensor in expert_ids:
                expert = int(expert_tensor.item())
                candidate_inputs = candidate_weights = None
                if calibration_ids is not None:
                    token_ids, positions = (calibration_ids == expert).nonzero(as_tuple=True)
                    if int(token_ids.numel()):
                        candidate_inputs = calibration_hidden.index_select(0, token_ids).to(
                            device=device, dtype=torch.float32
                        )
                        candidate_weights = calibration_weights[
                            token_ids, positions
                        ].to(device=device, dtype=torch.float32)
                factors, fallbacks = build_deployed_expert_factors(
                    gate_weights[expert],
                    up_weights[expert],
                    down_weights[expert],
                    gate_moments[expert],
                    up_moments[expert],
                    down_moments[expert],
                    expert_id=expert,
                    gate_up_rank=int(gate_up_rank),
                    down_rank=int(down_rank),
                    layer_seed=int(layer_seed),
                    candidate_inputs=candidate_inputs,
                    candidate_weights=candidate_weights,
                    scale_floor=scale_floor,
                    randomized=randomized,
                    storage_dtype=dtype,
                    gram_metric=gram_metric,
                    # calibration exists for the layer; this expert may still be
                    # absent from every shortlist, which is not a failure.
                    allow_uncalibrated=calibration_ids is not None,
                    activation=str(getattr(full_block, "activation", SILU)),
                    gram_shrink_target=layer_gram,
                    gram_shrink_alpha=shrink_alpha,
                    down_compensation=down_compensation,
                    use_recorded_moments=use_recorded_moments,
                    output_metric=factor_output_metric,
                )
                zero_gate_up += int(
                    fallbacks["gate_up_router_conditioned"]
                    or fallbacks["gate_up_weight_only"]
                )
                zero_down += int(fallbacks["down_weight_only"])
                for key in factor_lists:
                    factor_lists[key].append(factors[key])
            state_tiers.append({
                "expert_ids": expert_ids.detach().cpu(),
                "gate_up_rank": int(gate_up_rank),
                "down_rank": int(down_rank),
                **{
                    key: torch.stack(values).detach()
                    for key, values in factor_lists.items()
                },
            })
        return {
            "format_version": 1,
            "tiers": state_tiers,
            "zero_moment_gate_up_experts": int(zero_gate_up),
            "zero_moment_down_experts": int(zero_down),
        }

    @classmethod
    @torch.no_grad()
    def from_factor_state(
        cls,
        full_block: Qwen3MoeStackedBlock,
        *,
        full_ids: Sequence[int],
        tier_specs: Sequence[tuple[Sequence[int], int, int]],
        factor_state: dict,
    ) -> "Qwen3MoeHybridSVDBlock":
        gate_weights = full_block.gate_proj_stacked
        up_weights = full_block.up_proj_stacked
        down_weights = full_block.down_proj_stacked
        device, dtype = gate_weights.device, gate_weights.dtype
        experts, intermediate, hidden = map(int, gate_weights.shape)
        full_tensor = torch.as_tensor(full_ids, device=device, dtype=torch.long).reshape(-1)
        tensor_specs = [
            (
                torch.as_tensor(ids, device=device, dtype=torch.long).reshape(-1),
                int(gate_up_rank),
                int(down_rank),
            )
            for ids, gate_up_rank, down_rank in tier_specs
            if len(ids)
        ]
        block = cls(
            hidden,
            intermediate,
            experts,
            int(full_block.top_k),
            bool(full_block.norm_topk_prob),
            activation=str(getattr(full_block, "activation", SILU)),
            full_ids=full_tensor,
            tier_specs=tensor_specs,
            dtype=dtype,
            device=device,
        )
        block.router_weights.copy_(full_block.router_weights.to(dtype=dtype))
        block.router_fp32 = bool(getattr(full_block, "router_fp32", False))
        block.full_gate = nn.Parameter(gate_weights.index_select(0, full_tensor).clone())
        block.full_up = nn.Parameter(up_weights.index_select(0, full_tensor).clone())
        block.full_down = nn.Parameter(down_weights.index_select(0, full_tensor).clone())
        saved_tiers = factor_state.get("tiers") or []
        if len(saved_tiers) != len(block.svd_tiers):
            raise ValueError("factor artifact tier count does not match the plan")
        for tier, saved in zip(block.svd_tiers, saved_tiers):
            saved_ids = torch.as_tensor(saved["expert_ids"], dtype=torch.long)
            if not torch.equal(saved_ids.cpu(), tier.expert_ids.cpu()):
                raise ValueError("factor artifact expert ids do not match the plan")
            if (
                int(saved["gate_up_rank"]) != tier.gate_up_rank
                or int(saved["down_rank"]) != tier.down_rank
            ):
                raise ValueError("factor artifact ranks do not match the plan")
            for key in (
                "gate_output_factor",
                "up_output_factor",
                "gu_input_factor",
                "down_output_factor",
                "down_input_factor",
            ):
                target = getattr(tier, key)
                value = torch.as_tensor(saved[key]).to(device=device, dtype=dtype)
                if value.shape != target.shape:
                    raise ValueError(f"factor artifact has invalid {key} shape")
                target.copy_(value)
        block.zero_moment_gate_up_experts = int(
            factor_state.get("zero_moment_gate_up_experts", 0)
        )
        block.zero_moment_down_experts = int(
            factor_state.get("zero_moment_down_experts", 0)
        )
        return block

    @classmethod
    @torch.no_grad()
    def from_stacked(
        cls,
        full_block: Qwen3MoeStackedBlock,
        *,
        full_ids: Sequence[int],
        tier_specs: Sequence[tuple[Sequence[int], int, int]],
        gate_moments: torch.Tensor,
        up_moments: torch.Tensor,
        down_moments: torch.Tensor,
        calibration_hidden: torch.Tensor | None = None,
        router_candidates: int = 32,
        routing_batch_size: int = 2048,
        scale_floor: float = 1e-12,
        randomized: bool = True,
        layer_seed: int = 0,
        gram_metric: bool | None = None,
        gram_shrink_mode: str = "diagonal",
        down_compensation: bool = False,
        factor_calibration: str = "shortlist",
        factor_output_metric: str = "none",
    ) -> "Qwen3MoeHybridSVDBlock":
        state = cls.build_factor_state(
            full_block,
            tier_specs=tier_specs,
            gate_moments=gate_moments,
            up_moments=up_moments,
            down_moments=down_moments,
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
            # The block routes over exactly these experts, so the top-k modes
            # calibrate on that routing.
            retained_ids=[int(value) for value in full_ids] + [
                int(value) for ids, _, _ in tier_specs for value in ids
            ],
        )
        return cls.from_factor_state(
            full_block,
            full_ids=full_ids,
            tier_specs=tier_specs,
            factor_state=state,
        )

    def expert_parameter_count(self) -> int:
        full = sum(int(parameter.numel()) for parameter in (
            self.full_gate, self.full_up, self.full_down
        ))
        return full + sum(tier.expert_parameter_count() for tier in self.svd_tiers)

    def expert_parameter_ratio(self) -> float:
        original = self.num_experts * 3 * self.hidden_size * self.intermediate_size
        return float(self.expert_parameter_count() / original)


# The class was renamed in c7b7da1; the planner, two tools and three tests still
# import the old name, so keep it as an alias.
Qwen3MoeRoutedFunctionalSVDSubsetBlock = Qwen3MoeHybridSVDBlock
