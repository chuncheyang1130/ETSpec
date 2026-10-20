"""Activation-calibrated low-rank factors for expert projections.

Two layers, in dependency order:

  * the INPUT METRIC the SVD is calibrated in — a weighted diagonal second moment
    (``factor_metric: diagonal``, the only kind any of 151 recorded plans has
    used) or a full Gram, plus the shrinkage that makes a starved per-expert Gram
    usable at all;
  * the FACTORISATION — a truncated SVD taken in that metric, then rebalanced so
    both BF16 operands stay well scaled.

Merged here from the former ``skipcat.py`` on 2026-10-20. That module was named
after an interpolative "SkipCat" form (``low_rank_to_skipcat`` / ``apply_skipcat``
/ ``skipcat_parameter_count``) that nothing ever called, while the
activation-calibration helpers it also held are on the live path for every ours
run; the SkipCat half was dropped and the rest moved here.

NOTE on the Gram path: ``gram_shrinkage_coefficient`` and
``shrink_gram_to_diagonal`` are reachable only with ``factor_metric: gram`` or
``down_compensation: true``, and no recorded plan has ever set either — verified
by instrumenting a real b30 run, where ``input_metric_from_samples`` was called
2006 times, every one with ``gram=False``. They are kept because the planner can
still emit such a plan, not because the shipped pipeline exercises them.
"""

from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def gram_shrinkage_coefficient(
    weights: torch.Tensor,
    input_features: int,
) -> float:
    """How far to shrink a sample Gram toward its own diagonal.

    A ``d x d`` Gram estimated from ``N_eff = (sum w)^2 / sum w^2`` weighted
    samples has rank at most ``N_eff``.  When ``N_eff < d`` the estimate carries
    NO information in ``d - N_eff`` directions, yet ``G^(-1/2)`` still whitens
    along them -- with gain ``1/sqrt(scale_floor)``, i.e. 1e6 at the default
    floor.  The factorisation is then arbitrary off the calibration support: it
    fits the observed tokens (better than any constrained estimator can) and
    behaves randomly on everything else.  Measured on Qwen3-30B-A3B coding256,
    **22% of the experts that receive an SVD tier have N_eff < 2048**, and 9%
    have a literally singular Gram.

    ``alpha = d / (d + N_eff)`` is the standard bias/variance trade-off: it goes
    to 1 (pure diagonal) as samples vanish and to 0 (pure Gram) as they become
    abundant, with no free parameter.
    """

    w = weights.detach().double().flatten()
    total = float(w.sum())
    energy = float(w.square().sum())
    if total <= 0 or energy <= 0:
        return 1.0
    n_eff = total * total / energy
    d = float(int(input_features))
    return float(d / (d + n_eff))


@torch.no_grad()
def shrink_gram_to_diagonal(
    gram: torch.Tensor,
    alpha: float,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shrink a sample Gram toward a better-estimated target.

    ``target=None`` shrinks toward ``diag(G)``: ``(1 - alpha) * G + alpha *
    diag(G)``, leaving the diagonal exactly as measured, so ``alpha = 0`` is the
    raw Gram and ``alpha = 1`` is precisely the diagonal metric.

    ``target`` supplies a matrix instead -- in practice the LAYER Gram, pooled
    over every calibration token rather than one expert's routed subset.  That is
    the estimand SVD-LLM-style whitening actually assumes: one covariance per
    linear layer, measured on the full calibration set.  This code estimates one
    per EXPERT from only its routed tokens (~6% of tokens at top-8-of-128), which
    is why ``N_eff < d`` for 22% of tiered experts.  Shrinking toward the layer
    Gram fixes the starvation while KEEPING the off-diagonal correlation
    structure that whitening exists to exploit; shrinking toward the diagonal
    discards exactly that structure, which is why it only recovers parity with
    the diagonal metric rather than beating it.

    The target is trace-matched to ``gram`` before blending, so the two terms are
    commensurate regardless of how each was weighted or how many tokens each saw.
    """

    alpha = float(alpha)
    if alpha <= 0:
        return gram
    if target is None:
        if alpha >= 1:
            return torch.diag(torch.diagonal(gram).clone())
        diagonal = torch.diagonal(gram).clone()
        shrunk = gram * (1.0 - alpha)
        shrunk.diagonal().copy_(diagonal)
        return shrunk
    if target.shape != gram.shape:
        raise ValueError(
            f"shrinkage target {tuple(target.shape)} does not match the Gram "
            f"{tuple(gram.shape)}"
        )
    # The layer Gram is accumulated on the compute device and handed back on CPU,
    # so normalise the device explicitly rather than relying on PyTorch's 0-dim
    # scalar promotion to bridge it.
    target = target.to(device=gram.device, dtype=gram.dtype)
    if alpha >= 1:
        # Pure target. Returning it unscaled keeps the result EXACTLY independent
        # of the expert's own tokens -- the trace-matching scalar below would
        # otherwise perturb the low bits per expert. Scale is irrelevant anyway:
        # metric_square_roots normalises by the largest eigenvalue.
        return target
    reference = torch.diagonal(target).sum()
    if float(reference) <= 0:
        # A target with no energy carries no information; fall back to the
        # diagonal rather than blending in zeros and deflating the metric.
        return shrink_gram_to_diagonal(gram, alpha)
    scale = torch.diagonal(gram).sum() / reference
    return gram * (1.0 - alpha) + (target * scale) * alpha


@torch.no_grad()
def input_metric_from_samples(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    *,
    gram: bool,
    shrinkage: float | str = "auto",
    shrinkage_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted input metric: a diagonal second moment, or a full Gram.

    ``gram=False`` gives ``sum_t w_t * x_t**2`` (one entry per coordinate);
    ``gram=True`` gives ``X^T diag(w) X``.  Callers pick the kind once and every
    downstream consumer infers it from the returned tensor's rank, so a plan and
    the deployment it describes cannot disagree about the metric.

    ``shrinkage`` regularises the Gram toward its diagonal; ``"auto"`` uses
    :func:`gram_shrinkage_coefficient`.  It is ignored when ``gram=False``, so
    the diagonal path is bit-identical.  Pass ``0.0`` for the raw sample Gram --
    which is unusable in deployment, see the note there.
    """

    x = inputs.float()
    w = weights.float()
    if not gram:
        return (w.unsqueeze(1) * x.square()).sum(0)
    metric = (x * w.unsqueeze(1)).t() @ x
    alpha = (
        gram_shrinkage_coefficient(w, int(x.shape[1]))
        if isinstance(shrinkage, str)
        else float(shrinkage)
    )
    return shrink_gram_to_diagonal(metric, alpha, shrinkage_target)


@torch.no_grad()
def uninformative_input_metric(
    input_features: int,
    *,
    gram: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The metric meaning "no calibration evidence": ones, or the identity.

    Both reduce the calibrated factorisation to an ordinary weight SVD, which is
    the documented fallback for an expert the calibration never routed to.
    """

    if gram:
        return torch.eye(int(input_features), device=device, dtype=dtype)
    return torch.ones(int(input_features), device=device, dtype=dtype)


@torch.no_grad()
def metric_square_roots(
    second_moment: torch.Tensor,
    input_features: int,
    *,
    scale_floor: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Return ``(metric, inverse, is_matrix)`` for an input metric.

    A 1-D ``second_moment`` is the diagonal metric: ``sqrt(m / max(m))`` floored
    relative to its largest entry, returned as a vector so the caller keeps the
    elementwise fast path.  A 2-D ``second_moment`` is a full Gram
    ``G = Z^T diag(w) Z``; the metric is then ``G^(1/2)`` normalised by its
    largest eigenvalue, returned as a matrix.

    Minimising ``E_w||(W - W_r) z||^2`` is minimising ``||(W - W_r) G^(1/2)||_F``,
    so the rank-r optimum is the truncated SVD of ``W @ G^(1/2)`` mapped back by
    ``G^(-1/2)``.  A diagonal metric is the special case ``G = diag(m)``, which is
    only optimal when the input coordinates are uncorrelated.

    ``scale_floor`` bounds the smallest eigenvalue relative to the largest.  That
    keeps the inverse finite on directions the calibration never visited, which
    is the same role it plays for the diagonal metric, and it bounds the
    condition number of a Gram estimated from finitely many tokens.
    """

    moment = second_moment.detach().float().nan_to_num(nan=0.0)
    floor = float(scale_floor)
    if moment.ndim == 1:
        if moment.numel() != int(input_features):
            raise ValueError(
                f"diagonal metric has {moment.numel()} entries, expected {input_features}"
            )
        moment = moment.clamp_min(0)
        largest = float(moment.max().item())
        # A never-routed expert can become reachable after the router is
        # restricted to the retained set.  Fall back to ordinary weight SVD
        # rather than constructing a meaningless zero metric.
        scale = (
            torch.ones_like(moment)
            if largest <= 0
            else (moment / largest).clamp_min(floor).sqrt()
        )
        return scale, scale.reciprocal(), False
    if moment.ndim != 2 or moment.shape != (int(input_features), int(input_features)):
        raise ValueError(
            f"gram metric has shape {tuple(moment.shape)}, expected "
            f"({input_features}, {input_features})"
        )
    # eigh needs exact symmetry; an accumulated Z^T diag(w) Z is only symmetric
    # to rounding.  Do the decomposition in FP64 -- the Gram's dynamic range
    # spans many orders of magnitude and FP32 eigenvectors of the small
    # eigenvalues are not accurate enough to invert.
    gram = moment.double()
    gram = 0.5 * (gram + gram.t())
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    largest = float(values.max().item())
    if largest <= 0:
        identity = torch.eye(
            int(input_features), device=moment.device, dtype=moment.dtype
        )
        return identity, identity, True
    values = (values / largest).clamp_min(floor)
    root = values.sqrt()
    metric = (vectors * root) @ vectors.t()
    inverse = (vectors * root.reciprocal()) @ vectors.t()
    return (
        metric.to(moment.dtype),
        inverse.to(moment.dtype),
        True,
    )


@torch.no_grad()
def calibrated_low_rank_factors(
    weight: torch.Tensor,
    second_moment: torch.Tensor,
    rank: int,
    *,
    scale_floor: float = 1e-12,
    randomized: bool = False,
    oversample: int = 8,
    niter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``left, right`` for activation-calibrated truncated SVD.

    ``second_moment`` is the input metric: a 1-D diagonal, or a 2-D Gram
    ``Z^T diag(w) Z`` which is the correct metric when the input coordinates are
    correlated.  See :func:`metric_square_roots`.  The returned factors are in
    the original input coordinates, so ``inputs @ right.T @ left.T`` applies the
    approximation directly, whichever metric was used.
    """

    w = weight.detach().float()
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(w.shape)}")
    rank = int(rank)
    limit = min(w.shape)
    if not (0 < rank <= limit):
        raise ValueError(f"rank {rank} must be in [1, {limit}]")
    metric, inverse, is_matrix = metric_square_roots(
        second_moment.to(device=w.device),
        int(w.shape[1]),
        scale_floor=scale_floor,
    )
    weighted = w @ metric if is_matrix else w * metric.unsqueeze(0)
    if randomized and rank < limit:
        q = min(limit, rank + max(0, int(oversample)))
        u, s, v = torch.svd_lowrank(weighted, q=q, niter=int(niter))
        order = torch.argsort(s, descending=True)[:rank]
        u, s, vh = u[:, order], s[order], v[:, order].t()
    else:
        u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
        u, s, vh = u[:, :rank], s[:rank], vh[:rank]
    left = u * s.unsqueeze(0)
    right = vh @ inverse if is_matrix else vh / metric.unsqueeze(0)
    return left, right

@torch.no_grad()
def balance_low_rank_factors(
    left: torch.Tensor,
    right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rebase ``left @ right`` into balanced factors without changing it.

    Activation calibration can make ``right`` anisotropic because it divides by
    the input scales after decomposing the scaled weight.  We first obtain
    orthonormal bases for both factor spaces, then SVD only their small ``rank x
    rank`` core and split its singular values evenly between the two returned
    factors.  This keeps both BF16 operands well scaled while preserving the
    calibrated approximation in FP32.
    """

    l = left.detach().float()
    r = right.detach().float()
    if l.ndim != 2 or r.ndim != 2 or l.shape[1] != r.shape[0]:
        raise ValueError(f"incompatible low-rank factors {tuple(l.shape)} and {tuple(r.shape)}")

    q_left, triangular_left = torch.linalg.qr(l, mode="reduced")
    q_right, triangular_right = torch.linalg.qr(r.t(), mode="reduced")
    core = triangular_left @ triangular_right.t()
    core_left, singular_values, core_right = torch.linalg.svd(
        core, full_matrices=False
    )
    root = singular_values.clamp_min(0).sqrt()
    balanced_left = (q_left @ core_left) * root.unsqueeze(0)
    balanced_right = root.unsqueeze(1) * (core_right @ q_right.t())
    return balanced_left, balanced_right


@torch.no_grad()
def calibrated_balanced_low_rank_factors(
    weight: torch.Tensor,
    second_moment: torch.Tensor,
    rank: int,
    *,
    scale_floor: float = 1e-12,
    randomized: bool = False,
    oversample: int = 8,
    niter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return balanced ordinary factors for an activation-calibrated SVD."""

    left, right = calibrated_low_rank_factors(
        weight,
        second_moment,
        rank,
        scale_floor=scale_floor,
        randomized=randomized,
        oversample=oversample,
        niter=niter,
    )
    return balance_low_rank_factors(left, right)
