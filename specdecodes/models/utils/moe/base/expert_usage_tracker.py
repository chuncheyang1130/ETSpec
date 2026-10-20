"""
Mass-weighted expert-usage tracker + shared MoE utilities.

This module is the shared base for the Qwen3-MoE stacked family. It ships **no**
expert block of its own; it provides:

  1. The expert-usage tracker (`install_expert_usage_tracker`, `get_expert_usage`,
     `reset_expert_usage`, `pick_top_n_per_layer`, and the accept-aware
     `set_tracker_stash_mode` / `commit_expert_usage_accepted`). The tracker
     accumulates per-expert **routing mass** (sum of the actual top-k softmax
     weights after Qwen3's `norm_topk_prob=True` renormalization), not bincount hit
     counts, by re-running the *full* router as a forward pre-hook — so it works on
     the raw HF block and on every stacked swap variant.

  2. Small cross-block helpers: `_is_qwen3_moe_block`, `_compute_router_logits`, and
     `_set_module_by_name` (generic submodule replacement).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



_TRACKER_BUFFER = "_expert_usage_mass"
_TRACKER_HANDLE = "_expert_usage_handle"
# Accept-aware tracking: when stash mode is on, the hook defers (stashes) the
# current forward's per-token mass instead of committing it, so an external
# caller can later commit only the accepted tree positions (see
# `set_tracker_stash_mode` / `commit_expert_usage_accepted`).
_TRACKER_STASH_MODE = "_expert_usage_stash_mode"
_TRACKER_PENDING = "_expert_usage_pending"
# Optional second signal: per-expert IMPORTANCE = Σ g·‖f_e(x)‖ ("mass × activation"),
# the first-order cost of dropping the expert (routing value × output magnitude at
# the moment it fires). Opt-in via `install_expert_usage_tracker(..., track_activation=True)`
# and read with `get_expert_importance`. Off by default, so the plain mass path and
# every existing caller are unchanged.
_TRACKER_IMPORTANCE = "_expert_importance"
_TRACKER_TRACK_ACT = "_expert_track_activation"
# Per-expert ACTIVATION COUNT N_e = number of tokens for which the expert is in top-k.
# Paired with `_TRACKER_IMPORTANCE` so callers can form the frequency-NORMALIZED signal
# avg_e = importance_e / N_e = (1/N_e) Σ g·‖f_e‖ (REAP-style per-firing saliency), vs the
# frequency-WEIGHTED sum we track by default. Allocated only when track_activation=True.
_TRACKER_COUNT = "_expert_count"
# Exponent `p` on the activation magnitude in the importance signal: Σ g·‖f_e‖^p.
# p=1 (default) is the plain L2 norm — the first-order cost of dropping the expert.
# p=2 weighs each firing by its squared norm (energy), which sharpens the signal toward
# the few tokens where an expert injects a large vector. Set via
# `install_expert_usage_tracker(..., activation_power=...)`.
_TRACKER_ACT_POW = "_expert_act_power"
# Optional boolean mask over the current forward's `[batch, sequence]` positions.
# This lets an offline teacher-forced pass retain the prompt as causal context while
# accumulating statistics only for response tokens. `None` means track every token.
_TRACKER_TOKEN_MASK = "_expert_usage_token_mask"


def _is_qwen3_moe_block(module: nn.Module) -> bool:
    """Heuristic check: HF `Qwen3MoeSparseMoeBlock` OR a stacked swap variant.

    The stacked family is matched by name (`Qwen3MoeStackedBlock` and its
    INT4 subclass `Qwen3MoeStackedInt4Block`, the reduced-target block) so the
    mass tracker installs on it even when the target is itself an expert subset —
    the tracker reruns the *full* router (`router_weights`), so mass is still
    accumulated over all experts.
    """
    name = module.__class__.__name__
    if name == "Qwen3MoeSparseMoeBlock":
        return all(
            hasattr(module, a) for a in ("experts", "gate", "num_experts", "top_k")
        )
    if name in (
        "Qwen3MoeStackedBlock",
        "Qwen3MoeStackedInt4Block",
        "Qwen3MoeStackedSubsetBlock",  # offline fidelity-sweep block
    ):
        return all(
            hasattr(module, a) for a in ("router_weights", "num_experts", "top_k")
        )
    return False


def _compute_router_logits(module: nn.Module, flat: torch.Tensor) -> torch.Tensor:
    """Apply the block's router to flat hidden states, regardless of block type."""
    if hasattr(module, "router_weights"):       # Qwen3MoeStackedBlock
        return F.linear(flat, module.router_weights)
    return module.gate(flat)                    # HF Qwen3MoeSparseMoeBlock


@torch.no_grad()
def _expert_contribution_norm(
    module: nn.Module, flat: torch.Tensor,
    topk_idx: torch.Tensor, topk_vals: torch.Tensor,
    power: int = 1,
) -> torch.Tensor:
    """Per-expert `Σ g·‖f_e(x)‖^power` over this forward's firings -> `[E]` float32.

    `f_e(x) = W_down( SiLU(W_gate·x) ⊙ (W_up·x) )` is the vector the expert injects
    into the residual stream; weighting its L2 norm by the routing value `g` gives the
    first-order cost of dropping `e` on that token ("mass × activation, at the moment").
    `power=2` uses the squared norm (energy) instead, which weighs the few large-output
    firings far more heavily than the many small ones — a different saliency, not a
    rescaling, since the per-expert sums no longer rank the same way.
    Grouped by expert so no weights are replicated per token. Reads the stacked store
    (`gate/up/down_proj_stacked`) or, on a raw HF block, `experts[e]`; if neither is
    exposed (e.g. an INT4-packed store) it returns zeros (importance simply not tracked).
    This duplicates the expert math, so it is meant for the eager, offline calibration
    pass only — never the compiled decode path.
    """
    E = int(module.num_experts)
    out = torch.zeros(E, dtype=torch.float32, device=flat.device)
    Wg = getattr(module, "gate_proj_stacked", None)
    Wu = getattr(module, "up_proj_stacked", None)
    Wd = getattr(module, "down_proj_stacked", None)
    experts = getattr(module, "experts", None)
    stacked = Wg is not None and Wu is not None and Wd is not None
    if not stacked and experts is None:
        return out
    for e in topk_idx.unique().tolist():
        tok_idx, slot_idx = (topk_idx == e).nonzero(as_tuple=True)
        if tok_idx.numel() == 0:
            continue
        x_e = flat.index_select(0, tok_idx)                  # [n_e, H]
        g_e = topk_vals[tok_idx, slot_idx].float()           # [n_e]  routing value
        if stacked:
            h = F.silu(x_e @ Wg[e].t()) * (x_e @ Wu[e].t())  # [n_e, IM]
            f = h @ Wd[e].t()                                # [n_e, H]
        else:
            ex = experts[e]
            h = F.silu(ex.gate_proj(x_e)) * ex.up_proj(x_e)
            f = ex.down_proj(h)
        n_e = f.float().norm(dim=-1)                         # ‖f‖₂ per token
        if power != 1:
            n_e = n_e.pow(power)
        out[e] = (g_e * n_e).sum()                           # Σ g·‖f‖^power
    return out


# ---------------------------------------------------------------------------
# Mass-weighted expert-usage tracker (forward-hook based)
# ---------------------------------------------------------------------------
def _make_tracker_hook(block: nn.Module):
    """Forward pre-hook that accumulates per-expert routing MASS.

    Scatter-adds the actual top-k softmax weights (after Qwen3's
    `norm_topk_prob=True` renormalization), not `+1` per hit. Two experts
    with the same hit count but different routing confidence end up with
    different importance — high-confidence routes dominate, scraped
    low-confidence top-k slots get less weight.

    When `track_activation` is on for the block, the same firings also accumulate the
    `Σ g·‖f_e‖` importance signal (immediate-commit path only; stash mode stays
    mass-only, which is all the speculative accept-aware path needs).
    """

    def hook(module: nn.Module, inputs):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        flat = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            if hidden_states.dim() == 3
            else hidden_states
        )
        token_mask = getattr(module, _TRACKER_TOKEN_MASK, None)
        if token_mask is not None:
            token_mask = token_mask.reshape(-1).to(device=flat.device, dtype=torch.bool)
            if token_mask.numel() != flat.shape[0]:
                raise ValueError(
                    "expert tracker token mask has "
                    f"{token_mask.numel()} positions, but the MoE input has "
                    f"{flat.shape[0]} tokens"
                )
            if not bool(token_mask.any()):
                return
            flat = flat[token_mask]

        router_logits = _compute_router_logits(module, flat)
        weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_vals, topk_idx = torch.topk(weights, module.top_k, dim=-1)
        
        # ==========================================
        # Match Qwen3's `norm_topk_prob=True`: renormalize so each token's kept routing weights sum to 1.
        # ==========================================
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if getattr(module, _TRACKER_STASH_MODE, False):
            # Accept-aware mode: defer this forward's per-token mass. A later
            # `commit_expert_usage_accepted(...)` commits only the accepted tree
            # positions; padding / rejected positions are simply never committed.
            setattr(module, _TRACKER_PENDING, (topk_idx.detach(), topk_vals.detach()))
        else:
            buf = getattr(module, _TRACKER_BUFFER)
            buf.scatter_add_(
                0, topk_idx.flatten(), topk_vals.flatten().to(buf.dtype)
            )
            if getattr(module, _TRACKER_TRACK_ACT, False):
                imp = getattr(module, _TRACKER_IMPORTANCE, None)
                if imp is not None:
                    imp.add_(_expert_contribution_norm(
                        module, flat, topk_idx, topk_vals,
                        power=int(getattr(module, _TRACKER_ACT_POW, 1)),
                    ))
                cnt = getattr(module, _TRACKER_COUNT, None)
                if cnt is not None:                       # N_e += (#tokens where e is in top-k)
                    idx = topk_idx.flatten()
                    cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=cnt.dtype))

    return hook


def _module_device(module: nn.Module) -> torch.device:
    """Representative device for a block, robust to parameter-less modules.

    The INT4 stacked block (`Qwen3MoeStackedInt4Block`) holds its whole
    expert store as *buffers* (no `nn.Parameter`), so `next(module.parameters())`
    raises `StopIteration`. Fall back to buffers, then CPU.
    """
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.device("cpu")


def _maybe_alloc_importance(module: nn.Module, track_activation: bool,
                            activation_power: int = 1) -> None:
    """Flag a block for activation tracking and lazily allocate its importance buffer.

    Separate from the mass buffer so the idempotent re-install path can turn activation
    tracking on for an already-installed tracker without re-registering the hook.
    `activation_power` is the exponent `p` in `Σ g·‖f_e‖^p` (see `_TRACKER_ACT_POW`).
    """
    if not track_activation:
        return
    setattr(module, _TRACKER_TRACK_ACT, True)
    setattr(module, _TRACKER_ACT_POW, int(activation_power))
    if getattr(module, _TRACKER_IMPORTANCE, None) is None:
        imp_buf = torch.zeros(int(module.num_experts), dtype=torch.float32,
                              device=_module_device(module))
        if hasattr(module, _TRACKER_IMPORTANCE):
            setattr(module, _TRACKER_IMPORTANCE, imp_buf)
        else:
            module.register_buffer(_TRACKER_IMPORTANCE, imp_buf, persistent=False)
    if getattr(module, _TRACKER_COUNT, None) is None:
        cnt_buf = torch.zeros(int(module.num_experts), dtype=torch.float32,
                              device=_module_device(module))
        if hasattr(module, _TRACKER_COUNT):
            setattr(module, _TRACKER_COUNT, cnt_buf)
        else:
            module.register_buffer(_TRACKER_COUNT, cnt_buf, persistent=False)


def install_expert_usage_tracker(
    model: nn.Module, track_activation: bool = False, activation_power: int = 1,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register a mass-accumulating pre-hook on every Qwen3MoE sparse block.

    Idempotent — re-calling on a model that already has the tracker reuses
    the existing buffer/hook rather than registering a second one. The
    matching reset function is `reset_expert_usage`.

    `track_activation=True` also accumulates the per-expert `Σ g·‖f_e‖^p` importance
    signal (read via `get_expert_importance`), where `p = activation_power` (1 = plain
    L2 norm, 2 = squared/energy); it costs an extra eager expert pass per forward, so use
    it for offline calibration, not inference. Default off — the mass path and every
    existing caller are unaffected.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        _maybe_alloc_importance(module, track_activation, activation_power)
        if hasattr(module, _TRACKER_BUFFER) and getattr(module, _TRACKER_HANDLE, None) is not None:
            continue  # already installed

        mass_buf = torch.zeros(
            int(module.num_experts),
            dtype=torch.float32,
            device=_module_device(module),
        )
        if hasattr(module, _TRACKER_BUFFER):
            setattr(module, _TRACKER_BUFFER, mass_buf)
        else:
            module.register_buffer(_TRACKER_BUFFER, mass_buf, persistent=False)
        handle = module.register_forward_pre_hook(_make_tracker_hook(module))
        setattr(module, _TRACKER_HANDLE, handle)
        handles.append(handle)
    return handles


def remove_expert_usage_tracker(model: nn.Module) -> None:
    for _, module in model.named_modules():
        handle = getattr(module, _TRACKER_HANDLE, None)
        if handle is not None:
            handle.remove()
            delattr(module, _TRACKER_HANDLE)


def get_expert_usage(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read per-block expert-usage mass (keyed by module path)."""
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _TRACKER_BUFFER, None)
        if buf is None:
            continue
        out[name] = buf.detach().clone()
    return out


def get_expert_importance(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read per-block `Σ g·‖f_e‖` importance (keyed by module path).

    Only populated for blocks installed with `track_activation=True`; blocks without
    the buffer are omitted, so an empty dict means activation tracking was never on.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _TRACKER_IMPORTANCE, None)
        if buf is None:
            continue
        out[name] = buf.detach().clone()
    return out


def get_expert_count(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Read per-block activation count `N_e` (keyed by module path).

    Only populated for blocks installed with `track_activation=True`. Pair with
    `get_expert_importance` to form the per-firing average `importance / N_e`.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _TRACKER_COUNT, None)
        if buf is None:
            continue
        out[name] = buf.detach().clone()
    return out


def reset_expert_usage(model: nn.Module) -> None:
    """Zero the usage-mass accumulator on every Qwen3MoE block.

    Also clears accept-aware tracking state (stash mode + any pending mass) so a
    new prompt starts in commit-immediately mode — prefill tokens are all real
    and should accumulate directly.
    """
    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        buf = getattr(module, _TRACKER_BUFFER, None)
        if buf is not None:
            buf.zero_()
        imp = getattr(module, _TRACKER_IMPORTANCE, None)
        if imp is not None:
            imp.zero_()
        cnt = getattr(module, _TRACKER_COUNT, None)
        if cnt is not None:
            cnt.zero_()
        if getattr(module, _TRACKER_STASH_MODE, False):
            setattr(module, _TRACKER_STASH_MODE, False)
        if getattr(module, _TRACKER_PENDING, None) is not None:
            setattr(module, _TRACKER_PENDING, None)
        if getattr(module, _TRACKER_TOKEN_MASK, None) is not None:
            setattr(module, _TRACKER_TOKEN_MASK, None)


def set_tracker_token_mask(
    model: nn.Module, token_mask: Optional[torch.Tensor],
) -> None:
    """Restrict usage tracking to selected positions in the next full forward.

    The model still processes the complete sequence, preserving prompt context.
    Only the tracker's router mass, activation count, and importance accumulators
    are filtered. `token_mask` must flatten to the same number of tokens as each
    MoE input in that forward; pass `None` to resume tracking every position.
    """
    for _, module in model.named_modules():
        if _is_qwen3_moe_block(module):
            setattr(module, _TRACKER_TOKEN_MASK, token_mask)


def set_tracker_stash_mode(model: nn.Module, on: bool) -> None:
    """Toggle accept-aware tracking on every Qwen3MoE block.

    On: the hook stashes each forward's per-token mass instead of committing it
    (so a later `commit_expert_usage_accepted` commits only accepted positions).
    Off (default): the hook commits every processed token's mass immediately.
    """
    for _, module in model.named_modules():
        if _is_qwen3_moe_block(module):
            setattr(module, _TRACKER_STASH_MODE, bool(on))


@torch.no_grad()
def commit_expert_usage_accepted(model: nn.Module, positions: Optional[torch.Tensor]) -> None:
    """Commit the stashed per-token mass into the accumulator for `positions` only.

    `positions` are the accepted tree-token indices (e.g. `_verify`'s
    `hidden_indices`) into the most recent stashed forward. `None` commits all
    stashed tokens. Pending mass is cleared after the commit, so each round
    commits exactly once.
    """
    pos = None
    if positions is not None:
        pos = positions.reshape(-1).to(torch.long)

    for _, module in model.named_modules():
        if not _is_qwen3_moe_block(module):
            continue
        pending = getattr(module, _TRACKER_PENDING, None)
        if pending is None:
            continue
        topk_idx, topk_vals = pending
        setattr(module, _TRACKER_PENDING, None)

        if pos is not None:
            if pos.numel() == 0:
                continue  # nothing accepted this round
            sel = pos.to(topk_idx.device)
            topk_idx = topk_idx.index_select(0, sel)
            topk_vals = topk_vals.index_select(0, sel)

        buf = getattr(module, _TRACKER_BUFFER, None)
        if buf is None:
            continue
        buf.scatter_add_(0, topk_idx.flatten(), topk_vals.flatten().to(buf.dtype))


def pick_top_n_per_layer(
    expert_mass: Dict[str, torch.Tensor],
    top_n: int,
) -> Dict[str, torch.Tensor]:
    """For each layer, return the indices of the `top_n` highest-mass experts.

    Returns a dict mapping the same keys to a 1D long tensor of `top_n`
    expert ids, sorted ascending for deterministic equality checks.
    """
    kept: Dict[str, torch.Tensor] = {}
    for name, mass in expert_mass.items():
        if mass.numel() == 0:
            continue
        n = min(int(top_n), int(mass.numel()))
        _, ids = torch.topk(mass, k=n, largest=True, sorted=True)
        ids, _ = torch.sort(ids)
        kept[name] = ids.to(torch.long)
    return kept


# ---------------------------------------------------------------------------
# Module utilities
# ---------------------------------------------------------------------------
def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
