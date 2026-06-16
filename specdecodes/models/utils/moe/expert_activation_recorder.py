"""
Per-layer expert-activation recorder + plotting — for studying the *locality* of
MoE expert usage over a single generation (e.g. vanilla GSM8K decoding).

Hooks ONE MoE layer (default layer 0) and, every `window` generated tokens,
snapshots the cumulative per-expert routing **mass** (the same top-k softmax mass
the usage tracker uses — full router, original / un-redirected). Diffing
consecutive snapshots gives the per-window expert distribution, which reveals
whether the hot experts stay put (high locality) or drift across the sequence.

Prefill vs decode is split by the forward's token count: a forward with >1 token
is a prefill chunk (folded into `prompt_mass`); a 1-token forward is one
generated/decode token (counts toward the `window` snapshots). That matches the
`NaiveGenerator` loop (1 token per decode forward, chunked prefill).

NOTE: register a forward hook only fires reliably in **eager** mode — run the
recording pass with `compile_mode: null` (torch.compile may inline the module
forward and skip Python hooks).

Usage:
    rec = ExpertActivationRecorder(model, layer_index=0, window=16)
    ... run one generation ...
    rec.finalize()
    rec.save_npz("layer0_activation.npz")
    plot_expert_activation(rec, "layer0_activation.png")   # needs matplotlib
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .base.expert_usage_tracker import (
    _is_qwen3_moe_block,
    _compute_router_logits,
    _module_device,
)


class ExpertActivationRecorder:
    """Snapshot one MoE layer's cumulative expert mass every `window` decode tokens."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_index: int = 0,
        window: int = 16,
        top_k: Optional[int] = None,
    ):
        self.layer_index = int(layer_index)
        self.window = int(window)
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {window}")

        self.block, self.block_name = self._find_moe_block(model, self.layer_index)
        self.num_experts = int(self.block.num_experts)
        self.top_k = int(top_k if top_k is not None else self.block.top_k)
        dev = _module_device(self.block)

        # Decode-only cumulative mass (prefill kept separate so per-window deltas
        # are clean) + the prompt's mass for reference.
        self.decode_mass = torch.zeros(self.num_experts, dtype=torch.float32, device=dev)
        self.prompt_mass = torch.zeros(self.num_experts, dtype=torch.float32, device=dev)
        self._decode_tokens = 0
        self._prompt_tokens = 0
        # (decode_token_count, cumulative decode-only mass) at each window boundary.
        self.snapshots: List[Tuple[int, torch.Tensor]] = []

        self._handle = self.block.register_forward_pre_hook(self._hook)
        logging.info(
            "[ExpertActivationRecorder] hooked %s (E=%d, top_k=%d), window=%d tokens.",
            self.block_name, self.num_experts, self.top_k, self.window,
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _find_moe_block(model: torch.nn.Module, layer_index: int):
        moe = [(n, m) for n, m in model.named_modules() if _is_qwen3_moe_block(m)]
        if not moe:
            raise ValueError("No Qwen3-MoE blocks found on the model.")
        for n, m in moe:
            if f".layers.{layer_index}." in n or n.endswith(f"layers.{layer_index}.mlp"):
                return m, n
        if layer_index < len(moe):
            return moe[layer_index][1], moe[layer_index][0]
        raise ValueError(
            f"layer_index {layer_index} out of range ({len(moe)} MoE blocks found)."
        )

    # ------------------------------------------------------------------ hook
    @torch.no_grad()
    def _hook(self, module: torch.nn.Module, inputs):
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        flat = (
            hidden_states.reshape(-1, hidden_states.shape[-1])
            if hidden_states.dim() == 3
            else hidden_states
        )
        T = int(flat.shape[0])
        if T == 0:
            return

        logits = _compute_router_logits(module, flat)
        weights = F.softmax(logits, dim=1, dtype=torch.float)
        topk_vals, topk_idx = torch.topk(weights, self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        per_call = torch.zeros(self.num_experts, dtype=torch.float32, device=flat.device)
        per_call.scatter_add_(0, topk_idx.flatten(), topk_vals.flatten().float())

        if T > 1:
            # Prefill chunk — fold into the prompt baseline, don't count as decode tokens.
            self.prompt_mass += per_call
            self._prompt_tokens += T
            return

        # One generated (decode) token.
        self.decode_mass += per_call
        self._decode_tokens += 1
        if self._decode_tokens % self.window == 0:
            self.snapshots.append((self._decode_tokens, self.decode_mass.detach().cpu().clone()))

    # ------------------------------------------------------------------ lifecycle
    def finalize(self) -> "ExpertActivationRecorder":
        """Capture the trailing partial window (if any) and stop recording."""
        if self._decode_tokens > 0 and (
            not self.snapshots or self.snapshots[-1][0] != self._decode_tokens
        ):
            self.snapshots.append((self._decode_tokens, self.decode_mass.detach().cpu().clone()))
        self.remove()
        return self

    def remove(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------ data
    def window_labels(self) -> List[int]:
        return [c for c, _ in self.snapshots]

    def cumulative_matrix(self) -> torch.Tensor:
        """[num_windows, num_experts] decode-only cumulative mass at each boundary."""
        if not self.snapshots:
            return torch.empty(0, self.num_experts)
        return torch.stack([m for _, m in self.snapshots])

    def window_matrix(self) -> torch.Tensor:
        """[num_windows, num_experts] per-window mass (consecutive-snapshot deltas)."""
        cum = self.cumulative_matrix()
        if cum.numel() == 0:
            return cum
        base = torch.cat([torch.zeros(1, self.num_experts), cum[:-1]], dim=0)
        return cum - base

    def locality_summary(self, top_mult: int = 2) -> Dict[str, float]:
        """Scalar locality metrics for this layer (feeds the cross-layer depth profile).

        `experts_for_{50,90,99}` — # experts covering that share of the whole
        decode's mass (lower = more concentrated). `mean_window_cov90` — mean
        per-window 90%-coverage. `mean_consec_jaccard` — mean Jaccard of the
        top-(`top_mult`·top_k) expert set between consecutive windows (higher =
        more temporally stable / cacheable).
        """
        import numpy as np

        cum = self.cumulative_matrix().numpy()
        win = self.window_matrix().numpy()
        out: Dict[str, float] = {
            "layer": self.layer_index, "num_experts": self.num_experts, "top_k": self.top_k,
            "decode_tokens": (self.window_labels()[-1] if self.snapshots else 0),
            "experts_used": 0, "experts_for_50": 0, "experts_for_90": 0, "experts_for_99": 0,
            "mean_window_cov90": 0.0, "mean_consec_jaccard": float("nan"),
        }
        if cum.shape[0] == 0:
            return out

        total = cum[-1]
        tot = float(total.sum())
        if tot > 0:
            cs = np.cumsum(np.sort(total)[::-1]) / tot
            out["experts_for_50"] = int(np.searchsorted(cs, 0.50)) + 1
            out["experts_for_90"] = int(np.searchsorted(cs, 0.90)) + 1
            out["experts_for_99"] = int(np.searchsorted(cs, 0.99)) + 1
        out["experts_used"] = int((total > 1e-9).sum())

        covs = []
        for r in win:
            t = float(r.sum())
            if t <= 0:
                continue
            cs = np.cumsum(np.sort(r)[::-1]) / t
            covs.append(int(np.searchsorted(cs, 0.90)) + 1)
        if covs:
            out["mean_window_cov90"] = float(np.mean(covs))

        K = self.top_k * top_mult
        sets = [set(np.argsort(r)[::-1][:K].tolist()) for r in win]
        jac = [len(a & b) / len(a | b) for a, b in zip(sets[:-1], sets[1:]) if (a | b)]
        if jac:
            out["mean_consec_jaccard"] = float(np.mean(jac))
        return out

    def save_npz(self, path: str) -> str:
        import numpy as np

        np.savez(
            path,
            window_mass=self.window_matrix().numpy(),
            cumulative_mass=self.cumulative_matrix().numpy(),
            prompt_mass=self.prompt_mass.detach().cpu().numpy(),
            decode_token_counts=np.asarray(self.window_labels(), dtype=np.int64),
            layer_index=self.layer_index,
            num_experts=self.num_experts,
            top_k=self.top_k,
            window=self.window,
            prompt_tokens=self._prompt_tokens,
        )
        logging.info("[ExpertActivationRecorder] saved data -> %s", path)
        return path


def plot_expert_activation(
    recorder: ExpertActivationRecorder,
    out_path: str,
    normalize: bool = True,
    include_prompt: bool = True,
    title: Optional[str] = None,
) -> str:
    """Two-panel figure: per-window expert-mass heatmap + concentration line.

    Top panel — heatmap [window x expert]: each row is one `window`-token decode
    window, colored by that window's routing mass (row-normalized to a
    distribution when `normalize`). Stable bright columns = high locality; shifting
    columns = the active expert set drifts.

    Bottom panel — per window, the number of experts needed to cover 90% of that
    window's mass (lower = more concentrated).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # not installed, or a partial/namespace shadow
        raise RuntimeError(
            "matplotlib is required for plot_expert_activation "
            "(`pip install matplotlib`). The raw data is still available via "
            "ExpertActivationRecorder.save_npz(...)."
        ) from e

    import numpy as np

    W = recorder.window_matrix().numpy()  # [num_windows, E]
    if W.shape[0] == 0:
        raise ValueError("No decode windows recorded — generation too short for the window size?")

    labels = [f"+{c}" for c in recorder.window_labels()]
    if include_prompt:
        prompt = recorder.prompt_mass.detach().cpu().numpy()[None, :]
        W = np.concatenate([prompt, W], axis=0)
        labels = ["prompt"] + labels

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    heat = W / row_sums if normalize else W

    # 90%-mass coverage count per row.
    cover = []
    for row in W:
        s = np.sort(row)[::-1]
        tot = s.sum()
        if tot <= 0:
            cover.append(0)
            continue
        cum = np.cumsum(s) / tot
        cover.append(int(np.searchsorted(cum, 0.90) + 1))

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(max(8, recorder.num_experts * 0.09), max(5, len(labels) * 0.32 + 2)),
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True,
    )

    im = ax0.imshow(heat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax0.set_xlabel(f"expert id (layer {recorder.layer_index}, {recorder.num_experts} experts)")
    ax0.set_ylabel(f"decode window ({recorder.window} tokens each)")
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax0, label=("per-window mass share" if normalize else "mass"))
    ax0.set_title(title or f"Layer {recorder.layer_index} expert activation locality")

    ax1.plot(range(len(labels)), cover, marker="o", ms=3)
    ax1.set_xlabel("decode window")
    ax1.set_ylabel("# experts\nfor 90% mass")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7, rotation=90)
    ax1.set_ylim(0, recorder.num_experts)
    ax1.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("[ExpertActivationRecorder] saved plot -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Multi-layer recording (full depth profile in a single generation)
# ---------------------------------------------------------------------------
def available_moe_layers(model: torch.nn.Module) -> List[int]:
    """Sorted layer indices of every Qwen3-MoE block, parsed from module names."""
    idxs = []
    for n, m in model.named_modules():
        if _is_qwen3_moe_block(m):
            mt = re.search(r"layers\.(\d+)\.", n)
            if mt:
                idxs.append(int(mt.group(1)))
    return sorted(set(idxs))


def resolve_layer_indices(model: torch.nn.Module, spec: Union[int, str, Sequence[int], None]) -> List[int]:
    """Resolve a `layer` spec into concrete indices: int, list[int], or 'all'/None."""
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return available_moe_layers(model)
    if isinstance(spec, int):
        return [spec]
    return sorted({int(x) for x in spec})


class ExpertActivationRecorderSet:
    """Record several MoE layers at once (one `ExpertActivationRecorder` each).

    `layers` may be an int, a list of ints, or 'all'. `save_all` writes a per-layer
    `.npz` (+ optional `.png` heatmap) plus a cross-layer depth profile
    (`<name>_profile.npz/.png`) summarizing how concentration/stability vary with depth.
    """

    def __init__(self, model: torch.nn.Module, layers: Union[int, str, Sequence[int]] = "all", window: int = 16):
        idxs = resolve_layer_indices(model, layers)
        if not idxs:
            raise ValueError("No MoE layers resolved for recording.")
        self.window = int(window)
        self.recorders = [ExpertActivationRecorder(model, layer_index=i, window=window) for i in idxs]
        logging.info("[ExpertActivationRecorderSet] recording %d layer(s): %s", len(idxs), idxs)

    def finalize(self) -> "ExpertActivationRecorderSet":
        for r in self.recorders:
            r.finalize()
        return self

    def save_all(
        self,
        out_dir: str,
        name: str,
        per_layer_plot: Optional[bool] = None,
        profile_plot: bool = True,
    ) -> List[Dict[str, float]]:
        import os
        os.makedirs(out_dir, exist_ok=True)
        # Default: per-layer heatmaps only when recording a single layer (avoid 48 PNGs).
        if per_layer_plot is None:
            per_layer_plot = len(self.recorders) == 1

        for r in self.recorders:
            stem = os.path.join(out_dir, f"{name}_layer{r.layer_index}")
            r.save_npz(stem + ".npz")
            if per_layer_plot:
                try:
                    plot_expert_activation(r, stem + ".png")
                except Exception as e:
                    logging.warning("[recorder] per-layer plot skipped (L%d): %s", r.layer_index, e)

        summaries = [r.locality_summary() for r in self.recorders]
        self._save_profile_npz(os.path.join(out_dir, f"{name}_profile.npz"), summaries)
        if profile_plot and len(self.recorders) > 1:
            try:
                plot_layer_locality_profile(summaries, os.path.join(out_dir, f"{name}_profile.png"))
            except Exception as e:
                logging.warning("[recorder] depth-profile plot skipped: %s", e)
        return summaries

    @staticmethod
    def _save_profile_npz(path: str, summaries: List[Dict[str, float]]) -> None:
        import numpy as np
        keys = ["layer", "experts_for_50", "experts_for_90", "experts_for_99",
                "experts_used", "mean_window_cov90", "mean_consec_jaccard", "decode_tokens"]
        np.savez(path, **{k: np.asarray([s[k] for s in summaries], dtype=np.float64) for k in keys})
        logging.info("[recorder] saved depth profile -> %s", path)


def plot_layer_locality_profile(summaries: List[Dict[str, float]], out_path: str) -> str:
    """Two-panel depth profile across layers: concentration + temporal stability."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required (`pip install matplotlib`).") from e

    summaries = sorted(summaries, key=lambda s: s["layer"])
    layers = [s["layer"] for s in summaries]
    E = int(summaries[0]["num_experts"])
    top_k = int(summaries[0]["top_k"])

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(max(7, len(layers) * 0.18), 7), constrained_layout=True
    )

    ax0.plot(layers, [s["experts_for_50"] for s in summaries], marker="o", ms=3, label="50% mass")
    ax0.plot(layers, [s["experts_for_90"] for s in summaries], marker="o", ms=3, label="90% mass")
    ax0.plot(layers, [s["mean_window_cov90"] for s in summaries], marker="x", ms=3, ls="--",
             label="per-window 90% (mean)")
    ax0.axhline(top_k, color="gray", ls=":", lw=1, label=f"top_k={top_k}")
    ax0.set_xlabel("layer")
    ax0.set_ylabel("# experts")
    ax0.set_ylim(0, E)
    ax0.set_title(f"Expert-usage concentration vs depth (E={E}; lower = more concentrated)")
    ax0.legend(fontsize=7)
    ax0.grid(alpha=0.3)

    ax1.plot(layers, [s["mean_consec_jaccard"] for s in summaries], marker="o", ms=3, color="C3")
    ax1.set_xlabel("layer")
    ax1.set_ylabel(f"mean consec-window\ntop-{2 * top_k} Jaccard")
    ax1.set_ylim(0, 1)
    ax1.set_title("Temporal stability of hot set vs depth (higher = more local / cacheable)")
    ax1.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("[recorder] saved depth-profile plot -> %s", out_path)
    return out_path
