"""One description of an MoE block, whatever architecture produced it.

Qwen3-MoE is the native shape: a `gate` `nn.Linear` returning router logits,
softmax over all experts, top-k, renormalised over the winners, and
`experts[e].{gate,up,down}_proj.weight` with `[I, H]`/`[H, I]` shapes.

Gemma 4 joins through an adapter (`gemma4_adapter.Gemma4MoEAdapterBlock`) that
presents its inline MoE wiring in that same shape, with softmax routing. It
differs in ONE way that reaches the rest of the stack: its experts use GeLU-tanh,
not SiLU. `MoEBlockSpec.activation` carries that; it defaults to SiLU, so every
Qwen3 path is unchanged.

DeepSeek-V2 joins through `deepseek_v2_adapter.DeepseekV2MoEAdapterBlock` (the
routed half of its MoE; shared experts stay outside). It differs in routing:
`norm_topk_prob=False` -- the top-k weights are the raw softmax probabilities --
and fp32 router logits (`MoEBlockSpec.router_fp32`). For such routers, REMOVING
experts must not re-normalise the softmax over the survivors: see
`retained_topk_routing`.

Everything else -- the expert layout and top-k routing over a flat expert pool --
is identical, which is why the stacked store, the Triton kernels and the SVD
planner need no fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Block classes we recognise. The stacked/SVD swap variants are matched too so a
# tracker installs on an already-restructured model.
_HF_BLOCKS = ("Qwen3MoeSparseMoeBlock", "Gemma4MoEAdapterBlock", "DeepseekV2MoEAdapterBlock")
_SWAP_BLOCKS = (
    "Qwen3MoeStackedBlock",
    "Qwen3MoeStackedInt4Block",
    "Qwen3MoeStackedSubsetBlock",
)

SILU, GELU_TANH = "silu", "gelu_tanh"


def activation_fn(x: torch.Tensor, activation: str = SILU) -> torch.Tensor:
    """The expert's gating nonlinearity: `act(gate) * up` uses this on `gate`."""
    if activation == SILU:
        return F.silu(x)
    if activation == GELU_TANH:
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"unknown expert activation {activation!r}")


def block_activation(module: nn.Module) -> str:
    """The expert activation a block computes with. HF Qwen3 blocks carry no
    attribute and are SiLU; the Gemma 4 adapter and our stacked/hybrid blocks say
    so explicitly."""
    return str(getattr(module, "activation", SILU))


def activation_derivative(x: torch.Tensor, activation: str = SILU) -> torch.Tensor:
    """d act(x) / dx, elementwise -- the gate-side Jacobian of `act(gate) * up`."""
    if activation == SILU:
        s = torch.sigmoid(x)
        return s * (1.0 + x * (1.0 - s))
    if activation == GELU_TANH:
        k = 0.7978845608028654                     # sqrt(2 / pi)
        inner = k * (x + 0.044715 * x.pow(3))
        t = torch.tanh(inner)
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t.square()) * k * (1.0 + 3 * 0.044715 * x.square())
    raise ValueError(f"unknown expert activation {activation!r}")


@dataclass(frozen=True)
class MoEBlockSpec:
    """Architecture-independent summary of one MoE block."""

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    norm_topk_prob: bool
    activation: str = SILU
    router_fp32: bool = False       # router logits computed in fp32 (DeepSeek-V2)


def _router_module(module: nn.Module) -> Optional[nn.Module]:
    return getattr(module, "gate", None) or getattr(module, "router", None)


def is_moe_block(module: nn.Module) -> bool:
    """True for an unmodified HF MoE block or one of our swap variants."""
    name = module.__class__.__name__
    if name in _HF_BLOCKS:
        return hasattr(module, "experts") and _router_module(module) is not None
    if name in _SWAP_BLOCKS:
        return all(hasattr(module, a) for a in ("router_weights", "num_experts", "top_k"))
    return False


def describe_moe_block(module: nn.Module) -> MoEBlockSpec:
    """Read a block's routing + shape configuration. Raises if unrecognised."""
    if not is_moe_block(module):
        raise TypeError(f"{module.__class__.__name__} is not a recognised MoE block.")

    if module.__class__.__name__ in _SWAP_BLOCKS:
        return MoEBlockSpec(
            num_experts=int(module.num_experts),
            top_k=int(module.top_k),
            hidden_size=int(module.hidden_size),
            intermediate_size=int(module.intermediate_size),
            norm_topk_prob=bool(getattr(module, "norm_topk_prob", True)),
            activation=str(getattr(module, "activation", SILU)),
            router_fp32=bool(getattr(module, "router_fp32", False)),
        )

    sample = module.experts[0].gate_proj.weight
    intermediate_size, hidden_size = int(sample.shape[0]), int(sample.shape[1])

    if module.__class__.__name__ == "Gemma4MoEAdapterBlock":
        return MoEBlockSpec(
            num_experts=int(module.num_experts),
            top_k=int(module.top_k),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            norm_topk_prob=True,
            activation=str(module.activation),
        )

    if module.__class__.__name__ == "DeepseekV2MoEAdapterBlock":
        return MoEBlockSpec(
            num_experts=int(module.num_experts),
            top_k=int(module.top_k),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            norm_topk_prob=False,
            router_fp32=True,
        )

    # `getattr` defaults rather than attribute access: lightweight stand-ins for
    # this block (the self-generation calibration tests use one) define only the
    # pieces they exercise, and the pre-`moe_arch` call sites defaulted the same way.
    return MoEBlockSpec(                                    # Qwen3MoeSparseMoeBlock
        num_experts=int(getattr(module, "num_experts", len(module.experts))),
        top_k=int(module.top_k),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        norm_topk_prob=bool(getattr(module, "norm_topk_prob", True)),
    )


def router_weight(module: nn.Module) -> torch.Tensor:
    """The `[E, H]` router projection, however the block stores it."""
    if hasattr(module, "router_weights"):
        return module.router_weights
    router = _router_module(module)
    if router is None:
        raise AttributeError("block exposes neither .gate nor .router for the MoE router.")
    return router.weight


def compute_routing(module: nn.Module, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(topk_idx, topk_vals)` for a block, in GLOBAL expert ids.

    One entry point for every offline consumer (usage tracker, activation
    recorder, moment collector): softmax over all experts, top-k, then
    renormalised over the winners when the block does so.
    """
    spec = describe_moe_block(module)
    logits = F.linear(flat.float(), router_weight(module).float())
    scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    vals, idx = torch.topk(scores, k=int(spec.top_k), dim=-1)
    if spec.norm_topk_prob:
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return idx, vals


def router_logits(x: torch.Tensor, weight: torch.Tensor, *, fp32: bool = False) -> torch.Tensor:
    """`[T, E]` float32 router logits; computed in fp32 when the model's router is."""
    if fp32:
        return F.linear(x.float(), weight.float())
    return F.linear(x, weight).float()


def topk_routing(
    logits: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool = True,
    num_experts: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(weights, slots)` from `[T, E]` router logits.

    Softmax is normalised ACROSS experts, so the top-k is taken on the logits and
    the renormalisation happens afterwards over the winners. Returned weights are
    float32 when the logits are; callers cast.

    With ``norm_topk_prob=False`` the weights are softmax probabilities over the
    columns of ``logits``, so ``logits`` must cover the router's WHOLE expert
    pool: a softmax over a retained subset would silently inflate every
    survivor's weight. Pass ``num_experts`` to have that checked; restricted
    routing goes through `retained_topk_routing`.
    """
    if not norm_topk_prob and num_experts is not None and int(logits.shape[-1]) != int(num_experts):
        raise ValueError(
            f"norm_topk_prob=False routing over {int(logits.shape[-1])} of {int(num_experts)} experts "
            "would re-softmax over a subset; use retained_topk_routing"
        )
    values, slots = torch.topk(logits, k=int(top_k), dim=-1)
    if norm_topk_prob:
        weights = F.softmax(values, dim=-1)
    else:
        weights = F.softmax(logits, dim=-1).gather(-1, slots)
    return weights, slots


def retained_topk_routing(
    full_logits: torch.Tensor,
    retained_ids: torch.Tensor,
    top_k: int,
    *,
    norm_topk_prob: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(weights, slots)` when only `retained_ids` (GLOBAL ids) may be selected.

    `slots` index `retained_ids`. `full_logits` are the `[T, E]` logits of the
    ORIGINAL router over all experts, removed ones included.

    * ``norm_topk_prob=True`` (Qwen3, Gemma 4): top-k over the retained logits,
      softmax over the k winners. The full-softmax denominator cancels, so this is
      also what masking removed experts before or after the softmax gives.
    * ``norm_topk_prob=False`` (DeepSeek-V2): the router is left untouched --
      softmax over ALL experts, removed experts masked AFTER the softmax, top-k
      among the retained, weights = their original probabilities. A removed
      expert's probability mass is lost, and the promoted expert enters with its
      own raw probability. Re-softmaxing over the survivors instead (what deleting
      the removed router rows does) inflates every surviving weight by
      1 / (1 - removed mass); at ~60 % removal on DeepSeek-V2-Lite that collapses
      generation (MBPP-dev100 pass@1 1 vs 29 at b25).
    """
    retained_ids = retained_ids.to(full_logits.device).long()
    kept_logits = full_logits.index_select(-1, retained_ids)
    values, slots = torch.topk(kept_logits, k=int(top_k), dim=-1)
    if norm_topk_prob:
        return F.softmax(values, dim=-1), slots
    probs = F.softmax(full_logits, dim=-1).index_select(-1, retained_ids)
    return probs.gather(-1, slots), slots
