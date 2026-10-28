"""Give a transformers >= 5 Qwen3-MoE block its per-expert shape back.

transformers 5 fuses a Qwen3-MoE layer's experts into one `Qwen3MoeExperts`
module (`gate_up_proj [E, 2I, H]`, gate rows first, and `down_proj [E, H, I]`)
and moves `top_k` / `num_experts` / `norm_topk_prob` onto a `Qwen3MoeTopKRouter`
that returns `(logits, weights, indices)`. The rest of this stack (stacked store,
restructurer, usage tracker, calibration, planner) reads the transformers 4
surface: `block.gate` a linear router, `block.experts[e].{gate,up,down}_proj.weight`
and the routing attributes on the block.

`install_qwen3_fused_adapter` restores that surface in place, without copying a
weight: each expert's projections are views into the fused tensors, and the router
`nn.Linear` holds the router's own Parameter. The block keeps its class name, so
`moe_arch` recognises it as before; its forward is the transformers 4 computation
(softmax over experts -> top-k -> renormalise -> weighted sum of SiLU experts).
On transformers 4 the experts are already a `ModuleList` and this is a no-op.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gemma4_adapter import _ExpertView
from .moe_arch import SILU, activation_fn, topk_routing


def _is_fused_qwen3_block(module: nn.Module) -> bool:
    return (
        type(module).__name__ == "Qwen3MoeSparseMoeBlock"
        and type(getattr(module, "experts", None)).__name__ == "Qwen3MoeExperts"
    )


def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    weights, slots = topk_routing(
        F.linear(flat, self.gate.weight).float(), self.top_k,
        norm_topk_prob=self.norm_topk_prob,
    )
    out = torch.zeros_like(flat)
    for e in torch.unique(slots).tolist():
        token, pos = torch.where(slots == e)
        expert = self.experts[e]
        x = flat[token]
        middle = activation_fn(expert.gate_proj(x), SILU) * expert.up_proj(x)
        y = expert.down_proj(middle) * weights[token, pos].unsqueeze(1).to(flat.dtype)
        out.index_add_(0, token, y.to(out.dtype))
    return out.reshape(shape)


_PATCHED: dict[type, type] = {}


def _patched_block_class(base: type) -> type:
    """`base` under its own name, with the transformers 4 per-expert forward."""
    return type(base.__name__, (base,), {"forward": _forward, "__module__": base.__module__})


@torch.no_grad()
def install_qwen3_fused_adapter(model: nn.Module) -> int:
    """Adapt every fused Qwen3-MoE block in `model`. Returns the count (0 on transformers 4).

    Idempotent: an adapted block's experts are a `ModuleList` and are skipped.
    """
    count = 0
    for module in list(model.modules()):
        if not _is_fused_qwen3_block(module):
            continue
        experts, router = module.experts, module.gate
        num_experts, two_i, hidden = experts.gate_up_proj.shape
        inter = two_i // 2
        if experts.down_proj.shape != (num_experts, hidden, inter):
            raise ValueError(f"unexpected down_proj shape {tuple(experts.down_proj.shape)}")
        act = getattr(getattr(experts, "act_fn", None), "__class__", type(None)).__name__
        if act not in ("SiLU", "SiLUActivation"):
            raise NotImplementedError(f"Qwen3-MoE expert activation {act!r} is not supported")

        gate_up, down = experts.gate_up_proj.data, experts.down_proj.data
        module.experts = nn.ModuleList(
            _ExpertView(gate_up[e, :inter], gate_up[e, inter:], down[e])
            for e in range(num_experts)
        )
        linear = nn.Linear(hidden, num_experts, bias=False, device="meta")
        linear.weight = router.weight                      # the router's own Parameter
        module.gate = linear
        module.num_experts = int(router.num_experts)
        module.top_k = int(router.top_k)
        module.norm_topk_prob = bool(router.norm_topk_prob)
        module.__class__ = _PATCHED.setdefault(type(module), _patched_block_class(type(module)))
        count += 1
    return count
