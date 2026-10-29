"""Give each Gemma 4 decoder layer a Qwen3-shaped MoE block.

Gemma 4 (`Gemma4TextDecoderLayer`, transformers >= 5.5) has no MoE module: the
router, the fused experts and three extra norms sit directly on the decoder
layer, and its forward wires them together inline::

    residual ──► router(residual)                       # own norm, scale, H^-1/2
             └─► pre_feedforward_layernorm_2(residual) ──► experts ──► post_feedforward_layernorm_2
    dense mlp ─► post_feedforward_layernorm_1 ───────────────────────────────┘ (+)

The rest of this stack (restructurer, usage tracker, activation recorder,
planner) expects `block.gate` (a linear router over the block's own input),
`block.experts[e].{gate,up,down}_proj.weight` and `block(x) -> sum_e g_e f_e(x)`.
`install_gemma4_moe_adapter` builds exactly that as `layer.moe_block` and swaps
the layer's class for one whose forward calls it, so a compressed block can later
replace `moe_block` like any other.

Two folds make Gemma's routing identical to Qwen3's (softmax -> top-k -> renorm):

* **Router input.** Both the router's own norm and `pre_feedforward_layernorm_2`
  divide the same residual by the same RMS (same eps); they differ only by an
  elementwise factor, so ``router_in = x * c`` with ``x`` the expert input and
  ``c = router.scale * H**-0.5 / pre_feedforward_layernorm_2.weight``. Folding
  ``c`` into the router's columns is exact. The norm weight has entries down to
  ~1e-6, so the folded columns are large -- each product keeps its magnitude, so
  bf16 relative precision is unchanged; top-k agreement with the original router
  is checked on real activations, not assumed.
* **`per_expert_scale`** multiplies expert e's routing weight after renormalising;
  `f_e` is linear in `down_proj`, so it is folded into `down_proj[e]` (in place).

What is NOT folded, deliberately: `post_feedforward_layernorm_2` on the branch
output. It sits downstream of the block, so the block's computation is exact; it
only changes how damage to the block's output reaches the residual stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_arch import GELU_TANH, activation_fn, topk_routing


def _linear_with(weight: torch.Tensor) -> nn.Linear:
    out_f, in_f = weight.shape
    linear = nn.Linear(in_f, out_f, bias=False, device="meta")
    linear.weight = nn.Parameter(weight, requires_grad=False)   # shares storage, no copy
    return linear


class _ExpertView(nn.Module):
    """`gate_proj` / `up_proj` / `down_proj` as views into Gemma's fused tensors."""

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.gate_proj = _linear_with(gate)
        self.up_proj = _linear_with(up)
        self.down_proj = _linear_with(down)


class Gemma4MoEAdapterBlock(nn.Module):
    """Qwen3-shaped view of one Gemma 4 layer's routed experts.

    Input is the expert input ``pre_feedforward_layernorm_2(residual)``; output
    is the routed sum BEFORE ``post_feedforward_layernorm_2``.
    """

    activation = GELU_TANH

    def __init__(self, layer: nn.Module):
        super().__init__()
        router, experts = layer.router, layer.experts
        pre_norm = layer.pre_feedforward_layernorm_2
        num_experts, two_i, hidden = experts.gate_up_proj.shape
        inter = two_i // 2
        if experts.down_proj.shape != (num_experts, hidden, inter):
            raise ValueError(f"unexpected down_proj shape {tuple(experts.down_proj.shape)}")
        act = getattr(layer.config, "hidden_activation", None)
        if act not in ("gelu_pytorch_tanh", "gelu_tanh"):
            raise NotImplementedError(f"Gemma 4 expert activation {act!r} is not supported")

        self.num_experts = int(num_experts)
        self.top_k = int(router.config.top_k_experts)
        self.norm_topk_prob = True
        self.hidden_size = int(hidden)
        self.intermediate_size = int(inter)

        dtype = experts.gate_up_proj.dtype
        with torch.no_grad():
            norm_w = pre_norm.weight.detach().float()
            if bool((norm_w.abs() < 1e-30).any()):
                raise ValueError("pre_feedforward_layernorm_2 has zero weights; the router fold is undefined")
            col = router.scale.detach().float() * router.scalar_root_size / norm_w
            folded = router.proj.weight.detach().float() * col.unsqueeze(0)
            self.gate = nn.Linear(hidden, num_experts, bias=False, device=folded.device, dtype=dtype)
            self.gate.weight.copy_(folded.to(dtype))

            experts.down_proj.data.mul_(
                router.per_expert_scale.detach().to(dtype).view(-1, 1, 1)
            )
            gate_up, down = experts.gate_up_proj.data, experts.down_proj.data
            self.experts = nn.ModuleList(
                _ExpertView(gate_up[e, :inter], gate_up[e, inter:], down[e])
                for e in range(num_experts)
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        logits = F.linear(flat, self.gate.weight)
        weights, slots = topk_routing(
            logits.float(), self.top_k, norm_topk_prob=True
        )
        out = torch.zeros_like(flat)
        for e in torch.unique(slots).tolist():
            token, pos = torch.where(slots == e)
            expert = self.experts[e]
            x = flat[token]
            middle = activation_fn(expert.gate_proj(x), self.activation) * expert.up_proj(x)
            y = expert.down_proj(middle) * weights[token, pos].unsqueeze(1).to(flat.dtype)
            out.index_add_(0, token, y.to(out.dtype))
        return out.reshape(shape)


def _patched_layer_class(base: type) -> type:
    """`base` with the inline MoE wiring replaced by a call to `self.moe_block`."""

    def forward(
        self,
        hidden_states,
        per_layer_input=None,
        shared_kv_states=None,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_moe_block:
            dense = self.post_feedforward_layernorm_1(hidden_states)
            routed = self.moe_block(self.pre_feedforward_layernorm_2(residual))
            hidden_states = dense + self.post_feedforward_layernorm_2(routed)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states

    return type(f"{base.__name__}WithMoEBlock", (base,), {"forward": forward})


_PATCHED: dict[type, type] = {}


def is_gemma4_model(model: nn.Module) -> bool:
    cfg = getattr(model, "config", None)
    types = {getattr(cfg, "model_type", None), getattr(getattr(cfg, "text_config", None), "model_type", None)}
    return bool(types & {"gemma4", "gemma4_text"})


def prepare_moe_blocks(model: nn.Module) -> int:
    """Install whatever adapter `model` needs so every MoE layer is a module the
    stacked/hybrid builders recognise: Gemma 4 gets an MoE block per layer, a
    transformers >= 5 Qwen3-MoE gets its fused experts split into per-expert views,
    DeepSeek-V2 gets its routed experts moved into `mlp.moe_block`.
    No-op (returns 0) for Qwen3 on transformers 4."""
    if is_gemma4_model(model):
        return install_gemma4_moe_adapter(model)
    from .deepseek_v2_adapter import install_deepseek_v2_moe_adapter, is_deepseek_v2_model
    if is_deepseek_v2_model(model):
        return install_deepseek_v2_moe_adapter(model)
    from .qwen3_fused_adapter import install_qwen3_fused_adapter   # imports _ExpertView from here
    return install_qwen3_fused_adapter(model)


@torch.no_grad()
def install_gemma4_moe_adapter(model: nn.Module) -> int:
    """Swap every Gemma 4 MoE decoder layer onto `layer.moe_block`. Returns the count.

    Idempotent: layers already patched are skipped. The layer's `router` and
    `experts` attributes are removed afterwards; the expert storage lives on as
    views inside the adapter.
    """
    count = 0
    for module in list(model.modules()):
        cls = type(module)
        if cls.__name__ != "Gemma4TextDecoderLayer" or not getattr(module, "enable_moe_block", False):
            continue
        module.moe_block = Gemma4MoEAdapterBlock(module)
        del module.router
        del module.experts
        module.__class__ = _PATCHED.setdefault(cls, _patched_layer_class(cls))
        count += 1
    return count
