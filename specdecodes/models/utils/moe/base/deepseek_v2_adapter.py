"""Give each DeepSeek-V2 MoE layer a Qwen3-shaped routed block.

A DeepSeek-V2(-Lite) MoE layer (`DeepseekV2MoE` on transformers 4.5x,
`DeepseekV2Moe` on transformers 5) is a ROUTED part -- softmax router over
`n_routed_experts`, greedy top-k, **no top-k renormalisation**
(`norm_topk_prob=False`) -- plus always-on `shared_experts`. Layer 0
(`first_k_dense_replace=1`) is a dense MLP and is not touched.

`install_deepseek_v2_moe_adapter` moves the routed part into
`layer.mlp.moe_block` (a `DeepseekV2MoEAdapterBlock`: `gate` linear router,
`experts[e].{gate,up,down}_proj.weight`, routing attributes) and swaps the MoE
class for one whose forward is ``moe_block(x) + shared_experts(x)`` -- the same
op order as the HF block, so the adapted model is bit-identical. The
compression stack then only ever sees (and replaces) `moe_block`; the shared
experts stay FULL.

The adapter's own forward calls the HF router and experts unchanged, so the
routed computation is HF's on either transformers major version. No weight is
copied: on transformers 5 the per-expert projections are views into the fused
`gate_up_proj` / `down_proj` tensors.

`fix_deepseek_v2_yarn_softmax_scale` is a separate, model-level correctness fix
(see its docstring).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .gemma4_adapter import _ExpertView

_MOE_CLASSES = ("DeepseekV2MoE", "DeepseekV2Moe")


def is_deepseek_v2_model(model: nn.Module) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) == "deepseek_v2"


# --------------------------------------------------------------------------- YaRN
def _yarn_get_mscale(scale: float, mscale: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _rope_config(config) -> dict:
    # transformers 4: `rope_scaling`; transformers 5: `rope_parameters`.
    for key in ("rope_parameters", "rope_scaling"):
        value = getattr(config, key, None)
        if isinstance(value, dict) and value:
            return value
    return {}


@torch.no_grad()
def fix_deepseek_v2_yarn_softmax_scale(model: nn.Module) -> float:
    """Re-apply DeepSeek-V2's YaRN attention-softmax factor.

    The reference implementation (DeepSeek's `modeling_deepseek.py`, remote code)
    scales attention logits by ``qk_head_dim**-0.5 * mscale**2`` with
    ``mscale = 0.1 * mscale_all_dim * ln(factor) + 1`` when YaRN rope scaling is
    configured (DeepSeek-V2-Lite: factor 40, mscale_all_dim 0.707 -> x1.590).
    transformers' native `DeepseekV2Attention` (checked: 4.57.1 and 5.14.1) uses
    ``qk_head_dim**-0.5`` only -- its `DeepseekV3Attention` does apply the
    factor -- so the native model attends with too flat a softmax. Measured on
    DeepSeek-V2-Lite-Chat: KL(remote || native) 0.10-0.20 nats/token without
    the factor, 0.005-0.024 with it; MBPP-dev100 38 -> 26 without it.

    Only touches `DeepseekV2Attention` modules whose scaling is still the
    unscaled value, so it is idempotent and becomes a no-op if transformers
    starts applying the factor itself. Returns the multiplier applied (1.0 if
    nothing was changed).
    """
    if not is_deepseek_v2_model(model):
        return 1.0
    rope = _rope_config(model.config)
    rope_type = rope.get("rope_type", rope.get("type"))
    mscale_all_dim = rope.get("mscale_all_dim")
    if rope_type != "yarn" or not mscale_all_dim:
        return 1.0
    m = _yarn_get_mscale(float(rope["factor"]), float(mscale_all_dim))
    mult = m * m
    changed = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekV2Attention":
            continue
        base = float(module.qk_head_dim) ** -0.5
        if math.isclose(float(module.scaling), base, rel_tol=1e-9):
            module.scaling = base * mult
            changed += 1
    return mult if changed else 1.0


# ------------------------------------------------------------------ routed block
class DeepseekV2MoEAdapterBlock(nn.Module):
    """Qwen3-shaped view of one DeepSeek-V2 layer's ROUTED experts.

    Input is the MoE input; output is the routed sum (shared experts excluded).
    """

    norm_topk_prob = False          # DeepSeek-V2 never renormalises the top-k weights
    router_fp32 = True              # its gate computes logits in fp32

    def __init__(self, moe: nn.Module):
        super().__init__()
        router, experts = moe.gate, moe.experts
        self.hf_gate = router
        self.hf_experts = experts
        self.num_experts = int(getattr(router, "num_experts", None) or getattr(router, "n_routed_experts"))
        self.top_k = int(router.top_k)
        if getattr(router, "topk_method", "greedy") != "greedy":
            raise NotImplementedError("DeepSeek-V2 adapter supports topk_method='greedy' only")
        if float(getattr(router, "routed_scaling_factor", 1.0)) != 1.0:
            raise NotImplementedError("DeepSeek-V2 adapter supports routed_scaling_factor=1.0 only")
        if bool(getattr(router, "norm_topk_prob", False)) or bool(getattr(moe.config, "norm_topk_prob", False)):
            raise NotImplementedError("DeepSeek-V2 adapter assumes norm_topk_prob=False")
        hidden = int(router.weight.shape[1])
        self.gate = nn.Linear(hidden, self.num_experts, bias=False, device="meta")
        self.gate.weight = router.weight                                 # the router's own Parameter

        if isinstance(experts, nn.ModuleList):                           # transformers 4
            self.experts = experts                                       # DeepseekV2MLP per expert
            sample = experts[0].gate_proj.weight
            self.intermediate_size, self.hidden_size = int(sample.shape[0]), int(sample.shape[1])
            self._fused = False
        else:                                                            # transformers 5 (fused)
            num_experts, two_i, hidden = experts.gate_up_proj.shape
            inter = two_i // 2
            gate_up, down = experts.gate_up_proj.data, experts.down_proj.data
            self.experts = nn.ModuleList(
                _ExpertView(gate_up[e, :inter], gate_up[e, inter:], down[e]) for e in range(num_experts)
            )
            self.intermediate_size, self.hidden_size = int(inter), int(hidden)
            self._fused = True
        self.ep_rank, self.experts_per_rank = 0, self.num_experts        # read by HF `moe()` (v4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.view(-1, shape[-1])
        if self._fused:
            _, weights, idx = self.hf_gate(hidden_states)
            return self.hf_experts(flat, idx, weights).view(*shape)
        idx, weights = self.hf_gate(hidden_states)
        moe = type(self)._v4_moe
        return moe(self, flat, idx, weights).view(*shape)

    _v4_moe = None                  # set at install time to the HF class's `moe` function


def _patched_moe_class(base: type) -> type:
    """`base` whose forward is ``moe_block(x) + shared_experts(x)`` (HF op order)."""

    def forward(self, hidden_states):
        return self.moe_block(hidden_states) + self.shared_experts(hidden_states)

    return type(f"{base.__name__}WithMoEBlock", (base,), {"forward": forward})


_PATCHED: dict[type, type] = {}


@torch.no_grad()
def install_deepseek_v2_moe_adapter(model: nn.Module) -> int:
    """Move every DeepSeek-V2 layer's routed experts into `mlp.moe_block`.

    Also applies `fix_deepseek_v2_yarn_softmax_scale` (idempotent). Idempotent
    itself: already-adapted layers are skipped. Returns the number adapted.
    """
    fix_deepseek_v2_yarn_softmax_scale(model)
    count = 0
    for module in list(model.modules()):
        cls = type(module)
        if cls.__name__ not in _MOE_CLASSES or hasattr(module, "moe_block"):
            continue
        if hasattr(cls, "moe") and DeepseekV2MoEAdapterBlock._v4_moe is None:
            DeepseekV2MoEAdapterBlock._v4_moe = cls.moe
        module.moe_block = DeepseekV2MoEAdapterBlock(module)
        del module.gate
        del module.experts
        module.__class__ = _PATCHED.setdefault(cls, _patched_moe_class(cls))
        count += 1
    return count
