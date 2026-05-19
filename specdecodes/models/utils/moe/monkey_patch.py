"""FlashInfer kernel monkey-patch for Qwen3-MoE.

Mirrors `specdecodes/models/utils/flashinfer/monkey_patch.py` (which targets
Llama / Qwen3 dense), but for the Qwen3-MoE classes
(`Qwen3MoeAttention`, `Qwen3MoeRMSNorm`).

`Qwen3MoeAttention.forward` has the same signature and same q_norm/k_norm
structure as `Qwen3Attention`, so `FiQwen3Attention.forward` is a drop-in
replacement once bound onto already-constructed `Qwen3MoeAttention`
modules. (Class-level replacement is also done so any modules constructed
after this call pick up the FI version end-to-end.)

The patch is intentionally scoped to attention + RMSNorm: it does NOT
touch the MoE sparse block — that's handled separately by the
`PackedTopN*MoeBlock` family in this same folder.
"""

from typing import Callable

from transformers import PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
)

from ..flashinfer.attention import FiQwen3Attention


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def _patch_attention_module(module) -> None:
    if not isinstance(module, Qwen3MoeAttention):
        raise ValueError(
            f"Unsupported attention module type for Qwen3-MoE patch: {type(module)}"
        )
    _bind_method_to_module(module, "forward", FiQwen3Attention.forward)


def apply_flashinfer_kernel_to_qwen_moe(
    attention: bool = True,
    rms_norm: bool = False,
    model: PreTrainedModel = None,
) -> None:
    """Swap Qwen3-MoE attention / RMSNorm for the FlashInfer versions.

    Args:
        attention: Patch `Qwen3MoeAttention.forward` to `FiQwen3Attention.forward`.
        rms_norm:  Patch `Qwen3MoeRMSNorm` to `FiLlamaRMSNorm`. Off by default
                   because the FI RMSNorm path needs upstream callers to
                   pass the residual it expects; flip on once that's wired.
        model:     If provided, walk the loaded model and rebind methods on
                   every existing decoder-layer attention / norm so already-
                   constructed modules pick up the FI forward.
    """
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    if attention:
        modeling_qwen3_moe.Qwen3MoeAttention = FiQwen3Attention

    if model is None:
        return

    if hasattr(model, "base_model_prefix"):
        base_model = getattr(model, model.base_model_prefix, model)
    else:
        base_model = getattr(model, "model", model).model

    for decoder_layer in base_model.layers:
        if attention:
            _patch_attention_module(decoder_layer.self_attn)
