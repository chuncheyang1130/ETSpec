"""Packed top-N MoE block with **W4A16** experts (INT4 weights, bf16 activations).

Sibling of the FP8 block (`qwen3_moe_topn_fp8.py`). Same routing / materialize-
cache / soft top-K redirect plumbing inherited from `PackedTopNMoeBlock`; only
the expert storage and `_expert_forward` differ. Here the kept experts' gate /
up / down weights are quantized to 4 bits with **HQQ** (Half-Quadratic
Quantization) and stored packed 2-per-byte; activations are never quantized
(A16). The two Triton kernels dequantize the int4 weights to bf16 on the fly
and run bf16 tensor-core matmuls.

HQQ (Badri & Shaji, mobiusml/hqq) is calibration-free affine quant:

    W_q  = round(W * scale + zero).clamp(0, 2^nbits - 1)
    W_dq = (W_q - zero) / scale

`scale` is fixed from the per-group min/max; `zero` is refined by a
half-quadratic proximal solver that puts an L_p (p<1) penalty on the
reconstruction error, which is robust to weight outliers. We run it group-wise
along the contraction (input) dim — one (scale, zero) per
(expert, output-channel, group of `group_size`) — so it lines up with the
kernels' per-group dequant. For the kernel we store the dequant *step*
`s = 1/scale` and the float `zero`, so dequant is `W = (W_q - zero) * s`.

Layout (per kept-expert slot):
    gate_proj_packed_int4 : [top_n, IM, H // 2]   uint8 (2 int4 / byte along H)
    up_proj_packed_int4   : [top_n, IM, H // 2]   uint8
    down_proj_packed_int4 : [top_n, H, IM // 2]   uint8
    {gate,up}_proj_scale/zero : [top_n, IM, H // group_size] fp32
    down_proj_scale/zero      : [top_n, H, IM // group_size] fp32
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from transformers.models.qwen3_moe import Qwen3MoeConfig

from .qwen3_moe_topn import (
    PackedTopNMoeBlock,
    _is_qwen3_moe_block,
    _read_target_expert_weight,
    _set_module_by_name,
)
from .hqq_quantize import hqq_quantize_and_pack_int4
from .triton_fused_gate_up_int4_bmm_silu import triton_fused_gate_up_int4_bmm_silu
from .triton_fused_down_int4_bmm_reduction import triton_fused_down_int4_bmm_reduction


__all__ = [
    "PackedTopNINT4MoeBlock",
    "apply_packed_topn_int4_structure",
]


# ---------------------------------------------------------------------------
# W4A16 packed top-N block
# ---------------------------------------------------------------------------
class PackedTopNINT4MoeBlock(PackedTopNMoeBlock):
    """Packed top-N block with HQQ-INT4 weights and bf16 activations (W4A16).

    Inherits routing / materialize-cache / redirect plumbing from
    `PackedTopNMoeBlock`. Overrides expert storage (`_init_expert_weights`),
    the HQQ quantize+pack fill (`_materialize_expert_weights`), and the forward
    (`_routing_weights` reused from the FP8 sibling's sparse form, expert math
    via the two int4 kernels).
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        top_n: int,
        dtype: torch.dtype,
        device: torch.device | str,
        hidden_act: str,
        target_top_k: int,
        redirect_topk: int = 4,
        group_size: int = 128,
        nbits: int = 4,
    ):
        # bf16/fp16 compute dtype for activations + the silu/output path.
        self._compute_dtype = dtype
        self.nbits = int(nbits)
        if self.nbits != 4:
            raise ValueError(f"PackedTopNINT4MoeBlock is W4A16 only (got nbits={nbits}).")
        # group_size must be set before super().__init__ → it sizes the buffers.
        self.group_size = int(group_size)
        h = int(config.hidden_size)
        im = int(config.moe_intermediate_size)
        if h % self.group_size != 0 or im % self.group_size != 0:
            raise ValueError(
                f"group_size ({self.group_size}) must divide both hidden_size "
                f"({h}) and moe_intermediate_size ({im})."
            )
        if self.group_size % 2 != 0:
            raise ValueError(f"group_size ({self.group_size}) must be even (2 int4 / byte).")
        super().__init__(
            config=config,
            top_n=top_n,
            dtype=dtype,
            device=device,
            hidden_act=hidden_act,
            target_top_k=target_top_k,
            redirect_topk=redirect_topk,
        )

    # ----- overrides -----
    def _init_expert_weights(self, dtype: torch.dtype, device: torch.device | str) -> None:
        """Allocate packed int4 weight buffers + per-group HQQ scale/zero."""
        del dtype  # storage is uint8; compute dtype tracked via _compute_dtype
        ng_h = self.hidden_size // self.group_size      # groups along H (gate/up contraction)
        ng_im = self.intermediate_size // self.group_size    # groups along IM (down contraction)

        # Packed 4-bit weights (2 codes per byte along the contraction dim).
        self.register_buffer(
            "gate_proj_packed_int4",
            torch.zeros(
                self.top_n, 
                self.intermediate_size, 
                self.hidden_size // 2, 
                dtype=torch.uint8, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "up_proj_packed_int4",
            torch.zeros(
                self.top_n, 
                self.intermediate_size, 
                self.hidden_size // 2, 
                dtype=torch.uint8, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "down_proj_packed_int4",
            torch.zeros(
                self.top_n, 
                self.hidden_size, 
                self.intermediate_size // 2, 
                dtype=torch.uint8, device=device
            ),
            persistent=False,
        )
        # Per-group dequant step (1/hqq_scale) and float zero point.
        for name, n_out, ng in (
            ("gate_proj", self.intermediate_size, ng_h),
            ("up_proj",   self.intermediate_size, ng_h),
            ("down_proj", self.hidden_size,      ng_im),
        ):
            self.register_buffer(
                f"{name}_scale",
                torch.ones(self.top_n, n_out, ng, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                f"{name}_zero",
                torch.zeros(self.top_n, n_out, ng, dtype=torch.float32, device=device),
                persistent=False,
            )
        # bf16 probe so the inherited `materialize_from_target` reads the right
        # (float) dtype/device for the router buffers — NOT the uint8 storage.
        self.register_buffer(
            "_dtype_probe",
            torch.zeros(1, dtype=self._compute_dtype, device=device),
            persistent=False,
        )

    def reset_random_(self) -> None:
        """No random init — buffers are filled by `_materialize_expert_weights`."""
        pass

    def _packed_expert_parameters(self) -> List[torch.Tensor]:
        return [self.gate_proj_packed_int4, self.up_proj_packed_int4, self.down_proj_packed_int4]

    def _reference_param(self) -> torch.Tensor:
        # Float probe: drives the router-buffer dtype in the base materialize.
        return self._dtype_probe

    @torch.no_grad()
    def _materialize_expert_weights(
        self,
        target_block: nn.Module,
        kept_ids: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        svd_device: torch.device | str,
    ) -> bool:
        """Load kept experts from `target_block`, HQQ-quantize to int4, pack."""
        del svd_device, target_dtype  # storage is uint8; compute dtype is fixed

        gate_proj_real = torch.empty(
            self.top_n, 
            self.intermediate_size, 
            self.hidden_size, 
            dtype=self._compute_dtype, 
            device=target_device
        )
        up_proj_real = torch.empty(
            self.top_n, 
            self.intermediate_size, 
            self.hidden_size, 
            dtype=self._compute_dtype, 
            device=target_device
        )
        down_proj_real = torch.empty(
            self.top_n, 
            self.hidden_size, 
            self.intermediate_size, 
            dtype=self._compute_dtype, 
            device=target_device
        )
        for slot, eid in enumerate(kept_ids.tolist()):
            w_gate, w_up, w_down = _read_target_expert_weight(target_block, int(eid))
            gate_proj_real[slot].copy_(w_gate.to(device=target_device, dtype=self._compute_dtype))
            up_proj_real[slot].copy_(w_up.to(device=target_device, dtype=self._compute_dtype))
            down_proj_real[slot].copy_(w_down.to(device=target_device, dtype=self._compute_dtype))

        # Fused HQQ-INT4 quantize + 2-per-byte pack. Dispatches to a fused
        # Triton kernel on CUDA (registers-only, no per-iteration transients)
        # and falls back to the plain-torch reference on CPU.
        gate_packed, gs, gz = hqq_quantize_and_pack_int4(gate_proj_real, self.group_size)
        up_packed,   us, uz = hqq_quantize_and_pack_int4(up_proj_real,   self.group_size)
        down_packed, ds, dz = hqq_quantize_and_pack_int4(down_proj_real, self.group_size)

        self.gate_proj_packed_int4.copy_(gate_packed)
        self.up_proj_packed_int4.copy_(up_packed)
        self.down_proj_packed_int4.copy_(down_packed)

        self.gate_proj_scale.copy_(gs)
        self.gate_proj_zero.copy_(gz)
        
        self.up_proj_scale.copy_(us)
        self.up_proj_zero.copy_(uz)
        
        self.down_proj_scale.copy_(ds)
        self.down_proj_zero.copy_(dz)
        
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        x = x.view(-1, hidden)                                  # [T, H] bf16

        topn_routing_weights = self._routing_weights(x)         # [T, top_n]

        interm = triton_fused_gate_up_int4_bmm_silu(
            x,
            self.gate_proj_packed_int4, self.up_proj_packed_int4,
            self.gate_proj_scale, self.gate_proj_zero,
            self.up_proj_scale, self.up_proj_zero,
            group_size=self.group_size, dtype=self._compute_dtype,
        )                                                       # [E, T, IM] bf16

        out = triton_fused_down_int4_bmm_reduction(
            interm,
            self.down_proj_packed_int4,
            self.down_proj_scale, self.down_proj_zero,
            topn_routing_weights,
            group_size=self.group_size, dtype=self._compute_dtype,
        )                                                       # [T, H] bf16

        return out.view(bsz, seq_len, hidden)


# ---------------------------------------------------------------------------
# Build-time replacement
# ---------------------------------------------------------------------------
def apply_packed_topn_int4_structure(
    model: nn.Module,
    top_n: int,
    redirect_topk: int = 4,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    group_size: int = 128,
) -> int:
    """Swap every `Qwen3MoeSparseMoeBlock` for a `PackedTopNINT4MoeBlock`.

    Int4 buffers are left zero-filled; the real HQQ fill happens at generate
    time via `materialize_from_target` once the kept set is picked.
    """
    replaced = 0
    block_to_be_replaced = []
    for name, module in list(model.named_modules()):
        if _is_qwen3_moe_block(module):
            block_to_be_replaced.append((name, module))

    for name, module in tqdm(block_to_be_replaced, desc="Constructing Draft INT4 MoE Blocks"):
        target_top_k = int(getattr(module, "top_k"))
        block_dtype = dtype if dtype is not None else next(module.parameters()).dtype
        block_device = device if device is not None else next(module.parameters()).device

        new_block = PackedTopNINT4MoeBlock(
            config=model.config,
            top_n=int(top_n),
            redirect_topk=int(redirect_topk),
            dtype=block_dtype,
            device=block_device,
            hidden_act=model.config.hidden_act,
            target_top_k=target_top_k,
            group_size=int(group_size),
        )

        _set_module_by_name(model, name, new_block)
        replaced += 1

    logging.info(
        "[Packed-MoE-TopN-INT4] Replaced %d MoE blocks (top_n=%d, redirect_topk=%d, "
        "W4A16/HQQ, group_size=%d).",
        replaced,
        int(top_n),
        int(redirect_topk),
        int(group_size),
    )
    return replaced
