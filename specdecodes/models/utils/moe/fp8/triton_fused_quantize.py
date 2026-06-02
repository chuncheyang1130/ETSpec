import torch
import triton
import triton.language as tl
from typing import Tuple

LIB_NAME = "expspec"

_FP8_E4M3_MAX = 448.0

# ---------------------------------------------------------------------------
# FP8 Triton Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_quantize_bf16_to_fp8(
    x_ptr,                  # Input tensor (BF16/FP16/FP32)
    scale_ptr,              # Scale tensor (FP32)
    out_ptr,                # Output tensor (FP8)
    n_elements,             # Total number of elements
    elements_per_group,     # How many elements share the same scale
    scale_stride,           # 1 if per-expert, 0 if per-tensor
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses the multiplication, clamping, and casting into a single memory pass.
    Automatically handles both per-tensor (scalar) and per-expert (1D) scales.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate which group (expert) this thread belongs to
    group_idx = offsets // elements_per_group
    
    # Load X
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load Scale (Broadcasts automatically if scale_stride == 0)
    scale = tl.load(scale_ptr + (group_idx * scale_stride), mask=mask, other=1.0).to(tl.float32)
    
    # Math: Scale -> Clamp
    x_scaled = x * scale
    x_clamped = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
    
    # Cast and Store (tl.float8e4nv maps to torch.float8_e4m3fn)
    tl.store(out_ptr + offsets, x_clamped.to(tl.float8e4nv), mask=mask)


# ==========================================
# 1. Register the Custom Op
# ==========================================
@torch.library.custom_op(f"{LIB_NAME}::fused_quantize_bf16_to_fp8", mutates_args=())
def triton_fused_quantize_bf16_to_fp8(x: torch.Tensor, scale: torch.Tensor, is_per_expert: bool = False) -> torch.Tensor:
    """Wrapper to launch the Triton quantization kernel."""
    x_flat = x.view(-1)
    n_elements = x_flat.numel()
    out = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    
    scale = scale.view(-1).contiguous()
    
    if is_per_expert:
        B = x.shape[0]
        elements_per_group = n_elements // B
        scale_stride = scale.stride(0)
    else:
        elements_per_group = n_elements
        scale_stride = 0
        
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_quantize_bf16_to_fp8[grid](
        x_flat, scale, out,
        n_elements, elements_per_group, scale_stride,
        FP8_MAX=_FP8_E4M3_MAX,
        BLOCK_SIZE=1024,
    )
    
    return out.view_as(x)


# ==========================================
# 2. Register the Fake Tensor (Meta) Implementation
# ==========================================
@triton_fused_quantize_bf16_to_fp8.register_fake
def _fused_quantize_bf16_to_fp8_fake(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    is_per_expert: bool = False
) -> torch.Tensor:
    """
    Fake tensor implementation for torch.compile.
    The output is exactly the same shape as the input 'x', 
    but the dtype is forced to FP8 (e4m3fn).
    """
    return x.new_empty(x.shape, dtype=torch.float8_e4m3fn)