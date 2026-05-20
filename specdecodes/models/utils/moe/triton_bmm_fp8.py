"""Triton FP8 Batched Matrix Multiply with Per-Expert Dequantization and Scaling.

Replaces FlashInfer's bmm_fp8 with a torch.compile-friendly Triton kernel that:
- Handles per-expert scale factors (per-row/column scaling)
- Accumulates in FP32 for numerical stability
- Outputs in BF16
- Works seamlessly with torch.compile (registered as custom op)

This kernel is the fusion of:
  bmm_fp8(a_fp8, b_fp8, scale_a, scale_b, dtype=bfloat16)

into a single Triton kernel that applies scales directly.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# Set custom library name
LIB_NAME = "expspec"

# Create a custom namespace for our ops
lib_def = torch.library.Library(LIB_NAME, "DEF")

# Define the operation schema
lib_def.define("bmm_fp8(Tensor a_fp8, Tensor b_fp8, Tensor scale_a, Tensor scale_b, int dtype) -> Tensor")


@triton.jit
def _kernel_bmm_fp8_per_expert(
    # Input matrices (FP8)
    a_fp8_ptr,
    b_fp8_ptr,
    # Scales (passed as single-element pointers for torch.compile compatibility)
    scale_a_ptr,  # [1]: input activation scale (pass as tensor, not scalar)
    scale_b_ptr,  # [batch]: per-expert weight scale
    # Output
    out_ptr,
    # Metadata
    M,
    N,
    K,
    batch,
    stride_a_batch,
    stride_a_m,
    stride_a_k,
    stride_b_batch,
    stride_b_n,
    stride_b_k,
    stride_out_batch,
    stride_out_m,
    stride_out_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched FP8 matrix multiply with per-expert dequantization.

    Computes: out[b, m, n] = (a_fp8[b, m, :] @ b_fp8[b, n, :].T) * scale_a * scale_b[b]

    Args:
        a_fp8_ptr: [batch, M, K] FP8 matrix
        b_fp8_ptr: [batch, N, K] FP8 matrix (transposed layout, stored as [batch, N, K])
        scale_a: scalar dequantization scale for activation
        scale_b_ptr: [batch] per-expert dequantization scales
        out_ptr: [batch, M, N] output in BF16
        M, N, K: matrix dimensions
        batch: batch size (top_n)
        stride_*: strides for each tensor
        BLOCK_M, BLOCK_N, BLOCK_K: tile sizes
    """
    # 3D block grid: [batch, M_tiles, N_tiles]
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Bounds check
    if pid_batch >= batch:
        return

    # Tile offsets
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    # Pointer offsets for this batch
    a_ptr_base = a_fp8_ptr + pid_batch * stride_a_batch
    b_ptr_base = b_fp8_ptr + pid_batch * stride_b_batch

    # FP32 accumulator (no precision loss during matrix multiply)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load scalar scale factors from pointers (always single element)
    scale_a = tl.load(scale_a_ptr)  # scalar
    scale_b = tl.load(scale_b_ptr + pid_batch)  # per-expert

    # K-loop: process in blocks
    for k_offset in range(0, K, BLOCK_K):
        # K bounds for this iteration
        off_k_block = off_k + k_offset

        # Load A block: [BLOCK_M, BLOCK_K] in FP8
        # Layout: [batch, M, K] row-major
        a_ptr = (
            a_ptr_base
            + off_m[:, None] * stride_a_m   # [BLOCK_M, 1] * stride_a_m
            + off_k_block[None, :] * stride_a_k     # [1, BLOCK_K] * stride_a_k
        )   # [BLOCK_M, BLOCK_K] pointer offsets
        a_mask = (off_m[:, None] < M) & (off_k_block[None, :] < K)
        a_block = tl.load(a_ptr, mask=a_mask, other=0.0).to(tl.float32)

        # Load B block: [BLOCK_K, BLOCK_N] in FP8
        # Layout: [batch, N, K] row-major (but we read as transposed)
        # We want b[n, k] from b[batch, n, k] layout
        b_ptr = (
            b_ptr_base
            + off_n[None, :] * stride_b_n   # [1, BLOCK_N] * stride_b_n
            + off_k_block[:, None] * stride_b_k     # [BLOCK_K, 1] * stride_b_k
        )   # [BLOCK_K, BLOCK_N] pointer offsets (transposed access)
        b_mask = (off_n[None, :] < N) & (off_k_block[:, None] < K)
        b_block = tl.load(b_ptr, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += a @ b^T
        # a is [BLOCK_M, BLOCK_K], b is [BLOCK_K, BLOCK_N]
        acc += tl.dot(a_block, b_block)

    # Apply scales: dequantization
    # acc was accumulated in FP32, now scale
    acc = acc * scale_a * scale_b

    # Convert to output dtype (BF16)
    out = acc.to(tl.bfloat16)

    # Store output: [batch, M, N]
    partial_out_ptr = (
        out_ptr
        + pid_batch * stride_out_batch
        + off_m[:, None] * stride_out_m
        + off_n[None, :] * stride_out_n
    )
    out_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(partial_out_ptr, out, mask=out_mask)


def _triton_bmm_fp8_impl_kernel(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    dtype: int,  # Pass as int code to satisfy custom op requirements
) -> torch.Tensor:
    """Internal Triton kernel launch (actual computation)."""
    # Convert dtype int back to torch.dtype
    if isinstance(dtype, int):
        if dtype == 5:  # torch.bfloat16
            dtype_torch = torch.bfloat16
        elif dtype == 1:  # torch.float32
            dtype_torch = torch.float32
        elif dtype == 2:  # torch.float16
            dtype_torch = torch.float16
        else:
            dtype_torch = torch.bfloat16
    else:
        dtype_torch = dtype
    
    # Allocate output (works for both fake and real tensors)
    batch, M, K = a_fp8.shape
    batch2, N, K2 = b_fp8.shape  # b is [batch, N, K]
    assert batch == batch2 and K == K2, (
        f"Batch and K mismatch: a {a_fp8.shape} vs b {b_fp8.shape}"
    )
    
    out = torch.zeros((batch, M, N), dtype=dtype_torch, device=a_fp8.device)
    
    # Skip actual kernel execution for fake tensors or non-CUDA devices
    # torch.compile will trace this function with fake tensors during compilation
    if not isinstance(a_fp8, torch.Tensor):
        return out
    
    try:
        # Try to get data pointer - this will fail for fake tensors
        _ = a_fp8.data_ptr()
    except (RuntimeError, AttributeError):
        # Fake tensor - return zeros (used during torch.compile tracing)
        return out
    
    if not a_fp8.is_cuda or a_fp8.device.type != "cuda":
        return out

    # Ensure inputs are contiguous
    a_fp8 = a_fp8.contiguous()
    b_fp8 = b_fp8.contiguous()

    # Normalize scale_a: ensure it's a 1-element tensor
    if not isinstance(scale_a, torch.Tensor):
        scale_a = torch.tensor(scale_a, dtype=torch.float32, device=a_fp8.device)
    if scale_a.numel() != 1:
        raise ValueError(f"scale_a must be scalar, got shape {scale_a.shape}")
    scale_a = scale_a.unsqueeze(0) if scale_a.dim() == 0 else scale_a.view(1)
    scale_a = scale_a.to(dtype=torch.float32, device=a_fp8.device).contiguous()

    # Ensure scale_b is 1D [batch]
    if not isinstance(scale_b, torch.Tensor):
        scale_b = torch.tensor(scale_b, dtype=torch.float32, device=a_fp8.device)
    if scale_b.numel() == 1:
        scale_b = scale_b.expand(batch)
    elif scale_b.shape[0] != batch:
        raise ValueError(
            f"scale_b must have batch size {batch}, got shape {scale_b.shape}"
        )
    scale_b = scale_b.to(dtype=torch.float32, device=a_fp8.device).contiguous()

    # Block size tuning
    BLOCK_M = 16
    BLOCK_N = 256
    BLOCK_K = 128

    # Ensure not zero-sized
    if M == 0 or N == 0 or K == 0:
        return out

    # Grid: [batch, ceil(M/BLOCK_M), ceil(N/BLOCK_N)]
    grid = (batch, (M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    # Launch kernel
    _kernel_bmm_fp8_per_expert[grid](
        a_fp8,
        b_fp8,
        scale_a,
        scale_b,
        out,
        M,
        N,
        K,
        batch,
        a_fp8.stride(0),
        a_fp8.stride(1),
        a_fp8.stride(2),
        b_fp8.stride(0),
        b_fp8.stride(1),
        b_fp8.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


# Register the operation as a custom op for torch.compile
lib_impl = torch.library.Library(LIB_NAME, "IMPL")


def bmm_fp8_impl(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    dtype: int,
) -> torch.Tensor:
    """Custom op implementation registered with torch.library."""
    return _triton_bmm_fp8_impl_kernel(a_fp8, b_fp8, scale_a, scale_b, dtype)


def bmm_fp8_meta_impl(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    dtype: int,
) -> torch.Tensor:
    """Meta implementation for shape inference during torch.compile tracing.
    
    Returns a tensor with realistic scale so downstream operations (layer norms,
    activations) behave similarly to eager execution during compilation.
    """
    # Convert dtype int back to torch.dtype
    if isinstance(dtype, int):
        if dtype == 5:  # torch.bfloat16
            dtype_torch = torch.bfloat16
        elif dtype == 1:  # torch.float32
            dtype_torch = torch.float32
        elif dtype == 2:  # torch.float16
            dtype_torch = torch.float16
        else:
            dtype_torch = torch.bfloat16
    else:
        dtype_torch = dtype
    
    batch, M, K = a_fp8.shape
    _, N, _ = b_fp8.shape
    
    # Return shape-correct output with proper scale for realistic tracing
    # Use scale_a * scale_b as a heuristic for the output magnitude
    # This helps downstream operations (layer norms, etc.) behave realistically
    scale_factor = scale_a * scale_b.mean() if hasattr(scale_b, 'mean') else scale_a
    
    # Create output with realistic statistics for better tracing behavior
    # Using a small default scale (e.g., 0.01) to match typical intermediate activations
    output = torch.ones((batch, M, N), dtype=dtype_torch, device=a_fp8.device) * 0.01
    return output


# Register both meta (for torch.compile tracing) and CUDA (for actual execution)
lib_impl.impl("bmm_fp8", bmm_fp8_meta_impl, "Meta")
lib_impl.impl("bmm_fp8", bmm_fp8_impl, "CUDA")


def triton_bmm_fp8(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Batched FP8 matrix multiply with per-expert scales.

    This function uses torch.library registration with both Meta and CUDA dispatch.
    - Meta: Used during torch.compile tracing for shape inference (returns zeros)
    - CUDA: Used for actual execution (runs Triton kernel)

    Args:
        a_fp8: [batch, M, K] activation in FP8
        b_fp8: [batch, N, K] weight in FP8 (pre-transposed layout)
        scale_a: scalar or [1] activation dequantization scale (tensor)
        scale_b: [batch] or scalar per-expert weight dequantization scale (tensor)
        dtype: output dtype (default: torch.bfloat16)

    Returns:
        out: [batch, M, N] in dtype (typically BF16)

    Note:
        The kernel expects b in [batch, N, K] layout. Callers should pre-transpose
        their weights (e.g., self.gate_up_packed_fp8.transpose(-2, -1) passes
        [batch, hidden, 2*intermediate] which is [batch, K, N] → [batch, N, K] style).

    Example:
        >>> a = torch.randn(32, 128, 2048, dtype=torch.float8_e4m3fn)  # [batch, M, K]
        >>> b = torch.randn(32, 1024, 2048, dtype=torch.float8_e4m3fn) # [batch, N, K]
        >>> scale_a = torch.tensor(0.01)
        >>> scale_b = torch.randn(32)
        >>> out = triton_bmm_fp8(a, b, scale_a, scale_b)
        >>> assert out.shape == (32, 128, 1024)
    """
    # Convert dtype to int for custom op
    if dtype == torch.bfloat16:
        dtype_int = 5
    elif dtype == torch.float32:
        dtype_int = 1
    elif dtype == torch.float16:
        dtype_int = 2
    else:
        dtype_int = 5  # default to bfloat16
    
    # Call via the custom op - torch.compile will dispatch to Meta during tracing
    # and CUDA during actual execution
    return torch.ops.expspec.bmm_fp8(a_fp8, b_fp8, scale_a, scale_b, dtype_int)
