"""
Grouped Gate + Up Projection BMM + SiLU Fusion Kernel
Torch Compile Friendly Implementation with Meta Tensor Support
Naming Convention: E (Experts), T (Total Tokens = batch*seq_len), IM (Intermediate), H (Hidden)
"""

import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"

@triton.jit
def _fused_gate_up_gmm_silu(
    # Pointers
    x_ptr,                  # [T, H] (input activations, flattened across batch/seq)
    w_gate_ptr,             # [E, IM, H] (gate weights)
    w_up_ptr,               # [E, IM, H] (up weights)
    active_experts_ptr,     # [num_active_experts] (expert ids with work)
    token_offsets_ptr,      # [E + 1] (cumulative boundaries into sorted_token_ids per expert)
    sorted_token_ids_ptr,   # [T * top_k] (original flat positions, sorted by expert)
    out_ptr,                # [T * top_k, IM] (output activations, sorted-by-expert order)

    # Metadata
    T, IM, H, top_k,

    stride_x_t, stride_x_h,
    stride_w_gate_e, stride_w_gate_im, stride_w_gate_h,
    stride_w_up_e, stride_w_up_im, stride_w_up_h,
    stride_out_t, stride_out_im,
    
    BLOCK_T: tl.constexpr,
    BLOCK_IM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # ==========================================
    # Grid Coordinates & Early Exit: [B, T_tiles, IM_tiles]
    # ==========================================
    pid_e = tl.program_id(0)        # expert block index
    pid_t = tl.program_id(1)        # Token block index
    pid_im = tl.program_id(2)       # Intermediate feature block index
    
    # ===========================================
    # Find tokens which activate this expert 
    # ===========================================
    expert_id = tl.load(active_experts_ptr + pid_e)
    off_t_start = tl.load(token_offsets_ptr + expert_id)
    off_t_end = tl.load(token_offsets_ptr + expert_id + 1)
    
    # ==========================================
    # If no tokens assigned to this expert, exit early
    # ==========================================
    if pid_t * BLOCK_T >= (off_t_end - off_t_start):
        return
    
    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = off_t_start + pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_im = pid_im * BLOCK_IM + tl.arange(0, BLOCK_IM)
    off_h = tl.arange(0, BLOCK_H)
    
    # ==========================================
    # Find the original token indices for this block
    # ==========================================
    flatten_token_ids = tl.load(sorted_token_ids_ptr + off_t, mask=off_t < off_t_end, other=0)   # [BLOCK_T]
    token_ids = flatten_token_ids // top_k      # [BLOCK_T] Get the original token indices by dividing by top_k
    
    # ==========================================
    # Pointer offsets
    # ==========================================
    w_gate_ptr_base = w_gate_ptr + expert_id * stride_w_gate_e           # base for w_gate of this expert
    w_up_ptr_base = w_up_ptr + expert_id * stride_w_up_e                 # base for w_up of this expert

    # ==========================================
    # Initialize accumulators
    # ==========================================
    acc_gate = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_T, BLOCK_IM), dtype=tl.float32)

    # ==========================================
    # Main Loop
    # ==========================================
    for h_offset in range(0, H, BLOCK_H):
        # ==========================================
        # 1. Inner loop offset (H dimension)
        # ==========================================
        off_h_block = off_h + h_offset

        # ==========================================
        # 2. Load X block: [BLOCK_T, BLOCK_H] (gathered by token_ids)
        # ==========================================
        x_tile_ptr = (
            x_ptr
            + token_ids[:, None] * stride_x_t
            + off_h_block[None, :] * stride_x_h
        )
        x_mask = (off_t[:, None] < off_t_end) & (off_h_block[None, :] < H)
        x_block = tl.load(x_tile_ptr, mask=x_mask, other=0.0)

        # ==========================================
        # 3-1. Base mask for W (bounds checking against IM and H)
        # ==========================================
        w_mask = (off_im[None, :] < IM) & (off_h_block[:, None] < H)

        # ==========================================
        # 3-2. Load W_gate block: [BLOCK_H, BLOCK_IM] (Transposed read)
        # ==========================================
        w_gate_tile_ptr = (
            w_gate_ptr_base
            + off_im[None, :] * stride_w_gate_im
            + off_h_block[:, None] * stride_w_gate_h
        )
        w_gate_block = tl.load(w_gate_tile_ptr, mask=w_mask, other=0.0)

        # ==========================================
        # 3-3. Load W_up block: [BLOCK_H, BLOCK_IM] (Transposed read)
        # ==========================================
        w_up_tile_ptr = (
            w_up_ptr_base
            + off_im[None, :] * stride_w_up_im
            + off_h_block[:, None] * stride_w_up_h
        )
        w_up_block = tl.load(w_up_tile_ptr, mask=w_mask, other=0.0)

        # ==========================================
        # 4. Accumulate both: acc += x @ w
        # ==========================================
        acc_gate += tl.dot(x_block, w_gate_block).to(tl.float32)
        acc_up += tl.dot(x_block, w_up_block).to(tl.float32)

    # ==========================================
    # 5. Apply SiLU(gate) * up
    # ==========================================
    out = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    # ==========================================
    # 6. Store output block: [BLOCK_T, BLOCK_IM] in BF16 at sorted positions
    # ==========================================
    out_ptr_base = (
        out_ptr
        + off_t[:, None] * stride_out_t
        + off_im[None, :] * stride_out_im
    )
    out_mask = (off_t[:, None] < off_t_end) & (off_im[None, :] < IM)

    tl.store(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


# ==========================================
# Register the Custom Op
# ==========================================
@torch.library.custom_op(f"{LIB_NAME}::fused_gate_up_gmm_silu", mutates_args=())
def triton_fused_gate_up_gmm_silu(
    x: torch.Tensor,                  # [T, H] (BF16 activations, flattened)
    w_gate: torch.Tensor,             # [E, IM, H] (BF16 gate weights)
    w_up: torch.Tensor,               # [E, IM, H] (BF16 up weights)
    active_experts: torch.Tensor,     # [num_active_experts] (int64) expert ids with work
    token_offsets: torch.Tensor,      # [E + 1] (int64) cumulative offsets into sorted_token_ids
    sorted_token_ids: torch.Tensor,   # [T*top_k] (int64) original flat positions, sorted by expert
    top_k: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Python wrapper for the fused grouped BMM + SiLU kernel.

    For each (token, expert) assignment, computes:
        out[sorted_pos, :] = SiLU(x[token] @ w_gate[expert].T) * (x[token] @ w_up[expert].T)
    Output is in sorted-by-expert order to feed a downstream GMM-style down projection.
    """
    T, H = x.shape
    _, IM, _ = w_gate.shape
    N_total = T * top_k
    num_active_experts = active_experts.shape[0]

    # ==========================================
    # 1. Allocate output: [T*top_k, IM] (sorted-by-expert order)
    # ==========================================
    out = torch.zeros((N_total, IM), dtype=dtype, device=x.device)

    # ==========================================
    # 2. Ensure inputs are contiguous
    # ==========================================
    x, w_gate, w_up = x.contiguous(), w_gate.contiguous(), w_up.contiguous()
    active_experts, token_offsets, sorted_token_ids = active_experts.contiguous(), token_offsets.contiguous(), sorted_token_ids.contiguous()

    # ==========================================
    # 3. Early exit
    # ==========================================
    if num_active_experts == 0 or T == 0 or IM == 0 or H == 0:
        return out

    # ==========================================
    # 4. Launch Triton kernel
    # ==========================================
    BLOCK_T, BLOCK_IM, BLOCK_H = 4, 128, 128
    grid = (
        num_active_experts,
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(IM, BLOCK_IM),
    )

    _fused_gate_up_gmm_silu[grid](
        x, w_gate, w_up,
        active_experts, token_offsets, sorted_token_ids,
        out,
        T, IM, H, top_k,
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_IM=BLOCK_IM,
        BLOCK_H=BLOCK_H,
    )

    return out


# ==========================================
# Register a fake implementation for Torch Compile
# ==========================================
@triton_fused_gate_up_gmm_silu.register_fake
def _fused_gate_up_gmm_silu_fake(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    T, H1 = x.shape
    E1, IM1, H2 = w_gate.shape
    E2, IM2, H3 = w_up.shape

    assert E1 == E2, "Expert dimension must match"
    assert IM1 == IM2, "Intermediate dimension must match"
    assert H1 == H2 and H1 == H3, "Hidden dimension must match"

    N_total = T * top_k
    return x.new_empty((N_total, IM1), dtype=dtype)
