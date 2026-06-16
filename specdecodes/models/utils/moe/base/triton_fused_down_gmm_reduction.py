"""
Grouped Down Projection BMM + Sparse Reduction Kernel
Torch Compile Friendly Implementation with Meta Tensor Support
Naming Convention: E (Experts), T (Total Tokens = batch*seq_len), IM (Intermediate), H (Hidden)
"""
import torch
import triton
import triton.language as tl

LIB_NAME = "expspec"

@triton.jit
def _fused_down_gmm_reduction(
    # Pointers
    interm_ptr,             # [T*top_k, IM] (BF16 intermediate, sorted-by-expert)
    w_down_ptr,             # [E, H, IM] (BF16 down weights)
    active_experts_ptr,     # [num_active_experts] (expert ids with work)
    token_offsets_ptr,      # [E + 1] (cumulative boundaries into sorted_token_ids per expert)
    sorted_token_ids_ptr,   # [T*top_k] (original flat positions, sorted by expert)
    routing_weights_ptr,    # [T*top_k] (BF16 per-assignment routing weights, original order)
    out_ptr,                # [T, H] (BF16 FFN output activations)

    # Metadata
    T, IM, H, top_k,

    stride_interm_t, stride_interm_im,
    stride_w_down_e, stride_w_down_h, stride_w_down_im,
    stride_out_t, stride_out_h,

    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_IM: tl.constexpr,
):
    # ==========================================
    # Grid Coordinates: [num_active_experts, T_tiles, H_tiles]
    # ==========================================
    pid_e = tl.program_id(0)        # active expert index
    pid_t = tl.program_id(1)        # Token block index (within this expert's slice)
    pid_h = tl.program_id(2)        # Hidden feature block index

    # ===========================================
    # Find tokens which activate this expert
    # ===========================================
    expert_id = tl.load(active_experts_ptr + pid_e)
    off_t_start = tl.load(token_offsets_ptr + expert_id)
    off_t_end = tl.load(token_offsets_ptr + expert_id + 1)

    # ==========================================
    # If no tokens assigned to this expert tile, exit early
    # ==========================================
    if pid_t * BLOCK_T >= (off_t_end - off_t_start):
        return

    # ==========================================
    # Tile offsets
    # ==========================================
    off_t = off_t_start + pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    off_im = tl.arange(0, BLOCK_IM)

    # ==========================================
    # Find the original token indices and routing weights for this block
    # ==========================================
    flatten_token_ids = tl.load(sorted_token_ids_ptr + off_t, mask=off_t < off_t_end, other=0)   # [BLOCK_T]
    token_ids = flatten_token_ids // top_k          # [BLOCK_T] original token indices in [0, T)
    rw = tl.load(routing_weights_ptr + flatten_token_ids, mask=off_t < off_t_end, other=0.0).to(tl.float32)     # [BLOCK_T] routing weights for this block of tokens

    # ==========================================
    # Pointer offsets
    # ==========================================
    w_down_ptr_base = w_down_ptr + expert_id * stride_w_down_e   # base for w_down of this expert

    # ==========================================
    # Initialize accumulator
    # ==========================================
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # ==========================================
    # Main Loop (reduction over IM)
    # ==========================================
    for im_offset in range(0, IM, BLOCK_IM):
        # ==========================================
        # 1. Inner loop offset (IM dimension)
        # ==========================================
        off_im_block = off_im + im_offset

        # ==========================================
        # 2. Load Interm block: [BLOCK_T, BLOCK_IM] (already sorted by expert)
        # ==========================================
        interm_tile_ptr = (
            interm_ptr
            + off_t[:, None] * stride_interm_t
            + off_im_block[None, :] * stride_interm_im
        )
        interm_mask = (off_t[:, None] < off_t_end) & (off_im_block[None, :] < IM)
        interm_block = tl.load(interm_tile_ptr, mask=interm_mask, other=0.0)

        # ==========================================
        # 3. Load W_down block: [BLOCK_IM, BLOCK_H] (Transposed read)
        # ==========================================
        w_down_tile_ptr = (
            w_down_ptr_base
            + off_h[None, :] * stride_w_down_h
            + off_im_block[:, None] * stride_w_down_im
        )
        w_mask = (off_h[None, :] < H) & (off_im_block[:, None] < IM)
        w_down_block = tl.load(w_down_tile_ptr, mask=w_mask, other=0.0)

        # ==========================================
        # 4. Accumulate: acc += interm @ w_down
        # ==========================================
        acc += tl.dot(interm_block, w_down_block).to(tl.float32)

    # ==========================================
    # 5. Apply per-assignment routing weight
    # ==========================================
    out = acc * rw[:, None]     # Broadcast to [BLOCK_T, BLOCK_H]

    # ==========================================
    # 6. Atomic-add to output [T, H] at original token positions
    #    (multiple experts contribute to the same token; needs atomic)
    # ==========================================
    out_ptr_base = (
        out_ptr
        + token_ids[:, None] * stride_out_t
        + off_h[None, :] * stride_out_h
    )
    out_mask = (off_t[:, None] < off_t_end) & (off_h[None, :] < H)

    tl.atomic_add(out_ptr_base, out.to(tl.bfloat16), mask=out_mask)


# ==========================================
# Register the Custom Op
# ==========================================
@torch.library.custom_op(f"{LIB_NAME}::fused_down_gmm_reduction", mutates_args=())
def triton_fused_down_gmm_reduction(
    interm: torch.Tensor,             # [T*top_k, IM] (BF16, sorted-by-expert)
    w_down: torch.Tensor,             # [E, H, IM] (BF16)
    active_experts: torch.Tensor,     # [num_active_experts] (int64)
    token_offsets: torch.Tensor,      # [E + 1] (int64)
    sorted_token_ids: torch.Tensor,   # [T*top_k] (int64) original flat positions, sorted
    routing_weights: torch.Tensor,    # [T*top_k] (BF16) per-assignment weights, original order
    T: int,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Python wrapper for the fused grouped down BMM + weighted reduction kernel.

    For each (token, expert) assignment:
        out[token] += routing_weights[orig_pos] * (interm[sorted_pos, :] @ w_down[expert].T)
    """
    _, IM = interm.shape
    _, H, _ = w_down.shape
    num_active_experts = active_experts.shape[0]

    # ==========================================
    # 1. Allocate output: [T, H] (BF16, zero-initialized for atomic_add)
    # ==========================================
    out = torch.zeros((T, H), dtype=dtype, device=interm.device)

    # ==========================================
    # 2. Ensure inputs are stacked
    # ==========================================
    interm, w_down = interm.contiguous(), w_down.contiguous()
    active_experts, token_offsets, sorted_token_ids = active_experts.contiguous(), token_offsets.contiguous(), sorted_token_ids.contiguous()
    routing_weights = routing_weights.contiguous()

    # ==========================================
    # 3. Early exit
    # ==========================================
    if num_active_experts == 0 or T == 0 or H == 0 or IM == 0:
        return out

    # ==========================================
    # 4. Launch Triton kernel
    # ==========================================
    BLOCK_T, BLOCK_H, BLOCK_IM = 4, 256, 128
    grid = (
        num_active_experts,
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(H, BLOCK_H),
    )

    _fused_down_gmm_reduction[grid](
        interm, w_down,
        active_experts, token_offsets, sorted_token_ids, routing_weights,
        out,
        T, IM, H, top_k,
        interm.stride(0), interm.stride(1),
        w_down.stride(0), w_down.stride(1), w_down.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_IM=BLOCK_IM,
    )

    return out


# ==========================================
# Register a fake implementation for Torch Compile
# ==========================================
@triton_fused_down_gmm_reduction.register_fake
def _fused_down_gmm_reduction_fake(
    interm: torch.Tensor,
    w_down: torch.Tensor,
    active_experts: torch.Tensor,
    token_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    T: int,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _, IM1 = interm.shape
    _, H, IM2 = w_down.shape

    assert IM1 == IM2, "Intermediate dimension must match"

    return interm.new_empty((T, H), dtype=dtype)
