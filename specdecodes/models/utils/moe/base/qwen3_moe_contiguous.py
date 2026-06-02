import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_fused_gate_up_gmm_silu import triton_fused_gate_up_gmm_silu
from .triton_fused_down_gmm_reduction import triton_fused_down_gmm_reduction

import gc
from tqdm.auto import tqdm

def _is_qwen3_moe_block(module: nn.Module) -> bool:
    """Heuristic check for `Qwen3MoeSparseMoeBlock` without importing transformers."""
    return (
        module.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
        and hasattr(module, "experts")
        and hasattr(module, "gate")
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
    )
    
def get_grouped_matmul_metadata(topk_indices: torch.Tensor, num_experts: int):
    """
    Sorts tokens by their assigned expert so Triton can process them in contiguous blocks.
    """
    # ==========================================
    # 1. Flatten indices. Shape: [Tokens * top_k]
    # ==========================================
    flat_indices = topk_indices.view(-1)
    
    # ==========================================
    # 2. Sort the tokens by expert ID -> sorted_token_ids tells us the original position for each token
    # ==========================================
    expert_ids, sorted_token_ids = torch.sort(flat_indices)
    
    # ==========================================
    # 3. Find the boundaries (offsets) for each expert -> We use bincount to count how many tokens each expert got
    # =========================================
    n_tokens_per_expert = torch.bincount(expert_ids, minlength=num_experts)
    
    # ==========================================
    # 4. Cumulative sum gives us the start/end offsets -> DP vector
    # ==========================================
    zero_tensor = torch.zeros(1, dtype=torch.long, device=topk_indices.device)
    token_offsets = torch.cat([zero_tensor, torch.cumsum(n_tokens_per_expert, dim=0)])
    
    # ==========================================
    # 5. Find which experts actually have work to do
    # =q========================================
    active_experts = torch.nonzero(n_tokens_per_expert).squeeze(-1)
    
    return active_experts, token_offsets, sorted_token_ids

class Qwen3MoeContiguousMoeBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int, norm_topk_prob: bool):
        super().__init__()
    
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    @classmethod
    def from_huggingface(cls, hf_block):
        # ============================================
        # Hyperparameters from the original block
        # ============================================
        sample_w = hf_block.experts[0].gate_proj.weight
        target_device, target_dtype = sample_w.device, sample_w.dtype        
        hidden_size, intermediate_size = sample_w.shape[1], sample_w.shape[0]
        num_experts, top_k, norm_topk_prob = hf_block.num_experts, hf_block.top_k, hf_block.norm_topk_prob

        block = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob
        ).to(device=target_device, dtype=target_dtype)

        with torch.no_grad():
            # ================================================
            # 1. Register router weights: HF uses `.gate` (a Linear); some forks expose `.router`. Take whichever exists.
            # ================================================
            router_module = getattr(hf_block, "gate", None) or getattr(hf_block, "router", None)
            if router_module is None:
                raise AttributeError(
                    "hf_block exposes neither .gate nor .router for the MoE router."
                )
            block.register_buffer(
                "router_weights",
                router_module.weight.detach().clone().to(device=target_device, dtype=target_dtype),
            )

            # ================================================
            # 2-1. Build gate contiguous weights: [E, IM, H]
            # ================================================
            block.gate_proj_contiguous = nn.Parameter(
                torch.empty(num_experts, intermediate_size, hidden_size, device=target_device, dtype=target_dtype)
            )
            for e, expert in enumerate(hf_block.experts):
                block.gate_proj_contiguous.data[e].copy_(expert.gate_proj.weight.data)
                expert.gate_proj.weight = None  # Release source parameter to free memory
            
            # ================================================
            # 2-2. Build up contiguous weights: [E, IM, H]
            # ================================================
            block.up_proj_contiguous = nn.Parameter(
                torch.empty(num_experts, intermediate_size, hidden_size, device=target_device, dtype=target_dtype)
            )
            for e, expert in enumerate(hf_block.experts):
                block.up_proj_contiguous.data[e].copy_(expert.up_proj.weight.data)
                expert.up_proj.weight = None    # Release source parameter to free memory

            # ================================================
            # 2-3. Build down contiguous weights: [E, H, IM]
            # ================================================
            block.down_proj_contiguous = nn.Parameter(
                torch.empty(num_experts, hidden_size, intermediate_size, device=target_device, dtype=target_dtype)
            )
            for e, expert in enumerate(hf_block.experts):
                block.down_proj_contiguous.data[e].copy_(expert.down_proj.weight.data)
                expert.down_proj.weight = None  # Release source parameter to free memory

        return block
    
    def _routing_weights(self, x) -> torch.Tensor:
        routing_logits = F.linear(x, self.router_weights)   # [B, T, num_experts]
        topk_vals, topk_indices = torch.topk(routing_logits, k=self.top_k, dim=-1)   # Both [B, T, top_k]
        
        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1, dtype=torch.float32)           # [T, top_k] sums to 1
        else:
            global_softmax = F.softmax(routing_logits, dim=-1, dtype=torch.float32)
            topk_probs = torch.gather(global_softmax, -1, topk_indices)
            
        topk_probs = topk_probs.to(x.dtype)
        return topk_probs, topk_indices

    def forward(self, x):
        # ================================================
        # Flatten leading dims to [T, H]
        # ================================================
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        T = x_flat.shape[0]

        # ================================================
        # 1. Compute gating scores and top-k experts
        # ================================================
        topk_weights, topk_indices = self._routing_weights(x_flat)   # both [T, top_k]

        # ================================================
        # 2. Prepare grouped-matmul metadata (sort tokens by expert)
        # ================================================
        active_experts, token_offsets, sorted_token_ids = get_grouped_matmul_metadata(
            topk_indices=topk_indices,
            num_experts=self.num_experts,
        )

        # ================================================
        # 3. Fused gate + up + SiLU (GMM): [T*top_k, IM] in sorted-by-expert order
        # ================================================
        interm = triton_fused_gate_up_gmm_silu(
            x_flat,
            self.gate_proj_contiguous,
            self.up_proj_contiguous,
            active_experts,
            token_offsets,
            sorted_token_ids,
            top_k=self.top_k,
        )

        # ================================================
        # 4. Fused down + weighted reduction (GMM): [T, H]
        # ================================================
        out = triton_fused_down_gmm_reduction(
            interm,
            self.down_proj_contiguous,
            active_experts,
            token_offsets,
            sorted_token_ids,
            routing_weights=topk_weights.reshape(-1),
            T=T,
            top_k=self.top_k,
        )

        return out.view(bsz, seq_len, hidden)

def apply_contiguous_moe_block_to_qwen_moe(model: nn.Module) -> int:
    """
    Replace the original HF MoE block in Qwen3-MoE with the contiguous expert
    weight version (a drop-in for the target model's MoE so the GMM Triton
    kernels can read per-expert weights as one batched matmul).

    Returns the number of blocks replaced.
    """

    # ==========================================
    # 1: Collect targets (Avoids Iterator Mutation)
    # ==========================================
    block_to_replace = []

    for name, module in model.named_modules():
        if _is_qwen3_moe_block(module):
            block_to_replace.append((name, module))

    # ==========================================
    # 2: Replace with Contiguous MoE Block
    # ==========================================
    for absolute_name, hf_block in tqdm(block_to_replace, desc="Replacing MoE blocks with contiguous weight"):
        # Build new moe block with contiguous weights from the original HuggingFace block
        new_moe_block = Qwen3MoeContiguousMoeBlock.from_huggingface(hf_block)

        # Find parent module name and attribute name
        name_parts = absolute_name.split(".")
        parent_name = ".".join(name_parts[:-1])
        child_name = name_parts[-1]

        # Replace the original block with the contiguous version
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, child_name, new_moe_block)

        # Clean up
        del hf_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return len(block_to_replace)
            