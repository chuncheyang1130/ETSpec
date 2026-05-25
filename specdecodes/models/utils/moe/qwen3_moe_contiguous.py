import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe import Qwen3MoeConfig

def _is_qwen3_moe_block(module: nn.Module) -> bool:
    """Heuristic check for `Qwen3MoeSparseMoeBlock` without importing transformers."""
    return (
        module.__class__.__name__ == "Qwen3MoeSparseMoeBlock"
        and hasattr(module, "experts")
        and hasattr(module, "gate")
        and hasattr(module, "num_experts")
        and hasattr(module, "top_k")
    )

class Qwen3MoeContiguousMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
    
        # ================================================
        # Expert weight in contiguous memory
        # ================================================
        self.gate_proj_contiguous = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.hidden_size)
        )
        
        self.up_proj_contiguous = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.hidden_size)
        )
        
        self.down_proj_contiguous = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, config.intermediate_size)
        )
        
    @classmethod
    def from_huggingface(cls, hf_block):
        
        block = cls(config=hf_block.config)
        
        with torch.no_grad():
            # ================================================
            # 1. Register router weights
            # ================================================
            block.register_buffer(
                "router_weights",
                hf_block.router.weight.detach().clone()
            )
            
            # ================================================
            # 2. Stack Gate/Up/Down weights individually
            # ================================================
            stacked_gate = torch.stack(
                [expert.gate_proj.weight.data for expert in hf_block.experts], dim=0
            )
            
            stacked_up = torch.stack(
                [expert.up_proj.weight.data for expert in hf_block.experts], dim=0
            )
            
            stacked_down = torch.stack(
                [expert.down_proj.weight.data for expert in hf_block.experts], dim=0
            )
            
            # ================================================
            # 3. Copy Gate/Up/Down weights individually
            # ================================================            
            block.gate_proj_contiguous.data.copy_(stacked_gate)
            block.up_proj_contiguous.data.copy_(stacked_up)
            block.down_proj_contiguous.data.copy_(stacked_down)
            
        return block
    
    def forward(self, x):
        # TODO: Implement custom triton kernel
        pass
    
def apply_contiguous_moe_block_to_qwen_moe(model: nn.Module) -> None:
    """
    Replace the original MoE block in Qwen3-MoE with the contiguous expert weight version.

    Args:
        model (nn.Module): The Qwen3-MoE model instance to apply the contiguous MoE block to.
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
    for absolute_name, hf_block in block_to_replace:
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
        torch.cuda.empty_cache()
            