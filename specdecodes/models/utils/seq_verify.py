import torch
import numpy as np
from typing import Any, Optional, Tuple
from .lossy_seq_verify import edit_tolerance_verify, fly_verify, fly_verify_sequence, custom_verify, edit_tolerance_verify_v2
# from .fly_seq_verify import fly_verify, fly_verify_sequence
from specdecodes.models.utils.utils import get_normalized_entropy

import logging

def verify_seq(
    *,
    draft_ids: torch.Tensor,
    root_ind: int,
    logits: torch.Tensor,
    sample_token_fn,
    eos_token_id: int,
    logits_processor,
    do_sample: bool,
    skip_nodes: int = 0,
    tokenizer=None,
    verify_method: str = "exact",
    verify_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:

    method = str(verify_method or "exact").strip().lower()
    vk: dict[str, Any] = dict(verify_kwargs or {})
    
    target_ids = sample_token_fn(logits, logits_processor, do_sample, return_probs=False)[0]  # [T]
    draft_ids = draft_ids[root_ind:root_ind + target_ids.size(0)] # [T]
    
    if len(target_ids) == 1:
        return target_ids.unsqueeze(0), None, (1, 0)
    
    if method == "exact":
        # ---- Exact verifier (existing behavior) ----
        valid = (draft_ids[1:] == target_ids[:-1]) & (target_ids[:-1] != eos_token_id)
        accept_len = int(torch.cumprod(valid.to(torch.int64), dim=0).sum().item())
        sampled_tokens = target_ids[:accept_len + 1]
        
        cmp_len = target_ids.size(0) - 1
        total_len = cmp_len if accept_len == cmp_len else accept_len + 1    
        
        return sampled_tokens.unsqueeze(0), None, (total_len, int(accept_len))
    
    # ---- Lossy Verifier ----
    threshold = float(vk.get("threshold", 0.9))
    window_size = int(vk.get("window_size", 6))
    
    normalized_entropy = get_normalized_entropy(logits)  # [T]

    if method == "edit":
        max_edit = int(vk.get("max_edit", 2))
        
        # edit distance verification with max edit tolerance
        accept_len = edit_tolerance_verify(
            draft_ids=draft_ids[1:],
            target_ids=target_ids[:-1],
            entropy=normalized_entropy[:-1],
            eos_token_id=eos_token_id,
            threshold=threshold,
            window_size=window_size,
            max_edit=max_edit,
        )
        
    elif method == "edit_v2":
        max_edit = int(vk.get("max_edit", 2))
        # a more relaxed edit distance verification that allows some tolerance even when the draft token is correct
        accept_len = edit_tolerance_verify_v2(
            draft_ids=draft_ids[1:],
            target_ids=target_ids[:-1],
            entropy=normalized_entropy[:-1],
            eos_token_id=eos_token_id,
            threshold=threshold,
            window_size=window_size,
            max_edit=max_edit,
        )
        
    elif method == "fly":
        # ---- FLy verifier (Training-Free Loosely Speculative Decoding) ----
        # Implements the exact FLy algorithm from the paper
        # Reference: FLy-paper.txt, Section 2.2
        accept_len = fly_verify(
            draft_ids=draft_ids[1:],
            target_ids=target_ids[:-1],
            entropy=normalized_entropy[:-1],
            eos_token_id=eos_token_id,
            threshold=threshold,
            window_size=window_size,
        )

    elif method == "fly_sequence":
        # ---- FLy sequence verifier (variant that accepts token sequences) ----
        # Treats a sequence of tokens as a single unit for verification
        max_tolerance_seq_length = int(vk.get("max_tolerance_seq_length", 1))
        
        # FLy sequence verification
        accept_len = fly_verify_sequence(
            draft_ids=draft_ids[1:],
            target_ids=target_ids[:-1],
            entropy=normalized_entropy[:-1],
            eos_token_id=eos_token_id,
            threshold=threshold,
            window_size=window_size,
            max_tolerance_seq_length=max_tolerance_seq_length,
        )
        
    elif method == "custom":
        max_tolerance_seq_length = int(vk.get("max_tolerance_seq_length", 1))
        
        # custom verfication: test different seq length
        accept_len = custom_verify(
            draft_ids=draft_ids[1:],
            target_ids=target_ids[:-1],
            entropy=normalized_entropy[:-1],
            eos_token_id=eos_token_id,
            threshold=threshold,
            window_size=window_size,
            tolerance_seq_length=max_tolerance_seq_length,
        )
        
    else:
        raise ValueError(f"Unsupported verify_method: {verify_method}")
    
    sampled_tokens = torch.cat([draft_ids[1:1+accept_len], target_ids[accept_len:accept_len + 1]])
    
    cmp_len = target_ids.size(0) - 1
    total_len = cmp_len if accept_len == cmp_len else accept_len + 1
    
    return sampled_tokens.unsqueeze(0), None, (total_len, int(accept_len))