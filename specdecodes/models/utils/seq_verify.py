import torch
import numpy as np
from typing import Any, Optional, Tuple
from .lossy_seq_verify import lossy_seq_verify

def verify_seq(
    *,
    draft_ids: torch.Tensor,
    root_ind: int,
    logits: torch.Tensor,
    sample_token_fn,
    tokenizer,
    eos_token_id: Optional[int],
    logits_processor,
    do_sample: bool,
    skip_nodes: int = 0,
    verify_method: str = "exact",
    verify_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:

    method = str(verify_method or "exact").strip().lower()
    vk: dict[str, Any] = dict(verify_kwargs or {})
    
    global_ids = sample_token_fn(logits, logits_processor, do_sample, return_probs=False)[0]  # [T]
    d = draft_ids[root_ind:root_ind + global_ids.size(0)] # [T]
    
    if method == "exact":
        # ---- Exact verifier (existing behavior) ----
        valid = (d[1:] == global_ids[:-1]) & (global_ids[:-1] != eos_token_id)
        accept_len = int(torch.cumprod(valid.to(torch.int64), dim=0).sum().item())
        sampled_tokens = global_ids[:accept_len + 1]
        
    elif method == "lossy":
        vocab_size = logits.size(-1)
        # calculate entropy for each position to see whether target is confident on the prediction
        probs = torch.softmax(logits, dim=-1)   # p -> [1, T]
        log_probs = torch.log_softmax(logits, dim=-1)   # log(p) -> [1, T]
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # create a metric: confidence, which indicates the confidence of target model's token prediction
        max_entropy = np.log(vocab_size)     # By math, when all token has equal prob -> 1 / |V|, entropy = -sigma{ 1/|V| * log(1/|V|) } = sigma{ 1/|V| * log|V| } = log|V|
        confidence = 1 - entropy / max_entropy
        
        threshold = float(vk.get("threshold", 0.9))
        window_size = int(vk.get("window_size", 6))
        
        accept_len = lossy_seq_verify(
            draft_ids=draft_ids,
            target_ids=global_ids,
            tokenizer=tokenizer,
            confidence=confidence,
            eos_token_id=eos_token_id,
            threshold=threshold,
            window_size=window_size
        )

        sampled_tokens = torch.cat([draft_ids[1:1+accept_len], global_ids[accept_len:accept_len + 1]])
        
    else:
        raise ValueError(f"Unsupported verify_method: {verify_method}")

    cmp_len = global_ids.size(0) - 1
    total_len = cmp_len if accept_len == cmp_len else accept_len + 1
    
    return sampled_tokens.unsqueeze(0), None, (total_len, int(accept_len))