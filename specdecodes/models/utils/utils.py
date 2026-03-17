import torch
from dataclasses import dataclass
import logging

def invert_mask(mask, dtype): 
    # Inversion using bitwise NOT and multiplication
    return (~mask).to(dtype) * torch.finfo(dtype).min

def get_normalized_entropy(logits):
    vocab_size = logits.size(-1)
    
    # Compute probabilities from logits
    probs = torch.softmax(logits, dim=-1)   # p -> [1, T]
    log_probs = torch.log_softmax(logits, dim=-1)   # log(p) -> [1, T]
    
    # Compute entropy
    entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)  # entropy -> [T]
    max_entropy = torch.log(torch.tensor(vocab_size))  # max entropy happens when |V| tokens have equal probability 1 / |V|
    
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

@dataclass
class DraftParams:
    temperature: float = 1
    max_depth: int = 6
    topk_len: int = 10
    draft_threshold: float = 1.0
    do_sample: bool = False
    max_verify_tokens: int = None
    
    def __post_init__(self):
        self.max_sample_tokens = self.max_depth * self.topk_len + 1
        self.max_verify_tokens = min(self.max_sample_tokens, self.max_verify_tokens) if self.max_verify_tokens is not None else self.max_sample_tokens