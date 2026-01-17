"""
FLy (Training-Free Loosely Speculative Decoding) verification implementation.

This module implements the exact FLy algorithm from the paper:
"Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match"

Reference: FLy-paper.txt, Section 2.2

Algorithm:
1. Entropy-level gate: Compute normalized entropy h_j = -Σ(p log p) / log|V|
   - If h_j < θ (threshold): Strict rejection (reject all tokens from mismatch position)
   - If h_j ≥ θ: Defer to token-level window
2. Token-level deferred window: Monitor W tokens after mismatch
   - If no further mismatch in window: Accept the initial mismatch
   - If another mismatch occurs: Reject (target model is correcting)
"""

import torch
import numpy as np
from typing import Tuple


def fly_verify(
    *,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    logits: torch.Tensor,
    eos_token_id: int,
    entropy_threshold: float = 0.3,
    window_size: int = 6,
) -> int:
    """
    FLy verification: Accept semantically correct mismatches using entropy gate and deferred window.
    
    Args:
        draft_ids: Draft token sequence [K] (excluding root token)
        target_ids: Target token sequence [K] (excluding bonus token)
        logits: Target model logits [1, K, |V|] for the K comparison positions
        eos_token_id: End-of-sequence token ID
        entropy_threshold: Entropy threshold θ for gate (default: 0.3)
        window_size: Deferred window size W (default: 6)
    
    Returns:
        accept_len: Number of accepted tokens (excluding bonus token), in range [0, K]
    
    Algorithm (from FLy paper):
    1. Find mismatch positions J = {j | draft_ids[j] != target_ids[j]}
    2. For each mismatch j ∈ J:
       a. Compute normalized entropy h_j = -Σ(p log p) / log|V|
       b. Entropy gate: Gate(j) = Strict if h_j < θ, else Defer
       c. If Strict: reject all tokens from position j, return j
       d. If Defer: monitor window [j+1, j+W] for further mismatches
         - If no mismatch in window: accept mismatch at j
         - If mismatch in window: reject (course correction), return j
    3. Return min of all rejection positions, or K+1 if all accepted
    """
    K = draft_ids.size(0)
    assert target_ids.size(0) == K, f"draft_ids and target_ids must have same length K, got {K} and {target_ids.size(0)}"
    assert logits.size(1) == K, f"logits must have K positions, got {logits.size(1)} vs K={K}"
    
    # Compute match indicators: ∆_i = 1[draft_ids[i] == target_ids[i]]
    # Compare draft_ids[0:K] with target_ids[0:K]
    match_indicators = (draft_ids == target_ids) & (target_ids != eos_token_id)
    
    # Find mismatch positions: J = {j | ∆_j = 0}
    mismatch_positions = torch.where(~match_indicators)[0].tolist()
    
    # If no mismatches, accept all K draft tokens (bonus token handled separately)
    if not mismatch_positions:
        return K
    
    # Compute normalized entropy for each position (Equation 5 from paper)
    vocab_size = logits.size(-1)
    probs = torch.softmax(logits, dim=-1)  # [1, K, |V|]
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, K, |V|]
    
    # Entropy: -Σ_v p(v) log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)  # [1, K]
    
    # Normalized entropy: h_j = entropy / log|V| (Equation 5)
    max_entropy = np.log(vocab_size)
    normalized_entropy = entropy / max_entropy  # [1, K]
    normalized_entropy = normalized_entropy.squeeze(0)  # [K]
    
    # Find first rejection position
    # s_gate: min{j | Gate(j) = Strict} if exists, else K+1
    # s_defer: min{j | DeferDecide(j) = Reject} if exists, else K+1
    # s_fly = min(s_gate, s_defer)
    
    first_strict_reject = None
    first_defer_reject = None
    
    for j in mismatch_positions:
        # Step 1: Entropy-level gate (Equation 6)
        h_j = normalized_entropy[j].item()
        
        if h_j < entropy_threshold:
            # Gate = Strict: reject all tokens from position j
            first_strict_reject = j
            break
        
        # Gate = Defer: check token-level deferred window
        # Check if window extends beyond available tokens
        if j + window_size > K:
            # Boundary case: not enough tokens to judge, reject conservatively
            first_defer_reject = j
            break
        
        # Count mismatches in window [j+1, j+W] (Equation 8)
        # N_W(j) = Σ_{i=j+1}^{j+W} (1 - ∆_i)
        window_start = j + 1
        window_end = min(j + window_size + 1, K)
        window_matches = match_indicators[window_start:window_end]
        n_mismatches_in_window = (~window_matches).sum().item()
        
        # DeferDecide (Equation 9)
        # Accept if: h_j ≥ θ AND N_W(j) = 0 AND j + W ≤ K
        # Reject otherwise
        if n_mismatches_in_window > 0:
            # Another mismatch in window → target is correcting → reject
            first_defer_reject = j
            break
        # Otherwise, mismatch at j is accepted (no rejection yet)
    
    # Compute s_gate and s_defer (Equations 7 and 10)
    s_gate = first_strict_reject if first_strict_reject is not None else K + 1
    s_defer = first_defer_reject if first_defer_reject is not None else K + 1
    
    # Final acceptance length (Equation 11)
    # s_fly = min(s_gate, s_defer) is the first rejection position
    # If rejection at position j, we accept positions 0 to j-1, which is j tokens
    # So accept_len = min(s_gate, s_defer)
    s_fly = min(s_gate, s_defer)
    
    # If s_fly = K+1, all tokens accepted (including bonus handled separately)
    # Otherwise, s_fly is the rejection position, and we accept s_fly tokens
    accept_len = s_fly if s_fly <= K else K
    
    return int(accept_len)


def fly_verify_sequence(
    *,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    logits: torch.Tensor,
    eos_token_id: int,
    entropy_threshold: float = 0.3,
    window_size: int = 6,
    max_defer_sequence_length: int = 1,
) -> int:
    """
    FLy verification variant: Accept semantically correct token sequences using entropy gate and deferred window.
    
    This variant treats a sequence of tokens [j, j+k-1] as a single unit for verification.
    When a mismatch occurs at position j, it tries to accept a sequence of length k (1 to max_defer_sequence_length)
    by checking the entire sequence as a whole.
    
    Args:
        draft_ids: Draft token sequence [K] (excluding root token)
        target_ids: Target token sequence [K] (excluding bonus token)
        logits: Target model logits [1, K, |V|] for the K comparison positions
        eos_token_id: End-of-sequence token ID
        entropy_threshold: Entropy threshold θ for gate (default: 0.3)
        window_size: Deferred window size W (default: 6)
        max_defer_sequence_length: Maximum length of sequence that can be deferred and accepted (default: 1)
    
    Returns:
        accept_len: Number of accepted tokens (excluding bonus token), in range [0, K]
    
    Algorithm:
    1. Find mismatch positions J = {j | draft_ids[j] != target_ids[j]}
    2. For the first mismatch j ∈ J:
       a. Compute normalized entropy h_j = -Σ(p log p) / log|V|
       b. Entropy gate: Gate(j) = Strict if h_j < θ, else Defer
       c. If Strict: reject all tokens from position j, return j
       d. If Defer: try different sequence lengths k (from max_defer_sequence_length down to 1)
          - For candidate sequence [j, j+k-1]:
            * Entropy check: all positions in sequence must have entropy >= θ
            * Window check: check window [j+k, j+k+W-1] for mismatches
          - If sequence passes: accept entire sequence [j, j+k-1]
          - If no sequence passes: reject, return j
    3. After accepting a sequence (if any), continue checking subsequent positions with single-token FLy logic
    4. Return the total accept length
    """
    K = draft_ids.size(0)
    assert target_ids.size(0) == K, f"draft_ids and target_ids must have same length K, got {K} and {target_ids.size(0)}"
    assert logits.size(1) == K, f"logits must have K positions, got {logits.size(1)} vs K={K}"
    assert max_defer_sequence_length >= 1, f"max_defer_sequence_length must be >= 1, got {max_defer_sequence_length}"
    
    # Compute match indicators: ∆_i = 1[draft_ids[i] == target_ids[i]]
    match_indicators = (draft_ids == target_ids) & (target_ids != eos_token_id)
    
    # Find mismatch positions: J = {j | ∆_j = 0}
    mismatch_positions = torch.where(~match_indicators)[0].tolist()
    
    # If no mismatches, accept all K draft tokens (bonus token handled separately)
    if not mismatch_positions:
        return K
    
    # Compute normalized entropy for each position (Equation 5 from paper)
    vocab_size = logits.size(-1)
    probs = torch.softmax(logits, dim=-1)  # [1, K, |V|]
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, K, |V|]
    
    # Entropy: -Σ_v p(v) log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)  # [1, K]
    
    # Normalized entropy: h_j = entropy / log|V| (Equation 5)
    max_entropy = np.log(vocab_size)
    normalized_entropy = entropy / max_entropy  # [1, K]
    normalized_entropy = normalized_entropy.squeeze(0)  # [K]
    
    # Check first mismatch for sequence acceptance
    first_mismatch_j = mismatch_positions[0]
    h_j = normalized_entropy[first_mismatch_j].item()
    
    # Step 1: Entropy-level gate
    if h_j < entropy_threshold:
        # Gate = Strict: reject all tokens from position j
        return first_mismatch_j
    
    # Step 2: Gate = Defer: try different sequence lengths
    # Try from largest to smallest to find the maximum acceptable sequence
    sequence_accepted = False
    accept_sequence_length = 0
    
    for k in range(max_defer_sequence_length, 0, -1):
        # Check boundary: sequence [j, j+k-1] must be within [0, K)
        if first_mismatch_j + k > K:
            continue  # Sequence extends beyond available tokens, try smaller k
        
        # a. Entropy check: for sequence [j, j+k-1]
        #    - Match positions: directly accept (no entropy check needed)
        #    - Mismatch positions: must have entropy >= θ
        sequence_match_indicators = match_indicators[first_mismatch_j:first_mismatch_j+k]
        sequence_entropies = normalized_entropy[first_mismatch_j:first_mismatch_j+k]
        
        # Check entropy only for mismatch positions
        mismatch_positions_in_sequence = torch.where(~sequence_match_indicators)[0]
        if mismatch_positions_in_sequence.numel() > 0:
            # There are mismatches in the sequence, check their entropy
            mismatch_entropies = sequence_entropies[mismatch_positions_in_sequence]
            if not (mismatch_entropies >= entropy_threshold).all().item():
                continue  # Not all mismatch positions pass entropy check, try smaller k
        
        # b. Window check: check window [j+k, j+k+window_size-1] for mismatches
        window_start = first_mismatch_j + k
        window_end = min(first_mismatch_j + k + window_size, K)
        
        if window_start >= K:
            # Boundary case: sequence extends to the end, cannot check window
            # Conservative: only accept if k == 1 (single token)
            if k == 1:
                sequence_accepted = True
                accept_sequence_length = k
                break
            else:
                continue  # Try smaller k
        
        # Check window for mismatches
        window_matches = match_indicators[window_start:window_end]
        n_mismatches_in_window = (~window_matches).sum().item()
        
        # c. If window has no mismatches, accept the entire sequence
        if n_mismatches_in_window == 0:
            sequence_accepted = True
            accept_sequence_length = k
            break
    
    if sequence_accepted:
        # Accept sequence [j, j+accept_sequence_length-1]
        accept_len = first_mismatch_j + accept_sequence_length
        
        # Continue checking from position j+accept_sequence_length with single-token FLy logic
        # Find next mismatch after the accepted sequence
        next_pos = first_mismatch_j + accept_sequence_length
        while next_pos < K:
            if match_indicators[next_pos].item():
                # Match: accept and continue
                accept_len = next_pos + 1
                next_pos += 1
            else:
                # Mismatch: check with single-token FLy logic
                h_next = normalized_entropy[next_pos].item()
                if h_next < entropy_threshold:
                    # Strict reject
                    break
                
                # Defer: check window
                if next_pos + window_size > K:
                    # Boundary case: not enough tokens to judge
                    break
                
                window_start = next_pos + 1
                window_end = min(next_pos + window_size + 1, K)
                window_matches = match_indicators[window_start:window_end]
                n_mismatches = (~window_matches).sum().item()
                
                if n_mismatches > 0:
                    # Reject
                    break
                
                # Accept single token
                accept_len = next_pos + 1
                next_pos += 1
        
        return min(accept_len, K)
    else:
        # No sequence passed, reject at position j
        return first_mismatch_j
