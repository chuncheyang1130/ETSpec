import logging
import torch.nn as nn 
import torch
import transformers

def edit_tolerance_verify(
    *,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    entropy: torch.Tensor,
    eos_token_id: int,
    threshold: float = 0.9,
    window_size: int = 8,
    max_edit: int = 2
):
    # check if draft_ids match target_ids
    is_match = (draft_ids[:] == target_ids[:]) & (target_ids[:] != eos_token_id)
    not_eos = draft_ids[:] != eos_token_id
    max_length = int(torch.cumprod(not_eos.to(torch.int64), dim=0).sum().item()) 
    min_accept_len = int(torch.cumprod(is_match.to(torch.int64), dim=0).sum().item())
    # exact_match = int(torch.cumprod(is_match.to(torch.int64), dim=0).sum().item())
    
    # use for dp on accumulative edit count 
    mismatch_count = [0] * (max_length + 1)
    for i in range(1, max_length+1):
        if not is_match[i-1]:
            if entropy[i-1] < threshold:
                max_length = i-1    # strict reject from here -> set to end position
                break
            elif entropy[i-1] >= threshold:
                mismatch_count[i] = mismatch_count[i-1] + 1
        else:
            mismatch_count[i] = mismatch_count[i-1]
            
    # logging.debug(f"max_length: {max_length}, mismatch_count: {mismatch_count[max_length]}")
    # mismatch_status = ["X" if mismatch_count[i] > mismatch_count[i-1] else "O" for i in range(1, max_length+1)]
        
    # check mismatch token in window
    window_begin, window_end = 0, window_size   # window_end is exclusive
    accept_len = max_length
    
    while window_end <= max_length:
        if mismatch_count[window_end] - mismatch_count[window_begin] > max_edit or mismatch_count[window_end] >= 3:     # TODO: add a maximum edit count stopper check
            if window_begin == 0:
                accept_len = min_accept_len
            else:
                accept_len = window_end - 1
            break
        else:
            window_begin += 1
            window_end += 1
    
    # if accept_len == 0:        
    #     accept_len = max(accept_len, min_accept_len)
    # else:
    #     accept_len = window_end - 1

    # accept match tokens after edit tolerance checkpoint
    # while accept_len < max_length and is_match[accept_len]:
    #     accept_len += 1
        
    # fallback check
    while accept_len > 0:
        if not is_match[accept_len-1]:
            accept_len -= 1
        else:
            break
        
    # logging.info(f"max-length: {max_length} \t/ mismatch-count: {mismatch_count[max_length]} \t/ exact-match: {exact_match}\t/ accept-length: {accept_len} \t/ mismatch status: {''.join(mismatch_status)}")
    return accept_len

def edit_tolerance_verify_v2(
    *,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    entropy: torch.Tensor,
    eos_token_id: int,
    threshold: float = 0.9,
    window_size: int = 8,
    max_edit: int = 2,
    min_interval: int = 3
):
    
    # check if draft_ids match target_ids
    is_match = (draft_ids[:] == target_ids[:]) & (target_ids[:] != eos_token_id)
    not_eos = draft_ids[:] != eos_token_id
    max_length = int(torch.cumprod(not_eos.to(torch.int64), dim=0).sum().item()) 
    min_accept_len = int(torch.cumprod(is_match.to(torch.int64), dim=0).sum().item())
    # exact_match = int(torch.cumprod(is_match.to(torch.int64), dim=0).sum().item())
    
    # use for dp on accumulative edit count 
    mismatch_count = [0] * (max_length + 1)
    for i in range(1, max_length+1):
        if not is_match[i-1]:
            if entropy[i-1] < threshold:
                max_length = i-1    # strict reject from here -> set to end position
                break
            elif entropy[i-1] >= threshold:
                mismatch_count[i] = mismatch_count[i-1] + 1
        else:
            mismatch_count[i] = mismatch_count[i-1]
            
    # logging.debug(f"max_length: {max_length}, mismatch_count: {mismatch_count[max_length]}")
    # mismatch_status = ["X" if mismatch_count[i] > mismatch_count[i-1] else "O" for i in range(1, max_length+1)]
        
    # check mismatch token in window
    window_begin, window_end = 0, window_size   # window_end is exclusive
    accept_len = max_length
    
    while window_end <= max_length:
        if mismatch_count[window_end] - mismatch_count[window_begin] > max_edit or mismatch_count[window_end] >= 3:     # TODO: add a maximum edit count stopper check
            if window_begin != 0:
                accept_len = window_end - 1
                # fallback check
                while accept_len > 0:
                    if not is_match[accept_len-1]:
                        accept_len -= 1
                    else:
                        break
            else:
                accept_len = 0
            break
        else:
            window_begin += 1
            window_end += 1
            
    if accept_len == 0:
        window_begin, window_end = 0, window_size // 2
        while window_end <= max_length:
            if mismatch_count[window_end] - mismatch_count[window_begin] > 1 or mismatch_count[window_end] >= 3:
                if window_begin != 0:
                    accept_len = window_end - 1
                    # fallback check
                    while accept_len > 0:
                        if not is_match[accept_len-1]:
                            accept_len -= 1
                        else:
                            break
                else:
                    accept_len = 0
                break
            else:
                window_begin += 1
                window_end += 1
                
    accept_len = max(accept_len, min_accept_len)

    # logging.info(f"max-length: {max_length} \t/ mismatch-count: {mismatch_count[max_length]} \t/ exact-match: {exact_match}\t/ accept-length: {accept_len} \t/ mismatch status: {''.join(mismatch_status)}")
    return accept_len

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

def fly_verify(
    *,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    entropy: torch.Tensor,
    eos_token_id: int,
    threshold: float = 0.3,     # entropy threshold from paper is 0.3 -> confidence threshold is 0.7
    window_size: int = 6,      
) -> int:
    """
    FLy verification: Accept semantically correct mismatches using entropy gate and deferred window.
    
    Args:
        draft_ids: Draft token sequence [K] (excluding root token)
        target_ids: Target token sequence [K] (excluding bonus token)
        entropy: Normalized entropy for target tokens [K]
        eos_token_id: End-of-sequence token ID
        threshold: Entropy threshold for gate (default: 0.3)
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
    
    # Compute match indicators: ∆_i = 1[draft_ids[i] == target_ids[i]]
    # Compare draft_ids[0:K] with target_ids[0:K]
    is_match = (draft_ids == target_ids) & (target_ids != eos_token_id)
    
    # Find mismatch positions: J = {j | ∆_j = 0}
    mismatch_positions = torch.where(~is_match)[0].tolist()
    
    # If no mismatches, accept all K draft tokens (bonus token handled separately)
    if not mismatch_positions:
        return K
    
    # Find first rejection position
    # s_gate: min{j | Gate(j) = Strict} if exists, else K+1
    # s_defer: min{j | DeferDecide(j) = Reject} if exists, else K+1
    # s_fly = min(s_gate, s_defer)
    
    first_strict_reject = None
    first_defer_reject = None
    
    for j in mismatch_positions:
        # Step 1: Entropy-level gate (Equation 6)        
        if entropy[j] < threshold:
            # Gate = Strict: reject all tokens from position j
            first_strict_reject = j
            break
        
        # Gate = Defer: check token-level deferred window
        # Check if window extends beyond available tokens
        if j + window_size >= K:
            # Boundary case: not enough tokens to judge, reject conservatively
            first_defer_reject = j
            break
        
        # Count mismatches in window [j+1, j+W] (Equation 8)
        # N_W(j) = Σ_{i=j+1}^{j+W} (1 - ∆_i)
        window_start = j + 1
        window_end = min(j + window_size + 1, K)
        match_in_window = is_match[window_start:window_end]
        n_mismatch_in_window = (~match_in_window).sum().item()

        # DeferDecide (Equation 9)
        # Accept if: h_j ≥ θ AND N_W(j) = 0 AND j + W ≤ K
        # Reject otherwise
        if n_mismatch_in_window > 0:
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
    entropy: torch.Tensor,
    eos_token_id: int,
    threshold: float = 0.3,
    window_size: int = 6,
    max_tolerance_seq_length: int = 1,
) -> int:
    """
    FLy verification variant: Accept semantically correct token sequences using entropy gate and deferred window.
    
    This variant treats a sequence of tokens [j, j+k-1] as a single unit for verification.
    When a mismatch occurs at position j, it tries to accept a sequence of length k (1 to max_tolerance_seq_length)
    by checking the entire sequence as a whole.
    
    Args:
        draft_ids: Draft token sequence [K] (excluding root token)
        target_ids: Target token sequence [K] (excluding bonus token)
        entropy: Normalized entropy for target tokens [K]
        eos_token_id: End-of-sequence token ID
        threshold: Entropy threshold for gate (default: 0.3)
        window_size: Deferred window size W (default: 6)
        max_tolerance_seq_length: Maximum length of sequence that can be deferred and accepted (default: 1)

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
    assert max_tolerance_seq_length >= 1, f"max_tolerance_sequence_length must be >= 1, got {max_tolerance_seq_length}"
    
    # Compute match indicators: ∆_i = 1[draft_ids[i] == target_ids[i]]
    is_match = (draft_ids == target_ids) & (target_ids != eos_token_id)
    
    # Find mismatch positions: J = {j | ∆_j = 0}
    mismatch_positions = torch.where(~is_match)[0].tolist()
    
    # If no mismatches, accept all K draft tokens (bonus token handled separately)
    if not mismatch_positions:
        return K
    
    # Check first mismatch for sequence acceptance
    first_mismatch_position = mismatch_positions[0]
    
    # Step 1: Entropy-level gate
    if entropy[first_mismatch_position] < threshold:
        # Gate = Strict: reject all tokens from position j
        return first_mismatch_position
    
    # Step 2: Gate = Defer: try different sequence lengths
    # Try from largest to smallest to find the maximum acceptable sequence
    seq_accepted = False
    accept_seq_length = 0
    
    for k in range(max_tolerance_seq_length, 0, -1):
        # Check boundary: sequence [j, j+k-1] must be within [0, K)
        if first_mismatch_position + k > K:
            continue  # Sequence extends beyond available tokens, try smaller k
        
        # a. Entropy check: for sequence [j, j+k-1]
        #    - Match positions: directly accept (no entropy check needed)
        #    - Mismatch positions: must have entropy >= θ
        match_in_seq = is_match[first_mismatch_position:first_mismatch_position+k]
        seq_entropy = entropy[first_mismatch_position:first_mismatch_position+k]
        
        # Check entropy only for mismatch positions
        mismatch_positions_in_seq = torch.where(~match_in_seq)[0]
        if mismatch_positions_in_seq.numel() > 0:
            # There are mismatches in the sequence, check their entropy
            if not (seq_entropy[mismatch_positions_in_seq] >= threshold).all().item():
                continue  # Not all mismatch positions pass entropy check, try smaller k
        
        # b. Window check: check window [j+k, j+k+window_size-1] for mismatches
        window_start = first_mismatch_position + k
        window_end = min(first_mismatch_position + k + window_size, K)
        
        if window_start >= K:
            # Boundary case: sequence extends to the end, cannot check window
            # Conservative: only accept if k == 1 (single token)
            if k == 1:
                seq_accepted = True
                accept_seq_length = k
                break
            else:
                continue  # Try smaller k
        
        # Check window for mismatches
        match_in_window = is_match[window_start:window_end]
        n_mismatch_in_window = (~match_in_window).sum().item()
        
        # c. If window has no mismatches, accept the entire sequence
        if n_mismatch_in_window == 0:
            seq_accepted = True
            accept_seq_length = k
            break
        
    # logging.info(f"Inital accept_seq_length: {accept_seq_length}, seq_accepted: {seq_accepted}")
    
    if seq_accepted:
        # Accept sequence [j, j+accept_seq_length-1]
        accept_len = first_mismatch_position + accept_seq_length
        
        # Continue checking from position j+accept_seq_length with single-token FLy logic
        # Find next mismatch after the accepted sequence
        next_pos = first_mismatch_position + accept_seq_length
        while next_pos < K:
            if is_match[next_pos].item():
                # Match: accept and continue
                accept_len = next_pos + 1
                next_pos += 1
            else:
                # Mismatch: check with single-token FLy logic
                if entropy[next_pos] < threshold:
                    # Strict reject
                    break
                
                # Defer: check window
                if next_pos + window_size > K:
                    # Boundary case: not enough tokens to judge
                    break
                
                window_start = next_pos + 1
                window_end = min(next_pos + window_size + 1, K)
                match_in_window = is_match[window_start:window_end]
                n_mismatch_in_window = (~match_in_window).sum().item()
                
                if n_mismatch_in_window > 0:
                    # Reject
                    break
                
                # Accept single token
                accept_len = next_pos + 1
                next_pos += 1
                
        # logging.info(f"Final accept_seq_length: {accept_seq_length}, seq_accepted: {seq_accepted}")
        
        return min(accept_len, K)
    else:
        # No sequence passed, reject at position j
        # logging.info(f"Final accept_seq_length: {accept_seq_length}, seq_accepted: {seq_accepted}")
        return first_mismatch_position

def custom_verify(
    *,
    draft_ids: torch.Tensor,
    target_ids: torch.Tensor,
    entropy: torch.Tensor,
    eos_token_id: int,
    threshold: float = 0.3,
    window_size: int = 6,
    tolerance_seq_length: int = 1,
):
    K = draft_ids.size(0)
    assert target_ids.size(0) == K, f"draft_ids and target_ids must have same length K, got {K} and {target_ids.size(0)}"
    
    # Compute match indicators: ∆_i = 1[draft_ids[i] == target_ids[i]]
    is_match = (draft_ids == target_ids) & (target_ids != eos_token_id)
    
    # Find mismatch positions: J = {j | ∆_j = 0}
    mismatch_positions = torch.where(~is_match)[0].tolist()
    # If no mismatches, accept all K draft tokens (bonus token handled separately)
    if not mismatch_positions:
        return K
    
    target_confidence = entropy[:] < threshold
    
    for j in mismatch_positions:
        # target model is confident on its prediction
        if target_confidence[j]:
            return j
        
        # no reference window for tolerance sequence
        if j + tolerance_seq_length + window_size >= K:
            return j
        
        if not target_confidence[j]:
            # edit sequence: j:j+tolerance_seq_length
            tolerance_seq_reject = ~is_match[j:j+tolerance_seq_length] & target_confidence[j:j+tolerance_seq_length]
            if tolerance_seq_reject.count_nonzero() > 1:
                return j
            
            # check window after tolerance sequence
            window_mismatch = ~is_match[j+tolerance_seq_length: j+tolerance_seq_length+window_size]
            if window_mismatch.count_nonzero():
                return j
        
    # accept all draft tokens
    return K