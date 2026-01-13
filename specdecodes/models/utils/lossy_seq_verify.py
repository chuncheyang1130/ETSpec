import logging
import torch.nn as nn 
import torch
import transformers

NOP = 0
INSERT = 1
DELETE = 2
SUBSTITUTE = 3
KEEP = 4

def lossy_edit_distance_verify(
    *,
    draft_ids,
    target_ids,
    tokenizer,
    confidence,
    eos_token_id,
    threshold,
    window_size
):
    
    # check if eos token is in target tokens
    is_eos = (draft_ids[:-1] == eos_token_id)
    if is_eos.any():
        eos_index = is_eos.float().argmax()
    else:
        eos_index = len(draft_ids)
        
    M = min(len(draft_ids), eos_index)
    N = M
    # 
    # N = len(target_ids[:-1])    # Note: last token is potential bonus token
    INF = max(M, N) + 1

    # edit distance verify
    # dp: 2-D matrix to store edit counts
    # op: 2-D matrix to store the operations being applied
    edit_count_matrix = torch.zeros((N+1, M+1), dtype=torch.float)
    edit_count_matrix[:, 0] = torch.arange(N+1)
    edit_count_matrix[0, :] = torch.arange(M+1)
    
    # calculate edit count
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            draft_token_id = draft_ids[i-1]
            target_token_id = target_ids[i-1]
            
            # check matchness
            if draft_token_id == target_token_id:
                edit_count_matrix[i][j] = edit_count_matrix[i-1][j-1]
            else:
                edit_count_matrix[i][j] = 1 + min(edit_count_matrix[i, j-1], edit_count_matrix[i-1, j], edit_count_matrix[i-1, j-1])

    # backtrace
    path = []
    
    i, j = N, M
    while i > 0 or j > 0:
        current_val = edit_count_matrix[i][j]
        
        if i > 0 and j > 0:
            cost = 0 if target_ids[i-1] == draft_ids[j-1] else 1
            if edit_count_matrix[i-1][j-1] + cost == current_val:
                op_type = "KEEP" if cost == 0 else "SUB"
                
                # record (operation, target_token_id, draft_token_id)
                path.append((op_type, target_ids[i-1], draft_ids[j-1]))
                
                i -= 1
                j -= 1

                continue
            
        if i > 0 and edit_count_matrix[i-1][j] + 1 == current_val:
            path.append(("DEL", target_ids[i-1], '-'))
            i -= 1
            continue
        
        if j > 0 and edit_count_matrix[i][j-1] + 1 == current_val:
            path.append(("INS", '-', draft_ids[j-1]))
            j -= 1
            continue
                
    path.reverse()
    
    idx = 0
    for op_type, target_token_id, draft_token_id in path:
        if op_type == "KEEP":
            idx += 1
        elif op_type == "SUB":
            if confidence[0, idx].item() < threshold:
                logging.debug(f"SUBSTITUTE token '{tokenizer.decode(target_token_id)}' with '{tokenizer.decode(draft_token_id)}' at position {idx} (confidence: {confidence[0, idx]:.4f})")
                # draft_ids[idx] = target_token_id
                idx += 1
            else:
                logging.debug(f"REJECT SUBSTITUTE token '{tokenizer.decode(target_token_id)}' with '{tokenizer.decode(draft_token_id)}' at position {idx} (confidence: {confidence[0, idx]:.4f})")
                break
        elif op_type == "DEL":
            idx += 1
        elif op_type == "INS":
            if confidence[0, idx].item() < threshold:
                # logging.debug(f"INSERT token '{tokenizer.decode(target_token_id)}' at position {idx} (confidence: {confidence[idx]:.4f})")
                # draft_ids = torch.cat([draft_ids[:idx], draft_ids[idx+1:]], dim=0)
                idx += 1
            else:
                break
    return idx