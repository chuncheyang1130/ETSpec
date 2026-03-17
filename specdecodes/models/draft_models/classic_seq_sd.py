import torch
import torch.nn as nn
import nvtx
import logging

from specdecodes.models.utils.wandb_logger import wandb_logger
from specdecodes.models.utils.utils import get_normalized_entropy

from .base import DraftModelBase

    
class ClassicSDDraftModel(DraftModelBase):
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        input_ids, kwargs = self._align_forward_inputs_to_model_device(input_ids, kwargs)
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        do_sample = kwargs.get("do_sample", False)
        logits_processor = kwargs.get("logits_processor", None)
        is_lossy = kwargs.get("is_lossy", False)
        
        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("kv_init"):
            kv_len = self._get_kv_len_int()
            
        # 3) First forward pass
        with nvtx.annotate("draft_prefill", color="red"):
            cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=device)
            logits = self.prefill_forward(
                input_ids[:, kv_len:],
                # with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                logits_to_keep=1,
            )
            
            kv_len = input_len
            self.past_key_values.seq_len = input_len
            
        with nvtx.annotate("draft_sample", color="green"):
            if not is_lossy:
                sampled_probs = torch.softmax(logits / self.draft_params.temperature, dim=-1)
            else:
                sampled_probs = self._sample_probs(logits=logits, logits_warper=logits_processor, do_sample=do_sample)
            
            if not do_sample:
                sampled_token = torch.argmax(sampled_probs[:, -1:], dim=-1)
            else:
                sampled_token = torch.multinomial(sampled_probs[:, -1, :], num_samples=1)
                logging.debug(f"sampled_token shape: {sampled_token.shape}")
        
        # 4) Initialize sequential draft state (token buffer + cache position).
        self.token_ids = []
        self.token_ids.append(input_ids[:, -1:])
        
        # Draft Model is confident enough
        if get_normalized_entropy(logits) >= self.draft_params.draft_threshold:
            return torch.cat(self.token_ids, dim=-1)
        
        self.token_ids.append(sampled_token)
        self.cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)
        
        # 5) Main loop
        for depth_i in range(self.draft_params.max_depth-1):
            is_valid = self.speculate_once(do_sample=do_sample, logits_processor=logits_processor, is_lossy=is_lossy)
            if not is_valid:
                break
        
        return torch.cat(self.token_ids, dim=-1)
    
    @torch.no_grad()
    def speculate_once(self, **kwargs):
        token_ids = self.token_ids
        cache_position = self.cache_position
        
        # logits processor and do_sample flag be passed from kwargs 
        # lossy SD might need draft model to perform sampling 
        # this setting is mainly for llm  from same family (e.g. Qwen3 Family)
        do_sample = kwargs.get("do_sample", False)
        logits_processor = kwargs.get("logits_processor", None)
        is_lossy = kwargs.get("is_lossy", False)

        with nvtx.annotate("draft_forward", color="red"):
            logits = self(
                token_ids[-1], 
                # with_softmax=True if not do_sample else False,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                past_key_values=self.past_key_values.cache,
            )
            
            if is_lossy and get_normalized_entropy(logits) >= self.draft_params.draft_threshold:
                return False
            
        with nvtx.annotate("draft_sample", color="green"):
            if not is_lossy:
                sampled_probs = torch.softmax(logits / self.draft_params.temperature, dim=-1)
            else:
                sampled_probs = self._sample_probs(logits=logits, logits_warper=logits_processor, do_sample=do_sample)
                
            if not do_sample:
                sampled_token = torch.argmax(sampled_probs[:, -1, :], dim=-1, keepdim=True)
            else:
                sampled_token = torch.multinomial(sampled_probs[:, -1, :], num_samples=1)
                
            token_ids.append(sampled_token)
            
        if wandb_logger.get_flag("detailed_analysis", False):
            self.draft_prob.append(torch.max(sampled_probs[:, -1, :]).cpu().item())
            
        # Update internal state
        self.token_ids = token_ids
        self.cache_position += 1
        
        return True