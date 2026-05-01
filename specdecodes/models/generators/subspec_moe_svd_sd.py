"""Generator for MoE TopN-SVD speculative decoding.

Step 1 (build time): the draft's MoE blocks are swapped for
`PackedTopNSvdMoeBlock` instances with random-initialized packed tensors.

Step 2 (this file): after prefill, and again after every verification
round, we read per-layer expert-hit counts off the target, pick the top-N
most-activated experts per layer, and write those ids onto each draft
block's `kept_expert_ids` bookkeeping buffer. The packed SVD weights are
**not** refilled yet — that is the next step. The block forward keeps
running against whatever weights are currently in the packed tensors.
"""

import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from specdecodes.models.utils.moe.qwen3_moe_topn_svd import (
    get_expert_usage,
    install_expert_usage_tracker,
    pick_top_n_per_layer,
    reset_expert_routing,
    reset_expert_usage,
)

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


_TRACKER_INSTALLED = "_moe_svd_tracker_installed"


class MoESvdSDGeneratorBase(ClassicSDGeneratorBase):
    def _ensure_tracker_installed(self) -> None:
        if getattr(self, _TRACKER_INSTALLED, False):
            return
        install_expert_usage_tracker(self.target_model)
        setattr(self, _TRACKER_INSTALLED, True)

    def _topn(self) -> int:
        cfg = getattr(self.draft_model, "topn_svd_config", None) or {}
        return int(cfg.get("top_n", 32))

    def _pick_and_update_kept(self) -> None:
        """End-to-end: pick top-N per layer from the latest tracker counts,
        write the ids onto each draft block's `kept_expert_ids` buffer
        (step 2), and SVD-fill the packed tensors from the target's
        experts at those ids (step 3).

        Per-block SVD is cached on the kept set, so layers whose kept
        set didn't change pay nothing here.
        """
        counts = get_expert_usage(self.target_model)
        if not counts:
            return
        kept = pick_top_n_per_layer(counts, top_n=self._topn())
        if not kept:
            return

        # step 2: stash kept ids on the bookkeeping buffer
        self.draft_model.update_kept_ids(kept)

        # step 3: SVD-fill from target experts at those ids
        cfg = getattr(self.draft_model, "topn_svd_config", None) or {}
        svd_device = str(cfg.get("svd_device", "cuda:0"))
        self.draft_model.materialize_kept_from_target(
            self.target_model, kept, svd_device=svd_device,
        )

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        # Clear tracker state before the target forward so what we read
        # after only reflects this round.
        reset_expert_usage(self.target_model)
        reset_expert_routing(self.target_model)
        outputs = super()._tree_decoding(
            tree=tree,
            past_key_values=past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            device=device,
        )
        # Step 2: pick top-N per layer from this round's counts and stash
        # them on the draft blocks.
        with nvtx.annotate("topn_pick", color="purple"):
            self._pick_and_update_kept()
        return outputs

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"

        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )

        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)
            self.draft_model.set_past_key_values(past_key_values)
        else:
            raise ValueError("past_key_values is not provided")

        stream_callback = model_kwargs.get("stream_callback", None)

        with nvtx.annotate("expert_tracker_install"):
            self._ensure_tracker_installed()
            reset_expert_usage(self.target_model)
            reset_expert_routing(self.target_model)

        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device
            )
            outputs = self._chunked_prefill_forward(
                input_ids,
                past_key_values,
                prefill_chunk_size=self.prefill_chunk_size,
                use_position_ids=True,
            )
            next_token_logits = outputs.logits
            del outputs

        # Step 2: initial top-N pick from prefill counts. This seeds each
        # draft block's `kept_expert_ids` before the SD loop starts; per-round
        # picks inside `_tree_decoding` will keep refreshing it.
        with nvtx.annotate("topn_pick_initial", color="purple"):
            self._pick_and_update_kept()

        with nvtx.annotate("sample"):
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            cache_position = torch.arange(
                org_input_len,
                org_input_len + self.draft_params.max_verify_tokens,
                dtype=torch.long,
                device=input_ids.device,
            )
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                with nvtx.annotate("speculate", color="cyan"):
                    last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(last_token_id)

                with nvtx.annotate("target_decode", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    if self.cache_implementation == "dynamic":
                        past_key_values.crop(prev_kv_len)
                    outputs = self._tree_decoding(
                        tree,
                        past_key_values,
                        position_offset=input_ids.shape[1] - 1,
                        cache_position=cache_position,
                        device=input_ids.device,
                    )
                    next_token_logits = outputs.logits
                    del outputs

                with nvtx.annotate("verify"):
                    root_ind = 0
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                        tree,
                        root_ind,
                        next_token_logits,
                        logits_processor,
                        do_sample,
                    )
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    del next_token_logits

                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]

                with nvtx.annotate("stop_check"):
                    finished, input_ids, kept, prune_tokens = self._apply_tokenwise_stopping_criteria(
                        input_ids=input_ids,
                        sampled_tokens=sampled_tokens,
                        stopping_criteria=stopping_criteria,
                    )
                if kept.numel() > 0:
                    self._maybe_stream(stream_callback, kept)

                with nvtx.annotate("kv_reorder"):
                    past_key_values.reorder_cache_with_offset(
                        hidden_indices,
                        offset=prev_kv_len,
                        new_chunk_len=self.draft_params.max_verify_tokens,
                        dim=2,
                    )
                    past_key_values.seq_len += hidden_indices.shape[0]
                    if finished:
                        past_key_values.seq_len -= prune_tokens

        return input_ids


class MoESvdSDGenerator(SDProfilingMixin, MoESvdSDGeneratorBase):
    pass
