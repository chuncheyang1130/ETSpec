"""
Generator for ExpSpec — top-N expert subset speculative decoding.

The draft runs only `top_n` experts per MoE layer. 
The kept subset is **dynamic**, updated at following timings:
    - After prefill, before the first SD round. 
    - After every verification round
This makes it follow the current generation's hot experts instead of being fixed at build time. 
Each token chooses exactly top-k experts directly from the retained pool.

Build time:
    - Draft's MoE blocks are swapped for shared-weight stacked draft blocks (`Qwen3MoeStackedBlock`, `owns_store=False`) that alias the target's stacked weights.

Generate time:
    - Install the mass-weighted tracker on the target's MoE blocks (top-k softmax weights)
    - After prefill, and again after every verification round, pick the top-N highest-mass experts per layer
    - Refresh each draft block's selected experts from the target's matching experts.
"""

from typing import Any, Dict

import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from specdecodes.models.utils.moe.base.expert_usage_tracker import (
    get_expert_usage,
    install_expert_usage_tracker,
    pick_top_n_per_layer,
    reset_expert_usage,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from specdecodes.models.utils.moe.expert_usage_logger import ExpertUsageLogger

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


_TRACKER_INSTALLED = "_moe_topn_tracker_installed"


class ExpSpecSDGeneratorBase(ClassicSDGeneratorBase):
    _block_cls = (Qwen3MoeStackedBlock,)

    # Attribute on the draft model where the recipe stashes its config dict.
    _config_attr: str = "topn_subset_config"

    def _config(self) -> Dict[str, Any]:
        """The recipe-supplied config dict on the draft model, or `{}`."""
        return getattr(self.draft_model, self._config_attr, None) or {}

    def _materialize_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs forwarded to `materialize_kept_from_target`.

        Base (full-rank) needs none. SVD subclass overrides to inject
        `svd_device` for the SVD compute device.
        """
        return {}

    def _ensure_tracker_installed(self) -> None:
        if getattr(self, _TRACKER_INSTALLED, False):
            return
        install_expert_usage_tracker(self.target_model)
        setattr(self, _TRACKER_INSTALLED, True)

    def bind_draft_weights(self) -> None:
        """Bind shared-weight draft blocks to the target once, before generation.

        Called by the builder after the target is stacked and before the
        draft is compiled (so the compiled graph captures the bound weight
        pointers). Idempotent and outside the per-prompt `_generate` path.

        No-op for drafts that don't expose `bind_target_weights`.
        """
        fn = getattr(self.draft_model, "bind_target_weights", None)
        if callable(fn):
            fn(self.target_model)

    def init_cuda_graph_runner(self, device, kvCachePool=None):
        """Forward CUDA-graph capture to the draft, if it supports one.

        Called by `run.pipelines.utils.eval_utils.maybe_init_cuda_graph_runner`
        after warmup. No-op for drafts that don't expose the hook (e.g. the
        plain `ExpSpecSDDraftModel`).
        """
        fn = getattr(self.draft_model, "init_cuda_graph_runner", None)
        if callable(fn):
            fn(device=device)

    def _topn(self) -> int:
        return int(self._config().get("top_n", 32))

    def _usage_logger(self) -> ExpertUsageLogger:
        """Lazily build the per-round expert-usage logger (target/draft set)."""
        logger = getattr(self, "_usage_logger_obj", None)
        if logger is None:
            logger = ExpertUsageLogger(
                self.target_model, self.draft_model, self._block_cls, self._config()
            )
            self._usage_logger_obj = logger
        return logger

    def _defer_kept_update(self) -> bool:
        """If True, skip the in-`_tree_decoding` kept-set pick.

        Default False (the pick runs right after the target forward, before
        verification — using all processed tree tokens). A subclass that wants
        accept-aware tracking returns True here and instead picks *after*
        `_verify`, once the accepted positions are known.
        """
        return False

    def _pick_and_update_kept(self) -> None:
        """Pick top-N per layer from cumulative tracker counts, then refresh
        the draft's packed tensors + routing buffers from the target's
        experts at those ids.

        Per-block calls are cached on the kept set (`_last_filled_ids`), so
        layers whose kept set didn't change pay nothing here.
        """
        counts = get_expert_usage(self.target_model)
        if not counts:
            return
        kept = pick_top_n_per_layer(counts, top_n=self._topn())
        if not kept:
            return

        self.draft_model.materialize_kept_from_target(
            self.target_model, kept, **self._materialize_kwargs(),
        )

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        # Counts are *not* reset here: we let them accumulate across the
        # whole generation so picks are drawn from a dense, stable
        # distribution rather than a single tree's worth of tokens.
        logger = self._usage_logger()
        if logger.enabled:
            # Snapshot pre-round state so we can compute the round-local
            # delta after the target forward completes.
            logger.snapshot_pre_round()

        outputs = super()._tree_decoding(
            tree=tree,
            past_key_values=past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            device=device,
        )

        if logger.enabled:
            with nvtx.annotate("expert_usage_log", color="yellow"):
                logger.record_round_delta()

        if not self._defer_kept_update():
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
            # Reset cumulative counts at the start of every prompt; from this
            # point on they accumulate across prefill + every SD round.
            reset_expert_usage(self.target_model)

        logger = self._usage_logger()
        if logger.enabled:
            logger.reset()

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

        # Init / Refresh selected expert set
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

                if logger.enabled:
                    logger.set_last_round_accept_len(int(accept_len))

                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]

                with nvtx.annotate("stop_check"):
                    finished, input_ids, kept, prune_tokens = self._apply_tokenwise_stopping_criteria(
                        input_ids=input_ids,
                        sampled_tokens=sampled_tokens,
                        stopping_criteria=stopping_criteria,
                    )
                    
                    # if len(input_ids[0]) >= 128:
                    #     finished = True
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

        if logger.enabled:
            logger.dump()

        return input_ids


class ExpSpecSDGenerator(SDProfilingMixin, ExpSpecSDGeneratorBase):
    pass
