"""
NaiveMoEGenerator — target-only generator for per-prompt prefill-subset MoE models.

Isolates the expert-selection orchestration that the dense `NaiveGenerator` must not
carry. It runs its OWN `_generate` so selection is an EXPLICIT step at the
prefill->decode boundary — no forward hooks, no per-token branching:

  1. reset every prefill-subset block -> full + observe          (before prefill)
  2. prefill (eager) accumulates each prompt's per-expert routing mass
  3. select each block's working set from that mass              (after prefill)
  4. decode loop runs the (compiled) forward over the installed subset

The compiled decode `forward` is never touched, so it stays fully torch.compile-able.
For a model without prefill-subset blocks, steps 1 and 3 are no-ops.

`_generate` mirrors `NaiveGeneratorBase._generate` with the two extra steps (1) and
(3); keep them in sync if the base loop changes.
"""

from __future__ import annotations

import nvtx
import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria

from .naive import NaiveGeneratorBase
from ..utils.mixin import ProfilingMixin


def _is_prefill_subset(module) -> bool:
    return (hasattr(module, "reset_prompt") and hasattr(module, "select_from_prefill")
            and hasattr(module, "_observe"))


class NaiveMoEGeneratorBase(NaiveGeneratorBase):
    """NaiveGenerator with explicit per-prompt prefill-subset expert selection."""

    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(generator_kwargs, *model_args, **kwargs)
        self._ps_blocks = [m for m in self.target_model.modules() if _is_prefill_subset(m)]

    def _reset_prefill_subset(self) -> None:
        for b in self._ps_blocks:
            b.reset_prompt()

    def _select_prefill_subset(self) -> None:
        for b in self._ps_blocks:
            b.select_from_prefill()

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        assert self.target_model is not None, "target_model must be provided"

        input_ids = input_ids.clone()
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )

        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
        else:
            raise ValueError("past_key_values should be provided")

        stream_callback = model_kwargs.get("stream_callback", None)

        kv_len = past_key_values.get_seq_length()
        cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=input_ids.device)

        expert_recorder = self._maybe_start_expert_recorder()

        # (1) Per-prompt reset: full + observe routing before prefill.
        self._reset_prefill_subset()

        # (2) Prefill — accumulates each block's per-prompt routing mass (eager).
        with nvtx.annotate("prefill_chunked", color="orange"):
            outputs = self._chunked_prefill_forward(
                input_ids,
                past_key_values,
                prefill_chunk_size=self.prefill_chunk_size,
                use_position_ids=True,
            )
            next_token_logits = outputs.logits
            del outputs

        # (3) Selection — install each block's working set from the prefill mass,
        # eager and OUTSIDE the compiled decode forward.
        self._select_prefill_subset()

        with nvtx.annotate("sample"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cache_position = cache_position[-1:] + 1
            self._maybe_stream(stream_callback, next_tokens)

        # (4) Decode loop — compiled forward over the installed working set.
        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                with nvtx.annotate("target_forward", color="red"):
                    outputs = self.target_model(
                        next_tokens,
                        past_key_values=past_key_values.cache,
                        position_ids=cache_position.unsqueeze(0),
                        cache_position=cache_position,
                    )
                    next_token_logits = outputs.logits

                with nvtx.annotate("sample"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    cache_position += 1
                    past_key_values.seq_len += 1
                    self._maybe_stream(stream_callback, next_tokens)

                with nvtx.annotate("stop_check"):
                    finished = stopping_criteria(input_ids, None)

        self._maybe_finish_expert_recorder(expert_recorder)

        return input_ids


class NaiveMoEGenerator(ProfilingMixin, NaiveMoEGeneratorBase):
    pass
