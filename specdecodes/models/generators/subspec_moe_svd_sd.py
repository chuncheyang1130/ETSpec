"""Generator for MoE TopN-SVD speculative decoding.

Build time: the draft's MoE blocks are swapped for `PackedTopNSvdMoeBlock`
instances (random init).

Generate time:
  1. Install a tracker on the target's MoE blocks that accumulates per-expert
     hit counts across **all** target forwards (prefill + every round).
     Counts are *not* reset between rounds — picks are denser and more
     stable when drawn from cumulative usage than from a single tree.
  2. After prefill, and again after every verification round, pick the top-N
     most-activated experts per layer, then SVD-fill each draft block's
     packed tensors and refresh its router buffers from the target. The
     per-block fill is cached on the kept set, so layers whose pick didn't
     change pay nothing.

Optional per-round expert-usage logging (gated by `log_expert_usage` in
`topn_svd_config`): each verification round, snapshot the cumulative tracker
delta (= what experts the target activated *for this round's tree*), compare
against the kept set the draft was using, and report coverage / churn / top
experts. A single-line console summary fires per round; a richer per-round
record is appended to `expert_usage_log_path` (JSONL, one record per
`_generate` call) for offline analysis.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from specdecodes.models.utils.moe.qwen3_moe_topn_svd import (
    PackedTopNSvdMoeBlock,
    get_expert_usage,
    install_expert_usage_tracker,
    pick_top_n_per_layer,
    reset_expert_usage,
)

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


_TRACKER_INSTALLED = "_moe_svd_tracker_installed"
# Cap on how many round-local top experts to record per layer in the JSONL
# dump. Keeps file size bounded for long generations on wide MoEs.
_MAX_TOP_LOG = 16


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
        """Pick top-N per layer from cumulative tracker counts, then SVD-fill
        the packed tensors and refresh routing buffers from the target's
        experts at those ids.

        Per-block SVD is cached on the kept set (`_last_filled_ids`), so
        layers whose kept set didn't change pay nothing here.
        """
        counts = get_expert_usage(self.target_model)
        if not counts:
            return
        kept = pick_top_n_per_layer(counts, top_n=self._topn())
        if not kept:
            return

        cfg = getattr(self.draft_model, "topn_svd_config", None) or {}
        svd_device = str(cfg.get("svd_device", "cuda:0"))
        self.draft_model.materialize_kept_from_target(
            self.target_model, kept, svd_device=svd_device,
        )

    # ------------------------------------------------------------------
    # Per-verification expert-usage logging
    # ------------------------------------------------------------------

    def _expert_log_enabled(self) -> bool:
        cfg = getattr(self.draft_model, "topn_svd_config", None) or {}
        return bool(cfg.get("log_expert_usage", False))

    def _expert_log_path(self) -> Optional[str]:
        cfg = getattr(self.draft_model, "topn_svd_config", None) or {}
        path = cfg.get("expert_usage_log_path")
        return str(path) if path else None

    def _snapshot_kept_ids(self) -> Dict[str, List[int]]:
        # Iterate the *inner* transformer (`self.draft_model.model`) so the
        # module paths match the keys produced by `get_expert_usage` against
        # `self.target_model` — otherwise every name gets a stray `model.`
        # prefix from the draft wrapper, the lookup misses, and coverage
        # collapses to 0 across the board.
        out: Dict[str, List[int]] = {}
        for name, mod in self.draft_model.model.named_modules():
            if isinstance(mod, PackedTopNSvdMoeBlock):
                out[name] = mod.kept_expert_ids.detach().cpu().tolist()
        return out

    def _reset_usage_log(self) -> None:
        self._usage_log: List[Dict[str, Any]] = []
        self._usage_pre_cum: Dict[str, torch.Tensor] = {}
        self._usage_pre_kept: Dict[str, List[int]] = {}
        self._usage_prev_picks: Dict[str, List[int]] = {}
        self._usage_round_idx = 0

    def _record_round_delta(self) -> None:
        """Compute the per-layer cumulative delta from the snapshot taken
        before the just-finished target forward. Logs a one-line console
        summary and appends a per-round record to `self._usage_log`.
        """
        curr = get_expert_usage(self.target_model)
        if not curr:
            return

        kept_during = self._usage_pre_kept  # in use during this round's tree

        # Sanity: target and draft must agree on module paths, otherwise the
        # kept-id lookup misses every layer and coverage silently reports 0.
        if self._usage_round_idx == 0 and kept_during:
            unmatched = [n for n in curr if n not in kept_during]
            if unmatched:
                print(
                    f"[expert-usage WARNING] {len(unmatched)}/{len(curr)} target "
                    f"MoE layer paths have no matching draft kept-id entry — "
                    f"e.g. target='{unmatched[0]}', "
                    f"draft keys sample={list(kept_during.keys())[:1]}. "
                    f"Coverage numbers will be wrong."
                )

        layers: Dict[str, Dict[str, Any]] = {}
        agg_total = 0
        agg_in_kept = 0
        coverages: List[float] = []
        churns: List[int] = []
        round_idx = self._usage_round_idx

        for name, c_curr in curr.items():
            prev = self._usage_pre_cum.get(name)
            delta = (c_curr - prev) if prev is not None else c_curr
            delta_cpu = delta.detach().cpu().tolist()
            total = int(sum(delta_cpu))
            if total == 0:
                continue

            kept_ids = kept_during.get(name, [])
            kept_set = set(kept_ids)
            in_kept = int(sum(delta_cpu[i] for i in kept_ids if 0 <= i < len(delta_cpu)))
            coverage = in_kept / total

            # round-local top-K experts (id, count), descending by count.
            top_pairs = sorted(
                ((eid, cnt) for eid, cnt in enumerate(delta_cpu) if cnt > 0),
                key=lambda p: -p[1],
            )[:_MAX_TOP_LOG]

            # churn: how many of last round's picks are *not* in this round's
            # currently-active picks. Only meaningful from round 1 onwards.
            prev_picks = self._usage_prev_picks.get(name)
            if prev_picks is None:
                churn: Optional[int] = None
            else:
                churn = len(set(prev_picks) - kept_set)

            layers[name] = {
                "tot": total,
                "in_kept": in_kept,
                "cov": round(coverage, 4),
                "kept": kept_ids,
                "top": top_pairs,
                "churn": churn,
            }
            agg_total += total
            agg_in_kept += in_kept
            coverages.append(coverage)
            if churn is not None:
                churns.append(churn)

        if not layers:
            return

        agg_cov = agg_in_kept / agg_total if agg_total else 0.0
        mean_cov = sum(coverages) / len(coverages)
        min_cov = min(coverages)
        below_50 = sum(1 for c in coverages if c < 0.5)
        max_churn = max(churns) if churns else None
        mean_churn = (sum(churns) / len(churns)) if churns else None

        print(
            f"[expert-usage round {round_idx:>3}] "
            f"agg_cov={agg_cov:.3f} mean_cov={mean_cov:.3f} min_cov={min_cov:.3f} "
            f"layers<50%={below_50}/{len(coverages)} "
            f"max_churn={max_churn if max_churn is not None else '-'} "
            f"mean_churn={f'{mean_churn:.2f}' if mean_churn is not None else '-'}"
        )

        self._usage_log.append({
            "round": round_idx,
            "agg_total": agg_total,
            "agg_cov": round(agg_cov, 4),
            "mean_cov": round(mean_cov, 4),
            "min_cov": round(min_cov, 4),
            "layers_below_50pct": below_50,
            "n_layers": len(coverages),
            "max_churn": max_churn,
            "mean_churn": round(mean_churn, 4) if mean_churn is not None else None,
            "accept_len": None,  # filled in post-verify
            "layers": layers,
        })

        # Stash the kept-set snapshot from *during* this round so the next
        # round can compute churn against it. Picks for the *next* round are
        # made by `_pick_and_update_kept` immediately after this method.
        self._usage_prev_picks = kept_during
        self._usage_round_idx += 1

    def _set_last_round_accept_len(self, accept_len: int) -> None:
        if not self._usage_log:
            return
        self._usage_log[-1]["accept_len"] = int(accept_len)

    def _dump_usage_log(self) -> None:
        path = self._expert_log_path()
        if not path or not self._usage_log:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": getattr(self.draft_model, "topn_svd_config", None),
            "n_rounds": len(self._usage_log),
            "rounds": self._usage_log,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # End-of-generation aggregate summary.
        covs = [r["mean_cov"] for r in self._usage_log]
        accs = [r["accept_len"] for r in self._usage_log if r["accept_len"] is not None]
        if covs:
            print(
                f"[expert-usage summary] rounds={len(self._usage_log)} "
                f"mean_cov_over_rounds={sum(covs)/len(covs):.3f} "
                f"mean_accept_len={(sum(accs)/len(accs)):.2f} "
                f"-> appended to {path}"
            )

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        # Counts are *not* reset here: we let them accumulate across the
        # whole generation so picks are drawn from a dense, stable
        # distribution rather than a single tree's worth of tokens.
        if self._expert_log_enabled():
            # Snapshot pre-round state so we can compute the round-local
            # delta after the target forward completes.
            self._usage_pre_cum = get_expert_usage(self.target_model)
            self._usage_pre_kept = self._snapshot_kept_ids()

        outputs = super()._tree_decoding(
            tree=tree,
            past_key_values=past_key_values,
            position_offset=position_offset,
            cache_position=cache_position,
            device=device,
        )

        if self._expert_log_enabled():
            with nvtx.annotate("expert_usage_log", color="yellow"):
                self._record_round_delta()

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

        if self._expert_log_enabled():
            self._reset_usage_log()

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

                if self._expert_log_enabled():
                    self._set_last_round_accept_len(int(accept_len))

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

        if self._expert_log_enabled():
            self._dump_usage_log()

        return input_ids


class MoESvdSDGenerator(SDProfilingMixin, MoESvdSDGeneratorBase):
    pass
