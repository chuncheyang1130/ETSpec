"""Per-round expert-usage logging for the ExpSpec top-N generator.

Snapshots the target's cumulative routing-mass tracker before each
verification round, then *after* the round computes the round-local delta
(= which experts the target activated for this round's tree), compares it
against the kept set the draft was actually running, and reports
coverage / churn / top experts.

One console line fires per round; a richer per-round record is buffered and
flushed to a JSONL file (one record per `_generate` call) for offline
analysis. Everything is gated by `log_expert_usage` in the recipe config —
when disabled, `enabled` is False and the generator skips every call here.

Lifecycle (driven by the generator):
    reset()                  # start of each `_generate`
    snapshot_pre_round()     # before each target tree-decode
    record_round_delta()     # after it — logs one console line, buffers record
    set_last_round_accept_len(n)   # once verify knows the accept length
    dump()                   # end of `_generate` — flush JSONL + summary
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

from .base.expert_usage_tracker import get_expert_usage


# Cap on how many round-local top experts to record per layer in the JSONL
# dump. Keeps file size bounded for long generations on wide MoEs.
_MAX_TOP_LOG = 16


class ExpertUsageLogger:
    """Coverage / churn logger for top-N expert-subset speculative decoding.

    Construct once per generator (lazily, after `target_model` / `draft_model`
    are set). All state lives here rather than on the generator, so the SD
    pipeline in `expspec_sd.py` stays a clean target/draft loop.
    """

    def __init__(
        self,
        target_model: torch.nn.Module,
        draft_model: torch.nn.Module,
        block_cls: tuple,
        config: Optional[Dict[str, Any]],
    ):
        self._target_model = target_model
        self._draft_model = draft_model
        self._block_cls = block_cls
        self._config = config or {}
        self.reset()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("log_expert_usage", False))

    def _log_path(self) -> Optional[str]:
        path = self._config.get("expert_usage_log_path")
        return str(path) if path else None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear buffered records + per-round snapshots (call per `_generate`)."""
        self._log: List[Dict[str, Any]] = []
        self._pre_cum: Dict[str, torch.Tensor] = {}
        self._pre_kept: Dict[str, List[int]] = {}
        self._prev_picks: Dict[str, List[int]] = {}
        self._round_idx = 0

    def _snapshot_kept_ids(self) -> Dict[str, List[int]]:
        # Iterate the *inner* transformer (`draft_model.model`) so the module
        # paths match the keys `get_expert_usage` produces against the target —
        # otherwise the draft wrapper prepends a stray `model.`, the lookup
        # misses, and coverage silently collapses to 0 across the board.
        out: Dict[str, List[int]] = {}
        for name, mod in self._draft_model.model.named_modules():
            if isinstance(mod, self._block_cls):
                # Packed/FP8/INT4 blocks expose `kept_expert_ids`; the
                # shared-weight block renames it `selected_expert_ids`. Accept
                # whichever the block carries.
                ids = getattr(mod, "selected_expert_ids", None)
                if ids is None:
                    ids = mod.kept_expert_ids
                out[name] = ids.detach().cpu().tolist()
        return out

    def snapshot_pre_round(self) -> None:
        """Snapshot pre-round cumulative counts + the kept set in use, so the
        post-round delta can be attributed to the experts the draft ran."""
        self._pre_cum = get_expert_usage(self._target_model)
        self._pre_kept = self._snapshot_kept_ids()

    # ------------------------------------------------------------------
    # Per-round delta
    # ------------------------------------------------------------------

    def record_round_delta(self) -> None:
        """Compute the per-layer cumulative delta from the pre-round snapshot.

        Logs a one-line console summary and appends a per-round record to the
        in-memory buffer (flushed by `dump`).
        """
        curr = get_expert_usage(self._target_model)
        if not curr:
            return

        kept_during = self._pre_kept  # in use during this round's tree

        # Sanity: target and draft must agree on module paths, otherwise the
        # kept-id lookup misses every layer and coverage silently reports 0.
        if self._round_idx == 0 and kept_during:
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
        round_idx = self._round_idx

        for name, c_curr in curr.items():
            prev = self._pre_cum.get(name)
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
            prev_picks = self._prev_picks.get(name)
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

        self._log.append({
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
        # made by the generator's `_pick_and_update_kept` right after this.
        self._prev_picks = kept_during
        self._round_idx += 1

    def set_last_round_accept_len(self, accept_len: int) -> None:
        if self._log:
            self._log[-1]["accept_len"] = int(accept_len)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def dump(self) -> None:
        """Append the buffered rounds to the JSONL path + print an aggregate."""
        path = self._log_path()
        if not path or not self._log:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": self._config,
            "n_rounds": len(self._log),
            "rounds": self._log,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # End-of-generation aggregate summary.
        covs = [r["mean_cov"] for r in self._log]
        accs = [r["accept_len"] for r in self._log if r["accept_len"] is not None]
        if covs:
            print(
                f"[expert-usage summary] rounds={len(self._log)} "
                f"mean_cov_over_rounds={sum(covs)/len(covs):.3f} "
                f"mean_accept_len={(sum(accs)/len(accs)):.2f} "
                f"-> appended to {path}"
            )
