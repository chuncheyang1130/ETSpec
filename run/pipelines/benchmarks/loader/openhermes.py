"""
OpenHermes-2.5 calibration loader (category-stratified).

`teknium/OpenHermes-2.5` is a ~1M-example synthetic instruction/chat corpus
aggregated from many sources, tagged with ~28 task `category` values (coding,
roleplay, orca, riddle, summarization, ...). It spans general chat, math, and
code, which makes it a good *general* calibration set for routing-mass expert
selection: the kept set isn't biased toward a single task's experts.

This loader stratifies by `category`: it collects up to `per_category` prompts
from *every* category it sees, so each task type contributes routing mass and no
category-specific expert is starved (then pruned by `select_kept_by_coverage`).
It returns the raw first human turn of each conversation (no chat template — the
caller wraps it), so it slots straight into `expert_calibration.run_calibration`,
which applies the chat template and generates, accumulating decode-phase mass.

Size note: the corpus is ~1M rows, so this *streams* (no multi-GB download). It
stops as soon as the stream goes quiet — no new category and no bucket filled for
`patience` rows (default 2k) — or `scan_cap` (default 10k) is hit, whichever comes
first; in practice it scans only a few thousand rows. Rare categories that haven't
filled by then just under-fill (logged via the per-category counts).

Caveat carried from the calibration discussion: equal `per_category` counts are
NOT equal token mass — short-answer categories hit EOS early during generation
and contribute fewer decode tokens, so long-output categories (coding /
detailed_writing) dominate the accumulated mass. That's benign for a
general-coverage goal; balance by tokens if you need strict per-category parity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from random import Random
from typing import Any, Dict, List, Optional
from tqdm.auto import tqdm


def _save_prompt_cache(path: str, prompts: List[Dict[str, str]], use_cache: bool) -> None:
    """Persist the (small) selected prompt set so later runs skip the 1M-row stream."""
    if not use_cache:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f)
    logging.info("[openhermes] cached %d prompts -> %s", len(prompts), path)


def _first_human_turn(convs: Any) -> Optional[str]:
    """First human/user turn's text from a ShareGPT-style `conversations` list."""
    if not isinstance(convs, list):
        return None
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", turn.get("role", ""))).lower()
        if role in ("human", "user"):
            val = (turn.get("value", turn.get("content", "")) or "").strip()
            return val or None
    return None


def load_openhermes_dataset(
    query_version: str = "qwen",   # accepted for loader-interface parity; prompts used as-is
    mode: str = "stratified",      # "stratified" (per-category) | "random" (shuffle + pick n)
    per_category: int = 40,
    n: int = 1000,
    seed: int = 0,
    scan_cap: int = 10_000,
    patience: int = 2_000,
    shuffle_buffer: int = 10_000,
    cache_dir: str = "./expert_activation/calib_cache",
    use_cache: bool = True,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Stream `teknium/OpenHermes-2.5` -> calibration prompts.

    Returns `{"query", "category", "source"}` dicts; `query` is the raw first human
    turn (no chat template). Two modes:
      * "stratified" — up to `per_category` prompts per `category`, so every task
        type contributes (flattens the natural mix). Result shuffled by `seed`.
      * "random" — buffered-shuffle the stream (`shuffle_buffer`, `seed`) and take
        the first `n` prompts at the corpus's *natural* category frequency. Logs the
        realized category mix so you can confirm math/code aren't under-represented.
    """
    # Cache the (small) selected prompt set keyed by the params that determine it, so the
    # 1M-row corpus is streamed only ONCE — later runs load instantly (no network, no scan).
    cat_tag = "all" if not categories else "c" + hashlib.md5(
        ",".join(sorted(categories)).encode()).hexdigest()[:8]
    # Key only on the params that actually determine the selection: random uses `n`,
    # stratified uses `per_category` (n is irrelevant there) — so varying the unused one
    # doesn't cause a spurious re-stream.
    size_tag = f"n{n}" if mode == "random" else f"pc{per_category}"
    cache_path = os.path.join(
        cache_dir, f"openhermes_{mode}_{size_tag}_seed{seed}_{cat_tag}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            prompts = json.load(f)
        logging.info("[openhermes] loaded %d cached prompts <- %s (delete to re-stream)",
                     len(prompts), cache_path)
        return prompts

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "OpenHermes-2.5 calibration needs the 'datasets' library "
            "(pip install datasets)."
        ) from e

    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    if mode == "random":
        ds = ds.shuffle(seed=seed, buffer_size=int(shuffle_buffer))
        prompts: List[Dict[str, str]] = []
        cats: Counter = Counter()
        for row in tqdm(ds, desc="Loading OpenHermes (random)"):
            q = _first_human_turn(row.get("conversations"))
            if not q:
                continue
            cat = (str(row.get("category") or "").strip()) or "uncategorized"
            prompts.append({"query": q, "category": cat, "source": str(row.get("source", ""))})
            cats[cat] += 1
            if len(prompts) >= int(n):
                break
        logging.info("[openhermes] random sample: n=%d categories=%d", len(prompts), len(cats))
        logging.info("[openhermes] category mix: %s", dict(cats.most_common()))
        _save_prompt_cache(cache_path, prompts, use_cache)
        return prompts

    if mode != "stratified":
        raise ValueError(f"unknown mode {mode!r} (stratified|random)")

    # `categories` allow-list: collect ONLY these (pre-seed buckets so a never-seen name
    # is flagged as under-filled). `None` keeps the discover-all behaviour.
    allowed = set(categories) if categories else None
    buckets: Dict[str, List[Dict[str, str]]] = {c: [] for c in (categories or ())}
    scanned = 0
    last_progress = 0
    for row in tqdm(ds, desc="Loading OpenHermes Dataset"):
        scanned += 1
        if scanned > scan_cap:
            break
        q = _first_human_turn(row.get("conversations"))
        if not q:
            continue
        cat = (str(row.get("category") or "").strip()) or "uncategorized"
        if allowed is not None and cat not in allowed:
            continue                                 # skip non-target categories
        if cat not in buckets:                       # newly discovered category
            buckets[cat] = []
            last_progress = scanned
        if len(buckets[cat]) < per_category:         # filled a not-yet-full bucket
            buckets[cat].append({"query": q, "category": cat, "source": str(row.get("source", ""))})
            last_progress = scanned
        if allowed is not None:
            # Stop as soon as every target category is full (don't chase rare ones).
            if all(len(buckets[c]) >= per_category for c in allowed):
                break
        elif scanned - last_progress > patience:
            # Discover-all mode: stop once the stream goes quiet (no new category and no
            # bucket filled for `patience` rows). `scan_cap` is the hard ceiling.
            break

    prompts = [p for b in buckets.values() for p in b]
    counts = {c: len(b) for c, b in sorted(buckets.items())}
    under = {c: k for c, k in counts.items() if k < per_category}
    logging.info("[openhermes] scanned=%d categories=%d collected=%d per_category=%d",
                 scanned, len(buckets), len(prompts), per_category)
    logging.info("[openhermes] per-category counts: %s", counts)
    if under:
        logging.warning("[openhermes] under-filled categories (<%d): %s "
                        "(raise scan_cap, or the category name may differ in the corpus)",
                        per_category, under)

    Random(seed).shuffle(prompts)
    _save_prompt_cache(cache_path, prompts, use_cache)
    return prompts
