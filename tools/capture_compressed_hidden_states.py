#!/usr/bin/env python
"""Capture every MoE layer's input with the model ALREADY compressed by a plan.

The planner prices each (expert, action) on hidden states captured from the
full model, so layer l is always scored as if every layer before it were
exact. A deployed plan compresses them all at once, and layer l actually sees
inputs already perturbed upstream -- the teacher-forcing gap that per-layer
audits (``measure_plan_additive_gap.py``) cannot see by construction.

This tool replays the same calibration sequences as the planner's own cache
through the model restructured by ``--plan`` and writes a cache in the exact
format the planner reads (``--hidden-cache-dir ... --reuse-hidden-cache``).
Planning on it with a fresh ``--trace-dir`` re-measures every action, and
recalibrates every SVD factor, on the inputs the compressed model produces.
That makes staged compression possible: compress part of the way, re-capture,
re-plan the rest.

Row order is the replay order of ``--sequence-cache``, identical to the
planner's capture, so category and prompt ids line up with the original.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm.auto import tqdm

from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
    MoERoutedFunctionalSVDRestructurer,
)
from tools.evaluate_joint_svd_rank_grid import _make_generator, _safe_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--plan", required=True, help="plan to compress the model with")
    ap.add_argument("--moments", required=True)
    ap.add_argument("--source-hidden-cache-dir", required=True,
                    help="cache the plan's own SVD factors are calibrated on")
    ap.add_argument("--sequence-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = Path(args.out_dir)
    if (out / "metadata.json").exists():
        raise SystemExit(f"{out} already holds a cache; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    source_meta = json.loads(
        (Path(args.source_hidden_cache_dir) / "metadata.json").read_text()
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    plan = json.loads(Path(args.plan).read_text())
    module_names = list(plan["layers"])
    MoERoutedFunctionalSVDRestructurer.restructure_model(
        model,
        {
            "plan_path": args.plan,
            "moments_path": args.moments,
            "hidden_cache_dir": args.source_hidden_cache_dir,
            "category": "coding",
            "seed": 0,
        },
        torch.bfloat16,
        args.device,
    )

    hidden_size = int(model.config.hidden_size)
    modules = dict(model.named_modules())
    files, rows, paths, handles = {}, {}, {}, []
    for name in module_names:
        path = out / f"{_safe_name(name)}.bf16"
        files[name] = open(path, "wb")
        rows[name] = 0
        paths[name] = str(path)

        def hook(_module, inputs, name=name):
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            flat = hidden.reshape(-1, hidden.shape[-1])
            flat.detach().to("cpu", torch.bfloat16).contiguous().view(
                torch.uint16
            ).numpy().tofile(files[name])
            rows[name] += int(flat.shape[0])

        handles.append(modules[name].register_forward_pre_hook(hook))

    from specdecodes.models.utils.cache_utils import create_kv_cache

    sequences = torch.load(args.sequence_cache, map_location="cpu", weights_only=False)[
        "sequences"
    ]
    generator = _make_generator(model, tokenizer, args.chunk_size)
    replay_tokens = 0
    with torch.no_grad():
        for category, items in sequences.items():
            for sequence in tqdm(items, desc=f"Capturing {category} (compressed)"):
                replay_ids = sequence[:, :-1]
                if replay_ids.numel() == 0:
                    continue
                replay_tokens += int(replay_ids.numel())
                cache = create_kv_cache("dynamic")
                generator._chunked_prefill_forward(
                    replay_ids.to(args.device), cache,
                    prefill_chunk_size=int(args.chunk_size), use_position_ids=True,
                )
                del cache
    for handle in handles:
        handle.remove()
    for handle in files.values():
        handle.close()

    if replay_tokens != int(source_meta["replay_tokens"]):
        raise SystemExit(
            f"replayed {replay_tokens} tokens but the source cache has "
            f"{source_meta['replay_tokens']}; row ids would not line up"
        )
    for name in module_names:
        if rows[name] != replay_tokens:
            raise SystemExit(f"{name} captured {rows[name]} rows for {replay_tokens} tokens")
    metadata = {
        "format": "raw_bfloat16_as_uint16",
        "paths": paths,
        "rows": rows,
        "hidden_sizes": {name: hidden_size for name in module_names},
        "replay_tokens": replay_tokens,
        # Routing provenance of the PLANNER's analysis pool, which is still the
        # full original router: the compression lives in the inputs, not here.
        "routing_pool_mode": source_meta.get("routing_pool_mode", "full"),
        "routing_kept_path": source_meta.get("routing_kept_path"),
        "routing_pool_ids": source_meta["routing_pool_ids"],
        "captured_under_plan": args.plan,
        "captured_under_plan_sha256": hashlib.sha256(
            Path(args.plan).read_bytes()
        ).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logging.info("Saved compressed-context hidden cache -> %s (%d tokens)", out, replay_tokens)


if __name__ == "__main__":
    main()
