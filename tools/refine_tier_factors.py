#!/usr/bin/env python
"""Refine a plan's SVD-tier factors against each expert's true nonlinear output.

The deployed factors are fitted one matrix at a time: the shared gate/up input
factor and the two output factors from one truncated SVD, the down factors from
another. What the model consumes is the expert's OUTPUT,
``down(silu(gate x) * up x)``, and the two truncations interact through the
SiLU product. This tool starts from exactly the factors the restructurer would
deploy and runs a few hundred Adam steps per tiered expert on

    L_e = sum_t g_t^2 ||f_e(x_t) - fhat_e(x_t)||^2 / sum_t g_t^2 ||f_e(x_t)||^2

over the tokens the DEPLOYED router sends to that expert (top-k over the
plan's retained experts, renormalized weights ``g_t``). Nothing about the
tier's shape, rank or kernel changes: same five factors, same storage dtype.

Held-out tokens are whole calibration PROMPTS, not scattered positions, and a
refined factor set is kept only if it lowers the held-out loss; otherwise the
deployed SVD factors are written unchanged. Local wins have failed to transfer
before (Gram whitening won 10/11 local metrics and zeroed b60), so this is a
candidate to benchmark, never a result by itself.

Output is a factor artifact the restructurer loads with ``factors_dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from specdecodes.helpers.restructurer.moe_routed_functional_svd import (
    _action_tiers,
    _sampled_rows,
)
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeRoutedFunctionalSVDSubsetBlock,
)
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock

KEYS = (
    "gu_input_factor",
    "gate_output_factor",
    "up_output_factor",
    "down_input_factor",
    "down_output_factor",
)


def _safe_layer_name(name: str) -> str:
    return name.replace(".", "_")


def load_stacked_layer(model: str, layer: int, device: str) -> Qwen3MoeStackedBlock:
    """One MoE layer from the checkpoint shards, without loading the model."""

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    root = Path(snapshot_download(model, allow_patterns=["*.json"]))
    index = json.loads((root / "model.safetensors.index.json").read_text())
    prefix = f"model.layers.{layer}.mlp."
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        if key.startswith(prefix):
            by_shard.setdefault(shard, []).append(key)
    state = {}
    for shard, keys in by_shard.items():
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                state[key[len(prefix):]] = handle.get_tensor(key)
    config = AutoConfig.from_pretrained(model)
    with torch.device("meta"):
        block = Qwen3MoeSparseMoeBlock(config)
    block.load_state_dict(state, assign=True)
    block = block.to(device=device, dtype=torch.bfloat16)
    return Qwen3MoeStackedBlock.from_huggingface(block)


def sequence_holdout_mask(sequence_cache: str, rows: int, holdout: float, seed: int):
    """True for rows whose calibration prompt is held out."""

    sequences = torch.load(sequence_cache, map_location="cpu", weights_only=False)["sequences"]
    lengths = [
        int(sequence.shape[-1]) - 1
        for items in sequences.values()
        for sequence in items
        if int(sequence.shape[-1]) > 1
    ]
    if sum(lengths) != rows:
        raise ValueError(f"sequence cache has {sum(lengths)} tokens, hidden cache {rows}")
    generator = torch.Generator().manual_seed(seed)
    held = torch.rand(len(lengths), generator=generator) < float(holdout)
    return torch.repeat_interleave(held, torch.as_tensor(lengths))


def _build_state(stacked, tier_specs, snapshot, name, calibration_hidden, meta, layer_seed, retained):
    return Qwen3MoeRoutedFunctionalSVDSubsetBlock.build_factor_state(
        stacked,
        tier_specs=tier_specs,
        gate_moments=snapshot["sum_gate_input_metric"][name],
        up_moments=snapshot["sum_up_input_metric"][name],
        down_moments=snapshot["sum_g2_z2"][name],
        calibration_hidden=calibration_hidden,
        router_candidates=int(meta.get("router_candidates", 32)),
        routing_batch_size=2048,
        scale_floor=1e-12,
        randomized=True,
        layer_seed=layer_seed,
        gram_metric=False,
        gram_shrink_mode=str(meta.get("gram_shrink_target", "diagonal")),
        down_compensation=bool(meta.get("down_compensation", False)),
        factor_calibration=str(meta.get("factor_calibration", "shortlist")),
        factor_output_metric=str(meta.get("factor_output_metric") or "none"),
        retained_ids=retained,
    )


def load_layer_hidden(cache_dir: str, name: str) -> torch.Tensor:
    meta = json.loads((Path(cache_dir) / "metadata.json").read_text())
    rows, hidden = int(meta["rows"][name]), int(meta["hidden_sizes"][name])
    raw = torch.from_file(meta["paths"][name], shared=False, size=rows * hidden, dtype=torch.uint16)
    return raw.view(torch.bfloat16).reshape(rows, hidden)


def deployed_routing(x_all, stacked, kept_ids):
    """Top-k over the retained experts with the block's own normalization."""
    logits = F.linear(x_all, stacked.router_weights.index_select(0, kept_ids)).float()
    values, slots = torch.topk(logits, k=int(stacked.top_k), dim=-1)
    weights = F.softmax(values, dim=-1) if stacked.norm_topk_prob else \
        F.softmax(logits, dim=-1).gather(1, slots)
    return kept_ids[slots], weights


def routed_tokens(cache, expert, stacked, device):
    """(x, g^2, target, held-out flag) for one expert under one cache's routing."""
    x_all, chosen, weights, held_all = cache
    token, pos = (chosen == int(expert)).nonzero(as_tuple=True)
    x = x_all.index_select(0, token).float()
    g2 = weights[token, pos].float().square()
    with torch.no_grad():
        y = expert_forward(
            x,
            stacked.gate_proj_stacked[expert].float(),
            stacked.up_proj_stacked[expert].float(),
            stacked.down_proj_stacked[expert].float(),
        )
    return x, g2, y, held_all.index_select(0, token)


def expert_forward(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def tier_forward(x, p):
    latent = F.linear(x, p["gu_input_factor"])
    middle = F.silu(F.linear(latent, p["gate_output_factor"])) * F.linear(
        latent, p["up_output_factor"]
    )
    return F.linear(F.linear(middle, p["down_input_factor"]), p["down_output_factor"])


def weighted_loss(x, w2, target, p):
    """Energy-weighted relative output error, and its numerator/denominator."""

    err = (tier_forward(x, p) - target).square().sum(dim=1)
    num = (w2 * err).sum()
    den = (w2 * target.square().sum(dim=1)).sum().clamp_min(1e-30)
    return num / den


def refine_expert(x_tr, w_tr, y_tr, x_ho, w_ho, y_ho, init, *, steps, lr, batch, eval_every):
    params = {k: init[k].detach().float().clone().requires_grad_(True) for k in KEYS}
    with torch.no_grad():
        base_ho = float(weighted_loss(x_ho, w_ho, y_ho, init_f := {k: init[k].float() for k in KEYS}))
        base_tr = float(weighted_loss(x_tr, w_tr, y_tr, init_f))
    # Per-factor step scaled to the factor's own RMS, so one lr fits every
    # layer: Adam's update is ~lr * rms(factor) per element.
    groups = [
        {"params": [params[k]], "lr": lr * float(init[k].float().pow(2).mean().sqrt().clamp_min(1e-12))}
        for k in KEYS
    ]
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.99))
    best = base_ho
    best_state = None
    best_step = 0
    rows = int(x_tr.shape[0])
    gen = torch.Generator(device=x_tr.device).manual_seed(0)
    for step in range(1, steps + 1):
        if rows > batch:
            idx = torch.randint(0, rows, (batch,), device=x_tr.device, generator=gen)
            loss = weighted_loss(x_tr[idx], w_tr[idx], y_tr[idx], params)
        else:
            loss = weighted_loss(x_tr, w_tr, y_tr, params)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                held = float(weighted_loss(x_ho, w_ho, y_ho, params))
            if held < best:
                best, best_step = held, step
                best_state = {k: v.detach().clone() for k, v in params.items()}
    with torch.no_grad():
        final_tr = float(weighted_loss(x_tr, w_tr, y_tr, best_state or init_f))
    return best_state, base_tr, base_ho, final_tr, best, best_step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--moments", required=True)
    ap.add_argument("--hidden-cache-dir", required=True,
                    help="the cache the deployed factors are calibrated on (the method config's hidden_cache_dir)")
    ap.add_argument("--sequence-cache", required=True)
    ap.add_argument("--out-dir", default=None, help="write a factor artifact here (omit to only measure)")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="layer indices (default all)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--min-train-tokens", type=int, default=64)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="keep refined factors only if held-out loss drops by more than this fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report", default=None, help="per-expert JSONL report path")
    ap.add_argument("--train-extra-caches", nargs="*", default=[],
                    help="more hidden caches of the SAME sequences (e.g. captured under compressed "
                         "models) whose training prompts are added to the fit, each with its own routing")
    ap.add_argument("--eval-caches", nargs="*", default=[],
                    help="caches evaluated on held-out prompts only, never trained on (drift test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    plan = json.loads(Path(args.plan).read_text())
    meta = plan.get("meta", {})
    for key, bad in (("factor_metric", "gram"),):
        if str(meta.get(key)) == bad:
            raise SystemExit("gram plans are not supported")
    hidden_meta = json.loads((Path(args.hidden_cache_dir) / "metadata.json").read_text())
    moments = torch.load(args.moments, map_location="cpu", weights_only=False)
    snapshot = moments["category_snapshots"][str(meta.get("basis_category", "coding"))]
    names = list(plan["layers"])
    selected = range(len(names)) if args.layers is None else args.layers
    out = Path(args.out_dir) if args.out_dir else None
    if out is not None:
        if (out / "metadata.json").exists():
            raise SystemExit(f"{out} already holds a factor artifact")
        out.mkdir(parents=True, exist_ok=True)
        if args.layers is not None:
            raise SystemExit("--out-dir needs every layer; drop --layers")
    report = open(args.report, "w") if args.report else None
    device = torch.device(args.device)
    layer_files, totals = {}, {"base_ho": 0.0, "ref_ho": 0.0, "n": 0, "kept": 0}
    holdout_mask = None

    for layer_index in selected:
        name = names[layer_index]
        started = time.time()
        stacked = load_stacked_layer(args.model, int(name.split(".")[2]), args.device)
        full_ids, tier_specs = _action_tiers(plan["layers"][name])
        removed = set(int(v) for v in plan["layers"][name]["by_action"].get("removed", []))
        retained = [e for e in range(int(stacked.num_experts)) if e not in removed]

        rows = int(hidden_meta["rows"][name])
        hidden = int(hidden_meta["hidden_sizes"][name])
        raw = torch.from_file(hidden_meta["paths"][name], shared=False, size=rows * hidden, dtype=torch.uint16)
        all_hidden = raw.view(torch.bfloat16).reshape(rows, hidden)
        row_ids = _sampled_rows(meta, rows, args.seed)
        calibration_hidden = all_hidden.index_select(0, row_ids)
        # Exactly the restructurer's from_stacked factors (same rows, seed, retained
        # set, and the same forked RNG it seeds before the randomized SVD).
        layer_seed = args.seed + layer_index * 1_000_003
        rng_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(layer_seed)
            state = _build_state(stacked, tier_specs, snapshot, name, calibration_hidden,
                                 meta, layer_seed, retained)
        del calibration_hidden

        if holdout_mask is None:
            holdout_mask = sequence_holdout_mask(args.sequence_cache, rows, args.holdout, args.seed)
        del all_hidden, raw
        kept_ids = torch.as_tensor(retained, device=device)
        ho_dev = holdout_mask.to(device)

        def open_cache(cache_dir):
            x_c = load_layer_hidden(cache_dir, name).to(device)
            chosen_c, weights_c = deployed_routing(x_c, stacked, kept_ids)
            return (x_c, chosen_c, weights_c, ho_dev)

        train_caches = [open_cache(args.hidden_cache_dir)] + [open_cache(c) for c in args.train_extra_caches]
        eval_caches = {Path(c).name.split(".")[-1]: open_cache(c) for c in args.eval_caches}

        for tier in state["tiers"]:
            for slot, expert in enumerate(tier["expert_ids"].tolist()):
                parts = [routed_tokens(c, expert, stacked, device) for c in train_caches]
                x = torch.cat([p[0] for p in parts]); g2 = torch.cat([p[1] for p in parts])
                y = torch.cat([p[2] for p in parts]); held = torch.cat([p[3] for p in parts])
                del parts
                init = {k: tier[k][slot].to(device) for k in KEYS}
                n_tr, n_ho = int((~held).sum()), int(held.sum())
                record = {"layer": layer_index, "expert": int(expert), "rank": int(tier["gate_up_rank"]),
                          "train_tokens": n_tr, "holdout_tokens": n_ho}
                if n_tr < args.min_train_tokens or n_ho < 8:
                    record.update(skipped=True)
                    best, keep = None, False
                else:
                    best, base_tr, base_ho, ref_tr, ref_ho, best_step = refine_expert(
                        x[~held], g2[~held], y[~held], x[held], g2[held], y[held], init,
                        steps=args.steps, lr=args.lr, batch=args.batch, eval_every=args.eval_every,
                    )
                    keep = best is not None and ref_ho < base_ho * (1.0 - args.min_gain)
                    record.update(base_train=base_tr, base_holdout=base_ho, refined_train=ref_tr,
                                  refined_holdout=ref_ho if keep else base_ho, best_step=best_step, kept=keep)
                    totals["n"] += 1
                    totals["base_ho"] += base_ho
                    totals["ref_ho"] += ref_ho if keep else base_ho
                    totals["kept"] += int(keep)
                del x, g2, y, held
                final = {k: (best[k] if keep else init[k].float()) for k in KEYS}
                init_f = {k: init[k].float() for k in KEYS}
                for tag, cache in eval_caches.items():
                    xe, ge, ye, he = routed_tokens(cache, expert, stacked, device)
                    if int(he.sum()) >= 8:
                        with torch.no_grad():
                            record[f"eval_{tag}_base"] = float(weighted_loss(xe[he], ge[he], ye[he], init_f))
                            record[f"eval_{tag}_refined"] = float(weighted_loss(xe[he], ge[he], ye[he], final))
                        record[f"eval_{tag}_tokens"] = int(he.sum())
                    del xe, ge, ye, he
                if keep:
                    dtype = tier[KEYS[0]].dtype
                    for k in KEYS:
                        tier[k][slot] = best[k].to(device="cpu", dtype=dtype)
                if report:
                    report.write(json.dumps(record) + "\n")
                    report.flush()
        del train_caches, eval_caches
        if out is not None:
            for tier in state["tiers"]:
                for key, value in list(tier.items()):
                    if torch.is_tensor(value):
                        tier[key] = value.detach().cpu()
            filename = f"{_safe_layer_name(name)}.pt"
            torch.save(state, out / filename)
            layer_files[name] = filename
        n = max(totals["n"], 1)
        logging.info("%s done in %.0fs | running mean held-out rel. error: base %.4f -> refined %.4f (%d/%d kept)",
                     name, time.time() - started, totals["base_ho"] / n, totals["ref_ho"] / n,
                     totals["kept"], totals["n"])
        del stacked, state
        torch.cuda.empty_cache()

    if out is not None:
        metadata = {
            "format": "routed_functional_svd_selected_factors",
            "format_version": 1,
            "plan_path": str(Path(args.plan).resolve()),
            "plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
            "storage_dtype": "torch.bfloat16",
            "refined": {"steps": args.steps, "lr": args.lr, "holdout": args.holdout,
                        "hidden_cache_dir": args.hidden_cache_dir,
                        "mean_base_holdout": totals["base_ho"] / max(totals["n"], 1),
                        "mean_refined_holdout": totals["ref_ho"] / max(totals["n"], 1),
                        "kept": totals["kept"], "experts": totals["n"]},
            "layers": layer_files,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logging.info("Saved refined factor artifact -> %s", out)
    if report:
        report.close()


if __name__ == "__main__":
    main()
