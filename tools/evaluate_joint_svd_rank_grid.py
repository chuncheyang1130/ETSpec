#!/usr/bin/env python
"""Evaluate gate/up and down SVD ranks from complete routed expert outputs.

The projection-level SVD report cannot capture the interaction

    down(silu(gate(h)) * up(h)).

This tool replays the same deterministic calibration setup, records the input to
each MoE layer in a disk-backed BF16 cache, and then analyzes one layer at a time.
For each middle-tier expert it constructs:

* one activation-calibrated input-side basis shared by gate and up; and
* one independently activation-calibrated down basis.

Every requested ``(gate_up_rank, down_rank)`` pair is evaluated against the
original expert output and weighted by squared routing probability. Candidate
pairs are filtered using the final Cat+Skip parameter formula. The model itself
is never replaced by low-rank modules in this analysis.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from specdecodes.helpers.recipes.moe.expert_calibration import load_selection
from specdecodes.models.utils.moe.base.activation_svd import expert_projection_weight
from specdecodes.models.utils.moe.base.expert_usage_tracker import _is_qwen3_moe_block
from specdecodes.models.utils.moe.base.ordinary_svd import metric_square_roots
from tools.analyze_activation_svd import (
    _generate_category_sequences,
    _load_moments,
    _select_analysis_views,
    _select_projection_moments,
    _select_tiers,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")


def _safe_name(module_name: str) -> str:
    return module_name.replace(".", "_").replace("/", "_")


def skipcat_middle_parameters(
    gate_up_rank: int,
    down_rank: int,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> int:
    """Parameters for one middle expert after Cat and block skipping."""

    r_gu = int(gate_up_rank)
    r_down = int(down_rank)
    if not (0 < r_gu <= hidden_size):
        raise ValueError(f"invalid gate/up rank {r_gu} for hidden size {hidden_size}")
    if not (0 < r_down <= intermediate_size):
        raise ValueError(
            f"invalid down rank {r_down} for intermediate size {intermediate_size}"
        )
    gate_up = r_gu * (hidden_size + 2 * intermediate_size - r_gu)
    down = r_down * (hidden_size + intermediate_size - r_down)
    return int(gate_up + down)


def tiered_expert_parameter_ratio(
    gate_up_rank: int,
    down_rank: int,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    full_experts: int,
    svd_experts: int,
) -> float:
    """Fraction of all original expert parameters retained by the three tiers."""

    matrix = int(hidden_size) * int(intermediate_size)
    dense = int(num_experts) * 3 * matrix
    kept = int(full_experts) * 3 * matrix + int(svd_experts) * skipcat_middle_parameters(
        gate_up_rank,
        down_rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    return float(kept / dense)


def _decompose_calibrated(
    weight: torch.Tensor,
    second_moment: torch.Tensor,
    max_rank: int,
    *,
    randomized: bool,
    scale_floor: float,
):
    """Return ``U, s, V, metric`` for the SVD of ``weight`` in a calibrated metric.

    ``second_moment`` is either a 1-D diagonal or a 2-D Gram; ``metric`` is the
    ``(root, inverse, is_matrix)`` triple from
    :func:`~specdecodes.models.utils.moe.base.ordinary_svd.metric_square_roots` and is
    passed straight through to :func:`_apply_factors`.
    """

    w = weight.detach().float()
    moment = second_moment.detach().to(device=w.device, dtype=torch.float32)
    if w.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(w.shape)}")
    if float(moment.nan_to_num(nan=0.0).clamp_min(0).max().item()) <= 0:
        raise ValueError("cannot construct calibrated factors from an all-zero moment")
    # Normalizing by the largest entry/eigenvalue does not move the singular
    # vectors but makes the inverse numerically predictable, while the relative
    # floor protects directions unseen in a finite calibration sample.
    root, inverse, is_matrix = metric_square_roots(
        moment, int(w.shape[1]), scale_floor=float(scale_floor)
    )
    weighted = w @ root if is_matrix else w * root.unsqueeze(0)
    metric = (root, inverse, bool(is_matrix))
    max_rank = int(max_rank)
    limit = min(weighted.shape)
    if not (0 < max_rank <= limit):
        raise ValueError(f"rank {max_rank} exceeds matrix rank limit {limit}")
    if randomized and max_rank < limit:
        q = min(limit, max_rank + 8)
        u, s, v = torch.svd_lowrank(weighted, q=q, niter=2)
        order = torch.argsort(s, descending=True)
        return u[:, order], s[order], v[:, order], metric
    u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
    return u, s, vh.t(), metric


def _apply_factors(
    inputs: torch.Tensor,
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    metric: tuple[torch.Tensor, torch.Tensor, bool],
    rank: int,
) -> torch.Tensor:
    """Apply the original-coordinate rank prefix of calibrated SVD factors."""

    rank = int(rank)
    root, inverse, is_matrix = metric
    whitened = inputs.float() @ inverse if is_matrix else inputs.float() / root.unsqueeze(0)
    latent = whitened @ v[:, :rank]
    reconstruction = u[:, :rank] * s[:rank].unsqueeze(0)
    return latent @ reconstruction.t()


class _DiskHiddenCapture:
    """Append MoE inputs to one raw BF16 file per layer."""

    def __init__(self, model, module_names: Sequence[str], cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        modules = dict(model.named_modules())
        self.paths: Dict[str, str] = {}
        self.rows = {name: 0 for name in module_names}
        self.hidden_sizes: Dict[str, int] = {}
        self.handles = []
        self.files = {}
        for name in module_names:
            module = modules[name]
            hidden = int(expert_projection_weight(module, 0, "gate").shape[1])
            path = self.cache_dir / f"{_safe_name(name)}.bf16"
            handle = open(path, "wb")
            self.paths[name] = str(path)
            self.hidden_sizes[name] = hidden
            self.files[name] = handle
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        @torch.no_grad()
        def hook(_module, inputs) -> None:
            hidden = inputs[0] if isinstance(inputs, tuple) else inputs
            flat = hidden.reshape(-1, hidden.shape[-1])
            raw = (
                flat.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                .view(torch.uint16)
                .numpy()
            )
            raw.tofile(self.files[name])
            self.rows[name] += int(flat.shape[0])

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for handle in self.files.values():
            handle.close()
        self.files.clear()

    def metadata(self):
        return {
            "format": "raw_bfloat16_as_uint16",
            "paths": dict(self.paths),
            "rows": dict(self.rows),
            "hidden_sizes": dict(self.hidden_sizes),
        }


def _load_hidden_cache(metadata, name: str) -> torch.Tensor:
    path = metadata["paths"][name]
    rows = int(metadata["rows"][name])
    hidden = int(metadata["hidden_sizes"][name])
    expected_bytes = rows * hidden * 2
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"hidden cache size mismatch for {name}: {actual_bytes} != {expected_bytes}"
        )
    raw = torch.from_file(path, shared=False, size=rows * hidden, dtype=torch.uint16)
    return raw.view(torch.bfloat16).reshape(rows, hidden)


def _make_generator(model, tokenizer, chunk_size: int):
    from specdecodes.models.generators.naive import NaiveGenerator

    return NaiveGenerator(
        target_model=model,
        tokenizer=tokenizer,
        draft_model=None,
        draft_params=None,
        cache_implementation="static",
        profiling=False,
        profiling_verbose=False,
        generator_kwargs={"prefill_chunk_size": int(chunk_size)},
    )


@torch.no_grad()
def _capture_hidden_states(
    model,
    generator,
    sequences_by_category,
    module_names: Sequence[str],
    *,
    cache_dir: str,
    device: str,
    chunk_size: int,
):
    from specdecodes.models.utils.cache_utils import create_kv_cache

    capture = _DiskHiddenCapture(model, module_names, cache_dir)
    replay_tokens = 0
    try:
        for category, sequences in sequences_by_category.items():
            for sequence in tqdm(sequences, desc=f"Capturing {category} MoE inputs"):
                replay_ids = sequence[:, :-1]
                if replay_ids.numel() == 0:
                    continue
                replay_tokens += int(replay_ids.numel())
                replay_cache = create_kv_cache("dynamic")
                outputs = generator._chunked_prefill_forward(
                    replay_ids.to(device),
                    replay_cache,
                    prefill_chunk_size=int(chunk_size),
                    use_position_ids=True,
                )
                del outputs, replay_cache
    finally:
        capture.close()
    metadata = capture.metadata()
    metadata["replay_tokens"] = int(replay_tokens)
    for name in module_names:
        if int(metadata["rows"][name]) != replay_tokens:
            raise RuntimeError(
                f"{name} captured {metadata['rows'][name]} rows for {replay_tokens} replay tokens"
            )
    metadata_path = Path(cache_dir) / "metadata.json"
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    logging.info("Saved disk-backed hidden cache -> %s", metadata_path)
    return metadata


def _load_hidden_metadata(cache_dir: str):
    path = Path(cache_dir) / "metadata.json"
    with open(path) as handle:
        metadata = json.load(handle)
    return metadata


@torch.no_grad()
def _routing_for_hidden(module, hidden: torch.Tensor, *, device: str, batch_size: int):
    values = []
    indices = []
    dtype = expert_projection_weight(module, 0, "gate").dtype
    for start in range(0, hidden.shape[0], int(batch_size)):
        batch = hidden[start:start + int(batch_size)].to(device=device, dtype=dtype)
        routing = getattr(module, "_routing_weights", None)
        if not callable(routing):
            raise TypeError(
                f"{module.__class__.__name__} must expose restricted _routing_weights"
            )
        topk_values, topk_indices = routing(batch)
        values.append(topk_values.float().cpu())
        indices.append(topk_indices.long().cpu())
    return torch.cat(values), torch.cat(indices)


@torch.no_grad()
def _evaluate_layer(
    module,
    hidden: torch.Tensor,
    expert_ids: Sequence[int],
    scores: torch.Tensor,
    gate_moments: torch.Tensor,
    up_moments: torch.Tensor,
    down_moments: torch.Tensor,
    feasible_pairs,
    *,
    device: str,
    routing_batch_size: int,
    max_tokens_per_expert: int,
    randomized: bool,
    scale_floor: float,
    down_calibration_input: str,
    seed: int,
):
    routing_values, routing_indices = _routing_for_hidden(
        module, hidden, device=device, batch_size=routing_batch_size
    )
    max_gu_rank = max(pair[0] for pair in feasible_pairs)
    max_down_rank = max(pair[1] for pair in feasible_pairs)
    gu_ranks = sorted({pair[0] for pair in feasible_pairs})
    down_ranks_by_gu = {
        rank: sorted({r_down for r_gu, r_down in feasible_pairs if r_gu == rank})
        for rank in gu_ranks
    }
    aggregate = {
        f"gu{r_gu}_down{r_down}": {
            "numerator": 0.0,
            "denominator": 0.0,
            "expert_error_sum": 0.0,
            "importance_weighted_error": 0.0,
            "importance_sum": 0.0,
            "experts": 0,
            "samples": 0,
        }
        for r_gu, r_down in feasible_pairs
    }
    expert_reports = {}
    for expert_id in tqdm(list(expert_ids), desc="Joint expert rank grid", leave=False):
        token_idx, slot_idx = (routing_indices == int(expert_id)).nonzero(as_tuple=True)
        routed_count = int(token_idx.numel())
        if routed_count == 0:
            continue
        if routed_count > int(max_tokens_per_expert):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + int(expert_id) * 1009)
            chosen = torch.randperm(routed_count, generator=generator)[: int(max_tokens_per_expert)]
            token_idx = token_idx.index_select(0, chosen)
            slot_idx = slot_idx.index_select(0, chosen)
        x = hidden.index_select(0, token_idx).to(device=device, dtype=torch.float32)
        route = routing_values[token_idx, slot_idx].to(device=device, dtype=torch.float32)
        route_square = route.square()

        gate_weight = expert_projection_weight(module, expert_id, "gate").float()
        up_weight = expert_projection_weight(module, expert_id, "up").float()
        down_weight = expert_projection_weight(module, expert_id, "down").float()
        original_gate = x @ gate_weight.t()
        original_up = x @ up_weight.t()
        original_z = F.silu(original_gate) * original_up
        original_y = original_z @ down_weight.t()
        sampled_count = int(token_idx.numel())
        sampling_expansion = float(routed_count / max(sampled_count, 1))
        sampled_denominator = float(
            (route_square * original_y.square().sum(dim=1)).sum().item()
        )
        denominator = sampling_expansion * sampled_denominator
        if denominator <= 0:
            continue

        joint_moment = gate_moments[expert_id] + up_moments[expert_id]
        stacked_weight = torch.cat((gate_weight, up_weight), dim=0)
        u_gu, s_gu, v_gu, scale_gu = _decompose_calibrated(
            stacked_weight,
            joint_moment,
            max_gu_rank,
            randomized=randomized,
            scale_floor=scale_floor,
        )
        original_down_factors = None
        if down_calibration_input == "original":
            original_down_factors = _decompose_calibrated(
                down_weight,
                down_moments[expert_id],
                max_down_rank,
                randomized=randomized,
                scale_floor=scale_floor,
            )
        intermediate = int(gate_weight.shape[0])
        expert_pairs = {}
        importance = max(0.0, float(scores[expert_id].item()))
        for r_gu in gu_ranks:
            joined = _apply_factors(x, u_gu, s_gu, v_gu, scale_gu, r_gu)
            compressed_gate = joined[:, :intermediate]
            compressed_up = joined[:, intermediate:]
            compressed_z = F.silu(compressed_gate) * compressed_up
            if down_calibration_input == "compressed":
                conditional_down_moment = (
                    route_square.unsqueeze(1) * compressed_z.square()
                ).sum(dim=0)
                down_factors = _decompose_calibrated(
                    down_weight,
                    conditional_down_moment,
                    max(down_ranks_by_gu[r_gu]),
                    randomized=randomized,
                    scale_floor=scale_floor,
                )
            else:
                down_factors = original_down_factors
            u_down, s_down, v_down, scale_down = down_factors
            for r_down in down_ranks_by_gu[r_gu]:
                compressed_y = _apply_factors(
                    compressed_z, u_down, s_down, v_down, scale_down, r_down
                )
                sampled_numerator = float(
                    (
                        route_square
                        * (original_y - compressed_y).square().sum(dim=1)
                    )
                    .sum()
                    .item()
                )
                numerator = sampling_expansion * sampled_numerator
                relative_error = numerator / denominator
                key = f"gu{r_gu}_down{r_down}"
                expert_pairs[key] = relative_error
                agg = aggregate[key]
                agg["numerator"] += numerator
                agg["denominator"] += denominator
                agg["expert_error_sum"] += relative_error
                agg["importance_weighted_error"] += importance * relative_error
                agg["importance_sum"] += importance
                agg["experts"] += 1
                agg["samples"] += int(token_idx.numel())
            del joined, compressed_gate, compressed_up, compressed_z
            if down_calibration_input == "compressed":
                del u_down, s_down, v_down, scale_down, down_factors
        expert_reports[str(expert_id)] = {
            "routed_tokens": routed_count,
            "sampled_tokens": int(token_idx.numel()),
            "sampling_expansion": sampling_expansion,
            "ranking_score": importance,
            "output_energy": denominator,
            "relative_errors": expert_pairs,
        }

        del (
            x,
            gate_weight,
            up_weight,
            down_weight,
            original_gate,
            original_up,
            original_z,
            original_y,
            stacked_weight,
            u_gu,
            s_gu,
            v_gu,
            scale_gu,
        )
        if original_down_factors is not None:
            del original_down_factors, u_down, s_down, v_down, scale_down, down_factors

    summary = {}
    for key, agg in aggregate.items():
        summary[key] = {
            "global_relative_error": agg["numerator"] / max(agg["denominator"], 1e-30),
            "mean_expert_relative_error": agg["expert_error_sum"] / max(agg["experts"], 1),
            "importance_weighted_relative_error": (
                agg["importance_weighted_error"] / max(agg["importance_sum"], 1e-30)
            ),
            "importance_weighted_error_sum": float(agg["importance_weighted_error"]),
            "importance_sum": float(agg["importance_sum"]),
            "experts": int(agg["experts"]),
            "samples": int(agg["samples"]),
            "absolute_error": float(agg["numerator"]),
            "output_energy": float(agg["denominator"]),
        }
    return expert_reports, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--moments", required=True)
    parser.add_argument("--ranking-moments", default=None)
    parser.add_argument("--routing-kept-path", required=True)
    parser.add_argument("--analysis-category", default="coding")
    parser.add_argument("--full-experts", type=int, default=32)
    parser.add_argument("--svd-experts", type=int, default=64)
    parser.add_argument("--gate-up-ranks", nargs="+", type=int, default=[256, 320, 384, 424, 448, 512])
    parser.add_argument("--down-ranks", nargs="+", type=int, default=[128, 256, 384, 512])
    parser.add_argument("--max-expert-memory-ratio", type=float, default=0.50)
    parser.add_argument("--max-tokens-per-expert", type=int, default=1024)
    parser.add_argument("--sequence-cache", default=None)
    parser.add_argument("--hidden-cache-dir", default=None)
    parser.add_argument("--reuse-hidden-cache", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--routing-batch-size", type=int, default=2048)
    parser.add_argument("--max-cache-len", type=int, default=4096)
    parser.add_argument("--scale-floor", type=float, default=1e-12)
    parser.add_argument(
        "--down-calibration-input",
        choices=["compressed", "original"],
        default="compressed",
        help=(
            "rebuild each down basis from its candidate compressed SwiGLU input, "
            "or reuse the moments collected from the original gate/up"
        ),
    )
    parser.add_argument("--exact-svd", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=None)
    args = parser.parse_args()

    if not (0 < args.max_expert_memory_ratio <= 1):
        parser.error("--max-expert-memory-ratio must be in (0, 1]")
    if args.max_tokens_per_expert <= 0 or args.chunk_size <= 0:
        parser.error("token and chunk limits must be positive")
    if not args.gate_up_ranks or not args.down_ranks:
        parser.error("both rank lists must be non-empty")
    torch.manual_seed(int(args.seed))

    moment_artifact = _load_moments(args.moments)
    ranking_artifact = (
        _load_moments(args.ranking_moments)
        if args.ranking_moments
        else moment_artifact
    )
    scores = _select_analysis_views(ranking_artifact, args.analysis_category)[0]
    gate_moments = _select_projection_moments(
        moment_artifact, args.analysis_category, "gate"
    )
    up_moments = _select_projection_moments(
        moment_artifact, args.analysis_category, "up"
    )
    down_moments = _select_projection_moments(
        moment_artifact, args.analysis_category, "down"
    )
    full, svd, pruned = _select_tiers(scores, args.full_experts, args.svd_experts)
    if args.max_layers is not None:
        keep_names = list(svd)[: int(args.max_layers)]
        full = {name: full[name] for name in keep_names}
        svd = {name: svd[name] for name in keep_names}
        pruned = {name: pruned[name] for name in keep_names}
        scores = {name: scores[name] for name in keep_names}
        gate_moments = {name: gate_moments[name] for name in keep_names}
        up_moments = {name: up_moments[name] for name in keep_names}
        down_moments = {name: down_moments[name] for name in keep_names}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info("Loading %s directly on %s (%s)", args.model, args.device, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    ).eval()
    from specdecodes.models.utils.moe.base.apply_stacked_moe import apply_stacked_target

    logging.info("Converted %d MoE blocks to memory-safe stacked form", apply_stacked_target(model))
    routing_kept, routing_meta = load_selection(args.routing_kept_path)
    modules = dict(model.named_modules())
    for name, ids in routing_kept.items():
        if name in modules:
            modules[name].set_kept(torch.as_tensor(ids, dtype=torch.long))
    logging.info(
        "Restricted routing using %s (%s)",
        args.routing_kept_path,
        routing_meta.get("selection_category", "unknown category"),
    )

    module_names = list(svd)
    if not module_names or any(not _is_qwen3_moe_block(modules[name]) for name in module_names):
        raise RuntimeError("no supported MoE layers are available for joint analysis")
    sample_module = modules[module_names[0]]
    sample_gate = expert_projection_weight(sample_module, svd[module_names[0]][0], "gate")
    intermediate_size, hidden_size = map(int, sample_gate.shape)
    num_experts = int(sample_module.num_experts)
    feasible_pairs = []
    pair_memory = {}
    for r_gu in sorted(set(args.gate_up_ranks)):
        for r_down in sorted(set(args.down_ranks)):
            ratio = tiered_expert_parameter_ratio(
                r_gu,
                r_down,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                full_experts=args.full_experts,
                svd_experts=args.svd_experts,
            )
            key = f"gu{r_gu}_down{r_down}"
            pair_memory[key] = ratio
            if ratio <= float(args.max_expert_memory_ratio) + 1e-12:
                feasible_pairs.append((int(r_gu), int(r_down)))
    if not feasible_pairs:
        raise ValueError("no rank pairs satisfy the requested expert-memory budget")
    logging.info("Evaluating %d feasible rank pair(s)", len(feasible_pairs))

    base = os.path.splitext(args.out)[0]
    sequence_cache = args.sequence_cache or base + ".sequences.pt"
    hidden_cache_dir = args.hidden_cache_dir or base + ".hidden_states"
    if os.path.exists(sequence_cache):
        sequence_artifact = _load_moments(sequence_cache)
        sequences = sequence_artifact["sequences"]
        generator = _make_generator(model, tokenizer, args.chunk_size)
        logging.info("Loaded generated calibration sequences <- %s", sequence_cache)
    else:
        from run.pipelines.benchmarks.loader.openhermes import load_openhermes_dataset

        meta = moment_artifact.get("meta") or {}
        categories = list(meta.get("categories") or [args.analysis_category])
        per_category = int(meta.get("per_category") or 128)
        gen_tokens = int(meta.get("gen_tokens") or 256)
        rows = load_openhermes_dataset(
            mode="stratified",
            per_category=per_category,
            categories=categories,
            scan_cap=300_000,
        )
        prompts = {
            category: [row["query"] for row in rows if row["category"] == category]
            for category in categories
        }
        generator, sequences, skipped = _generate_category_sequences(
            model,
            tokenizer,
            prompts,
            device=args.device,
            gen_tokens=gen_tokens,
            max_cache_len=args.max_cache_len,
            chunk_size=args.chunk_size,
        )
        os.makedirs(os.path.dirname(sequence_cache) or ".", exist_ok=True)
        torch.save(
            {
                "sequences": sequences,
                "meta": {
                    "model": args.model,
                    "categories": categories,
                    "per_category": per_category,
                    "gen_tokens": gen_tokens,
                    "enable_thinking": False,
                    "routing_kept_path": args.routing_kept_path,
                    "skipped": skipped,
                },
            },
            sequence_cache,
        )
        logging.info("Saved generated calibration sequences -> %s", sequence_cache)

    metadata_path = Path(hidden_cache_dir) / "metadata.json"
    if args.reuse_hidden_cache and metadata_path.exists():
        hidden_metadata = _load_hidden_metadata(hidden_cache_dir)
        missing = sorted(set(module_names) - set(hidden_metadata.get("paths", {})))
        if missing:
            raise KeyError(f"hidden cache is missing layers: {missing[:4]}")
        logging.info("Reusing disk-backed hidden cache <- %s", metadata_path)
    else:
        hidden_metadata = _capture_hidden_states(
            model,
            generator,
            sequences,
            module_names,
            cache_dir=hidden_cache_dir,
            device=args.device,
            chunk_size=args.chunk_size,
        )

    del generator, sequences
    gc.collect()
    torch.cuda.empty_cache()

    global_aggregate = {
        f"gu{r_gu}_down{r_down}": {
            "numerator": 0.0,
            "denominator": 0.0,
            "expert_error_sum": 0.0,
            "importance_weighted_error": 0.0,
            "importance_sum": 0.0,
            "experts": 0,
            "samples": 0,
        }
        for r_gu, r_down in feasible_pairs
    }
    layer_reports = {}
    for layer_index, name in enumerate(module_names):
        logging.info("Analyzing joint expert outputs for %s", name)
        hidden = _load_hidden_cache(hidden_metadata, name)
        experts, layer_summary = _evaluate_layer(
            modules[name],
            hidden,
            svd[name],
            scores[name],
            gate_moments[name],
            up_moments[name],
            down_moments[name],
            feasible_pairs,
            device=args.device,
            routing_batch_size=args.routing_batch_size,
            max_tokens_per_expert=args.max_tokens_per_expert,
            randomized=not args.exact_svd,
            scale_floor=args.scale_floor,
            down_calibration_input=args.down_calibration_input,
            seed=args.seed + layer_index * 1_000_003,
        )
        layer_reports[name] = {"summary": layer_summary, "experts": experts}
        for key, values in layer_summary.items():
            agg = global_aggregate[key]
            agg["numerator"] += values["absolute_error"]
            agg["denominator"] += values["output_energy"]
            agg["expert_error_sum"] += (
                values["mean_expert_relative_error"] * values["experts"]
            )
            agg["importance_weighted_error"] += (
                values["importance_weighted_error_sum"]
            )
            agg["importance_sum"] += values["importance_sum"]
            agg["experts"] += values["experts"]
            agg["samples"] += values["samples"]
        del hidden
        gc.collect()
        torch.cuda.empty_cache()

    summary = {}
    for key, agg in global_aggregate.items():
        summary[key] = {
            "gate_up_rank": int(key.split("_")[0][2:]),
            "down_rank": int(key.split("_")[1][4:]),
            "expert_parameter_ratio": pair_memory[key],
            "global_relative_error": agg["numerator"] / max(agg["denominator"], 1e-30),
            "mean_expert_relative_error": agg["expert_error_sum"] / max(agg["experts"], 1),
            "importance_weighted_relative_error": (
                agg["importance_weighted_error"] / max(agg["importance_sum"], 1e-30)
            ),
            "experts": int(agg["experts"]),
            "samples": int(agg["samples"]),
        }
    best_key = min(summary, key=lambda key: summary[key]["global_relative_error"])
    report = {
        "meta": {
            "model": args.model,
            "moments": args.moments,
            "ranking_moments": args.ranking_moments,
            "routing_kept_path": args.routing_kept_path,
            "analysis_category": args.analysis_category,
            "full_experts": int(args.full_experts),
            "svd_experts": int(args.svd_experts),
            "pruned_experts": int(num_experts - args.full_experts - args.svd_experts),
            "max_expert_memory_ratio": float(args.max_expert_memory_ratio),
            "max_tokens_per_expert": int(args.max_tokens_per_expert),
            "sequence_cache": sequence_cache,
            "hidden_cache_dir": hidden_cache_dir,
            "enable_thinking": False,
            "scale_floor": float(args.scale_floor),
            "randomized_svd": not args.exact_svd,
            "down_calibration_input": args.down_calibration_input,
            "down_factorization": (
                "original down weight recalibrated separately on each candidate "
                "compressed post-SwiGLU input"
                if args.down_calibration_input == "compressed"
                else "original down weight calibrated on original post-SwiGLU moments"
            ),
            "objective": "sum routed_probability^2 * ||original_expert_output - compressed_expert_output||^2",
        },
        "best_pair": {"key": best_key, **summary[best_key]},
        "summary": summary,
        "pair_memory_all_requested": pair_memory,
        "layers": layer_reports,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    logging.info("Saved joint SVD rank-grid report -> %s", args.out)
    for key, values in sorted(summary.items(), key=lambda item: item[1]["global_relative_error"]):
        logging.info(
            "%s: memory=%.4f%%, global complete-expert error=%.6f",
            key,
            100 * values["expert_parameter_ratio"],
            values["global_relative_error"],
        )


if __name__ == "__main__":
    main()
