"""Recipe for a routed-functional heterogeneous expert-compression plan."""

from __future__ import annotations

import os

from ..base_recipe import BaseRecipe
from ...restructurer.moe_routed_functional_svd import (
    MoERoutedFunctionalSVDRestructurer,
)


class Recipe(BaseRecipe):
    def __init__(
        self,
        plan_path: str,
        moments_path: str,
        hidden_cache_dir: str | None = None,
        factors_dir: str | None = None,
        category: str = "coding",
        router_candidates: int | None = None,
        routing_batch_size: int = 2048,
        randomized: bool = True,
        scale_floor: float = 1e-12,
        seed: int = 0,
        routing_penalty_scale: float | None = None,
        tier_output_ablation: str = "none",
        tier_routing_penalty: dict | None = None,
        removed_policy: str = "reroute",
        inductor_cc: str | None = None,
        inductor_cxx: str | None = None,
    ) -> None:
        super().__init__()
        for label, path in (("plan_path", plan_path), ("moments_path", moments_path)):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} not found: {path}")
        if hidden_cache_dir is not None:
            metadata = os.path.join(hidden_cache_dir, "metadata.json")
            if not os.path.isfile(metadata):
                raise FileNotFoundError(f"hidden-cache metadata not found: {metadata}")
        if factors_dir is not None:
            metadata = os.path.join(factors_dir, "metadata.json")
            if not os.path.isfile(metadata):
                raise FileNotFoundError(f"factor-artifact metadata not found: {metadata}")
        if tier_output_ablation not in ("none", "zero"):
            raise ValueError(
                f"tier_output_ablation must be 'none' or 'zero', got {tier_output_ablation!r}"
            )
        if tier_routing_penalty is not None:
            if routing_penalty_scale:
                raise ValueError(
                    "tier_routing_penalty and routing_penalty_scale are exclusive"
                )
            tier_routing_penalty = {
                str(action): float(value) for action, value in tier_routing_penalty.items()
            }
            if any(value < 0 for value in tier_routing_penalty.values()):
                raise ValueError("tier_routing_penalty values are logit discounts, >= 0")
        for label, path in (("CC", inductor_cc), ("CXX", inductor_cxx)):
            if path is not None:
                if not os.path.isfile(path) or not os.access(path, os.X_OK):
                    raise FileNotFoundError(f"Inductor compiler is not executable: {path}")
                os.environ[label] = path
        self.restructurer = MoERoutedFunctionalSVDRestructurer
        self.config = {
            "kind": "routed_functional_svd",
            "plan_path": plan_path,
            "moments_path": moments_path,
            "hidden_cache_dir": hidden_cache_dir,
            "factors_dir": factors_dir,
            "category": str(category),
            "routing_batch_size": int(routing_batch_size),
            "randomized": bool(randomized),
            "scale_floor": float(scale_floor),
            "seed": int(seed),
            # Discounts tiered experts' router logits at deployment. It changes
            # no factor, so one factor artifact serves an entire sweep. None
            # means "whatever the plan was priced with"; an explicit value
            # overrides that and is reported as off-plan.
            "routing_penalty_scale": (
                None if routing_penalty_scale is None else float(routing_penalty_scale)
            ),
        }
        # Ablation, not a deployment: "zero" silences every low-rank expert while
        # it keeps its router slot and weight, isolating slot-keeping from
        # reconstruction. "none" is the untouched block.
        self.config["tier_output_ablation"] = str(tier_output_ablation)
        # Per-action router-logit discount, e.g. {"gu128_down128": 1.3}: every
        # expert on that tier loses the amount from its logit before top-k, so
        # it wins fewer slots and weighs less in the ones it keeps. Full experts
        # are never discounted. None is the untouched router.
        self.config["tier_routing_penalty"] = tier_routing_penalty
        if removed_policy not in ("reroute", "drop"):
            raise ValueError(f"removed_policy must be 'reroute' or 'drop', got {removed_policy!r}")
        self.config["removed_policy"] = str(removed_policy)
        if router_candidates is not None:
            self.config["router_candidates"] = int(router_candidates)

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        return {"structure_config": dict(self.config)}, {"structure_config": {}}
