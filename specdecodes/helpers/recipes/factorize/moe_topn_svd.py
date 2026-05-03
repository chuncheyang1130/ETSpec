"""Recipe: TopN-Expert SVD draft for Qwen3-MoE.

The actual SVD replacement happens at generate-time once expert usage has
been observed during prefill, so this recipe only stores the configuration.
The generator (`MoESvdSDGenerator`) reads `topn_svd_config` from the draft
model and invokes the factorizer.
"""

from typing import Any, Dict

from ..base_recipe import BaseRecipe
from ...factorizers.moe_topn_svd import MoETopNSVDFactorizer
from ...offloaders.offloader import Offloader


class Recipe(BaseRecipe):
    def __init__(self, topn_svd_config: Dict[str, Any] | None = None):
        super().__init__()
        self.factorizer = MoETopNSVDFactorizer
        # Required so `BaseRecipe.apply_offloading` actually runs and so the
        # builder's `load_model_and_tokenizer` knows to load the model on CPU
        # first (it checks `self.recipe.offloader` before deciding the
        # initial device_map). Switch back to None to disable offloading.
        self.offloader = Offloader

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        # Stash the config on the draft so generator-side code can read it
        # (currently informational; real SVD-fill uses these values in step 2).
        if draft_model is not None:
            setattr(draft_model, "topn_svd_config", dict(self.topn_svd_config))
            
        svd_config = {
            "top_n": 16,
            # `rank` is the SVD rank for the gate/up factorization.
            "rank": 1560,
            # `rank_down` is the SVD rank for the down factorization. If left
            # unset (None), it defaults to `rank` inside the factorizer.
            "rank_down": None,
            "svd_device": "cuda:0",
            # Round-to-round tracking driver:
            #   "tree"     - count routing on every position in the verification
            #                tree (all candidates, including rejected branches).
            #   "verified" - count routing only on accepted tokens, summed over
            #                a sliding window of the last `verified_window`
            #                accepted tokens.
            "tracking_mode": "verified",
            "verified_window": 16,
        }

        # ------------------------------------------------------------------
        # Build target's device_map.
        #   * experts of layers [0, cpu_expert_layer_start) → `device`
        #   * experts of layers [cpu_expert_layer_start, num_layers) → "cpu"
        #     (Offloader streams them onto GPU around each target forward)
        #   * gate (router), attention, layernorms, embed, norm, lm_head → `device`
        #
        # The map keys are *parameter-bearing module* names (parents of
        # weight/buffer tensors) so `Offloader.check_device_map` is satisfied.
        # ------------------------------------------------------------------
        cpu_expert_layer_start = 16
        device_map: Dict[str, str] = {}

        for name, _ in target_model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            parts = layer_name.split(".")
            is_expert_param = (
                len(parts) >= 5
                and parts[0] == "model"
                and parts[1] == "layers"
                and parts[3] == "mlp"
                and parts[4] == "experts"
                and parts[2].isdigit()
            )
            if is_expert_param and int(parts[2]) >= cpu_expert_layer_start:
                device_map[layer_name] = "cpu"
            else:
                device_map[layer_name] = device

        # Buffers (rotary inv_freq, etc.) always stay on `device`.
        for name, _ in target_model.named_buffers():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = device

        if "lm_head" not in device_map:
            device_map["lm_head"] = device

        # Drive the build-time replacement: the builder calls `apply_svd`,
        # which calls `MoETopNSVDFactorizer.factorize_model(svd_config=...)`,
        # which swaps each Qwen3MoE block for a PackedTopNSvdMoeBlock with
        # random-initialized packed tensors.
        target_config: Dict[str, Any] = {"device_map": device_map}
        draft_config: Dict[str, Any] = {"svd_config": svd_config}
        return target_config, draft_config
