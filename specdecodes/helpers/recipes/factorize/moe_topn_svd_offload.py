"""Recipe: TopN-Expert SVD draft for Qwen3-MoE (with CPU expert offload)."""

from typing import Any, Dict

from ..base_recipe import BaseRecipe
from ...factorizers.moe_topn_svd import MoETopNSVDFactorizer
from ...offloaders.offloader import Offloader


_DEFAULTS: Dict[str, Any] = {
    "top_n": 16,
    "rank": 1560,
    "rank_down": None,
    "svd_device": "cuda:0",
}


class Recipe(BaseRecipe):
    def __init__(
        self,
        topn_svd_config: Dict[str, Any] | None = None,
        cpu_expert_layer_start: int = 16,
    ):
        super().__init__()
        self.factorizer = MoETopNSVDFactorizer
        # Required so `BaseRecipe.apply_offloading` actually runs and so the
        # builder's `load_model_and_tokenizer` knows to load the model on CPU
        # first (it checks `self.recipe.offloader` before deciding the
        # initial device_map). Switch to None to disable offloading.
        self.offloader = Offloader
        self.topn_svd_config: Dict[str, Any] = dict(topn_svd_config or {})
        self.cpu_expert_layer_start = int(cpu_expert_layer_start)

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        svd_config: Dict[str, Any] = {**_DEFAULTS, **self.topn_svd_config}
        if draft_model is not None:
            setattr(draft_model, "topn_svd_config", svd_config)

        # ------------------------------------------------------------------
        # Build target's device_map.
        #   * experts of layers [0, cpu_expert_layer_start) → `device`
        #   * experts of layers [cpu_expert_layer_start, num_layers) → "cpu"
        #     (Offloader streams them onto GPU around each target forward)
        #   * gate (router), attention, layernorms, embed, norm, lm_head → `device`
        # ------------------------------------------------------------------
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
            if is_expert_param and int(parts[2]) >= self.cpu_expert_layer_start:
                device_map[layer_name] = "cpu"
            else:
                device_map[layer_name] = device

        for name, _ in target_model.named_buffers():
            layer_name = ".".join(name.split(".")[:-1])
            device_map[layer_name] = device

        if "lm_head" not in device_map:
            device_map["lm_head"] = device

        target_config: Dict[str, Any] = {"device_map": device_map}
        draft_config: Dict[str, Any] = {"svd_config": svd_config}
        return target_config, draft_config
