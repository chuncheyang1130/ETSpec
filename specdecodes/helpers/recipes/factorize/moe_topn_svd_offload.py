"""Recipe: TopN-Expert SVD draft for Qwen3-MoE (with CPU expert offload).

Inherits from the no-offload SVD recipe and adds:
  * `Offloader` to stream CPU experts onto GPU around each target forward.
  * A custom target `device_map` that pins experts in layers
    `[cpu_expert_layer_start, num_layers)` to CPU.

Constructed with no arguments — fixed config is hardcoded inline.
"""

from typing import Any, Dict

from ...offloaders.offloader import Offloader

from .moe_topn_svd_no_offload import Recipe as MoeTopNSVDNoOffloadRecipe


class Recipe(MoeTopNSVDNoOffloadRecipe):
    # Layers `[cpu_expert_layer_start, num_layers)` have their experts
    # pinned to CPU; `Offloader` streams them onto GPU around each target
    # forward.
    cpu_expert_layer_start: int = 16

    def __init__(self):
        super().__init__()
        # Required so `BaseRecipe.apply_offloading` actually runs and so the
        # builder's `load_model_and_tokenizer` knows to load the model on CPU
        # first (it checks `self.recipe.offloader` before deciding the
        # initial device_map). Set to None on a subclass to disable offload.
        self.offloader = Offloader

    def generate_configurations(
        self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device
    ):
        cfg = {
            "top_n": 16,
            "rank": 1560,
            "rank_down": 1560,
            "svd_device": "cuda:0",
            "log_expert_usage": False,
            "expert_usage_log_path": None,
        }

        if draft_model is not None:
            setattr(draft_model, "topn_svd_config", cfg)

        # Custom device_map:
        #   * experts of layers [0, cpu_expert_layer_start) -> `device`
        #   * experts of layers [cpu_expert_layer_start, num_layers) -> "cpu"
        #     (Offloader streams them onto GPU around each target forward)
        #   * gate (router), attention, layernorms, embed, norm, lm_head -> `device`
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

        return {"device_map": device_map}, {"svd_config": cfg}
