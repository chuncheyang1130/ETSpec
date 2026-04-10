"""Factorize recipe: shared-basis SVD for MoE MLP experts."""

from ..base_recipe import BaseRecipe
from hqq.core.quantize import *
from ...factorizers.svd import MoEMLPSharedBasisSVDFactorizer


class Recipe(BaseRecipe):
    def __init__(self):
        super().__init__()
        self.factorizer = MoEMLPSharedBasisSVDFactorizer
        self.quantizer = None
        self.offloader = None

    def generate_configurations(self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device):
        svd_config = {
            "rank": 64,
            "expert_group_size": 4,
            "modules_to_factor": [
                r".*gate_proj$",
                r".*up_proj$",
                # r".*down_proj$",
            ],
        }
        
        # quant_config = {}
        # mlp_quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)

        # layer_cnt = len(target_model.model.layers)
        # quant_start = 0
        # quant_end = layer_cnt
        # for i in range(quant_start, quant_end):
        #     for j in range(len(target_model.model.layers[i].mlp.experts)):
        #         quant_config[f"model.layers.{i}.mlp.experts.{j}.gate_proj"] = mlp_quant_config
        #         quant_config[f"model.layers.{i}.mlp.experts.{j}.up_proj"] = mlp_quant_config
        #         quant_config[f"model.layers.{i}.mlp.experts.{j}.down_proj"] = mlp_quant_config

        # device_map = {}
        # for name, _ in target_model.named_parameters():
        #     layer_name = ".".join(name.split(".")[:-1])
        #     if layer_name in quant_config:
        #         device_map[layer_name] = "cpu"
        #     else:
        #         device_map[layer_name] = device
        # for name, _ in target_model.named_buffers():
        #     layer_name = ".".join(name.split(".")[:-1])
        #     device_map[layer_name] = device
            
        draft_config = {
            "svd_config": svd_config,
            # "quant_config": quant_config
        }
        target_config = {
            # "device_map": device_map,
        }
        
        return target_config, draft_config
