"""Factorize recipe: shared-basis SVD for normal chat-model MLP projections."""

from ..base_recipe import BaseRecipe
from ...factorizers.mlp_svd import MLPSharedBasisSVDFactorizer


class Recipe(BaseRecipe):
    def __init__(self):
        super().__init__()
        self.factorizer = MLPSharedBasisSVDFactorizer
        self.quantizer = None
        self.offloader = None

    def generate_configurations(self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device):
        svd_config = {
            "rank": 64,
            "modules_to_factor": [
                r"model\.layers\.\d+\.mlp\.gate_proj$",
                r"model\.layers\.\d+\.mlp\.up_proj$",
            ],
        }

        draft_config = {
            "svd_config": svd_config,
        }
        target_config = {}

        return target_config, draft_config
