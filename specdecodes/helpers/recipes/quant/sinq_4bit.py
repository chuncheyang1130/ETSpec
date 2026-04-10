from ..base_recipe import BaseRecipe
from ...quantizers.sinq import SINQQuantizer
from hqq.core.quantize import *
from sinq.sinqlinear import BaseQuantizeConfig

class Recipe(BaseRecipe):
    def __init__(self):
        super().__init__()
        # Assign quantizer and offloader objects.
        self.quantizer = SINQQuantizer
        self.offloader = None
        self.factorizer = None

    def generate_configurations(self, target_model, draft_model, max_length, cpu_offload_gb, dtype, device):
        # Quantization
        quant_cfg = BaseQuantizeConfig(
            nbits=4,            # quantization bit-width
            group_size=64,      # group size
            tiling_mode="1D",   # tiling strategy
            method="sinq"       # quantization method ("asinq" for the calibrated version)
        )

        # Configs
        target_config = {
            "quant_config": quant_cfg,
        }
        draft_config = None
        
        return target_config, draft_config