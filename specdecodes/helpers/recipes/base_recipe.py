import logging
from typing import Any, Dict, Optional, Tuple


class BaseRecipe:
    """Base recipe for quantization, factorization (SVD), and offloading transforms."""

    def __init__(self):
        self.quantizer = None
        self.factorizer = None
        self.offloader = None

    def generate_configurations(
        self,
        target_model: Any,
        draft_model: Any,
        max_length: int,
        cpu_offload_gb: Optional[int],
        dtype: Any,
        device: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}

    def apply_quantization(self, model: Any, quant_config: Dict[str, Any], dtype: Any, device: str):
        if self.quantizer is None:
            logging.info("No quantizer provided; skipping quantization.")
            return
        self.quantizer.quantize_model(model, quant_config, dtype, device)

    def apply_svd(self, model: Any, svd_config: Dict[str, Any], dtype: Any, device: str):
        if self.factorizer is None:
            logging.info("No factorizer provided; skipping SVD.")
            return

        self.factorizer.factorize_model(model, svd_config, dtype, device)

    def apply_offloading(self, model: Any, device_map: Dict[str, Any], draft_model: Any = None):
        if self.offloader is None:
            logging.info("No offloader provided; skipping offloading.")
            return
        return self.offloader(model, device_map=device_map, draft_model=draft_model)
