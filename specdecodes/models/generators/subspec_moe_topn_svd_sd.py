"""Generator for MoE TopN-SVD speculative decoding.

`MoESvdSDGenerator` inherits the picker / tracker / refresh / logging loop
from `MoeTopNSubsetSDGeneratorBase`. The two SVD-specific bits live here:

  * `_config_attr` — points at `topn_svd_config` on the draft model
    (the no-SVD base reads from `topn_subset_config`).
  * `_materialize_kwargs` — injects `svd_device` (where the actual SVD
    decomposition runs) into the `materialize_kept_from_target` call.

The actual SVD decomposition itself happens inside the draft block's
overridden `_materialize_expert_weights` template hook (see
`qwen3_moe_topn_svd.py`). Because `PackedTopNSvdMoeBlock` is a subclass
of `PackedTopNMoeBlock`, the base generator's isinstance walks match SVD
blocks transparently.

Build time: the draft's MoE blocks are swapped for `PackedTopNSvdMoeBlock`
instances (random init).

Generate time: see `subspec_moe_topn_sd.py` — same picker / tracker /
refresh / logging loop. Only the materialize step differs (SVD fit vs.
weight copy), and that lives inside the block class.
"""

from typing import Any, Dict

from .subspec_moe_topn_sd import MoeTopNSubsetSDGeneratorBase
from ..utils.mixin import SDProfilingMixin


class MoESvdSDGeneratorBase(MoeTopNSubsetSDGeneratorBase):
    """Top-N + shared-basis-SVD speculative-decoding generator.

    Differs from the no-SVD base only in:
      * config-dict attribute name (`topn_svd_config`).
      * `svd_device` forwarded down into the materialize call so the
        actual SVD math runs on the right device.
    """

    _config_attr = "topn_svd_config"

    def _materialize_kwargs(self) -> Dict[str, Any]:
        return {"svd_device": str(self._config().get("svd_device", "cuda:0"))}


class MoESvdSDGenerator(SDProfilingMixin, MoESvdSDGeneratorBase):
    pass
