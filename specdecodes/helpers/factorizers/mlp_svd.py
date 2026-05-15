from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from specdecodes.models.utils.llama_modeling import SVD_LlamaMLP 

class MLPSharedBasisSVDFactorizer:
    """Factorize each layer's MLP gate/up projections with one shared basis."""

    @classmethod
    def factorize_model(
        cls,
        model: Any,
        svd_config: Optional[Dict[str, Any]],
        compute_dtype: Any,
        device: str,
    ) -> None:

        if not svd_config:
            return

        rank_ratio = float(svd_config.get("rank_ratio", 0.5))
        # rank = int(model.config.hidden_size * rank_ratio)
        rank = 2560
        
        svd_device = svd_config.get("svd_device", "cuda:0")
        
        core_model = model if hasattr(model, "layers") else model.model
        layers = getattr(core_model, "layers", None)

        if layers is None:
            logging.warning("[SVD-MLP] model.layers not found; skipping.")
            return
        
        for layer in tqdm(layers):
            new_mlp = cls._build_svd_llama_mlp(
                orig_mlp=layer.mlp,
                rank=rank,
                svd_device=svd_device,
            )

            layer.mlp = new_mlp

    @classmethod
    def _build_svd_llama_mlp(
        cls,
        orig_mlp: nn.Module,
        rank: int,
        svd_device: Optional[str] = "cuda:0",
    ) -> Optional[SVD_LlamaMLP]:
        
        gate_proj_weight = orig_mlp.gate_proj.weight
        up_proj_weight = orig_mlp.up_proj.weight
        
        u_mat, vh_mat = cls._build_shared_basis(
            gate_weight=gate_proj_weight,
            up_weight=up_proj_weight,
            rank=rank,
            svd_device=svd_device,
        )

        return SVD_LlamaMLP(
            orig_mlp=orig_mlp,
            u_gate_up_weight=u_mat,
            vh_gate_up_weight=vh_mat,
        )

    @classmethod
    def _build_shared_basis(
        cls,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        rank: int,
        svd_device: str,
    ):

        out_dtype = gate_weight.dtype
        target_device = gate_weight.device

        stacked = torch.cat(
            [
                gate_weight.detach().to(device=svd_device, dtype=torch.float32),
                up_weight.detach().to(device=svd_device, dtype=torch.float32),
            ],
            dim=0,
        )

        try:
            U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        except RuntimeError:
            logging.exception("[SVD-MLP] SVD failed for one layer; skipping.")
            return None
        
        # logging.info(f"[SVD-MLP] Original weight shape: {stacked.shape}, U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}, rank: {rank}")
        U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
        S_sqrt = torch.diag(torch.sqrt(S_r))
        U_scaled, Vh_scaled = U_r @ S_sqrt, S_sqrt @ Vh_r

        U_scaled, Vh_scaled = U_scaled.to(device=target_device, dtype=out_dtype), Vh_scaled.to(device=target_device, dtype=out_dtype)
        U_scaled.requires_grad = False
        Vh_scaled.requires_grad = False

        return U_scaled, Vh_scaled