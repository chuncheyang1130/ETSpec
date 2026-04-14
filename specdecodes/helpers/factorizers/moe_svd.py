from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FACTOR_DEVICE = torch.device("cuda:0")


@dataclass
class _LinearRef:
    name: str
    module: nn.Linear


@dataclass
class _MoEBlockRef:
    name: str
    module: nn.Module


class SharedVhLinear(nn.Module):
    """Linear module that stores the shared right-singular basis."""

    def __init__(self, v_hat: torch.Tensor, freeze_params: bool):
        super().__init__()
        self.v_hat = nn.Parameter(v_hat)
        if freeze_params:
            self.v_hat.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.v_hat)


class FactorizedLinear(nn.Module):
    """Linear module that consumes a shared Vh module and stores the U-side matrix."""

    def __init__(self, shared_vh: SharedVhLinear, u_mat: torch.Tensor, bias: Optional[torch.Tensor], freeze_params: bool):
        super().__init__()
        self.shared_vh = shared_vh
        self.u_mat = nn.Parameter(u_mat)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
        if freeze_params:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low_rank = self.shared_vh(x)
        return F.linear(x_low_rank, self.u_mat, self.bias)


class FactorizedMoEExpertGroup(nn.Module):
    """A group of experts sharing one Vh basis; each expert keeps its own U/down weights."""

    def __init__(
        self,
        shared_vh: SharedVhLinear,
        u_mat_gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        act_fn,
        freeze_params: bool,
    ):
        super().__init__()
        self.shared_vh = shared_vh
        self.u_mat_gate_up_proj = nn.Parameter(u_mat_gate_up_proj)
        self.down_proj = nn.Parameter(down_proj)
        self.act_fn = act_fn
        if freeze_params:
            for p in self.parameters():
                p.requires_grad = False

    def forward_expert(self, hidden_states: torch.Tensor, local_expert_idx: int) -> torch.Tensor:
        hidden_states_low_rank = self.shared_vh(hidden_states)
        gate, up = F.linear(hidden_states_low_rank, self.u_mat_gate_up_proj[local_expert_idx]).chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, self.down_proj[local_expert_idx])


class FactorizedMoEExperts(nn.Module):
    """Qwen3-MoE block that owns all experts with grouped shared Vh bases."""

    def __init__(
        self,
        groups: List[FactorizedMoEExpertGroup],
        expert_to_group: List[int],
        expert_to_local: List[int],
        num_experts: int,
        freeze_params: bool,
    ):
        super().__init__()
        self.groups = nn.ModuleList(groups)
        self.expert_to_group = expert_to_group
        self.expert_to_local = expert_to_local
        self.num_experts = int(num_experts)
        if freeze_params:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            group_idx = self.expert_to_group[expert_idx]
            local_idx = self.expert_to_local[expert_idx]
            current_hidden_states = self.groups[group_idx].forward_expert(hidden_states[token_idx], local_idx)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

class MoEMLPSharedBasisSVDFactorizer:
    """Apply shared-basis SVD over MoE expert MLP projections."""

    DEFAULT_MODULES_TO_FACTOR = [
        r"model\.layers\.\d+\.mlp\.gate_proj$",
        r"model\.layers\.\d+\.mlp\.up_proj$",
        r"model\.layers\.\d+\.mlp\.down_proj$",
        r"model\.layers\.\d+\.mlp\.experts\.gate_up_proj$",
        r".*experts\.\d+\.w1$",
        r".*experts\.\d+\.w2$",
        r".*experts\.\d+\.w3$",
        r".*experts\.\d+\.gate_proj$",
        r".*experts\.\d+\.up_proj$",
        r".*experts\.\d+\.down_proj$",
    ]

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

        rank = int(svd_config.get("rank", 64))
        expert_group_size = max(1, int(svd_config.get("expert_group_size", 4)))
        modules_to_factor = svd_config.get("modules_to_factor") or svd_config.get("include_patterns")
        if modules_to_factor is None:
            modules_to_factor = cls.DEFAULT_MODULES_TO_FACTOR
        freeze_params = bool(svd_config.get("freeze_params", True))
        svd_device = svd_config.get("svd_device", "cpu")
        verbose = bool(svd_config.get("verbose", True))

        compiled_patterns = [re.compile(pattern) for pattern in modules_to_factor]
        groups = cls._collect_groups(model, compiled_patterns)
        moe_block_refs = cls._collect_moe_blocks(model, compiled_patterns)
        if not groups and not moe_block_refs:
            logging.warning("[SVD-MoE] No modules matched modules_to_factor.")
            return

        if verbose:
            logging.info(
                "[SVD-MoE] Found %d linear groups and %d MoE blocks for fixed-rank factorization.",
                len(groups),
                len(moe_block_refs),
            )

        replaced = 0
        for group_key, refs in groups.items():
            shared_vh = cls._build_shared_vh(
                refs,
                rank=rank,
                freeze_params=freeze_params,
                svd_device=svd_device,
            )
            if shared_vh is None:
                continue

            for ref in refs:
                u_mat = cls._project_to_u_mat(ref.module.weight.data, shared_vh.v_hat)
                new_mod = FactorizedLinear(
                    shared_vh=shared_vh,
                    u_mat=u_mat,
                    bias=ref.module.bias.data if ref.module.bias is not None else None,
                    freeze_params=freeze_params,
                )
                cls._set_module_by_name(model, ref.name, new_mod)
                replaced += 1

            if verbose:
                logging.info(
                    "[SVD-MoE] Group %s: modules=%d, rank=%d",
                    group_key,
                    len(refs),
                    shared_vh.v_hat.shape[0],
                )

        for ref in moe_block_refs:
            gate_up_weight = ref.module.gate_up_proj.data
            down_proj = ref.module.down_proj.data
            num_experts = gate_up_weight.shape[0]

            group_modules: List[FactorizedMoEExpertGroup] = []
            expert_to_group = [-1] * num_experts
            expert_to_local = [-1] * num_experts
            group_failed = False

            for start_idx in range(0, num_experts, expert_group_size):
                end_idx = min(start_idx + expert_group_size, num_experts)
                group_gate_up_weight = gate_up_weight[start_idx:end_idx]
                group_down_proj = down_proj[start_idx:end_idx]

                group_shared_vh = cls._build_shared_vh_from_weights(
                    weights=group_gate_up_weight,
                    rank=rank,
                    freeze_params=freeze_params,
                    svd_device=svd_device,
                )
                if group_shared_vh is None:
                    group_failed = True
                    break

                group_u_mat_gate_up = cls._project_to_u_mat(group_gate_up_weight, group_shared_vh.v_hat)
                group_module = FactorizedMoEExpertGroup(
                    shared_vh=group_shared_vh,
                    u_mat_gate_up_proj=group_u_mat_gate_up,
                    down_proj=group_down_proj,
                    act_fn=ref.module.act_fn,
                    freeze_params=freeze_params,
                )

                group_idx = len(group_modules)
                group_modules.append(group_module)
                for expert_idx in range(start_idx, end_idx):
                    expert_to_group[expert_idx] = group_idx
                    expert_to_local[expert_idx] = expert_idx - start_idx

            if group_failed or any(idx < 0 for idx in expert_to_group) or any(idx < 0 for idx in expert_to_local):
                logging.warning("[SVD-MoE] Skipping MoE block %s because grouped SVD did not complete.", ref.name)
                continue

            new_mod = FactorizedMoEExperts(
                groups=group_modules,
                expert_to_group=expert_to_group,
                expert_to_local=expert_to_local,
                num_experts=num_experts,
                freeze_params=freeze_params,
            )
            cls._set_module_by_name(model, ref.name, new_mod)
            replaced += 1

            if verbose:
                logging.info(
                    "[SVD-MoE] MoE block %s: experts=%d, group_size=%d, groups=%d",
                    ref.name,
                    num_experts,
                    expert_group_size,
                    len(group_modules),
                )

        logging.info("[SVD-MoE] Replaced %d modules with shared-basis low-rank blocks.", replaced)

    @staticmethod
    def _matches(module_name: str, compiled_patterns: List[re.Pattern]) -> bool:
        return any(p.search(module_name) for p in compiled_patterns)

    @staticmethod
    def _group_key(module_name: str) -> str:
        return re.sub(r"\.(gate_proj|up_proj|w1|w3)$", ".gate_up_shared", module_name)

    @classmethod
    def _collect_groups(cls, model: nn.Module, compiled_patterns: List[re.Pattern]) -> Dict[str, List[_LinearRef]]:
        groups: Dict[str, List[_LinearRef]] = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not cls._matches(name, compiled_patterns):
                continue
            key = cls._group_key(name)
            groups.setdefault(key, []).append(_LinearRef(name=name, module=mod))
        return groups

    @classmethod
    def _collect_moe_blocks(
        cls, model: nn.Module, compiled_patterns: List[re.Pattern]
    ) -> List[_MoEBlockRef]:
        refs: List[_MoEBlockRef] = []
        for name, mod in model.named_modules():
            if not hasattr(mod, "gate_up_proj") or not hasattr(mod, "down_proj"):
                continue

            gate_up = getattr(mod, "gate_up_proj")
            down_proj = getattr(mod, "down_proj")
            if not isinstance(gate_up, torch.Tensor) or not isinstance(down_proj, torch.Tensor):
                continue
            if gate_up.dim() != 3 or down_proj.dim() != 3:
                continue

            synthetic_names = (
                f"{name}.gate_up_proj",
                f"{name}.gate_proj",
                f"{name}.up_proj",
            )
            if not any(cls._matches(n, compiled_patterns) for n in synthetic_names):
                continue

            refs.append(_MoEBlockRef(name=name, module=mod))
        return refs

    @classmethod
    def _build_shared_vh(
        cls,
        refs: List[_LinearRef],
        rank: int,
        freeze_params: bool,
        svd_device: str,
    ) -> Optional[SharedVhLinear]:
        if not refs:
            return None

        first = refs[0].module
        in_features = first.in_features
        out_dtype = first.weight.dtype
        stacked = torch.cat(
            [r.module.weight.detach().to(device=svd_device, dtype=torch.float32) for r in refs],
            dim=0,
        )
        return cls._build_shared_vh_from_stacked(
            stacked=stacked,
            in_features=in_features,
            rank=rank,
            out_dtype=out_dtype,
            freeze_params=freeze_params,
            error_context="one group",
        )

    @classmethod
    def _build_shared_vh_from_weights(
        cls,
        weights: torch.Tensor,
        rank: int,
        freeze_params: bool,
        svd_device: str,
    ) -> Optional[SharedVhLinear]:
        if weights.dim() < 2:
            return None

        in_features = weights.shape[-1]
        out_dtype = weights.dtype
        stacked = weights.detach().to(device=svd_device, dtype=torch.float32).reshape(-1, in_features)
        return cls._build_shared_vh_from_stacked(
            stacked=stacked,
            in_features=in_features,
            rank=rank,
            out_dtype=out_dtype,
            freeze_params=freeze_params,
            error_context="one expert group",
        )

    @classmethod
    def _build_shared_vh_from_stacked(
        cls,
        stacked: torch.Tensor,
        in_features: int,
        rank: int,
        out_dtype: torch.dtype,
        freeze_params: bool,
        error_context: str,
    ) -> Optional[SharedVhLinear]:
        try:
            _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
        except RuntimeError:
            logging.exception("[SVD-MoE] SVD failed for %s; skipping.", error_context)
            return None

        resolved_rank = min(max(1, int(rank)), in_features, vh.shape[0])
        v_hat = vh[:resolved_rank, :].to(device=FACTOR_DEVICE, dtype=out_dtype)
        return SharedVhLinear(v_hat=v_hat, freeze_params=freeze_params)

    @staticmethod
    def _project_to_u_mat(weight: torch.Tensor, v_hat: nn.Parameter) -> torch.Tensor:
        return (weight.to(torch.float32) @ v_hat.data.to(torch.float32).transpose(0, 1)).to(
            device=FACTOR_DEVICE,
            dtype=weight.dtype,
        )

    @staticmethod
    def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
        parts = module_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
