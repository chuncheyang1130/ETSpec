"""Correctness tests for balanced ordinary low-rank factors.

`ordinary_svd.calibrated_balanced_low_rank_factors` is used by the live routed
functional SVD block (`qwen3_moe_routed_functional_svd.py`). These two tests were
split out of the former `test_moe_hybrid_svd.py` when the hybrid MoE recipe,
restructurer and block were removed (2026-10-14); the hybrid-specific tests went
with them.
"""

import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.ordinary_svd import (
    balance_low_rank_factors,
    calibrated_balanced_low_rank_factors,
)


@torch.no_grad()
def test_balancing_exactly_reparameterizes_low_rank_product():
    torch.manual_seed(3)
    left = torch.randn(7, 4) * 20
    right = torch.randn(4, 6) * 0.03
    balanced_left, balanced_right = balance_low_rank_factors(left, right)
    expected = left @ right
    actual = balanced_left @ balanced_right
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-4)
    left_norm = left.norm()
    right_norm = right.norm()
    assert abs(float(torch.log(left_norm / right_norm))) > 2
    assert abs(float(torch.log(balanced_left.norm() / balanced_right.norm()))) < 0.1


@torch.no_grad()
def test_calibrated_balanced_factors_apply_directly():
    torch.manual_seed(4)
    weight = torch.randn(7, 6)
    moment = torch.rand(6).add_(0.2)
    inputs = torch.randn(11, 6)
    left, right = calibrated_balanced_low_rank_factors(weight, moment, rank=6)
    expected = F.linear(inputs, weight)
    actual = F.linear(F.linear(inputs, right), left)
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-4)
