"""CPU tests for the unified grouped-matmul forward.

The GMM kernels themselves are Triton and need a GPU (`test_moe_gmm_parity.py` and
`test_moe_int4_gmm_parity.py` cover those). What is pinned here is the contract
around them:

  1. every live grouped-MoE block takes `forward` from the ONE `GroupedMoEForward`
     mixin and overrides all four hooks it calls. bf16, INT4 and INT2 used to carry
     equivalent copies of that forward, so a regression here means a copy crept back.
  2. the forward wires the four steps together in the documented order and shapes —
     testable without a GPU by implementing the hooks with plain tensors.
"""

import torch

from specdecodes.models.utils.moe.base.grouped_moe import GroupedMoEForward
from specdecodes.models.utils.moe.base.qwen3_moe_profile import Qwen3MoeProfileBlock
from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from specdecodes.models.utils.moe.hqq.qwen3_moe_stacked_int2 import Qwen3MoeStackedInt2Block
from specdecodes.models.utils.moe.hqq.qwen3_moe_stacked_int4 import Qwen3MoeStackedInt4Block

HOOKS = ("_routing_weights", "_grouped_matmul_metadata",
         "_gmm_gate_up_silu", "_gmm_down_reduce")

# The blocks that actually run in recorded experiments: bf16 target/vanilla, and the
# two precision subclasses that swap the kernels but not the pipeline.
GROUPED_BLOCKS = (Qwen3MoeStackedBlock, Qwen3MoeStackedInt4Block, Qwen3MoeStackedInt2Block)


def test_one_forward_serves_every_grouped_block():
    for cls in GROUPED_BLOCKS:
        assert GroupedMoEForward in cls.__mro__, cls.__name__
        assert cls.forward is GroupedMoEForward.forward, (
            f"{cls.__name__} defines its own forward again"
        )


def test_grouped_blocks_supply_every_hook_the_forward_calls():
    """Nothing reaches the mixin's NotImplementedError stubs at runtime."""
    for cls in GROUPED_BLOCKS:
        for hook in HOOKS:
            assert getattr(cls, hook) is not getattr(GroupedMoEForward, hook), (
                f"{cls.__name__} never overrides {hook}"
            )


def test_profile_block_keeps_its_plain_torch_forward():
    """The analysis block must NOT inherit the Triton forward — it runs on CPU."""
    assert Qwen3MoeProfileBlock.forward is not GroupedMoEForward.forward


class _RecordingBlock(GroupedMoEForward, torch.nn.Module):
    """Hook implementations in plain torch, recording what the forward passed."""

    def __init__(self, num_experts=4, top_k=2, hidden=3, interm=5):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.hidden, self.interm = hidden, interm
        self.seen = {}

    def _routing_weights(self, x):
        self.seen["route_shape"] = tuple(x.shape)
        T = x.shape[0]
        weights = torch.full((T, self.top_k), 0.5)
        ids = torch.zeros(T, self.top_k, dtype=torch.long)
        return weights, ids

    def _grouped_matmul_metadata(self, topk_ids):
        self.seen["meta_shape"] = tuple(topk_ids.shape)
        return "active", "offsets", "sorted"

    def _gmm_gate_up_silu(self, x_flat, active, token_offsets, sorted_token_ids, k):
        self.seen["gate_up"] = (tuple(x_flat.shape), active, token_offsets,
                                sorted_token_ids, k)
        return torch.zeros(x_flat.shape[0] * k, self.interm)

    def _gmm_down_reduce(self, interm, active, token_offsets, sorted_token_ids,
                         routing_weights, T, k):
        self.seen["down"] = (tuple(interm.shape), active, token_offsets,
                             sorted_token_ids, tuple(routing_weights.shape), T, k)
        return torch.arange(T * self.hidden, dtype=torch.float32).view(T, self.hidden)


@torch.no_grad()
def test_forward_wires_the_four_steps_in_order():
    block = _RecordingBlock()
    bsz, seq_len = 2, 3
    T, k, H = bsz * seq_len, block.top_k, block.hidden

    out = block(torch.zeros(bsz, seq_len, H))

    # Leading dims are flattened for the kernels and restored on the way out.
    assert block.seen["route_shape"] == (T, H)
    assert out.shape == (bsz, seq_len, H)
    # Routing ids drive the metadata, whose three outputs reach BOTH GMM hooks as-is.
    assert block.seen["meta_shape"] == (T, k)
    assert block.seen["gate_up"][1:] == ("active", "offsets", "sorted", k)
    assert block.seen["down"][1:4] == ("active", "offsets", "sorted")
    # gate/up emits one row per (token, expert) slot; down reduces back to tokens.
    assert block.seen["gate_up"][0] == (T, H)
    assert block.seen["down"][0] == (T * k, block.interm)
    # The weights are flattened to [T*k] to match that row order, and T/k forwarded.
    assert block.seen["down"][4] == (T * k,)
    assert block.seen["down"][5:] == (T, k)
    # The view is a reshape, not a copy of something reordered.
    assert torch.equal(out.reshape(T, H), torch.arange(T * H, dtype=torch.float32).view(T, H))


@torch.no_grad()
def test_forward_handles_a_single_token_step():
    """Decode calls it with seq_len=1; the flatten/restore must survive that."""
    block = _RecordingBlock()
    out = block(torch.zeros(1, 1, block.hidden))
    assert block.seen["route_shape"] == (1, block.hidden)
    assert block.seen["down"][5:] == (1, block.top_k)
    assert out.shape == (1, 1, block.hidden)
