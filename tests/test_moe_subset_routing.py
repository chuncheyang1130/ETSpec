"""CPU tests for retained-expert routing semantics.

Subset routing must select top-k after restricting the router to retained experts.
When every full-router winner was pruned, the next-best retained experts must be
promoted with nonzero normalized weights.
"""

import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.qwen3_moe_stacked import Qwen3MoeStackedBlock
from specdecodes.models.utils.moe.base.qwen3_moe_subset import Qwen3MoeSubsetBlock
from specdecodes.models.utils.moe.base.qwen3_moe_profile import Qwen3MoeProfileBlock


def _router_fixture():
    # For x=1 the full expert ranking is 0 > 1 > 2 > 3 > 4. Use a deliberately
    # shuffled retained order so the test also checks local-slot renumbering:
    # local slot 0 -> global 4, slot 1 -> global 2, slot 2 -> global 3.
    router = torch.tensor([[10.0], [9.0], [8.0], [7.0], [6.0]])
    kept_ids = torch.tensor([4, 2, 3], dtype=torch.long)
    x = torch.ones(1, 1)
    expected_weights = F.softmax(torch.tensor([[8.0, 7.0]]), dim=-1)
    return router, kept_ids, x, expected_weights


@torch.no_grad()
def test_compact_subset_promotes_next_retained_experts():
    router, kept_ids, x, expected_weights = _router_fixture()
    block = Qwen3MoeSubsetBlock(
        hidden_size=1,
        intermediate_size=1,
        num_experts=5,
        top_k=2,
        norm_topk_prob=True,
        kept=3,
        dtype=torch.float32,
        device="cpu",
    )
    block.router_weights.copy_(router)
    block.selected_expert_ids.copy_(kept_ids)

    weights, local_slots = block._routing_weights(x)

    assert local_slots.shape == (1, block.top_k)
    assert local_slots.tolist() == [[1, 2]]
    assert torch.all(weights > 0)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


@torch.no_grad()
def test_compact_subset_routing_is_torch_compilable():
    router, kept_ids, x, expected_weights = _router_fixture()
    block = Qwen3MoeSubsetBlock(
        hidden_size=1,
        intermediate_size=1,
        num_experts=5,
        top_k=2,
        norm_topk_prob=True,
        kept=3,
        dtype=torch.float32,
        device="cpu",
    )
    block.router_weights.copy_(router)
    block.selected_expert_ids.copy_(kept_ids)

    compiled_routing = torch.compile(
        block._routing_weights, backend="eager", fullgraph=True
    )
    weights, local_slots = compiled_routing(x)

    assert local_slots.tolist() == [[1, 2]]
    assert torch.allclose(weights, expected_weights)


@torch.no_grad()
def test_shared_store_subset_preserves_global_expert_ids():
    router, kept_ids, x, expected_weights = _router_fixture()
    block = Qwen3MoeStackedBlock(
        hidden_size=1,
        intermediate_size=1,
        num_experts=5,
        top_k=2,
        norm_topk_prob=True,
        kept=3,
        owns_store=False,
        dtype=torch.float32,
        device="cpu",
    )
    block.router_weights.copy_(router)
    block.selected_expert_ids.copy_(kept_ids)

    weights, global_ids = block._routing_weights(x)

    assert global_ids.shape == (1, block.top_k)
    assert global_ids.tolist() == [[2, 3]]
    assert torch.all(weights > 0)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


@torch.no_grad()
def test_profile_subset_matches_serving_routing():
    router, kept_ids, x, expected_weights = _router_fixture()
    block = Qwen3MoeProfileBlock(
        hidden_size=1,
        intermediate_size=1,
        num_experts=5,
        top_k=2,
        norm_topk_prob=True,
        kept=3,
        owns_store=False,
        dtype=torch.float32,
        device="cpu",
    )
    block.router_weights.copy_(router)
    block.selected_expert_ids.copy_(kept_ids)

    probs = block._routing_probs(x)

    assert torch.count_nonzero(probs[0]).item() == block.top_k
    assert torch.nonzero(probs[0], as_tuple=False).flatten().tolist() == [2, 3]
    assert torch.allclose(probs[:, [2, 3]], expected_weights)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


@torch.no_grad()
def test_subset_rejects_pool_smaller_than_topk():
    try:
        Qwen3MoeSubsetBlock(
            hidden_size=1,
            intermediate_size=1,
            num_experts=5,
            top_k=4,
            norm_topk_prob=True,
            kept=2,
            dtype=torch.float32,
            device="cpu",
        )
    except ValueError as exc:
        assert "exactly top_k" in str(exc)
    else:
        raise AssertionError("expected a retained pool smaller than top_k to be rejected")


@torch.no_grad()
def test_dynamic_subset_rejects_duplicate_experts():
    block = Qwen3MoeStackedBlock(
        hidden_size=1,
        intermediate_size=1,
        num_experts=5,
        top_k=2,
        norm_topk_prob=True,
        kept=3,
        owns_store=False,
        dtype=torch.float32,
        device="cpu",
    )
    try:
        block.set_kept(torch.tensor([1, 1, 2]))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("expected duplicate retained expert ids to be rejected")


@torch.no_grad()
def test_dynamic_subset_rejects_pool_smaller_than_topk():
    block = Qwen3MoeStackedBlock(
        hidden_size=1,
        intermediate_size=1,
        num_experts=5,
        top_k=2,
        norm_topk_prob=True,
        kept=3,
        owns_store=False,
        dtype=torch.float32,
        device="cpu",
    )
    try:
        block.set_kept(torch.tensor([3]))
    except ValueError as exc:
        assert "top_k=2" in str(exc)
    else:
        raise AssertionError("expected a retained pool smaller than top_k to be rejected")
