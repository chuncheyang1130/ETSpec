"""DeepSeek-V2 support: routing semantics for norm_topk_prob=False routers, the adapter,
the YaRN softmax-scale fix, and all-FULL identity.

CPU tests use toy routers and a tiny randomly initialised DeepseekV2 model. The
real-model identity test runs only when DEEPSEEK_V2_PATH points at a
DeepSeek-V2(-Lite) checkpoint and a GPU is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from specdecodes.models.utils.moe.base.moe_arch import retained_topk_routing, topk_routing
from specdecodes.models.utils.moe.base.qwen3_moe_routed_functional_svd import (
    Qwen3MoeHybridSVDBlock,
    factor_calibration_selection,
)


# --------------------------------------------------------------------------- toy router
def _toy(seed=0, tokens=64, experts=16, hidden=24):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=g) * 2.0
    retained = torch.sort(torch.randperm(experts, generator=g)[:10]).values
    return logits, retained


def test_retained_routing_raw_probabilities():
    logits, retained = _toy()
    weights, slots = retained_topk_routing(logits, retained, 4, norm_topk_prob=False)
    probs = F.softmax(logits, dim=-1)
    chosen = retained[slots]
    # weights are the ORIGINAL probabilities of the chosen experts
    torch.testing.assert_close(weights, probs.gather(1, chosen))
    # removed experts never selected; the chosen are the top-4 retained by probability
    masked = probs.clone()
    removed = torch.ones(logits.shape[1], dtype=torch.bool)
    removed[retained] = False
    masked[:, removed] = -1.0
    expected = torch.topk(masked, 4, dim=-1).indices
    assert torch.equal(torch.sort(chosen, -1).values, torch.sort(expected, -1).values)
    # no renormalisation: the removed mass is lost, so weights are strictly below
    # what a re-softmax over the survivors would give
    resoftmax = F.softmax(logits.index_select(1, retained), dim=-1).gather(1, slots)
    assert bool((weights < resoftmax).all())


def test_retained_routing_renormalised_unchanged():
    logits, retained = _toy(1)
    weights, slots = retained_topk_routing(logits, retained, 4, norm_topk_prob=True)
    values = logits.index_select(1, retained).gather(1, slots)
    torch.testing.assert_close(weights, F.softmax(values, dim=-1))


def test_topk_routing_refuses_subset_resoftmax():
    logits, retained = _toy(2)
    with pytest.raises(ValueError):
        topk_routing(logits.index_select(1, retained), 4, norm_topk_prob=False, num_experts=logits.shape[1])
    topk_routing(logits, 4, norm_topk_prob=False, num_experts=logits.shape[1])          # full pool: fine
    topk_routing(logits.index_select(1, retained), 4, norm_topk_prob=True, num_experts=logits.shape[1])


def test_factor_calibration_weights_raw_probabilities():
    logits, retained = _toy(3)
    mask = torch.zeros(logits.shape[1], dtype=torch.bool)
    mask[retained] = True
    ids, w = factor_calibration_selection(logits, factor_calibration="topk_g1", top_k=4,
                                          norm_topk_prob=False, router_candidates=8, retained=mask)
    assert bool(mask[ids].all())
    torch.testing.assert_close(w, F.softmax(logits, dim=-1).gather(1, ids))


def _toy_full_block(norm_topk_prob, experts=16, hidden=24, inter=12, top_k=4, seed=4):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        gate_proj_stacked=torch.randn(experts, inter, hidden, generator=g) / 4,
        up_proj_stacked=torch.randn(experts, inter, hidden, generator=g) / 4,
        down_proj_stacked=torch.randn(experts, hidden, inter, generator=g) / 4,
        router_weights=torch.randn(experts, hidden, generator=g),
        top_k=top_k, norm_topk_prob=norm_topk_prob, router_fp32=True,
    )


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_hybrid_block_removal_semantics(norm_topk_prob):
    full = _toy_full_block(norm_topk_prob)
    _, retained = _toy(5)
    block = Qwen3MoeHybridSVDBlock.from_factor_state(
        full, full_ids=retained.tolist(), tier_specs=[], factor_state={"tiers": []})
    x = torch.randn(1, 32, 24, generator=torch.Generator().manual_seed(6))
    y = block(x)[0]                                          # CPU -> eager forward

    flat = x[0]
    logits = flat @ full.router_weights.T
    w, slots = retained_topk_routing(logits, retained, full.top_k, norm_topk_prob=norm_topk_prob)
    ids = retained[slots]
    ref = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(full.top_k):
            e = int(ids[t, j])
            h = F.silu(full.gate_proj_stacked[e] @ flat[t]) * (full.up_proj_stacked[e] @ flat[t])
            ref[t] += w[t, j] * (full.down_proj_stacked[e] @ h)
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------- tiny DeepseekV2
def _tiny_deepseek_v2():
    transformers = pytest.importorskip("transformers")
    try:
        from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
    except ImportError:
        pytest.skip("transformers has no native DeepseekV2")
    rope = {"type": "yarn", "rope_type": "yarn", "factor": 40.0, "mscale": 0.707, "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0}
    kwargs = dict(vocab_size=128, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
                  num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=8,
                  n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1, kv_lora_rank=16,
                  q_lora_rank=None, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8,
                  norm_topk_prob=False, topk_method="greedy", routed_scaling_factor=1.0, n_group=1,
                  topk_group=1, max_position_embeddings=4096 * 40, rope_theta=10000.0)
    if int(transformers.__version__.split(".")[0]) >= 5:
        cfg = DeepseekV2Config(rope_parameters={**rope, "rope_theta": 10000.0}, **kwargs)
    else:
        cfg = DeepseekV2Config(rope_scaling=rope, **kwargs)
    torch.manual_seed(0)
    return DeepseekV2ForCausalLM(cfg).eval()


@torch.no_grad()
def test_yarn_softmax_scale_fix_is_gated_and_idempotent():
    from specdecodes.models.utils.moe.base.deepseek_v2_adapter import fix_deepseek_v2_yarn_softmax_scale
    model = _tiny_deepseek_v2()
    attn = [m for m in model.modules() if type(m).__name__ == "DeepseekV2Attention"]
    base = attn[0].qk_head_dim ** -0.5
    m = 0.1 * 0.707 * math.log(40.0) + 1.0
    if not math.isclose(attn[0].scaling, base):
        pytest.skip("this transformers version already applies the YaRN softmax factor")
    assert math.isclose(fix_deepseek_v2_yarn_softmax_scale(model), m * m, rel_tol=1e-9)
    assert all(math.isclose(a.scaling, base * m * m, rel_tol=1e-9) for a in attn)
    assert fix_deepseek_v2_yarn_softmax_scale(model) == 1.0                    # second call: no-op
    assert all(math.isclose(a.scaling, base * m * m, rel_tol=1e-9) for a in attn)


def _all_full_factor_dir(model, tmp_path):
    names = [n for n, mod in model.named_modules() if type(mod).__name__ == "DeepseekV2MoEAdapterBlock"]
    experts = model.get_submodule(names[0]).num_experts
    plan = {"meta": {"basis_category": "coding"}, "layers": {
        n: {"by_action": {"removed": [], "full": list(range(experts))},
            "expert_actions": {str(e): "full" for e in range(experts)}} for n in names}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    fdir = tmp_path / "factors"
    fdir.mkdir()
    layers = {}
    for n in names:
        fn = n.replace(".", "_") + ".pt"
        torch.save({"tiers": []}, fdir / fn)
        layers[n] = fn
    sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    (fdir / "metadata.json").write_text(json.dumps({"plan_sha256": sha, "layers": layers}))
    return str(plan_path), str(fdir), names


@torch.no_grad()
def test_tiny_adapter_and_all_full_identity(tmp_path):
    from specdecodes.helpers.restructurer.moe_routed_functional_svd import MoERoutedFunctionalSVDRestructurer
    from specdecodes.models.utils.moe.base.gemma4_adapter import prepare_moe_blocks
    from specdecodes.models.utils.moe.base.deepseek_v2_adapter import fix_deepseek_v2_yarn_softmax_scale
    model = _tiny_deepseek_v2()
    fix_deepseek_v2_yarn_softmax_scale(model)
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))
    ref = model(ids).logits
    assert prepare_moe_blocks(model) == 2                   # layers 1, 2 (layer 0 is dense)
    assert torch.equal(model(ids).logits, ref)              # the adapter is bit-identical
    plan_path, fdir, _ = _all_full_factor_dir(model, tmp_path)
    MoERoutedFunctionalSVDRestructurer.restructure_model(
        model, {"plan_path": plan_path, "factors_dir": fdir}, torch.float32, "cpu")
    assert any(type(m).__name__ == "Qwen3MoeHybridSVDBlock" for m in model.modules())
    torch.testing.assert_close(model(ids).logits, ref, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------- real checkpoint
@pytest.mark.skipif(not (os.environ.get("DEEPSEEK_V2_PATH") and torch.cuda.is_available()),
                    reason="set DEEPSEEK_V2_PATH to a DeepSeek-V2(-Lite) checkpoint (needs one ~40 GB GPU)")
@torch.no_grad()
def test_real_model_all_full_identity(tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from specdecodes.helpers.restructurer.moe_routed_functional_svd import MoERoutedFunctionalSVDRestructurer
    from specdecodes.models.utils.moe.base.deepseek_v2_adapter import fix_deepseek_v2_yarn_softmax_scale
    from specdecodes.models.utils.moe.base.gemma4_adapter import prepare_moe_blocks

    path = os.environ["DEEPSEEK_V2_PATH"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    fix_deepseek_v2_yarn_softmax_scale(model)
    prompts = ["Write a Python function that checks whether a number is prime.",
               "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"",
               "Explain the difference between a list and a tuple in Python."]
    batches = [torch.tensor([tok.apply_chat_template([{"role": "user", "content": p}],
                                                     add_generation_prompt=True, tokenize=True)], device="cuda:0")
               for p in prompts]
    ref = [model(b, use_cache=False).logits.float() for b in batches]

    prepare_moe_blocks(model)
    for b, r in zip(batches, ref):                           # the adapter is bit-identical
        assert torch.equal(model(b, use_cache=False).logits.float(), r)
    probe_layers = (1, 13, len(model.model.layers) - 1)
    x = torch.randn(1, 256, model.config.hidden_size, device="cuda:0", dtype=torch.bfloat16,
                    generator=torch.Generator(device="cuda:0").manual_seed(0))
    layer_ref = {i: model.model.layers[i].mlp.moe_block(x).float() for i in probe_layers}

    plan_path, fdir, _ = _all_full_factor_dir(model, tmp_path)
    MoERoutedFunctionalSVDRestructurer.restructure_model(
        model, {"plan_path": plan_path, "factors_dir": fdir}, torch.bfloat16, "cuda:0")
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "shared_memory_per_block_optin", 1 << 30) < 160 * 1024:
        # The grouped Triton kernels' tiles need ~135-160 KB shared memory; on smaller
        # GPUs check the same block through its eager forward.
        for m in model.modules():
            if type(m).__name__ == "Qwen3MoeHybridSVDBlock":
                m.forward = m._eager_forward
    # Per layer: same routing and experts; the block rounds routing weights to bf16
    # and accumulates in a different order than HF, so ~0.5 % relative, not bitwise.
    # A routing/semantics bug shows up as tens of percent.
    for i in probe_layers:
        y = model.model.layers[i].mlp.moe_block(x).float()
        rel = float((y - layer_ref[i]).norm() / layer_ref[i].norm())
        print(f"layer {i}: rel {rel:.4f}")
        assert rel < 0.01, (i, rel)
    for b, r in zip(batches, ref):
        out = model(b, use_cache=False).logits.float()
        lp, lq = torch.log_softmax(r, -1), torch.log_softmax(out, -1)
        kl = float((lp.exp() * (lp - lq)).sum(-1).mean())
        agree = float((out.argmax(-1) == r.argmax(-1)).float().mean())
        print(f"KL {kl:.4f} top-1 {agree:.3f} positions {out.shape[1]}")
        assert kl < 0.05 and agree >= 0.9, (kl, agree)
