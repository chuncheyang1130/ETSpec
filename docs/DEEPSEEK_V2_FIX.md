# DeepSeek-V2(-Lite): why routed-functional compression collapsed, and the fix

Branch `moe-compress-deepseek-bug-fix`, based on `moe-compress` @ `c7b7da1`.

## Symptom

Applying the routed-functional compression (AUTH recipe: mean_loss planner, jacobian output
metric, router candidates 32, coding basis, η = 1.32) to DeepSeek-V2-Lite at b25 collapsed
generation: outputs did not terminate, looped, or contained no code.

All numbers below come from DeepSeek-V2-Lite-Chat on MBPP-dev100 (100 problems; seed 0;
T = 0.7, top-p 0.8, top-k 20; max_length 2048). At this size, ±1 SE is about 5 points.

## Root cause 1: removal semantics for routers without top-k renormalisation

DeepSeek-V2 uses `norm_topk_prob=False`: an expert's weight is its raw softmax probability
over all 64 routed experts. The compressed block removed experts by deleting their router
rows and taking the softmax over the retained rows only. For Qwen3 (`norm_topk_prob=True`)
this makes no difference, because the top-k renormalisation cancels the softmax
denominator. For DeepSeek it inflates every surviving weight by 1 / (1 − removed mass).
At b25 about 60% of the routed experts are removed.

| b25, same plan, removed experts handled by … | pass@1 | overlength | repetition | TF-KL vs base |
|---|---|---|---|---|
| re-softmax over retained experts (old) | 1 | 43 % | 47 % | 0.50 |
| mask after the full softmax, raw probabilities (fix) | 29 | 4 % | 6 % | 0.24 |
| base model (uncompressed) | 38 | 1 % | 1 % | – |

REAP pruning at b25 shows the same effect: 0 with the old semantics, 9 with the fix. The
planner cannot see the problem: on the real removal sets, the old semantics produce
1.2–1.4× the layer's routed-output amplitude in damage, while the planner predicts about
0.55.

## Root cause 2: missing YaRN softmax factor in native transformers DeepseekV2

DeepSeek's reference implementation (`modeling_deepseek.py`, remote code) scales attention
logits by `qk_head_dim**-0.5 * mscale**2`, with
`mscale = 0.1 * mscale_all_dim * ln(factor) + 1`. For V2-Lite (factor 40,
mscale_all_dim 0.707) that is a ×1.590 factor.

transformers' native `DeepseekV2Attention` uses `qk_head_dim**-0.5` only. This is true of
both 4.57.1 and 5.14.1 (checked in the source: `self.scaling = self.qk_head_dim ** (-0.5)`).
`DeepseekV3Attention` in the same releases does apply the factor.

The effect on the base model:
- KL(remote ‖ native) per token is 0.10–0.20 without the factor and 0.005–0.024 with it.
  The residual is the remote code's bf16 RoPE cache.
- MBPP-dev100 for the base model drops from 38 to 26 without the factor.
- With b25 compression on the unfixed model, overlength rises to 13–21% (pass@1 12–14).

The repo's flow needs transformers ≥ 4.54, so it runs DeepSeek through the native class and
gets this bug.

## Not the cause: the removal-damage model

The planner priced removal for `norm_topk_prob=False` as a re-softmax over the survivors.
The exact single-removal model under the fixed semantics is δ = p_r·f_r − p_e·f_e, using raw
probabilities.

On 6 layers of held-out tokens, all three candidate models were compared: the Qwen3 renorm
formula, the old branch, and the exact one.
- They rank removal sets almost identically: Spearman 0.98–0.99 against realized damage.
- Single-removal prices differ by 3–5%.
- At b25 the plans share about 95% of their actions and have the same TF-KL (0.2426 vs
  0.2429).
- pass@1 for these plans differs only within noise (22–29).

The exact model is included here as a correctness fix (separate commit). It does not
change results measurably.

## What this branch changes

| commit | change |
|---|---|
| feat: DeepSeek-V2 MoE adapter and YaRN fix | `deepseek_v2_adapter.py`: moves each layer's routed experts into `mlp.moe_block`. It exposes the Qwen3 shape: `gate` linear, per-expert views, `norm_topk_prob=False`, fp32 router. Shared experts and dense layer 0 stay untouched. Works on transformers 4 and 5 (fused experts) and is bit-identical to HF. `fix_deepseek_v2_yarn_softmax_scale` is gated on yarn rope scaling with `mscale_all_dim`, idempotent, and a no-op once transformers applies the factor. It is called by the adapter and after loading in `builder.py`. `moe_arch` recognises the block; `prepare_moe_blocks` dispatches to it. |
| fix: removal semantics for `norm_topk_prob=False` | `moe_arch.retained_topk_routing`: softmax over ALL experts, removed experts masked after the softmax, top-k among the retained, weights = original probabilities. `norm_topk_prob=True` is unchanged. Used by `Qwen3MoeStackedBlock` (subset mode), `Qwen3MoeHybridSVDBlock` (incl. `removed_policy=drop`), the profile block and `factor_calibration_selection`. `topk_routing(..., num_experts=)` raises on a subset re-softmax, and the old branch is now an assertion. Router logits are computed in fp32 where the source router does. |
| fix: exact removal-damage model | Streaming trace: δ = p_r·f_r − p_e·f_e for `norm_topk_prob=False`, fp32 router logits, and a guard against restricted routing pools for such routers. |
| fix: class alias | `Qwen3MoeRoutedFunctionalSVDSubsetBlock = Qwen3MoeHybridSVDBlock`. At c7b7da1 the planner, two tools and three tests still import the old name, so the planner fails to import. |
| feat: calibration tool | `tools/calibrate_deepseek_v2.py`: moments, sequence cache and hidden cache in the existing formats. It uses HF `generate`, because `NaiveGenerator`'s static cache does not handle MLA's 192/128 K/V head dims. |
| test | `tests/test_deepseek_v2.py` (toy-router semantics, guard, tiny-model adapter and all-FULL identity, real-checkpoint identity), plus one updated assertion in `test_moe_routed_functional_svd.py` for the new semantics. |

## How to run

```bash
# tests (CPU; the real-checkpoint test runs when DEEPSEEK_V2_PATH is set and a GPU is free)
pytest tests/test_deepseek_v2.py
DEEPSEEK_V2_PATH=/path/to/DeepSeek-V2-Lite-Chat pytest tests/test_deepseek_v2.py -k real

# calibration (256 OpenHermes coding prompts, greedy 256 new tokens)
python tools/calibrate_deepseek_v2.py --model $DSV2 --out-dir expert_activation/dsv2_coding256

# plan, b25 (K scaled from Qwen3's 51/128 to 26/64)
C=expert_activation/dsv2_coding256
python tools/plan_routed_functional_compression.py --model $DSV2 \
  --moments $C/moments.pt --sequence-cache $C/sequences.pt --hidden-cache-dir $C/hidden_states \
  --reuse-hidden-cache --trace-dir $C/traces --planner mean_loss --trace-mode streaming \
  --rank-pairs 128:128 256:256 384:384 --factor-output-metric jacobian --router-candidates 32 \
  --basis-category coding --memory-ratio 0.25 --min-layer-memory-ratio 0.245133 \
  --min-retained-experts 26 --tier-price 1.32 --out plan_dsv2_b25.json
```

## Test results on this branch

Hardware for the GPU test: RTX A6000.

- `tests/test_deepseek_v2.py`: 9 passed on transformers 4.57.1. The 8 CPU tests also pass
  on 5.14.1, where the fused-experts adapter path and `rope_parameters` are exercised.
- Real checkpoint:
  - the adapter is bit-identical to HF;
  - the all-FULL restructured block differs per layer by 0.42% relative (bf16 routing
    weights and a different accumulation order);
  - end-to-end KL is 0.001–0.017 with top-1 agreement 0.95–1.0 on three prompts.
- Existing tests: `test_moe_routed_functional_svd`, `test_routed_functional_compression`,
  `test_activation_svd`, `test_selfgen_calibration` and `test_ordinary_svd` give 131 passed,
  with one assertion updated for the new semantics.

## Known limitations

- The MBPP / TF-KL numbers above were produced with our earlier port of this fix on top of
  `589e28c`, using a standalone HF-generate eval harness. They were not re-run on this
  branch; only the tests above were.
- Not verified for DeepSeek: the `run.main` eval path with `cache_implementation: static`
  and `compile_mode: max-autotune`. The static KV cache with MLA is untested.
- Not verified on our hardware: the planner's trace build and the grouped Triton forward.
  The kernels' tiles need about 135–160 KB of shared memory, and our RTX 6000 Ada / A6000
  have 99 KB. The real-checkpoint test therefore uses the hybrid block's eager forward on
  such GPUs. This limitation is not DeepSeek-specific.
- The adapter supports greedy top-k with `routed_scaling_factor = 1` only (V2-Lite). The
  full DeepSeek-V2 (group-limited greedy) is not covered.
- `routing_penalty` (retired) is rejected for `norm_topk_prob=False` blocks. The HQQ INT4
  subclass keeps bf16 router logits.
- `tests/test_routed_svd_hqq.py` already fails to import at c7b7da1
  (`Qwen3MoeRoutedFunctionalSVDInt4Block` renamed). Not touched here.
