"""CPU tests for fixed-prompt, response-only self-calibration."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from specdecodes.helpers.recipes.moe.expert_calibration import (
    SELFGEN_PROMPTS,
    load_calibration_prompts,
    run_calibration,
    run_response_only_calibration,
)
from specdecodes.models.utils.moe.base.expert_usage_tracker import (
    get_expert_usage,
    install_expert_usage_tracker,
    remove_expert_usage_tracker,
    reset_expert_usage,
    set_tracker_token_mask,
)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Minimal block whose class name intentionally matches the tracker heuristic."""

    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.top_k = 1
        self.gate = nn.Linear(2, 2, bias=False)
        self.experts = nn.ModuleList([_ToyExpert(), _ToyExpert()])
        with torch.no_grad():
            self.gate.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))

    def forward(self, hidden_states):
        return hidden_states


class _ToyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)


class _ToyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def __init__(self):
        self.thinking_flags = []

    def apply_chat_template(self, messages, **kwargs):
        self.thinking_flags.append(kwargs.get("enable_thinking"))
        return torch.tensor([[1, 3]], dtype=torch.long)


class _ToyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 2)
        self.mlp = Qwen3MoeSparseMoeBlock()
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[1] = torch.tensor([1.0, 0.0])
            self.embed.weight[3] = torch.tensor([1.0, 0.0])
            self.embed.weight[5] = torch.tensor([1.0, 0.0])
            self.embed.weight[6] = torch.tensor([-1.0, 0.0])

    def forward(self, input_ids, **kwargs):
        hidden = self.mlp(self.embed(input_ids))
        return SimpleNamespace(logits=torch.zeros(
            *hidden.shape[:-1], 8, device=hidden.device
        ))

    def generate(self, input_ids, **kwargs):
        response = torch.tensor([[5, 6, 2]], device=input_ids.device)
        return torch.cat([input_ids, response], dim=1)


class _FakeNaiveGenerator:
    """Small stand-in that exposes the sequence-position contract of generation."""

    def __init__(self, target_model, **kwargs):
        self.target_model = target_model
        target_model.prefill_forward = target_model.forward

    def generate(self, input_ids, **kwargs):
        response = torch.tensor([[5, 6]], device=input_ids.device)
        return torch.cat([input_ids, response], dim=1)

    def _chunked_prefill_forward(
        self, input_ids, past_key_values, *, prefill_chunk_size, **kwargs
    ):
        outputs = None
        for start in range(0, input_ids.shape[1], int(prefill_chunk_size)):
            outputs = self.target_model(input_ids[:, start:start + int(prefill_chunk_size)])
        return outputs


class _FakeDynamicCache:
    pass


@torch.no_grad()
def test_tracker_token_mask_preserves_context_but_counts_selected_positions_only():
    model = _ToyCausalModel()
    install_expert_usage_tracker(model)
    reset_expert_usage(model)
    ids = torch.tensor([[5, 6, 5]])
    set_tracker_token_mask(model, torch.tensor([[False, True, False]]))
    model(ids)
    mass = get_expert_usage(model)["mlp"]
    remove_expert_usage_tracker(model)

    assert torch.equal(mass, torch.tensor([0.0, 1.0]))


@torch.no_grad()
def test_response_only_calibration_excludes_prompt_and_special_tokens():
    model = _ToyCausalModel()
    tokenizer = _ToyTokenizer()
    signals, stats = run_response_only_calibration(
        model,
        tokenizer,
        ["a prompt"],
        gen_tokens=3,
        device="cpu",
        continuations_per_prompt=2,
        track_activation=True,
    )

    # Each continuation contributes token 5 -> expert 0 and token 6 -> expert 1.
    # Prompt positions and the generated EOS token are excluded.
    assert torch.equal(signals["mass"]["mlp"], torch.tensor([2.0, 2.0]))
    assert torch.equal(signals["count"]["mlp"], torch.tensor([2.0, 2.0]))
    assert torch.isfinite(signals["importance"]["mlp"]).all()
    assert torch.isfinite(signals["avg"]["mlp"]).all()
    assert stats["selfgen_sequences"] == 2
    assert stats["selfgen_response_tokens"] == 4
    assert tokenizer.thinking_flags == [False]


def test_selfgen_prompt_bank_is_fixed_and_bounded_by_n():
    assert len(SELFGEN_PROMPTS) == 16
    assert load_calibration_prompts("selfgen", 3) == SELFGEN_PROMPTS[:3]
    assert load_calibration_prompts("selfgen", 100) == SELFGEN_PROMPTS


@torch.no_grad()
def test_fast_replay_tracks_generation_inputs_except_final_sample(monkeypatch):
    monkeypatch.setattr(
        "specdecodes.models.generators.naive.NaiveGenerator", _FakeNaiveGenerator
    )
    monkeypatch.setattr(
        "specdecodes.models.utils.cache_utils.create_kv_cache",
        lambda *args, **kwargs: _FakeDynamicCache(),
    )
    model = _ToyCausalModel()
    tokenizer = _ToyTokenizer()

    signals = run_calibration(
        model,
        tokenizer,
        ["a prompt"],
        gen_tokens=2,
        device="cpu",
        cache_implementation="dynamic",
        fast_replay=True,
        track_activation=True,
    )

    # Prompt [1, 3] and generated token 5 route to expert 0. Final token 6 would
    # route to expert 1, but generation stops before that token is consumed.
    assert signals["count"]["mlp"].sum().item() == 3
    assert signals["count"]["mlp"][0].item() == 3
    assert tokenizer.thinking_flags == [False]
