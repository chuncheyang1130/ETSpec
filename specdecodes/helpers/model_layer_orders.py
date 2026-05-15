
# order of layers in the model
def get_llama_layer_order(model_config):
    layers = []
    layers.append('model.embed_tokens')
    layers.append('model.rotary_emb')
    for i in range(model_config.num_hidden_layers):
        layers.append(f'model.layers.{i}.input_layernorm')
        layers.append(f'model.layers.{i}.self_attn.q_proj')
        layers.append(f'model.layers.{i}.self_attn.k_proj')
        layers.append(f'model.layers.{i}.self_attn.v_proj')
        layers.append(f'model.layers.{i}.self_attn.o_proj')
        layers.append(f'model.layers.{i}.post_attention_layernorm')
        layers.append(f'model.layers.{i}.mlp.gate_proj')
        layers.append(f'model.layers.{i}.mlp.up_proj')
        layers.append(f'model.layers.{i}.mlp.down_proj')
    layers.append('model.norm')
    layers.append('lm_head')
    return layers

def get_qwen_layer_order(model_config):
    layers = []
    layers.append('model.embed_tokens')
    layers.append('model.rotary_emb')
    for i in range(model_config.num_hidden_layers):
        layers.append(f'model.layers.{i}.input_layernorm')
        layers.append(f'model.layers.{i}.self_attn.q_proj')
        layers.append(f'model.layers.{i}.self_attn.k_proj')
        layers.append(f'model.layers.{i}.self_attn.v_proj')
        layers.append(f'model.layers.{i}.self_attn.o_proj')
        layers.append(f'model.layers.{i}.post_attention_layernorm')
        layers.append(f'model.layers.{i}.mlp.gate_proj')
        layers.append(f'model.layers.{i}.mlp.up_proj')
        layers.append(f'model.layers.{i}.mlp.down_proj')
    layers.append('model.norm')
    layers.append('lm_head')
    return layers

def get_qwen3_layer_order(model_config):
    layers = []
    layers.append('model.embed_tokens')
    layers.append('model.rotary_emb')
    for i in range(model_config.num_hidden_layers):
        layers.append(f'model.layers.{i}.input_layernorm')
        layers.append(f'model.layers.{i}.self_attn.q_proj')
        layers.append(f'model.layers.{i}.self_attn.q_norm')
        layers.append(f'model.layers.{i}.self_attn.k_proj')
        layers.append(f'model.layers.{i}.self_attn.k_norm')
        layers.append(f'model.layers.{i}.self_attn.v_proj')
        layers.append(f'model.layers.{i}.self_attn.o_proj')
        layers.append(f'model.layers.{i}.post_attention_layernorm')
        layers.append(f'model.layers.{i}.mlp.gate_proj')
        layers.append(f'model.layers.{i}.mlp.up_proj')
        layers.append(f'model.layers.{i}.mlp.down_proj')
    layers.append('model.norm')
    layers.append('lm_head')
    return layers

def get_qwen3_moe_layer_order(model_config):
    layers = []
    layers.append('model.embed_tokens')
    layers.append('model.rotary_emb')

    num_experts = int(getattr(model_config, 'num_experts', 0) or 0)
    decoder_sparse_step = int(getattr(model_config, 'decoder_sparse_step', 1) or 1)
    mlp_only_layers = set(getattr(model_config, 'mlp_only_layers', []) or [])

    for i in range(model_config.num_hidden_layers):
        layers.append(f'model.layers.{i}.input_layernorm')
        layers.append(f'model.layers.{i}.self_attn.q_proj')
        layers.append(f'model.layers.{i}.self_attn.q_norm')
        layers.append(f'model.layers.{i}.self_attn.k_proj')
        layers.append(f'model.layers.{i}.self_attn.k_norm')
        layers.append(f'model.layers.{i}.self_attn.v_proj')
        layers.append(f'model.layers.{i}.self_attn.o_proj')
        layers.append(f'model.layers.{i}.post_attention_layernorm')

        # Mirrors `Qwen3MoeDecoderLayer.__init__`'s choice between
        # `Qwen3MoeSparseMoeBlock` and `Qwen3MoeMLP` for `self.mlp`.
        is_moe_layer = (
            i not in mlp_only_layers
            and num_experts > 0
            and (i + 1) % decoder_sparse_step == 0
        )

        if is_moe_layer:
            # Sparse MoE block: a router Linear (`gate`) + a ModuleList of
            # per-expert MLPs, each with its own gate/up/down projections.
            layers.append(f'model.layers.{i}.mlp.gate')
            for e in range(num_experts):
                layers.append(f'model.layers.{i}.mlp.experts.{e}.gate_proj')
                layers.append(f'model.layers.{i}.mlp.experts.{e}.up_proj')
                layers.append(f'model.layers.{i}.mlp.experts.{e}.down_proj')
        else:
            # Dense MLP layer (no router, no experts).
            layers.append(f'model.layers.{i}.mlp.gate_proj')
            layers.append(f'model.layers.{i}.mlp.up_proj')
            layers.append(f'model.layers.{i}.mlp.down_proj')

    layers.append('model.norm')
    layers.append('lm_head')
    return layers


MODEL_TYPE_GET_LAYER_ORDER = {
    "gemma": None,
    "gemma2": None,
    "llama": get_llama_layer_order,
    "granite": None,
    "mllama": None,
    "mllama_text_model": None,
    "mistral": None,
    "mixtral": None,
    "qwen2": get_qwen_layer_order,
    "qwen2_vl": None,
    "qwen3": get_qwen3_layer_order,
    "qwen3_moe": get_qwen3_moe_layer_order,
    "phi3": None,
}