from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import torch
from specdecodes.models.utils.utils import DraftParams

@dataclass
class AppConfig:
    # Base configurations
    method: str = "classic_sd"
    vram_limit_gb: Optional[int] = None
    seed: int = 0
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    
    # Model paths
    llm_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Token that actually ends an assistant turn. Defaults to the
    # tokenizer's own `eos_token`, which is right for Qwen3/Llama. Gemma 4
    # reports `<eos>` there, but its chat template closes assistant turns
    # with `<turn|>` -- so without this override nothing ever stops.
    eos_token: Optional[str] = None
    draft_model_path: Optional[str] = None
    
    # Generation parameters
    max_length: int = 2048
    do_sample: bool = False
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    
    # Generator-specific configurations
    generator_kwargs: Dict[str, Any] = field(default_factory=dict)
    draft_params: Optional[DraftParams] = None
    
    # Recipe
    recipe: Any = None
    cpu_offload_gb: Optional[int] = None
    
    # Additional configurations
    cache_implementation: Union[str, Dict[str, Optional[str]]] = "dynamic"
    warmup_iter: int = 0
    compile_mode: Optional[Union[str, Dict[str, Optional[str]]]] = None
    # torch.compile `dynamic` for the target model. False (default) specializes every
    # Python int -- right for Qwen3, whose static caches report positions as
    # tensors. Gemma 4's sliding-window cache layers report an int that changes every
    # token (transformers >= 5), so they need None: compile once specialized, then once
    # generic, instead of recompiling per token until the recompile limit.
    compile_dynamic: Optional[bool] = False
    
    # Profiling
    generator_profiling: bool = True
    profiling_verbose: bool = True
    print_time: bool = True
    print_message: bool = True
    
    # Benchmarking/logging
    out_dir: Optional[str] = None
    log_dir: str = "experiments"

    # Settings snapshot (resolved config + CLI context)
    config_path: Optional[str] = None
    settings_snapshot: Optional[Dict[str, Any]] = None

    # Research toggles (set via YAML/CLI)
    detailed_analysis: bool = False
    nvtx_profiling: bool = False
    nsys_output: str = "nsight_report"

    def update(self, new_config: Dict[str, Any]):
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
