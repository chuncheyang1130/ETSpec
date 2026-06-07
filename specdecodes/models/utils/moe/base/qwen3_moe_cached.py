"""
Two-tier LRU-cached (offload-friendly) MoE block — reuses the base GMM kernels.

Offload counterpart of `Qwen3MoeContiguousMoeBlock`. Instead of keeping all `E`
experts resident on the GPU, it keeps a **CPU master** of every expert plus a
fixed-capacity GPU **pool** of `C < E` slots, split into two tiers:

  * **Hot / pinned tier** — slots ``[0, num_pinned)``. Holds the draft's kept
    experts; **never evicted**. The shared-weight draft block aliases the pool
    and reads these slots directly (slot ``j`` holds the ``j``-th kept expert),
    so the draft is load-free and its weight pointers stay stable for CUDA
    graphs. Refreshed in place via `set_hot_tier`.
  * **Cold / streaming tier** — slots ``[num_pinned, C)``. An LRU cache for the
    experts only the verification pass needs; eviction is confined here so it
    never touches the pinned tier.

On forward it routes (full top-k, identical to the contiguous block), then:
  * runs **one GMM "hot pass"** over the touched pinned experts (no loads), and
  * streams the touched cold experts through the cold tier in **waves** of at
    most ``C - num_pinned`` at a time, one GMM per wave.
Each (token, expert) assignment is computed exactly once (hot pass or a single
cold wave); the down-projection `atomic_add`s into per-pass `[T, H]` buffers
which are summed, so the result equals the exact MoE output.

``num_pinned = 0`` degenerates to a single-tier LRU pool over the whole `C`
slots — the original behaviour (and the `vanilla` baseline) byte-for-byte.

Why the kernels need no changes
-------------------------------
`triton_fused_gate_up_gmm_silu` / `triton_fused_down_gmm_reduction` index the
weight tensors and `token_offsets` purely by the entries of `active_experts`
(`w_gate_ptr + expert_id * stride_w_gate_e`). They don't care whether that index
is a global expert id or a pool slot id, so we remap routed expert-ids -> slot
ids and build the metadata over the `C`-slot space.

Notes / scope
-------------
  * Compute dtype is **bf16** (the reused base GMM kernels are bf16). FP8 master
    + dequant-on-load is a separate follow-up (`master_dtype` stays separable).
  * **Eager-only**: residency resolution is data-dependent (`torch.unique`,
    dict LRU, conditional H->D copies). The *target* is eager anyway. The draft
    that aliases the pinned tier stays graph-friendly because the pool tensor
    identity and its `selected_expert_ids` are stable; only pinned-slot
    *contents* change, in place, outside any captured region (via `set_hot_tier`).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_fused_gate_up_gmm_silu import triton_fused_gate_up_gmm_silu
from .triton_fused_down_gmm_reduction import triton_fused_down_gmm_reduction
from .qwen3_moe_contiguous import _is_qwen3_moe_block, get_grouped_matmul_metadata
from .qwen3_moe_topn import _set_module_by_name

import gc
from tqdm.auto import tqdm


def _alloc_cpu(shape, dtype: torch.dtype, pin: bool) -> torch.Tensor:
    """Allocate a CPU tensor, falling back to non-pinned if pinning fails/OOMs."""
    if pin:
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError as exc:  # pinned (page-locked) allocation can fail
            logging.warning(
                "[Cached-MoE] pin_memory alloc failed (%s); falling back to pageable.",
                exc,
            )
    return torch.empty(shape, dtype=dtype, device="cpu")


class Qwen3MoeCachedMoeBlock(nn.Module):
    """Offload MoE block: CPU expert master + two-tier (pinned + LRU) GPU pool.

    Drop-in for the target's MoE block. Routing matches
    `Qwen3MoeContiguousMoeBlock`; only the expert *storage* differs.

    Attributes
    ----------
    gate_pool : [C, IM, H]   resident gate weights (bf16, GPU)  slots [0,P)=hot, [P,C)=cold
    up_pool   : [C, IM, H]   resident up   weights (bf16, GPU)
    down_pool : [C, H, IM]   resident down weights (bf16, GPU)
    gate_cpu/up_cpu/down_cpu : [E, ...] CPU master (master_dtype, optionally pinned)
    router_weights : [E, H]  (buffer, follows the model to GPU)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        capacity: int,
        num_pinned: int = 0,
        compute_dtype: torch.dtype = torch.bfloat16,
        master_dtype: Optional[torch.dtype] = None,
        device: torch.device | str = "cuda",
        pin_memory: bool = True,
    ):
        super().__init__()

        if not (1 <= capacity <= num_experts):
            raise ValueError(
                f"capacity ({capacity}) must be in [1, num_experts={num_experts}]."
            )
        if not (0 <= num_pinned <= capacity):
            raise ValueError(
                f"num_pinned ({num_pinned}) must be in [0, capacity={capacity}]."
            )
        if compute_dtype != torch.bfloat16:
            # The reused base GMM kernels are bf16-only (see module docstring).
            raise ValueError(
                f"compute_dtype must be torch.bfloat16 for the base GMM kernels, got {compute_dtype}."
            )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.norm_topk_prob = bool(norm_topk_prob)

        self.capacity = int(capacity)
        self.num_pinned = int(num_pinned)
        self.cold_capacity = self.capacity - self.num_pinned
        self.compute_dtype = compute_dtype
        self.master_dtype = master_dtype or compute_dtype
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory)

        # `capacity` is the sentinel slot id used to mark out-of-pass assignments
        # (a row the kernels never touch because it is excluded from active slots).
        self._sentinel = self.capacity

        # --- GPU pool (resident experts). Plain attributes so module `.to(...)`
        # ---  never relocates or duplicates them; we manage device explicitly.
        self.gate_pool = torch.empty(
            self.capacity, self.intermediate_size, self.hidden_size,
            dtype=compute_dtype, device=self.device,
        )
        self.up_pool = torch.empty(
            self.capacity, self.intermediate_size, self.hidden_size,
            dtype=compute_dtype, device=self.device,
        )
        self.down_pool = torch.empty(
            self.capacity, self.hidden_size, self.intermediate_size,
            dtype=compute_dtype, device=self.device,
        )

        # --- CPU master (all experts). Filled by `from_huggingface`.
        self.gate_cpu: Optional[torch.Tensor] = None
        self.up_cpu: Optional[torch.Tensor] = None
        self.down_cpu: Optional[torch.Tensor] = None

        # --- Router (small; lives on GPU with the model).
        self.register_buffer(
            "router_weights",
            torch.zeros(self.num_experts, self.hidden_size, dtype=compute_dtype, device=self.device),
            persistent=False,
        )

        # --- Tier bookkeeping (CPU-side python state).
        #     `_pinned`: gid -> slot in [0, num_pinned)            (never evicted)
        #     `_cold`  : gid -> slot in [num_pinned, capacity), LRU(front)..MRU(back)
        self._pinned: Dict[int, int] = {}
        self._cold: "OrderedDict[int, int]" = OrderedDict()
        self._free_cold: List[int] = list(range(self.num_pinned, self.capacity))
        self._slot_expert: List[int] = [-1] * self.capacity

        self.reset_stats()

    # ------------------------------------------------------------------ build

    @classmethod
    def from_huggingface(
        cls,
        hf_block,
        capacity: int,
        num_pinned: int = 0,
        pin_memory: bool = True,
        device: Optional[torch.device | str] = None,
        master_dtype: Optional[torch.dtype] = None,
    ) -> "Qwen3MoeCachedMoeBlock":
        """Build a cached block from an HF `Qwen3MoeSparseMoeBlock`.

        Streams every expert into the CPU master (releasing HF sources), copies
        the router to GPU, and leaves the pool empty (populated on demand /
        `set_hot_tier`).
        """
        sample_w = hf_block.experts[0].gate_proj.weight
        src_device, src_dtype = sample_w.device, sample_w.dtype
        intermediate_size, hidden_size = sample_w.shape[0], sample_w.shape[1]
        num_experts = int(hf_block.num_experts)
        top_k = int(hf_block.top_k)
        norm_topk_prob = bool(hf_block.norm_topk_prob)

        dev = torch.device(device) if device is not None else src_device
        m_dtype = master_dtype or src_dtype

        block = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            capacity=capacity,
            num_pinned=num_pinned,
            compute_dtype=torch.bfloat16,
            master_dtype=m_dtype,
            device=dev,
            pin_memory=pin_memory,
        )

        with torch.no_grad():
            router_module = getattr(hf_block, "gate", None) or getattr(hf_block, "router", None)
            if router_module is None:
                raise AttributeError("hf_block exposes neither .gate nor .router for the MoE router.")
            block.router_weights.data.copy_(
                router_module.weight.detach().to(device=dev, dtype=torch.bfloat16)
            )

            block.gate_cpu = _alloc_cpu((num_experts, intermediate_size, hidden_size), m_dtype, pin_memory)
            block.up_cpu = _alloc_cpu((num_experts, intermediate_size, hidden_size), m_dtype, pin_memory)
            block.down_cpu = _alloc_cpu((num_experts, hidden_size, intermediate_size), m_dtype, pin_memory)

            for e, expert in enumerate(hf_block.experts):
                block.gate_cpu[e].copy_(expert.gate_proj.weight.data.to(m_dtype))
                block.up_cpu[e].copy_(expert.up_proj.weight.data.to(m_dtype))
                block.down_cpu[e].copy_(expert.down_proj.weight.data.to(m_dtype))
                expert.gate_proj.weight = None
                expert.up_proj.weight = None
                expert.down_proj.weight = None

        return block

    # ------------------------------------------------------------------ tiers

    def reset_stats(self) -> None:
        self.stats: Dict[str, int] = {
            "forwards": 0, "waves": 0, "lookups": 0,
            "hits": 0, "misses": 0, "loads": 0, "evictions": 0,
        }

    def pool_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate_pool, up_pool, down_pool) for a shared draft to alias."""
        return self.gate_pool, self.up_pool, self.down_pool

    @torch.no_grad()
    def set_hot_tier(self, global_ids: Sequence[int]) -> None:
        """Pin `global_ids` into the hot tier: slot ``j`` <- expert ``global_ids[j]``.

        In-place H->D copies into the stable pinned slots ``[0, num_pinned)`` so
        a draft aliasing the pool reads the kept experts in order. Any of these
        experts currently in the cold tier are released back to the cold pool.
        """
        ids = [int(e) for e in global_ids]
        if len(ids) != self.num_pinned:
            raise ValueError(
                f"set_hot_tier expects num_pinned={self.num_pinned} ids, got {len(ids)}."
            )
        for j, e in enumerate(ids):
            self.gate_pool[j].copy_(self.gate_cpu[e], non_blocking=True)
            self.up_pool[j].copy_(self.up_cpu[e], non_blocking=True)
            self.down_pool[j].copy_(self.down_cpu[e], non_blocking=True)
            cold_slot = self._cold.pop(e, None)
            if cold_slot is not None:
                self._free_cold.append(cold_slot)
                self._slot_expert[cold_slot] = -1
            self._slot_expert[j] = e
        self._pinned = {e: j for j, e in enumerate(ids)}

    @torch.no_grad()
    def _ensure_cold_resident(self, wave: Sequence[int]) -> None:
        """Make every (non-pinned) expert in `wave` resident in the cold tier."""
        protected = set(wave)
        for e in wave:
            self.stats["lookups"] += 1
            slot = self._cold.get(e)
            if slot is not None:
                self._cold.move_to_end(e)
                self.stats["hits"] += 1
                continue
            self.stats["misses"] += 1
            slot = self._free_cold.pop() if self._free_cold else self._evict_cold(protected)
            self.gate_pool[slot].copy_(self.gate_cpu[e], non_blocking=True)
            self.up_pool[slot].copy_(self.up_cpu[e], non_blocking=True)
            self.down_pool[slot].copy_(self.down_cpu[e], non_blocking=True)
            self._cold[e] = slot
            self._slot_expert[slot] = e
            self.stats["loads"] += 1

    def _evict_cold(self, protected: set) -> int:
        """Evict the LRU cold expert not in `protected`; return its freed slot."""
        for victim in list(self._cold.keys()):  # LRU-first
            if victim in protected:
                continue
            slot = self._cold.pop(victim)
            self._slot_expert[slot] = -1
            self.stats["evictions"] += 1
            return slot
        raise RuntimeError(
            "Cached-MoE: no evictable cold slot — cold working set exceeds cold capacity."
        )

    @staticmethod
    def _partition(experts: List[int], chunk: int) -> List[List[int]]:
        """Chunk `experts` into groups of at most `chunk` (>=1)."""
        if not experts:
            return []
        if chunk <= 0:
            raise RuntimeError(
                "Cached-MoE: cold tier has 0 slots (num_pinned == capacity) but "
                "verification routed to a non-pinned expert."
            )
        return [experts[i : i + chunk] for i in range(0, len(experts), chunk)]

    @torch.no_grad()
    def prefetch(self, expert_ids: Sequence[int]) -> None:
        """Warm the cold tier with `expert_ids` (external prefetch policy hook)."""
        ids = [int(e) for e in expert_ids if int(e) not in self._pinned][-self.cold_capacity :]
        if ids:
            self._ensure_cold_resident(ids)

    # ---------------------------------------------------------------- routing

    def _routing_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identical to the contiguous block: top-k logits -> (weights, indices)."""
        routing_logits = F.linear(x, self.router_weights)
        topk_vals, topk_indices = torch.topk(routing_logits, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_probs = F.softmax(topk_vals, dim=-1, dtype=torch.float32)
        else:
            global_softmax = F.softmax(routing_logits, dim=-1, dtype=torch.float32)
            topk_probs = torch.gather(global_softmax, -1, topk_indices)
        return topk_probs.to(x.dtype), topk_indices

    def _remap(self, topk_indices: torch.Tensor, experts: List[int], slots: List[int]) -> torch.Tensor:
        """Map `experts` -> their `slots` for this pass; everything else -> sentinel."""
        dev = topk_indices.device
        slot_map = torch.full((self.num_experts,), self._sentinel, dtype=torch.long, device=dev)
        if experts:
            slot_map[torch.tensor(experts, dtype=torch.long, device=dev)] = torch.tensor(
                slots, dtype=torch.long, device=dev
            )
        return slot_map[topk_indices]

    def _run_slot_gmm(
        self, x_flat: torch.Tensor, slot_idx: torch.Tensor, routing_weights_flat: torch.Tensor, T: int
    ) -> torch.Tensor:
        """One GMM over the resident pool for the assignments mapped to real slots."""
        active_slots, token_offsets, sorted_token_ids = get_grouped_matmul_metadata(
            topk_indices=slot_idx,
            num_experts=self.capacity + 1,  # +1 sentinel bin
        )
        active_slots = active_slots[active_slots < self.capacity]
        if active_slots.numel() == 0:
            return torch.zeros((T, self.hidden_size), dtype=self.compute_dtype, device=x_flat.device)
        interm = triton_fused_gate_up_gmm_silu(
            x_flat, self.gate_pool, self.up_pool,
            active_slots, token_offsets, sorted_token_ids, top_k=self.top_k,
        )
        return triton_fused_down_gmm_reduction(
            interm, self.down_pool,
            active_slots, token_offsets, sorted_token_ids,
            routing_weights=routing_weights_flat, T=T, top_k=self.top_k,
        )

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        T = x_flat.shape[0]

        topk_weights, topk_indices = self._routing_weights(x_flat)
        routing_weights_flat = topk_weights.reshape(-1)

        distinct = torch.unique(topk_indices).tolist()
        self.stats["forwards"] += 1
        out = torch.zeros((T, hidden), dtype=self.compute_dtype, device=x_flat.device)

        # Hot pass: pinned experts are always resident — one GMM, zero loads.
        pinned_touched = [e for e in distinct if e in self._pinned]
        if pinned_touched:
            self.stats["waves"] += 1
            self.stats["lookups"] += len(pinned_touched)
            self.stats["hits"] += len(pinned_touched)
            slot_idx = self._remap(topk_indices, pinned_touched, [self._pinned[e] for e in pinned_touched])
            out = out + self._run_slot_gmm(x_flat, slot_idx, routing_weights_flat, T)

        # Cold passes: stream the tail through the cold tier in waves.
        cold_touched = [e for e in distinct if e not in self._pinned]
        for wave in self._partition(cold_touched, self.cold_capacity):
            self.stats["waves"] += 1
            self._ensure_cold_resident(wave)
            slot_idx = self._remap(topk_indices, wave, [self._cold[e] for e in wave])
            out = out + self._run_slot_gmm(x_flat, slot_idx, routing_weights_flat, T)

        return out.view(bsz, seq_len, hidden)

    # ------------------------------------------------------------- introspect

    def get_expert_weight(self, eid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate, up, down) for expert `eid` from the CPU master (CPU tensors).

        Used by the shared draft's redirect-footprint computation and by
        `_read_target_expert_weight` when the target is a cached block.
        """
        e = int(eid)
        return self.gate_cpu[e], self.up_cpu[e], self.down_cpu[e]

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, capacity={self.capacity}, "
            f"num_pinned={self.num_pinned}, top_k={self.top_k}, hidden={self.hidden_size}, "
            f"intermediate={self.intermediate_size}, master_dtype={self.master_dtype}"
        )


# ---------------------------------------------------------------------------
# Build-time replacement
# ---------------------------------------------------------------------------
def apply_cached_moe_block_to_qwen_moe(
    model: nn.Module,
    capacity: int,
    num_pinned: int = 0,
    pin_memory: bool = True,
    device: Optional[torch.device | str] = None,
) -> int:
    """Replace every HF `Qwen3MoeSparseMoeBlock` with a `Qwen3MoeCachedMoeBlock`."""
    block_to_replace = [
        (name, module)
        for name, module in model.named_modules()
        if _is_qwen3_moe_block(module)
    ]

    for name, hf_block in tqdm(block_to_replace, desc="Replacing MoE blocks with two-tier expert cache"):
        new_block = Qwen3MoeCachedMoeBlock.from_huggingface(
            hf_block, capacity=capacity, num_pinned=num_pinned,
            pin_memory=pin_memory, device=device,
        )
        _set_module_by_name(model, name, new_block)
        del hf_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    logging.info(
        "[Cached-MoE] Replaced %d MoE blocks (capacity=%d, num_pinned=%d).",
        len(block_to_replace), int(capacity), int(num_pinned),
    )
    return len(block_to_replace)
