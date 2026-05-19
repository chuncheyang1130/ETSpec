"""CUDA-graph capture variant of `ExpSpecSDDraftModel`.

Identical to the eager base in build / materialize / routing — the only
difference is the per-step tree forward. After `init_cuda_graph_runner`
has been called, every `speculate_once` invocation copies fresh inputs
into pre-allocated static buffers and replays the captured graph
instead of running eager. The prefill / first-speculate forward
(`input_len=1`) stays eager because its shape differs from the
captured `[1, topk_len]` shape.

The capture itself is block-agnostic: it just records `self(...)` once,
so the packed top-N block's soft-redirect routing (via `redirect_P`) is
captured along with the rest of the forward.

Capture pre-conditions (caller's responsibility):
  * Draft `past_key_values` is set (`set_past_key_values`) and uses a
    **static** cache implementation with a known `max_cache_len`.
  * The draft has been materialized at least once (kept_ids picked,
    packed expert tensors filled, `redirect_P` built).
    `materialize_from_target` later only does in-place copies into the
    same Parameters (including `redirect_P`), so subsequent
    re-materialization does **not** invalidate the captured graph.

Out-of-scope / known limits:
  * Re-binding `past_key_values` to a different cache instance after
    capture invalidates the graph (the captured kernels hold the old
    pointers). Reuse the same cache across requests, or rebuild the
    runner.
  * `compile_mode.draft` should be set to `null` in the recipe for
    this draft — stacking torch.compile's autograph on top of an
    explicit `torch.cuda.graph` capture is redundant and can fight
    over allocator state.
"""

from __future__ import annotations

import torch
import nvtx

from .expspec_sd import ExpSpecSDDraftModel


class ExpSpecSDCgDraftModel(ExpSpecSDDraftModel):
    """ExpSpec draft model whose tree-step forward is captured as a CUDA graph.

    Capture happens once via `init_cuda_graph_runner`; replay happens
    inside `speculate_once`. Until capture, behavior is identical to the
    eager `ExpSpecSDDraftModel`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cg_graph: torch.cuda.CUDAGraph | None = None
        # Static buffers (allocated lazily in init_cuda_graph_runner).
        self._cg_input_ids_buf: torch.Tensor | None = None
        self._cg_position_ids_buf: torch.Tensor | None = None
        self._cg_cache_position_buf: torch.Tensor | None = None
        self._cg_attn_mask_buf: torch.Tensor | None = None  # already inverted
        self._cg_output_buf: torch.Tensor | None = None
        self._cg_capture_max_cache_len: int | None = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_cuda_graph_runner(self, device: torch.device) -> None:
        """Allocate static buffers and capture the `[1, topk_len]` tree forward.

        Idempotent — a second call is a no-op while a graph is held.
        """
        if self._cg_graph is not None:
            return
        if not hasattr(self, "past_key_values") or self.past_key_values is None:
            raise RuntimeError(
                "init_cuda_graph_runner: past_key_values must be set on the draft "
                "model (call set_past_key_values) before capture."
            )
        cache = self.past_key_values.cache
        max_cache_len = getattr(cache, "max_cache_len", None)
        if max_cache_len is None:
            raise RuntimeError(
                "init_cuda_graph_runner: draft past_key_values must expose "
                "`max_cache_len` (use a static cache implementation)."
            )

        topk_len = int(self.draft_params.topk_len)
        max_cache_len = int(max_cache_len)

        # Output dtype: prefer lm_head.weight; otherwise fall back to any model param.
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            mask_dtype = lm_head.weight.dtype
        else:
            try:
                mask_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                mask_dtype = torch.float16

        # Static input buffers.
        self._cg_input_ids_buf = torch.zeros((1, topk_len), dtype=torch.long, device=device)
        self._cg_position_ids_buf = torch.zeros((1, topk_len), dtype=torch.long, device=device)
        self._cg_cache_position_buf = torch.zeros((topk_len,), dtype=torch.long, device=device)
        # Inverted attention mask: same shape that `invert_mask` produces from
        # the static TreeMaskCache buffer. `_tree_step` will copy in fresh
        # contents before each replay.
        self._cg_attn_mask_buf = torch.full(
            (1, 1, topk_len, max_cache_len),
            fill_value=torch.finfo(mask_dtype).min,
            dtype=mask_dtype,
            device=device,
        )
        self._cg_capture_max_cache_len = max_cache_len

        for buf in (
            self._cg_input_ids_buf,
            self._cg_position_ids_buf,
            self._cg_cache_position_buf,
            self._cg_attn_mask_buf,
        ):
            try:
                torch._dynamo.mark_static_address(buf)
            except Exception:
                pass

        self.model.eval()

        # Warmup + capture on a side stream, matching the FI capture path.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            for _ in range(2):
                _ = self(
                    self._cg_input_ids_buf,
                    with_softmax=True,
                    past_key_values=cache,
                    position_ids=self._cg_position_ids_buf,
                    attention_mask=self._cg_attn_mask_buf,
                    cache_position=self._cg_cache_position_buf,
                )

            torch.cuda.current_stream().wait_stream(stream)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, stream=stream):
                self._cg_output_buf = self(
                    self._cg_input_ids_buf,
                    with_softmax=True,
                    past_key_values=cache,
                    position_ids=self._cg_position_ids_buf,
                    attention_mask=self._cg_attn_mask_buf,
                    cache_position=self._cg_cache_position_buf,
                )

        self._cg_graph = cg
        print(
            f"[expspec-cg] captured draft forward "
            f"(L={topk_len}, max_cache_len={max_cache_len}, dtype={mask_dtype})."
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _draft_forward_step(
        self,
        token_ids: torch.Tensor,         # [1, topk_len]
        position_ids: torch.Tensor,      # [1, topk_len]
        attention_mask: torch.Tensor,    # [1, 1, topk_len, max_cache_len], already inverted
        cache_position: torch.Tensor,    # [topk_len]
    ) -> torch.Tensor:
        """Copy inputs into static buffers and replay the captured graph."""
        self._cg_input_ids_buf.copy_(token_ids, non_blocking=True)
        self._cg_position_ids_buf.copy_(position_ids, non_blocking=True)
        self._cg_cache_position_buf.copy_(cache_position, non_blocking=True)
        # `attention_mask` is the inverted [1,1,L,max_cache_len] tensor produced
        # by `TreeMaskCache.get_tree_mask` — already masked-out beyond the
        # current prefix length, so a straight copy is correct.
        self._cg_attn_mask_buf.copy_(attention_mask, non_blocking=True)
        self._cg_graph.replay()
        return self._cg_output_buf

    # ------------------------------------------------------------------
    # Per-step override
    # ------------------------------------------------------------------

    @torch.no_grad()
    def speculate_once(self, **kwargs):
        # Eager fallback until init_cuda_graph_runner has captured.
        if self._cg_graph is None:
            return super().speculate_once(**kwargs)

        tree_attention_mask = self.tree_mask_cache.get_tree_mask()  # inverted
        token_ids = self.token_ids
        parent_probs = self.parent_probs
        position_ids = self.position_ids
        cache_position = self.cache_position

        with nvtx.annotate("draft_forward_cg", color="red"):
            sampled_probs = self._draft_forward_step(
                token_ids=token_ids,
                position_ids=position_ids,
                attention_mask=tree_attention_mask,
                cache_position=cache_position,
            )

        with nvtx.annotate("draft_sample", color="green"):
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                parent_probs,
                self.draft_params.topk_len,
            )
            parent_probs = child_probs

        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)

        # In-place state update (same shapes/addresses across calls).
        self.token_ids = token_ids
        self.parent_probs = parent_probs
        self.position_ids += 1
        self.cache_position += self.draft_params.topk_len
