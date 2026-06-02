"""Router #3: TopK + per-expert sigmoid scoring (DeepSeek-V2 / V3 style).

Algorithm:

  1. ``router_logits = linear_router(hidden_states)``        shape (T, E)
  2. ``scores = sigmoid(router_logits)``                     shape (T, E)
  3. ``expert_index = topk(scores, k)``                      shape (T, k)
  4. ``topk_weights = gather(scores, expert_index)``         shape (T, k)
  5. ``expert_affinities = scatter(zeros(T, E), expert_index, topk_weights)``

Differences from #1 (TopK + softmax):

  * Sigmoid is **independent per expert** — no row normalization,
    no softmax cross-expert competition. The activation of expert
    ``e`` is a function only of its own logit. This is what DeepSeek-V2
    and V3 call "sigmoid gating".
  * Affinity values are in ``[0, 1]`` but generally **don't sum to 1**.
    Use with ``normalize_top_k_affinities=True`` (the demo default) for
    apples-to-apples comparison with #1; omit to keep DeepSeek-V3's
    unnormalized scale.
  * Numerically more forgiving in bf16 than full-row softmax: each
    sigmoid only involves one logit, so there's no large-E reduction
    to lose precision on.
  * The price: the affinities are no longer a categorical distribution,
    so the routing-entropy metric (`H(p_expert)` in
    ``moe_demo/profile/analyze_routing.py``) must be computed *after*
    normalization or it's not interpretable.

Group A: keeps the same chosen experts as Qwen's TopK+softmax up to
the same numerical noise that would change a logit tie-breaker, so
accuracy drift is minimal on pretrained Qwen (unless ``norm_topk_prob``
is set differently).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import SwappableRouter, _zeros_affinities_from_topk


def _build_cls():
    from neuronx_distributed.modules.moe.routing import RouterBase

    class TopKSigmoidRouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "topk_sigmoid"
        ROUTER_GROUP = "A"
        HAS_LEARNABLE_PARAMS = True
        REQUIRES_CUSTOM_MOE = False

        def __init__(
            self,
            *,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            sequence_parallel_enabled: bool = False,
            sequence_dimension: Optional[int] = None,
            use_nki_router: bool = False,
            normalize_topk: bool = True,
            bias: bool = False,
            **_unused,
        ):
            RouterBase.__init__(
                self,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                act_fn="sigmoid",
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                dtype=dtype,
                device=device if device is not None else torch.device("cpu"),
                bias=bias,
            )
            self._use_nki_router = use_nki_router
            self.normalize_topk = normalize_topk

        def forward(
            self, hidden_states
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self._use_nki_router:
                from .fused_router_kernel import fused_router_call
                return fused_router_call(
                    hidden_states=hidden_states,
                    linear_router=self.linear_router,
                    top_k=self.top_k,
                    score_mode="sigmoid",
                    normalize_topk=self.normalize_topk,
                )

            # router_logits: (T, E)
            router_logits = self.get_router_logits(hidden_states)
            # scores in fp32 to keep sigmoid stable; cast back at the end.
            scores = torch.sigmoid(router_logits.to(torch.float32))

            topk_weights, expert_index = torch.topk(scores, self.top_k, dim=1)
            if self.normalize_topk:
                topk_weights = topk_weights / topk_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-9)
            topk_weights = topk_weights.to(hidden_states.dtype)

            expert_affinities = _zeros_affinities_from_topk(
                router_logits, expert_index, topk_weights
            )
            expert_affinities = expert_affinities.to(hidden_states.dtype)
            expert_index = expert_index.detach().to(torch.long)
            return router_logits, expert_affinities, expert_index

    return TopKSigmoidRouter


TopKSigmoidRouter = _build_cls()


__all__ = ["TopKSigmoidRouter"]
