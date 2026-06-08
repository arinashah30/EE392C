"""Router #5: Sparsemax + TopK.

Sparsemax (Martins & Astudillo, 2016) is a Euclidean projection of
the logits onto the probability simplex. Unlike softmax which always
produces strictly positive probabilities, sparsemax produces exact
zeros for experts the projection deems unhelpful.

Mathematically (per row of logits ``z ∈ R^E``):

  sparsemax(z) = argmin_{p ∈ Δ^{E-1}} ||p - z||²
              = (z - τ(z))_+

where ``τ(z)`` is computed in closed form:

  Sort z descending: z_(1) >= z_(2) >= ... >= z_(E)
  Let k_max = max { k : 1 + k * z_(k) > sum_{j<=k} z_(j) }
  τ(z) = (sum_{j<=k_max} z_(j) - 1) / k_max
  sparsemax(z)_i = max(z_i - τ(z), 0)

Properties for MoE routing:

  * **Exact sparsity**: experts with low scores get probability ZERO,
    not just small probability. The number of "active" experts
    (those with non-zero probability) is data-dependent and can be
    larger or smaller than the chosen ``top_k``. We still take
    ``top_k`` of the non-zero entries by score so the dispatch shape
    matches the rest of the MoE pipeline.
  * **Bounded support**: matches DSelect-K (Hazimeh et al. 2021)
    behavior — the sparsemax mass concentrates on a small set of
    confident experts, and downstream ``top_k`` truncation rarely
    drops mass.
  * **Differentiable a.e.**: unlike top-k itself, sparsemax has
    well-defined gradients (the Jacobian is the projection onto the
    support of the active set), which makes it cheap to T1-fine-tune.
  * Compared to softmax-over-topk (#2), sparsemax can produce
    *fewer* than top_k non-zero entries — this exposes a routing
    decision that softmax cannot make and is a key reason this router
    can outperform #2 on highly-confident tokens (e.g. code, symbols).

Group A: doesn't reorder the topk choice for the highest-confidence
tokens (where softmax and sparsemax produce the same argmax), so
pretrained Qwen quality is preserved when we use it. Drift comes
from the medium-confidence tokens where sparsemax zeroes-out a few
mid-rank experts that softmax would have given small positive mass.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import SwappableRouter, _zeros_affinities_from_topk


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Closed-form sparsemax over ``dim`` (Martins & Astudillo, 2016).

    Implemented with the sort-based ``k_max`` rule; vectorized over
    every dimension except ``dim``. Used by the torch reference path.
    """
    orig_dtype = logits.dtype
    # Compute in fp32 to keep the sort + cumsum stable in bf16.
    z = logits.to(torch.float32)

    # Sort descending along the active dim.
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    range_k = torch.arange(
        1, z.shape[dim] + 1, device=z.device, dtype=z.dtype
    )
    # Broadcast range to match z_sorted shape on `dim`.
    shape = [1] * z.dim()
    shape[dim] = -1
    range_k = range_k.view(shape)

    cumsum = torch.cumsum(z_sorted, dim=dim)
    is_active = (1.0 + range_k * z_sorted) > cumsum
    # k_max: largest index where is_active is True (per row).
    k_max = is_active.to(torch.int64).sum(dim=dim, keepdim=True)
    # Guard k_max >= 1 (always true in theory; clamp for fp safety).
    k_max = k_max.clamp_min(1)

    # tau = (sum_{j<=k_max} z_(j) - 1) / k_max
    sum_top = cumsum.gather(dim, k_max - 1)
    tau = (sum_top - 1.0) / k_max.to(z.dtype)

    out = torch.clamp(z - tau, min=0.0)
    return out.to(orig_dtype)


def _build_cls():
    from neuronx_distributed.modules.moe.routing import RouterBase

    class SparsemaxTopKRouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "sparsemax_topk"
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
            bias: bool = False,
            **_unused,
        ):
            # We pass act_fn="softmax" to RouterBase as a placeholder; we
            # override the score path ourselves in forward().
            RouterBase.__init__(
                self,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                act_fn="softmax",
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                dtype=dtype,
                device=device if device is not None else torch.device("cpu"),
                bias=bias,
            )
            self._use_nki_router = use_nki_router

        def forward(
            self, hidden_states
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self._use_nki_router:
                from .fused_router_kernel import fused_router_call
                return fused_router_call(
                    hidden_states=hidden_states,
                    linear_router=self.linear_router,
                    top_k=self.top_k,
                    score_mode="sparsemax",
                )

            # router_logits: (T, E)
            router_logits = self.get_router_logits(hidden_states)
            scores = sparsemax(router_logits, dim=-1)
            # The top_k entries by sparsemax score — most rows will have
            # at most top_k non-zero anyway, but we always return a
            # constant-shape (T, top_k) so the dispatcher stays static.
            topk_weights, expert_index = torch.topk(scores, self.top_k, dim=1)
            # Re-normalize so the weights sum to 1 — needed because some
            # rows lose mass when sparsemax produced fewer than top_k
            # nonzeros (the missing weights are zeros; normalization
            # turns the surviving ones into a valid distribution).
            sums = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_weights = topk_weights / sums
            topk_weights = topk_weights.to(hidden_states.dtype)

            expert_affinities = _zeros_affinities_from_topk(
                router_logits, expert_index, topk_weights
            )
            expert_affinities = expert_affinities.to(hidden_states.dtype)
            expert_index = expert_index.detach().to(torch.long)
            return router_logits, expert_affinities, expert_index

    return SparsemaxTopKRouter


SparsemaxTopKRouter = _build_cls()


__all__ = ["SparsemaxTopKRouter", "sparsemax"]
