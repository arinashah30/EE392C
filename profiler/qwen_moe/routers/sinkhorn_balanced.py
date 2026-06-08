"""Router #6: Sinkhorn-balanced TopK.

Extends NxD's :class:`RouterSinkhorn` from top-1 to top-K routing.
The Sinkhorn-Knopp algorithm iteratively normalizes the rows and
columns of the (positive) score matrix so that every token's row
sums to 1 and every expert's column sums to ``T / E``. This drives
the routing toward a *balanced* assignment where every expert sees
approximately the same number of tokens.

Algorithm (matches Megatron-LM, Korthikanti et al. 2022):

  1. ``router_logits = linear_router(hidden_states)``     shape (T, E)
  2. ``cost = exp(router_logits)`` (fp32 for stability)
  3. Sinkhorn iterations (``num_iters``, default 30):
       d0[t]    = (1/T) * 1 / (sum_e d1[e] * cost[t, e])
       d1[e]    = (1/E) * 1 / (sum_t d0[t] * cost[t, e])
  4. Balanced cost: ``balanced[t, e] = d1[e] * cost[t, e] * d0[t]``
  5. ``expert_index = topk(balanced, k)``               shape (T, top_k)
  6. ``topk_weights = sigmoid(router_logits).gather(1, expert_index)``  (T, k)
  7. ``expert_affinities = scatter_zeros_then(weights)``  shape (T, E)

Notes:

  * NxD's :class:`RouterSinkhorn` was top-1 only. We extend it
    in-place: the Sinkhorn iterations are unchanged; we replace
    ``argmax`` with ``topk``.
  * Sinkhorn runs only at *inference* in NxD's implementation
    (no-grad). At T1 fine-tune time we still allow gradients to
    flow through the underlying ``sigmoid(router_logits)`` weights
    (the Sinkhorn step is a detached re-weighting that decides
    which experts are picked — like a routing oracle, not a
    differentiable score).
  * Fixed number of iterations (default 30) so the compiled graph
    is static. The ``sinkhorn_tol`` knob is purely diagnostic.

Group B: balances expert load by construction, which deliberately
deviates from what Qwen's pretrained TopK+softmax chose. Without a
T1 fine-tune (``bench/finetune_router.py``) this router will route
~10–20% of tokens to a different expert set than the baseline and
MMLU will drop accordingly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import SwappableRouter, _zeros_affinities_from_topk


def _build_cls():
    from neuronx_distributed.modules.moe.routing import RouterSinkhorn

    class SinkhornBalancedRouter(SwappableRouter, RouterSinkhorn):
        ROUTER_NAME = "sinkhorn_balanced"
        ROUTER_GROUP = "B"
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
            use_nki_router: bool = False,  # unused (no NKI shell for Sinkhorn)
            sinkhorn_iterations: int = 30,
            sinkhorn_tol: Optional[float] = None,
            **_unused,
        ):
            # NxD's base class enforces top_k==1. We bypass that check
            # by jumping past it: construct the underlying RouterBase
            # state ourselves rather than calling RouterSinkhorn.__init__.
            from neuronx_distributed.modules.moe.routing import RouterBase
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
            )
            self.sinkhorn_iterations = sinkhorn_iterations
            self.sinkhorn_tol = sinkhorn_tol

        def forward(
            self, hidden_states
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            router_logits = self.get_router_logits(hidden_states)

            # Sinkhorn balancing runs in fp32 to avoid bf16 overflow on
            # exp(logits) when |logits| is large. The result is a
            # detached cost matrix the topk runs on.
            with torch.no_grad():
                if self.training:
                    sinkroute = RouterSinkhorn._sinkhorn(
                        router_logits.detach().to(torch.float32),
                        num_iters=self.sinkhorn_iterations,
                        tol=self.sinkhorn_tol,
                    )
                else:
                    # Inference: still apply Sinkhorn so the *chosen
                    # experts* are balanced. (NxD's reference skips
                    # Sinkhorn at inference for top-1; we keep it for
                    # top-k so balance holds.)
                    sinkroute = RouterSinkhorn._sinkhorn(
                        router_logits.detach().to(torch.float32),
                        num_iters=self.sinkhorn_iterations,
                        tol=None,
                    )
                # topk over balanced scores; indices are what we route on.
                _, expert_index = torch.topk(sinkroute, self.top_k, dim=1)

            # Weights themselves use the *original* sigmoid scores so
            # backprop still flows through linear_router during T1.
            sigmoid_scores = torch.sigmoid(router_logits.to(torch.float32))
            topk_weights = sigmoid_scores.gather(1, expert_index)
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

    return SinkhornBalancedRouter


SinkhornBalancedRouter = _build_cls()


__all__ = ["SinkhornBalancedRouter"]
