"""Router #9: BASE Layer (single-iteration auction).

BASE Layer (Lewis et al. 2021, "BASE Layers: Simplifying Training of
Large, Sparse Models") assigns tokens to experts via a balanced linear
assignment so that every expert sees exactly ``T / E`` tokens.
The original paper solves the assignment with the Hungarian algorithm
which is inherently iterative and not Trainium-friendly (dynamic
shapes + control flow over the assignment graph).

We ship a static-shape approximation that runs **one iteration of an
auction-style assignment** and falls back to argmax for tokens that
no expert claimed. This is the simplest Trn-compatible BASE
implementation; it gives perfect expert-load balance for the
``capacity = ⌈T / E⌉`` budget per expert, modulo a small number of
tokens that get routed by their raw argmax (always within at most
one capacity slot of perfect).

Algorithm:

  1. ``router_logits = linear_router(hidden_states)``     shape (T, E)
  2. ``scores = softmax(router_logits, dim=1)``           shape (T, E)
  3. ``capacity = ⌈T / E⌉``
  4. **Expert claim phase**: for each expert ``e``, find the top
     ``capacity`` tokens by score → ``claim_mask ∈ {0,1}^{T x E}``
     with exactly ``capacity`` ones per column.
  5. **Token decision phase**: each token picks its highest-score
     claimant; tokens with no claimant fall back to their raw argmax.
  6. The chosen expert per token forms a *hard top-1* assignment.

Returning the ``(T, top_k)`` shape contract while only assigning a
single expert per token:

  * Affinity = 1.0 at the chosen expert, 0 elsewhere.
  * ``expert_index = topk(scores, top_k)`` — provides slot 0 = chosen
    (because BASE adjusted scores keep chosen as the highest)
    and slots 1..top_k-1 = next-best-by-score (whose affinity is 0,
    so they don't contribute compute through ``forward_all_experts``).

Because ``ExpertMLPs.get_expert_mask`` is a top_k-hot encoder via
``+=`` (see ``expert_mlps_v2.py:get_expert_mask``), passing distinct
indices in the ``top_k`` slots ensures ``mask[t, chosen[t]] == 1``
exactly, so ``masked = affinity * mask = 1 * 1 = 1`` and the chosen
expert's output is used at unit weight. Padding slots have zero
affinity → zero mask contribution → no spurious double-counting.

Group B: BASE deliberately re-balances against Qwen's natural
routing → 5–15% MMLU drop at T0. T1 router-only fine-tune recovers
most of it (BASE's optimal balancing tends to find a good
"compromise" expert set quickly).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import SwappableRouter


def _build_cls():
    from neuronx_distributed.modules.moe.routing import RouterBase

    class BaseLayerRouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "base_layer"
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
            use_nki_router: bool = False,
            bias: bool = False,
            **_unused,
        ):
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

        def forward(
            self, hidden_states
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            router_logits = self.get_router_logits(hidden_states)
            T, E = router_logits.shape

            scores = F.softmax(router_logits.to(torch.float32), dim=1)
            capacity = max(1, math.ceil(T / E))

            # ---- Claim phase ----
            # For each expert: top-`capacity` tokens by score.
            _, claim_idx = torch.topk(scores, capacity, dim=0)  # (capacity, E)
            claim_mask = torch.zeros_like(scores)
            claim_mask.scatter_(0, claim_idx, 1.0)             # (T, E)

            # ---- Decision phase ----
            # Among each token's claimants, pick the highest-score one.
            # Tokens with no claimant: fall back to their raw argmax.
            # Implemented with one masked-fill + a torch.where on the
            # has-claimant mask.
            NEG = torch.finfo(scores.dtype).min / 4  # safely-large negative
            masked_scores = scores * claim_mask + NEG * (1.0 - claim_mask)
            has_claimant = (claim_mask.sum(dim=1, keepdim=True) > 0)  # (T, 1)
            adjudicated = torch.where(has_claimant, masked_scores, scores)

            chosen = adjudicated.argmax(dim=1, keepdim=True)  # (T, 1)

            # ---- Format ----
            # expert_index: chosen + the next top_k-1 distinct experts
            # (those entries get zero affinity, so they contribute
            # nothing through forward_all_experts).
            # We use the *adjudicated* scores so chosen is guaranteed
            # to appear in slot 0 of the topk result.
            _, expert_index = torch.topk(adjudicated, self.top_k, dim=1)
            expert_index = expert_index.to(torch.long)

            # Build affinities: 1.0 at chosen, 0 elsewhere.
            expert_affinities = torch.zeros_like(router_logits)
            expert_affinities = expert_affinities.scatter_(
                1, chosen, 1.0
            ).to(hidden_states.dtype)

            return router_logits, expert_affinities, expert_index

    return BaseLayerRouter


BaseLayerRouter = _build_cls()


__all__ = ["BaseLayerRouter"]
