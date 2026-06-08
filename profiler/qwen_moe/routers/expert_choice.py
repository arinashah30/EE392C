"""Router #8: Expert Choice routing.

Inverts the optimization direction of standard TopK routing: instead of
each *token* picking ``top_k`` experts, each *expert* picks the
``T_per_expert`` highest-scoring tokens (Zhou et al. 2022,
"Mixture-of-Experts with Expert Choice Routing").

  T_per_expert = ⌈top_k · T / E⌉

For Qwen at prefill (T=128, E=60, top_k=4) → T_per_expert = 9.
Total token-expert assignments: T_per_expert · E = 540 ≈ top_k · T = 512.

Algorithm:

  1. ``router_logits = linear_router(hidden_states)``         shape (T, E)
  2. ``scores = softmax(logits, dim=0)``  (softmax over TOKENS, the
     transpose of the standard softmax). Each column is now a
     probability distribution *across tokens*: ``Pr(expert e picks token t)``.
  3. For each expert ``e``, take ``topk(scores[:, e], T_per_expert)``
     along the token axis. This produces a binary expert-choice mask
     ``ec_mask ∈ {0, 1}^{T x E}`` where exactly ``T_per_expert``
     entries per column are set.
  4. Per-token, find the top-``top_k`` experts that picked it. Tokens
     picked by 0 experts get top_k zero weights (effectively no output);
     tokens picked by ``> top_k`` experts have the lowest-affinity
     extras dropped — both are honest to Expert Choice's behavior at
     ``T_per_expert · E ≈ top_k · T``.

Why this preserves the standard MoE block contract:

The function still returns ``(logits, affinities, indices)`` with
shapes ``(T, E)``, ``(T, E)``, ``(T, top_k)`` respectively — identical
to :class:`RouterTopK`. The expert-choice constraint is folded into
the *affinities* (zeros for tokens an expert did not pick), so the
existing ``ExpertMLPs.forward_all_experts`` path produces the correct
output without modification.

This means the dedicated :class:`MoEExpertChoice` wrapper does not
need to override the forward pass. We still ship it as a thin subclass
of ``MoE`` so the cost model and aggregate report can identify
expert-choice deployments by class.

Group B: hugely changes which experts see which tokens (compared to
pretrained Qwen). T0 inference will drop MMLU. The router is
trainable (``linear_router.weight`` is the same shape as Qwen's
gate), so T1 router-only fine-tune recovers most of the gap.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import SwappableRouter


def _build_classes():
    from neuronx_distributed.modules.moe.routing import RouterBase
    from neuronx_distributed.modules.moe.model import MoE

    class ExpertChoiceRouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "expert_choice"
        ROUTER_GROUP = "B"
        HAS_LEARNABLE_PARAMS = True
        REQUIRES_CUSTOM_MOE = True

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
            tokens_per_expert: Optional[int] = None,  # optional override
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
            self._tokens_per_expert_override = tokens_per_expert

        def _tokens_per_expert(self, T: int) -> int:
            if self._tokens_per_expert_override is not None:
                return int(self._tokens_per_expert_override)
            return max(1, math.ceil(self.top_k * T / self.num_experts))

        def forward(
            self, hidden_states
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # router_logits: (T, E)
            router_logits = self.get_router_logits(hidden_states)
            T, E = router_logits.shape

            # Transpose softmax: prob distribution over tokens, per expert.
            # fp32 for stability across the long token axis.
            scores_per_expert = F.softmax(
                router_logits.to(torch.float32), dim=0
            )

            T_per_e = self._tokens_per_expert(T)
            # (T_per_e, E): scores of top-T_per_e tokens for each expert.
            # (T_per_e, E): indices of those tokens in the token axis.
            _, ec_topk_indices = torch.topk(scores_per_expert, T_per_e, dim=0)

            # Build the (T, E) binary mask: 1 where an expert picked the token.
            ec_mask = torch.zeros_like(scores_per_expert)
            ec_mask.scatter_(0, ec_topk_indices, 1.0)

            # Per-token affinities = scores * mask. Zero everywhere an
            # expert did not pick the token. This is what
            # ExpertMLPs.forward_all_experts multiplies into the expert
            # output, so unselected tokens contribute zero.
            masked_scores = scores_per_expert * ec_mask  # (T, E)

            # Per-token, pick the top_k highest-affinity experts that
            # selected this token. Tokens with <top_k picks get
            # duplicate-index padding; their topk_weights at those slots
            # are zero (because masked_scores there is zero), so the
            # dense (T, E) affinity tensor stays sparse correctly.
            topk_weights, expert_index = torch.topk(
                masked_scores, self.top_k, dim=1
            )

            # Normalize the surviving weights to sum to 1 per token.
            # Tokens with zero total mass stay zero (clamp_min keeps
            # this safe). Without this, ExpertMLPs would weight each
            # surviving expert by a raw softmax probability rather than
            # a normalized affinity, which is *correct* Expert Choice
            # behavior — but ``normalize_top_k_affinities=True`` is the
            # demo's setting, so we honor it here.
            sums = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_weights = topk_weights / sums
            topk_weights = topk_weights.to(hidden_states.dtype)

            expert_affinities = torch.zeros_like(router_logits)
            expert_affinities = expert_affinities.scatter_(
                1, expert_index, topk_weights
            )
            expert_index = expert_index.detach().to(torch.long)
            return router_logits, expert_affinities, expert_index

    class MoEExpertChoice(MoE):
        """Identical to NxD's ``MoE`` block; subclassed for clarity and
        so the cost-model / aggregate-report code can identify
        expert-choice deployments by class.

        The expert-choice constraint is enforced *inside the router* by
        folding it into the per-token affinities (see
        :class:`ExpertChoiceRouter`), which means the standard
        ``ExpertMLPs.forward_all_experts`` path already produces the
        correct output. No forward override needed.
        """

        ROUTER_KIND = "expert_choice"

    return ExpertChoiceRouter, MoEExpertChoice


ExpertChoiceRouter, MoEExpertChoice = _build_classes()


__all__ = ["ExpertChoiceRouter", "MoEExpertChoice"]
