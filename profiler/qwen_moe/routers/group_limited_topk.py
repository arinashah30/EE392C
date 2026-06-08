"""Router #4: group-limited TopK with auxiliary-loss-free balancing.

DeepSeek-V3's ``noaux_tc`` routing strategy (Liu et al., 2024). The
``E`` experts are partitioned into ``n_group`` groups; the router
first selects the ``topk_group`` highest-scoring groups, then runs
``topk(top_k)`` on the union of selected groups (other groups are
masked to zero).

For Qwen1.5-MoE-A2.7B's ``E=60``, ``top_k=4`` the default partition
is ``n_group=12`` groups of 5 experts each, with ``topk_group=4``.
This means each token activates 4 of the 12 groups and picks its
top-4 experts from the union of those 4 groups (size 20).

Algorithm (matches NxD's :class:`GroupLimitedRouter`):

  1. ``router_logits = linear_router(hidden_states)``     shape (T, E)
  2. ``scores = sigmoid(router_logits)``                  shape (T, E)
  3. Add ``e_score_correction_bias`` (per-expert bias used by the
     no-aux-loss balancing rule from DeepSeek-V3, trainable):
     ``scores_for_choice = scores + bias.unsqueeze(0)``
  4. Group-level scores: ``group_scores[t, g] = top2(scores_for_choice[t, group_g]).sum()``
  5. ``group_idx = topk(group_scores, k=topk_group)``     shape (T, topk_group)
  6. Build a binary mask ``score_mask[t, e]`` = 1 if expert ``e``'s
     group is among ``group_idx[t]``, else 0.
  7. ``masked = scores_for_choice.masked_fill(~score_mask, 0)``
  8. ``expert_index = topk(masked, k=top_k)``             shape (T, top_k)

Why this is Group A (accuracy-preserving):

  * Score function is **sigmoid** (same as #3), so the chosen experts
    overlap heavily with what Qwen's pretrained TopK+softmax would
    pick — provided we don't enable the bias.
  * The bias term defaults to zero at construction time; it's only
    learned during training. For T0 inference on pretrained Qwen
    we ship a zero bias so the router degrades gracefully to "pick
    the topk_group highest-scoring groups, then topk within."
  * If the user supplies an ``e_score_correction_bias`` via a T1
    fine-tune checkpoint, the bias is loaded transparently by the
    factory's ``_maybe_load_finetune_checkpoint``.

Wrapping NxD's :class:`GroupLimitedRouter` directly so we get the
same compiled kernel path. The wrapper adds:
  * the ``ROUTER_NAME``/``ROUTER_GROUP``/``HAS_LEARNABLE_PARAMS`` class
    attrs the factory inspects;
  * the ``e_score_correction_bias`` parameter (NxD's base class
    references this attribute but doesn't construct it);
  * the optional ``--use-nki-router`` shortcut that routes through
    the fused NKI shell.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .base import SwappableRouter


def _build_cls():
    from neuronx_distributed.modules.moe.routing import GroupLimitedRouter

    class GroupLimitedTopKRouter(SwappableRouter, GroupLimitedRouter):
        ROUTER_NAME = "group_limited_topk"
        ROUTER_GROUP = "A"
        HAS_LEARNABLE_PARAMS = True
        REQUIRES_CUSTOM_MOE = False

        def __init__(
            self,
            *,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            n_group: int = 12,
            topk_group: int = 4,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            sequence_parallel_enabled: bool = False,
            sequence_dimension: Optional[int] = None,
            use_nki_router: bool = False,
            **_unused,
        ):
            if num_experts % n_group != 0:
                raise ValueError(
                    f"group_limited_topk: num_experts ({num_experts}) must be "
                    f"divisible by n_group ({n_group})"
                )
            if not (1 <= topk_group <= n_group):
                raise ValueError(
                    f"group_limited_topk: topk_group ({topk_group}) must be "
                    f"in [1, n_group={n_group}]"
                )
            GroupLimitedRouter.__init__(
                self,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                n_group=n_group,
                topk_group=topk_group,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                dtype=dtype,
                device=device if device is not None else torch.device("cpu"),
            )
            # NxD's GroupLimitedRouter references self.e_score_correction_bias
            # but does not construct it. We declare it here so the param is
            # always present (zero-init), and T1 fine-tune checkpoints can
            # populate it.
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(num_experts, dtype=dtype, device=self.device),
                requires_grad=True,
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
                    score_mode="group_limited",
                    score_mode_extra={
                        "n_group": self.n_group,
                        "topk_group": self.topk_group,
                        "e_score_correction_bias": self.e_score_correction_bias,
                    },
                )
            return GroupLimitedRouter.forward(self, hidden_states)

    return GroupLimitedTopKRouter


GroupLimitedTopKRouter = _build_cls()


__all__ = ["GroupLimitedTopKRouter"]
