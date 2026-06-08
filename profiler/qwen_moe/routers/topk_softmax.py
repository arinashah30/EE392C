"""Router #1: classical TopK + softmax (Qwen / Mixtral / Switch baseline).

This is a thin wrapper over NxD's :class:`RouterTopK` so that the
factory's single dispatch path covers the baseline router too. The
wrapper exists so that:

  * :data:`SwappableRouter.ROUTER_NAME` is set to ``"topk_softmax"``
    (used by NEFF cache keys and the aggregate report).
  * The factory's optional ``router_ft_checkpoint`` and
    ``use_nki_router`` kwargs are accepted (and silently ignored
    for the torch path) without the caller having to know which
    router is the baseline.
  * Subclassing keeps the existing NxD ``preshard_hook`` and
    ``store_transposed_weights`` machinery wired without
    duplication.

Algorithm (matches the existing baseline in
``moe_demo/qwen_moe/neuron_modeling_qwen_moe.py``):

  1. ``router_logits = linear_router(hidden_states)``   shape ``(T, E)``
  2. ``expert_affinities = softmax(router_logits, dim=1, fp64)``
  3. ``expert_index = topk(router_logits, k)``
  4. Return ``(router_logits, expert_affinities, expert_index)``

Normalization of the topk weights to sum to 1 is handled downstream
by ``ExpertMLPs(normalize_top_k_affinities=True)``; we don't
duplicate it here.
"""
from __future__ import annotations

from typing import Optional

import torch

from .base import SwappableRouter, _import_router_base


def _build_topk_softmax_cls():
    """Lazy-built class that inherits from NxD's RouterTopK at runtime."""
    from neuronx_distributed.modules.moe.routing import RouterTopK

    class TopKSoftmaxRouter(SwappableRouter, RouterTopK):
        ROUTER_NAME = "topk_softmax"
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
            apply_act_fn_over_topk: bool = False,
            bias: bool = False,
            **_unused,
        ):
            # NxD RouterTopK does the heavy lifting; SwappableRouter is
            # a marker mixin so __init__ delegates straight through.
            RouterTopK.__init__(
                self,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                dtype=dtype,
                device=device if device is not None else torch.device("cpu"),
                bias=bias,
                act_fn="softmax",
                apply_act_fn_over_topk=apply_act_fn_over_topk,
            )
            self._use_nki_router = use_nki_router

        def forward(self, hidden_states):
            if self._use_nki_router:
                # The fused-NKI shell handles linear+softmax+topk in one
                # kernel; falls back to torch if the NKI components fail
                # to import (e.g. on a CPU dev box).
                from .fused_router_kernel import fused_router_call
                return fused_router_call(
                    hidden_states=hidden_states,
                    linear_router=self.linear_router,
                    top_k=self.top_k,
                    score_mode="softmax",
                )
            return RouterTopK.forward(self, hidden_states)

    return TopKSoftmaxRouter


# Module-level class for `from qwen_moe.routers.topk_softmax import TopKSoftmaxRouter`.
# We resolve it eagerly here because importing NxD at module load is
# fine on the inference venv (everything else in qwen_moe/ already does so).
TopKSoftmaxRouter = _build_topk_softmax_cls()


__all__ = ["TopKSoftmaxRouter"]
