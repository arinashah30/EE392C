"""Router #2: TopK first, softmax only over the chosen ``k`` values.

Algorithm:

  1. ``router_logits = linear_router(hidden_states)``   shape ``(T, E)``
  2. ``topk_values, expert_index = topk(router_logits, k)``
  3. ``topk_weights = softmax(topk_values, dim=-1)``  (size ``(T, k)``)
  4. Scatter ``topk_weights`` back into a dense ``(T, E)`` zero-filled
     tensor at ``expert_index`` to produce ``expert_affinities``.

Why this is numerically cheaper than #1 (``topk_softmax``):

The Qwen baseline does ``softmax(logits, dtype=fp64)`` over all ``E``
experts and then runs ``topk`` on the *logits* (not the affinities)
to pick ``k``. The fp64 cast is there because the baseline applies
softmax over the full ``E=60``-wide vector where the un-cast bf16
softmax loses precision on the tail (subtracting `max(logits)` and
exponentiating ~55 entries in bf16 can flush small differences to
zero, perturbing both the chosen experts and their weights).

This router takes the same chosen experts (because topk on logits
selects the same set as topk on softmax(logits)) but only ever
softmaxes over a ``k=4``-wide vector, which is stable in bf16. That
saves:

  * the fp64 cast on the full ``(T, E)`` matrix (4× the bytes through
    SBUF on Neuron);
  * the larger softmax reduction (5*T*E scalar ops vs. 5*T*k);
  * the larger ``scatter`` of weights into the dense affinities tensor
    (we write ``T*k`` values instead of ``T*E``).

Identical chosen experts to #1, but with a smaller fp32 reduction
window and (in the NKI shell) a fused implementation that elides the
intermediate write of dense ``(T, E)`` weights to HBM.

NxD's ``RouterTopK`` already supports this via ``apply_act_fn_over_topk=True``;
we expose it here under a distinct name + group so the benchmark
treats it as a separate kernel.
"""
from __future__ import annotations

from typing import Optional

import torch

from .base import SwappableRouter


def _build_cls():
    from neuronx_distributed.modules.moe.routing import RouterTopK

    class TopKSoftmaxOverTopKRouter(SwappableRouter, RouterTopK):
        ROUTER_NAME = "topk_softmax_over_topk"
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
                apply_act_fn_over_topk=True,
            )
            self._use_nki_router = use_nki_router

        def forward(self, hidden_states):
            if self._use_nki_router:
                from .fused_router_kernel import fused_router_call
                return fused_router_call(
                    hidden_states=hidden_states,
                    linear_router=self.linear_router,
                    top_k=self.top_k,
                    score_mode="softmax_over_topk",
                )
            return RouterTopK.forward(self, hidden_states)

    return TopKSoftmaxOverTopKRouter


TopKSoftmaxOverTopKRouter = _build_cls()


__all__ = ["TopKSoftmaxOverTopKRouter"]
