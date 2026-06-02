"""Hierarchical router: coarse ``linear_group`` logits, then TopK over masked experts.

Mirrors NxD ``RouterHierarchical`` (separate group head + per-expert logits). For
Qwen1.5-MoE-A2.7B (``E=60``, ``top_k=4``) use ``n_group=12``, ``topk_group=4``.

Pretrained checkpoints only contain ``linear_router`` (the usual gate); ``linear_group``
is additional and starts randomly unless you load a fine-tuned checkpoint.

Standalone CPU tests: NxD ``RouterBase`` expects parallel state (same as other demo routers).
Initialize a 1-rank process group and ``initialize_model_parallel(tp=1, pp=1)`` before
``RouterFactory.create(...)``, or exercise the router only inside ``run_qwen_moe_trn2`` / ModelBuilder.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributed import ProcessGroup

from neuronx_distributed.parallel_layers import mappings

from .base import SwappableRouter
from .base import _import_router_base


def _build_cls():
    RouterBase = _import_router_base()

    class HierarchicalRouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "hierarchical"
        ROUTER_GROUP = "B"
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
            sequence_parallel_enabled: bool = False,
            sequence_dimension: Optional[int] = None,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            bias: bool = False,
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
            act_fn: str = "softmax",
            apply_act_fn_over_topk: bool = False,
            jitter_eps: float = 0.0,
            store_transposed_weights: bool = False,
            use_nki_router: bool = False,
            **_unused,
        ):
            if use_nki_router:
                raise NotImplementedError(
                    "hierarchical: no fused NKI router shell; run without --use-nki-router."
                )
            if num_experts % n_group != 0:
                raise ValueError(
                    f"n_group={n_group} must divide num_experts={num_experts}"
                )
            if not (1 <= topk_group <= n_group):
                raise ValueError(
                    f"topk_group must be in [1, n_group], got topk_group={topk_group}, n_group={n_group}"
                )
            experts_per_group = num_experts // n_group
            max_routable = topk_group * experts_per_group
            if top_k > max_routable:
                raise ValueError(
                    f"top_k={top_k} exceeds number of experts visible after group selection "
                    f"({max_routable} = topk_group * experts_per_group)"
                )

            RouterBase.__init__(
                self,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                act_fn=act_fn,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_dimension=sequence_dimension,
                dtype=dtype,
                device=device if device is not None else torch.device("cpu"),
                bias=bias,
                tensor_model_parallel_group=tensor_model_parallel_group,
                jitter_eps=jitter_eps,
                store_transposed_weights=store_transposed_weights,
                apply_act_fn_over_topk=apply_act_fn_over_topk,
            )
            self.n_group = n_group
            self.topk_group = topk_group
            self.experts_per_group = experts_per_group

            dev = device if device is not None else torch.device("cpu")
            self.linear_group = nn.Linear(
                hidden_size, n_group, dtype=dtype, device=dev, bias=bias
            )
            setattr(
                self.linear_group.weight,
                "sequence_parallel_enabled",
                sequence_parallel_enabled,
            )
            if bias:
                setattr(
                    self.linear_group.bias,
                    "sequence_parallel_enabled",
                    sequence_parallel_enabled,
                )

        def _router_and_group_logits(self, hidden_states):
            h = hidden_states.to(dtype=self.linear_router.weight.dtype)
            if self.jitter_eps != 0.0 and self.training:
                h = h * torch.empty_like(h).uniform_(
                    1.0 - self.jitter_eps, 1.0 + self.jitter_eps
                )

            group_logits = self.linear_group(h)
            router_logits = self.linear_router(h)

            if self._if_training_gather_for_sp():
                group_logits = mappings.gather_from_sequence_parallel_region(
                    group_logits,
                    sequence_dimension=self.sequence_dimension,
                    to_model_parallel=False,
                    process_group=self.tensor_parallel_group,
                )
                router_logits = mappings.gather_from_sequence_parallel_region(
                    router_logits,
                    sequence_dimension=self.sequence_dimension,
                    to_model_parallel=False,
                    process_group=self.tensor_parallel_group,
                )

            group_logits = group_logits.view(-1, self.n_group)
            router_logits = router_logits.view(-1, self.num_experts)
            return router_logits, group_logits

        def forward(
            self, hidden_states: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            router_logits, group_logits = self._router_and_group_logits(hidden_states)

            _, group_idx = torch.topk(group_logits, self.topk_group, dim=1)

            epg = self.experts_per_group
            T = router_logits.shape[0]
            device = router_logits.device
            offsets = torch.arange(epg, device=device, dtype=torch.long).view(1, 1, epg)
            expert_flat = group_idx.unsqueeze(-1) * epg + offsets
            expert_mask = torch.zeros(T, self.num_experts, dtype=torch.bool, device=device)
            expert_mask.scatter_(1, expert_flat.reshape(T, -1), True)

            neg_large = torch.finfo(router_logits.dtype).min / 2
            masked_logits = router_logits.masked_fill(~expert_mask, neg_large)
            _, expert_index = torch.topk(masked_logits, self.top_k, dim=1)

            full = self.apply_activation_fn(router_logits)
            picked = full.gather(1, expert_index)
            expert_affinities = torch.zeros_like(full).scatter(1, expert_index, picked)
            expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
            expert_index = expert_index.detach().to(dtype=torch.long)
            return router_logits, expert_affinities, expert_index

    return HierarchicalRouter


HierarchicalRouter = _build_cls()

__all__ = ["HierarchicalRouter"]
