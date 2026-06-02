"""Shared API contract for the swappable Qwen-MoE router kernels.

Every router under ``moe_demo/qwen_moe/routers/`` exposes a single
``nn.Module`` whose ``forward(hidden_states)`` returns the same
three-tuple as NxD's ``RouterTopK``:

  - ``router_logits``      shape ``(T, E)``   raw scores per expert
  - ``expert_affinities``  shape ``(T, E)``   normalized weights (zeros
                                              outside the chosen subset
                                              for sparse routers)
  - ``expert_index``       shape ``(T, top_k)`` chosen expert ids per
                                                token (``torch.long``)

``T = batch * seq_len`` is the flattened token dimension. The contract
matches ``RouterBase.forward`` so the ``MoE`` block consumes any router
listed in :class:`RouterFactory` without modification.

We add a lightweight :class:`SwappableRouter` base class on top of
``RouterBase`` so all routers gain:

  * a ``ROUTER_NAME`` class attribute (the factory key);
  * a ``ROUTER_GROUP`` class attribute (``"A"`` or ``"B"``);
  * a ``HAS_LEARNABLE_PARAMS`` class attribute (``False`` only for
    :class:`HashRouter`);
  * a uniform ``__repr__`` that includes the router name (handy when
    inspecting compiled graphs).

Routers that need to alter the MoE forward pass shape (Expert Choice,
SoftMoE) ship a paired ``MoE*`` subclass in their module file. The
factory exposes :meth:`RouterFactory.build_moe` to construct the right
``MoE`` wrapper for any router.
"""
from __future__ import annotations

import math
from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


# The torch.distributed.ProcessGroup type is exposed at runtime by the
# NxD inference venv; we keep a forward reference so this module imports
# cleanly even when torch.distributed isn't fully wired (e.g. importing
# from a CPU-only dev box for static analysis).
try:  # pragma: no cover - import-time shim
    from torch.distributed import ProcessGroup as _ProcessGroup
except ImportError:  # pragma: no cover
    _ProcessGroup = object  # type: ignore[assignment]


def _import_router_base():
    """Lazy import of NxD's :class:`RouterBase` so this file is usable
    in a unit-test environment without ``neuronx_distributed`` installed.
    The actual swappable routers subclass the returned class.
    """
    from neuronx_distributed.modules.moe.routing import RouterBase
    return RouterBase


class SwappableRouter(nn.Module):
    """Marker / mixin every router in the factory inherits.

    Subclasses MUST implement
    :meth:`forward` (returning the 3-tuple above) and set the three
    class attributes ``ROUTER_NAME``, ``ROUTER_GROUP``,
    ``HAS_LEARNABLE_PARAMS``.

    Concrete subclasses are expected to also inherit from NxD's
    :class:`RouterBase` so that ``MoE.__init__`` accepts them. The split
    exists so that test code can ``isinstance(router, SwappableRouter)``
    without dragging in NxD.
    """

    ROUTER_NAME: str = "unset"
    ROUTER_GROUP: str = "unset"  # "A" = preserve baseline accuracy, "B" = SOTA-but-disruptive
    HAS_LEARNABLE_PARAMS: bool = True
    REQUIRES_CUSTOM_MOE: bool = False  # True for #8 expert_choice and #10 soft_moe

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"name={self.ROUTER_NAME}, group={self.ROUTER_GROUP}, "
            f"trainable={self.HAS_LEARNABLE_PARAMS}"
        )


def _zeros_affinities_from_topk(
    router_logits: torch.Tensor,
    expert_index: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Helper: build a dense ``(T, E)`` affinity tensor from a sparse
    ``(T, top_k)`` index + weight pair.

    Mirrors the ``RouterTopK.forward`` path with
    ``apply_act_fn_over_topk=True``: zeros everywhere except at the
    chosen ``(token, expert)`` slots where we write the topk weight.

    Cast back to the router_logits dtype on return (the MoE block
    expects ``expert_affinities`` to share the input hidden-state dtype).
    """
    out = torch.zeros_like(router_logits, dtype=topk_weights.dtype)
    out = out.scatter_(1, expert_index, topk_weights)
    return out


def _normalize_topk_affinities(topk_weights: torch.Tensor) -> torch.Tensor:
    """Renormalize per-token topk weights so they sum to 1.

    Identical to ``ExpertMLPs.normalize_top_k_affinities`` but
    pre-computed at the router level so it shows up in the routing
    pipeline cost model (rather than buried inside the expert MLPs).
    """
    s = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return topk_weights / s


def init_linear_router(
    hidden_size: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool = False,
) -> nn.Linear:
    """Identical construction used by NxD's RouterBase. Exposed here so
    routers that don't inherit from RouterBase (e.g. :class:`HashRouter`
    which has no learnable params, or :class:`SoftMoERouter` which has
    a 3-D weight tensor) can still call into the same init path.
    """
    layer = nn.Linear(hidden_size, num_experts, dtype=dtype, device=device, bias=bias)
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


# Re-export for downstream files that want the symbol but want to avoid
# the lazy-import dance.
__all__ = [
    "SwappableRouter",
    "_zeros_affinities_from_topk",
    "_normalize_topk_affinities",
    "init_linear_router",
    "_import_router_base",
]
