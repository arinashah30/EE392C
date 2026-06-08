"""Router factory for the Qwen-MoE demo.

Usage from ``neuron_modeling_qwen_moe.py``:

.. code-block:: python

    from qwen_moe.routers import RouterFactory

    router = RouterFactory.create(
        name=neuron_config.router_kernel,
        num_experts=hf_cfg.num_experts,
        top_k=hf_cfg.num_experts_per_tok,
        hidden_size=hf_cfg.hidden_size,
        dtype=hf_cfg.torch_dtype,
        device=torch.device("cpu"),
        sequence_parallel_enabled=False,
        sequence_dimension=1,
        use_nki_router=neuron_config.use_nki_router,
        router_ft_checkpoint=neuron_config.router_ft_checkpoint,
        **(neuron_config.router_kwargs or {}),
    )

    moe = RouterFactory.build_moe(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=shared_experts,
        sequence_parallel_enabled=False,
        sequence_dimension=1,
    )

The factory accepts a ``name`` string ∈ :data:`ROUTER_REGISTRY` (see
the keys for the canonical list) and returns a fully-constructed
:class:`SwappableRouter`.

The factory also exposes :meth:`RouterFactory.build_moe`, which returns
the correct ``MoE`` wrapper for a given router. Most routers slot into
the stock ``neuronx_distributed.modules.moe.MoE`` block, but
:class:`ExpertChoiceRouter` and :class:`SoftMoERouter` change the
dispatch shape and ship paired subclasses (``MoEExpertChoice`` and
``MoESoftMix``) in their module files.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

import torch

from .base import SwappableRouter

# All routers are lazy-imported on demand: the package only needs to
# pay the import cost of the routers it actually instantiates, and
# routers that import optional dependencies (e.g. NKI shells) don't
# break ``from qwen_moe.routers import RouterFactory`` on hosts that
# lack those dependencies.
_ROUTER_MODULES: Dict[str, str] = {
    # Group A: accuracy-preserving (compatible with pretrained Qwen TopK weights).
    "topk_softmax":            "qwen_moe.routers.topk_softmax",
    "topk_softmax_over_topk":  "qwen_moe.routers.topk_softmax_over_topk",
    "topk_sigmoid":            "qwen_moe.routers.topk_sigmoid",
    "group_limited_topk":      "qwen_moe.routers.group_limited_topk",
    "sparsemax_topk":          "qwen_moe.routers.sparsemax_topk",
    # Group B: SOTA-but-disruptive (need router-only fine-tune to recover quality).
    "hierarchical":            "qwen_moe.routers.hierarchical_routing",
    "sinkhorn_balanced":       "qwen_moe.routers.sinkhorn_balanced",
    "hash_routing":            "qwen_moe.routers.hash_routing",
    "expert_choice":           "qwen_moe.routers.expert_choice",
    "base_layer":              "qwen_moe.routers.base_layer",
    "soft_moe":                "qwen_moe.routers.soft_moe",
}


# The class name inside each module that ``create()`` instantiates.
_ROUTER_CLASS_NAMES: Dict[str, str] = {
    "topk_softmax":            "TopKSoftmaxRouter",
    "topk_softmax_over_topk":  "TopKSoftmaxOverTopKRouter",
    "topk_sigmoid":            "TopKSigmoidRouter",
    "group_limited_topk":      "GroupLimitedTopKRouter",
    "sparsemax_topk":          "SparsemaxTopKRouter",
    "hierarchical":            "HierarchicalRouter",
    "sinkhorn_balanced":       "SinkhornBalancedRouter",
    "hash_routing":            "HashRouter",
    "expert_choice":           "ExpertChoiceRouter",
    "base_layer":              "BaseLayerRouter",
    "soft_moe":                "SoftMoERouter",
}


# Routers that require a custom MoE wrapper. The factory's
# :meth:`build_moe` consults this to pick the right wrapper class.
_CUSTOM_MOE_MODULES: Dict[str, str] = {
    "expert_choice": "qwen_moe.routers.expert_choice",
    "soft_moe":      "qwen_moe.routers.soft_moe",
    "hash_routing":  "qwen_moe.routers.hash_routing",
}


_CUSTOM_MOE_CLASS_NAMES: Dict[str, str] = {
    "expert_choice": "MoEExpertChoice",
    "soft_moe":      "MoESoftMix",
    "hash_routing":  "MoEHash",
}


ROUTER_REGISTRY = tuple(_ROUTER_MODULES.keys())


def _load_router_class(name: str) -> Type[SwappableRouter]:
    """Resolve ``name`` to a concrete router class via lazy import."""
    import importlib
    if name not in _ROUTER_MODULES:
        raise KeyError(
            f"Unknown router kernel {name!r}. "
            f"Available: {sorted(_ROUTER_MODULES)}"
        )
    module = importlib.import_module(_ROUTER_MODULES[name])
    cls = getattr(module, _ROUTER_CLASS_NAMES[name])
    return cls


def _load_custom_moe_class(name: str):
    """Resolve the MoE wrapper class for routers that change dispatch."""
    import importlib
    module = importlib.import_module(_CUSTOM_MOE_MODULES[name])
    return getattr(module, _CUSTOM_MOE_CLASS_NAMES[name])


def _maybe_load_finetune_checkpoint(
    router: SwappableRouter, checkpoint_path: Optional[str]
) -> None:
    """Merge a router-only fine-tune state dict into the router.

    Layout: ``{"layers.{l}.mlp.router.linear_router.weight": tensor}``
    keyed exactly as :func:`convert_qwen_moe_to_neuron_state_dict`
    emits. The same checkpoint can carry weights for all 24 layers; we
    only read the keys that match this router's own parameter names.

    Silently no-ops if ``checkpoint_path`` is None or empty so the
    default (Tier-T0, no fine-tune) path is unchanged.
    """
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"--router-ft-checkpoint points at {checkpoint_path!r} which does not exist."
        )

    try:
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    except ImportError:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # State-dict keys may be either fully-qualified (with the
    # ``layers.{l}.mlp.router.`` prefix from the NxD layout) or local
    # to the router module. Strip the prefix so ``load_state_dict``
    # below works against the router's own ``named_parameters``.
    local_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if ".router." in k:
            local_state[k.split(".router.", 1)[1]] = v
        else:
            local_state[k] = v

    missing, unexpected = router.load_state_dict(local_state, strict=False)
    # Routers with no learnable params (HashRouter) won't have any keys
    # to consume; that's a no-op, not an error.
    if missing and router.HAS_LEARNABLE_PARAMS:
        raise RuntimeError(
            f"Fine-tune checkpoint at {checkpoint_path!r} missing keys: {missing}"
        )


class RouterFactory:
    """Factory class for the 10 swappable Qwen-MoE routers.

    Kept as a class (rather than a free function) so the test plan and
    profiling scripts can introspect :data:`ROUTER_REGISTRY` and
    :meth:`group_of`/``has_learnable_params`` without instantiating.
    """

    @staticmethod
    def registry() -> tuple:
        return ROUTER_REGISTRY

    @staticmethod
    def group_of(name: str) -> str:
        return _load_router_class(name).ROUTER_GROUP

    @staticmethod
    def has_learnable_params(name: str) -> bool:
        return _load_router_class(name).HAS_LEARNABLE_PARAMS

    @staticmethod
    def requires_custom_moe(name: str) -> bool:
        return _load_router_class(name).REQUIRES_CUSTOM_MOE

    @staticmethod
    def create(
        name: str,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        use_nki_router: bool = False,
        router_ft_checkpoint: Optional[str] = None,
        **router_kwargs: Any,
    ) -> SwappableRouter:
        """Construct the router for the given ``name``.

        ``router_kwargs`` is the per-router escape hatch (e.g.
        ``n_group`` / ``topk_group`` for group_limited_topk, ``num_slots``
        for soft_moe). The wire-up code in :mod:`run_qwen_moe_trn2`
        passes whatever the user provided on the CLI.
        """
        cls = _load_router_class(name)
        if device is None:
            device = torch.device("cpu")

        router = cls(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            use_nki_router=use_nki_router,
            **router_kwargs,
        )

        _maybe_load_finetune_checkpoint(router, router_ft_checkpoint)
        return router

    @staticmethod
    def build_moe(
        *,
        router: SwappableRouter,
        expert_mlps,
        shared_experts=None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        return_router_logits: bool = False,
        return_expert_index: bool = False,
        **moe_kwargs: Any,
    ):
        """Build the appropriate ``MoE`` wrapper for ``router``.

        - Routers with ``REQUIRES_CUSTOM_MOE = False`` get the stock
          ``neuronx_distributed.modules.moe.MoE``.
        - Routers with ``REQUIRES_CUSTOM_MOE = True`` get the
          per-router subclass (``MoEExpertChoice`` for expert_choice,
          ``MoESoftMix`` for soft_moe).
        """
        if router.REQUIRES_CUSTOM_MOE:
            moe_cls = _load_custom_moe_class(router.ROUTER_NAME)
        else:
            from neuronx_distributed.modules.moe.model import MoE as moe_cls

        return moe_cls(
            router=router,
            expert_mlps=expert_mlps,
            shared_experts=shared_experts,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            return_router_logits=return_router_logits,
            return_expert_index=return_expert_index,
            **moe_kwargs,
        )


__all__ = [
    "RouterFactory",
    "ROUTER_REGISTRY",
    "SwappableRouter",
]
