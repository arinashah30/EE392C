"""Router #7: HASH-layer-style routing with zero learnable parameters.

Two families (see ``hash_mode``):

**Paper (Roller et al. 2021, arXiv:2106.04426)** — routes on **input token id**:

  * ``paper_random`` / ``paper_balanced``: fixed lookup ``expert = table[token_id]``.
  * Balanced table: greedy assign frequent tokens to emptiest expert (Sec. 3.1).
  * Build tables with ``bench/build_paper_hash_lookup.py``.

**Hidden projection** — routes on hidden states (like the teacher gate):

  * ``hidden_proj``: single global ``(H,1)`` projection; ``hash_seed`` selects it.
  * ``distilled_hidden``: per-layer ``(H,1)`` projections fit offline to match the
    teacher gate on calibration text (``bench/distill_hash_routing.py``).
  * ``optimize_hash_seed.py`` — global seed search (load-balance or overlap).

Group B floor baseline: no T1 fine-tune (zero learnable params).
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.modules.moe import token_shuffling

from .base import SwappableRouter, _zeros_affinities_from_topk


def _load_paper_lookup(path: str):
    _moe = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _moe not in sys.path:
        sys.path.insert(0, _moe)
    from bench.paper_hash import load_lookup
    return load_lookup(path)


def _hash_mode_family(mode: str) -> str:
    """Normalize hash_mode for metadata compatibility checks."""
    if mode in ("paper_teacher_argmax", "paper_teacher_balanced"):
        return "paper_teacher"
    return mode


_PRIME = 7  # coprime to {8, 16, 60, 64, 128}


def _build_cls():
    from neuronx_distributed.modules.moe.model import MoE
    from neuronx_distributed.modules.moe.routing import RouterBase

    class HashRouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "hash_routing"
        ROUTER_GROUP = "B"
        HAS_LEARNABLE_PARAMS = False
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
            hash_mode: str = "hidden_proj",
            hash_seed: int = 12345,
            hash_lookup_path: Optional[str] = None,
            hash_distill_path: Optional[str] = None,
            layer_idx: int = 0,
            vocab_size: int = 151936,
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
            )
            self.hash_mode = str(hash_mode)
            self.layer_idx = int(layer_idx)
            self.vocab_size = int(vocab_size)
            for p in self.linear_router.parameters():
                p.requires_grad_(False)

            offsets = (
                torch.arange(top_k, device=self.device, dtype=torch.long) * _PRIME
            )
            self.register_buffer("hash_offsets", offsets)
            self.register_buffer("bigram_keys", torch.zeros(0, dtype=torch.long))
            self.register_buffer("bigram_experts", torch.zeros(0, dtype=torch.long))
            self.register_buffer("unigram_lookup", torch.zeros(1, dtype=torch.long))

            if self.hash_mode.startswith("paper_"):
                if not hash_lookup_path or not os.path.isfile(hash_lookup_path):
                    raise ValueError(
                        f"hash_mode={self.hash_mode!r} requires "
                        f"--router-kwarg hash_lookup_path=<file> from "
                        "bench/build_paper_hash_lookup.py or "
                        "bench/build_teacher_hash_lookup.py"
                    )
                lookup, meta, bkeys, bex, buni = _load_paper_lookup(hash_lookup_path)
                built = meta.get("hash_mode", "")
                if built and _hash_mode_family(built) != _hash_mode_family(self.hash_mode):
                    raise ValueError(
                        f"lookup was built for {built!r} but hash_mode={self.hash_mode!r}"
                    )
                self.register_buffer(
                    "token_expert_lookup", lookup.to(self.device).long(),
                )
                if bkeys is not None and bkeys.numel() > 0:
                    self.bigram_keys = bkeys.to(self.device).long()
                    self.bigram_experts = bex.to(self.device).long()
                    self.unigram_lookup = buni.to(self.device).long()
                self.random_proj = None
            elif self.hash_mode == "distilled_hidden":
                path = hash_distill_path or hash_lookup_path
                if not path or not os.path.isfile(path):
                    raise ValueError(
                        "hash_mode='distilled_hidden' requires "
                        "--router-kwarg hash_distill_path=<file> from "
                        "bench/distill_hash_routing.py"
                    )
                obj = torch.load(path, map_location="cpu", weights_only=False)
                meta = obj.get("meta", {})
                if meta.get("hash_mode") not in (None, "distilled_hidden"):
                    raise ValueError(f"distill file hash_mode={meta.get('hash_mode')!r}")
                projs = obj["projections"]
                if projs.dim() == 3 and projs.shape[0] > self.layer_idx:
                    proj = projs[self.layer_idx]
                else:
                    proj = projs.reshape(hidden_size, 1)
                self.register_buffer("random_proj", proj.to(dtype).to(self.device))
                self.token_expert_lookup = None
            else:
                if self.hash_mode != "hidden_proj":
                    raise ValueError(
                        f"unknown hash_mode={self.hash_mode!r}; use hidden_proj, "
                        "distilled_hidden, or paper_*"
                    )
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(hash_seed))
                proj = torch.randn(hidden_size, 1, generator=gen, dtype=dtype)
                self.register_buffer("random_proj", proj.to(self.device))
                self.token_expert_lookup = None

        def _route_from_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
            """``token_ids`` (T,) -> ``expert_index`` (T, top_k)."""
            flat = token_ids.reshape(-1).long()
            lookup = self.token_expert_lookup
            if lookup.dim() == 2:
                base = lookup[self.layer_idx, flat]
            elif self.bigram_keys.numel() > 0:
                prev = torch.roll(flat, 1, 0)
                prev[0] = flat[0]
                pair_key = prev * self.vocab_size + flat
                base = self.unigram_lookup[flat]
                idx = torch.searchsorted(self.bigram_keys, pair_key)
                idx = idx.clamp(max=max(self.bigram_keys.numel() - 1, 0))
                match = self.bigram_keys[idx] == pair_key
                base = torch.where(match, self.bigram_experts[idx], base)
            else:
                base = lookup[flat]
            expert_index = (base.unsqueeze(1) + self.hash_offsets.unsqueeze(0)) % self.num_experts
            return expert_index.to(torch.long)

        def _route_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
            x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float32)
            hash_signal = x @ self.random_proj.to(torch.float32)
            base = (hash_signal.abs() * 1024.0).reshape(-1).to(torch.long) % self.num_experts
            expert_index = (base.unsqueeze(1) + self.hash_offsets.unsqueeze(0)) % self.num_experts
            return expert_index.to(torch.long)

        def forward(
            self,
            hidden_states,
            token_ids: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            T_H = hidden_states.dtype
            if self.hash_mode.startswith("paper_"):
                if token_ids is None:
                    raise RuntimeError(
                        "paper hash modes require token_ids in the traced graph; "
                        "use MoEHash + decoder input_ids plumbing."
                    )
                expert_index = self._route_from_token_ids(token_ids)
            elif self.hash_mode in ("hidden_proj", "distilled_hidden"):
                expert_index = self._route_from_hidden(hidden_states)
            else:
                expert_index = self._route_from_hidden(hidden_states)

            T = expert_index.shape[0]
            topk_weights = torch.full(
                (T, self.top_k),
                1.0 / float(self.top_k),
                dtype=T_H,
                device=hidden_states.device,
            )
            with torch.no_grad():
                router_logits = self.linear_router(
                    hidden_states.reshape(-1, hidden_states.shape[-1])
                )
            expert_affinities = _zeros_affinities_from_topk(
                router_logits, expert_index, topk_weights
            )
            return router_logits, expert_affinities, expert_index

    class MoEHash(MoE):
        """Passes ``routing_token_ids`` into :class:`HashRouter` for paper modes."""

        ROUTER_KIND = "hash_routing"

        def _call_router(self, hidden_states, routing_token_ids=None):
            if routing_token_ids is not None:
                flat = routing_token_ids.reshape(-1)
                return self.router(hidden_states, token_ids=flat)
            return self.router(hidden_states)

        def forward(
            self,
            hidden_states,
            padding_mask=None,
            is_speculative_decoding=False,
            residual=None,
            routing_token_ids=None,
        ):
            seq_len = hidden_states.shape[self.sequence_dimension]
            paper_mode = getattr(self.router, "hash_mode", "").startswith("paper_")
            hidden_hash = getattr(self.router, "hash_mode", "") in (
                "hidden_proj", "distilled_hidden",
            )
            if (
                not paper_mode
                and not hidden_hash
                and (seq_len == 1 or is_speculative_decoding)
                and self.moe_fused_tkg is not None
            ):
                return self.moe_fused_tkg(hidden_states, residual=residual)
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = hidden_states.clone()
                return self._forward_compute_bound(
                    hidden_states, padding_mask, routing_token_ids=routing_token_ids,
                ) + (residual,)
            return self._forward_compute_bound(
                hidden_states, padding_mask, routing_token_ids=routing_token_ids,
            )

        def _forward_compute_bound(self, hidden_states, padding_mask=None, routing_token_ids=None):
            if self.rmsnorm is not None:
                hidden_states = self.rmsnorm(hidden_states)

            shuffle_permutation = None
            if self.token_shuffle_group_size > 1:
                hidden_states, shuffle_permutation = token_shuffling.token_shuffle(
                    hidden_states, seed=self.token_shuffle_seed,
                )
                if self.token_shuffle_seed is not None:
                    self.shuffle_permutation = shuffle_permutation

            s0, s1, _ = hidden_states.shape
            total_tokens = (
                s0 * s1 * self.tensor_parallel_group.size()
                if self.sequence_parallel_enabled else s0 * s1
            )
            use_index_calc_kernel = self.expert_mlps.use_index_calc_kernel(total_tokens)
            expert_affinities_masked_full = None
            early_affinities_ag = (
                use_index_calc_kernel
                and self.sequence_parallel_enabled
                and self.router.sequence_parallel_enabled
            )
            if early_affinities_ag:
                router_logits, expert_affinities, expert_index = self._call_router(
                    hidden_states, routing_token_ids,
                )
                expert_affinities_masked_full = self.expert_mlps.get_full_expert_affinities_masked(
                    expert_affinities, expert_index,
                )

            if self.sequence_parallel_enabled:
                full_hidden_states = mappings.gather_from_sequence_parallel_region(
                    hidden_states,
                    sequence_dimension=self.sequence_dimension,
                    to_model_parallel=False,
                    process_group=self.tensor_parallel_group,
                )
            else:
                full_hidden_states = hidden_states

            full_hidden_states_shape = full_hidden_states.shape
            hidden_states_shape = hidden_states.shape
            seq_len = full_hidden_states_shape[self.sequence_dimension]

            if not early_affinities_ag:
                if self.router.sequence_parallel_enabled:
                    router_logits, expert_affinities, expert_index = self._call_router(
                        hidden_states, routing_token_ids,
                    )
                else:
                    router_logits, expert_affinities, expert_index = self._call_router(
                        full_hidden_states, routing_token_ids,
                    )

            if not self.ep_enabled:
                expert_affinities = mappings.copy_to_tensor_model_parallel_region(
                    expert_affinities,
                )
            full_hidden_states = full_hidden_states.reshape(-1, full_hidden_states_shape[-1])
            output = self.expert_mlps(
                hidden_states=full_hidden_states,
                expert_affinities=expert_affinities,
                expert_index=expert_index,
                seq_len=seq_len,
                padding_mask=padding_mask,
                expert_affinities_masked_full=expert_affinities_masked_full,
            )
            output = self._apply_shared_experts(
                output, full_hidden_states, hidden_states, hidden_states_shape, seq_len,
            )
            output = output.view(full_hidden_states_shape)

            if self.sequence_parallel_enabled:
                if self.ep_enabled:
                    output = mappings.reduce_scatter_to_sequence_parallel_region(
                        output, self.sequence_dimension,
                        process_group=parallel_state.get_world_group(),
                    )
                else:
                    output = mappings.reduce_scatter_to_sequence_parallel_region(
                        output, self.sequence_dimension,
                        process_group=self.tensor_parallel_group,
                    )
            else:
                if self.ep_enabled:
                    output = mappings.reduce_from_tensor_model_parallel_region(
                        output, process_group=parallel_state.get_world_group(),
                    )
                else:
                    output = mappings.reduce_from_tensor_model_parallel_region(
                        output, process_group=self.tensor_parallel_group,
                    )

            if self.token_shuffle_group_size > 1:
                output = token_shuffling.token_unshuffle(output, shuffle_permutation)

            return_op = (output,)
            if self.expert_mlps.return_bias:
                return_op += (None,)
            if self.return_router_logits:
                return_op += (router_logits,)
            if self.return_expert_index:
                return_op += (expert_index,)
            return return_op

    return HashRouter, MoEHash


HashRouter, MoEHash = _build_cls()

__all__ = ["HashRouter", "MoEHash"]
