"""Router #10: SoftMoE — slot-based dense soft mixture of experts.

SoftMoE (Puigcerver et al. 2023, "From Sparse to Soft Mixtures of
Experts") removes the hard ``top_k`` selection entirely. Each expert
owns ``num_slots`` learned dispatch slots; each slot is a soft
weighted average over **all** tokens. The expert MLP processes each
slot as if it were a token. Per-token output is then reconstructed
as a soft weighted average over **all** ``E * num_slots`` slot
outputs.

Algorithm (Section 2 of the paper):

  Notation: T tokens, H hidden, E experts, S slots/expert. Slot
  weight matrix Φ ∈ R^{H × (E·S)}.

  1. ``phi = hidden_states @ slot_weights``        shape (T, E·S)
  2. ``D = softmax(phi, dim=0)``                   "dispatch", shape (T, E·S)
                                                   (column-softmax over tokens)
  3. ``C = softmax(phi, dim=1)``                   "combine",  shape (T, E·S)
                                                   (row-softmax over slots)
  4. ``X_slot = D.T @ hidden_states``              shape (E·S, H)
  5. Reshape: ``X_slot`` ∈ R^{E × S × H}, run the expert MLPs in
     parallel (each expert sees S slot-vectors as input):
     ``Y_slot = ExpertMLP(X_slot)``                shape (E, S, H)
  6. ``Y = Y_slot.reshape(E·S, H)``
  7. ``output = C @ Y``                            shape (T, H)

Properties:

  * **No hard top_k.** Every expert contributes a soft amount to
    every token output. The compute is *dense* over all E experts
    (i.e. no token-to-expert sparsity), so total FLOPs are
    ``E·S·H·(2I)`` regardless of effective "activated" experts.
  * **Slots replace experts as the unit of dispatch.** ``E·S`` is
    chosen to match ``T · top_k`` so the slot count is roughly the
    same as the token-expert assignments in standard TopK MoE.
  * **Differentiable end-to-end.** Both D and C are smooth softmaxes,
    so T1 fine-tuning is well-defined without straight-through
    estimators.
  * **Slowest router on Trn2.** The dense expert pass is what makes
    SoftMoE expensive — but the router itself is just two GEMMs
    (the H × (E·S) projection plus the two softmaxes), so the
    *router*-side cost is modest. Expected: top throughput-per-watt
    on small T, falls off on large T.

Group B: changes the entire MoE layer's input/output relationship.
T0 inference on pretrained Qwen will produce broken outputs (the
experts were never trained against soft-mix dispatch). T1 router
fine-tune helps but typically leaves SoftMoE 2–5 MMLU points below
TopK on Qwen scale — this is the canonical "needs T2 LoRA on
experts" router.

Implementation note: the standard ``MoE._forward_compute_bound``
doesn't fit SoftMoE's shape contract (no per-token expert
selection). We ship a :class:`MoESoftMix` subclass with its own
``_forward_compute_bound`` that consumes the router's slot
dispatch + combine weights directly. The router's standard
``forward()`` returns *placeholder* (T, E)-shaped affinities so
isolated diagnostic tools (``analyze_routing.py``) still work.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import SwappableRouter


def _build_classes():
    from neuronx_distributed.modules.moe.routing import RouterBase
    from neuronx_distributed.modules.moe.model import MoE
    from neuronx_distributed.parallel_layers import mappings, parallel_state

    class SoftMoERouter(SwappableRouter, RouterBase):
        ROUTER_NAME = "soft_moe"
        ROUTER_GROUP = "B"
        HAS_LEARNABLE_PARAMS = True
        REQUIRES_CUSTOM_MOE = True

        def __init__(
            self,
            *,
            num_experts: int,
            top_k: int,  # Used to size num_slots if num_slots is unset.
            hidden_size: int,
            num_slots: Optional[int] = None,
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

            # Default to S = top_k so total slots E*S ≈ top_k * T for
            # T = E (i.e. about one slot per expert per top-k position).
            # Most published SoftMoE recipes use S = top_k or S = top_k+1.
            self.num_slots = int(num_slots) if num_slots is not None else top_k

            # The standard linear_router (H, E) is left in place but
            # not used by the SoftMoE forward; we override its output
            # in `forward()` only to satisfy the SwappableRouter
            # contract for diagnostic tools.

            # The real SoftMoE projection: slot_weights ∈ R^{H × (E·S)}.
            self.slot_weights = nn.Parameter(
                torch.empty(
                    hidden_size,
                    num_experts * self.num_slots,
                    dtype=dtype,
                    device=self.device,
                ),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(self.slot_weights, a=math.sqrt(5))

        def compute_dispatch_combine(
            self, hidden_flat: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """The real soft-MoE projection used by :class:`MoESoftMix`.

            Returns a 3-tuple ``(D, C, phi)``:
              * ``D``: dispatch weights, shape ``(T, E·S)``, softmax over tokens.
              * ``C``: combine weights, shape ``(T, E·S)``, softmax over slots.
              * ``phi``: raw slot logits, shape ``(T, E·S)`` (returned for
                logging / load-balance diagnostics).
            """
            orig_dtype = hidden_flat.dtype
            x = hidden_flat.to(self.slot_weights.dtype)
            phi = x @ self.slot_weights  # (T, E·S)

            # softmax over tokens (column) and over slots (row) -- both
            # fp32 for stability across the large dimensions.
            phi_fp = phi.to(torch.float32)
            D = F.softmax(phi_fp, dim=0).to(orig_dtype)
            C = F.softmax(phi_fp, dim=1).to(orig_dtype)
            return D, C, phi

        def forward(
            self, hidden_states
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """SwappableRouter contract stub.

            SoftMoE doesn't use per-token expert selection, so this
            return is a *placeholder* used only by the host-side
            diagnostic tools (``analyze_routing.py``). The real
            dispatch happens inside :class:`MoESoftMix._forward_compute_bound`.

            The placeholder reports an "effective" expert assignment:
            for each token, take the topk experts (across summed slot
            weights) so the routing-entropy histograms still produce
            interpretable output.
            """
            T_H_dtype = hidden_states.dtype
            hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            D, C, phi = self.compute_dispatch_combine(hidden_flat)

            # Summarize per-token affinity across slots: sum the combine
            # weights belonging to each expert -> (T, E).
            T = hidden_flat.shape[0]
            C_grouped = C.view(T, self.num_experts, self.num_slots).sum(dim=2)
            # Use phi reduced over slots as router_logits proxy.
            phi_grouped = phi.view(T, self.num_experts, self.num_slots).sum(dim=2)

            topk_weights, expert_index = torch.topk(C_grouped, self.top_k, dim=1)
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-9)
            expert_affinities = torch.zeros_like(C_grouped)
            expert_affinities = expert_affinities.scatter_(
                1, expert_index, topk_weights
            ).to(T_H_dtype)
            return phi_grouped, expert_affinities, expert_index.to(torch.long)

    class MoESoftMix(MoE):
        """MoE wrapper that overrides ``_forward_compute_bound`` with
        the dense slot-based dispatch + soft combine described in
        Puigcerver et al. 2023.

        Mirrors the optional behaviors of NxD's ``MoE``:
          * input/output reshape between (B, S, H) and (T, H);
          * shared-experts add at the end (via the inherited
            :meth:`_apply_shared_experts` helper);
          * delayed all-reduce when TP > 1.
        Skipped (don't apply to soft-MoE):
          * token shuffling (incompatible with the dense slot path);
          * the index-calc fast path / blockwise NKI kernel (sparse
            only).
        """

        ROUTER_KIND = "soft_moe"

        def _forward_compute_bound(self, hidden_states, padding_mask=None):
            if self.rmsnorm is not None:
                hidden_states = self.rmsnorm(hidden_states)

            # --- bring hidden_states to (T, H) in the same shape as MoE does ---
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

            hidden_flat = full_hidden_states.reshape(-1, full_hidden_states_shape[-1])
            T = hidden_flat.shape[0]
            H = hidden_flat.shape[-1]
            E = self.router.num_experts
            S = self.router.num_slots

            # --- 1. Dispatch & combine weights ---
            D, C, phi = self.router.compute_dispatch_combine(hidden_flat)
            # D, C: (T, E·S)

            # --- 2. Slot inputs ---
            # X_slot[:, h] = D.T @ hidden_flat[:, h]
            # shape: (E·S, H)
            slot_inputs = D.transpose(0, 1) @ hidden_flat  # (E·S, H)
            slot_inputs = slot_inputs.view(E, S, H).contiguous()

            # --- 3. Per-expert MLP over slots ---
            # ExpertMLPs' mlp_op expects shape (E or 1, C, H). With
            # input (E, S, H) it returns (E, S, H).
            mlp_op = self.expert_mlps.get_mlp_op()
            slot_outputs = mlp_op(slot_inputs, expert_indices=None)
            # If the underlying mlp_op happens to add a leading dim
            # (some EP paths do), squeeze it back.
            if slot_outputs.dim() == 4 and slot_outputs.shape[0] == 1:
                slot_outputs = slot_outputs.squeeze(0)
            # slot_outputs: (E, S, H)
            slot_outputs = slot_outputs.reshape(E * S, H)

            # --- 4. Combine slot outputs into per-token output ---
            output = C @ slot_outputs  # (T, H)

            # --- 5. Add shared experts (reuses MoE._apply_shared_experts) ---
            output = self._apply_shared_experts(
                output, hidden_flat, hidden_states,
                hidden_states_shape, seq_len,
            )
            output = output.view(full_hidden_states_shape)

            # --- 6. Sequence-parallel / TP all-reduce, mirroring MoE ---
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

            return_op = (output,)
            if self.expert_mlps.return_bias:
                return_op += (None,)
            if self.return_router_logits:
                return_op += (phi,)
            if self.return_expert_index:
                # Placeholder: SoftMoE doesn't have a hard expert_index.
                # Return the router's stub (top_k experts by combined
                # slot weight) so downstream code that expects a
                # (T, top_k) integer tensor still works.
                _, eff_idx = torch.topk(
                    C.view(T, E, S).sum(dim=2), self.router.top_k, dim=1
                )
                return_op += (eff_idx.to(torch.long),)
            return return_op

    return SoftMoERouter, MoESoftMix


SoftMoERouter, MoESoftMix = _build_classes()


__all__ = ["SoftMoERouter", "MoESoftMix"]
