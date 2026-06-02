# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PyTorch Qwen1.5-MoE / Qwen2-MoE model for NXD inference.

Cloned from examples/inference/mixtral/neuron_modeling_mixtral.py and
adapted for Qwen2-MoE-style architecture:

  * 60 routed experts (top-k=4) with per-expert intermediate size 1408
  * one shared-expert MLP (modeled in NxD as N shared sub-experts whose
    fused intermediate size matches HF's shared_expert_intermediate_size)
  * a *per-token sigmoid scalar gate* over the shared-expert output (this
    is unique to Qwen2-MoE; NxD's stock SharedExperts has no such gate, so
    we subclass it as :class:`GatedSharedExperts`)
  * QKV projections with bias enabled (Qwen2 default qkv_bias=True)
  * MHA at the default Qwen1.5-MoE-A2.7B size (16 attn = 16 KV heads); the
    GQA wiring still works because num_kv_heads == num_attn_heads
"""
import gc
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from modules.custom_calls import CustomRMSNorm
from modules.gqa import (
    GQA,
    BaseGroupQueryAttention,
)
from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed.utils.sampling import Sampler

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import Qwen2MoeForCausalLM, Qwen2MoePreTrainedModel
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeRMSNorm

from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.utils import RotaryEmbedding
from modules.config import MoENeuronConfig

from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.parallel_layers import parallel_state, utils
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)

# Swappable router factory: picks one of the 10 router kernels defined
# under ``qwen_moe/routers/`` based on neuron_config.router_kernel. Default
# (``"topk_softmax"``) reproduces the original RouterTopK construction
# below so existing NEFF cache artifacts continue to match.
from qwen_moe.routers import RouterFactory

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def _num_shared_experts(hf_config) -> int:
    """Translate HF's monolithic shared expert (intermediate=shared_expert_intermediate_size)
    into NxD's "N sub-experts of size moe_intermediate_size" form. NxD's SharedExperts
    just multiplies these together: expert_dim = intermediate_size * num_shared_experts.
    Mathematically identical, but the user-facing knob mirrors how many "shared experts"
    Qwen describes (4 for Qwen1.5-MoE-A2.7B: 5632 / 1408 = 4).
    """
    n = hf_config.shared_expert_intermediate_size // hf_config.moe_intermediate_size
    if n * hf_config.moe_intermediate_size != hf_config.shared_expert_intermediate_size:
        raise ValueError(
            f"shared_expert_intermediate_size ({hf_config.shared_expert_intermediate_size}) "
            f"is not a multiple of moe_intermediate_size ({hf_config.moe_intermediate_size}); "
            f"cannot factor into NxD num_shared_experts."
        )
    return n


class GatedSharedExperts(SharedExperts):
    """NxD SharedExperts + Qwen's per-token sigmoid scalar gate.

    HF reference (Qwen2MoeSparseMoeBlock):

        shared = self.shared_expert(hidden_states)
        shared = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared

    `shared_expert_gate` is a single Linear(hidden, 1, bias=False); its
    weight is replicated across TP ranks (no need to shard one row).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_expert_gate = nn.Linear(
            self.hidden_size, 1, bias=False, dtype=self.dtype
        )

    def forward(self, x, seq_len):
        gate_score = torch.sigmoid(self.shared_expert_gate(x))
        out = super().forward(x, seq_len)
        return gate_score * out


def convert_qwen_moe_to_neuron_state_dict(neuron_state_dict, neuron_config):
    """HF Qwen2MoE -> NxD MoE state-dict layout.

    Per layer ``l``:

      * ``mlp.gate.weight``                 -> ``layers.{l}.mlp.router.linear_router.weight``
      * If ``neuron_config.router_kernel == "hierarchical"``, also adds
        ``layers.{l}.mlp.router.linear_group.weight`` (shape ``[n_group, H]``)
        since HF checkpoints have no group head; initialized to zeros so you
        can recompile/load; replace with a fine-tuned tensor when available.
      * for each routed expert ``e`` (0..num_experts-1):
            ``mlp.experts.{e}.gate_proj.weight`` (shape [I, H], I=moe_intermediate_size)
          + ``mlp.experts.{e}.up_proj.weight``   (same shape)
          fused as ``mlp.expert_mlps.mlp_op.gate_up_proj.weight`` of shape
          (E, H, 2*I) with [..., :I] = gate, [..., I:] = up.
            ``mlp.experts.{e}.down_proj.weight`` ([H, I]) packed as
          ``mlp.expert_mlps.mlp_op.down_proj.weight`` of shape (E, I, H).
      * Shared expert: ``mlp.shared_expert.{gate,up,down}_proj.weight`` ->
        ``mlp.shared_experts.{gate,up,down}_proj.weight`` (layouts already
        match because expert_dim = num_shared_experts * moe_intermediate_size
        = shared_expert_intermediate_size).
      * ``mlp.shared_expert_gate.weight`` ->
        ``mlp.shared_experts.shared_expert_gate.weight``.

    Q/K/V biases pass through unchanged; the GQA preshard hook reads
    ``self_attn.{q,k,v}_proj.bias`` directly.
    """
    assert neuron_config.glu_mlp is True, "Only GLU MLP is supported for Qwen2-MoE"

    hf_cfg = neuron_config.hf_config
    num_experts = hf_cfg.num_experts

    expert_ft_checkpoint = getattr(neuron_config, "expert_ft_checkpoint", None)
    if expert_ft_checkpoint:
        neuron_state_dict = _apply_expert_lora_overlay(
            neuron_state_dict, expert_ft_checkpoint,
        )

    for l in range(hf_cfg.num_hidden_layers):  # noqa: E741
        # --- attention: Qwen2 has qkv_bias=True but no o_proj bias. The
        # NeuronAttentionBase uses a single `self.bias` flag for both QKV
        # and O projections, so the traced graph allocates an o_proj.bias
        # parameter that the HF state dict lacks. Inject a zero tensor so
        # weight-loading succeeds (a zero bias is mathematically a no-op).
        o_proj_w_key = f"layers.{l}.self_attn.o_proj.weight"
        o_proj_b_key = f"layers.{l}.self_attn.o_proj.bias"
        if o_proj_b_key not in neuron_state_dict:
            o_proj_w = neuron_state_dict[o_proj_w_key]
            neuron_state_dict[o_proj_b_key] = torch.zeros(
                o_proj_w.shape[0], dtype=o_proj_w.dtype, device=o_proj_w.device,
            )

        # --- router ---
        gate_w = neuron_state_dict[f"layers.{l}.mlp.gate.weight"]
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            gate_w.detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.mlp.gate.weight"]

        router_kernel = getattr(neuron_config, "router_kernel", "topk_softmax")
        if router_kernel == "hierarchical":
            rk = getattr(neuron_config, "router_kwargs", None) or {}
            n_group = int(rk.get("n_group", 12))
            hidden_size = hf_cfg.hidden_size
            # nn.Linear(hidden, n_group) -> weight (n_group, hidden); HF has no analogue.
            neuron_state_dict[f"layers.{l}.mlp.router.linear_group.weight"] = torch.zeros(
                n_group,
                hidden_size,
                dtype=gate_w.dtype,
                device=gate_w.device,
            )
            # Optional bias if HierarchicalRouter is constructed with bias=True
            if bool(rk.get("bias")):
                neuron_state_dict[f"layers.{l}.mlp.router.linear_group.bias"] = torch.zeros(
                    n_group, dtype=gate_w.dtype, device=gate_w.device
                )

        # --- routed experts: fuse gate_proj + up_proj into gate_up_proj ---
        sample_key = f"layers.{l}.mlp.experts.0.gate_proj.weight"
        intermediate_size, hidden_size = neuron_state_dict[sample_key].shape
        device = neuron_state_dict[sample_key].device
        dtype = neuron_state_dict[sample_key].dtype

        gate_up_proj = torch.empty(
            num_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device
        )
        for e in range(num_experts):
            gate_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"].T.detach().clone()
            )
            up_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"].T.detach().clone()
            )

            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
            up_proj_slice.copy_(up_proj_weights)

            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.gate_proj.weight"]
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.up_proj.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # --- routed experts: down_proj ---
        down_proj = torch.empty(
            num_experts, intermediate_size, hidden_size, dtype=dtype, device=device
        )
        for e in range(num_experts):
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"].T.detach().clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.mlp.experts.{e}.down_proj.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        # --- shared experts (rename only; layouts already match) ---
        for proj in ("gate_proj", "up_proj", "down_proj"):
            src = f"layers.{l}.mlp.shared_expert.{proj}.weight"
            dst = f"layers.{l}.mlp.shared_experts.{proj}.weight"
            neuron_state_dict[dst] = neuron_state_dict[src].detach().clone()
            del neuron_state_dict[src]

        # --- per-token sigmoid gate over shared-expert output ---
        src_gate = f"layers.{l}.mlp.shared_expert_gate.weight"
        dst_gate = f"layers.{l}.mlp.shared_experts.shared_expert_gate.weight"
        neuron_state_dict[dst_gate] = neuron_state_dict[src_gate].detach().clone()
        del neuron_state_dict[src_gate]

        gc.collect()

    # Tier-T1 router fine-tune overlay (see josh/moe_demo bench REPORT §6).
    router_ft_checkpoint = getattr(neuron_config, "router_ft_checkpoint", None)
    if router_ft_checkpoint:
        neuron_state_dict = _apply_ft_overlay(
            neuron_state_dict,
            router_ft_checkpoint,
            label="router",
        )

    return neuron_state_dict


def _apply_expert_lora_overlay(
    neuron_state_dict: dict,
    checkpoint_path: str,
) -> dict:
    """Merge LoRA adapter checkpoints into HF expert weights (before NxD fusion)."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"--expert-ft-checkpoint {checkpoint_path!r} does not exist."
        )
    meta_path = checkpoint_path + ".json"
    scaling = None
    if os.path.isfile(meta_path):
        import json
        meta = json.load(open(meta_path, encoding="utf-8"))
        scaling = float(meta.get("lora_scaling", 0))
        if scaling <= 0 and meta.get("lora_alpha") and meta.get("lora_rank"):
            scaling = float(meta["lora_alpha"]) / float(meta["lora_rank"])

    try:
        from safetensors.torch import load_file as _load_safetensors
        ft_state = _load_safetensors(checkpoint_path)
    except ImportError:
        ft_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    lora_a_keys = [k for k in ft_state if k.endswith(".lora_A")]
    if not lora_a_keys:
        return _apply_ft_overlay(
            neuron_state_dict,
            checkpoint_path,
            label="expert_merged",
            key_filter=lambda k: ".mlp.experts." in k and k.endswith(".weight"),
        )

    if scaling is None:
        scaling = 4.0

    n_merged = 0
    for a_key in lora_a_keys:
        b_key = a_key.replace(".lora_A", ".lora_B")
        w_key = a_key.replace(".lora_A", ".weight")
        if b_key not in ft_state or w_key not in neuron_state_dict:
            continue
        lora_a = ft_state[a_key].float()
        lora_b = ft_state[b_key].float()
        weight = neuron_state_dict[w_key].float()
        delta = (lora_b @ lora_a) * scaling
        neuron_state_dict[w_key] = (weight + delta).to(
            dtype=neuron_state_dict[w_key].dtype,
            device=neuron_state_dict[w_key].device,
        )
        n_merged += 1
    print(
        f"[ft-overlay:expert_lora] merged {n_merged} expert projections from "
        f"{checkpoint_path!r} (scaling={scaling})",
        flush=True,
    )
    return neuron_state_dict


def _apply_ft_overlay(
    neuron_state_dict: dict,
    checkpoint_path: str,
    *,
    label: str,
    key_filter=None,
) -> dict:
    """Override selected keys in ``neuron_state_dict`` from a safetensors file."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"FT checkpoint {checkpoint_path!r} does not exist ({label})."
        )
    try:
        from safetensors.torch import load_file as _load_safetensors
        ft_state = _load_safetensors(checkpoint_path)
    except ImportError:
        ft_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    n_overrides = 0
    for ft_key, ft_tensor in ft_state.items():
        if key_filter is not None and not key_filter(ft_key):
            continue
        if ft_key not in neuron_state_dict:
            print(
                f"[ft-overlay:{label}] WARNING: skip unknown key {ft_key!r}",
                flush=True,
            )
            continue
        target = neuron_state_dict[ft_key]
        ft_tensor = ft_tensor.to(dtype=target.dtype, device=target.device)
        if ft_tensor.shape != target.shape:
            raise RuntimeError(
                f"[ft-overlay:{label}] shape mismatch on {ft_key!r}: "
                f"FT {tuple(ft_tensor.shape)} vs neuron {tuple(target.shape)}"
            )
        neuron_state_dict[ft_key] = ft_tensor
        n_overrides += 1
    print(
        f"[ft-overlay:{label}] applied {n_overrides} tensors from {checkpoint_path!r}",
        flush=True,
    )
    return neuron_state_dict


def get_rmsnorm_cls(neuron_config):
    # On NXD use the fused custom kernel; on CPU fall back to HF RMSNorm.
    return Qwen2MoeRMSNorm if neuron_config.on_cpu else CustomRMSNorm


class NeuronQwenMoeAttention(NeuronAttentionBase):
    def __init__(self, neuron_config: MoENeuronConfig):
        super().__init__()
        self.neuron_config = neuron_config
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.num_attention_heads = neuron_config.hf_config.num_attention_heads
        self.num_key_value_heads = neuron_config.hf_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = neuron_config.hf_config.max_position_embeddings
        self.rope_theta = neuron_config.hf_config.rope_theta
        self.padding_side = neuron_config.padding_side
        self.torch_dtype = neuron_config.hf_config.torch_dtype

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwenMoeAttention has to be initialized in a distributed env. Please use "
                "neuronx_distributed module to initialize a distributed env."
            )
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.fused_qkv = False
        self.clip_qkv = None
        # Qwen2 has biases on Q/K/V projections (qkv_bias=True by default).
        # The o_proj has no bias in HF Qwen2 -- but GQA shares one self.bias flag for
        # both QKV and O. The o_proj bias allocated here will be left as zeros (HF
        # never reads/writes it), which is numerically a no-op.
        self.bias = getattr(neuron_config.hf_config, "qkv_bias", True)

        self.init_gqa_properties()

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class NeuronQwenMoeDecoderLayer(nn.Module):
    """Decoder layer wired with NxD's MoE module + GatedSharedExperts."""

    def __init__(self, neuron_config: MoENeuronConfig, layer_idx: int):
        super().__init__()
        hf_cfg = neuron_config.hf_config
        self.hidden_size = hf_cfg.hidden_size
        self.self_attn = NeuronQwenMoeAttention(neuron_config=neuron_config)

        # Build the router via the swappable factory. The default
        # router_kernel is "topk_softmax" which is byte-equivalent to
        # constructing RouterTopK directly (the factory's
        # ``topk_softmax`` subclass adds a thin SwappableRouter mixin
        # only). Other 9 kernels are documented in
        # ``moe_demo/qwen_moe/routers/README.md``.
        router_kernel = getattr(neuron_config, "router_kernel", "topk_softmax")
        self._router_kernel = router_kernel
        use_nki_router = getattr(neuron_config, "use_nki_router", False)
        router_ft_checkpoint = getattr(neuron_config, "router_ft_checkpoint", None)
        router_kwargs = dict(getattr(neuron_config, "router_kwargs", None) or {})
        router = RouterFactory.create(
            name=router_kernel,
            num_experts=hf_cfg.num_experts,
            top_k=hf_cfg.num_experts_per_tok,
            hidden_size=hf_cfg.hidden_size,
            dtype=hf_cfg.torch_dtype,
            device=torch.device("cpu"),
            sequence_parallel_enabled=False,
            sequence_dimension=1,
            use_nki_router=use_nki_router,
            router_ft_checkpoint=router_ft_checkpoint,
            layer_idx=layer_idx,
            vocab_size=hf_cfg.vocab_size,
            **router_kwargs,
        )
        # block_size large enough that
        #   total_tokens * top_k < block_size
        # for the compiled prefill shape (=> ExpertMLPsV2.forward dispatches
        # to forward_all_experts instead of forward_blockwise). The NxD
        # inference venv ships only the shard-on-intermediate / shard-on-block
        # kernels; the shard-hidden one used by forward_blockwise at LNC=2 is
        # a NotImplementedError stub. forward_all_experts has higher FLOPs
        # (no token-to-expert sparsity) but is correct and well-supported
        # for the small prefill shapes this wrapper targets.
        max_prefill_tokens = max(neuron_config.batch_size, 1) * max(neuron_config.max_context_length, 1)
        block_size = max(2 * max_prefill_tokens * hf_cfg.num_experts_per_tok, 512)
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
        expert_mlps = ExpertMLPs(
            num_experts=hf_cfg.num_experts,
            top_k=hf_cfg.num_experts_per_tok,
            hidden_size=hf_cfg.hidden_size,
            intermediate_size=hf_cfg.moe_intermediate_size,
            hidden_act=hf_cfg.hidden_act,
            glu_mlp=neuron_config.glu_mlp,
            capacity_factor=neuron_config.capacity_factor,
            normalize_top_k_affinities=hf_cfg.norm_topk_prob,
            block_size=block_size,
            logical_nc_config=lnc,
        )
        shared_experts = GatedSharedExperts(
            hidden_size=hf_cfg.hidden_size,
            intermediate_size=hf_cfg.moe_intermediate_size,
            num_shared_experts=_num_shared_experts(hf_cfg),
            hidden_act=hf_cfg.hidden_act,
            dtype=hf_cfg.torch_dtype,
        )
        # Use RouterFactory.build_moe so routers that need a custom MoE
        # wrapper (expert_choice, soft_moe) pick the right subclass.
        # For all other routers this returns the standard
        # ``neuronx_distributed.modules.moe.model.MoE``.
        self.mlp = RouterFactory.build_moe(
            router=router,
            expert_mlps=expert_mlps,
            shared_experts=shared_experts,
            sequence_parallel_enabled=False,
            sequence_dimension=1,
        )
        self.mlp.eval()

        self.input_layernorm = get_rmsnorm_cls(neuron_config)(
            hf_cfg.hidden_size,
            eps=hf_cfg.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls(neuron_config)(
            hf_cfg.hidden_size,
            eps=hf_cfg.rms_norm_eps,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            input_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated; use `attention_mask` instead."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Only MoEHash (hash_routing) accepts routing_token_ids; stock MoE does not.
        if input_ids is not None and self._router_kernel == "hash_routing":
            hidden_states = self.mlp(
                hidden_states, routing_token_ids=input_ids,
            )[0]
        else:
            hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value)


class NeuronQwenMoeModel(NeuronBaseModel, Qwen2MoePreTrainedModel):
    """Traceable Qwen2-MoE backbone."""

    _model_cls = Qwen2MoePreTrainedModel

    def get_model_output(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            active_mask: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Same as ``NeuronBaseModel.get_model_output`` but passes ``input_ids``
        into each decoder layer for paper-style hash routing."""
        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        hidden_states = inputs_embeds
        next_decoder_cache = ()
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                input_ids=input_ids,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        return (hidden_states, next_decoder_cache)

    def setup_attr_for_model(self, neuron_config: MoENeuronConfig):
        self.on_device_sampling = neuron_config.on_device_sampling
        self.tp_degree = neuron_config.tp_degree
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.num_attention_heads = neuron_config.hf_config.num_attention_heads
        self.num_key_value_heads = neuron_config.hf_config.num_key_value_heads
        self.max_batch_size = neuron_config.max_batch_size
        self.buckets = neuron_config.buckets

    def init_model(self, neuron_config: MoENeuronConfig):
        hf_cfg = neuron_config.hf_config
        self.padding_idx = hf_cfg.pad_token_id
        self.vocab_size = hf_cfg.vocab_size

        self.embed_tokens = ParallelEmbedding(
            hf_cfg.vocab_size,
            hf_cfg.hidden_size,
            self.padding_idx,
            dtype=hf_cfg.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronQwenMoeDecoderLayer(neuron_config, layer_idx)
             for layer_idx in range(hf_cfg.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls(neuron_config)(self.hidden_size, eps=hf_cfg.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(hf_cfg.hidden_size, self.vocab_size, bias=False)


class NeuronQwenMoeForCausalLM(NeuronBaseForCausalLM, Qwen2MoePreTrainedModel):
    """Drop-in replacement for Qwen2MoeForCausalLM that runs on Neuron."""

    _model_cls = NeuronQwenMoeModel

    def __init__(self, model_path: str, neuron_config: MoENeuronConfig):
        super().__init__(model_path, neuron_config)
        self.sampler = Sampler(neuron_config)

    @staticmethod
    def load_hf_model(model_path):
        return Qwen2MoeForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, neuron_config: MoENeuronConfig) -> dict:
        return convert_qwen_moe_to_neuron_state_dict(state_dict, neuron_config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.neuron_config.hf_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        # MoE NEFFs on trn2 default to platform LNC=2 unless --lnc is explicit.
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
        compiler_args += f" --lnc={lnc}"
        return compiler_args
