"""
Tensor name mapper for AWS Neuron profiling / Neuron Explorer JSON.

Maps generic NEFF names (input0, input1, …) to semantic names and simulator
categories (weight, kv_cache, activation, …) using LLaMA-style layout heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dmsim.trace.schema import TensorCategory


@dataclass
class SemanticTensorInfo:
    """Semantic information about a tensor."""
    semantic_name: str          # e.g., "layer_5.cache_k"
    category: str               # e.g., "kv_cache", "attention_weight", "mlp_weight"
    layer_index: int            # -1 for non-layer tensors
    component: str              # e.g., "key_cache", "wq", "gate_proj"
    description: str            # Human-readable description
    shape_description: str      # e.g., "(batch, seq_len, n_kv_heads, head_dim)"


class LLaMANameMapper:
    """
    Maps tensor names for LLaMA-style transformer models.
    
    Expected tensor layout (based on neuronx-distributed tracing):
    - input0: tokens [batch, seq_len]
    - input1: position [batch] or [1]
    - input2: attention_mask [batch, seq_len]
    - input3-N: KV cache (2 per layer: cache_k, cache_v)
    - inputN+1: embedding.weight
    - inputN+2 onwards: layer weights (repeating pattern)
    - Last few: output projection, final norm
    """
    
    # Categories for grouping tensors
    CATEGORY_RUNTIME = "runtime_input"
    CATEGORY_KV_CACHE = "kv_cache"
    CATEGORY_EMBEDDING = "embedding"
    CATEGORY_ATTENTION = "attention_weight"
    CATEGORY_MLP = "mlp_weight"
    CATEGORY_NORM = "norm_weight"
    CATEGORY_OUTPUT = "output_weight"
    CATEGORY_WEIGHT = "weight"
    CATEGORY_UNKNOWN = "unknown"
    
    def __init__(self, 
                 n_layers: int = 16,
                 batch_size: int = 1,
                 seq_len: int = 128,
                 hidden_size: int = 2048,
                 n_heads: int = 32,
                 n_kv_heads: int = 8,
                 head_dim: int = 64,
                 vocab_size: int = 128256,
                 tp_degree: int = 4):
        """
        Initialize the mapper with model configuration.
        
        These can be auto-detected from tensor shapes if not provided.
        """
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.tp_degree = tp_degree
        
        # Computed values
        self.n_kv_heads_per_tp = max(n_kv_heads // tp_degree, 1)
        self.kv_cache_shape = f"[{batch_size}, {seq_len}, {self.n_kv_heads_per_tp}, {head_dim}]"
        
        # Cache for mappings
        self._mapping_cache: Dict[str, SemanticTensorInfo] = {}
    
    @classmethod
    def auto_detect_config(cls, tensors: List[Dict]) -> 'LLaMANameMapper':
        """
        Auto-detect model configuration from tensor shapes.
        
        Args:
            tensors: List of tensor info dicts with 'variable_name', 'shape', 'type'
        
        Returns:
            Configured LLaMANameMapper instance
        """
        # Find KV cache tensors to detect config
        kv_cache_tensors = []
        embedding_tensor = None
        
        for t in tensors:
            shape_str = t.get('shape', '')
            var_name = t.get('variable_name', '')
            
            # Parse shape string "[1 128 2 64]" -> (1, 128, 2, 64)
            shape = cls._parse_shape(shape_str)
            
            if shape and cls._is_kv_cache_shape(shape) and var_name.startswith("input"):
                kv_cache_tensors.append((var_name, shape))
            elif shape and len(shape) == 2:
                # Could be embedding if large vocab
                if shape[0] > 10000:  # Large vocab dimension
                    embedding_tensor = (var_name, shape)
        
        # Infer config from KV cache
        if kv_cache_tensors:
            _, shape = kv_cache_tensors[0]
            batch_size = shape[0]
            seq_len = shape[1]
            n_kv_heads_per_tp = shape[2]
            head_dim = shape[3]
            n_layers = max(1, len(kv_cache_tensors) // 2)  # K and V per layer
        else:
            # Defaults
            batch_size, seq_len, n_kv_heads_per_tp, head_dim = 1, 128, 2, 64
            n_layers = 16
        
        # Infer from embedding
        if embedding_tensor:
            _, shape = embedding_tensor
            vocab_size = shape[0]
            hidden_per_tp = shape[1]
            # Estimate TP degree
            if hidden_per_tp in [512, 1024, 2048]:
                tp_degree = 2048 // hidden_per_tp if hidden_per_tp <= 2048 else 1
                hidden_size = hidden_per_tp * tp_degree
            else:
                tp_degree = 4
                hidden_size = 2048
        else:
            vocab_size = 128256
            hidden_size = 2048
            tp_degree = 4
        
        return cls(
            n_layers=n_layers,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_kv_heads=n_kv_heads_per_tp * tp_degree,
            head_dim=head_dim,
            vocab_size=vocab_size,
            tp_degree=tp_degree
        )
    
    @staticmethod
    def _parse_shape(shape_str: str) -> Optional[Tuple[int, ...]]:
        """Parse shape string like '[1 128 2 64]' to tuple."""
        if not shape_str:
            return None
        # Remove brackets and split
        shape_str = shape_str.strip('[]')
        parts = shape_str.split()
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return None

    @staticmethod
    def _is_kv_cache_shape(shape: Tuple[int, ...]) -> bool:
        """True for attention KV tensors, not compiler temporaries.

        Supports common layouts seen in Neuron profiles:
        - 4D: (batch, seq, n_kv_heads, head_dim)
        - 5D: (batch, seq, n_kv_heads, 2, head_dim) where the 2 packs K/V
        """
        if len(shape) == 4:
            if shape[2] == 1 and shape[3] == 1:
                return False
            _batch, _seq, n_heads, head_dim = shape
            return n_heads <= 16 and head_dim in (64, 128)
        if len(shape) == 5:
            _batch, _seq, n_heads, kv_pack, head_dim = shape
            if kv_pack != 2:
                return False
            return n_heads <= 16 and head_dim in (64, 128)
        return False
    
    def map_tensor(self, var_name: str, shape_str: str, tensor_type: str) -> SemanticTensorInfo:
        """
        Map a tensor to its semantic name and category.
        
        Args:
            var_name: Variable name like "input0", "output5"
            shape_str: Shape string like "[1 128 2 64]"
            tensor_type: Type like "IN", "OUT", "WEIGHT"
        
        Returns:
            SemanticTensorInfo with semantic name and metadata
        """
        # Check cache
        cache_key = f"{var_name}:{shape_str}:{tensor_type}"
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]
        
        # Parse the variable name to get index
        if var_name.startswith('input'):
            try:
                idx = int(var_name.replace('input', ''))
            except ValueError:
                idx = -1
            result = self._map_input_tensor(idx, var_name, shape_str)
        elif var_name.startswith('output'):
            try:
                idx = int(var_name.replace('output', ''))
            except ValueError:
                idx = -1
            result = self._map_output_tensor(idx, var_name, shape_str)
        elif tensor_type == 'WEIGHT':
            result = self._map_weight_tensor(var_name, shape_str)
        else:
            result = SemanticTensorInfo(
                semantic_name=var_name,
                category=self.CATEGORY_UNKNOWN,
                layer_index=-1,
                component="unknown",
                description=f"Unknown tensor: {var_name}",
                shape_description=shape_str
            )
        
        self._mapping_cache[cache_key] = result
        return result
    
    def _map_input_tensor(self, idx: int, var_name: str, shape_str: str) -> SemanticTensorInfo:
        """Map an input tensor based on its index and shape."""
        shape = self._parse_shape(shape_str)
        
        # Runtime inputs (first 3)
        if idx == 0:
            return SemanticTensorInfo(
                semantic_name="tokens",
                category=self.CATEGORY_RUNTIME,
                layer_index=-1,
                component="token_ids",
                description="Input token IDs",
                shape_description="(batch, seq_len)"
            )
        elif idx == 1:
            return SemanticTensorInfo(
                semantic_name="position",
                category=self.CATEGORY_RUNTIME,
                layer_index=-1,
                component="position_index",
                description="Sequence position index for decode",
                shape_description="(batch,) or (1,)"
            )
        elif idx == 2:
            return SemanticTensorInfo(
                semantic_name="attention_mask",
                category=self.CATEGORY_RUNTIME,
                layer_index=-1,
                component="attention_mask",
                description="Attention mask for padding/causal",
                shape_description="(batch, seq_len)"
            )

        # Heuristic KV detection: some exports use higher input indices and/or 5D packed shapes.
        if shape and self._is_kv_cache_shape(shape):
            # Prefer stable semantic names even when the input index doesn't match the
            # "input3..input(3+2*n_layers)" convention.
            shape_desc = (
                "(batch, seq_len, n_kv_heads, head_dim)"
                if len(shape) == 4
                else "(batch, seq_len, n_kv_heads, {k,v}, head_dim)"
            )
            return SemanticTensorInfo(
                semantic_name=f"{var_name}_kv_cache",
                category=self.CATEGORY_KV_CACHE,
                layer_index=-1,
                component="kv_cache",
                description="KV cache (shape-detected)",
                shape_description=shape_desc,
            )
        
        # KV Cache (input3 to input3 + 2*n_layers - 1)
        kv_start = 3
        kv_end = kv_start + 2 * self.n_layers
        
        if kv_start <= idx < kv_end:
            if shape and self._is_kv_cache_shape(shape):
                relative_idx = idx - kv_start
                layer_idx = relative_idx // 2
                is_key = (relative_idx % 2) == 0

                cache_type = "cache_k" if is_key else "cache_v"
                cache_name = "Key Cache" if is_key else "Value Cache"

                return SemanticTensorInfo(
                    semantic_name=f"layer_{layer_idx}.{cache_type}",
                    category=self.CATEGORY_KV_CACHE,
                    layer_index=layer_idx,
                    component=cache_type,
                    description=f"Layer {layer_idx} Attention {cache_name}",
                    shape_description="(batch, seq_len, n_kv_heads, head_dim)",
                )
        
        # Embedding weight (first tensor after KV cache)
        if idx == kv_end:
            return SemanticTensorInfo(
                semantic_name="embedding.weight",
                category=self.CATEGORY_EMBEDDING,
                layer_index=-1,
                component="embedding",
                description="Token embedding weight matrix",
                shape_description="(vocab_size, hidden_size/tp)"
            )
        
        # Layer weights (repeating pattern of 9 per layer)
        weights_start = kv_end + 1
        weights_per_layer = 9
        
        if weights_start <= idx < weights_start + self.n_layers * weights_per_layer:
            relative_idx = idx - weights_start
            layer_idx = relative_idx // weights_per_layer
            weight_offset = relative_idx % weights_per_layer
            
            return self._map_layer_weight(layer_idx, weight_offset, shape_str)
        
        # Output projection and final norm (last tensors)
        total_layer_weights = weights_start + self.n_layers * weights_per_layer
        
        if idx == total_layer_weights:
            return SemanticTensorInfo(
                semantic_name="output.weight",
                category=self.CATEGORY_OUTPUT,
                layer_index=-1,
                component="output_projection",
                description="Output projection (lm_head) weight",
                shape_description="(vocab_size/tp, hidden_size)"
            )
        elif idx == total_layer_weights + 1:
            return SemanticTensorInfo(
                semantic_name="final_norm.weight",
                category=self.CATEGORY_NORM,
                layer_index=-1,
                component="final_norm",
                description="Final layer normalization weight",
                shape_description="(hidden_size,)"
            )
        
        # Fallback based on shape. If the mapper still cannot recognize this input
        # but it has a high-rank activation-like shape, classify it as activation.
        inferred = self._infer_from_shape(var_name, shape_str, shape)
        if inferred.category == self.CATEGORY_UNKNOWN and shape and len(shape) >= 5:
            return SemanticTensorInfo(
                semantic_name=f"{var_name}_activation",
                category=self.CATEGORY_RUNTIME,
                layer_index=-1,
                component="activation",
                description="Intermediate activation (shape-detected)",
                shape_description=shape_str or "(activation)",
            )
        return inferred
    
    def _map_layer_weight(self, layer_idx: int, offset: int, shape_str: str) -> SemanticTensorInfo:
        """Map a per-layer weight based on offset within the layer."""
        
        # Layer weight pattern (based on observed data):
        # 0: wq [2048, 512] - attention query
        # 1: wk [128, 2048] - attention key (split/transposed)
        # 2: attention_norm [2048] - attention layer norm
        # 3: wv [128, 2048] - attention value (split/transposed)
        # 4: wo [512, 2048] - attention output
        # 5: gate_proj [2048, 2048] - MLP gate projection
        # 6: up_proj [2048, 2048] - MLP up projection  
        # 7: mlp_norm [2048] - MLP layer norm
        # 8: down_proj [2048, 2048] - MLP down projection
        
        weight_map = {
            0: ("attention.wq.weight", self.CATEGORY_ATTENTION, "wq", "Query projection weight", "(hidden, n_heads*head_dim/tp)"),
            1: ("attention.wk.weight", self.CATEGORY_ATTENTION, "wk", "Key projection weight", "(hidden/tp, hidden) or split"),
            2: ("attention_norm.weight", self.CATEGORY_NORM, "attention_norm", "Pre-attention RMSNorm weight", "(hidden_size,)"),
            3: ("attention.wv.weight", self.CATEGORY_ATTENTION, "wv", "Value projection weight", "(hidden/tp, hidden) or split"),
            4: ("attention.wo.weight", self.CATEGORY_ATTENTION, "wo", "Output projection weight", "(n_heads*head_dim/tp, hidden)"),
            5: ("mlp.gate_proj.weight", self.CATEGORY_MLP, "gate_proj", "MLP gate projection (SwiGLU)", "(hidden, intermediate/tp)"),
            6: ("mlp.up_proj.weight", self.CATEGORY_MLP, "up_proj", "MLP up projection", "(hidden, intermediate/tp)"),
            7: ("mlp_norm.weight", self.CATEGORY_NORM, "mlp_norm", "Pre-MLP RMSNorm weight", "(hidden_size,)"),
            8: ("mlp.down_proj.weight", self.CATEGORY_MLP, "down_proj", "MLP down projection", "(intermediate/tp, hidden)"),
        }
        
        if offset in weight_map:
            name_suffix, category, component, desc, shape_desc = weight_map[offset]
            return SemanticTensorInfo(
                semantic_name=f"layer_{layer_idx}.{name_suffix}",
                category=category,
                layer_index=layer_idx,
                component=component,
                description=f"Layer {layer_idx} {desc}",
                shape_description=shape_desc
            )
        
        # Fallback
        return SemanticTensorInfo(
            semantic_name=f"layer_{layer_idx}.weight_{offset}",
            category=self.CATEGORY_UNKNOWN,
            layer_index=layer_idx,
            component=f"weight_{offset}",
            description=f"Layer {layer_idx} unknown weight at offset {offset}",
            shape_description=shape_str
        )
    
    def _map_output_tensor(self, idx: int, var_name: str, shape_str: str) -> SemanticTensorInfo:
        """Map an output tensor."""
        shape = self._parse_shape(shape_str)
        
        # output0 is typically the logits
        if idx == 0:
            return SemanticTensorInfo(
                semantic_name="logits",
                category=self.CATEGORY_OUTPUT,
                layer_index=-1,
                component="logits",
                description="Output logits for next token prediction",
                shape_description="(batch, vocab_size) or (vocab_size,)"
            )
        
        # Other outputs are typically updated KV cache
        # Pattern: output1-N are KV cache updates (similar to inputs)
        if shape and self._is_kv_cache_shape(shape):
            relative_idx = idx - 1
            layer_idx = relative_idx // 2
            is_key = (relative_idx % 2) == 0
            
            cache_type = "cache_k" if is_key else "cache_v"
            cache_name = "Key Cache" if is_key else "Value Cache"
            
            return SemanticTensorInfo(
                semantic_name=f"layer_{layer_idx}.{cache_type}_out",
                category=self.CATEGORY_KV_CACHE,
                layer_index=layer_idx,
                component=f"{cache_type}_output",
                description=f"Layer {layer_idx} Updated {cache_name}",
                shape_description="(batch, seq_len, n_kv_heads, head_dim)"
            )
        
        return SemanticTensorInfo(
            semantic_name=var_name,
            category=self.CATEGORY_OUTPUT,
            layer_index=-1,
            component="output",
            description=f"Model output {idx}",
            shape_description=shape_str
        )
    
    def _map_weight_tensor(self, var_name: str, shape_str: str) -> SemanticTensorInfo:
        """Map a WEIGHT type tensor (compiler-generated masks, etc.)."""
        # These are typically compiler-generated tensors like bp_mask_*, identity_*
        
        if 'bp_mask' in var_name:
            return SemanticTensorInfo(
                semantic_name=var_name,
                category=self.CATEGORY_UNKNOWN,
                layer_index=-1,
                component="compiler_mask",
                description="Compiler-generated broadcast mask",
                shape_description=shape_str
            )
        elif 'identity' in var_name:
            return SemanticTensorInfo(
                semantic_name=var_name,
                category=self.CATEGORY_UNKNOWN,
                layer_index=-1,
                component="compiler_identity",
                description="Compiler-generated identity tensor",
                shape_description=shape_str
            )
        
        return SemanticTensorInfo(
            semantic_name=var_name,
            category=self.CATEGORY_WEIGHT,
            layer_index=-1,
            component="weight",
            description=f"Weight tensor: {var_name}",
            shape_description=shape_str
        )
    
    def _infer_from_shape(self, var_name: str, shape_str: str, shape: Optional[Tuple[int, ...]]) -> SemanticTensorInfo:
        """Infer tensor type from shape when index-based mapping fails."""
        if shape is None:
            return SemanticTensorInfo(
                semantic_name=var_name,
                category=self.CATEGORY_UNKNOWN,
                layer_index=-1,
                component="unknown",
                description=f"Unknown tensor: {var_name}",
                shape_description=shape_str
            )
        
        if self._is_kv_cache_shape(shape):
            return SemanticTensorInfo(
                semantic_name=f"{var_name}_kv_cache",
                category=self.CATEGORY_KV_CACHE,
                layer_index=-1,
                component="kv_cache",
                description="Likely KV cache tensor",
                shape_description="(batch, seq_len, n_kv_heads, head_dim)"
            )
        
        # Embedding: 2D with large vocab
        if len(shape) == 2 and shape[0] > 10000:
            return SemanticTensorInfo(
                semantic_name="embedding_like",
                category=self.CATEGORY_EMBEDDING,
                layer_index=-1,
                component="embedding",
                description="Likely embedding or output projection",
                shape_description="(vocab_size, hidden_size)"
            )
        
        # Norm weight: 1D
        if len(shape) == 1:
            return SemanticTensorInfo(
                semantic_name=f"{var_name}_norm",
                category=self.CATEGORY_NORM,
                layer_index=-1,
                component="norm",
                description="Likely normalization weight",
                shape_description="(hidden_size,)"
            )
        
        # Default
        return SemanticTensorInfo(
            semantic_name=var_name,
            category=self.CATEGORY_UNKNOWN,
            layer_index=-1,
            component="unknown",
            description=f"Tensor with shape {shape_str}",
            shape_description=shape_str
        )
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description for a category."""
        descriptions = {
            self.CATEGORY_RUNTIME: "Runtime Input (tokens, position, mask)",
            self.CATEGORY_KV_CACHE: "KV Cache (attention key/value storage)",
            self.CATEGORY_EMBEDDING: "Embedding (token embedding matrix)",
            self.CATEGORY_ATTENTION: "Attention Weight (Q/K/V/O projections)",
            self.CATEGORY_MLP: "MLP Weight (feed-forward network)",
            self.CATEGORY_NORM: "Normalization Weight (RMSNorm/LayerNorm)",
            self.CATEGORY_OUTPUT: "Output (logits, final projection)",
            self.CATEGORY_UNKNOWN: "Unknown/Compiler-generated",
        }
        return descriptions.get(category, category)


def create_mapper_for_tensors(tensors: List[Dict]) -> LLaMANameMapper:
    """
    Create an appropriately configured mapper for a list of tensors.
    
    Args:
        tensors: List of tensor dicts with 'variable_name', 'shape', 'type'
    
    Returns:
        Configured LLaMANameMapper
    """
    return LLaMANameMapper.auto_detect_config(tensors)


# Convenience function for quick mapping
def mapper_category_to_sim(category: str, *, neff_type: str = "") -> TensorCategory:
    """Map mapper category strings to dmsim TensorCategory."""
    if category == LLaMANameMapper.CATEGORY_KV_CACHE:
        return TensorCategory.KV_CACHE
    if category in (
        LLaMANameMapper.CATEGORY_ATTENTION,
        LLaMANameMapper.CATEGORY_MLP,
        LLaMANameMapper.CATEGORY_EMBEDDING,
        LLaMANameMapper.CATEGORY_NORM,
        LLaMANameMapper.CATEGORY_WEIGHT,
    ):
        return TensorCategory.WEIGHT
    if category in (LLaMANameMapper.CATEGORY_RUNTIME, LLaMANameMapper.CATEGORY_OUTPUT):
        return TensorCategory.ACTIVATION
    if neff_type == "WEIGHT" and category != LLaMANameMapper.CATEGORY_UNKNOWN:
        return TensorCategory.WEIGHT
    return TensorCategory.OTHER


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    return slug or "tensor"


@dataclass(frozen=True)
class CatalogEntry:
    variable_name: str
    semantic_name: str
    category: TensorCategory
    bytes: int
    neff_type: str
    shape: str

    @property
    def tensor_id(self) -> str:
        return _slugify(self.semantic_name or self.variable_name)


class NeffTensorCatalog:
    """NEFF tensor table (neff_node) plus DMA variable resolution."""

    def __init__(self, device: dict):
        nodes = [
            node
            for node in device.get("neff_node") or []
            if str(node.get("variable_name") or "").strip()
        ]
        mapper_tensors = [
            {
                "variable_name": node.get("variable_name", ""),
                "shape": node.get("shape", ""),
                "type": node.get("type", "IN"),
            }
            for node in nodes
        ]
        self._mapper = create_mapper_for_tensors(mapper_tensors) if mapper_tensors else LLaMANameMapper()
        self.by_variable: dict[str, CatalogEntry] = {}
        self.by_size: dict[int, list[CatalogEntry]] = {}

        for node in nodes:
            entry = self._entry_from_node(node)
            self.by_variable[entry.variable_name] = entry
            self.by_size.setdefault(entry.bytes, []).append(entry)

    def entries(self) -> list[CatalogEntry]:
        return list(self.by_variable.values())

    def _entry_from_node(self, node: dict) -> CatalogEntry:
        variable = str(node.get("variable_name", ""))
        shape = str(node.get("shape", ""))
        neff_type = str(node.get("type", "IN") or "IN")
        try:
            size_bytes = int(node.get("size") or 0)
        except (TypeError, ValueError):
            size_bytes = 0

        info = self._mapper.map_tensor(variable, shape, neff_type)
        category = mapper_category_to_sim(info.category, neff_type=neff_type)
        return CatalogEntry(
            variable_name=variable,
            semantic_name=info.semantic_name,
            category=category,
            bytes=size_bytes,
            neff_type=neff_type,
            shape=shape,
        )

    def resolve_dma(
        self,
        variable: str,
        transfer_bytes: int,
        *,
        src: str = "",
        dst: str = "",
        read_shape: object = None,
    ) -> CatalogEntry | None:
        """Best-effort link of a DMA variable to a catalog entry."""
        if variable in self.by_variable:
            return self.by_variable[variable]

        if re.fullmatch(r"input\d+", variable) or re.fullmatch(r"output\d+", variable):
            return self.by_variable.get(variable)

        route_weight = "WEIGHT" in src.upper()
        if route_weight:
            return CatalogEntry(
                variable_name=variable,
                semantic_name=variable,
                category=TensorCategory.WEIGHT,
                bytes=transfer_bytes,
                neff_type="WEIGHT",
                shape="",
            )

        shape_category = _category_from_dma_shape(read_shape)
        if shape_category is not None:
            return CatalogEntry(
                variable_name=variable,
                semantic_name=variable,
                category=shape_category,
                bytes=transfer_bytes,
                neff_type="IN",
                shape=str(read_shape),
            )

        candidates = self.by_size.get(transfer_bytes, [])
        if len(candidates) == 1:
            base = candidates[0]
            return CatalogEntry(
                variable_name=variable,
                semantic_name=base.semantic_name,
                category=base.category,
                bytes=max(transfer_bytes, base.bytes),
                neff_type=base.neff_type,
                shape=base.shape,
            )

        if len(candidates) > 1:
            picked = _pick_ambiguous(candidates, src=src, dst=dst)
            if picked:
                return CatalogEntry(
                    variable_name=variable,
                    semantic_name=picked.semantic_name,
                    category=picked.category,
                    bytes=max(transfer_bytes, picked.bytes),
                    neff_type=picked.neff_type,
                    shape=picked.shape,
                )

        info = self._mapper.map_tensor(variable, "", "WEIGHT" if route_weight else "IN")
        return CatalogEntry(
            variable_name=variable,
            semantic_name=info.semantic_name,
            category=mapper_category_to_sim(info.category),
            bytes=transfer_bytes,
            neff_type="IN",
            shape="",
        )


def _category_from_dma_shape(read_shape: object) -> TensorCategory | None:
    if not read_shape:
        return None
    dims: list[int] = []
    if isinstance(read_shape, list):
        for part in read_shape:
            if isinstance(part, list):
                dims.extend(int(x) for x in part)
            else:
                try:
                    dims.append(int(part))
                except (TypeError, ValueError):
                    pass
    if LLaMANameMapper._is_kv_cache_shape(tuple(dims)):
        return TensorCategory.KV_CACHE
    if len(dims) == 2 and dims[0] > 1000:
        return TensorCategory.WEIGHT
    return None


def _pick_ambiguous(
    candidates: list[CatalogEntry],
    *,
    src: str,
    dst: str,
) -> CatalogEntry | None:
    src_u, dst_u = src.upper(), dst.upper()
    if "WEIGHT" in src_u:
        weights = [c for c in candidates if c.category == TensorCategory.WEIGHT]
        if len(weights) == 1:
            return weights[0]
    kv = [c for c in candidates if c.category == TensorCategory.KV_CACHE]
    if len(kv) == 1:
        return kv[0]
    if "VIRTUAL" in dst_u or "REMOTE" in dst_u:
        activations = [c for c in candidates if c.category == TensorCategory.ACTIVATION]
        if len(activations) == 1:
            return activations[0]
    return None


def build_catalog(device: dict) -> NeffTensorCatalog:
    return NeffTensorCatalog(device)


def map_tensor_name(var_name: str, shape: str, tensor_type: str = "IN",
                    mapper: Optional[LLaMANameMapper] = None) -> SemanticTensorInfo:
    """
    Quick function to map a single tensor name.
    
    Args:
        var_name: Variable name like "input0"
        shape: Shape string like "[1 128 2 64]"
        tensor_type: Type like "IN", "OUT", "WEIGHT"
        mapper: Optional pre-configured mapper
    
    Returns:
        SemanticTensorInfo
    """
    if mapper is None:
        mapper = LLaMANameMapper()
    return mapper.map_tensor(var_name, shape, tensor_type)
