from dmsim.trace.schema import TensorCategory
from dmsim.trace.tensor_name_mapper import (
    LLaMANameMapper,
    NeffTensorCatalog,
    QwenMoENameMapper,
    classify_kv_shape,
    create_mapper_for_tensors,
    is_kv_cache_shape,
    mapper_category_to_sim,
    parse_shape,
)


def test_mapper_kv_cache_indices() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 128]", "type": "IN"},
        {"variable_name": "input1", "shape": "[1]", "type": "IN"},
        {"variable_name": "input2", "shape": "[1 128]", "type": "IN"},
        {"variable_name": "input3", "shape": "[1 128 2 64]", "type": "IN"},
        {"variable_name": "input4", "shape": "[1 128 2 64]", "type": "IN"},
    ]
    mapper = create_mapper_for_tensors(tensors)
    k_info = mapper.map_tensor("input3", "[1 128 2 64]", "IN")
    assert k_info.category == LLaMANameMapper.CATEGORY_KV_CACHE
    assert k_info.semantic_name == "layer_0.cache_k"
    assert mapper_category_to_sim(k_info.category) == TensorCategory.KV_CACHE


def test_nxdi_kv_tile_shapes_map_to_kv_cache() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 1]", "type": "IN"},
        {"variable_name": "input1", "shape": "[1]", "type": "IN"},
        {"variable_name": "input2", "shape": "[1 1]", "type": "IN"},
        {"variable_name": "input37", "shape": "[128 16 2 64]", "type": "IN"},
        {"variable_name": "input39", "shape": "[128 16 2 2 32]", "type": "IN"},
    ]
    mapper = create_mapper_for_tensors(tensors)
    k_tile = mapper.map_tensor("input37", "[128 16 2 64]", "IN")
    v_tile = mapper.map_tensor("input39", "[128 16 2 2 32]", "IN")
    assert k_tile.category == LLaMANameMapper.CATEGORY_KV_CACHE
    assert v_tile.category == LLaMANameMapper.CATEGORY_KV_CACHE
    assert mapper_category_to_sim(k_tile.category) == TensorCategory.KV_CACHE


def test_nxdi_workspace_shape_is_not_kv_cache() -> None:
    workspace = parse_shape("[8 128 16 2 128]")
    assert workspace is not None
    assert classify_kv_shape(workspace) is None
    assert not is_kv_cache_shape(workspace)


def test_llama_decode_neff_selects_llama_mapper() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 1]", "type": "IN"},
        {"variable_name": "input1", "shape": "[1]", "type": "IN"},
        {"variable_name": "input2", "shape": "[1 1]", "type": "IN"},
    ]
    for layer in range(16):
        tensors.append(
            {"variable_name": f"input{35 + layer * 2}", "shape": "[1 2 256 64]", "type": "IN"}
        )
        tensors.append(
            {"variable_name": f"input{36 + layer * 2}", "shape": "[1 2 256 64]", "type": "IN"}
        )
    for i in range(16):
        tensors.append(
            {"variable_name": f"input{100 + i}", "shape": "[8 128 16 2 128]", "type": "IN"}
        )
    tensors.append({"variable_name": "input180", "shape": "[32064 2048]", "type": "IN"})

    mapper = create_mapper_for_tensors(tensors)
    assert isinstance(mapper, LLaMANameMapper)
    assert mapper.n_layers == 16

    k0 = mapper.map_tensor("input35", "[1 2 256 64]", "IN")
    v0 = mapper.map_tensor("input36", "[1 2 256 64]", "IN")
    assert k0.semantic_name == "layer_0.cache_k"
    assert v0.semantic_name == "layer_0.cache_v"

    workspace = mapper.map_tensor("input100", "[8 128 16 2 128]", "IN")
    assert workspace.category == LLaMANameMapper.CATEGORY_KV_CACHE
    assert workspace.component == "kv_staging"

    lm_head = mapper.map_tensor("input180", "[32064 2048]", "IN")
    assert lm_head.semantic_name == "output.weight"
    assert mapper_category_to_sim(lm_head.category) == TensorCategory.WEIGHT


def test_catalog_resolves_input_and_dma_weight_route() -> None:
    device = {
        "neff_node": [
            {"variable_name": "input0", "shape": "[1 128]", "size": "8", "type": "IN"},
            {"variable_name": "input1", "shape": "[1]", "size": "8", "type": "IN"},
            {"variable_name": "input2", "shape": "[1 128]", "size": "8", "type": "IN"},
            {"variable_name": "input3", "shape": "[1 128 2 64]", "size": "65536", "type": "IN"},
            {"variable_name": "input4", "shape": "[1 128 2 64]", "size": "65536", "type": "IN"},
            {"variable_name": "input100", "shape": "[128 2048]", "size": "524288", "type": "IN"},
        ],
        "dma": [],
    }
    catalog = NeffTensorCatalog(device)
    entry = catalog.by_variable["input3"]
    assert entry.semantic_name == "layer_0.cache_k"
    assert entry.category == TensorCategory.KV_CACHE

    weight_dma = catalog.resolve_dma("transpose.99_sg0000", 524288, src="WEIGHT", dst="SB")
    assert weight_dma is not None
    assert weight_dma.category == TensorCategory.WEIGHT

    compound = catalog.resolve_dma("input3, input4", 4096)
    assert compound is not None
    assert compound.category == TensorCategory.KV_CACHE

    collective = catalog.resolve_dma("all_gather.2_sg0000, output0", 4096)
    assert collective is not None
    assert collective.category == TensorCategory.ACTIVATION


def test_nxdi_norm_shards_assign_layer_index() -> None:
    tensors = [
        {"variable_name": f"input{38 + i * 4}", "shape": "[128 16]", "type": "IN"}
        for i in range(33)
    ]
    mapper = LLaMANameMapper(n_layers=16)
    mapper._build_layer_groups(tensors)
    attn = mapper.map_tensor("input38", "[128 16]", "IN")
    mlp = mapper.map_tensor("input42", "[128 16]", "IN")
    final = mapper.map_tensor(tensors[-1]["variable_name"], "[128 16]", "IN")
    assert attn.semantic_name == "layer_0.attention_norm.weight"
    assert mlp.semantic_name == "layer_0.mlp_norm.weight"
    assert final.semantic_name == "final_norm.weight"


def test_coalesced_memloc_dma_maps_to_activation() -> None:
    device = {
        "neff_node": [
            {"variable_name": "input0", "shape": "[1 1]", "size": "8", "type": "IN"},
        ],
        "dma": [],
    }
    catalog = NeffTensorCatalog(device)
    resolved = catalog.resolve_dma("Coalesced_memloc_split_0", 4096)
    assert resolved is not None
    assert resolved.category == TensorCategory.ACTIVATION


def test_compiler_weight_slots_map_to_activation() -> None:
    mapper = LLaMANameMapper()
    info = mapper.map_tensor("t25415_sg0000", "[32 128 1 1]", "WEIGHT")
    assert info.category == LLaMANameMapper.CATEGORY_RUNTIME
    assert mapper_category_to_sim(info.category) == TensorCategory.ACTIVATION

    device = {
        "neff_node": [
            {"variable_name": "t25415_sg0000", "shape": "[32 128 1 1]", "size": "4096", "type": "WEIGHT"},
        ],
        "dma": [],
    }
    catalog = NeffTensorCatalog(device)
    assert catalog.by_variable["t25415_sg0000"].category == TensorCategory.ACTIVATION
