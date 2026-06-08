from dmsim.trace.schema import TensorCategory
from dmsim.trace.tensor_name_mapper import (
    LLaMANameMapper,
    NeffTensorCatalog,
    create_mapper_for_tensors,
    mapper_category_to_sim,
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
    assert mapper_category_to_sim(k_info.category) == TensorCategory.KV_CACHE


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
    assert entry.semantic_name == "input3_kv_cache"
    assert entry.category == TensorCategory.KV_CACHE

    weight_dma = catalog.resolve_dma("transpose.99_sg0000", 524288, src="WEIGHT", dst="SB")
    assert weight_dma is not None
    assert weight_dma.category == TensorCategory.WEIGHT
