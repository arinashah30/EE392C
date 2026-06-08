from dmsim.trace.schema import TensorCategory
from dmsim.trace.tensor_name_mapper import (
    QwenMoENameMapper,
    classify_kv_shape,
    create_mapper_for_tensors,
    is_kv_cache_shape,
    mapper_category_to_sim,
)


def test_classify_bhsd_kv_shape() -> None:
    assert classify_kv_shape((1, 4, 256, 128)) == "bhsd"
    assert is_kv_cache_shape((1, 4, 256, 128))
    assert classify_kv_shape((1, 128, 2, 64)) == "bshd"


def test_qwen_moe_mapper_kv_and_expert_weights() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 2]", "type": "IN"},
        {"variable_name": "input1", "shape": "[1 256 2]", "type": "IN"},
        {"variable_name": "input2", "shape": "[1 1 2]", "type": "IN"},
        {"variable_name": "input3", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input4", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input5", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input6", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input10", "shape": "[60 2048]", "type": "IN"},
        {"variable_name": "input11", "shape": "[60 2048]", "type": "IN"},
        {"variable_name": "input20", "shape": "[2048 512]", "type": "IN"},
        {"variable_name": "input22", "shape": "[2048 512]", "type": "IN"},
        {"variable_name": "input21", "shape": "[512 2048]", "type": "IN"},
        {"variable_name": "input51", "shape": "[151936 512]", "type": "IN"},
        {"variable_name": "input52", "shape": "[37984 2048]", "type": "IN"},
    ]
    mapper = create_mapper_for_tensors(tensors)
    assert isinstance(mapper, QwenMoENameMapper)
    assert mapper.n_layers == 2

    k0 = mapper.map_tensor("input3", "[1 4 256 128]", "IN")
    v0 = mapper.map_tensor("input4", "[1 4 256 128]", "IN")
    assert k0.semantic_name == "layer_0.cache_k"
    assert v0.semantic_name == "layer_0.cache_v"
    assert mapper_category_to_sim(k0.category) == TensorCategory.KV_CACHE

    router = mapper.map_tensor("input10", "[60 2048]", "IN")
    assert router.semantic_name == "layer_0.moe.router.gate.weight"
    assert mapper_category_to_sim(router.category) == TensorCategory.WEIGHT

    wq = mapper.map_tensor("input20", "[2048 512]", "IN")
    assert wq.semantic_name == "layer_0.attention.wq.weight"

    embed = mapper.map_tensor("input52", "[37984 2048]", "IN")
    assert embed.semantic_name == "embedding.weight"
