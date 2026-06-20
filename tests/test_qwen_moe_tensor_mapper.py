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
    assert classify_kv_shape((1, 2, 256, 64)) == "bhsd"
    assert classify_kv_shape((8, 128, 16, 64)) is None


def test_qwen_moe_tp_sharded_expert_and_shared_weights() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input1", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input10", "shape": "[60 2048 704]", "type": "IN"},
        {"variable_name": "input11", "shape": "[60 352 2048]", "type": "IN"},
        {"variable_name": "input20", "shape": "[2048 1408]", "type": "IN"},
        {"variable_name": "input21", "shape": "[1408 2048]", "type": "IN"},
    ]
    mapper = QwenMoENameMapper.auto_detect_config(tensors)
    assert mapper.moe_intermediate_size == 1408
    gate_up = mapper.map_tensor("input10", "[60 2048 704]", "IN")
    down = mapper.map_tensor("input11", "[60 352 2048]", "IN")
    sgate = mapper.map_tensor("input20", "[2048 1408]", "IN")
    sdown = mapper.map_tensor("input21", "[1408 2048]", "IN")
    assert gate_up.semantic_name == "layer_0.moe.expert.gate_up.weight"
    assert down.semantic_name == "layer_0.moe.expert.down_proj.weight"
    assert sgate.semantic_name == "layer_0.mlp.shared.gate_proj.weight"
    assert sdown.semantic_name == "layer_0.mlp.shared.down_proj.weight"
    assert mapper_category_to_sim(gate_up.category) == TensorCategory.WEIGHT


def test_qwen_moe_expert_3d_shapes() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input1", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input2", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input3", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input10", "shape": "[60 2048 2816]", "type": "IN"},
        {"variable_name": "input11", "shape": "[60 1408 2048]", "type": "IN"},
    ]
    mapper = QwenMoENameMapper.auto_detect_config(tensors)
    gate_up = mapper.map_tensor("input10", "[60 2048 2816]", "IN")
    down = mapper.map_tensor("input11", "[60 1408 2048]", "IN")
    assert gate_up.semantic_name == "layer_0.moe.expert.gate_up.weight"
    assert down.semantic_name == "layer_0.moe.expert.down_proj.weight"


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
    assert wq.semantic_name == "layer_0.attention.qkv_proj.weight"

    embed = mapper.map_tensor("input52", "[37984 2048]", "IN")
    assert embed.semantic_name == "embedding.weight"


def test_qwen_fused_qkv_when_one_slot_per_layer() -> None:
    tensors = [
        {"variable_name": f"input{52 + i * 17}", "shape": "[2048 512]", "type": "IN"}
        for i in range(24)
    ]
    mapper = QwenMoENameMapper(n_layers=24, hidden_size=2048, n_heads=16, head_dim=128, tp_degree=4)
    mapper._build_index_maps(tensors)
    for i, t in enumerate(tensors):
        info = mapper.map_tensor(t["variable_name"], t["shape"], "IN")
        assert info.semantic_name == f"layer_{i}.attention.qkv_proj.weight"


def test_qwen_post_attention_norm_vectors() -> None:
    tensors = [
        {"variable_name": f"input{55 + i}", "shape": "[2048]", "type": "IN"}
        for i in range(73)
    ]
    mapper = QwenMoENameMapper(n_layers=24, hidden_size=2048)
    mapper._build_index_maps(tensors)
    final = mapper.map_tensor("input127", "[2048]", "IN")
    assert final.semantic_name == "final_norm.weight"
    layer0 = mapper.map_tensor("input55", "[2048]", "IN")
    assert layer0.semantic_name == "layer_0.norm.weight"
    layer1 = mapper.map_tensor("input56", "[2048]", "IN")
    assert layer1.semantic_name == "layer_0.norm.weight"


def test_qwen_wo_and_bias_tp_shard_names() -> None:
    wo = [
        {"variable_name": f"input{54 + i}", "shape": "[512 2048]", "type": "IN"}
        for i in range(6)
    ]
    bias = [
        {"variable_name": f"input{200 + i}", "shape": "[512]", "type": "IN"}
        for i in range(6)
    ]
    tensors = wo + bias
    mapper = QwenMoENameMapper(n_layers=2, hidden_size=2048, n_heads=16, head_dim=128, tp_degree=4)
    mapper._build_index_maps(tensors)
    assert mapper.map_tensor("input54", "[512 2048]", "IN").semantic_name == "layer_0.attention.wo.weight.tp0"
    assert mapper.map_tensor("input55", "[512 2048]", "IN").semantic_name == "layer_0.attention.wo.weight.tp1"
    assert mapper.map_tensor("input57", "[512 2048]", "IN").semantic_name == "layer_1.attention.wo.weight.tp0"
    assert mapper.map_tensor("input200", "[512]", "IN").semantic_name == "layer_0.attention.bias.tp0"
    assert mapper.map_tensor("input201", "[512]", "IN").semantic_name == "layer_0.attention.bias.tp1"
    assert mapper.map_tensor("input203", "[512]", "IN").semantic_name == "layer_1.attention.bias.tp0"


def test_qwen_output_kv_uses_bhsd_shape_description() -> None:
    tensors = [
        {"variable_name": "input0", "shape": "[1 2]", "type": "IN"},
        {"variable_name": "input3", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "input4", "shape": "[1 4 256 128]", "type": "IN"},
        {"variable_name": "output0", "shape": "[151936]", "type": "OUT"},
        {"variable_name": "output1", "shape": "[1 4 256 128]", "type": "OUT"},
        {"variable_name": "output2", "shape": "[1 4 256 128]", "type": "OUT"},
    ]
    mapper = QwenMoENameMapper.auto_detect_config(tensors)
    assert len(mapper._output_kv_slots) == 2

    k_out = mapper.map_tensor("output1", "[1 4 256 128]", "OUT")
    v_out = mapper.map_tensor("output2", "[1 4 256 128]", "OUT")
    assert k_out.semantic_name == "layer_0.cache_k_out"
    assert v_out.semantic_name == "layer_0.cache_v_out"
    assert k_out.shape_description == "(batch, n_kv_heads, seq_len, head_dim)"
    assert v_out.shape_description == "(batch, n_kv_heads, seq_len, head_dim)"
