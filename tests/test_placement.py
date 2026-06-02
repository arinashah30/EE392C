from pathlib import Path

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.policies.placement import assign_home_levels
from dmsim.trace.schema import TensorCategory, TensorRecord, Trace, TraceMetadata


ROOT = Path(__file__).resolve().parents[1]


def _trace_with_reads(
    tensors: list[TensorRecord],
    reads: list[tuple[str, int, int]],
) -> Trace:
    """Build a trace: reads is (tensor_id, bytes, count) after one kernel_end."""
    events = [{"type": "kernel_end", "t_ns": 0.0, "kernel_id": 0, "core_id": 0}]
    t = 1.0
    for tensor_id, nbytes, count in reads:
        for _ in range(count):
            events.append(
                {
                    "type": "access",
                    "op": "read",
                    "tensor_id": tensor_id,
                    "bytes": nbytes,
                    "target_level": "sbuf",
                    "core_id": 0,
                    "t_ns": t,
                }
            )
            t += 1.0
    return Trace(
        version=1,
        metadata=TraceMetadata(workload="placement_test"),
        tensors=tensors,
        events=events,
    )


def test_spill_uses_policy_fallback_not_yaml_order() -> None:
    """LtRAM overflow spills to hbm per policy, not sequential stram."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    tensors = [
        TensorRecord(id="w1", name="w1", bytes=250_000, category=TensorCategory.WEIGHT),
        TensorRecord(id="w2", name="w2", bytes=250_000, category=TensorCategory.WEIGHT),
    ]
    trace = _trace_with_reads(
        tensors,
        [("w1", 65536, 40), ("w2", 4096, 2)],
    )
    homes = assign_home_levels(tensors, hierarchy, policy, trace=trace)
    assert homes["w1"] == "ltram"
    assert homes["w2"] == "hbm"
    assert "stram" not in homes.values()


def test_spill_ranks_by_residency_replay() -> None:
    """Few large home-level reads beat many tiny ones under residency replay."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    tensors = [
        TensorRecord(
            id="w_many_small",
            name="w_many_small",
            bytes=20_000,
            category=TensorCategory.WEIGHT,
        ),
        TensorRecord(
            id="w_few_large",
            name="w_few_large",
            bytes=370_000,
            category=TensorCategory.WEIGHT,
        ),
    ]
    trace = _trace_with_reads(
        tensors,
        [("w_many_small", 4096, 80), ("w_few_large", 65536, 8)],
    )
    homes = assign_home_levels(tensors, hierarchy, policy, trace=trace)
    assert homes["w_many_small"] == "hbm"
    assert homes["w_few_large"] == "ltram"


def test_worst_case_spills_highest_retention() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    policy = policy.model_copy(update={"spill_victim_order": "worst_case"})
    tensors = [
        TensorRecord(id="w1", name="w1", bytes=250_000, category=TensorCategory.WEIGHT),
        TensorRecord(id="w2", name="w2", bytes=250_000, category=TensorCategory.WEIGHT),
    ]
    trace = _trace_with_reads(
        tensors,
        [("w1", 65536, 40), ("w2", 4096, 2)],
    )
    homes = assign_home_levels(tensors, hierarchy, policy, trace=trace)
    assert homes["w1"] == "hbm"
    assert homes["w2"] == "ltram"


def test_disabled_level_uses_policy_fallback() -> None:
    """Policy homes to stram but baseline hierarchy disables it → fallback hbm."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    tensors = [
        TensorRecord(id="kv", name="kv", bytes=4096, category=TensorCategory.KV_CACHE),
    ]
    homes = assign_home_levels(tensors, hierarchy, policy)
    assert homes["kv"] == "hbm"


def test_tiered_stram_spills_to_hbm_per_policy() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    stram_cap = next(
        level.capacity_bytes
        for level in hierarchy.enabled_levels
        if level.id == "stram"
    )
    tensor_bytes = stram_cap // 2 + 1
    tensors = [
        TensorRecord(
            id="kv_hot",
            name="kv_hot",
            bytes=tensor_bytes,
            category=TensorCategory.KV_CACHE,
            core_id=0,
        ),
        TensorRecord(
            id="kv_cold",
            name="kv_cold",
            bytes=tensor_bytes,
            category=TensorCategory.KV_CACHE,
            core_id=0,
        ),
    ]
    trace = _trace_with_reads(
        tensors,
        [("kv_hot", 8192, 40), ("kv_cold", 8192, 2)],
    )
    homes = assign_home_levels(tensors, hierarchy, policy, trace=trace)
    assert homes["kv_hot"] == "stram"
    assert homes["kv_cold"] == "hbm"
