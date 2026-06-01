from pathlib import Path

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.policies.placement import assign_home_levels
from dmsim.trace.schema import TensorCategory, TensorRecord


ROOT = Path(__file__).resolve().parents[1]


def test_spill_uses_policy_fallback_not_yaml_order() -> None:
    """LtRAM overflow spills to hbm per policy, not sequential stram."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    tensors = [
        TensorRecord(id="w1", name="w1", bytes=600_000_000, category=TensorCategory.WEIGHT),
        TensorRecord(id="w2", name="w2", bytes=600_000_000, category=TensorCategory.WEIGHT),
    ]
    # w1 is hot; w2 is cold — best_case should spill w2 to hbm first.
    homes = assign_home_levels(
        tensors,
        hierarchy,
        policy,
        access_counts={"w1": 100, "w2": 1},
    )
    assert homes["w1"] == "ltram"
    assert homes["w2"] == "hbm"
    assert "stram" not in homes.values()


def test_worst_case_spills_most_accessed() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    policy = policy.model_copy(update={"spill_victim_order": "worst_case"})
    tensors = [
        TensorRecord(id="w1", name="w1", bytes=600_000_000, category=TensorCategory.WEIGHT),
        TensorRecord(id="w2", name="w2", bytes=600_000_000, category=TensorCategory.WEIGHT),
    ]
    homes = assign_home_levels(
        tensors,
        hierarchy,
        policy,
        access_counts={"w1": 100, "w2": 1},
    )
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


def test_tiered_stram_spills_to_ltram_best_case() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    stram_cap = next(
        level.capacity_bytes
        for level in hierarchy.enabled_levels
        if level.id == "stram"
    )
    tensors = [
        TensorRecord(
            id="kv_hot",
            name="kv_hot",
            bytes=stram_cap // 2 + 1,
            category=TensorCategory.KV_CACHE,
            core_id=0,
        ),
        TensorRecord(
            id="kv_cold",
            name="kv_cold",
            bytes=stram_cap // 2 + 1,
            category=TensorCategory.KV_CACHE,
            core_id=0,
        ),
    ]
    homes = assign_home_levels(
        tensors,
        hierarchy,
        policy,
        access_counts={"kv_hot": 50, "kv_cold": 1},
    )
    assert homes["kv_hot"] == "stram"
    assert homes["kv_cold"] == "ltram"
