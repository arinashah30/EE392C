"""Residency-aware placement replay matches hop charging."""

from pathlib import Path

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.policies.placement import assign_home_levels
from dmsim.sim.engine import run_simulation
from dmsim.sim.placement_replay import build_retention_by_residency_replay
from dmsim.sim.transfer import latency_ns
from dmsim.trace.schema import TensorCategory, TensorRecord, Trace, TraceMetadata


ROOT = Path(__file__).resolve().parents[1]


def test_build_retention_counts_home_level_hops_only() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem_10hbm.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    trace = Trace(
        version=1,
        metadata=TraceMetadata(workload="test"),
        tensors=[
            TensorRecord(
                id="w",
                name="w",
                bytes=4096,
                category=TensorCategory.WEIGHT,
            )
        ],
        events=[
            {
                "type": "access",
                "op": "read",
                "tensor_id": "w",
                "bytes": 8192,
                "target_level": "sbuf",
                "core_id": 0,
            },
            {
                "type": "access",
                "op": "read",
                "tensor_id": "w",
                "bytes": 8192,
                "target_level": "sbuf",
                "core_id": 0,
            },
        ],
    )
    homes = {"w": "ltram"}
    retention, home_hop_bytes = build_retention_by_residency_replay(
        trace, hierarchy, policy, homes, "ltram", "hbm"
    )
    assert home_hop_bytes["w"] == 8192
    sbuf = hierarchy.level_by_id("sbuf")
    ltram = hierarchy.level_by_id("ltram")
    hbm = hierarchy.level_by_id("hbm")
    one_hop = latency_ns(hierarchy, 8192, from_level=hbm, to_level=sbuf) - latency_ns(
        hierarchy, 8192, from_level=ltram, to_level=sbuf
    )
    assert retention["w"] == int(one_hop)


def test_best_case_beats_worst_case_on_max_core_time_for_constrained_ltram() -> None:
    """Synthetic trace: best placement must not exceed worst simulated time."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    # Two weights; LtRAM fits one. Hot tensor benefits from LtRAM on every layer.
    tensors = [
        TensorRecord(id="hot", name="hot", bytes=250_000, category=TensorCategory.WEIGHT),
        TensorRecord(id="cold", name="cold", bytes=250_000, category=TensorCategory.WEIGHT),
    ]
    events = []
    t = 0.0
    events.append(
        {
            "type": "kernel_end",
            "t_ns": t,
            "kernel_id": 0,
            "core_id": 0,
        }
    )
    t += 1.0
    for _ in range(32):
        events.append(
            {
                "type": "access",
                "op": "read",
                "tensor_id": "hot",
                "bytes": 65536,
                "target_level": "sbuf",
                "core_id": 0,
                "t_ns": t,
            }
        )
        t += 1.0
    for _ in range(4):
        events.append(
            {
                "type": "access",
                "op": "read",
                "tensor_id": "cold",
                "bytes": 4096,
                "target_level": "sbuf",
                "core_id": 0,
                "t_ns": t,
            }
        )
        t += 1.0

    trace = Trace(
        version=1,
        metadata=TraceMetadata(workload="synthetic_spill"),
        tensors=tensors,
        events=events,
    )
    best_p = policy
    worst_p = policy.model_copy(update={"spill_victim_order": "worst_case"})
    best_h = assign_home_levels(tensors, hierarchy, best_p, trace=trace)
    worst_h = assign_home_levels(tensors, hierarchy, worst_p, trace=trace)
    assert best_h["hot"] == "ltram"
    assert best_h["cold"] == "hbm"
    assert worst_h["hot"] == "hbm"
    assert worst_h["cold"] == "ltram"

    best_t = run_simulation(trace, hierarchy, best_p).total_time_ns
    worst_t = run_simulation(trace, hierarchy, worst_p).total_time_ns
    assert best_t <= worst_t
