from pathlib import Path

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.sim.engine import run_simulation
from dmsim.sim.transfer import latency_ns
from dmsim.trace.schema import AccessEvent, TensorCategory, TensorRecord, Trace, TraceMetadata

ROOT = Path(__file__).resolve().parents[1]


def test_per_core_time_sums_then_worst_core_wins() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=4
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    nbytes = 1_000_000
    hbm = hierarchy.level_by_id("hbm")
    sbuf = hierarchy.level_by_id("sbuf")
    hop_lat = latency_ns(hierarchy, nbytes, from_level=hbm, to_level=sbuf)

    trace = Trace(
        metadata=TraceMetadata(
            workload="per_core_time",
            neuron_core_ids=[0, 1, 2, 3],
        ),
        tensors=[
            TensorRecord(id=f"w{i}", name="weight", bytes=nbytes, category=TensorCategory.WEIGHT)
            for i in range(4)
        ],
        events=[
            AccessEvent(
                t_ns=0.0,
                tensor_id=f"w{i}",
                op="read",
                bytes=nbytes,
                target_level="sbuf",
                core_id=i,
            ).model_dump()
            for i in range(4)
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert all(result.time_by_core_ns[i] == hop_lat for i in range(4))
    assert result.total_time_ns == hop_lat
    assert result.total_time_ns == max(result.time_by_core_ns.values())


def test_same_core_accumulates_multiple_transfers() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    nbytes = 500_000
    hbm = hierarchy.level_by_id("hbm")
    sbuf = hierarchy.level_by_id("sbuf")
    hop_lat = latency_ns(hierarchy, nbytes, from_level=hbm, to_level=sbuf)

    trace = Trace(
        metadata=TraceMetadata(workload="serial_core0"),
        tensors=[
            TensorRecord(id="a", name="w1", bytes=nbytes, category=TensorCategory.WEIGHT),
            TensorRecord(id="b", name="w2", bytes=nbytes, category=TensorCategory.WEIGHT),
        ],
        events=[
            AccessEvent(
                t_ns=0, tensor_id="a", op="read", bytes=nbytes, target_level="sbuf", core_id=0
            ).model_dump(),
            AccessEvent(
                t_ns=1, tensor_id="b", op="read", bytes=nbytes, target_level="sbuf", core_id=0
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.time_by_core_ns[0] == 2 * hop_lat
    assert result.total_time_ns == 2 * hop_lat
