from pathlib import Path

import pytest

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.sim.engine import run_simulation
from dmsim.trace.schema import (
    AccessEvent,
    KernelBoundaryEvent,
    TensorCategory,
    TensorRecord,
    Trace,
    TraceMetadata,
)


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data/traces/synthetic_decode.json"


@pytest.fixture
def trace() -> Trace:
    from dmsim.trace.schema import load_trace

    return load_trace(TRACE)


def test_baseline_runs(trace: Trace) -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    result = run_simulation(trace, hierarchy, policy)
    assert result.total_time_ns > 0
    assert result.hbm_traffic_bytes > 0


def test_diff_mem_reduces_hbm_traffic(trace: Trace) -> None:
    baseline_h = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    diff_h = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem_25hbm.yaml", num_cores=1
    )
    baseline_p = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    diff_p = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")

    baseline = run_simulation(trace, baseline_h, baseline_p)
    candidate = run_simulation(trace, diff_h, diff_p)
    assert "ltram->sbuf" in candidate.transfers_by_hop
    assert "ltram->sbuf" not in baseline.transfers_by_hop
    assert candidate.hbm_traffic_bytes < baseline.hbm_traffic_bytes


def test_kernel_wipe_forces_reload(trace: Trace) -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    result = run_simulation(trace, hierarchy, policy)
    assert result.kernel_wipes >= 1


def test_kernel_wipe_scoped_to_core() -> None:
    """kernel_end with core_id only resets that core's scratch residency."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=2
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="wipe_scope"),
        tensors=[
            TensorRecord(
                id="t0",
                name="w0",
                bytes=4096,
                category=TensorCategory.WEIGHT,
                core_id=0,
            ),
            TensorRecord(
                id="t1",
                name="w1",
                bytes=4096,
                category=TensorCategory.WEIGHT,
                core_id=1,
            ),
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="t0",
                op="read",
                bytes=4096,
                target_level="sbuf",
                core_id=0,
            ).model_dump(),
            AccessEvent(
                t_ns=1,
                tensor_id="t1",
                op="read",
                bytes=4096,
                target_level="sbuf",
                core_id=1,
            ).model_dump(),
            KernelBoundaryEvent(
                type="kernel_end", t_ns=1000.0, kernel_id=1, core_id=0
            ).model_dump(),
            AccessEvent(
                t_ns=2000,
                tensor_id="t0",
                op="read",
                bytes=4096,
                target_level="sbuf",
                core_id=0,
            ).model_dump(),
            AccessEvent(
                t_ns=2001,
                tensor_id="t1",
                op="read",
                bytes=4096,
                target_level="sbuf",
                core_id=1,
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    # Two initial loads + one reload for core 0 after wipe; core 1 scratch hit.
    assert result.transfers_by_hop["hbm->sbuf"] == 3


def test_ltram_homed_access_charges_direct_hop() -> None:
    """LtRAM→SBUF is one direct hop (no automatic StRAM staging)."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="ltram_direct"),
        tensors=[
            TensorRecord(
                id="w",
                name="weight",
                bytes=8192,
                category=TensorCategory.WEIGHT,
            )
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="w",
                op="read",
                bytes=8192,
                target_level="sbuf",
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.transfers_by_hop == {"ltram->sbuf": 1}
    assert result.energy_by_level_pJ.get("stram", 0.0) == 0.0


def test_stram_home_persists_across_kernel_wipe() -> None:
    """StRAM is not in wipe_levels_on_boundary — home copy survives kernel_end."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert "stram" not in hierarchy.kernel.wipe_levels_on_boundary
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="stram_kernel"),
        tensors=[
            TensorRecord(
                id="kv",
                name="cache_k",
                bytes=4096,
                category=TensorCategory.KV_CACHE,
            )
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="kv",
                op="read",
                bytes=4096,
                target_level="sbuf",
            ).model_dump(),
            KernelBoundaryEvent(type="kernel_end", t_ns=1000.0, kernel_id=1).model_dump(),
            AccessEvent(
                t_ns=2000,
                tensor_id="kv",
                op="read",
                bytes=4096,
                target_level="sbuf",
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.transfers_by_hop == {}
    assert result.energy_by_level_pJ.get("stram", 0.0) > 0.0


def test_stram_read_to_sbuf_is_local_not_dma() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="stram_local"),
        tensors=[
            TensorRecord(
                id="kv",
                name="cache_k",
                bytes=8192,
                category=TensorCategory.KV_CACHE,
            )
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="kv",
                op="read",
                bytes=8192,
                target_level="sbuf",
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert "stram->sbuf" not in result.transfers_by_hop
    assert result.energy_by_level_pJ.get("sbuf", 0.0) == 0.0
    assert result.energy_by_level_pJ.get("stram", 0.0) > 0.0
    stram = hierarchy.level_by_id("stram")
    from dmsim.sim.transfer import access_latency_ns

    assert result.total_time_ns == access_latency_ns(stram, "read", 8192, hierarchy)


def test_hbm_homed_access_single_hop_on_baseline_stack() -> None:
    """With only PSUM/SBUF/HBM enabled, HBM-homed load is one adjacent hop."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="hbm_direct"),
        tensors=[
            TensorRecord(
                id="w",
                name="weight",
                bytes=8192,
                category=TensorCategory.WEIGHT,
            )
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="w",
                op="read",
                bytes=8192,
                target_level="sbuf",
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.transfers_by_hop == {"hbm->sbuf": 1}
