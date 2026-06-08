"""Regression tests for simulator costing invariants (golden metrics on small traces)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.sim.engine import SimulationResult, run_simulation
from dmsim.sim.transfer import hops_between, latency_ns
from dmsim.trace.schema import (
    AccessEvent,
    KernelBoundaryEvent,
    TensorCategory,
    TensorRecord,
    Trace,
    TraceMetadata,
    load_trace,
)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data/traces/synthetic_decode.json"


def _run(hierarchy_path: str, policy_path: str, *, cores: int = 1) -> SimulationResult:
    trace = load_trace(SYNTHETIC)
    hierarchy = load_hierarchy(ROOT / hierarchy_path, num_cores=cores)
    policy = load_policy(ROOT / policy_path)
    return run_simulation(trace, hierarchy, policy)


def test_golden_baseline_synthetic_decode() -> None:
    result = _run(
        "configs/hierarchy/trainium2_baseline.yaml",
        "configs/policies/baseline_hbm.yaml",
    )
    assert result.total_time_ns == pytest.approx(19126.04347826087, rel=0, abs=1e-6)
    assert result.total_energy_pJ == pytest.approx(384407961.6, rel=0, abs=1)
    assert result.refresh_energy_pJ == 0.0
    assert result.hbm_read_bytes == 6815744
    assert result.hbm_write_bytes == 0
    assert result.transfers_by_hop == {"hbm->sbuf": 5}
    assert result.kernel_wipes == 2


def test_golden_tiered_synthetic_decode() -> None:
    result = _run(
        "configs/hierarchy/trainium2_diff_mem.yaml",
        "configs/policies/decode_tiered.yaml",
    )
    assert result.total_time_ns == pytest.approx(11686.959217391304, rel=0, abs=1e-6)
    assert result.total_energy_pJ == pytest.approx(39426457.6, rel=0, abs=1)
    assert result.transfers_by_hop == {"ltram->sbuf": 2}
    assert result.energy_by_level_pJ.get("stram", 0.0) == pytest.approx(20971520.0, rel=0, abs=1)
    assert "hbm->sbuf" not in result.transfers_by_hop


def test_scratch_hit_after_load_skips_interconnect() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="scratch"),
        tensors=[
            TensorRecord(id="w", name="w", bytes=4096, category=TensorCategory.WEIGHT),
        ],
        events=[
            AccessEvent(
                t_ns=0, tensor_id="w", op="read", bytes=4096, target_level="sbuf"
            ).model_dump(),
            AccessEvent(
                t_ns=100, tensor_id="w", op="read", bytes=4096, target_level="sbuf"
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.transfers_by_hop == {"hbm->sbuf": 1}
    sbuf = hierarchy.level_by_id("sbuf")
    expected = latency_ns(
        hierarchy, 4096, from_level=hierarchy.level_by_id("hbm"), to_level=sbuf
    ) + latency_ns(hierarchy, 4096, from_level=sbuf)
    assert result.total_time_ns == pytest.approx(expected, rel=0, abs=1e-6)


def test_refresh_energy_between_events_on_stram() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="refresh"),
        tensors=[
            TensorRecord(
                id="kv", name="kv", bytes=1024, category=TensorCategory.KV_CACHE
            ),
        ],
        events=[
            AccessEvent(
                t_ns=0, tensor_id="kv", op="read", bytes=1024, target_level="sbuf"
            ).model_dump(),
            AccessEvent(
                t_ns=200_000, tensor_id="kv", op="read", bytes=1024, target_level="sbuf"
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.refresh_energy_pJ > 0.0
    assert result.refresh_cycles_by_level.get("stram", 0) > 0


def test_hops_between_is_single_direct_edge() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hops_between(hierarchy, "hbm", "sbuf") == [("hbm", "sbuf")]
    assert hops_between(hierarchy, "ltram", "sbuf") == [("ltram", "sbuf")]
    assert hops_between(hierarchy, "sbuf", "sbuf") == []


def test_dma_hop_slower_than_datapath_for_small_read() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    sbuf = hierarchy.level_by_id("sbuf")
    hbm = hierarchy.level_by_id("hbm")
    nbytes = 8192
    dma = latency_ns(hierarchy, nbytes, from_level=hbm, to_level=sbuf)
    datapath = latency_ns(hierarchy, nbytes, from_level=sbuf)
    assert dma > datapath
