from pathlib import Path

import pytest

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.sim.engine import run_simulation
from dmsim.trace.schema import (
    AccessEvent,
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
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    baseline_p = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    diff_p = load_policy(ROOT / "configs/policies/decode_tiered.yaml")

    baseline = run_simulation(trace, baseline_h, baseline_p)
    candidate = run_simulation(trace, diff_h, diff_p)
    assert candidate.hbm_traffic_bytes <= baseline.hbm_traffic_bytes


def test_retention_corrupts_stale_stram() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_tiered.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="retention_test"),
        tensors=[
            TensorRecord(
                id="kv",
                name="kv_cache",
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
            AccessEvent(
                t_ns=5_000_000,
                tensor_id="kv",
                op="read",
                bytes=4096,
                target_level="sbuf",
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.retention_evictions >= 1
    assert result.corrupt_accesses >= 1


def test_kernel_wipe_forces_reload(trace: Trace) -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    result = run_simulation(trace, hierarchy, policy)
    assert result.kernel_wipes >= 1


def test_hbm_homed_access_skips_stram_ltram_hop_keys() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
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
    assert "hbm->sbuf" in result.transfers_by_hop
    assert "hbm->ltram" not in result.transfers_by_hop
    assert "ltram->stram" not in result.transfers_by_hop
