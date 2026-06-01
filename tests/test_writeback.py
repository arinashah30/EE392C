from pathlib import Path

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


def test_write_to_hbm_charges_sbuf_path() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="hbm_writeback"),
        tensors=[
            TensorRecord(
                id="out",
                name="gather.1",
                bytes=4096,
                category=TensorCategory.OTHER,
            )
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="out",
                op="write",
                bytes=4096,
                target_level="hbm",
                core_id=0,
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.hbm_write_bytes == 4096
    assert result.transfers_by_hop.get("sbuf->hbm", 0) == 1
    assert result.hbm_read_bytes == 0


def test_write_to_hbm_direct_with_ltram_in_stack() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem_25hbm.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/decode_ltram_only.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="hbm_writeback_ltram_stack"),
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
                op="write",
                bytes=4096,
                target_level="hbm",
                core_id=0,
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.hbm_write_bytes == 4096
    assert result.transfers_by_hop == {"sbuf->hbm": 1}


def test_same_level_sbuf_write_is_omitted() -> None:
    """Write with source==target (SBUF scratch) charges no time or energy."""
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    policy = load_policy(ROOT / "configs/policies/baseline_hbm.yaml")
    trace = Trace(
        metadata=TraceMetadata(workload="sbuf_write_scratch"),
        tensors=[
            TensorRecord(
                id="out",
                name="output",
                bytes=4096,
                category=TensorCategory.OTHER,
            )
        ],
        events=[
            AccessEvent(
                t_ns=0,
                tensor_id="out",
                op="read",
                bytes=4096,
                target_level="sbuf",
                core_id=0,
            ).model_dump(),
            AccessEvent(
                t_ns=100,
                tensor_id="out",
                op="write",
                bytes=4096,
                target_level="sbuf",
                core_id=0,
            ).model_dump(),
        ],
    )
    result = run_simulation(trace, hierarchy, policy)
    assert result.transfers_by_hop == {"hbm->sbuf": 1}
    assert result.total_time_ns > 0
    # Second event is same-level write — only the read's interconnect + local read cost
    reread = run_simulation(
        Trace(
            metadata=TraceMetadata(workload="read_only"),
            tensors=trace.tensors,
            events=[trace.events[0]],
        ),
        hierarchy,
        policy,
    )
    assert result.total_time_ns == reread.total_time_ns
    assert result.total_energy_pJ == reread.total_energy_pJ
    assert result.hbm_write_bytes == 0
    assert result.hbm_read_bytes == 4096
