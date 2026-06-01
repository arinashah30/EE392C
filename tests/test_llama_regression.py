"""Regression metrics for the 4-core Llama decode trace (committed simulator)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dmsim.config.loader import load_hierarchy, load_policy
from dmsim.sim.engine import run_simulation
from dmsim.trace.schema import load_trace

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "data/traces/llama32_1b_decode_4core_min1_no_unknown.json"

pytestmark = pytest.mark.skipif(
    not LLAMA.is_file(),
    reason="Llama trace not present",
)


def _run(hierarchy: str, policy: str):
    trace = load_trace(LLAMA)
    h = load_hierarchy(ROOT / hierarchy, num_cores=4)
    p = load_policy(ROOT / policy)
    return run_simulation(trace, h, p)


def test_area_budget_stram_capacity_50sbuf() -> None:
    """50% SBUF die area → StRAM: capacity = 0.5 × area(SBUF) × ρ_eDRAM / 8."""
    h = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml",
        num_cores=4,
    )
    assert h.level_by_id("stram").capacity_bytes == 66_180_616
    assert h.area_budget_notes["stram_area_um2"] == "48131357.3770"


def test_llama_baseline_metrics() -> None:
    r = _run(
        "configs/hierarchy/trainium2_baseline.yaml",
        "configs/policies/baseline_hbm.yaml",
    )
    assert r.total_time_ns == pytest.approx(4_446_607.0260902075, rel=0, abs=1e-3)
    assert r.total_energy_pJ == pytest.approx(1_670_092_155_883.6238, rel=0, abs=1e3)
    assert r.transfers_by_hop == {"hbm->sbuf": 4070, "sbuf->hbm": 134396}


def test_llama_tiered_50sbuf_25hbm_metrics() -> None:
    r = _run(
        "configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml",
        "configs/policies/decode_tiered.yaml",
    )
    assert r.total_time_ns == pytest.approx(4_413_934.2668554755, rel=0, abs=1e-3)
    assert r.total_energy_pJ == pytest.approx(140_827_660_844_846.94, rel=0, abs=1e3)
    assert r.hbm_read_bytes == 15_970_428
    assert r.transfers_by_hop == {
        "hbm->sbuf": 2774,
        "ltram->sbuf": 868,
        "sbuf->hbm": 134396,
    }
    assert "stram->sbuf" not in r.transfers_by_hop
