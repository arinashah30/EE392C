from pathlib import Path

from dmsim.config.loader import load_hierarchy

ROOT = Path(__file__).resolve().parents[1]


def test_area_budget_reduces_sbuf_and_hbm() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml",
        num_cores=4,
    )
    sbuf = hierarchy.level_by_id("sbuf")
    stram = hierarchy.level_by_id("stram")
    hbm = hierarchy.level_by_id("hbm")
    ltram = hierarchy.level_by_id("ltram")

    assert stram.enabled
    assert stram.scope == "per_core"
    assert ltram.enabled
    assert sbuf.capacity_bytes < 29_360_128
    assert hierarchy.area_budget_notes.get("stram_scope") == "per_core"
    assert hbm.capacity_bytes < int(96 * (1024**3))
    assert "sbuf_removed_per_core_bytes" in hierarchy.area_budget_notes
    assert "hbm_removed_bytes" in hierarchy.area_budget_notes


def test_fraction_area_budget() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml",
        num_cores=4,
    )
    sbuf = hierarchy.level_by_id("sbuf")
    stram = hierarchy.level_by_id("stram")
    hbm = hierarchy.level_by_id("hbm")
    ltram = hierarchy.level_by_id("ltram")
    nominal_sbuf = 29_360_128
    nominal_hbm = int(96 * (1024**3))
    sbuf_density = 2.44
    stram_density = 11.0
    hbm_density = 1.1
    ltram_density = 200.0

    assert hierarchy.area_budget_notes.get("stram_replaces_sbuf_fraction") == "0.5"
    assert hierarchy.area_budget_notes.get("ltram_replaces_hbm_fraction") == "0.25"

    # 50% of SBUF die area → StRAM; SBUF keeps 50% of nominal capacity.
    assert sbuf.capacity_bytes == int(nominal_sbuf * 0.5)
    sbuf_area = nominal_sbuf * 8 / sbuf_density
    expected_stram = int(0.5 * sbuf_area * stram_density / 8)
    assert stram.capacity_bytes == expected_stram
    assert stram.capacity_bytes > nominal_sbuf // 2  # eDRAM denser than SRAM

    # 25% of HBM die area → LtRAM; HBM keeps 75% of nominal capacity.
    assert hbm.capacity_bytes == int(nominal_hbm * 0.75)
    hbm_area = nominal_hbm * 8 / hbm_density
    expected_ltram = int(0.25 * hbm_area * ltram_density / 8)
    assert ltram.capacity_bytes == expected_ltram


def test_baseline_skips_area_tradeoff() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml",
        num_cores=4,
    )
    assert hierarchy.level_by_id("sbuf").capacity_bytes == 29_360_128
    assert not hierarchy.area_budget_notes
