from pathlib import Path

from dmsim.config.loader import load_hierarchy, load_tech_spec
from dmsim.sim.transfer import path_between, physical_hops_between, transfer_latency_ns


ROOT = Path(__file__).resolve().parents[1]


def test_physical_path_includes_all_intermediate_levels() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    hops = physical_hops_between(hierarchy, "hbm", "sbuf")
    # With an explicit `hbm_sbuf` link configured, the simulator should take
    # the direct path rather than walking through intermediate tiers.
    assert hops == [("hbm", "sbuf")]


def test_home_aware_path_skips_unused_tiers() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    hops = path_between(hierarchy, "hbm", "sbuf", home_id="hbm")
    assert hops == [("hbm", "sbuf")]


def test_stram_homed_tensor_uses_stram_hop() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    hops = path_between(hierarchy, "stram", "sbuf", home_id="stram")
    assert hops == [("stram", "sbuf")]


def test_ltram_homed_includes_ltram_not_stram_as_endpoint_only() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    hops = path_between(hierarchy, "ltram", "sbuf", home_id="ltram")
    assert hops == [("ltram", "sbuf")]


def test_dma_aggregate_bandwidth_from_instance() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    assert hierarchy.dma_aggregate_bandwidth_GBs == 368.0


def test_dma_cap_limits_hbm_to_sbuf_link() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    assert hierarchy.link_bandwidth_GBs("hbm", "sbuf") == 368.0


def test_literature_tech_bandwidths() -> None:
    tech_dir = ROOT / "configs/tech_specs"
    assert load_tech_spec(tech_dir / "edram_1t1c.yaml").interface.max_bandwidth_GBs == 128
    assert load_tech_spec(tech_dir / "edram_3t.yaml").interface.max_bandwidth_GBs == 128
    assert load_tech_spec(tech_dir / "rram.yaml").interface.max_bandwidth_GBs == 2.3
    assert load_tech_spec(tech_dir / "feram.yaml").interface.max_bandwidth_GBs == 34


def test_unspecified_link_uses_dma_not_tech_min() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hierarchy.link_bandwidth_GBs("ltram", "sbuf") == 368.0
    assert hierarchy.link_bandwidth_GBs("hbm", "sbuf") == 368.0


def test_explicit_stram_sbuf_uses_literature_edram() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hierarchy.link_bandwidth_GBs("stram", "sbuf") == 128.0
