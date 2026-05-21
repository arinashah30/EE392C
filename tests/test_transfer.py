from pathlib import Path

from dmsim.config.loader import load_hierarchy
from dmsim.sim.transfer import path_between, physical_hops_between


ROOT = Path(__file__).resolve().parents[1]


def test_physical_path_includes_all_intermediate_levels() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    hops = physical_hops_between(hierarchy, "hbm", "sbuf")
    assert hops == [
        ("hbm", "ltram"),
        ("ltram", "stram"),
        ("stram", "sbuf"),
    ]


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
