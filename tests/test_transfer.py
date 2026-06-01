from pathlib import Path

from dmsim.config.loader import load_hierarchy
from dmsim.sim.transfer import (
    datapath_read_latency_ns,
    hops_between,
    path_between,
    transfer_latency_ns,
)


ROOT = Path(__file__).resolve().parents[1]


def test_hbm_to_sbuf_single_adjacent_hop_baseline() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    assert hops_between(hierarchy, "hbm", "sbuf") == [("hbm", "sbuf")]


def test_ltram_to_sbuf_is_direct_hop() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hops_between(hierarchy, "ltram", "sbuf") == [("ltram", "sbuf")]
    assert path_between(hierarchy, "ltram", "sbuf", home_id="ltram") == [
        ("ltram", "sbuf")
    ]


def test_hbm_to_sbuf_is_direct_hop_with_ltram_in_stack() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem_25hbm.yaml", num_cores=1
    )
    assert path_between(hierarchy, "hbm", "sbuf", home_id="hbm") == [("hbm", "sbuf")]
    assert hops_between(hierarchy, "hbm", "sbuf") == [("hbm", "sbuf")]
    assert hops_between(hierarchy, "sbuf", "hbm") == [("sbuf", "hbm")]


def test_stram_to_sbuf_uses_on_chip_bandwidth() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hierarchy.link_bandwidth_GBs("stram", "sbuf") == 10000.0


def test_off_chip_to_on_chip_uses_dma_bandwidth() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hierarchy.link_bandwidth_GBs("hbm", "sbuf") == 368.0
    assert hierarchy.link_bandwidth_GBs("ltram", "stram") == 368.0


def test_off_chip_to_off_chip_uses_dma_bandwidth() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_diff_mem.yaml", num_cores=1
    )
    assert hierarchy.link_bandwidth_GBs("hbm", "ltram") == 368.0


def test_interconnect_bandwidth_defaults() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    assert hierarchy.dma_bandwidth_GBs == 368.0
    assert hierarchy.on_chip_bandwidth_GBs == 10000.0


def test_transfer_latency_uses_access_latencies_plus_bytes_over_bw() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    sbuf = hierarchy.level_by_id("sbuf")
    hbm = hierarchy.level_by_id("hbm")
    nbytes = 1_000_000
    lat = transfer_latency_ns(hierarchy, hbm, sbuf, nbytes)
    bw_bytes_per_ns = 368.0
    expected = (
        hbm.tech.access.read_latency_ns
        + nbytes / bw_bytes_per_ns
        + sbuf.tech.access.write_latency_ns
    )
    assert lat == expected


def test_datapath_read_uses_tech_max_bandwidth_not_dma() -> None:
    hierarchy = load_hierarchy(
        ROOT / "configs/hierarchy/trainium2_baseline.yaml", num_cores=1
    )
    sbuf = hierarchy.level_by_id("sbuf")
    nbytes = 8192
    lat = datapath_read_latency_ns(sbuf, nbytes, hierarchy)
    assert lat == sbuf.tech.access.read_latency_ns + nbytes / hierarchy.on_chip_bandwidth_GBs
    assert lat < transfer_latency_ns(
        hierarchy, hierarchy.level_by_id("hbm"), sbuf, nbytes
    )
