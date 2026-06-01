from __future__ import annotations

from dmsim.config.models import ResolvedHierarchy, ResolvedLevel


def level_order(hierarchy: ResolvedHierarchy) -> list[str]:
    return [level.id for level in hierarchy.enabled_levels]


def hops_between(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
) -> list[tuple[str, str]]:
    """
    Direct memory-to-memory hop for a transfer.

    Transfers are one logical edge ``source → dest``. Multi-hop paths (e.g.
    staging through LtRAM) must appear as separate trace access events, not as
    an automatic walk along ``levels:`` order in YAML.
    """
    if source_id == dest_id:
        return []
    hierarchy.level_by_id(source_id)
    hierarchy.level_by_id(dest_id)
    return [(source_id, dest_id)]


# Backward-compatible alias
physical_hops_between = hops_between


def path_between(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
    *,
    home_id: str | None = None,
) -> list[tuple[str, str]]:
    """Hops used for access costing in ``_charge_path`` (always one direct edge)."""
    _ = home_id
    return hops_between(hierarchy, source_id, dest_id)


def transfer_latency_ns(
    hierarchy: ResolvedHierarchy,
    from_level: ResolvedLevel,
    to_level: ResolvedLevel,
    nbytes: int,
) -> float:
    bw_GBs = hierarchy.link_bandwidth_GBs(from_level.id, to_level.id)
    bw_bytes_per_ns = bw_GBs * 1e9 / 1e9
    hop_transfer = nbytes / bw_bytes_per_ns if bw_bytes_per_ns > 0 else 0.0
    read_lat = from_level.tech.access.read_latency_ns
    write_lat = to_level.tech.access.write_latency_ns
    return read_lat + hop_transfer + write_lat


def datapath_read_latency_ns(
    level: ResolvedLevel, nbytes: int, hierarchy: ResolvedHierarchy
) -> float:
    """On-chip read to compute (SBUF scratch hit, StRAM direct read)."""
    return level.tech.access.read_latency_ns + nbytes / hierarchy.on_chip_bandwidth_GBs


def access_latency_ns(
    level: ResolvedLevel, op: str, nbytes: int, hierarchy: ResolvedHierarchy
) -> float:
    if op == "read":
        return datapath_read_latency_ns(level, nbytes, hierarchy)
    line = level.tech.interface.line_size_bytes
    lines = max(1, (nbytes + line - 1) // line)
    return level.tech.access.write_latency_ns * lines


def access_energy_pJ(level: ResolvedLevel, op: str, nbytes: int) -> float:
    bits = nbytes * 8
    if op == "read":
        return bits * level.tech.access.read_energy_pJ_per_bit
    return bits * level.tech.access.write_energy_pJ_per_bit


def transfer_energy_pJ(
    hierarchy: ResolvedHierarchy,
    from_level: ResolvedLevel,
    to_level: ResolvedLevel,
    nbytes: int,
) -> float:
    return access_energy_pJ(from_level, "read", nbytes) + access_energy_pJ(to_level, "write", nbytes)


def transfer_latency_between_levels(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
    nbytes: int,
) -> float:
    """Latency for one direct transfer between levels."""
    from_level = hierarchy.level_by_id(source_id)
    to_level = hierarchy.level_by_id(dest_id)
    return transfer_latency_ns(hierarchy, from_level, to_level, nbytes)


def transfer_energy_between_levels(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
    nbytes: int,
) -> float:
    """Energy for one direct transfer between levels."""
    from_level = hierarchy.level_by_id(source_id)
    to_level = hierarchy.level_by_id(dest_id)
    return transfer_energy_pJ(hierarchy, from_level, to_level, nbytes)
