from __future__ import annotations

from dmsim.config.models import ResolvedHierarchy, ResolvedLevel


def level_order(hierarchy: ResolvedHierarchy) -> list[str]:
    return [level.id for level in hierarchy.enabled_levels]


def physical_hops_between(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
) -> list[tuple[str, str]]:
    """Every adjacent link along the hierarchy between source and dest."""
    order = level_order(hierarchy)
    i = order.index(source_id)
    j = order.index(dest_id)
    if i == j:
        return []
    step = 1 if j > i else -1
    hops: list[tuple[str, str]] = []
    idx = i
    while idx != j:
        nxt = idx + step
        hops.append((order[idx], order[nxt]))
        idx = nxt
    return hops


def path_between(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
    *,
    home_id: str | None = None,
) -> list[tuple[str, str]]:
    """
    Logical transfer hops for a tensor access.

    Only levels that are the source, destination, or persistent home appear
    as endpoints. Intermediate tiers (e.g. StRAM/LtRAM) are skipped when data
    is homed in HBM — matching a page table that routes directly from home.
    """
    order = level_order(hierarchy)
    i = order.index(source_id)
    j = order.index(dest_id)
    if i == j:
        return []

    anchor = home_id if home_id is not None else source_id
    anchors = {source_id, dest_id, anchor}
    step = 1 if j > i else -1

    waypoints: list[str] = []
    idx = i
    while True:
        level_id = order[idx]
        if level_id in anchors and (not waypoints or waypoints[-1] != level_id):
            waypoints.append(level_id)
        if idx == j:
            break
        idx += step

    return list(zip(waypoints, waypoints[1:]))


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


def access_latency_ns(level: ResolvedLevel, op: str, nbytes: int) -> float:
    bits = nbytes * 8
    line = level.tech.interface.line_size_bytes
    lines = max(1, (nbytes + line - 1) // line)
    per_line = (
        level.tech.access.read_latency_ns
        if op == "read"
        else level.tech.access.write_latency_ns
    )
    return per_line * lines


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
    """Sum latency across all physical links between two hierarchy levels."""
    total = 0.0
    for hop_from, hop_to in physical_hops_between(hierarchy, source_id, dest_id):
        from_level = hierarchy.level_by_id(hop_from)
        to_level = hierarchy.level_by_id(hop_to)
        total += transfer_latency_ns(hierarchy, from_level, to_level, nbytes)
    return total


def transfer_energy_between_levels(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
    nbytes: int,
) -> float:
    """Sum energy across all physical links between two hierarchy levels."""
    total = 0.0
    for hop_from, hop_to in physical_hops_between(hierarchy, source_id, dest_id):
        from_level = hierarchy.level_by_id(hop_from)
        to_level = hierarchy.level_by_id(hop_to)
        total += transfer_energy_pJ(hierarchy, from_level, to_level, nbytes)
    return total
