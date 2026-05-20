from __future__ import annotations

from dmsim.config.models import ResolvedHierarchy, ResolvedLevel


def level_order(hierarchy: ResolvedHierarchy) -> list[str]:
    return [level.id for level in hierarchy.enabled_levels]


def path_between(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
) -> list[tuple[str, str]]:
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
