from __future__ import annotations

from dmsim.config.models import ResolvedHierarchy, ResolvedLevel


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


def latency_ns(
    hierarchy: ResolvedHierarchy,
    nbytes: int,
    *,
    from_level: ResolvedLevel,
    to_level: ResolvedLevel | None = None,
) -> float:
    """
    Latency for a memory access.

    - ``to_level`` set: interconnect hop (read at source + nbytes / link BW +
      write at dest).
    - ``to_level`` None: local read at ``from_level`` (datapath:
      ``on_chip_bandwidth_GBs``).
    """
    if to_level is not None:
        bw_GBs = hierarchy.link_bandwidth_GBs(from_level.id, to_level.id)
        hop_transfer = nbytes / bw_GBs if bw_GBs > 0 else 0.0
        return (
            from_level.tech.access.read_latency_ns
            + hop_transfer
            + to_level.tech.access.write_latency_ns
        )
    return (
        from_level.tech.access.read_latency_ns
        + nbytes / hierarchy.on_chip_bandwidth_GBs
    )


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
    return access_energy_pJ(from_level, "read", nbytes) + access_energy_pJ(
        to_level, "write", nbytes
    )
