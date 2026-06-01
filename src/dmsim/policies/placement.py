"""Static tensor placement: map trace categories to hierarchy home levels.

Placement runs once at simulation start. It does not move tensors during the
trace — only adjusts homes when a tier is disabled or over capacity.

Policy YAML (`configs/policies/`) supplies category → desired level,
per-level spill/fallback targets (`fallback_by_level`; default hbm), and
spill victim ordering (`spill_victim_order`: best_case vs worst_case by trace
access count).
Hierarchy YAML supplies which levels exist and their capacities.
"""

from __future__ import annotations

from collections import defaultdict

from dmsim.config.models import PolicyConfig, ResolvedHierarchy
from dmsim.trace.schema import TensorRecord


def assign_home_levels(
    tensors: list[TensorRecord],
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
    *,
    access_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    """Assign each tensor a persistent home memory level for the simulation."""
    enabled_ids = {level.id for level in hierarchy.enabled_levels}
    homes: dict[str, str] = {}
    counts = access_counts or {}

    for tensor in tensors:
        category_key = tensor.category.value
        desired = policy.home_level_by_category.get(category_key, "hbm")

        if desired not in enabled_ids:
            desired = _fallback_level(desired, policy, enabled_ids)

        homes[tensor.id] = desired

    _enforce_capacities(tensors, homes, hierarchy, policy, counts)
    return homes


def _fallback_level(
    desired: str,
    policy: PolicyConfig,
    enabled_ids: set[str],
) -> str:
    """Pick an enabled home when policy desired level is disabled."""
    if desired in enabled_ids:
        return desired
    target = _resolve_spill_target(desired, policy, enabled_ids)
    if target is not None:
        return target
    return "hbm" if "hbm" in enabled_ids else next(iter(enabled_ids))


def _resolve_spill_target(
    level_id: str,
    policy: PolicyConfig,
    enabled_ids: set[str],
) -> str | None:
    """Follow ``policy.fallback_by_level`` until an enabled level != ``level_id``."""
    visited: set[str] = {level_id}
    current = policy.fallback_for(level_id)
    while True:
        if current in visited:
            return None
        visited.add(current)
        if current in enabled_ids and current != level_id:
            return current
        current = policy.fallback_for(current)


def _spill_victim_key(
    tensor: TensorRecord,
    access_counts: dict[str, int],
    policy: PolicyConfig,
) -> tuple:
    """Sort key for picking spill victims (lower = evicted first in best_case)."""
    access = access_counts.get(tensor.id, 0)
    if policy.spill_victim_order == "worst_case":
        return (-access, -tensor.bytes, tensor.id)
    return (access, tensor.bytes, tensor.id)


def _pick_spill_victim(
    pool: list[TensorRecord],
    access_counts: dict[str, int],
    policy: PolicyConfig,
) -> TensorRecord:
    return min(pool, key=lambda tensor: _spill_victim_key(tensor, access_counts, policy))


def _enforce_capacities(
    tensors: list[TensorRecord],
    homes: dict[str, str],
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
    access_counts: dict[str, int],
) -> None:
    """Spill oversized tensors via policy fallbacks; updates ``homes`` in place."""
    enabled_ids = {level.id for level in hierarchy.enabled_levels}
    by_level: dict[str, list[TensorRecord]] = {
        level.id: [] for level in hierarchy.enabled_levels
    }

    for tensor in tensors:
        by_level[homes[tensor.id]].append(tensor)

    for level in hierarchy.enabled_levels:
        pool_tensors = by_level[level.id]
        if not pool_tensors:
            continue

        if level.scope == "per_core":
            _enforce_per_core_pool(
                pool_tensors, level, homes, policy, enabled_ids, access_counts
            )
        else:
            _enforce_chip_pool(
                pool_tensors, level, homes, policy, enabled_ids, access_counts
            )


def _enforce_chip_pool(
    pool_tensors: list[TensorRecord],
    level,
    homes: dict[str, str],
    policy: PolicyConfig,
    enabled_ids: set[str],
    access_counts: dict[str, int],
) -> None:
    """Enforce one chip-wide capacity pool (LtRAM or HBM)."""
    total = sum(tensor.bytes for tensor in pool_tensors)
    if total <= level.capacity_bytes:
        return
    spill_target = _resolve_spill_target(level.id, policy, enabled_ids)
    if spill_target is None:
        return
    pool = list(pool_tensors)
    while total > level.capacity_bytes and pool:
        victim = _pick_spill_victim(pool, access_counts, policy)
        pool.remove(victim)
        homes[victim.id] = spill_target
        total -= victim.bytes


def _enforce_per_core_pool(
    pool_tensors: list[TensorRecord],
    level,
    homes: dict[str, str],
    policy: PolicyConfig,
    enabled_ids: set[str],
    access_counts: dict[str, int],
) -> None:
    """Enforce per-NeuronCore capacity (SBUF, StRAM)."""
    spill_target = _resolve_spill_target(level.id, policy, enabled_ids)
    if spill_target is None:
        return

    by_core: dict[int, list[TensorRecord]] = defaultdict(list)
    for tensor in pool_tensors:
        by_core[tensor.core_id if tensor.core_id is not None else 0].append(tensor)

    for core_tensors in by_core.values():
        total = sum(tensor.bytes for tensor in core_tensors)
        if total <= level.capacity_bytes:
            continue
        pool = list(core_tensors)
        while total > level.capacity_bytes and pool:
            victim = _pick_spill_victim(pool, access_counts, policy)
            pool.remove(victim)
            homes[victim.id] = spill_target
            total -= victim.bytes
