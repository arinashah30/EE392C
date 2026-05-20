from __future__ import annotations

from dmsim.config.models import PolicyConfig, ResolvedHierarchy
from dmsim.trace.schema import TensorRecord, TensorCategory


def assign_home_levels(
    tensors: list[TensorRecord],
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
) -> dict[str, str]:
    enabled_ids = {level.id for level in hierarchy.enabled_levels}
    homes: dict[str, str] = {}

    for tensor in tensors:
        category_key = tensor.category.value
        desired = policy.home_level_by_category.get(category_key, "hbm")

        if desired not in enabled_ids:
            # Fall back toward HBM along enabled stack
            desired = _fallback_level(desired, hierarchy, enabled_ids)

        homes[tensor.id] = desired

    _enforce_capacities(tensors, homes, hierarchy)
    return homes


def _fallback_level(
    desired: str,
    hierarchy: ResolvedHierarchy,
    enabled_ids: set[str],
) -> str:
    order = [level.id for level in hierarchy.enabled_levels]
    if desired in enabled_ids:
        return desired
    try:
        idx = next(i for i, level in enumerate(hierarchy.levels) if level.id == desired)
    except StopIteration:
        return "hbm" if "hbm" in enabled_ids else next(iter(enabled_ids))

    for level in hierarchy.levels[idx + 1 :]:
        if level.enabled and level.id in enabled_ids:
            return level.id
    return "hbm"


def _enforce_capacities(
    tensors: list[TensorRecord],
    homes: dict[str, str],
    hierarchy: ResolvedHierarchy,
) -> None:
    """Greedy spill to next level when a per-chip pool overflows."""
    order = [level.id for level in hierarchy.enabled_levels]
    by_level: dict[str, list[TensorRecord]] = {level_id: [] for level_id in order}

    for tensor in tensors:
        by_level[homes[tensor.id]].append(tensor)

    for level in hierarchy.enabled_levels:
        if level.scope == "per_core":
            continue
        pool = by_level[level.id]
        total = sum(tensor.bytes for tensor in pool)
        if total <= level.capacity_bytes:
            continue
        spill_target = _next_level_id(level.id, order, hierarchy)
        if spill_target is None:
            continue
        pool.sort(key=lambda tensor: tensor.bytes, reverse=True)
        while total > level.capacity_bytes and pool:
            victim = pool.pop()
            homes[victim.id] = spill_target
            total -= victim.bytes
            by_level[spill_target].append(victim)


def _next_level_id(
    current: str,
    order: list[str],
    hierarchy: ResolvedHierarchy,
) -> str | None:
    try:
        idx = order.index(current)
    except ValueError:
        return None
    for level_id in order[idx + 1 :]:
        level = hierarchy.level_by_id(level_id)
        if level.enabled:
            return level_id
    return None
