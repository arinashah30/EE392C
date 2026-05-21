from __future__ import annotations

from collections import defaultdict

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
    """Greedy spill to next level when a memory pool overflows."""
    order = [level.id for level in hierarchy.enabled_levels]
    by_level: dict[str, list[TensorRecord]] = {level_id: [] for level_id in order}

    for tensor in tensors:
        by_level[homes[tensor.id]].append(tensor)

    for level in hierarchy.enabled_levels:
        pool_tensors = by_level[level.id]
        if not pool_tensors:
            continue

        if level.scope == "per_core":
            _enforce_per_core_pool(pool_tensors, level, homes, order, hierarchy)
        else:
            _enforce_chip_pool(pool_tensors, level, homes, order, hierarchy)


def _enforce_chip_pool(
    pool_tensors: list[TensorRecord],
    level,
    homes: dict[str, str],
    order: list[str],
    hierarchy: ResolvedHierarchy,
) -> None:
    total = sum(tensor.bytes for tensor in pool_tensors)
    if total <= level.capacity_bytes:
        return
    spill_target = _next_level_id(level.id, order, hierarchy)
    if spill_target is None:
        return
    pool_tensors.sort(key=lambda tensor: tensor.bytes, reverse=True)
    while total > level.capacity_bytes and pool_tensors:
        victim = pool_tensors.pop()
        homes[victim.id] = spill_target
        total -= victim.bytes


def _enforce_per_core_pool(
    pool_tensors: list[TensorRecord],
    level,
    homes: dict[str, str],
    order: list[str],
    hierarchy: ResolvedHierarchy,
) -> None:
    """Per-core levels (SBUF, StRAM): capacity applies independently on each NeuronCore."""
    by_core: dict[int, list[TensorRecord]] = defaultdict(list)
    for tensor in pool_tensors:
        by_core[tensor.core_id if tensor.core_id is not None else 0].append(tensor)

    spill_target = _next_level_id(level.id, order, hierarchy)
    if spill_target is None:
        return

    for core_tensors in by_core.values():
        total = sum(tensor.bytes for tensor in core_tensors)
        if total <= level.capacity_bytes:
            continue
        core_tensors.sort(key=lambda tensor: tensor.bytes, reverse=True)
        while total > level.capacity_bytes and core_tensors:
            victim = core_tensors.pop()
            homes[victim.id] = spill_target
            total -= victim.bytes


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
