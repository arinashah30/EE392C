"""Residency-aware replay for placement spill (matches simulator hop charging)."""

from __future__ import annotations

from collections import defaultdict

from dmsim.config.models import PolicyConfig, ResolvedHierarchy
from dmsim.sim.residency import FastBufferState, LevelPoolState, TensorResidency
from dmsim.sim.transfer import latency_ns
from dmsim.trace.schema import AccessEvent, KernelBoundaryEvent, Trace


def build_retention_by_residency_replay(
    trace: Trace,
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
    homes: dict[str, str],
    pool_level: str,
    spill_target: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Per-tensor benefit (ns) of homing in ``pool_level`` vs ``spill_target``.

    Counts ``latency_ns(spill→target) - latency_ns(pool→target)`` only on trace
    reads that actually incur an interconnect hop **from** ``pool_level`` (the
    tensor's home), matching when the simulator charges a home-level DMA.

    Also returns ``home_hop_bytes`` (Σ event bytes on those hops) for tie-breaks
    when per-hop latency delta is constant across transfer sizes.
    """
    tensor_map = trace.tensor_map()
    residency: dict[str, TensorResidency] = {
        tensor_id: TensorResidency(home_level=home, resident_level=home)
        for tensor_id, home in homes.items()
    }
    pools: dict[str, LevelPoolState] = {
        level.id: LevelPoolState(capacity_bytes=level.capacity_bytes)
        for level in hierarchy.enabled_levels
        if level.scope != "per_core"
    }
    fast_buffers: dict[int, dict[str, FastBufferState]] = {}
    retention: dict[str, float] = defaultdict(float)
    home_hop_bytes: dict[str, int] = defaultdict(int)

    pool_resolved = hierarchy.level_by_id(pool_level)
    spill_resolved = hierarchy.level_by_id(spill_target)
    target_id = policy.default_access_target
    to_level = hierarchy.level_by_id(target_id)

    _bootstrap_near_memory_homes(
        tensor_map, homes, hierarchy, residency, fast_buffers, pools
    )

    for event in trace.parsed_events():
        if isinstance(event, KernelBoundaryEvent):
            _handle_kernel_boundary(
                event, hierarchy, fast_buffers, residency, tensor_map
            )
            continue
        if not isinstance(event, AccessEvent):
            continue
        if event.op != "read":
            continue

        tensor = tensor_map.get(event.tensor_id)
        if tensor is None:
            continue

        state = residency[event.tensor_id]
        target = event.target_level or policy.default_access_target
        source = _source_level_for_access(event, state, policy, hierarchy)

        if _is_direct_stram_read(event, state, policy):
            continue

        if source != target and source == state.home_level == pool_level:
            lat_pool = latency_ns(
                hierarchy,
                event.bytes,
                from_level=pool_resolved,
                to_level=to_level,
            )
            lat_spill = latency_ns(
                hierarchy,
                event.bytes,
                from_level=spill_resolved,
                to_level=to_level,
            )
            retention[event.tensor_id] += int(lat_spill - lat_pool)
            home_hop_bytes[event.tensor_id] += event.bytes

        _advance_access_residency(
            event, hierarchy, policy, residency, fast_buffers
        )

    return dict(retention), dict(home_hop_bytes)


def _fast_buffer(
    fast_buffers: dict[int, dict[str, FastBufferState]],
    core_id: int,
    level_id: str,
) -> FastBufferState:
    if core_id not in fast_buffers:
        fast_buffers[core_id] = {}
    if level_id not in fast_buffers[core_id]:
        fast_buffers[core_id][level_id] = FastBufferState()
    return fast_buffers[core_id][level_id]


def _bootstrap_near_memory_homes(
    tensor_map: dict,
    homes: dict[str, str],
    hierarchy: ResolvedHierarchy,
    residency: dict[str, TensorResidency],
    fast_buffers: dict[int, dict[str, FastBufferState]],
    pools: dict[str, LevelPoolState],
) -> None:
    deepest = hierarchy.enabled_levels[-1].id
    wipe_ids = set(hierarchy.kernel.wipe_levels_on_boundary)
    for tensor_id, home_level in homes.items():
        if home_level == deepest or home_level in wipe_ids:
            continue
        tensor = tensor_map[tensor_id]
        level = hierarchy.level_by_id(home_level)
        if level.scope == "per_core":
            core_id = tensor.core_id if tensor.core_id is not None else 0
            _install_in_fast_buffer(
                hierarchy,
                fast_buffers,
                residency,
                core_id,
                home_level,
                tensor_id,
                tensor.bytes,
            )
        else:
            pool = pools.get(home_level)
            if pool is not None:
                pool.install(tensor_id, tensor.bytes)
        residency[tensor_id].resident_level = home_level


def _install_in_fast_buffer(
    hierarchy: ResolvedHierarchy,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    residency: dict[str, TensorResidency],
    core_id: int,
    level_id: str,
    tensor_id: str,
    nbytes: int,
) -> None:
    level = hierarchy.level_by_id(level_id)
    if level.scope != "per_core":
        return
    buffer = _fast_buffer(fast_buffers, core_id, level_id)
    if tensor_id not in buffer.occupants and buffer.used_bytes + nbytes > level.capacity_bytes:
        _evict_from_fast_buffer(buffer, residency)
    if tensor_id in buffer.occupants:
        buffer.used_bytes -= buffer.occupants[tensor_id]
    buffer.occupants[tensor_id] = nbytes
    buffer.used_bytes += nbytes


def _evict_from_fast_buffer(
    buffer: FastBufferState,
    residency: dict[str, TensorResidency],
) -> None:
    if not buffer.occupants:
        return
    victim_id = next(iter(buffer.occupants))
    buffer.used_bytes -= buffer.occupants.pop(victim_id)
    state = residency.get(victim_id)
    if state is not None:
        state.resident_level = state.home_level


def _tensor_core_id(tensor_map: dict, tensor_id: str) -> int:
    tensor = tensor_map.get(tensor_id)
    if tensor is None or tensor.core_id is None:
        return 0
    return tensor.core_id


def _kernel_wipe_cores(
    event: KernelBoundaryEvent,
    fast_buffers: dict[int, dict[str, FastBufferState]],
) -> list[int]:
    if event.core_id is not None:
        return [event.core_id]
    cores = list(fast_buffers.keys())
    return cores if cores else [0]


def _handle_kernel_boundary(
    event: KernelBoundaryEvent,
    hierarchy: ResolvedHierarchy,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    residency: dict[str, TensorResidency],
    tensor_map: dict,
) -> None:
    if event.type != "kernel_end":
        return
    wipe_ids = set(hierarchy.kernel.wipe_levels_on_boundary)
    core_set = set(_kernel_wipe_cores(event, fast_buffers))
    for core_id in core_set:
        for level_id in wipe_ids:
            if level_id in fast_buffers.get(core_id, {}):
                fast_buffers[core_id][level_id].clear()
    for tensor_id, state in residency.items():
        if _tensor_core_id(tensor_map, tensor_id) not in core_set:
            continue
        if state.resident_level in wipe_ids:
            state.resident_level = state.home_level


def _source_level_for_access(
    event: AccessEvent,
    state: TensorResidency,
    policy: PolicyConfig,
    hierarchy: ResolvedHierarchy,
) -> str:
    resident = state.resident_level or state.home_level
    target = event.target_level or policy.default_access_target
    if event.op != "write":
        return resident
    try:
        target_level = hierarchy.level_by_id(target)
    except KeyError:
        return resident
    if target_level.interconnect == "off_chip":
        return policy.default_access_target
    return resident


def _is_direct_stram_read(
    event: AccessEvent,
    state: TensorResidency,
    policy: PolicyConfig,
) -> bool:
    if event.op != "read":
        return False
    target = event.target_level or policy.default_access_target
    if target != policy.default_access_target:
        return False
    if state.home_level != "stram":
        return False
    resident = state.resident_level or state.home_level
    return resident == state.home_level


def _advance_access_residency(
    event: AccessEvent,
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
    residency: dict[str, TensorResidency],
    fast_buffers: dict[int, dict[str, FastBufferState]],
) -> None:
    state = residency[event.tensor_id]
    target = event.target_level or policy.default_access_target
    source = _source_level_for_access(event, state, policy, hierarchy)

    if _is_direct_stram_read(event, state, policy):
        return

    if source != target:
        _install_in_fast_buffer(
            hierarchy,
            fast_buffers,
            residency,
            event.core_id,
            target,
            event.tensor_id,
            event.bytes,
        )
        state.resident_level = target
    elif event.op == "write":
        return
