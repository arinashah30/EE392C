from __future__ import annotations

from dataclasses import dataclass, field

from dmsim.config.models import PolicyConfig, ResolvedHierarchy
from dmsim.policies.placement import assign_home_levels
from dmsim.sim.residency import FastBufferState, LevelPoolState, TensorResidency
from dmsim.sim.transfer import (
    access_energy_pJ,
    access_latency_ns,
    path_between,
    transfer_energy_between_levels,
    transfer_latency_between_levels,
)
from dmsim.trace.schema import (
    AccessEvent,
    KernelBoundaryEvent,
    Trace,
    TraceEvent,
)


@dataclass
class SimulationResult:
    hierarchy_name: str
    policy_name: str
    trace_workload: str
    total_time_ns: float
    total_energy_pJ: float
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    retention_evictions: int = 0
    corrupt_accesses: int = 0
    kernel_wipes: int = 0
    transfers_by_hop: dict[str, int] = field(default_factory=dict)
    energy_by_level_pJ: dict[str, float] = field(default_factory=dict)
    latency_by_level_ns: dict[str, float] = field(default_factory=dict)

    @property
    def hbm_traffic_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes


def run_simulation(
    trace: Trace,
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
) -> SimulationResult:
    tensor_map = trace.tensor_map()
    homes = assign_home_levels(trace.tensors, hierarchy, policy)

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
    _seed_home_allocations(
        tensor_map, homes, hierarchy, residency, fast_buffers, pools
    )

    result = SimulationResult(
        hierarchy_name=hierarchy.name,
        policy_name=policy.name,
        trace_workload=trace.metadata.workload,
        total_time_ns=0.0,
        total_energy_pJ=0.0,
    )

    for event in trace.parsed_events():
        if isinstance(event, KernelBoundaryEvent):
            _handle_kernel_boundary(event, hierarchy, fast_buffers, residency, result)
            continue
        if isinstance(event, AccessEvent):
            _handle_access(
                event,
                hierarchy,
                policy,
                tensor_map,
                residency,
                pools,
                fast_buffers,
                result,
            )

    return result


def _seed_home_allocations(
    tensor_map: dict,
    homes: dict[str, str],
    hierarchy: ResolvedHierarchy,
    residency: dict[str, TensorResidency],
    fast_buffers: dict[int, dict[str, FastBufferState]],
    pools: dict[str, LevelPoolState],
) -> None:
    """Reserve capacity for tensors homed in persistent per-core levels (e.g. StRAM)."""
    for tensor_id, home_level in homes.items():
        level = hierarchy.level_by_id(home_level)
        if level.scope != "per_core" or home_level in hierarchy.kernel.wipe_levels_on_boundary:
            continue
        tensor = tensor_map[tensor_id]
        core_id = tensor.core_id if tensor.core_id is not None else 0
        _install_in_fast_buffer(
            hierarchy,
            fast_buffers,
            core_id,
            home_level,
            tensor_id,
            tensor.bytes,
            pools,
        )
        residency[tensor_id].resident_level = home_level


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


def _handle_kernel_boundary(
    event: KernelBoundaryEvent,
    hierarchy: ResolvedHierarchy,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    residency: dict[str, TensorResidency],
    result: SimulationResult,
) -> None:
    if event.type != "kernel_end":
        return
    wipe_ids = set(hierarchy.kernel.wipe_levels_on_boundary)
    cores = [event.core_id] if event.core_id is not None else list(fast_buffers.keys())
    if not cores and wipe_ids:
        cores = [0]
    for core_id in cores:
        for level_id in wipe_ids:
            if level_id in fast_buffers.get(core_id, {}):
                fast_buffers[core_id][level_id].clear()
    for state in residency.values():
        if state.resident_level in wipe_ids:
            state.resident_level = state.home_level
    result.kernel_wipes += 1


def _handle_access(
    event: AccessEvent,
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
    tensor_map: dict,
    residency: dict[str, TensorResidency],
    pools: dict[str, LevelPoolState],
    fast_buffers: dict[int, dict[str, FastBufferState]],
    result: SimulationResult,
) -> None:
    tensor = tensor_map.get(event.tensor_id)
    if tensor is None:
        raise KeyError(f"unknown tensor_id in trace: {event.tensor_id}")

    state = residency[event.tensor_id]
    target = event.target_level or policy.default_access_target
    nbytes = event.bytes
    core_id = event.core_id

    source_level = state.resident_level or state.home_level
    home_level = hierarchy.level_by_id(state.home_level)

    if _check_retention_expired(state, hierarchy, event.t_ns):
        state.corrupt = True
        result.retention_evictions += 1
        state.resident_level = None

    if state.corrupt:
        result.corrupt_accesses += 1
        source_level = state.home_level
        reload_from = _deepest_enabled(hierarchy)
        _charge_path(
            hierarchy,
            reload_from,
            state.home_level,
            nbytes,
            result,
            count_hbm=True,
            home_id=state.home_level,
        )
        state.corrupt = False
        state.resident_level = state.home_level
        _touch_home(state, event.t_ns)
        source_level = state.home_level

    if source_level != target:
        _charge_path(
            hierarchy,
            source_level,
            target,
            nbytes,
            result,
            count_hbm=True,
            home_id=state.home_level,
        )
        _install_in_fast_buffer(
            hierarchy, fast_buffers, core_id, target, event.tensor_id, nbytes, pools
        )
        state.resident_level = target
    else:
        level = hierarchy.level_by_id(target)
        lat = access_latency_ns(level, event.op, nbytes)
        eng = access_energy_pJ(level, event.op, nbytes)
        result.total_time_ns += lat
        result.total_energy_pJ += eng
        _accumulate_level(result, target, lat, eng)

    _touch_home(state, event.t_ns)


def _touch_home(state: TensorResidency, t_ns: float) -> None:
    state.last_home_touch_ns = t_ns


def _check_retention_expired(
    state: TensorResidency,
    hierarchy: ResolvedHierarchy,
    t_ns: float,
) -> bool:
    if state.last_home_touch_ns is None:
        return False
    level = hierarchy.level_by_id(state.home_level)
    if not level.has_retention:
        return False
    assert level.tech.retention_s is not None
    elapsed_s = (t_ns - state.last_home_touch_ns) * 1e-9
    return elapsed_s > level.tech.retention_s


def _deepest_enabled(hierarchy: ResolvedHierarchy) -> str:
    return hierarchy.enabled_levels[-1].id


def _charge_path(
    hierarchy: ResolvedHierarchy,
    source_id: str,
    dest_id: str,
    nbytes: int,
    result: SimulationResult,
    *,
    count_hbm: bool,
    home_id: str,
) -> None:
    if source_id == dest_id:
        return
    for hop_from, hop_to in path_between(
        hierarchy, source_id, dest_id, home_id=home_id
    ):
        lat = transfer_latency_between_levels(hierarchy, hop_from, hop_to, nbytes)
        eng = transfer_energy_between_levels(hierarchy, hop_from, hop_to, nbytes)
        result.total_time_ns += lat
        result.total_energy_pJ += eng
        hop_key = f"{hop_from}->{hop_to}"
        result.transfers_by_hop[hop_key] = result.transfers_by_hop.get(hop_key, 0) + 1
        _accumulate_level(result, hop_from, lat * 0.5, eng * 0.5)
        _accumulate_level(result, hop_to, lat * 0.5, eng * 0.5)
        if count_hbm and hop_from == "hbm":
            result.hbm_read_bytes += nbytes
        if count_hbm and hop_to == "hbm":
            result.hbm_write_bytes += nbytes


def _accumulate_level(
    result: SimulationResult,
    level_id: str,
    latency_ns: float,
    energy_pJ: float,
) -> None:
    result.latency_by_level_ns[level_id] = (
        result.latency_by_level_ns.get(level_id, 0.0) + latency_ns
    )
    result.energy_by_level_pJ[level_id] = (
        result.energy_by_level_pJ.get(level_id, 0.0) + energy_pJ
    )


def _install_in_fast_buffer(
    hierarchy: ResolvedHierarchy,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    core_id: int,
    level_id: str,
    tensor_id: str,
    nbytes: int,
    pools: dict[str, LevelPoolState],
) -> None:
    level = hierarchy.level_by_id(level_id)
    if level.scope != "per_core":
        pool = pools.get(level_id)
        if pool is not None:
            if not pool.can_fit(nbytes) and tensor_id not in pool.occupants:
                _evict_lru(pool)
            pool.install(tensor_id, nbytes)
        return

    buffer = _fast_buffer(fast_buffers, core_id, level_id)
    if tensor_id not in buffer.occupants and buffer.used_bytes + nbytes > level.capacity_bytes:
        _evict_lru_fast(buffer)
    if tensor_id in buffer.occupants:
        buffer.used_bytes -= buffer.occupants[tensor_id]
    buffer.occupants[tensor_id] = nbytes
    buffer.used_bytes += nbytes


def _evict_lru(pool: LevelPoolState) -> None:
    if not pool.occupants:
        return
    victim = next(iter(pool.occupants))
    pool.remove(victim)


def _evict_lru_fast(buffer: FastBufferState) -> None:
    if not buffer.occupants:
        return
    victim = next(iter(buffer.occupants))
    buffer.used_bytes -= buffer.occupants.pop(victim)
