"""Trace-driven memory hierarchy simulator. See README.md in this package."""

from __future__ import annotations

from dataclasses import dataclass, field

from dmsim.config.models import PolicyConfig, ResolvedHierarchy
from dmsim.policies.placement import assign_home_levels
from dmsim.sim.residency import FastBufferState, LevelPoolState, TensorResidency
from dmsim.sim.transfer import (
    access_energy_pJ,
    access_latency_ns,
    path_between,
    transfer_energy_pJ,
    transfer_latency_ns,
)
from dmsim.trace.schema import (
    AccessEvent,
    KernelBoundaryEvent,
    Trace,
)


@dataclass
class SimulationResult:
    hierarchy_name: str
    policy_name: str
    trace_workload: str
    total_time_ns: float
    total_energy_pJ: float
    time_by_core_ns: dict[int, float] = field(default_factory=dict)
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    retention_evictions: int = 0
    corrupt_accesses: int = 0
    kernel_wipes: int = 0
    refresh_energy_pJ: float = 0.0
    refresh_cycles_by_level: dict[str, int] = field(default_factory=dict)
    transfers_by_hop: dict[str, int] = field(default_factory=dict)
    energy_by_level_pJ: dict[str, float] = field(default_factory=dict)
    latency_by_level_ns: dict[str, float] = field(default_factory=dict)

    @property
    def hbm_traffic_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def worst_core_id(self) -> int | None:
        if not self.time_by_core_ns:
            return None
        return max(self.time_by_core_ns, key=self.time_by_core_ns.get)


def run_simulation(
    trace: Trace,
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
) -> SimulationResult:
    tensor_map = trace.tensor_map()
    homes = assign_home_levels(
        trace.tensors, hierarchy, policy, access_counts=trace.access_counts()
    )
    deepest = _deepest_enabled(hierarchy)

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
        tensor_map, homes, hierarchy, residency, fast_buffers, pools, deepest
    )
    _bootstrap_near_memory_homes(
        tensor_map, homes, hierarchy, residency, fast_buffers, pools, deepest
    )

    result = SimulationResult(
        hierarchy_name=hierarchy.name,
        policy_name=policy.name,
        trace_workload=trace.metadata.workload,
        total_time_ns=0.0,
        total_energy_pJ=0.0,
    )

    parsed = trace.parsed_events()
    prev_t_ns: float | None = parsed[0].t_ns if parsed else None

    for event in parsed:
        if prev_t_ns is not None and event.t_ns > prev_t_ns:
            _apply_refresh_energy_between(
                hierarchy,
                fast_buffers,
                pools,
                prev_t_ns,
                event.t_ns,
                result,
            )
        prev_t_ns = event.t_ns
        if isinstance(event, KernelBoundaryEvent):
            _handle_kernel_boundary(
                event, hierarchy, fast_buffers, residency, tensor_map, result
            )
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
                deepest,
            )

    result.total_time_ns = max(result.time_by_core_ns.values()) if result.time_by_core_ns else 0.0
    return result


def _add_core_latency(result: SimulationResult, core_id: int, latency_ns: float) -> None:
    result.time_by_core_ns[core_id] = (
        result.time_by_core_ns.get(core_id, 0.0) + latency_ns
    )


def _apply_refresh_energy_between(
    hierarchy: ResolvedHierarchy,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    pools: dict[str, LevelPoolState],
    start_t_ns: float,
    end_t_ns: float,
    result: SimulationResult,
) -> None:
    if end_t_ns <= start_t_ns:
        return

    for level in hierarchy.enabled_levels:
        tech = level.tech
        interval_s = tech.refresh_interval_s
        if interval_s is None:
            if not tech.volatile:
                continue
            if not tech.retention_s or tech.retention_s <= 0:
                continue
            interval_s = tech.retention_s
        if interval_s <= 0:
            continue
        interval_ns = interval_s * 1e9
        energy_pJ_per_bit = (
            tech.refresh_energy_pJ_per_bit
            if tech.refresh_energy_pJ_per_bit is not None
            else tech.access.write_energy_pJ_per_bit
        )

        if level.scope == "per_core":
            occupied = 0
            for core_state in fast_buffers.values():
                buf = core_state.get(level.id)
                if buf is not None:
                    occupied += buf.used_bytes
        else:
            occupied = pools.get(level.id).used_bytes if level.id in pools else 0
        if occupied <= 0:
            continue

        start_tick = int(start_t_ns // interval_ns)
        end_tick = int(end_t_ns // interval_ns)
        refreshes = max(0, end_tick - start_tick)
        if refreshes == 0:
            continue

        energy = refreshes * (occupied * 8) * energy_pJ_per_bit
        result.refresh_energy_pJ += energy
        result.total_energy_pJ += energy
        result.refresh_cycles_by_level[level.id] = result.refresh_cycles_by_level.get(level.id, 0) + refreshes
        _accumulate_level(result, level.id, 0.0, energy)


def _seed_home_allocations(
    tensor_map: dict,
    homes: dict[str, str],
    hierarchy: ResolvedHierarchy,
    residency: dict[str, TensorResidency],
    fast_buffers: dict[int, dict[str, FastBufferState]],
    pools: dict[str, LevelPoolState],
    deepest: str,
) -> None:
    for tensor_id, home_level in homes.items():
        if home_level != deepest:
            continue
        level = hierarchy.level_by_id(home_level)
        tensor = tensor_map[tensor_id]
        if level.scope == "per_core":
            if home_level in hierarchy.kernel.wipe_levels_on_boundary:
                continue
            core_id = tensor.core_id if tensor.core_id is not None else 0
            _install_in_fast_buffer(
                hierarchy,
                fast_buffers,
                core_id,
                home_level,
                tensor_id,
                tensor.bytes,
                pools,
                residency,
            )
        else:
            pool = pools.get(home_level)
            if pool is not None:
                pool.install(tensor_id, tensor.bytes)


def _bootstrap_near_memory_homes(
    tensor_map: dict,
    homes: dict[str, str],
    hierarchy: ResolvedHierarchy,
    residency: dict[str, TensorResidency],
    fast_buffers: dict[int, dict[str, FastBufferState]],
    pools: dict[str, LevelPoolState],
    deepest: str,
) -> None:
    """Install near-memory homes at t=0 (decode: weights already programmed before profile)."""
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
                core_id,
                home_level,
                tensor_id,
                tensor.bytes,
                pools,
                residency,
            )
        else:
            pool = pools.get(home_level)
            if pool is not None:
                pool.install(tensor_id, tensor.bytes)
        state = residency[tensor_id]
        state.resident_level = home_level
        state.initialized_at_home = True


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


def _source_level_for_access(
    event: AccessEvent,
    state: TensorResidency,
    policy: PolicyConfig,
    hierarchy: ResolvedHierarchy,
) -> str:
    """
    Where the access reads data from.

    Trace ``write`` events to off-chip ``target_level`` are SBUF→memory flushes
    (Neuron DMA SBUF→HBM), even when ``resident_level`` is already at home.
    """
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
    """
    Trace loads into SBUF but tensor is homed in StRAM and resident at home.

    Charge ``access_latency_ns(stram)`` — same line-local model as SBUF scratch
    hits, no interconnect ``nbytes/BW`` term.
    """
    if event.op != "read":
        return False
    target = event.target_level or policy.default_access_target
    if target != policy.default_access_target:
        return False
    if state.home_level != "stram":
        return False
    resident = state.resident_level or state.home_level
    return resident == state.home_level


def _tensor_core_id(tensor_map: dict, tensor_id: str) -> int:
    tensor = tensor_map.get(tensor_id)
    if tensor is None or tensor.core_id is None:
        return 0
    return tensor.core_id


def _kernel_wipe_cores(
    event: KernelBoundaryEvent,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    wipe_ids: set[str],
) -> list[int]:
    """NeuronCore ids whose fast buffers and scratch residency are reset."""
    if event.core_id is not None:
        return [event.core_id]
    cores = list(fast_buffers.keys())
    if not cores and wipe_ids:
        return [0]
    return cores


def _handle_kernel_boundary(
    event: KernelBoundaryEvent,
    hierarchy: ResolvedHierarchy,
    fast_buffers: dict[int, dict[str, FastBufferState]],
    residency: dict[str, TensorResidency],
    tensor_map: dict,
    result: SimulationResult,
) -> None:
    if event.type != "kernel_end":
        return
    wipe_ids = set(hierarchy.kernel.wipe_levels_on_boundary)
    cores = _kernel_wipe_cores(event, fast_buffers, wipe_ids)
    core_set = set(cores)
    for core_id in cores:
        for level_id in wipe_ids:
            if level_id in fast_buffers.get(core_id, {}):
                fast_buffers[core_id][level_id].clear()
    for tensor_id, state in residency.items():
        if state.resident_level not in wipe_ids:
            continue
        if _tensor_core_id(tensor_map, tensor_id) in core_set:
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
    deepest: str,
) -> None:
    tensor = tensor_map.get(event.tensor_id)
    if tensor is None:
        raise KeyError(f"unknown tensor_id in trace: {event.tensor_id}")

    state = residency[event.tensor_id]
    target = event.target_level or policy.default_access_target
    nbytes = event.bytes
    core_id = event.core_id

    resident_level = state.resident_level or state.home_level
    scratch_hit = resident_level == target and resident_level != state.home_level
    source_level = _source_level_for_access(event, state, policy, hierarchy)

    if not scratch_hit and _check_retention_expired(state, hierarchy, event.t_ns):
        state.corrupt = True
        result.retention_evictions += 1
        state.resident_level = None

    if state.corrupt:
        result.corrupt_accesses += 1
        source_level = state.home_level
        reload_from = deepest
        _charge_path(
            hierarchy,
            reload_from,
            state.home_level,
            nbytes,
            result,
            home_level=state.home_level,
            core_id=core_id,
            count_hbm=True,
        )
        state.corrupt = False
        state.resident_level = state.home_level
        _touch_home(state, event.t_ns)
        source_level = state.home_level

    if _is_direct_stram_read(event, state, policy):
        _charge_local_access(
            hierarchy, state.home_level, "read", nbytes, core_id, result
        )
        _touch_home(state, event.t_ns)
        return

    if source_level != target:
        _charge_path(
            hierarchy,
            source_level,
            target,
            nbytes,
            result,
            home_level=state.home_level,
            core_id=core_id,
            count_hbm=True,
        )
        _install_in_fast_buffer(
            hierarchy, fast_buffers, core_id, target, event.tensor_id, nbytes, pools, residency
        )
        state.resident_level = target
        if source_level == state.home_level:
            _touch_home(state, event.t_ns)
    else:
        # Same-level writes (e.g. SBUF scratch hit after SB→OUTPUT ingest) are
        # in-place touches — no separate DMA on Trainium; skip local cost.
        if event.op == "write":
            return
        _charge_local_access(
            hierarchy, target, event.op, nbytes, core_id, result
        )
        if target == state.home_level:
            _touch_home(state, event.t_ns)


def _charge_local_access(
    hierarchy: ResolvedHierarchy,
    level_id: str,
    op: str,
    nbytes: int,
    core_id: int,
    result: SimulationResult,
) -> None:
    """Line-granularity local read/write (SBUF scratch hits, direct StRAM reads)."""
    level = hierarchy.level_by_id(level_id)
    lat = access_latency_ns(level, op, nbytes)
    eng = access_energy_pJ(level, op, nbytes)
    _add_core_latency(result, core_id, lat)
    result.total_energy_pJ += eng
    _accumulate_level(result, level_id, lat, eng)


def _touch_home(state: TensorResidency, t_ns: float) -> None:
    """Refresh retention timer at the persistent home tier (not SBUF/PSUM scratch)."""
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
    home_level: str,
    core_id: int,
    count_hbm: bool,
) -> None:
    if source_id == dest_id:
        return
    for hop_from, hop_to in path_between(
        hierarchy, source_id, dest_id, home_id=home_level
    ):
        from_level = hierarchy.level_by_id(hop_from)
        to_level = hierarchy.level_by_id(hop_to)
        lat = transfer_latency_ns(hierarchy, from_level, to_level, nbytes)
        eng = transfer_energy_pJ(hierarchy, from_level, to_level, nbytes)
        _add_core_latency(result, core_id, lat)
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
    residency: dict[str, TensorResidency],
) -> None:
    level = hierarchy.level_by_id(level_id)
    if level.scope != "per_core":
        pool = pools.get(level_id)
        if pool is not None and (pool.can_fit(nbytes) or tensor_id in pool.occupants):
            pool.install(tensor_id, nbytes)
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
    """Drop one SBUF occupant (no writeback); resident returns to home."""
    if not buffer.occupants:
        return
    victim_id = next(iter(buffer.occupants))
    buffer.used_bytes -= buffer.occupants.pop(victim_id)
    state = residency.get(victim_id)
    if state is not None:
        state.resident_level = state.home_level
