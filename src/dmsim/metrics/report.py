from __future__ import annotations

import json
from pathlib import Path

from dmsim.sim.engine import SimulationResult


def format_report(result: SimulationResult) -> str:
    lines = [
        f"=== {result.hierarchy_name} / {result.policy_name} ===",
        f"workload: {result.trace_workload}",
        f"total_time_ns: {result.total_time_ns:,.0f}  (worst core"
        + (f" nc{result.worst_core_id}" if result.worst_core_id is not None else "")
        + ")",
        f"total_energy_pJ: {result.total_energy_pJ:,.0f}",
        f"refresh_energy_pJ: {result.refresh_energy_pJ:,.0f}",
        f"hbm_read_bytes: {result.hbm_read_bytes:,}",
        f"hbm_write_bytes: {result.hbm_write_bytes:,}",
        f"hbm_traffic_bytes: {result.hbm_traffic_bytes:,}",
        f"kernel_wipes: {result.kernel_wipes}",
        "",
        "transfers_by_hop:",
    ]
    for hop, count in sorted(result.transfers_by_hop.items()):
        lines.append(f"  {hop}: {count}")
    if result.time_by_core_ns:
        lines.append("")
        lines.append("time_by_core_ns:")
        for core_id in sorted(result.time_by_core_ns):
            lines.append(f"  nc{core_id}: {result.time_by_core_ns[core_id]:,.0f}")
    lines.append("")
    lines.append("energy_by_level_pJ:")
    for level_id, energy in sorted(result.energy_by_level_pJ.items()):
        lines.append(f"  {level_id}: {energy:,.0f}")
    return "\n".join(lines)


def compare_results(
    baseline: SimulationResult,
    candidate: SimulationResult,
) -> dict:
    def delta(a: float, b: float) -> float:
        if a == 0:
            return 0.0 if b == 0 else float("inf")
        return (b - a) / a * 100.0

    return {
        "baseline": baseline.hierarchy_name,
        "candidate": candidate.hierarchy_name,
        "time_ns": {
            "baseline": baseline.total_time_ns,
            "candidate": candidate.total_time_ns,
            "pct_change": delta(baseline.total_time_ns, candidate.total_time_ns),
        },
        "energy_pJ": {
            "baseline": baseline.total_energy_pJ,
            "candidate": candidate.total_energy_pJ,
            "pct_change": delta(baseline.total_energy_pJ, candidate.total_energy_pJ),
        },
        "hbm_traffic_bytes": {
            "baseline": baseline.hbm_traffic_bytes,
            "candidate": candidate.hbm_traffic_bytes,
            "pct_change": delta(baseline.hbm_traffic_bytes, candidate.hbm_traffic_bytes),
        },
    }


def write_report_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
