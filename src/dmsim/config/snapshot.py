from __future__ import annotations

from pathlib import Path

import yaml

from dmsim.config.models import PolicyConfig, ResolvedHierarchy


def hierarchy_snapshot(
    hierarchy: ResolvedHierarchy,
    hierarchy_path: Path | None = None,
) -> dict:
    levels = []
    for level in hierarchy.levels:
        if not level.enabled:
            continue
        levels.append(
            {
                "id": level.id,
                "scope": level.scope,
                "tech": level.tech.name,
                "capacity_bytes": level.capacity_bytes,
                "retention_s": level.tech.retention_s,
            }
        )
    return {
        "hierarchy_file": str(hierarchy_path) if hierarchy_path else None,
        "hierarchy_name": hierarchy.name,
        "instance": hierarchy.instance.model_dump(),
        "area_budget": hierarchy.area_budget.model_dump(),
        "area_budget_notes": dict(hierarchy.area_budget_notes),
        "levels": levels,
        "links_GBs": dict(hierarchy.links_GBs),
        "kernel": hierarchy.kernel.model_dump(),
    }


def policy_snapshot(policy: PolicyConfig, policy_path: Path | None = None) -> dict:
    return {
        "policy_file": str(policy_path) if policy_path else None,
        "policy_name": policy.name,
        "home_level_by_category": dict(policy.home_level_by_category),
        "default_access_target": policy.default_access_target,
    }


def trace_snapshot(trace) -> dict:
    return {
        "workload": trace.metadata.workload,
        "source": trace.metadata.source,
        "neuron_core_ids": list(trace.metadata.neuron_core_ids),
        "num_tensors": len(trace.tensors),
        "num_events": len(trace.events),
    }


def load_yaml_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def build_run_report(
    *,
    hierarchy: ResolvedHierarchy,
    policy: PolicyConfig,
    trace,
    result,
    hierarchy_path: Path | None = None,
    policy_path: Path | None = None,
    label: str | None = None,
) -> dict:
    payload = {
        "label": label or hierarchy.name,
        "configuration": {
            "hierarchy": hierarchy_snapshot(hierarchy, hierarchy_path),
            "policy": policy_snapshot(policy, policy_path),
            "trace": trace_snapshot(trace),
        },
        "results": {
            **result.__dict__,
            "transfers_by_hop": dict(result.transfers_by_hop),
            "energy_by_level_pJ": dict(result.energy_by_level_pJ),
            "latency_by_level_ns": dict(result.latency_by_level_ns),
        },
    }
    if hierarchy_path and hierarchy_path.exists():
        payload["configuration"]["hierarchy_yaml"] = load_yaml_config(hierarchy_path)
    return payload


def build_compare_report(
    *,
    baseline_hierarchy,
    candidate_hierarchy,
    baseline_policy,
    candidate_policy,
    trace,
    baseline_result,
    candidate_result,
    comparison: dict,
    paths: dict[str, Path | None] | None = None,
) -> dict:
    paths = paths or {}
    return {
        "comparison": comparison,
        "baseline": build_run_report(
            hierarchy=baseline_hierarchy,
            policy=baseline_policy,
            trace=trace,
            result=baseline_result,
            hierarchy_path=paths.get("baseline_hierarchy"),
            policy_path=paths.get("baseline_policy"),
            label="baseline",
        ),
        "candidate": build_run_report(
            hierarchy=candidate_hierarchy,
            policy=candidate_policy,
            trace=trace,
            result=candidate_result,
            hierarchy_path=paths.get("candidate_hierarchy"),
            policy_path=paths.get("candidate_policy"),
            label="candidate",
        ),
    }
