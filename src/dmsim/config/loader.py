from __future__ import annotations

from pathlib import Path

import yaml

from dmsim.config.models import (
    HierarchyConfig,
    InstanceSpec,
    PolicyConfig,
    ResolvedHierarchy,
    ResolvedLevel,
    TechnologySpec,
)


def _repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    return path


def _resolve_path(repo_root: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return repo_root / path


def load_tech_spec(path: Path) -> TechnologySpec:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return TechnologySpec.model_validate(data)


def load_instance(path: Path) -> InstanceSpec:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return InstanceSpec.model_validate(data)


def load_policy(path: Path) -> PolicyConfig:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return PolicyConfig.model_validate(data)


def load_hierarchy(
    path: Path,
    *,
    repo_root: Path | None = None,
    tech_dir: Path | None = None,
) -> ResolvedHierarchy:
    root = repo_root or _repo_root(path.parent)
    tech_root = tech_dir or (root / "configs" / "tech_specs")

    with path.open() as handle:
        raw = yaml.safe_load(handle)
    config = HierarchyConfig.model_validate(raw)

    instance_path = (
        _resolve_path(root, config.instance)
        if config.instance
        else root / "configs/instances/trn2_3xlarge.yaml"
    )
    instance = load_instance(instance_path)
    hbm_bytes = int(instance.hbm_gib_per_chip * (1024**3))

    resolved: list[ResolvedLevel] = []
    enabled_index = 0
    for level in config.levels:
        if not level.enabled:
            resolved.append(
                ResolvedLevel(
                    id=level.id,
                    index=-1,
                    tech=load_tech_spec(tech_root / f"{level.tech}.yaml"),
                    scope=level.scope,
                    capacity_bytes=level.capacity_bytes or 0,
                    enabled=False,
                )
            )
            continue

        tech = load_tech_spec(tech_root / f"{level.tech}.yaml")
        capacity = level.capacity_bytes
        if level.id == "hbm":
            capacity = hbm_bytes

        if capacity is None or capacity <= 0:
            raise ValueError(f"level {level.id} requires capacity_bytes or HBM instance spec")

        resolved.append(
            ResolvedLevel(
                id=level.id,
                index=enabled_index,
                tech=tech,
                scope=level.scope,
                capacity_bytes=capacity,
                enabled=True,
            )
        )
        enabled_index += 1

    return ResolvedHierarchy(
        name=config.name,
        instance=instance,
        levels=resolved,
        links_GBs=config.links_GBs,
        kernel=config.kernel,
        tech_dir=tech_root,
        repo_root=root,
    )
