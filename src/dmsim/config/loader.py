from __future__ import annotations

from pathlib import Path

import yaml

from dmsim.config.area_budget import apply_area_budget
from dmsim.config.models import (
    HierarchyConfig,
    InstanceSpec,
    InterconnectConfig,
    InterconnectDomain,
    LevelConfig,
    PolicyConfig,
    ResolvedHierarchy,
    ResolvedLevel,
    TechnologySpec,
)


def _resolve_interconnect(
    level_id: str, interconnect: InterconnectConfig
) -> InterconnectDomain:
    domain = interconnect.level_domain.get(level_id)
    if domain is None:
        raise ValueError(
            f"level {level_id!r} missing from interconnect.level_domain in hierarchy YAML"
        )
    return domain


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


def _capacity_set_by_area_budget(config: HierarchyConfig, level_id: str) -> bool:
    if not config.area_budget.enabled:
        return False
    budget = config.area_budget
    if level_id == "stram" and budget.stram_replaces_sbuf_fraction is not None:
        return True
    if level_id == "ltram" and budget.ltram_replaces_hbm_fraction is not None:
        return True
    return False


def load_hierarchy(
    path: Path,
    *,
    repo_root: Path | None = None,
    tech_dir: Path | None = None,
    num_cores: int | None = None,
) -> ResolvedHierarchy:
    root = repo_root or _repo_root(path.parent)
    tech_root = tech_dir or (root / "configs" / "tech_specs")

    with path.open() as handle:
        raw = yaml.safe_load(handle)
    config = HierarchyConfig.model_validate(raw)
    for level in config.levels:
        _resolve_interconnect(level.id, config.interconnect)

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
                    interconnect=_resolve_interconnect(level.id, config.interconnect),
                    enabled=False,
                )
            )
            continue

        tech = load_tech_spec(tech_root / f"{level.tech}.yaml")
        capacity = level.capacity_bytes
        if level.id == "hbm":
            capacity = hbm_bytes

        if capacity is None:
            raise ValueError(f"level {level.id} requires capacity_bytes or HBM instance spec")
        if capacity <= 0 and not _capacity_set_by_area_budget(config, level.id):
            raise ValueError(f"level {level.id} requires capacity_bytes or HBM instance spec")

        resolved.append(
            ResolvedLevel(
                id=level.id,
                index=enabled_index,
                tech=tech,
                scope=level.scope,
                capacity_bytes=capacity,
                interconnect=_resolve_interconnect(level.id, config.interconnect),
                enabled=True,
            )
        )
        enabled_index += 1

    cores = num_cores if num_cores is not None else instance.cores_per_chip

    hierarchy = ResolvedHierarchy(
        name=config.name,
        instance=instance,
        levels=resolved,
        interconnect=config.interconnect,
        kernel=config.kernel,
        area_budget=config.area_budget,
        num_cores=cores,
        tech_dir=tech_root,
        repo_root=root,
    )
    apply_area_budget(hierarchy, config.area_budget, num_cores=cores)
    return hierarchy
