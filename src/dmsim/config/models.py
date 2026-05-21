from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AccessSpec(BaseModel):
    read_latency_ns: float
    write_latency_ns: float
    read_energy_pJ_per_bit: float
    write_energy_pJ_per_bit: float


class InterfaceSpec(BaseModel):
    line_size_bytes: int = 64
    max_bandwidth_GBs: float


class TechnologySpec(BaseModel):
    name: str
    class_: str = Field(alias="class")
    volatile: bool = True
    retention_s: float | None = None
    cell_density_bits_per_um2: float | None = None
    access: AccessSpec
    interface: InterfaceSpec

    model_config = {"populate_by_name": True}


class InstanceSpec(BaseModel):
    name: str
    num_chips: int = 1
    cores_per_chip: int = 8
    hbm_gib_per_chip: float = 96


class LevelConfig(BaseModel):
    id: str
    enabled: bool = True
    tech: str
    scope: Literal["per_core", "per_chip", "global"] = "per_chip"
    capacity_bytes: int | None = None


class KernelConfig(BaseModel):
    wipe_levels_on_boundary: list[str] = Field(default_factory=lambda: ["psum", "sbuf"])


class AreaBudgetConfig(BaseModel):
    """Trade StRAM area against SBUF and LtRAM area against HBM at constant die area."""

    enabled: bool = True
    nominal_sbuf_bytes_per_core: int | None = None
    nominal_hbm_gib_per_chip: float | None = None
    sbuf_reference_density_bits_per_um2: float = 2.44
    hbm_reference_density_bits_per_um2: float = 1.1
    # If set (0–1), size StRAM/LtRAM from fractions of nominal SBUF/HBM area.
    stram_replaces_sbuf_fraction: float | None = None
    ltram_replaces_hbm_fraction: float | None = None


class HierarchyConfig(BaseModel):
    name: str
    description: str | None = None
    instance: str | None = None
    links_GBs: dict[str, float] = Field(default_factory=dict)
    levels: list[LevelConfig]
    kernel: KernelConfig = Field(default_factory=KernelConfig)
    area_budget: AreaBudgetConfig = Field(default_factory=AreaBudgetConfig)

    @field_validator("levels")
    @classmethod
    def at_least_one_enabled(cls, levels: list[LevelConfig]) -> list[LevelConfig]:
        if not any(level.enabled for level in levels):
            raise ValueError("hierarchy must have at least one enabled level")
        return levels


class PolicyConfig(BaseModel):
    name: str
    description: str | None = None
    home_level_by_category: dict[str, str]
    default_access_target: str = "sbuf"


class ResolvedLevel(BaseModel):
    id: str
    index: int
    tech: TechnologySpec
    scope: Literal["per_core", "per_chip", "global"]
    capacity_bytes: int
    enabled: bool = True

    @property
    def has_retention(self) -> bool:
        return self.tech.retention_s is not None and self.tech.retention_s > 0


class ResolvedHierarchy(BaseModel):
    name: str
    instance: InstanceSpec
    levels: list[ResolvedLevel]
    links_GBs: dict[str, float]
    kernel: KernelConfig
    area_budget: AreaBudgetConfig
    area_budget_notes: dict[str, str] = Field(default_factory=dict)
    tech_dir: Path
    repo_root: Path

    model_config = {"arbitrary_types_allowed": True}

    @property
    def enabled_levels(self) -> list[ResolvedLevel]:
        return [level for level in self.levels if level.enabled]

    def level_by_id(self, level_id: str) -> ResolvedLevel:
        for level in self.levels:
            if level.id == level_id:
                return level
        raise KeyError(f"unknown level id: {level_id}")

    def index_of(self, level_id: str) -> int:
        return self.level_by_id(level_id).index

    def link_bandwidth_GBs(self, from_id: str, to_id: str) -> float:
        key = f"{from_id}_{to_id}"
        rev = f"{to_id}_{from_id}"
        if key in self.links_GBs:
            return self.links_GBs[key]
        if rev in self.links_GBs:
            return self.links_GBs[rev]
        a = self.level_by_id(from_id).tech.interface.max_bandwidth_GBs
        b = self.level_by_id(to_id).tech.interface.max_bandwidth_GBs
        return min(a, b)
