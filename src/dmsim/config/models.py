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
    # Periodic refresh for volatile tiers (e.g. HBM 64 ms, eDRAM ~40 µs).
    refresh_interval_s: float | None = None
    refresh_energy_pJ_per_bit: float | None = None
    cell_density_bits_per_um2: float | None = None
    access: AccessSpec
    interface: InterfaceSpec

    model_config = {"populate_by_name": True}


class InstanceSpec(BaseModel):
    name: str
    num_chips: int = 1
    cores_per_chip: int = 8
    hbm_gib_per_chip: float = 96
    # Trainium2 NeuronCore DMA: aggregate delivery bandwidth caps memory transfers.
    dma_bytes_per_ns_per_engine: float = 23.0
    dma_engines_per_neuron_core: int = 16
    dma_cap_transfer_bandwidth: bool = True


InterconnectDomain = Literal["on_chip", "off_chip"]


class InterconnectConfig(BaseModel):
    """Per-core transfer bandwidth and level domain (see hierarchy YAML ``interconnect:``)."""

    dma_bandwidth_GBs: float
    on_chip_bandwidth_GBs: float
    level_domain: dict[str, InterconnectDomain]


class LevelConfig(BaseModel):
    id: str
    enabled: bool = True
    tech: str
    scope: Literal["per_core", "per_chip", "global"] = "per_chip"
    capacity_bytes: int | None = None
    # Override tech ``refresh_interval_s`` for this level (simulator only).
    refresh_interval_s: float | None = None


class KernelConfig(BaseModel):
    """Levels cleared on traced ``kernel_end`` (fast-buffer occupants + resident reset).

    Default: ``[psum, sbuf]`` only. Near-memory homes (``stram``, ``ltram``) and ``hbm``
    persist unless explicitly listed. Add a level id here to wipe it each kernel boundary.

    Wipe scope follows ``KernelBoundaryEvent.core_id``: one NeuronCore when set, or every
    core that already has fast-buffer state when ``core_id`` is omitted (single-core ingest).
    """

    wipe_levels_on_boundary: list[str] = Field(default_factory=lambda: ["psum", "sbuf"])


class AreaBudgetConfig(BaseModel):
    """Trade StRAM area against SBUF and LtRAM area against HBM at constant die area."""

    enabled: bool = True
    nominal_sbuf_bytes_per_core: int | None = None
    nominal_hbm_gib_per_chip: float | None = None
    sbuf_reference_density_bits_per_um2: float = 2.44
    hbm_reference_density_bits_per_um2: float = 1.1
    # Fraction (0–1) of nominal SBUF/HBM die area traded to StRAM/LtRAM.
    stram_replaces_sbuf_fraction: float = 0.0
    ltram_replaces_hbm_fraction: float = 0.0


class HierarchyConfig(BaseModel):
    name: str
    description: str | None = None
    instance: str | None = None
    interconnect: InterconnectConfig
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
    # When a level is disabled or over capacity, spill/fallback to this target.
    # Unlisted levels default to hbm.
    fallback_by_level: dict[str, str] = Field(default_factory=dict)
    # best_case: spill least-accessed tensors first; worst_case: most-accessed first.
    spill_victim_order: Literal["best_case", "worst_case"] = "best_case"

    def fallback_for(self, level_id: str) -> str:
        return self.fallback_by_level.get(level_id, "hbm")


class ResolvedLevel(BaseModel):
    id: str
    index: int
    tech: TechnologySpec
    scope: Literal["per_core", "per_chip", "global"]
    capacity_bytes: int
    interconnect: InterconnectDomain
    enabled: bool = True
    refresh_interval_s: float | None = None

    @property
    def effective_refresh_interval_s(self) -> float | None:
        if self.refresh_interval_s is not None:
            if self.refresh_interval_s <= 0:
                return None
            return self.refresh_interval_s
        return self.tech.refresh_interval_s

class ResolvedHierarchy(BaseModel):
    name: str
    instance: InstanceSpec
    levels: list[ResolvedLevel]
    interconnect: InterconnectConfig
    kernel: KernelConfig
    area_budget: AreaBudgetConfig
    num_cores: int = 1
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

    @property
    def dma_bandwidth_GBs(self) -> float:
        return self.interconnect.dma_bandwidth_GBs

    @property
    def on_chip_bandwidth_GBs(self) -> float:
        return self.interconnect.on_chip_bandwidth_GBs

    def link_bandwidth_GBs(self, from_id: str, to_id: str) -> float:
        """Per-core GB/s for one hop: on-chip if both on-chip, else DMA."""
        from_level = self.level_by_id(from_id)
        to_level = self.level_by_id(to_id)
        if (
            from_level.interconnect == "on_chip"
            and to_level.interconnect == "on_chip"
        ):
            return self.on_chip_bandwidth_GBs
        return self.dma_bandwidth_GBs
