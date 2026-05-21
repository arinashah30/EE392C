from __future__ import annotations

from dmsim.config.models import AreaBudgetConfig, ResolvedHierarchy, ResolvedLevel


def _density(level: ResolvedLevel, fallback: float | None) -> float:
    if level.tech.cell_density_bits_per_um2 is not None:
        return level.tech.cell_density_bits_per_um2
    if fallback is not None:
        return fallback
    raise ValueError(
        f"level {level.id} needs cell_density_bits_per_um2 or area_budget fallback density"
    )


def _bytes_for_area(area_um2: float, density_bits_per_um2: float) -> int:
    return int(area_um2 * density_bits_per_um2 / 8)


def _area_for_bytes(capacity_bytes: int, density_bits_per_um2: float) -> float:
    return (capacity_bytes * 8) / density_bits_per_um2


def apply_area_budget(
    hierarchy: ResolvedHierarchy,
    budget: AreaBudgetConfig,
    *,
    num_cores: int,
) -> dict[str, str]:
    """
    Constant-area tradeoffs (two modes):

    1. Fraction mode: StRAM takes ``stram_replaces_sbuf_fraction`` of SBUF area
       per core; LtRAM takes ``ltram_replaces_hbm_fraction`` of HBM area.

    2. Capacity mode: StRAM/LtRAM ``capacity_bytes`` in hierarchy YAML trade
       against SBUF (per core) and HBM using tech densities.
    """
    notes: dict[str, str] = {}
    if not budget.enabled:
        return notes

    levels = {level.id: level for level in hierarchy.levels}
    sbuf = levels.get("sbuf")
    stram = levels.get("stram")
    ltram = levels.get("ltram")
    hbm = levels.get("hbm")

    if sbuf and sbuf.enabled:
        nominal = budget.nominal_sbuf_bytes_per_core or sbuf.capacity_bytes
        sbuf.capacity_bytes = nominal
        notes["sbuf_nominal_per_core"] = str(nominal)

    if hbm and hbm.enabled:
        nominal_hbm = int(
            (budget.nominal_hbm_gib_per_chip or hierarchy.instance.hbm_gib_per_chip)
            * (1024**3)
        )
        hbm.capacity_bytes = nominal_hbm
        notes["hbm_nominal"] = str(nominal_hbm)

    if stram and stram.enabled and sbuf and sbuf.enabled:
        sbuf_density = _density(sbuf, budget.sbuf_reference_density_bits_per_um2)
        stram_density = _density(stram, None)
        nominal_sbuf = sbuf.capacity_bytes

        if budget.stram_replaces_sbuf_fraction is not None:
            frac = budget.stram_replaces_sbuf_fraction
            stram.capacity_bytes = int(nominal_sbuf * frac)
            remove_per_core = stram.capacity_bytes
            stram_area = _area_for_bytes(stram.capacity_bytes, stram_density)
            notes["mode"] = "capacity_fraction"
            notes["stram_replaces_sbuf_fraction"] = str(frac)
        else:
            stram_area = _area_for_bytes(stram.capacity_bytes, stram_density)
            remove_per_core = _bytes_for_area(stram_area, sbuf_density)
            notes["mode"] = "capacity"

        if stram.scope == "per_chip":
            remove_total = remove_per_core
            remove_per_core = remove_total // max(1, num_cores)
            notes["stram_scope"] = "per_chip"
            notes["sbuf_removed_total_bytes"] = str(remove_total)
        else:
            notes["stram_scope"] = "per_core"

        new_sbuf = max(0, nominal_sbuf - remove_per_core)
        notes["stram_area_um2_per_core"] = f"{stram_area:.4f}"
        notes["stram_capacity_bytes_per_core"] = str(stram.capacity_bytes)
        notes["sbuf_removed_per_core_bytes"] = str(remove_per_core)
        notes["sbuf_capacity_per_core_after"] = str(new_sbuf)
        sbuf.capacity_bytes = new_sbuf

    if ltram and ltram.enabled and hbm and hbm.enabled:
        ltram_density = _density(ltram, None)
        hbm_density = _density(hbm, budget.hbm_reference_density_bits_per_um2)
        nominal_hbm = hbm.capacity_bytes

        if budget.ltram_replaces_hbm_fraction is not None:
            frac = budget.ltram_replaces_hbm_fraction
            ltram.capacity_bytes = int(nominal_hbm * frac)
            remove_hbm = ltram.capacity_bytes
            ltram_area = _area_for_bytes(ltram.capacity_bytes, ltram_density)
            new_hbm = max(0, nominal_hbm - remove_hbm)
            notes["ltram_replaces_hbm_fraction"] = str(frac)
        else:
            ltram_area = _area_for_bytes(ltram.capacity_bytes, ltram_density)
            remove_hbm = _bytes_for_area(ltram_area, hbm_density)
            new_hbm = max(0, nominal_hbm - remove_hbm)

        notes["ltram_area_um2"] = f"{ltram_area:.4f}"
        notes["ltram_capacity_bytes"] = str(ltram.capacity_bytes)
        notes["hbm_removed_bytes"] = str(remove_hbm)
        notes["hbm_capacity_after"] = str(new_hbm)
        hbm.capacity_bytes = new_hbm

    hierarchy.area_budget_notes = notes
    return notes
