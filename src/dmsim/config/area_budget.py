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
    """capacity_bytes = area_um² × density_bits_per_um² / 8"""
    return int(area_um2 * density_bits_per_um2 / 8)


def _area_for_bytes(capacity_bytes: int, density_bits_per_um2: float) -> float:
    """area_um² = capacity_bytes × 8 / density_bits_per_um²"""
    return (capacity_bytes * 8) / density_bits_per_um2


def _split_by_area_fraction(
    nominal_capacity_a: int,
    density_a: float,
    density_b: float,
    replace_area_fraction: float,
    *,
    pool_scale: int = 1,
) -> tuple[int, int, float, float]:
    """
    Iso-area trade: fraction ``replace_area_fraction`` of donor pool A's die area
    is repurposed for recipient B; capacities follow tech density.

    - ``area_pool = _area_for_bytes(nominal_capacity_a * pool_scale, density_a)``
    - ``capacity_b = _bytes_for_area(replace_area_fraction * area_pool, density_b)``
    - ``capacity_a = int(nominal_capacity_a * (1 - replace_area_fraction))``

    ``pool_scale`` sizes the shared area pool (e.g. all cores' SBUF when StRAM is
    per_chip). Returned ``capacity_a`` uses the same units as ``nominal_capacity_a``.
    """
    total_capacity_a = nominal_capacity_a * pool_scale
    total_area = _area_for_bytes(total_capacity_a, density_a)
    replaced_area = replace_area_fraction * total_area
    capacity_b = _bytes_for_area(replaced_area, density_b)
    capacity_a = int(nominal_capacity_a * (1.0 - replace_area_fraction))
    return capacity_a, capacity_b, replaced_area, total_area


def apply_area_budget(
    hierarchy: ResolvedHierarchy,
    budget: AreaBudgetConfig,
    *,
    num_cores: int,
) -> dict[str, str]:
    """
    Iso-area (constant die area) tradeoffs for differentiated hierarchies.

    Fractions of nominal SBUF/HBM **die area** move to StRAM/LtRAM; recipient
    ``capacity_bytes = area × density / 8``. Donor pools keep ``(1 - fraction)``
    of nominal byte capacity. See ``docs/AREA_BUDGET.md``.
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
        frac = budget.stram_replaces_sbuf_fraction
        pool_scale = max(1, num_cores) if stram.scope == "per_chip" else 1
        new_sbuf, stram.capacity_bytes, stram_area, _ = _split_by_area_fraction(
            nominal_sbuf,
            sbuf_density,
            stram_density,
            frac,
            pool_scale=pool_scale,
        )
        remove_per_core = nominal_sbuf - new_sbuf
        notes["stram_replaces_sbuf_fraction"] = str(frac)

        if stram.scope == "per_chip":
            remove_total = remove_per_core * max(1, num_cores)
            notes["stram_scope"] = "per_chip"
            notes["sbuf_removed_total_bytes"] = str(remove_total)
        else:
            notes["stram_scope"] = "per_core"

        sbuf.capacity_bytes = max(0, new_sbuf)
        notes["stram_area_um2"] = f"{stram_area:.4f}"
        notes["stram_capacity_bytes"] = str(stram.capacity_bytes)
        notes["sbuf_removed_per_core_bytes"] = str(remove_per_core)
        notes["sbuf_capacity_per_core_after"] = str(sbuf.capacity_bytes)

    if ltram and ltram.enabled and hbm and hbm.enabled:
        ltram_density = _density(ltram, None)
        hbm_density = _density(hbm, budget.hbm_reference_density_bits_per_um2)
        nominal_hbm = hbm.capacity_bytes
        frac = budget.ltram_replaces_hbm_fraction
        new_hbm, ltram.capacity_bytes, ltram_area, _ = _split_by_area_fraction(
            nominal_hbm,
            hbm_density,
            ltram_density,
            frac,
        )
        remove_hbm = nominal_hbm - new_hbm
        hbm.capacity_bytes = new_hbm
        notes["ltram_replaces_hbm_fraction"] = str(frac)
        notes["ltram_area_um2"] = f"{ltram_area:.4f}"
        notes["ltram_capacity_bytes"] = str(ltram.capacity_bytes)
        notes["hbm_removed_bytes"] = str(remove_hbm)
        notes["hbm_capacity_after"] = str(hbm.capacity_bytes)

    hierarchy.area_budget_notes = notes
    return notes
