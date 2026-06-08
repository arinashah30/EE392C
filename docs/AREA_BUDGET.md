# Iso-area (constant die area) budget

Differentiated hierarchies (`trainium2_diff_mem*.yaml`) model **fixed on-die area** for the nominal SBUF pool (per NeuronCore) and nominal HBM pool (per chip). A configurable **fraction of that area** is repurposed for StRAM or LtRAM; **byte capacities change** with each technology’s `cell_density_bits_per_um2`.

Implementation: [`src/dmsim/config/area_budget.py`](../src/dmsim/config/area_budget.py). Tests: [`tests/test_area_budget.py`](../tests/test_area_budget.py).

---

## Area ↔ capacity

For a tier with density `ρ` (bits/µm²):

```text
area_um² = capacity_bytes × 8 / ρ
capacity_bytes = area_um² × ρ / 8
```

(`_area_for_bytes` / `_bytes_for_area` in `area_budget.py`.)

---

## Trade (one donor A → recipient B)

Given nominal capacity `C_A` for pool A, donor density `ρ_A`, recipient density `ρ_B`, and area fraction `f` ∈ [0, 1]:

1. **Die area of the nominal pool** (optionally scaled by `pool_scale`, e.g. all cores’ SBUF when StRAM is per-chip):
   ```text
   area_pool = C_A × pool_scale × 8 / ρ_A
   ```
2. **Area moved to B:**
   ```text
   area_B = f × area_pool
   ```
3. **Recipient capacity (iso-area → density sets bytes):**
   ```text
   capacity_B = area_B × ρ_B / 8
   ```
4. **Donor capacity after trade (byte accounting):**
   ```text
   capacity_A = (1 − f) × C_A
   ```

StRAM/LtRAM YAML `capacity_bytes: 0` is a placeholder; `apply_area_budget()` fills `capacity_B` from the formulas above.

**Not** iso-area: assigning `capacity_B = f × C_A` (byte fraction only). That keeps die area only when `ρ_B = ρ_A`.

---

## Defaults in this repo (50% SBUF / 25% HBM)

| Trade | Fraction `f` | Donor after | Recipient capacity |
|--------|----------------|-------------|---------------------|
| SBUF → StRAM (per core) | `stram_replaces_sbuf_fraction: 0.5` | 50% of nominal SBUF bytes/core | `0.5 × area(SBUF_nominal) × ρ_StRAM / 8` |
| HBM → LtRAM (per chip) | `ltram_replaces_hbm_fraction: 0.25` | 75% of nominal HBM bytes | `0.25 × area(HBM_nominal) × ρ_LtRAM / 8` |

Example (nominal SBUF 29 360 128 B/core, `ρ_SBUF=2.44`, `ρ_StRAM=11`):

```text
area_pool ≈ 9.63×10⁷ µm²/core
area_StRAM = 0.5 × area_pool
capacity_StRAM ≈ 6.62×10⁷ B/core   (eDRAM denser than SRAM → more bytes than 50% of SBUF bytes)
capacity_SBUF = 14 680 064 B/core
```

Densities come from `configs/tech_specs/` (`cell_density_bits_per_um2`), with optional fallbacks `sbuf_reference_density_bits_per_um2` / `hbm_reference_density_bits_per_um2` in hierarchy YAML when a level omits density.

---

## Scope

| Level | Scope | Area pool |
|-------|--------|-----------|
| SBUF / StRAM | `per_core` (Trainium2) | Each core’s nominal SBUF area independently |
| HBM / LtRAM | `per_chip` | Whole chip nominal HBM area |

StRAM is **not** carved from chip-wide SBUF once and split across cores; each core trades against its own SBUF die budget.

---

## Compare output

`dmsim compare` prints `=== iso-area budget (constant die area) ===` with resolved capacities and note fields (`stram_area_um2`, `ltram_capacity_bytes`, etc.).

Regression expectations for the Llama 4-core trace are in [`tests/test_llama_regression.py`](../tests/test_llama_regression.py).

---

## Related docs

- Placement uses resolved capacities: [`PLACEMENT_AND_EVICTION.md`](PLACEMENT_AND_EVICTION.md)
- Bandwidth is **not** scaled by area (only pool sizes): [`cost-model/README.md`](cost-model/README.md)
