# Results guide — milestones, tables, and plots

## Milestone completion checklist

| Milestone | Llama | Qwen (M5) | Status |
|-----------|-------|-----------|--------|
| **M1 — Baseline profile** | ✅ 4.49 ms · 26.18 MB HBM | ✅ 6.52 ms · 34.83 MB HBM | Complete (embedded in all JSON) |
| **M2 — StRAM only** | ✅ 1T1C + 3T sweeps | ✅ 1T1C + 3T sweeps | Complete |
| **M3 — LtRAM only** | ✅ RRAM + FeRAM sweeps | ✅ RRAM + FeRAM sweeps | Complete |
| **M4 — Full hierarchy** | ✅ 4 configs | ✅ 4 configs | Complete |
| **M5 — Repeat on Qwen** | n/a | ✅ M2–M4 replicated | Complete |

**You officially have all simulation results for the professor milestones.**

Optional (not required): standalone `MILESTONES_1.md`, worst-case spill curves, per-run JSON for Qwen sweeps.

---

## Generate plots

```bash
pip install matplotlib   # if needed
python3 scripts/plot_milestone_results.py
```

Output directory: `results/plots/`

**Y-axis convention:** HBM and latency use **positive improvement/reduction from baseline (%)** — higher is better. Energy uses the same scale but **negative values mean higher energy** (StRAM refresh overhead); panels note this where relevant.

### Recommended slide deck (4 figures)

| Slide | File | Contents |
|-------|------|----------|
| 1 — Baseline | `m1_baseline_comparison.png` | Time, HBM, energy, hops (Llama vs Qwen) |
| 2 — M2 & M3 sweeps | `presentation_m2_m3_sweep_dashboard.png` | 2×3 grid: HBM + latency + energy for StRAM & LtRAM sweeps |
| 3 — M4 configs | `presentation_m4_dashboard.png` | HBM + latency + energy for all 4 hierarchy designs |
| 4 — Synthesis | `presentation_cross_milestone_dashboard.png` | Best headline per milestone, all three metrics |

Optional appendix: `presentation_tech_dashboard.png` (1T1C vs 3T, RRAM vs FeRAM), `m4_ltram_fills_by_config.png` (placement evidence).

Individual single-metric PNGs are still generated for detail / appendix.

---

## Tier 1 — Include in report (required figures)

| ID | Plot | File | Key finding |
|----|------|------|-------------|
| P1 | **M1 baseline comparison** | `m1_baseline_comparison.png` | Qwen has ~33% more HBM traffic and ~45% more `hbm→sbuf` hops |
| P2 | **M2 Llama vs Qwen** | `m2_stram_hbm_reduction.png` | Llama 18.7% reduction @ 75%; Qwen plateaus 4.2% @ 25% |
| P3 | **M2 StRAM tech (1T1C vs 3T)** | `m2_stram_tech_comparison.png` | 1T1C dominates; 3T weak at low fractions |
| P4 | **M3 Llama vs Qwen** | `m3_ltram_hbm_reduction.png` | Llama saturates @ 25%; Qwen improves to 75% |
| P5 | **M3 LtRAM tech (RRAM vs FeRAM)** | `m3_ltram_tech_comparison.png` | Same HBM at 25%+ Llama; RRAM lower energy |
| P6 | **M4 hierarchy configs** | `m4_hierarchy_hbm_reduction.png` | Llama best = Primary 50/25; Qwen best = Capacity 25/50 |
| P7 | **M4 ltram fills** | `m4_ltram_fills_by_config.png` | MoE needs 50% LtRAM (643 fills vs 493 @ 25%) |
| P8 | **Cross-milestone synthesis** | `cross_milestone_best_hbm_reduction.png` | M4 wins for both; gap larger on Llama |

---

## Tier 2 — Strong supplementary figures

| ID | Plot | Data source | Purpose |
|----|------|-------------|---------|
| P9 | M2 time Δ vs SBUF fraction (Llama vs Qwen) | sweep JSON, `time_pct` | Latency nearly flat; justify HBM as primary metric |
| P10 | M3 time Δ vs HBM fraction (Llama vs Qwen) | sweep JSON | Same |
| P11 | M4 time Δ grouped bars (4 configs) | M4 compare JSON | Latency improvements < 1% |
| P12 | M3 `ltram→sbuf` fills vs HBM fraction | sweep JSON | Placement evidence; Qwen plateau shift |
| P13 | M2 `stram→sbuf` fills vs SBUF fraction | sweep JSON | Qwen shows hops; Llama often local-read only |
| P14 | M4 `hbm→sbuf` migration (baseline vs configs) | M4 JSON | Traffic shifts off HBM |
| P15 | StRAM energy vs fraction (1T1C) with HBM overlay | sweep JSON | Refresh dominates; explain energy caveat |
| P16 | LtRAM energy: RRAM vs FeRAM @ 25% | sweep JSON | FeRAM higher read energy, lower write |
| P17 | Spill sensitivity: best vs worst @ 25% LtRAM | sweep JSON | Robustness of headline claims |
| P18 | Spill sensitivity: best vs worst @ 50% StRAM | sweep JSON | Same for M2 |

---

## Tier 3 — Optional / appendix

| ID | Plot | Purpose |
|----|------|---------|
| P19 | Absolute HBM traffic (MB) not just % Δ | Easier for non-experts |
| P20 | Cross-domain traffic vs HBM traffic (M4) | Shows refresh / off-chip nuance |
| P21 | Per-model M4 radar (time, HBM, energy, fills) | Single-page config comparison |
| P22 | Tensor category breakdown (baseline traces) | Motivate tiered placement |
| P23 | Iso-area capacity table figure | SBUF/StRAM/LtRAM/HBM bytes per config |
| P24 | `sbuf→hbm` writeback (constant across configs) | Shows spill path unchanged |
| P25 | 3T eDRAM energy (near-flat vs 1T1C refresh spike) | Technology tradeoff deep-dive |
| P26 | FeRAM vs RRAM energy @ equal HBM reduction | M3 technology choice |
| P27 | Qwen capacity 25/50 vs Llama capacity 25/50 | Identical on Llama, wins on Qwen |
| P28 | Near-core 50/10 comparison both models | KV-heavy vs weight-heavy design |
| P29 | Combined tiered: stacked bar (HBM + LtRAM + StRAM traffic) | Full traffic accounting |
| P30 | Speedup ratio Qwen/Llama benefit | Normalized cross-model view |

---

## Tier 1 — Required tables

| ID | Table | Source |
|----|-------|--------|
| T1 | **Experimental setup** (trace, instance, ingest, policies) | All milestone MD files |
| T2 | **M1 baselines** (time, HBM, hops, energy) | Below |
| T3 | **M2 full sweep** (10–75%, 1T1C + 3T, both models) | `MILESTONES_2_3.md`, `MILESTONE_5_QWEN.md` |
| T4 | **M3 full sweep** (10–75%, RRAM + FeRAM, both models) | Same |
| T5 | **M4 all configs** (4 designs × 2 models) | `MILESTONES_4.md`, `MILESTONE_5_QWEN.md` |
| T6 | **Headline synthesis** (best config per milestone per model) | Derived |
| T7 | **Transfer hop evidence** (`hbm→sbuf`, `ltram→sbuf`, `stram→sbuf`) | M4 + sweep JSON |

### T2 — Baseline (copy-paste)

| Model | Time | HBM traffic | `hbm→sbuf` | Energy |
|-------|------|-------------|------------|--------|
| Llama 3.2-1B | 4.49 ms | 26.18 MB | 4,070 | 17.52 TJ |
| Qwen1.5-MoE-A2.7B | 6.52 ms | 34.83 MB | 8,409 | 385.88 TJ |

### T6 — Headline synthesis

| Model | Best M2 (1T1C) | Best M3 (RRAM) | Best M4 | Winning M4 config |
|-------|----------------|----------------|---------|-------------------|
| Llama | −12.1% @ 50% SBUF | −16.6% @ 25% HBM | **−28.6%** | Primary 50/25 |
| Qwen | −4.2% @ 25%+ SBUF | −9.2% @ 75% HBM | **−8.2%** | Capacity 25/50 |

---

## Key findings to call out in text

1. **Llama benefits more from differentiation** — denser relative HBM reduction at every milestone.
2. **StRAM: 1T1C ≫ 3T** on both models; Qwen KV working set limits StRAM benefit to −4.2%.
3. **LtRAM: Llama saturates @ 25%** (all weights fit, 868 fills); **Qwen keeps improving to 75%** (MoE experts).
4. **M4 Primary 50/25 wins on Llama**; **Capacity 25/50 wins on Qwen** — design must match weight footprint.
5. **Energy is misleading for StRAM** — report HBM traffic + time; note refresh separately.
6. **`stram→sbuf` may be zero in M4** — local-read model; use M2 sweeps or refresh energy as StRAM evidence.

---

## Source files

| Content | Llama | Qwen |
|---------|-------|------|
| M2/M3 sweeps | `data/traces/memory_sweeps/consolidated_*.json` | `data/traces/memory_sweeps_qwen/consolidated_*.json` |
| M4 compares | `results/m4_llama_*_tiered.json` | `results/m5_qwen/m4_*_tiered.json` |
| Write-ups | `MILESTONES_2_3.md`, `MILESTONES_4.md` | `MILESTONE_5_QWEN.md` |
