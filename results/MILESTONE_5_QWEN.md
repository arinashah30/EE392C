# Milestone 5 — Qwen1.5-MoE-A2.7B decode (M2, M3, M4 replicates)

**Trace:** `data/traces/qwen1_5_moe_trn2_3_4core_min1_no_unknown.json`  
**Instance:** trn2.3xlarge (4 active NeuronCores)  
**Spill order:** `best_case` (residency-aware replay)  
**Baseline hierarchy:** `trainium2_baseline.yaml` + `baseline_hbm.yaml`

## Baseline reference (Qwen)


| Metric                 | Value     |
| ---------------------- | --------- |
| Time (worst core)      | 6.52 ms   |
| HBM traffic            | 34.83 MB  |
| `hbm→sbuf` hops        | 8,409     |
| `sbuf→hbm` (writeback) | 186,348   |
| Total energy           | 385.88 TJ |


*Llama baseline for comparison:* 4.49 ms · 26.18 MB HBM · 4,070 `hbm→sbuf` · 17.52 TJ

---

## Milestone 3 — LtRAM (RRAM + FeRAM), by-tensor-type placement

**Policy:** `decode_ltram_only.yaml` — weights → LtRAM, KV/other → HBM  
**Iso-area trade:** X% of nominal HBM die area → LtRAM (StRAM disabled)

### Verdict: **Meets expectations — MoE weights need more LtRAM than Llama**

Unlike Llama (plateau at 25%), Qwen MoE weights continue to benefit through **75% HBM→LtRAM** trade. At 25% only **493** `ltram→sbuf` fills vs **868** on Llama; at 75% RRAM reaches **753** fills.

### Table M3-A — LtRAM RRAM (`decode_ltram_only`, iso-area from HBM)


| HBM area → LtRAM | Time Δ     | HBM traffic Δ | Energy Δ   | `hbm→sbuf` | `ltram→sbuf` |
| ---------------- | ---------- | ------------- | ---------- | ---------- | ------------ |
| **Baseline**     | —          | —             | —          | 8,409      | 0            |
| 10%              | −0.09%     | −1.7%         | −2.2%      | 8,199      | 210          |
| **25%**          | **−0.26%** | **−6.5%**     | **−5.8%**  | 7,916      | **493**      |
| 50%              | −0.31%     | −8.2%         | −11.5%     | 7,766      | 643          |
| **75%**          | **−0.38%** | **−9.2%**     | **−17.2%** | 7,656      | **753**      |


*Source:* `data/traces/memory_sweeps_qwen/consolidated_ltram_rram.json`

### Table M3-B — LtRAM FeRAM (`decode_ltram_only`, iso-area from HBM)


| HBM area → LtRAM | Time Δ     | HBM traffic Δ | Energy Δ  | `hbm→sbuf` | `ltram→sbuf` |
| ---------------- | ---------- | ------------- | --------- | ---------- | ------------ |
| **Baseline**     | —          | —             | —         | 8,409      | 0            |
| 10%              | −0.05%     | −1.0%         | −1.1%     | 8,247      | 162          |
| 25%              | −0.12%     | −2.5%         | −2.5%     | 8,156      | 253          |
| 50%              | −0.26%     | −6.4%         | −5.5%     | 7,921      | 488          |
| **75%**          | **−0.27%** | **−7.1%**     | **−8.2%** | 7,866      | **543**      |


*Source:* `data/traces/memory_sweeps_qwen/consolidated_ltram_feram.json`

### Table M3-C — Technology comparison at 25% HBM → LtRAM


| Technology | HBM traffic          | Time        | Energy vs baseline | `ltram→sbuf` |
| ---------- | -------------------- | ----------- | ------------------ | ------------ |
| Baseline   | 34.83 MB             | 6.52 ms     | —                  | 0            |
| **RRAM**   | **32.57 MB (−6.5%)** | **6.51 ms** | **−5.8%**          | **493**      |
| **FeRAM**  | **33.97 MB (−2.5%)** | **6.51 ms** | **−2.5%**          | **253**      |


### Table M3-D — Qwen vs Llama at 25% HBM → LtRAM (RRAM)


| Model             | HBM traffic Δ | `ltram→sbuf` | Plateau fraction           |
| ----------------- | ------------- | ------------ | -------------------------- |
| Llama 3.2-1B      | **−16.6%**    | 868          | 25%                        |
| Qwen1.5-MoE-A2.7B | **−6.5%**     | 493          | **>75%** (still improving) |


---

## Milestone 2 — StRAM (1T1C + 3T eDRAM), by-tensor-type placement

**Policy:** `decode_stram_only.yaml` — KV cache → StRAM, weights/other → HBM  
**Iso-area trade:** X% of nominal SBUF die area → StRAM per core (LtRAM disabled)

### Verdict: **Partially meets expectations — smaller benefit than Llama**

Qwen decode has a larger working set and more HBM traffic overall. StRAM reduces HBM traffic modestly (**−4.2%** best case for 1T1C at 25%+) vs **−12.1%** on Llama at 50%. `stram→sbuf` hops appear but plateau at **201** (1T1C) / **201** (3T @ 75%).

### Table M2-A — StRAM 1T1C eDRAM (`decode_stram_only`, iso-area from SBUF)


| SBUF area → StRAM | Time Δ     | HBM traffic Δ | Energy Δ  | `hbm→sbuf` | `stram→sbuf` |
| ----------------- | ---------- | ------------- | --------- | ---------- | ------------ |
| **Baseline**      | —          | —             | —         | 8,409      | 0            |
| 10%               | −0.14%     | −2.0%         | +3.7%     | 8,120      | 101          |
| **25%**           | **−0.29%** | **−4.2%**     | **+7.6%** | 7,807      | **201**      |
| 50%               | −0.29%     | −4.2%         | +7.6%     | 7,807      | 201          |
| 75%               | −0.29%     | −4.2%         | +7.6%     | 7,807      | 201          |


*Source:* `data/traces/memory_sweeps_qwen/consolidated_stram_edram_1t1c.json`

### Table M2-B — StRAM 3T eDRAM (`decode_stram_only`, iso-area from SBUF)


| SBUF area → StRAM | Time Δ     | HBM traffic Δ | Energy Δ   | `hbm→sbuf` | `stram→sbuf` |
| ----------------- | ---------- | ------------- | ---------- | ---------- | ------------ |
| **Baseline**      | —          | —             | —          | 8,409      | 0            |
| 10%               | −0.03%     | −0.7%         | +0.01%     | 8,312      | 36           |
| 25%               | −0.10%     | −1.8%         | +0.02%     | 8,143      | 93           |
| 50%               | −0.19%     | −3.6%         | +0.04%     | 7,892      | 181          |
| **75%**           | **−0.23%** | **−4.2%**     | **+0.04%** | 7,807      | **201**      |


*Source:* `data/traces/memory_sweeps_qwen/consolidated_stram_edram_3t.json`

### Table M2-C — Technology comparison at 50% SBUF → StRAM


| Technology     | HBM traffic          | Time        | `stram→sbuf` |
| -------------- | -------------------- | ----------- | ------------ |
| Baseline       | 34.83 MB             | 6.52 ms     | 0            |
| **1T1C eDRAM** | **33.37 MB (−4.2%)** | **6.50 ms** | 201          |
| **3T eDRAM**   | **33.58 MB (−3.6%)** | **6.51 ms** | 181          |


---

## Milestone 4 — Full hierarchy (StRAM + LtRAM), tiered placement

**Policy:** `decode_tiered.yaml` (weights→LtRAM, KV/hidden/activation→StRAM, other→HBM)  
**Technologies:** StRAM `edram_1t1c`, LtRAM `rram`

### Verdict: **Capacity-heavy wins on MoE — unlike Llama**

On Llama, primary **50/25** was best (−28.6% HBM). On Qwen MoE, **capacity-heavy 25/50** is best (**−8.2%** HBM) because weights spill beyond 25% LtRAM. Primary **50/25** and balanced **25/25** are identical (−6.5%). StRAM activity shows via refresh energy; `**stram→sbuf` does not appear** in M4 hop counts (local-read model for homed StRAM resident tensors).

### Table M4-A — Full hierarchy design points


| Design             | StRAM ← SBUF | LtRAM ← HBM | StRAM / core | LtRAM / chip |
| ------------------ | ------------ | ----------- | ------------ | ------------ |
| **Primary**        | 50%          | 25%         | 60 MiB       | 1.56 GiB     |
| Balanced           | 25%          | 25%         | 30 MiB       | 1.56 GiB     |
| Near-core heavy    | 50%          | 10%         | 60 MiB       | 0.62 GiB     |
| **Capacity heavy** | 25%          | **50%**     | 30 MiB       | **3.12 GiB** |


### Table M4-B — Results vs baseline


| Design               | StRAM / LtRAM trade | Time        | Time Δ     | HBM traffic  | HBM Δ      | Total energy  | Energy Δ   |
| -------------------- | ------------------- | ----------- | ---------- | ------------ | ---------- | ------------- | ---------- |
| Baseline             | —                   | 6.52 ms     | —          | 34.83 MB     | —          | 385.88 TJ     | —          |
| Primary (50/25)      | 50% / 25%           | 6.51 ms     | −0.27%     | 32.54 MB     | −6.55%     | 363.65 TJ     | −5.8%      |
| Balanced (25/25)     | 25% / 25%           | 6.51 ms     | −0.27%     | 32.54 MB     | −6.55%     | 363.65 TJ     | −5.8%      |
| Near-core (50/10)    | 50% / 10%           | 6.52 ms     | −0.09%     | 34.20 MB     | −1.79%     | 377.57 TJ     | −2.2%      |
| **Capacity (25/50)** | 25% / **50%**       | **6.50 ms** | **−0.31%** | **31.96 MB** | **−8.22%** | **341.56 TJ** | **−11.5%** |


### Table M4-C — Transfer hops (tiered placement evidence)


| Design               | `hbm→sbuf` | `ltram→sbuf` | `stram→sbuf` | `sbuf→hbm` |
| -------------------- | ---------- | ------------ | ------------ | ---------- |
| Baseline             | 8,409      | 0            | 0            | 186,348    |
| Primary (50/25)      | 7,904      | 493          | 0            | 186,348    |
| Balanced (25/25)     | 7,904      | 493          | 0            | 186,348    |
| Near-core (50/10)    | 8,187      | 210          | 0            | 186,348    |
| **Capacity (25/50)** | **7,754**  | **643**      | 0            | 186,348    |


### Table M4-D — Qwen vs Llama M4 headline (primary vs capacity-heavy)


| Model             | Best M4 config     | HBM traffic Δ | `ltram→sbuf` | `stram→sbuf` |
| ----------------- | ------------------ | ------------- | ------------ | ------------ |
| Llama 3.2-1B      | Primary 50/25      | **−28.6%**    | 868          | 140          |
| Qwen1.5-MoE-A2.7B | **Capacity 25/50** | **−8.2%**     | 643          | 0*           |


*StRAM active via local reads + refresh; see M2 caveats.*

---

## Summary: milestone completion (Qwen)


| Milestone               | Status         | Headline result (best_case)                                                |
| ----------------------- | -------------- | -------------------------------------------------------------------------- |
| **M2 — StRAM**          | ✅ with caveats | 1T1C @ 25%+ SBUF: **−4.2% HBM**, 201 `stram→sbuf`; much smaller than Llama |
| **M3 — LtRAM**          | ✅ complete     | RRAM @ 75% HBM: **−9.2% HBM**, **753 LtRAM fills**; 25% only −6.5%         |
| **M4 — Full hierarchy** | ✅ complete     | **Capacity 25/50:** **−8.2% HBM**, 643 `ltram→sbuf`; beats primary 50/25   |


---

## Output files


| Milestone      | Consolidated / results JSON                                         |
| -------------- | ------------------------------------------------------------------- |
| M2 — 1T1C      | `data/traces/memory_sweeps_qwen/consolidated_stram_edram_1t1c.json` |
| M2 — 3T        | `data/traces/memory_sweeps_qwen/consolidated_stram_edram_3t.json`   |
| M3 — RRAM      | `data/traces/memory_sweeps_qwen/consolidated_ltram_rram.json`       |
| M3 — FeRAM     | `data/traces/memory_sweeps_qwen/consolidated_ltram_feram.json`      |
| M4 — Primary   | `results/m5_qwen/m4_50sbuf_25hbm_tiered.json`                       |
| M4 — Balanced  | `results/m5_qwen/m4_25sbuf_25hbm_tiered.json`                       |
| M4 — Near-core | `results/m5_qwen/m4_50sbuf_10hbm_tiered.json`                       |
| M4 — Capacity  | `results/m5_qwen/m4_25sbuf_50hbm_tiered.json`                       |


---

## Report paragraph (copy-paste)

**Milestone 5 (Qwen1.5-MoE-A2.7B):** On 4-core decode, baseline HBM traffic is **34.8 MB** (vs 26.2 MB Llama). LtRAM-only placement (`decode_ltram_only`, RRAM) reduces HBM traffic by up to **9.2%** at 75% iso-area HBM→LtRAM trade, with benefit continuing past 25% unlike Llama (MoE expert weights). StRAM-only (`decode_stram_only`, 1T1C) achieves **−4.2%** HBM at 25%+ SBUF trade — smaller than Llama’s −12% due to larger KV/working set. Full hierarchy with `decode_tiered` is best with **capacity-heavy 25/50** (**−8.2%** HBM, **643 ltram→sbuf**), outperforming primary 50/25 (−6.5%) because 25% LtRAM is insufficient for all MoE weights.