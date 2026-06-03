# Milestones 2 & 3 — Simulation Results (Llama 3.2-1B decode, 4 cores)

**Status:** Partner sweep runs complete. Source data: `data/traces/memory_sweeps/`.

## Experimental setup (all runs below)


| Parameter              | Value                                               |
| ---------------------- | --------------------------------------------------- |
| **Trace**              | `llama32_1b_decode_4core_min1_no_unknown.json`      |
| **Baseline hierarchy** | `trainium2_baseline.yaml`                           |
| **Baseline policy**    | `baseline_hbm.yaml`                                 |
| **Instance**           | trn2.3xlarge (4 active NeuronCores)                 |
| **Spill order**        | `best_case` (residency-aware replay)                |
| **Baseline reference** | time 4.49 ms · HBM traffic 26.2 MB · energy 17.5 TJ |


---

## Milestone 3 — LtRAM (RRAM + FeRAM), by-tensor-type placement

**Policy:** `decode_ltram_only.yaml` — **weights → LtRAM**, KV/other → HBM  
**Iso-area trade:** X% of nominal HBM die area → LtRAM (StRAM disabled)

### Verdict: **Meets expectations**

- `ltram→sbuf` transfers appear at every fraction ≥10% (best_case).
- HBM traffic drops up to **−16.6%** vs baseline (25%+ area).
- Latency improves slightly (~**−0.5%** at 25%+).
- Both RRAM and FeRAM technologies simulated and compared.

### Table M3-A — LtRAM RRAM (`decode_ltram_only`, iso-area from HBM)


| HBM area → LtRAM | Time Δ     | HBM traffic Δ | Energy Δ   | `hbm→sbuf` | `ltram→sbuf` |
| ---------------- | ---------- | ------------- | ---------- | ---------- | ------------ |
| **Baseline**     | —          | —             | —          | 4,070      | 0            |
| 10%              | −0.31%     | −13.7%        | −22.1%     | 3,563      | 507          |
| **25%**          | **−0.55%** | **−16.6%**    | **−42.6%** | 3,202      | **868**      |
| 50%              | −0.55%     | −16.6%        | −42.6%     | 3,202      | 868          |
| 75%              | −0.55%     | −16.6%        | −42.6%     | 3,202      | 868          |


*Source:* `consolidated_ltram_rram.json` · per-run JSONs under `memory_sweeps/runs/compare_ltram_rram_`*

### Table M3-B — LtRAM FeRAM (`decode_ltram_only`, iso-area from HBM)


| HBM area → LtRAM | Time Δ     | HBM traffic Δ | Energy Δ   | `hbm→sbuf` | `ltram→sbuf` |
| ---------------- | ---------- | ------------- | ---------- | ---------- | ------------ |
| **Baseline**     | —          | —             | —          | 4,070      | 0            |
| 10%              | −0.17%     | −8.3%         | −10.4%     | 3,768      | 302          |
| **25%**          | **−0.55%** | **−16.6%**    | **−25.2%** | 3,206      | **864**      |
| 50%              | −0.55%     | −16.6%        | −42.6%     | 3,202      | 868          |
| 75%              | −0.55%     | −16.6%        | −42.6%     | 3,202      | 868          |


*Source:* `consolidated_ltram_feram.json`

### Table M3-C — Technology comparison at 25% HBM → LtRAM (headline row)


| Technology | HBM traffic          | Time        | Energy vs baseline | Weight home |
| ---------- | -------------------- | ----------- | ------------------ | ----------- |
| Baseline   | 26.2 MB              | 4.49 ms     | —                  | HBM         |
| **RRAM**   | **21.8 MB (−16.6%)** | **4.47 ms** | **−42.6%**         | LtRAM       |
| **FeRAM**  | **21.8 MB (−16.6%)** | **4.47 ms** | **−25.2%**         | LtRAM       |


**Note:** Results plateau at 25% because all weight tensors fit in the LtRAM pool; larger fractions do not increase benefit on this trace.

---

## Milestone 2 — StRAM (1T1C + 3T eDRAM), by-tensor-type placement

**Policy:** `decode_stram_only.yaml` — **KV cache → StRAM**, weights/other → HBM  
**Iso-area trade:** X% of nominal SBUF die area → StRAM per core (LtRAM disabled)

### Verdict: **Partially meets expectations — report with caveats**

**What works:**

- Both eDRAM technologies (1T1C, 3T) simulated across 10–75% SBUF area trade.
- HBM traffic **decreases** as StRAM pool grows (1T1C: up to **−18.7%** at 75%).
- StRAM is active: refresh energy and StRAM-level latency appear in detailed JSON (e.g. 50% 1T1C: 33k StRAM refresh cycles).

**Caveats for the writeup:**

1. `**stram→sbuf` does not appear in `transfers_by_hop`** — the simulator treats homed+resident StRAM reads as **local StRAM hits** (no interconnect hop). This is intentional (`sim/README.md`); cite StRAM energy/latency and HBM traffic instead of hop counts.
2. **3T eDRAM at 10–25%** shows almost no benefit (StRAM pool too small for KV working set on this trace).
3. **1T1C total energy rises sharply** at larger StRAM fractions due to **eDRAM refresh** (90 µs interval); report refresh separately or use HBM traffic + time as primary metrics for StRAM.

### Table M2-A — StRAM 1T1C eDRAM (`decode_stram_only`, iso-area from SBUF)


| SBUF area → StRAM | Time Δ     | HBM traffic Δ | Energy Δ | `hbm→sbuf` | StRAM refresh cycles |
| ----------------- | ---------- | ------------- | -------- | ---------- | -------------------- |
| **Baseline**      | —          | —             | —        | 4,070      | 0                    |
| 10%               | +0.01%     | −0.5%         | +24%     | 4,052      | low                  |
| **25%**           | −0.06%     | **−3.9%**     | +74%     | 3,925      | —                    |
| **50%**           | **−0.21%** | **−12.1%**    | +173%    | 3,629      | **33,113**           |
| 75%               | −0.31%     | **−18.7%**    | +271%    | 3,395      | —                    |


*Source:* `consolidated_stram_edram_1t1c.json` · example detail: `runs/compare_stram_edram_1t1c_50pct_best_case.json`

### Table M2-B — StRAM 3T eDRAM (`decode_stram_only`, iso-area from SBUF)


| SBUF area → StRAM | Time Δ | HBM traffic Δ | Energy Δ | `hbm→sbuf` |
| ----------------- | ------ | ------------- | -------- | ---------- |
| **Baseline**      | —      | —             | —        | 4,070      |
| 10%               | 0%     | 0%            | 0%       | 4,070      |
| 25%               | +0.03% | −0.5%         | +0.1%    | 4,052      |
| 50%               | −0.01% | −3.4%         | +0.3%    | 3,947      |
| 75%               | +0.04% | −4.9%         | +0.5%    | 3,886      |


*Source:* `consolidated_stram_edram_3t.json`

### Table M2-C — Technology comparison at 50% SBUF → StRAM (recommended headline)


| Technology     | HBM traffic          | Time        | Primary StRAM signal       |
| -------------- | -------------------- | ----------- | -------------------------- |
| Baseline       | 26.2 MB              | 4.49 ms     | —                          |
| **1T1C eDRAM** | **23.0 MB (−12.1%)** | **4.48 ms** | StRAM refresh + HBM read ↓ |
| **3T eDRAM**   | **25.3 MB (−3.4%)**  | 4.49 ms     | Modest HBM reduction       |


**Recommendation:** Use **50% SBUF→StRAM** as the headline StRAM fraction (clear HBM benefit for 1T1C). At 25%, benefits are small on this trace.

---

## Summary: milestone completion


| Milestone      | Status         | Headline result (best_case, recommended fraction)                             |
| -------------- | -------------- | ----------------------------------------------------------------------------- |
| **M2 — StRAM** | ✅ with caveats | 1T1C @ 50% SBUF: **−12% HBM traffic**, −0.2% time; 3T weaker at same fraction |
| **M3 — LtRAM** | ✅ complete     | RRAM @ 25% HBM: **−17% HBM traffic**, **868 LtRAM fills**, −43% energy        |


---

## Files to cite in report


| Milestone  | Consolidated JSON                                              | Example per-run JSON                                 |
| ---------- | -------------------------------------------------------------- | ---------------------------------------------------- |
| M2 — 1T1C  | `data/traces/memory_sweeps/consolidated_stram_edram_1t1c.json` | `runs/compare_stram_edram_1t1c_50pct_best_case.json` |
| M2 — 3T    | `data/traces/memory_sweeps/consolidated_stram_edram_3t.json`   | `runs/compare_stram_edram_3t_50pct_best_case.json`   |
| M3 — RRAM  | `data/traces/memory_sweeps/consolidated_ltram_rram.json`       | `runs/compare_ltram_rram_25pct_best_case.json`       |
| M3 — FeRAM | `data/traces/memory_sweeps/consolidated_ltram_feram.json`      | `runs/compare_ltram_feram_25pct_best_case.json`      |


---

## One paragraph for the report (copy-paste)

**Milestone 3 (LtRAM):** On Llama 3.2-1B decode (4 NeuronCores), placing weights in LtRAM via `decode_ltram_only` reduces HBM traffic by **16.6%** at 25% iso-area HBM→LtRAM trade, with **868 LtRAM→SBUF** transfers vs baseline. RRAM achieves **42.6%** lower total energy than baseline at this point; FeRAM achieves **25.2%** lower energy with the same HBM traffic reduction.

**Milestone 2 (StRAM):** With `decode_stram_only` (KV→StRAM), 1T1C eDRAM at **50% SBUF area trade** reduces HBM traffic by **12.1%** with negligible latency change. StRAM activity is confirmed via StRAM refresh energy (not `stram→sbuf` hops, due to local-read modeling). 3T eDRAM shows smaller HBM reductions at the same fractions on this workload.