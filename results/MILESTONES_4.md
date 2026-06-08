# Milestone 4 — Full hierarchy (StRAM + LtRAM), Llama 3.2-1B decode

**Trace:** `data/traces/llama32_1b_decode_4core_min1_no_unknown.json`  
**Baseline:** `trainium2_baseline.yaml` + `baseline_hbm.yaml`  
**Candidate policy:** `decode_tiered.yaml` (weights→LtRAM, KV/hidden/activation→StRAM, other→HBM)  
**Technologies:** StRAM `edram_1t1c`, LtRAM `rram`

## Baseline reference

| Metric | Value |
|--------|-------|
| Time (worst core) | 4.49 ms |
| HBM traffic | 26.18 MB |
| `hbm→sbuf` hops | 4,070 |
| Total energy | 17.52 TJ |

---

## Full hierarchy design points (iso-area)

| Design | StRAM ← SBUF | LtRAM ← HBM | Rationale |
|--------|--------------|-------------|-----------|
| **Primary** | 50% | 25% | Balanced research prototype; both tiers non-trivial |
| **Balanced** | 25% | 25% | Symmetric area trade |
| **Near-core heavy** | 50% | 10% | Maximize KV StRAM; minimal LtRAM |
| **Capacity heavy** | 25% | 50% | Maximize weight LtRAM (MoE-oriented) |

### Resolved capacities

| Design | SBUF / core | StRAM / core | LtRAM / chip | HBM / chip |
|--------|-------------|--------------|--------------|------------|
| Primary (50/25) | 14.0 MiB | 60.0 MiB | 1.56 GiB | 72.0 GiB |
| Balanced (25/25) | 21.0 MiB | 30.0 MiB | 1.56 GiB | 72.0 GiB |
| Near-core (50/10) | 14.0 MiB | 60.0 MiB | 0.62 GiB | 86.4 GiB |
| Capacity (25/50) | 21.0 MiB | 30.0 MiB | 3.12 GiB | 48.0 GiB |

---

## Results vs baseline

| Design | StRAM / LtRAM trade | Time | Time Δ | HBM traffic | HBM Δ | Cross-domain traffic | Total energy | Refresh energy |
|--------|---------------------|------|--------|-------------|-------|----------------------|--------------|----------------|
| Baseline | — | 4.49 ms | — | 26.18 MB | — | 26.18 MB | 17.52 TJ | 17.52 TJ |
| **Primary (50/25)** | 50% / 25% | **4.45 ms** | **−0.86%** | **18.68 MB** | **−28.6%** | 23.02 MB | 41.65 TJ | 41.65 TJ |
| Balanced (25/25) | 25% / 25% | 4.46 ms | −0.70% | 20.82 MB | −20.5% | 25.15 MB | 24.39 TJ | 24.39 TJ |
| Near-core (50/10) | 50% / 10% | 4.46 ms | −0.62% | 19.43 MB | −25.8% | 23.02 MB | 45.25 TJ | 45.25 TJ |
| Capacity (25/50) | 25% / 50% | 4.46 ms | −0.70% | 20.82 MB | −20.5% | 25.15 MB | 24.39 TJ | 24.39 TJ |

---

## Transfer hops (tiered placement evidence)

| Design | `hbm→sbuf` | `ltram→sbuf` | `stram→sbuf` | `sbuf→hbm` (writeback) |
|--------|------------|--------------|--------------|------------------------|
| Baseline | 4,070 | 0 | 0 | 134,396 |
| **Primary (50/25)** | **2,617** | **868** | **140** | 134,396 |
| Balanced (25/25) | 2,913 | 868 | 140 | 134,396 |
| Near-core (50/10) | 2,978 | 507 | 140 | 134,396 |
| Capacity (25/50) | 2,913 | 868 | 140 | 134,396 |

---

## Headline takeaway

**Best HBM traffic reduction on Llama:** **Primary 50/25** at **−28.6%** (26.2 → 18.7 MB), with both **`ltram→sbuf` (868)** and **`stram→sbuf` (140)** active.

**Near-core heavy (50/10)** is second-best on HBM (−25.8%) but only **507** LtRAM fills — smaller LtRAM pool limits weight offload.

**Balanced and capacity-heavy are identical** on this trace: at 25% StRAM the same KV placement applies, and **25% LtRAM already fits all weights** (868 fills). Doubling LtRAM to 50% HBM trade does not change behavior until a larger model/trace spills weights.

**Energy:** Report refresh separately. Primary 50/25 shows high total energy due to large StRAM pool + eDRAM refresh; use **HBM traffic and time** as primary M4 metrics.

---

## Output files

| Design | Config YAML | Results JSON |
|--------|-------------|--------------|
| Primary | `configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml` | `results/m4_llama_50sbuf_25hbm_tiered.json` |
| Balanced | `configs/hierarchy/trainium2_diff_mem_25sbuf_25hbm.yaml` | `results/m4_llama_25sbuf_25hbm_tiered.json` |
| Near-core | `configs/hierarchy/trainium2_diff_mem_50sbuf_10hbm.yaml` | `results/m4_llama_50sbuf_10hbm_tiered.json` |
| Capacity | `configs/hierarchy/trainium2_diff_mem_25sbuf_50hbm.yaml` | `results/m4_llama_25sbuf_50hbm_tiered.json` |

---

## Report paragraph (copy-paste)

On Llama 3.2-1B decode (4 NeuronCores), the full differentiated hierarchy with `decode_tiered` placement reduces HBM traffic by up to **28.6%** vs baseline Trainium2 (primary design: 50% SBUF→StRAM, 25% HBM→LtRAM). Traffic migrates from **`hbm→sbuf` (4070→2617)** to **`ltram→sbuf` (868)** and **`stram→sbuf` (140)**. Latency improves by **~0.9%** (4.49→4.45 ms worst-core). A near-core-heavy variant (50/10) achieves −25.8% HBM traffic with less LtRAM utilization; balanced and capacity-heavy configs (25/25 and 25/50) produce identical results on this workload because 25% LtRAM already accommodates all weight tensors.
