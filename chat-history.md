# EE392C dmsim — Session Chat History

This document summarizes the full discussion from a Cursor agent session on the **differentiated memory simulator (`dmsim`)** for AWS Trainium2 (PSUM → SBUF → StRAM → LtRAM → HBM). It is written for sharing with a project partner.

---

## Project context (carried into the session)

### Goals

- Build a **trace-driven analytical simulator** comparing baseline Trainium2 (PSUM + SBUF + HBM) vs. expanded hierarchy with **StRAM** (per-core eDRAM) and **LtRAM** (per-chip RRAM).
- Input: **Neuron Explorer JSON** exports (`profile.json` + per-core `*_nc_*_session_*.json`), not raw NEFF/NTFF.
- Focus: **LLM decode**, constant-area tradeoffs (StRAM area from SBUF; LtRAM area from HBM).
- Example profile: `data/traces/neuron_profile_json_4-19/`, model key `124050204400345`, 4 NeuronCores (0–3).

### Placement policy (`configs/policies/decode_tiered.yaml`)

| Category     | Home level |
|-------------|------------|
| weight      | LtRAM      |
| kv_cache    | StRAM      |
| hidden      | StRAM      |
| activation  | StRAM      |
| other       | HBM        |

Baseline policy keeps everything in HBM.

### Known issues at session start

1. **Tensor labels**: After ingest, ~1616/1628 tensors were `other`/HBM; almost no `weight` or `kv_cache` — decode policy barely applied.
2. **Transfer paths**: `path_between()` walked every enabled level (e.g. `hbm→ltram→stram→sbuf`), inflating energy vs. baseline even for HBM-homed data.
3. User had a prior script `src/dmsim/trace/tensor_name_mapper.py` (`LLaMANameMapper`) to map `input0`, `input1`, … to semantic names (KV cache, attention/MLP weights, etc.).

---

## Message 1 — User: Integrate tensor name mapper

**User request:** The problem is with tensor labels. There should be better classification (weights, KV cache, etc.). Adapt `tensor_name_mapper.py` so tensors are named correctly and the simulator works. Logic could move into `neuron_json_ingest.py` if that makes sense.

### What we found in the profile JSON

- Device JSON contains **`neff_node[]`**: `variable_name` (`input0`, `input10`, …), `shape`, `type` (`IN`, `OUT`, `WEIGHT`), `size` (bytes).
- DMA records mostly use **compiler names** (`transpose.148_sg0000`, `concatenate.*`, `unknown`) — not `inputN`.
- Only a tiny fraction of DMA `variable` fields match `input\d+`.
- Annotations occasionally reference `tensor_name: input1`.

### What we implemented

**`tensor_name_mapper.py`** (package module):

- `mapper_category_to_sim()` — maps mapper categories → `TensorCategory` (`weight`, `kv_cache`, `activation`, `other`).
- `NeffTensorCatalog` — built from `neff_node[]`; uses `LLaMANameMapper.auto_detect_config()`.
- `resolve_dma()` — links DMA variables to catalog entries via:
  - Direct `inputN` / `outputN` match
  - `WEIGHT→SB` DMA routes → weight
  - 4D `read_shape` heuristics → KV cache
  - Unique byte-size match to NEFF tensors
  - Mapper fallback for compiler names

**`neuron_json_ingest.py`**:

- Seeds trace tensors from NEFF catalog (semantic names, e.g. `layer_7.mlp.gate_proj.weight`).
- Resolves DMA access `tensor_id` to catalog when possible.

**KV / layer detection fix:**

- Auto-detect was treating all 4D tensors as KV (~98 “layers”).
- Added `_is_kv_cache_shape()` — true KV shapes like `(batch, seq, n_kv_heads, head_dim)`, excluding compiler temps like `(32, 128, 1, 1)`.
- Only `input*` tensors with KV shapes count toward layer count → **16 layers** on the example profile.

**Category mapping tweaks:**

- `output` / logits → `ACTIVATION` (not weight).
- Compiler `WEIGHT` nodes with `unknown` category → `OTHER`, not weight.

**Tests added:**

- `tests/test_tensor_name_mapper.py`
- `tests/test_neuron_ingest.py::test_ingest_classifies_weights_and_kv`

### Ingest results (single core, after mapping)

| Category    | ~Count |
|------------|--------|
| other      | 392    |
| weight     | 146    |
| kv_cache   | 64     |
| activation | 5      |

Much better than ~1600 `other`, but compiler DMA names still dominate access events.

---

## Message 2 — User: How does access energy work in real hardware? Fix path issue.

**User question:** If data is not in StRAM/LtRAM, should that still count as an access? Probably not — page table knows data is in HBM and skips StRAM/LtRAM. Fix so read/write energies are only added when a read/write occurs.

### Explanation (real hardware)

1. Each tensor has a **home** (persistent location from placement / page tables).
2. Compute needs data in **SBUF** (or PSUM) — that incurs SBUF access energy.
3. On a miss, the memory system fetches from **home**, not from every tier in the stack.
4. StRAM/LtRAM energy applies only when those cells are **actually accessed** (or retained/refreshed), not merely because they exist in the address map.

### What was wrong in the simulator

`path_between(hbm, sbuf)` always walked **all** enabled intermediate levels: `hbm→ltram→stram→sbuf`, charging access energy at LtRAM and StRAM even when **home = HBM**.

### What we fixed

**`src/dmsim/sim/transfer.py`:**

- `physical_hops_between()` — every adjacent link on the hierarchy (for summing link cost).
- `path_between(..., home_id=...)` — **waypoints** only at `{source, dest, home}`; skipped tiers are not separate hop endpoints in metrics.
- `transfer_*_between_levels()` — sums physical link latency/energy across skipped tiers for one logical hop (e.g. `hbm→sbuf`).

**`src/dmsim/sim/engine.py`:**

- `_charge_path()` passes `home_id` from tensor placement.

**Tests:**

- `tests/test_transfer.py`
- `tests/test_sim.py::test_hbm_homed_access_skips_stram_ltram_hop_keys` — confirms `hbm→ltram` / `ltram→stram` do not appear in `transfers_by_hop` for HBM-homed weights.

**Caveat (important):** Logical `hbm→sbuf` still **sums energy over all physical links** in the differentiated hierarchy (3 links vs. 1 in baseline). Hop *labels* were fixed; total energy for HBM-homed traffic can still be ~4× baseline per transfer for that reason.

---

## Message 3 — User: Rerun simulation (50% StRAM / 25% LtRAM)

**User request:** Rerun with `configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml` after fixes.

### Command run

```bash
PYTHONPATH=src python3 -m dmsim.cli pipeline \
  --profile-dir data/traces/neuron_profile_json_4-19 \
  --model-key 124050204400345 \
  --trace-cache data/traces/ingested_all_cores.json \
  --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml \
  --baseline-policy configs/policies/baseline_hbm.yaml \
  --candidate-policy configs/policies/decode_tiered.yaml \
  --output data/traces/sim_results_50sbuf_25hbm.json
```

### Area budget (50% / 25% fractions)

| Pool            | After tradeoff      |
|-----------------|---------------------|
| SBUF per core   | 14 MiB (50% of 28)  |
| StRAM per core  | 14 MiB              |
| LtRAM per chip  | 24 GiB (25% of 96)  |
| HBM per chip    | 72 GiB              |

### Trace (re-ingested)

- **2468 tensors**, **11864 events**, cores **0–3**
- Written to `data/traces/ingested_all_cores.json`

### Results vs baseline

| Metric          | Baseline   | 50% StRAM / 25% LtRAM | Change   |
|-----------------|------------|------------------------|----------|
| Total time      | 37.1 ms    | 37.7 ms                | +1.6%    |
| Total energy    | 10.4 J     | 40.2 J                 | +285%    |
| HBM traffic     | 162.7 MB   | 161.9 MB               | **−0.5%** |
| Retention evictions | 0      | 0                      | —        |

**Baseline transfers:** `hbm→sbuf` (1410)

**Candidate transfers:** `hbm→sbuf` (1402), `ltram→sbuf` (8), `hbm→ltram` (4), `ltram→hbm` (4) — no `hbm→ltram→stram→sbuf` chain in hop list.

Full JSON report: `data/traces/sim_results_50sbuf_25hbm.json`

---

## Message 4 — User: Why are results still the same? Isn’t decode_tiered used?

**User concern:** Results look unchanged after tensor mapping and energy fixes. Is the simulator using `decode_tiered.yaml`?

### Answer: Yes, the policy is used — but the trace doesn’t exercise it for most bytes

#### Policy is loaded correctly

Confirmed in `sim_results_50sbuf_25hbm.json` → `candidate.configuration.policy` shows `decode_tiered` with `weight→ltram`, `kv_cache→stram`, etc.

#### Tensor categories after ingest (all cores)

| Category    | Tensors |
|------------|---------|
| other      | 1608    |
| weight     | 584     |
| kv_cache   | 256     |
| activation | 20      |

#### Final homes after capacity spill

| Home  | Count | Notes |
|-------|-------|-------|
| HBM   | 1608  | All `other` |
| LtRAM | 860   | All weights + all KV + all activations |
| StRAM | **0** | KV+activation wanted StRAM (276 tensors) but **spilled** (14 MiB/core insufficient) |

#### Access events — the critical gap

| Category    | Access events | Access bytes |
|------------|---------------|--------------|
| other      | 1944          | **181.5 MB** |
| activation | few           | 0.9 MB       |
| kv_cache   | **0**         | 0            |
| weight     | **0**         | 0            |

**Neuron DMA traffic is tied to compiler tensor IDs** (`transpose.*`, `concatenate.*`), which remain `other` → HBM. NEFF catalog tensors (`layer_*.cache_k`, `layer_*.attention.wq.weight`, …) are classified correctly but **almost no DMA events reference those IDs**.

So ~99% of simulated traffic behaves like baseline (HBM home, `hbm→sbuf`), which explains **unchanged HBM traffic**.

#### Why energy is still ~4× baseline

1. **Same logical traffic** — mostly `hbm→sbuf`.
2. **Physical energy sum** — for diff mem, one logical `hbm→sbuf` still sums **3 physical links** (`hbm→ltram→stram→sbuf`). Measured ratio ~**4.26×** per 64 KiB transfer vs. baseline’s single `hbm→sbuf` link.
3. Hop *labels* were fixed; **total link energy for bypassed tiers was not removed**.

### What would need to change to see policy impact

1. **Attribute DMA bytes to semantic tensors** (or classify compiler names / size-match aggressively) so `weight` and `kv_cache` access events are non-zero.
2. **StRAM capacity / spill model** — or larger StRAM fraction so KV can actually home in StRAM.
3. **Energy model for bypass** — HBM-homed data should not pay LtRAM/StRAM access energy on the bypass path (only HBM + SBUF, or explicit NoC cost).

---

## Message 5 — User: Save chat history

**User request:** Write entire discussion history into `chat-history.md` for sharing with partner.

(This file.)

---

## Key files touched in this session

| File | Role |
|------|------|
| `src/dmsim/trace/tensor_name_mapper.py` | LLaMA mapper, `NeffTensorCatalog`, `mapper_category_to_sim` |
| `src/dmsim/trace/neuron_json_ingest.py` | Catalog seeding + DMA resolution |
| `src/dmsim/sim/transfer.py` | Home-aware `path_between`, physical hop helpers |
| `src/dmsim/sim/engine.py` | Pass `home_id` into `_charge_path` |
| `tests/test_tensor_name_mapper.py` | Mapper/catalog unit tests |
| `tests/test_transfer.py` | Path / hop tests |
| `tests/test_neuron_ingest.py` | Ingest + classification test |
| `tests/test_sim.py` | HBM-homed hop key test |
| `configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml` | 50% StRAM / 25% LtRAM area fractions |
| `configs/policies/decode_tiered.yaml` | Tiered placement policy |
| `data/traces/ingested_all_cores.json` | Re-ingested trace (2468 tensors) |
| `data/traces/sim_results_50sbuf_25hbm.json` | Latest compare results |

---

## Suggested next steps (for the team)

1. **DMA → semantic tensor attribution** — roll compiler DMA variables into catalog tensor IDs so access events hit `weight` / `kv_cache`.
2. **Bypass energy model** — for `home=hbm`, charge only HBM + SBUF (or one aggregated NoC cost), not full LtRAM/StRAM chain.
3. **Re-run pipeline** after (1) and (2); compare HBM traffic and energy by level.
4. Optional: report **homes vs. access bytes** in CLI output so policy effectiveness is visible per run.

---

## CLI quick reference

```bash
# Re-ingest all cores
PYTHONPATH=src python3 -m dmsim.cli ingest \
  --profile-dir data/traces/neuron_profile_json_4-19 \
  --model-key 124050204400345 \
  --output data/traces/ingested_all_cores.json

# Pipeline: ingest + baseline vs candidate
PYTHONPATH=src python3 -m dmsim.cli pipeline \
  --profile-dir data/traces/neuron_profile_json_4-19 \
  --model-key 124050204400345 \
  --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml \
  --output data/traces/sim_results_50sbuf_25hbm.json

pytest tests/
```

---

*Generated from Cursor agent session — May 2026.*
