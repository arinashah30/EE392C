# Placement and eviction (simple guide)

How **dmsim** decides where tensors “live” and what happens when memory fills up or a kernel ends.

Code lives in `src/dmsim/policies/placement.py` and `src/dmsim/sim/engine.py`.

---

## Two ideas to remember

**Home** — Where a tensor is supposed to stay long-term (like weights in LtRAM or HBM).

**SBUF (and PSUM)** — Small fast scratch space near the compute. The trace usually says “bring data **into SBUF** for this op.” After a kernel finishes, the model **clears SBUF** and the next op must **load again from home**.

---

## The two placement policies

Policies are YAML files under `configs/policies/`. They only answer: **by tensor type, which memory tier is home?**

### Baseline — everything in HBM

File: `configs/policies/baseline_hbm.yaml`

| Tensor type | Home |
|-------------|------|
| weights, KV, activations, other | **HBM** |

Use with the simple hierarchy (`trainium2_baseline.yaml`): only PSUM, SBUF, and HBM turned on.

### Decode tiered — split across new memories

File: `configs/policies/decode_tiered.yaml`

| Tensor type | Home |
|-------------|------|
| weights | **LtRAM** |
| KV cache | **StRAM** |
| hidden / activation | **StRAM** |
| other (unmapped names) | **HBM** |

Use with a “differentiated” hierarchy (`trainium2_diff_mem_50sbuf_25hbm.yaml`) that turns on StRAM and LtRAM and shrinks SBUF/HBM to match fixed chip area.

**Note:** Ingest often labels many tensors as **`other`** because decode NEFF names/shapes don’t match the LLaMA name mapper. Then they stay on **HBM** even with `decode_tiered`.

---

## How placement is chosen (once at start)

1. Read each tensor’s **category** from the trace.
2. Look up **home** in the policy table above.
3. If that memory tier is **off** in the hierarchy, push the tensor **down** toward HBM.
4. If too many bytes are assigned to one tier **vs its capacity**, move the **largest** tensors to the **next slower** tier until it fits.

This runs **once** when the simulation starts. It does **not** move tensors around during the run.

**Area budget** (on the 50%/25% hierarchy): StRAM steals half of SBUF’s area per core; LtRAM steals a quarter of HBM per chip. The policy table stays the same; **capacities** change.

---

## What happens on each memory access in the trace

For each access event (in time order):

1. Figure out where data is now (**home**, unless it was recently loaded into SBUF).
2. The trace says where it needs to go (almost always **SBUF**).
3. If those differ → **charge time and energy** to move bytes along the memory stack.
4. If already in SBUF → only charge a **cheap local** access.

**HBM traffic** in the report counts bytes that cross **into or out of HBM** on that path.

### How transfer hops are computed

Code: `_charge_path` in `src/dmsim/sim/engine.py` uses **`path_between`** (home-aware), then charges **one** link per logical hop with `transfer_latency_ns` / `transfer_energy_pJ`.

Example: weight **home = LtRAM**, load to SBUF → logical hop **`ltram → sbuf` only** (not `ltram → stram → sbuf`, not via HBM).

| Tensor home | Hop charged (time & energy) |
|-------------|-----------------------------|
| HBM | `hbm → sbuf` |
| LtRAM | `ltram → sbuf` |
| StRAM | `stram → sbuf` |

`physical_hops_between` in `transfer.py` still exists for tests/tools that need the full linear stack walk; the simulator does **not** use it for access costing anymore.

### Transfer time and bandwidth

Per hop: `read_latency + nbytes / bandwidth + write_latency` (`transfer_latency_ns`).

- **Unspecified hops** default to **Trainium DMA** (`16 × 23 B/ns ≈ 368 GB/s`), then `min(link, 368)` when `dma_cap_transfer_bandwidth: true`. LtRAM→SBUF and HBM→SBUF use this unless you override `links_GBs`.
- **Literature macro bandwidths** live in `configs/tech_specs/*.yaml` as `max_bandwidth_GBs`. They do **not** automatically slow every hop (RRAM 2.3 GB/s is a crossbar read rate, not the NeuronCore DMA delivery path). To model a macro-limited hop, set it in hierarchy YAML (e.g. `stram_sbuf: 128` for 1T1C eDRAM).

| Tech file | Macro `max_bandwidth_GBs` | Typical explicit link |
|-----------|---------------------------|------------------------|
| `edram_1t1c.yaml` | 128 GB/s aggregate | `stram_sbuf: 128` |
| `edram_3t.yaml` | 128 GB/s | `stram_sbuf: 128` (3T StRAM) |
| `rram.yaml` | 2.3 GB/s read | optional `ltram_sbuf: 2.3` for array-limited study |
| `feram.yaml` | 34 GB/s | optional `stram_sbuf: 34` if FeRAM StRAM |

Neuron profile `dma[].duration` is **not** used for latency today.

---

## Three ways data gets kicked out or reset

### 1. End of kernel — SBUF cleared (main one)

After every **`kernel_end`** in the trace:

- **SBUF and PSUM are wiped** on that core.
- Tensors are considered back at **home** (not sitting in SBUF anymore).

So **every kernel** that needs SBUF data pays the **full load from home** again. The simulator does **not** keep hot data in SBUF across kernels.

Configured in hierarchy YAML: `kernel.wipe_levels_on_boundary: [psum, sbuf]`.

### 2. Pool full — drop one cached tensor

If SBUF (or StRAM, etc.) runs out of space while tracking what’s cached:

- Remove **one** older entry (first in a simple list — not a full LRU).
- **No extra traffic** is modeled for that eviction (no writeback to HBM).

### 3. StRAM retention timer (only if home is StRAM)

StRAM tech has a short **retention** (~40 µs). If a tensor **homes in StRAM** and nothing touches it for longer than that:

- Next access is treated as **stale/corrupt**.
- Model reloads from **HBM → home** and counts a retention eviction.

Weights in LtRAM and data in HBM do **not** use this rule.

---

## What the simulator does *not* do

- Does not model **writing back** evicted SBUF data to HBM/StRAM.
- Does not model **loading weights into LtRAM once**; only what the trace’s access events say.
- Does not overlap compute and memory time (everything adds up sequentially).
- Trace ingest often **misses small writes** (see profiler docs) — so “writes” look rare even on real hardware.

---

## Example commands

```bash
cd /home/ubuntu/EE392C
export PYTHONPATH=src

# Baseline
python3 -m dmsim.cli run \
  --hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --policy configs/policies/baseline_hbm.yaml \
  --trace data/traces/llama32_1b_decode_4core_min1.json

# Compare baseline vs tiered
python3 -m dmsim.cli compare \
  --trace data/traces/llama32_1b_decode_4core_min1.json \
  --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --candidate-hierarchy configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml \
  --baseline-policy configs/policies/baseline_hbm.yaml \
  --candidate-policy configs/policies/decode_tiered.yaml \
  --output data/traces/sim_results.json
```

---

## Where to look in the repo

| What | Where |
|------|--------|
| Policy tables | `configs/policies/` |
| Memory stack + kernel wipe | `configs/hierarchy/` |
| Placement + spill | `src/dmsim/policies/placement.py` |
| Access loop + wipe + retention | `src/dmsim/sim/engine.py` |
| Energy/latency per technology | `configs/tech_specs/` |
