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
3. If those differ → **charge time and energy** for **one direct hop** (`source → target`). Multi-tier staging must appear as separate trace events.
4. If already at target (SBUF scratch hit, StRAM direct read, or same-level read) → **cheap local** access via `_charge_local_access`.
5. Same-level **writes** to SBUF are **omitted** (no cost).

**HBM traffic** in the report counts bytes that cross **into or out of HBM** on charged interconnect hops.

### How transfer hops are computed

Code: `_charge_path` in `src/dmsim/sim/engine.py` calls **`path_between`**, which returns **one direct edge** `(source, dest)` — it does **not** walk intermediate YAML levels. The optional `home_id` argument is **ignored**.

Example: weight **home = LtRAM**, load to SBUF → **`ltram → sbuf` only** (not `ltram → stram → sbuf`, not via HBM).

| Tensor home | Hop charged (time & energy) |
|-------------|-----------------------------|
| HBM | `hbm → sbuf` |
| LtRAM | `ltram → sbuf` |
| StRAM | **Local read at StRAM** when homed+resident there — **not** `stram → sbuf` |

`physical_hops_between` in `transfer.py` is a **backward-compatible alias** of `hops_between` (same direct-edge behavior).

### Transfer time and bandwidth

Per hop: `read_latency + nbytes / bandwidth + write_latency` (`transfer_latency_ns`).

Hierarchy YAML sets interconnect bandwidth: **DMA hops** use `dma_bandwidth_GBs`; **datapath reads** (SBUF scratch, StRAM direct read) use `on_chip_bandwidth_GBs`. Hierarchy sets:

```yaml
interconnect:
  dma_bandwidth_GBs: 368        # off-chip ↔ anything
  on_chip_bandwidth_GBs: 10000  # on-chip ↔ on-chip only
```

Each level is **`on_chip`** (PSUM, SBUF, StRAM) or **`off_chip`** (HBM, LtRAM). For a hop, if **both** endpoints are on-chip → on-chip BW; otherwise → DMA BW.

Neuron profile `dma[].duration` is **not** used for latency today.

---

## Three ways data gets kicked out or reset

### 1. End of kernel — SBUF/PSUM cleared (main one)

After every **`kernel_end`** in the trace (for the affected NeuronCore(s)):

- **SBUF and PSUM are wiped** on that core (levels in `kernel.wipe_levels_on_boundary`).
- **`resident_level`** resets to **home** only for tensors on the **same core(s)** whose resident tier was wiped.

**Core scope** (both fast buffers and residency use the same rule):

| `kernel_end.core_id` | Effect |
|----------------------|--------|
| Set (e.g. `0`) | Core 0 only |
| Omitted (`null`) | All cores that already have fast-buffer state (typical single-core ingest) |

Multi-core merged traces set `core_id` per core in [`merge_traces`](../../src/dmsim/trace/neuron_json_ingest.py).

**StRAM and LtRAM are not wiped** by default — near-memory homes persist across kernels on every core. StRAM-homed KV stays at StRAM; reads to SBUF use the **StRAM direct read** path (local, not `stram→sbuf`).

So **every kernel** that needs SBUF data from HBM/LtRAM pays the **full load from home** again after that core’s wipe. Other cores’ SBUF scratch can remain hot until their own `kernel_end`.

Configured in hierarchy YAML: `kernel.wipe_levels_on_boundary: [psum, sbuf]`.

### 2. Pool full — drop one cached tensor

If SBUF (or StRAM, etc.) runs out of space while tracking what’s cached:

- Remove **one** older entry (first in a simple list — not a full LRU).
- **No extra traffic** is modeled for that eviction (no writeback to HBM).

### 3. Background refresh (StRAM / HBM)

Between trace events, the simulator charges **refresh energy** for occupied bytes at each level’s `refresh_interval_s` (from tech YAML, overridable per level). **StRAM data is assumed to stay valid** while allocated — there is no retention-expiry / corrupt-reload path. LtRAM is non-volatile and has no refresh in the default specs.

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
