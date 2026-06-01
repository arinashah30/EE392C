# Simulation engine (`dmsim.sim`)

Trace-driven analytical model for Trainium2 memory hierarchy studies. This package **replays** ingested Neuron profiles and accumulates **time**, **energy**, and **HBM traffic** — it does not re-execute the model.

**Cost model reference (hops, latency, energy, HBM traffic):** [`docs/cost-model/README.md`](../../../docs/cost-model/README.md)

**Public API:** [`sim/__init__.py`](__init__.py) exports `run_simulation`, `SimulationResult`.

**CLI entry:** [`cli.py`](../cli.py) — `dmsim run` loads trace/hierarchy/policy and calls `run_simulation`.

| Area | Code |
|------|------|
| Trace ingest | [`trace/neuron_json_ingest.py`](../trace/neuron_json_ingest.py) |
| Trace types / load | [`trace/schema.py`](../trace/schema.py) |
| Placement | [`policies/placement.py`](../policies/placement.py) |
| Policy YAML → model | [`config/loader.py`](../config/loader.py) `load_policy` |
| Hierarchy YAML → model | [`config/loader.py`](../config/loader.py) `load_hierarchy` |
| Resolved config types | [`config/models.py`](../config/models.py) |
| Simulation engine | [`sim/engine.py`](engine.py), [`sim/transfer.py`](transfer.py), [`sim/residency.py`](residency.py) |

Illustrative ids and sizes below are simplified.

---

## Table of contents

1. [What this simulator does](#1-what-this-simulator-does)
2. [Terminology](#2-terminology)
3. [Inputs: hierarchy, policy, and trace](#3-inputs-hierarchy-policy-and-trace)
4. [Placement (once at start)](#4-placement-once-at-start)
5. [Runtime state during replay](#5-runtime-state-during-replay)
6. [Event replay loop](#6-event-replay-loop)
7. [How transfer cost is computed](#7-how-transfer-cost-is-computed)
8. [Processing an access (`_handle_access`)](#8-processing-an-access-_handle_access)
9. [Output: `SimulationResult`](#9-output-simulationresult)
10. [Limitations and comparisons](#10-limitations-and-comparations)
11. [Module reference](#11-module-reference)

### Code map (section → source)

| Doc section | Primary code |
|-------------|--------------|
| §3 Hierarchy | [`config/loader.py`](../config/loader.py) `load_hierarchy` · [`config/models.py`](../config/models.py) `ResolvedHierarchy` |
| §3 Policy | [`config/loader.py`](../config/loader.py) `load_policy` · [`config/models.py`](../config/models.py) `PolicyConfig` |
| §3 Trace ingest | [`trace/neuron_json_ingest.py`](../trace/neuron_json_ingest.py) |
| §3 Trace types | [`trace/schema.py`](../trace/schema.py) `Trace`, `TensorRecord`, `AccessEvent` |
| §4 Placement | [`policies/placement.py`](../policies/placement.py) `assign_home_levels` |
| §5 Runtime state | [`residency.py`](residency.py) |
| §6 Event loop | [`engine.py`](engine.py) `run_simulation` |
| §6 Kernel wipe | [`engine.py`](engine.py) `_handle_kernel_boundary` · ingest `_kernel_events_from_layers` |
| §7 Transfer cost | [`transfer.py`](transfer.py) `path_between` · [`engine.py`](engine.py) `_charge_path` · `_add_core_latency` |
| §8 Access handler | [`engine.py`](engine.py) `_handle_access` · `_source_level_for_access` |
| §9 Output | [`engine.py`](engine.py) `SimulationResult` |

---

## 1. What this simulator does

You captured a real Llama decode run on Trainium (DMA moves, kernel boundaries). You want to ask:

> If weights lived in **LtRAM** instead of HBM, and KV in **StRAM**, how would **time**, **energy**, and **HBM traffic** change — given the *same* access pattern?

The simulator **does not** re-run the model. It walks a **trace** (ordered list of events) and **charges cost** for each memory movement the trace records.

High-level pipeline:

```text
trace + hierarchy + policy
        │
        ▼
  assign_home_levels()          ← once: tensor → home level
        │
        ▼
  run_simulation()              ← replay events in time order
        │
        ├── kernel_end → reset fast buffers (SBUF/PSUM)
        └── access     → charge transfer or local access
        │
        ▼
  SimulationResult              ← time, energy, HBM bytes, per-hop counts
```

Three inputs answer three questions:

| Input | Question |
|-------|----------|
| **Trace** | What memory traffic happened on hardware? |
| **Policy** | Where should each tensor **category** live long-term? |
| **Hierarchy** | What memory tiers exist, with what capacity and link speed? |

**Code:** [`cli.py`](../cli.py) orchestrates load + sim · [`engine.py`](engine.py) `run_simulation` (L48) · [`placement.py`](../policies/placement.py) `assign_home_levels` (L18) called at [`engine.py`](engine.py) L54.

---

## 2. Terminology

Read this section first — later sections use these terms without re-defining them.

### Memory levels

A **level** is a tier in the memory stack (identified by a string id such as `sbuf`, `hbm`).

```text
  per_core (one copy per NeuronCore)     per_chip (shared)
  ┌────────┐
  │  PSUM  │  partial-sum scratch; wiped each kernel
  ├────────┤
  │  SBUF  │  compute-side buffer; most trace loads target here
  ├────────┤
  │ StRAM  │  optional near-core tier (eDRAM-style)
  └────────┘
       │ interconnect links (bandwidth in hierarchy YAML)
       ▼
  ┌────────┐     ┌────────┐
  │ LtRAM  │────▶│  HBM   │  optional chip-wide tiers
  └────────┘     └────────┘
```

| Level | Scope | Typical role |
|-------|-------|--------------|
| `psum` | per core | Partial sums; cleared on kernel boundary |
| `sbuf` | per core | Operand buffer for compute; trace `target_level` default |
| `stram` | per core | Policy home for KV / activations (when enabled) |
| `ltram` | per chip | Policy home for weights (when enabled) |
| `hbm` | per chip | Baseline home; spill target; deepest tier |

### Tensor and category

A **tensor** is a named block of bytes in the trace catalog (`TensorRecord`). Each tensor has a **category** (`weight`, `kv_cache`, `hidden`, `activation`, `other`) used by the **policy** to pick a home level.

### Home vs resident

| Term | Meaning | When set | Changes during replay? |
|------|---------|----------|------------------------|
| **home_level** | Long-term placement from policy (+ capacity spill) | Once, at start | No |
| **resident_level** | Where the simulator thinks a copy of the tensor is **right now** | Updated on each access; reset on kernel wipe | Yes |

**Home** is “where it belongs.” **Resident** is “where we last moved it for costing.” They diverge when data is DMA’d into SBUF but homed in LtRAM.

### Trace events

The trace is a time-ordered list of events. Only two kinds matter to the engine:

| Event type | Purpose |
|------------|---------|
| **`AccessEvent`** | Move or touch `bytes` of `tensor_id` toward `target_level` |
| **`KernelBoundaryEvent`** | Mark layer/kernel start or end (`kernel_end` triggers SBUF wipe) |

### Hop

A **hop** is one **adjacent** memory-to-memory link (e.g. `stram → sbuf`). The hierarchy is a set of differentiated tiers connected by explicit links.

**Routing** ([`transfer.py`](transfer.py)):

| Function | Use |
|----------|-----|
| [`path_between`](transfer.py) | **Access costing** in `_charge_path` — one direct edge |
| [`hops_between`](transfer.py) | Same as `path_between` — direct `source → dest` (no stack walk) |

Every interconnect move is **one direct hop** between source and destination levels (e.g. `hbm → sbuf`, `sbuf → hbm`, `ltram → sbuf`). The `levels:` list order in hierarchy YAML does **not** insert intermediate tiers. Multi-hop staging must appear as **separate trace access events**.

### Other terms

| Term | Meaning |
|------|---------|
| **Placement** | Mapping each trace tensor to a `home_level` via policy + spill (`fallback_by_level`) |
| **Kernel wipe** | On traced `kernel_end`, clear SBUF/PSUM and reset `resident_level` to home |
| **Local hit** | `source_level == target_level` — no interconnect; local latency/energy only |
| **Writeback** | Trace `write` to off-chip `target_level` — modeled as **`sbuf → target`** (SBUF flush) |
| **Touch home** | Update `last_home_touch_ns` when the **home** tier is read (not on SBUF-only scratch hits) |
| **Scratch hit** | `resident == target != home` (e.g. SBUF cache) — local access only, no retention |

**Code:** `TensorRecord` / categories → [`trace/schema.py`](../trace/schema.py) · `TensorResidency` → [`residency.py`](residency.py) · routing → [`transfer.py`](transfer.py).

---

## 3. Inputs: hierarchy, policy, and trace

### Hierarchy (`configs/hierarchy/*.yaml`)

Declares which levels are **enabled**, their **capacities**, **tech specs** (latency, energy, retention), **link bandwidths**, and kernel wipe config:

```yaml
interconnect:
  dma_bandwidth_GBs: 368
  on_chip_bandwidth_GBs: 10000
  level_domain:
    psum: on_chip
    sbuf: on_chip
    stram: on_chip
    ltram: off_chip
    hbm: off_chip

kernel:
  wipe_levels_on_boundary: [psum, sbuf]
```

See [`configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml`](../../../configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml) for a tiered stack example.

`interconnect.level_domain` marks each level `on_chip` or `off_chip` (drives DMA vs on-chip bandwidth). `num_cores` comes from the trace or instance spec (used for area budget and per-core time buckets).

**Code:** YAML → [`config/loader.py`](../config/loader.py) `load_hierarchy` → [`ResolvedHierarchy`](../config/models.py). Hop BW: `link_bandwidth_GBs` (DMA if either level is off-chip, else on-chip; **per NeuronCore**).

### Policy (`configs/policies/*.yaml`)

Maps **category → home level**, **spill/fallback targets**, and the default access target:

```yaml
# configs/policies/decode_tiered.yaml
name: decode_tiered
home_level_by_category:
  weight: ltram
  kv_cache: stram
  hidden: stram
  activation: stram
  other: hbm
default_access_target: sbuf
fallback_by_level:
  psum: sbuf
  sbuf: hbm
  stram: ltram    # StRAM full → spill to LtRAM
  ltram: hbm      # LtRAM full → spill to HBM
spill_victim_order: best_case   # or worst_case
```

| Policy field | Role |
|--------------|------|
| `home_level_by_category` | Initial home per tensor category |
| `fallback_by_level` | Where to move tensors when a level is **disabled** or **over capacity** |
| `spill_victim_order` | `best_case` (spill least-accessed) or `worst_case` (spill most-accessed); ties break on smaller/larger bytes |
| *(omitted level in fallback)* | Falls back to **`hbm`** via `PolicyConfig.fallback_for()` |

Example [`decode_ltram_only.yaml`](../../../configs/policies/decode_ltram_only.yaml): weights home in `ltram`, but `fallback_by_level: { ltram: hbm }` so overflow/disabled LtRAM goes **directly to HBM** — not through StRAM in the YAML stack order.

Policy does **not** decide per-access routing — the trace’s `target_level` does (with `default_access_target` as fallback).

**Code:** YAML → [`config/loader.py`](../config/loader.py) `load_policy` (L45) → [`config/models.py`](../config/models.py) `PolicyConfig` (L91) · `fallback_for()` (L99). `default_access_target` used in [`engine.py`](engine.py) `_handle_access` (L247).

### Trace (ingested JSON)

Written by [`trace/neuron_json_ingest.py`](../trace/neuron_json_ingest.py), read by [`trace/schema.py`](../trace/schema.py) `load_trace` (L91).

**Ingest entry points:** `ingest_neuron_json_profile` (L156) · DMA → accesses: `_build_from_device_dma` (L425) · kernel events: `_kernel_events_from_layers` (L688).

**Top-level shape:**

```json
{
  "version": 1,
  "metadata": {
    "workload": "llama32_decode_4core",
    "source": "neuron_json_ingest",
    "neuron_core_id": 0
  },
  "tensors": [ "..." ],
  "events": [ "..." ]
}
```

**`TensorRecord`** — catalog entry (looked up by `tensor_id` on each access). Defined [`trace/schema.py`](../trace/schema.py) L19–24; `TensorCategory` enum L11–16.

```json
{
  "id": "linear_weight_12",
  "name": "linear_12.weight",
  "bytes": 4194304,
  "category": "weight",
  "core_id": 0
}
```

```json
{
  "id": "input95",
  "name": "input95",
  "bytes": 131072,
  "category": "kv_cache",
  "core_id": 0
}
```

**`AccessEvent`** — one DMA or runtime copy to charge. Defined [`trace/schema.py`](../trace/schema.py) L33–40.

```json
{
  "type": "access",
  "t_ns": 1205000.0,
  "tensor_id": "linear_weight_12",
  "op": "read",
  "bytes": 65536,
  "target_level": "sbuf",
  "core_id": 0
}
```

Ingest maps Neuron DMA routes to `op` + `target_level` in [`trace/neuron_json_ingest.py`](../trace/neuron_json_ingest.py) `_map_dma_to_access` (L613). Runtime tensor copies: `_merge_system_tensor_events` (L646).

| DMA route (tokens) | `op` | `target_level` |
|--------------------|------|----------------|
| HBM → SBUF | `read` | `sbuf` |
| SBUF → HBM | `write` | `hbm` |
| weight/input → SBUF | `read` | `sbuf` |
| SBUF → OUTPUT | `write` | `sbuf` |

**`KernelBoundaryEvent`** — from Neuron `device.layer_summary`. Type: [`trace/schema.py`](../trace/schema.py) L43–47 · emitted by [`trace/neuron_json_ingest.py`](../trace/neuron_json_ingest.py) `_kernel_events_from_layers` (L688).

```json
{ "type": "kernel_start", "t_ns": 1000000.0, "kernel_id": 3, "core_id": null }
```

```json
{ "type": "kernel_end",   "t_ns": 1180000.0, "kernel_id": 3, "core_id": null }
```

Only **`kernel_end`** is handled by the engine today (`kernel_start` is stored but ignored). Parsed and sorted in [`trace/schema.py`](../trace/schema.py) `Trace.parsed_events` (L75–85); dispatched in [`engine.py`](engine.py) L94–96.

Events are replayed **sorted by `t_ns`**, not file order.

---

## 4. Placement (once at start)

Before replay, [`assign_home_levels`](../policies/placement.py) (L19–47) runs **once**. Called from [`run_simulation`](engine.py) L54.

Steps in code:

1. Map `tensor.category` → `policy.home_level_by_category` (L29–36)
2. **Disabled tier** → [`_fallback_level`](../policies/placement.py) (L41) walks `policy.fallback_by_level` via [`_resolve_spill_target`](../policies/placement.py) (L56)
3. **Over capacity** → [`_enforce_capacities`](../policies/placement.py) (L74) spills victims to policy fallback via [`_pick_spill_victim`](../policies/placement.py) (L88)

**Spill victim order** uses access counts from the trace ([`Trace.access_counts()`](../trace/schema.py)) passed from [`run_simulation`](engine.py):

| `spill_victim_order` | Victim picked first |
|----------------------|---------------------|
| `best_case` (default) | **Least** accessed tensor (then smaller bytes on ties) |
| `worst_case` | **Most** accessed tensor (then larger bytes on ties) |

Tensors with no access events count as **0** accesses (spilled before hot tensors in `best_case`).

Spill/fallback resolution ([`_resolve_spill_target`](../policies/placement.py) L56–69):

```python
# decode_ltram_only: fallback_by_level ltram → hbm
_resolve_spill_target("ltram", policy, enabled_ids)  # → "hbm"

# decode_tiered: stram → ltram → hbm (chain stops at first enabled != source)
_resolve_spill_target("stram", policy, enabled_ids)  # → "ltram"
```

Unlisted levels default to **`hbm`**. If the fallback chain loops or lands on the same level (e.g. HBM full with `hbm: hbm`), spill stops and homes stay put.

**Not used for spill:** hierarchy YAML `levels:` list order. Placement spill follows **policy** only. **`levels:` order is also not used for transfer routing** — all moves are direct memory-to-memory edges.

**Output** — plain dict `tensor_id → level id`:

```python
homes = {
    "linear_weight_12": "ltram",                    # weight → policy
    "input95": "stram",                             # kv_cache → policy
    "Coalesced_memloc_split_1_sg0000": "hbm",       # other → policy
}
```

This seeds **`TensorResidency`**, then [`_bootstrap_near_memory_homes`](engine.py) installs chip-wide and per-core near-memory homes at **t = 0**:

```python
residency["linear_weight_12"] = TensorResidency(
    home_level="ltram",
    resident_level="ltram",
    initialized_at_home=True,
)
```

**Decode assumption:** persistent tensors are already in their home before the profiled window (compile / model load). Trace replay only charges **home → SBUF** reloads. HBM-homed tensors are pre-installed via [`_seed_home_allocations`](engine.py). SBUF/PSUM are never pre-seeded.

See [`docs/PLACEMENT_AND_EVICTION.md`](../../../docs/PLACEMENT_AND_EVICTION.md) for area budget and spill details.

---

## 5. Runtime state during replay

While [`run_simulation`](engine.py) (L48) walks the trace, it mutates three structures alongside **`residency`**.

**Code:** pools created L61–65 · `fast_buffers` L67 · state types in [`residency.py`](residency.py).

### `TensorResidency` — per tensor

Defined [`residency.py`](residency.py) L7–11:

```python
@dataclass
class TensorResidency:
    home_level: str
    resident_level: str | None = None
    last_home_touch_ns: float | None = None   # retention timer at home
    corrupt: bool = False                      # home data stale → reload from HBM
    initialized_at_home: bool = False          # True after t=0 bootstrap
```

After a load access charges `ltram→sbuf`:

```python
residency["linear_weight_12"] = TensorResidency(
    home_level="ltram",        # unchanged
    resident_level="sbuf",
    last_home_touch_ns=1205000.0,
    corrupt=False,
)
```

Updated by [`_handle_access`](engine.py) (L232) · reset on wipe in [`_handle_kernel_boundary`](engine.py) (L226–228).

### `FastBufferState` — per core, per fast level

Defined [`residency.py`](residency.py) L15–23. Cleared in `_handle_kernel_boundary` (L223–225) via `FastBufferState.clear()` (L21).

```python
fast_buffers = {
    0: {
        "sbuf": FastBufferState(
            occupants={"linear_weight_12": 65536, "input95": 131072},
            used_bytes=196608,
        ),
    },
}
# after kernel_end on sbuf: clear() → occupants={}, used_bytes=0
```

### `LevelPoolState` — chip-wide pools

Defined [`residency.py`](residency.py) L27–44. Used from [`_install_in_fast_buffer`](engine.py) (L368) and [`_seed_home_allocations`](engine.py) (L165).

Tracks LtRAM/HBM capacity and occupants:

```python
pools = {
    "ltram": LevelPoolState(
        capacity_bytes=2_000_000_000,
        used_bytes=12582912,
        occupants={"linear_weight_12": 4194304, "linear_weight_13": 4194304},
    ),
    "hbm": LevelPoolState(capacity_bytes=..., used_bytes=..., occupants={...}),
}
```

SBUF cache eviction (when full) drops one occupant **without writeback traffic** — `resident_level` returns to `home_level`. **Home-tier chip pools never evict**; placement spill reserves capacity up front.

### Near-memory homes at t = 0 (decode)

For decode traces, [`_bootstrap_near_memory_homes`](engine.py) reserves LtRAM/StRAM (and similar) before the event loop — no staging transfers are charged. Weights and other homed tensors are assumed programmed into near memory before inference profiling starts.

After a **kernel wipe**, `resident_level` returns to `home_level`; the next trace access reloads **home → SBUF** (e.g. `ltram → sbuf` for weights, not HBM).

---

## 6. Event replay loop

[`run_simulation(trace, hierarchy, policy)`](engine.py) (L48–109):

```48:109:src/dmsim/sim/engine.py
def run_simulation(...) -> SimulationResult:
    homes = assign_home_levels(trace.tensors, hierarchy, policy)
    ...
    for event in parsed:
        ...
        if isinstance(event, KernelBoundaryEvent):
            _handle_kernel_boundary(...)
        if isinstance(event, AccessEvent):
            _handle_access(...)
    return result
```

Step-by-step:

```text
1. homes ← assign_home_levels()          [placement.py L18, engine.py L54]
2. Build residency, pools, fast_buffers  [engine.py L56–67]
3. _seed_home_allocations()              [engine.py L68, fn L165]
4. For each event sorted by t_ns:        [schema.py parsed_events L75]
       gap?  → _apply_refresh_energy_between()  [engine.py L85, fn L112]
       kernel_end? → _handle_kernel_boundary()  [engine.py L95, fn L209]
       access?     → _handle_access()           [engine.py L98, fn L232]
5. Return SimulationResult               [engine.py L26]
```

### Kernel boundaries

A **kernel wipe** happens when the trace contains **`kernel_end`** before the next access — not inferred from time gaps alone.

| Step | Code |
|------|------|
| Ingest emits events from `layer_summary` | [`neuron_json_ingest.py`](../trace/neuron_json_ingest.py) `_kernel_events_from_layers` (L688) |
| Hierarchy lists wiped levels | `kernel.wipe_levels_on_boundary` in hierarchy YAML → `KernelConfig` |
| Engine handles `kernel_end` | [`engine.py`](engine.py) `_handle_kernel_boundary` (L209) |

On `kernel_end` only ([`engine.py`](engine.py) `_handle_kernel_boundary` · `_kernel_wipe_cores`):
   - Clears `FastBufferState` for **wipe_levels_on_boundary** tiers on affected core(s) (default `psum`, `sbuf` — **not** StRAM/LtRAM)
   - Sets `resident_level = home_level` for tensors on those same core(s) when resident was a wiped tier (`TensorRecord.core_id`, default `0`)
   - Increments `kernel_wipes` — **no bytes charged** (no implicit flush)

**Core scope:**

| `kernel_end.core_id` | Buffers + residency |
|----------------------|---------------------|
| **Set** | That NeuronCore only |
| **`null`** | Every core in `fast_buffers` (single-core ingest); fallback core `0` if none |

[`merge_traces`](../trace/neuron_json_ingest.py) stamps each core’s `kernel_end` with that core’s id. [`_kernel_events_from_layers`](../trace/neuron_json_ingest.py) (single-core ingest) omits `core_id` → chip-wide wipe among cores that have run.

**Test:** [`tests/test_sim.py`](../../tests/test_sim.py) `test_kernel_wipe_scoped_to_core`.

If the profile has **no** `layer_summary`, there are no kernel events → SBUF is never reset (usually too optimistic on reload cost).

**Example time-ordered slice:**

```json
[
  { "type": "kernel_end", "t_ns": 1000000.0, "kernel_id": 2 },
  { "type": "access", "t_ns": 1205000.0, "tensor_id": "linear_weight_12",
    "op": "read", "bytes": 65536, "target_level": "sbuf", "core_id": 0 },
  { "type": "access", "t_ns": 1206000.0, "tensor_id": "linear_weight_12",
    "op": "read", "bytes": 65536, "target_level": "sbuf", "core_id": 0 },
  { "type": "access", "t_ns": 1210000.0, "tensor_id": "Coalesced_memloc_split_1_sg0000",
    "op": "write", "bytes": 65600, "target_level": "hbm", "core_id": 0 },
  { "type": "kernel_end", "t_ns": 1250000.0, "kernel_id": 3 }
]
```

### Typical load timeline

Tensor `linear_weight_12`, **home = `ltram`**:

| `t_ns` | Event | `resident_level` after | Charged |
|--------|--------|-------------------------|---------|
| 1.0e6 | `kernel_end` | `ltram` (was `sbuf`, wiped) | — |
| 1.205e6 | `read` 64 KiB → `sbuf` | `sbuf` | `ltram → sbuf` |
| 1.206e6 | `read` 64 KiB → `sbuf` (same tensor) | `sbuf` | local SBUF hit |
| 1.25e6 | `kernel_end` | `ltram` | — |

Without `kernel_end` between layers, the second read would still hit SBUF and skip the reload cost.

### Writeback timeline

Tensor with **home = `hbm`**, trace includes SBUF→HBM DMA (`op: write`, `target_level: hbm`):

| `t_ns` | Event | `resident_level` after | Charged |
|--------|--------|-------------------------|---------|
| 1.20e6 | `read` → `sbuf` | `sbuf` | `hbm → sbuf` |
| 1.21e6 | `write` 65 KiB → `hbm` | `hbm` | `sbuf → hbm`; `hbm_write_bytes += 65600` |

Even if `resident` was already `hbm` before the write, [`_source_level_for_access`](engine.py) treats off-chip **writes** as **`sbuf → target`** (Neuron SBUF→HBM flush semantics).

**Home** never changes; only **resident** tracks the last simulated location.

---

## 7. How transfer cost is computed

[`transfer.py`](transfer.py) defines hop routing and per-link cost; [`engine.py`](engine.py) `_charge_path` walks the hop list and updates `SimulationResult`.

### How hops are obtained

On an interconnect move (`source_level != target`), `_charge_path` calls [`path_between`](transfer.py), which returns **one direct hop** `(source, dest)` regardless of other tiers enabled in the hierarchy. Writebacks (`sbuf → hbm`), loads (`ltram → sbuf`, `hbm → sbuf`), and reloads all use the same rule.

### Chip time (per core)

| Metric | Aggregation |
|--------|-------------|
| **`time_by_core_ns[c]`** | **Sum** of hop + local access latencies on core `c` (`event.core_id`) |
| **`total_time_ns`** | **`max(time_by_core_ns)`** — worst-case core (cores assumed parallel) |
| **`latency_by_level_ns`** | **Sum chip-wide** — accounting breakdown, not wall-clock |
| **`total_energy_pJ`** | **Sum chip-wide** |
| **`hbm_*_bytes`, `corrupt_accesses`** | **Sum chip-wide** |

Trace `t_ns` is used for **event order** and **retention/refresh** gaps, not for adding latency into `total_time_ns`.

### Bandwidth (DMA vs on-chip)

Device tech `max_bandwidth_GBs` and legacy `links_GBs` are **not** used for transfer time.

`dma_bandwidth_GBs` and `on_chip_bandwidth_GBs` in hierarchy YAML are **per NeuronCore** (each hop’s `nbytes/BW` uses that core’s rate).

| Level class | Levels | Per-core hop bandwidth when paired with |
|-------------|--------|--------------------------------|
| **on_chip** | PSUM, SBUF, StRAM | Another on-chip level → **10 000 GB/s** |
| **off_chip** | HBM, LtRAM | Any hop touching off-chip → **368 GB/s** (DMA) |

Set in hierarchy YAML under `interconnect:` (`dma_bandwidth_GBs`, `on_chip_bandwidth_GBs`, `level_domain`).

```python
# stram → sbuf: both on_chip
link_bandwidth_GBs("stram", "sbuf")  # → 10000

# hbm → sbuf: off_chip + on_chip
link_bandwidth_GBs("hbm", "sbuf")    # → 368

# ltram → stram: off_chip + on_chip
link_bandwidth_GBs("ltram", "stram") # → 368
```

### Per-hop and local cost

| Function | When used |
|----------|-----------|
| `transfer_latency_ns` / `transfer_energy_pJ` | **One** adjacent hop in `_charge_path` |
| `_charge_local_access` → `access_latency_ns` / `access_energy_pJ` | SBUF scratch hits, local at home, **StRAM direct read** |

Single-hop latency:

```text
read_latency(source) + nbytes / bandwidth + write_latency(dest)
```

`bandwidth` from [`ResolvedHierarchy.link_bandwidth_GBs`](../config/models.py) using the table above.

### `_charge_path` accounting

For **each** hop in `path_between(source, dest)` (always zero or one direct edge):

1. Add that hop’s latency to **`time_by_core_ns[core_id]`** and energy to **`total_energy_pJ`**
2. Increment `transfers_by_hop["stram->sbuf"]` (etc.)
3. Split 50/50 into `latency_by_level_ns` / `energy_by_level_pJ`
4. If the hop touches **hbm**: update `hbm_read_bytes` / `hbm_write_bytes`

### Retention and `_touch_home`

**Retention** applies to the **home** tier (e.g. StRAM eDRAM with `retention_s` in the tech spec). SBUF/PSUM are scratch buffers wiped on `kernel_end`.

| Situation | Retention check? | `_touch_home`? |
|-----------|------------------|----------------|
| **SBUF scratch hit** (`resident == target == sbuf`, home ≠ sbuf) | No — serving cached scratch | No |
| **Load from home** (`source == home` on interconnect) | Yes | Yes (home was read) |
| **Local hit at home** | Yes | Yes |
| **Corrupt reload** (HBM → home) | Yes | Yes |

`_touch_home` only updates `last_home_touch_ns` when the persistent **home** copy is accessed, not when an access only touches SBUF.

---

## 8. Processing an access (`_handle_access`)

Core question: **reload from home, or cheap hit at target?**

**Code:** [`engine.py`](engine.py) `_handle_access` · `_source_level_for_access` · `_charge_path`.

Given:

```json
{
  "type": "access",
  "t_ns": 1205000.0,
  "tensor_id": "linear_weight_12",
  "op": "read",
  "bytes": 65536,
  "target_level": "sbuf",
  "core_id": 0
}
```

### Steps

| Step | Behavior |
|------|----------|
| Lookup tensor / residency | `TensorResidency` for `tensor_id` |
| Scratch hit? | `resident == target != home` → skip retention; local access only |
| **Source level** | [`_source_level_for_access`](engine.py): reads use `resident`; **off-chip writes** use `sbuf` |
| Retention | [`_check_retention_expired`](engine.py) vs trace `t_ns` when not scratch hit |
| Corrupt reload | `deepest → home`, then continue |
| **StRAM direct read?** | [`_is_direct_stram_read`](engine.py): home+resident at `stram`, read to SBUF → `_charge_local_access(stram)`, return |
| Transfer vs local | `_charge_path` if `source != target`; else local read or **omit same-level write** |
| Touch home | [`_touch_home`](engine.py) when home tier is read on interconnect or StRAM direct read |

**Direction** is `(source) → target`. Loads use `resident` (or home after wipe). Trace **`write` to an off-chip `target_level`** is modeled as **`sbuf → target`** (SBUF flush), even when `resident_level` is already at home. On-chip writes (`target_level: sbuf`) still use `resident`.

### Flow diagram

```text
                    ┌─────────────────────────┐
                    │ Retention expired?      │
                    └───────────┬─────────────┘
                          yes   │   no
                    ┌───────────▼───────────┐
                    │ deepest → home reload │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ StRAM direct read?    │
                    └───────────┬───────────┘
                          yes   │   no
                    ┌───────────▼──┐          │
                    │ local stram  │          │
                    └──────────────┘          │
                                ┌─────────────▼─────────────┐
                                │ source_level != target ?  │
                                └─────────┬─────────┬───────┘
                                     yes  │         │ no
                           ┌──────────────▼──┐  ┌───▼──────────────┐
                           │ _charge_path    │  │ op == write?     │
                           │ install target  │  └───┬──────────┬───┘
                           └─────────────────┘  yes │          │ no
                                              ┌─────▼──┐  ┌────▼─────────┐
                                              │ omit   │  │ local access │
                                              └────────┘  └──────────────┘
```

### What writebacks are and are not included

**Included** when the trace has them (e.g. ingest **SBUF → HBM** → `op: write`, `target_level: hbm`):

- Charged as **one direct hop** **`sbuf → hbm`** via `_source_level_for_access`
- Coalesced / gather buffers are usually category **`other`**, not a separate output class

**Not included** (not synthesized):

- SBUF cache eviction (drop only, no flush cost)
- Implicit flush on `kernel_end` wipe
- On-chip traffic missing from the Neuron profile

---

## 9. Output: `SimulationResult`

Defined [`engine.py`](engine.py) L26–45. Returned by `run_simulation` (L109). Printed by [`cli.py`](../cli.py) `run` command.

| Field | Meaning |
|--------|---------|
| `total_time_ns` | Worst-core time = `max(time_by_core_ns)` |
| `time_by_core_ns` | Per-core sum of hop + local latencies (ns) |
| `worst_core_id` | Core id that set `total_time_ns` (report only) |
| `total_energy_pJ` | Summed over all cores and events |
| `hbm_read_bytes` / `hbm_write_bytes` | Bytes on hops from/to **hbm** only |
| `hbm_traffic_bytes` | property: read + write |
| `transfers_by_hop` | Count per logical hop key (`ltram->sbuf`) |
| `energy_by_level_pJ`, `latency_by_level_ns` | Per-tier breakdown |
| `kernel_wipes`, `retention_evictions`, `corrupt_accesses` | Event counters |
| `refresh_energy_pJ`, `refresh_cycles_by_level` | Background refresh between trace timestamps |

Example:

```python
SimulationResult(
    hierarchy_name="trainium2_diff_mem_50sbuf_25hbm",
    policy_name="decode_tiered",
    trace_workload="llama32_decode_4core",
    total_time_ns=4.82e8,
    total_energy_pJ=1.03e12,
    hbm_read_bytes=1_500_000_000,
    hbm_write_bytes=320_000_000,
    transfers_by_hop={
        "ltram->sbuf": 412,
        # stram->sbuf absent when StRAM direct read applies
        "hbm->sbuf": 45,
        "sbuf->hbm": 168,
    },
    energy_by_level_pJ={"sbuf": 2.1e11, "ltram": 4.5e11, "hbm": 3.2e11},
    latency_by_level_ns={"sbuf": 1.2e8, "ltram": 2.0e8},
    kernel_wipes=32,
    retention_evictions=0,
    corrupt_accesses=0,
)
```

---

## 10. Limitations and comparisons

### Baseline vs candidate

| | Baseline | Candidate (example) |
|---|----------|------------------------|
| Stack | PSUM → SBUF → HBM | + StRAM, LtRAM |
| Policy | all → HBM | weight→ltram, kv→stram, … |
| Typical hop after wipe | `hbm→sbuf` | `ltram→sbuf`; StRAM-homed KV → **local at stram** (not `stram→sbuf`) |
| HBM traffic | bytes on hops that touch HBM | often lower |

### What is *not* modeled

- **Compute** overlapped with memory (only interconnect + local access latencies are summed)
- **Shared HBM port** contention when all cores hit HBM at once
- Implicit writebacks on `kernel_end` (SBUF/PSUM cleared without flush cost)
- SBUF wiped each traced `kernel_end` → may **over-reload** vs hardware if profile missed writebacks
- Imperfect tensor classification (`other` bucket for many DMA variables)

---

## 11. Module reference

Per-file index with line anchors. Paths relative to `src/dmsim/`.

### [`sim/residency.py`](residency.py)

| Symbol | Lines | Role |
|--------|-------|------|
| `TensorResidency` | L7–11 | `home_level`, `resident_level`, retention fields |
| `FastBufferState` | L15–23 | Per-core SBUF/PSUM occupancy; `clear()` |
| `LevelPoolState` | L27–44 | Chip-wide pool; `can_fit`, `install`, `remove` |

### [`sim/transfer.py`](transfer.py)

| Symbol | Lines | Role |
|--------|-------|------|
| `level_order` | — | Enabled level ids in hierarchy order |
| `hops_between` | — | Direct hop `source → dest` |
| `path_between` | — | Same as `hops_between` (used by `_charge_path`) |
| `physical_hops_between` | — | Alias of `hops_between` |
| `transfer_latency_ns` | — | Single direct link latency |
| `transfer_energy_pJ` | — | Single direct link energy |
| `transfer_latency_between_levels` | — | Direct transfer latency |
| `transfer_energy_between_levels` | — | Direct transfer energy |
| `access_latency_ns` | — | Local access latency |
| `access_energy_pJ` | — | Local access energy |

### [`sim/engine.py`](engine.py)

| Symbol | Lines | Role |
|--------|-------|------|
| `SimulationResult` | — | Output dataclass (`time_by_core_ns`, `worst_core_id`, …) |
| `run_simulation` | — | Placement + event loop; `total_time_ns = max` per-core sums |
| `_add_core_latency` | — | Accumulate latency into `time_by_core_ns[core]` |
| `_apply_refresh_energy_between` | — | Volatile tier refresh between trace timestamps |
| `_seed_home_allocations` | — | Install deepest-home tensors at t=0 |
| `_bootstrap_near_memory_homes` | — | Near-memory homes at t=0 (decode) |
| `_source_level_for_access` | — | `sbuf` source for off-chip writes |
| `_handle_kernel_boundary` | — | Per-core SBUF/PSUM wipe on `kernel_end` |
| `_kernel_wipe_cores` | — | Resolve affected NeuronCore ids from event + fast_buffers |
| `_tensor_core_id` | — | Tensor → NeuronCore for residency wipe scope |
| `_handle_access` | — | One access (see §8) |
| `_handle_access` | — | Per-event routing, StRAM direct read, same-level write omit |
| `_is_direct_stram_read` | — | StRAM home+resident → local read, no hop |
| `_charge_local_access` | — | Line-granularity local latency/energy |
| `_source_level_for_access` | — | Writeback = SBUF source |
| `_charge_path` | — | `path_between` + per-core latency + HBM bytes |
| `_install_in_fast_buffer` | — | SBUF occupancy; evict without writeback |
| `_evict_from_fast_buffer` | — | Drop one SBUF occupant; resident ← home |
| `_check_retention_expired` | — | StRAM-style retention (trace `t_ns`) |
| `_touch_home` | — | Update `last_home_touch_ns` |
| `_deepest_enabled` | — | Deepest enabled tier (corrupt reload from) |

### [`policies/placement.py`](../policies/placement.py)

| Symbol | Lines | Role |
|--------|-------|------|
| `assign_home_levels` | L19 | Policy map + capacity spill → `homes` dict |
| `_fallback_level` | L41 | Disabled tier → policy fallback chain |
| `_resolve_spill_target` | L56 | Walk `fallback_by_level` until enabled target |
| `_pick_spill_victim` | L88 | Choose victim by `spill_victim_order` + access counts |
| `_enforce_capacities` | L74 | Spill orchestration |
| `_enforce_chip_pool` | L99 | LtRAM/HBM spill |
| `_enforce_per_core_pool` | L119 | SBUF/StRAM spill |

### [`trace/schema.py`](../trace/schema.py)

| Symbol | Lines | Role |
|--------|-------|------|
| `TensorCategory` | L11 | Enum for policy lookup |
| `TensorRecord` | L19 | Trace tensor catalog entry |
| `AccessEvent` | L33 | One charged memory access |
| `KernelBoundaryEvent` | L43 | Kernel start/end marker |
| `Trace.parsed_events` | L75 | Parse + sort events for replay |
| `load_trace` | L91 | Load ingested JSON |

### [`trace/neuron_json_ingest.py`](../trace/neuron_json_ingest.py)

| Symbol | Lines | Role |
|--------|-------|------|
| `ingest_neuron_json_profile` | L156 | Main ingest entry |
| `_build_from_device_dma` | L425 | DMA records → tensors + accesses |
| `_map_dma_to_access` | L613 | Route → `op`, `target_level` |
| `_kernel_events_from_layers` | L688 | `layer_summary` → kernel events |
| `_merge_system_tensor_events` | L646 | Runtime tensor read/write events |

### [`config/loader.py`](../config/loader.py) · [`config/models.py`](../config/models.py)

| Symbol | File | Lines | Role |
|--------|------|-------|------|
| `load_policy` | loader | L45 | Policy YAML → `PolicyConfig` |
| `load_hierarchy` | loader | L62 | Hierarchy YAML → `ResolvedHierarchy` |
| `PolicyConfig` | models | L91 | `home_level_by_category`, `fallback_by_level`, `spill_victim_order`, `fallback_for()` |
| `Trace.access_counts` | schema | L90 | Access events per tensor for spill ordering |
| `ResolvedHierarchy` | models | L111 | Levels, links, kernel config |
| `link_bandwidth_GBs` | models | L145 | Effective hop bandwidth |

See [`docs/PLACEMENT_AND_EVICTION.md`](../../../docs/PLACEMENT_AND_EVICTION.md) for policy tables and area budget.
