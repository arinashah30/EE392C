# Cost model documentation plan

This plan defines how the **dmsim simulation cost model** is documented: hops, latency, energy, and HBM traffic. The goal is a path from **high-level replay semantics** down to **specific functions and data structures**.

## Audience and scope

**Audience:** anyone comparing hierarchies (`dmsim.cli compare`), debugging surprising sim numbers, or extending the simulator.

**In scope:**

- How a trace `AccessEvent` becomes charged hops, time, energy, and HBM bytes
- Definitions of `SimulationResult` metrics and how they aggregate
- Tech YAML and hierarchy YAML inputs that feed the formulas

**Out of scope (separate docs):**

- Trace ingest from Neuron profiles → [`../NEURON_PROFILE.md`](../NEURON_PROFILE.md)
- Placement, spill, eviction → [`../PLACEMENT_AND_EVICTION.md`](../PLACEMENT_AND_EVICTION.md)
- Full simulator tour → [`../../src/dmsim/sim/README.md`](../../src/dmsim/sim/README.md)

## Documentation structure

| File | Topic | Primary code |
|------|--------|--------------|
| [`README.md`](README.md) | Index, glossary, end-to-end flow | [`engine.py`](../../src/dmsim/sim/engine.py) `run_simulation` |
| [`01-hops.md`](01-hops.md) | Hop determination (`source`, `target`, routing) | [`transfer.py`](../../src/dmsim/sim/transfer.py), [`engine.py`](../../src/dmsim/sim/engine.py) `_handle_access` |
| [`02-latency.md`](02-latency.md) | Time metrics (`total_time_ns`, per-core, per-level) | [`transfer.py`](../../src/dmsim/sim/transfer.py), [`engine.py`](../../src/dmsim/sim/engine.py) `_charge_path` |
| [`03-energy.md`](03-energy.md) | Energy metrics (access, transfer, refresh) | [`transfer.py`](../../src/dmsim/sim/transfer.py), [`engine.py`](../../src/dmsim/sim/engine.py) `_apply_refresh_energy_between` |
| [`04-hbm-traffic.md`](04-hbm-traffic.md) | HBM read/write byte accounting | [`engine.py`](../../src/dmsim/sim/engine.py) `_charge_path` |

## Reading order

### High level → low level (conceptual)

1. [`README.md`](README.md) — what the simulator replays and what it outputs
2. [`01-hops.md`](01-hops.md) — when an interconnect move happens and which edge is charged
3. [`02-latency.md`](02-latency.md) — how each hop/local access becomes nanoseconds
4. [`03-energy.md`](03-energy.md) — parallel energy accounting
5. [`04-hbm-traffic.md`](04-hbm-traffic.md) — subset of hops that touch HBM

### Low level → high level (code walk)

1. [`transfer.py`](../../src/dmsim/sim/transfer.py) — pure cost functions (no trace state)
2. [`engine.py`](../../src/dmsim/sim/engine.py) `_handle_access` → `_charge_path` — residency decides source/target
3. [`SimulationResult`](../../src/dmsim/sim/engine.py) — aggregated outputs
4. [`dmsim.cli compare`](../../src/dmsim/cli.py) — user-facing comparison JSON

## Core design principles (to reflect in every doc)

1. **Direct hops only** — one logical edge per interconnect charge; no automatic walk through YAML `levels:` order. Multi-stage paths require multiple trace events.
2. **Residency drives source** — `TensorResidency.resident_level` vs `target_level` determines interconnect vs local access.
3. **Trace `t_ns` does not add wall time** — used for event order and refresh/retention gaps, not summed into `total_time_ns`.
4. **Per-core parallel time** — latency accumulates per `core_id`; `total_time_ns = max(time_by_core_ns)`.
5. **Chip-wide sums for accounting** — `latency_by_level_ns`, `energy_by_level_pJ`, HBM bytes sum across cores.
6. **StRAM direct read** — homed+resident at StRAM, trace read to SBUF → local at StRAM (`_is_direct_stram_read`), not `stram→sbuf`.
7. **Same-level writes omitted** — `source == target`, `op == write` → zero latency/energy.
8. **Kernel wipe is selective and per-core** — only `wipe_levels_on_boundary` tiers reset; StRAM/LtRAM persist by default. Fast-buffer clear and `resident_level` reset share the same NeuronCore scope (`kernel_end.core_id` or all active cores).

## Deliverables checklist

- [x] Plan (this file)
- [x] Index + glossary ([`README.md`](README.md))
- [x] Hops doc with source/target decision tree
- [x] Latency doc with formulas and metric table
- [x] Energy doc (transfer + local + refresh)
- [x] HBM traffic doc with counting rules and edge cases
- [x] Cross-links between docs and to source files
- [x] Example trace event + resulting state/metrics

## Future extensions (not written yet)

- Worked example notebook using `synthetic_decode.json`
- Diagram of ingest DMA → `AccessEvent` → sim charge
- FAQ for “why is LtRAM time similar to HBM?” (ties to residency + writeback dominance)
