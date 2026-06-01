# EE392C — Differentiated Memory Simulator (dmsim)

Trace-driven analytical simulator for exploring on-die **StRAM** (1T1C / 3T eDRAM) and **LtRAM** (RRAM / FeRAM) in an expanded AWS Trainium2 memory hierarchy.

## Memory hierarchy (per chip)

Closest to compute → farthest:

**PSUM → SBUF → StRAM → LtRAM → HBM**

- **PSUM / SBUF**: Trainium2 on-chip buffers (2 MiB / 28 MiB per NeuronCore). Wiped on each kernel boundary (matches hardware).
- **StRAM / LtRAM**: Research inserts (capacities in YAML are placeholders you can resize).
- **HBM**: 96 GiB per Trainium2 chip ([Trainium2 arch](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/about-neuron/arch/neuron-hardware/trainium2.html)).

### Instances


| Instance        | Chips | Sim scope                                                         |
| --------------- | ----- | ----------------------------------------------------------------- |
| `trn2.3xlarge`  | 1     | Default (`configs/instances/trn2_3xlarge.yaml`)                   |
| `trn2.48xlarge` | 16    | Set `instance: configs/instances/trn2_48xlarge.yaml` in hierarchy |


Simulation runs **per chip**; multi-chip studies scale HBM pools via instance config later.

## Quick start

```bash
cd /path/to/EE392C
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Single run
dmsim run \
  --hierarchy configs/hierarchy/trainium2_diff_mem.yaml \
  --policy configs/policies/decode_tiered.yaml \
  --trace data/traces/synthetic_decode.json

# Baseline vs differentiated (same trace)
dmsim compare \
  --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --candidate-hierarchy configs/hierarchy/trainium2_diff_mem.yaml \
  --baseline-policy configs/policies/baseline_hbm.yaml \
  --candidate-policy configs/policies/decode_tiered.yaml \
  --trace data/traces/synthetic_decode.json

pytest
```

## Updating technology metrics

Edit files under `configs/tech_specs/` (latency, energy, retention, bandwidth). Reference them from hierarchy levels via `tech: <name>`.

Swap StRAM technology:

```yaml
# configs/hierarchy/trainium2_diff_mem.yaml
- id: stram
  tech: edram_3t   # was edram_1t1c
```

Disable a level:

```yaml
- id: ltram
  enabled: false
```

## Neuron Explorer profiles — which format to use?

**Use the Neuron Explorer JSON export** (`profile.json` + per-core `*_nc_*_session_*.json`), not the raw NEFF/NTFF bundle, as your primary input.


|                     | JSON export                            | Raw NEFF/NTFF             |
| ------------------- | -------------------------------------- | ------------------------- |
| Parsing             | Standard `json`                        | Protobuf + NEFF toolchain |
| DMA / tensor detail | `dma[]`, `annotation`, `layer_summary` | NTFF (~40MB/core)         |
| Ingest              | `**dmsim ingest`**                     | Stub only                 |


Keep NEFF/NTFF for Neuron Explorer UI; run the simulator on ingested traces.

```bash
# Ingest all NeuronCores in the profile (default)
dmsim ingest \
  --profile-dir data/traces/neuron_profile_json_4-19 \
  --model-key 124050204400345 \
  --output data/traces/ingested_all_cores.json

# Full pipeline: ingest all cores → baseline vs StRAM/LtRAM (constant area)
dmsim pipeline \
  --profile-dir data/traces/neuron_profile_json_4-19 \
  --model-key 124050204400345 \
  --baseline-hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --candidate-hierarchy configs/hierarchy/trainium2_diff_mem.yaml \
  --output data/traces/sim_results_all_cores.json
```

**Constant-area tradeoffs** (`trainium2_diff_mem.yaml`): **StRAM is per NeuronCore** — each core's StRAM area is subtracted from that core's SBUF (not split across the chip). LtRAM remains per-chip and trades against HBM. Densities come from `configs/tech_specs/`.

Details: [docs/NEURON_PROFILE.md](docs/NEURON_PROFILE.md).

After ingest, quick **matplotlib** charts (tensor counts / bytes by category, top tensors by size):

```bash
pip install matplotlib
python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core.json
# PNGs under profiler/out/llama32_1b_decode_4core_viz/
```

## Trace format

```json
{
  "tensors": [
    { "id": "kv_cache", "name": "layer0.kv", "bytes": 8388608, "category": "kv_cache" }
  ],
  "events": [
    { "type": "kernel_start", "t_ns": 0, "kernel_id": 1, "core_id": 0 },
    { "type": "access", "t_ns": 100, "tensor_id": "kv_cache", "op": "read",
      "bytes": 1048576, "target_level": "sbuf", "core_id": 0 },
    { "type": "kernel_end", "t_ns": 500, "kernel_id": 1, "core_id": 0 }
  ]
}
```

Categories: `weight`, `kv_cache`, `hidden`, `activation`, `other` (auto-classified from names if omitted).

## Model behavior

- **Sequential analytical time**: event latencies sum on a single timeline (no compute/memory overlap yet).
- **Refresh (StRAM/HBM)**: background refresh energy is charged between trace events at `refresh_interval_s` (tech default or per-level override). The simulator assumes refresh is frequent enough that StRAM data does not expire.
- **Kernel boundaries**: `kernel_end` wipes PSUM/SBUF fast buffers and resets residency so the next access reloads from home.

## Project layout

```
configs/          # hierarchy, tech specs, policies, instances
data/traces/      # normalized workload traces
src/dmsim/        # simulator package
tests/
experiments/      # experiment descriptors
```

## Decode placement policy

`configs/policies/decode_tiered.yaml`:


| Category | Home level (differentiated) |
| -------- | --------------------------- |
| weight   | LtRAM                       |
| kv_cache | StRAM                       |
| hidden   | StRAM                       |


Baseline policy keeps all categories in HBM.