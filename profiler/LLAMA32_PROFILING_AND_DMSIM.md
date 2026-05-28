# Llama 3.2 1B profiling and dmsim — session summary

End-to-end notes for **Llama-3.2-1B-Instruct** on **trn2.3xlarge** (4 NeuronCores, LNC=1, TP=4): capture Neuron profiles, ingest into **dmsim**, visualize, simulate.

See also: [NEURON_PROFILE.md](NEURON_PROFILE.md) (JSON format and DMA ingest details).

---

## Goal

1. Run inference on Trainium with all 4 cores.
2. Capture Neuron Explorer profiles (NTFF → JSON) under `/dev/shm`.
3. Ingest into dmsim with tensor categories (weights, KV cache, etc.).
4. Visualize and run the memory-hierarchy simulator.

---

## Working path: `capture_and_export.sh`

**This is the supported workflow.** It compiles (optional), runs inference with Neuron inspect enabled, and exports Explorer JSON.

It calls `profiler/run_llama32_1b_trn2.py`, which uses `**neuronx_distributed_inference.inference_demo`** (`NeuronLlamaForCausalLM`).

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH"
export NEURON_LOGICAL_NC_CONFIG=1
export HF_HOME=/dev/shm/huggingface

cd /home/ubuntu/EE392C/profiler

# First time (compile + profile + export JSON):
COMPILE=1 ./capture_and_export.sh

# After compile:
./capture_and_export.sh

# Custom prompt:
PROMPT="I believe the meaning of life is" ./capture_and_export.sh
```

**Environment variables** (optional overrides):


| Variable            | Default                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| `MODEL_PATH`        | `/dev/shm/Llama-3.2-1B-Instruct`                                                            |
| `TRACED`            | `/dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256`                |
| `OUT`               | `$TRACED/profile`                                                                           |
| `LNC`               | `1`                                                                                         |
| `TP_DEGREE`         | `4`                                                                                         |
| `COMPILE`           | `0` (set `COMPILE=1` to compile first)                                                      |
| `SKIP_RUN`          | `0` (set `SKIP_RUN=1` to re-export JSON from existing NTFFs)                                |
| `ENABLE_DGE_NOTIFS` | `0` (set `1` to set `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` for richer dynamic DMA metadata) |


**Artifacts:**


| Item                | Path                                                                          |
| ------------------- | ----------------------------------------------------------------------------- |
| HF weights          | `/dev/shm/Llama-3.2-1B-Instruct`                                              |
| NXDI compiled model | `/dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/` |
| Profile + JSON      | `$TRACED/profile/` (`profile.json`, `*_nc_*_model_*.json`, NTFFs)             |


Set `NEURON_LOGICAL_NC_CONFIG=1` before any Neuron import (the script and runner both do this). Compile clears stale `/tmp/nxd_model` via `--clean-cache`.

### Manual equivalent (what the script runs)

```bash
python run_llama32_1b_trn2.py compile \
  --model-path /dev/shm/Llama-3.2-1B-Instruct \
  --compiled-model-path /dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/ \
  --lnc 1 --tp-degree 4 --clean-cache

python run_llama32_1b_trn2.py run \
  --model-path /dev/shm/Llama-3.2-1B-Instruct \
  --compiled-model-path /dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/ \
  --lnc 1 --tp-degree 4 \
  --prompt "The capital of France is"

neuron-explorer view -d "$TRACED/profile" --output-format json --disable-ui \
  --output-file "$TRACED/profile/profile.json" --json-pretty-print
```

### Export noise

Many `ERRO invalid DMA duration - transfer rate is invalid` lines from `neuron-explorer view` are usually harmless if the script exits 0 and `profile.json` is written.

### Note on earlier `llama2/LlamaRunner` attempts

An early version of `run_llama32_1b_trn2.py` wrapped `**llama2/LlamaRunner**` (HF + NxD examples). That path could compile but often failed on decode with `NRT_EXEC_OOB` on `token_generation_model`. **The current runner and `capture_and_export.sh` do not use that stack.**

---

## Why 12 JSON files, not 4?

Explorer exports **one device JSON per (NeuronCore × model execution)**.

Typical Llama 3.2 run: **3 NEFF phases × 4 cores = 12 files**:


| Model hash (suffix) | NEFF role                             |
| ------------------- | ------------------------------------- |
| `677185323126852`   | `context_encoding_model` (prefill)    |
| `446048307616134`   | `token_generation_model` (decode)     |
| `579692064539910`   | `layout_opt/graph.neff` (load/warmup) |


Pick phase at ingest with `--model-key` (substring match). For decode simulation, use `446048307616134`.

---

## dmsim ingest and tensor name mapper

Ingest builds `NeffTensorCatalog` from `neff_node[]` and runs `LLaMANameMapper` (`src/dmsim/trace/tensor_name_mapper.py`). You do not load files into the mapper manually.

```bash
cd /home/ubuntu/EE392C
export PYTHONPATH=src

PROFILE=/dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/profile

# All 4 cores, decode
python -m dmsim.cli ingest \
  --profile-dir "$PROFILE" \
  --model-key 446048307616134 \
  --output data/traces/llama32_1b_decode_4core.json

# Single core, prefill
python -m dmsim.cli ingest \
  --profile-dir "$PROFILE" \
  --nc 0 \
  --model-key 677185323126852 \
  --output data/traces/llama32_1b_prefill_nc0.json
```

**Inspect mapper output** (sort by numeric input index — not string order):

```bash
PYTHONPATH=src python3 <<PY
import re, json
from pathlib import Path
from dmsim.trace.neuron_json_ingest import discover_profile_dir, resolve_device_json
from dmsim.trace.tensor_name_mapper import NeffTensorCatalog

p = discover_profile_dir(Path("$PROFILE"))
dev = json.load(open(resolve_device_json(p, 0, "446048307616134")))
cat = NeffTensorCatalog(dev)

def idx(e):
    m = re.search(r"\d+", e.variable_name)
    return int(m.group()) if m else 0

for e in sorted(cat.entries(), key=idx):
    if e.category.value in ("weight", "kv_cache"):
        print(f"{e.variable_name:12} -> {e.semantic_name:40} {e.category.value}")
PY
```

---

## Visualization

### Ingested dmsim trace

```bash
pip install matplotlib
python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core.json
# → profiler/out/llama32_1b_decode_4core_viz/trace_summary.png
#   profiler/out/llama32_1b_decode_4core_viz/trace_top_tensors.png
```


| Panel  | Meaning                                            |
| ------ | -------------------------------------------------- |
| Left   | Tensor count by category                           |
| Middle | Sum of static `tensor.bytes` (NEFF catalog)        |
| Right  | Sum of `access` event bytes (traffic after ingest) |


### Raw Neuron device JSON (hardware)

```bash
python profiler/visualize_profiling.py \
  "$PROFILE/i-..._nc_0_model_446048307616134.json" \
  -o profiler/out/decode_nc0_hardware.png
```

### Neuron Explorer UI

```bash
neuron-explorer view -d "$PROFILE" -p 8081
# SSH: ssh -L 8081:localhost:8081 user@host
```

### dmsim simulation

```bash
python -m dmsim.cli run \
  --hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --policy configs/policies/decode_tiered.yaml \
  --trace data/traces/llama32_1b_decode_4core.json
```

---

## Why decode DMA rows are `unknown`

On Llama **decode** exports (`446048307616134`), almost every `dma[]` row looks like:

```json
{
  "variable": "unknown",
  "source": [["unknown"]],
  "dest": ["unknown"],
  "queue_type": "software_dynamic",
  "transfer_size": 256
}
```

This is **not a broken capture** — byte totals are correct. It is an **Explorer attribution gap**: the hardware records that a transfer happened and how many bytes moved, but Explorer cannot bind that NTFF record to a NEFF tensor symbol.

### Root cause: dynamic DMA (DGE) + missing DGE notifications at capture

Device JSON `warnings[]` on decode often includes:

> Missing additional dynamic DMA metadata … Try capturing with `--enable-dge-notifs` or `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`

`**capture_and_export.sh` does not set that flag by default** (to avoid `NRT_EXEC_SW_NQ_OVERFLOW` on DGE-heavy graphs). Without it, Explorer cannot correlate most `software_dynamic` / `hardware_dynamic` transfers to NEFF tensors → `variable=unknown`. **Re-exporting JSON does not help**; you must **re-capture** with DGE notifications enabled:

```bash
ENABLE_DGE_NOTIFS=1 ./capture_and_export.sh
# or: export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 before run
```

If the run hangs or fails with `NRT_EXEC_SW_NQ_OVERFLOW`, disable the flag and keep the dmsim ingest heuristic.

NXDI’s `token_generation_model` streams most HBM→SBUF traffic through **dynamic DMA queues**:


| `queue_type`       | Share of unknown-route decode rows (nc0) |
| ------------------ | ---------------------------------------- |
| `software_dynamic` | ~75%                                     |
| `hardware_dynamic` | ~24%                                     |
| `instruction`      | ~1%                                      |


These correspond to **Descriptor Generation Engine (DGE)** transfers — addresses resolved at runtime (weight tile streaming, gathers, indirect indexing), not fixed compile-time `WEIGHT→SB` / `INPUT→SB` descriptors. Neuron Explorer’s NTFF→NEFF correlation works well for static routes; for dynamic queues it often has no `variable`, `function`, or `subgraph` to emit.

Measured on this instance’s profile (nc0):


| Phase   | `unknown→unknown` rows        | Bytes on unknown route                | Rows with `function` | `annotation` (Tensor Viewer) |
| ------- | ----------------------------- | ------------------------------------- | -------------------- | ---------------------------- |
| Decode  | **100%** (151,316 / 151,351)  | 622.5 MB (= `summary.hbm_read_bytes`) | 0                    | 0                            |
| Prefill | **98.8%** (183,637 / 185,957) | 694 MB                                | 2,320                | 0                            |


Prefill is slightly better (a few thousand rows name compiler temps like `intermediate35, intermediate36` on `VIRTUAL→VIRTUAL` routes). Decode has essentially none of that metadata.

**What still works:** `summary.hbm_read_bytes`, `layer_summary` kernel timing, and `neff_node[]` static catalog (tensor names/sizes). **What’s missing:** per-transfer tensor identity in `dma[]`.

### What you cannot fix downstream

Ingest or a tensor name mapper cannot recover `layer_7.mlp.gate_proj.weight` from rows with no `variable`, no `function`, no `subgraph`, and no `read_shape`. Category-level approximations (below) are the practical downstream answer.

### Profile `warnings[]` — what they mean


| Warning                                           | Cause                                                          | What to do                                                                                                                                                                          |
| ------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Missing additional dynamic DMA metadata**       | Capture without DGE notifications                              | `ENABLE_DGE_NOTIFS=1 ./capture_and_export.sh` (re-capture, not re-export). If you get `NRT_EXEC_SW_NQ_OVERFLOW` or timeout, leave flag off and use dmsim `hbm_traffic_`* ingest.    |
| **NEFF missing compiler metrics**                 | NEFF built with older `neuronx-cc` than current tools          | `COMPILE=1` after upgrading compiler/DLAMI (`neuronx-cc` on this AMI was 2.25.x while tools are 2.30.x). Helps Explorer summaries / Tensor Viewer, not a substitute for DGE notifs. |
| **Inputs/weights size > measured HBM read/write** | Static NEFF tensor sizes vs incomplete dynamic-DMA measurement | Usually improves when DGE metadata is present; otherwise expected when most DMA stays `unknown`.                                                                                    |


### Partial workarounds (future / optional)


| Approach                                    | What you get                                         | Limitation                                     |
| ------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| dmsim ingest heuristic (implemented)        | Category-level HBM traffic (`hbm_traffic_weight`, …) | Not per-layer                                  |
| Time-align DMA with `layer_summary` windows | Traffic per kernel/layer                             | Still not per-tensor                           |
| `--include-system-events`                   | Sparse `nrt_tensor_read` events                      | Opaque runtime tensor IDs                      |
| Prefill JSON for ingest                     | Slightly more named routes                           | Wrong phase for decode sim; still ~99% unknown |


---

## Neuron SDK / Explorer upgrades (checked May 2026)

Recent Neuron releases improved Explorer UI and tooling, but **do not fix dynamic-DMA tensor naming in JSON exports** on this workload.

### Versions on this instance


| Component                               | Version                          |
| --------------------------------------- | -------------------------------- |
| `neuron-explorer` / `aws-neuronx-tools` | **2.30.10.0** (built 2026-05-18) |
| `aws-neuronx-runtime-lib`               | 2.32.31.0                        |
| `aws-neuronx-dkms` (driver)             | 2.28.0.0                         |
| `neuronx-cc` (compiler)                 | 2.25.3371.0                      |
| `torch-neuronx`                         | 2.9.0.2.14                       |
| `neuronx-distributed-inference`         | 0.10.17970                       |


`apt` reports **2.30.10.0 is the latest** `aws-neuronx-tools` candidate on this AMI.

### Relevant release notes


| Release    | Date         | DMA / Explorer relevance                                                                                                                              |
| ---------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2.29.1** | May 1, 2026  | “Fixed Neuron Explorer to display DMA information correctly” — **UI Device Profile pane display bug**, not dynamic-DMA→tensor correlation in exports. |
| **2.29.0** | Apr 9, 2026  | Explorer out of beta; System Trace HBM usage; Tensor Viewer (NEFF-level aggregates, not per-dynamic-DMA row).                                         |
| **2.28.0** | Feb 25, 2026 | Tensor Viewer + Database Viewer added.                                                                                                                |
| **2.30.0** | May 2026     | Region Highlighter, PCIe transfer viz, Dependency Chain Viewer — no mention of dynamic DMA attribution.                                               |


### Re-export test (same NTFFs, current Explorer)

Re-running `neuron-explorer view` on the existing capture with **2.30.10.0** produced **byte-identical** `dma[]` arrays (151,351 rows decode; same 99.98% unknown). Upgrading Explorer alone does not retroactively name dynamic DMA in JSON.

```bash
# Re-export only (no re-run); same attribution as before on our capture:
SKIP_RUN=1 ./capture_and_export.sh
```

A **fresh capture** after a full DLAMI/SDK upgrade (compiler + runtime + tools together) is worth retrying periodically, but as of 2.30.x on Llama 3.2 1B decode, expect the same pattern. For per-tensor DMA on dynamic workloads, file an AWS Neuron feature request for **runtime DGE→NEFF symbol correlation in NTFF/JSON exports**.

---

## DMA ingest issue and fix

### Symptom

First ingest of Llama decode produced only **4 access events** (activation) across 4 cores, while weights had large static sizes but **no access traffic** in charts.

### Cause

Original ingest only kept DMA rows with explicit routes (`WEIGHT→SB`, `VIRTUAL→SB`, etc.) and dropped the ~99% unknown dynamic rows above. See [Why decode DMA rows are `unknown](#why-decode-dma-rows-are-unknown)`.

### Fix (`src/dmsim/trace/neuron_json_ingest.py`)

For unknown routes on `software_dynamic` / `hardware_dynamic` / `instruction` queues:

1. Treat as **HBM → SBUF read** (`target_level: sbuf`).
2. Split bytes across synthetic tensors `hbm_traffic_weight`, `hbm_traffic_kv_cache`, `hbm_traffic_other`, … using NEFF catalog size ratios.

After fix, 4-core decode ingest shows ~~1.5 GB weight, ~1.0 GB other, ~16 MB kv_cache access bytes (~~2.5 GB total ≈ 621 MB × 4).

**Re-ingest after updating ingest code:**

```bash
python -m dmsim.cli ingest \
  --profile-dir "$PROFILE" \
  --model-key 446048307616134 \
  --output data/traces/llama32_1b_decode_4core.json
```

Details: [NEURON_PROFILE.md](NEURON_PROFILE.md#dma-record-shape-abbreviated).

---

## Tensor mapper quirks (decode)

- Decode NEFF uses shapes like `[1 2 256 64]` for many `input3+` slots; mapper expects prefill-style KV `[1 128 2 64]`, so early inputs may classify as `other` in the catalog.
- Named weights in the catalog appear from ~`input19 `(`layer_0.attention.wq.weight`, etc.).
- `cat.entries()[:20]` is string-sorted (`input10` before `input3`); sort numerically as above.

---

## File map


| File                                    | Role                             |
| --------------------------------------- | -------------------------------- |
| `profiler/capture_and_export.sh`        | **Main capture + JSON export**   |
| `profiler/run_llama32_1b_trn2.py`       | NXDI compile/run wrapper         |
| `profiler/visualize_trace.py`           | Charts from ingested dmsim trace |
| `profiler/visualize_profiling.py`       | Charts from raw device JSON      |
| `profiler/NEURON_PROFILE.md`            | JSON format reference            |
| `src/dmsim/trace/neuron_json_ingest.py` | Profile JSON → dmsim trace       |
| `src/dmsim/trace/tensor_name_mapper.py` | `inputN` → semantic names        |


---

## Open limitations

1. Unattributed dynamic DMA (~100% on decode) cannot be tied to specific layers; synthetic `hbm_traffic_`* tensors approximate **category-level** HBM traffic.
2. HBM→SBUF loads are recorded with `target_level: sbuf` (destination), not `hbm`.
3. Skip layout-opt captures (`579692064539910`) for simulation; use decode or prefill keys.
4. Neuron SDK 2.29.1–2.30.x re-export does not improve dynamic-DMA naming for this capture; upstream fix requires Explorer/runtime correlation for `software_dynamic` / `hardware_dynamic` queues.

---

## Full pipeline (copy-paste)

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_LOGICAL_NC_CONFIG=1
export HF_HOME=/dev/shm/huggingface

cd /home/ubuntu/EE392C/profiler
COMPILE=1 ./capture_and_export.sh    # or ./capture_and_export.sh if already compiled

cd /home/ubuntu/EE392C
export PYTHONPATH=src
PROFILE=/dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256/profile

python -m dmsim.cli ingest \
  --profile-dir "$PROFILE" \
  --model-key 446048307616134 \
  --min-transfer-bytes 1 \
  --output data/traces/llama32_1b_decode_4core.json

python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core.json

python -m dmsim.cli run \
  --hierarchy configs/hierarchy/trainium2_baseline.yaml \
  --policy configs/policies/decode_tiered.yaml \
  --trace data/traces/llama32_1b_decode_4core.json
```

