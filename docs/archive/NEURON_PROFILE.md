# Neuron Explorer profile formats (EE392C example)

This document describes the two profile bundles in `data/` and how `dmsim ingest` uses them.

## Layout comparison

### JSON export — `data/traces/neuron_profile_json_4-19/`


| File                              | Role                                                                                |
| --------------------------------- | ----------------------------------------------------------------------------------- |
| `profile.json`                    | System trace: HBM usage samples, `trace_event` (runtime API), `device_profile_list` |
| `i-*_pid_*_nc_<N>_session_0.json` | Device trace for one NeuronCore (large, one line JSON)                              |


### Raw capture — `data/neuron_profile/1776672348756756905/`


| File                  | Role                                       |
| --------------------- | ------------------------------------------ |
| `neff_*_vnc_<N>.neff` | Compiled graph (binary)                    |
| `*_vnc_<N>.ntff`      | Device hardware trace (binary, ~40MB/core) |
| `ntrace.pb`           | System trace protobuf                      |
| `trace_info.pb`       | Index of paths + metadata                  |


The JSON export is produced from the same capture session (`trace_info.pb` references the same `ntff` paths).

## JSON: `profile.json` (system)

Top-level sections used by dmsim:

- `**system_profile_metadata**` — `first_ts_ns`, `last_ts_ns`, `hbm_capacity_bytes`, `ntff_version`
- `**device_profile_list**` — maps `nc_id` → per-core JSON filename; multiple entries if multiple NEFF hashes were captured
- `**trace_event**` (~76k in the example) — runtime events with `timestamp` (ns), `nc_idx`, `name`, `size`, optional `tensor_id`

Useful event names:


| `name`                                      | Meaning                                  |
| ------------------------------------------- | ---------------------------------------- |
| `nrt_tensor_read` / `nrt_tensor_write`      | Runtime tensor access                    |
| `dmem_buf_copyin` / `dmem_buf_copyout`      | Host/device copies                       |
| `nrt_dma_mem_alloc` / `nrt_dma_mem_dealloc` | HBM allocator                            |
| `kbl_exec_pre` / `kbl_exec_post`            | Kernel schedule boundaries (system view) |


Optional ingest flag: `--include-system-events` (off by default to avoid double-counting DMA).

## JSON: per-NeuronCore device file

Example: `i-0703a89c7c8d99cc1_pid_7508_nc_0_session_0.json` (~594MB)

Important top-level keys:


| Key                          | Content                                                                     |
| ---------------------------- | --------------------------------------------------------------------------- |
| `neff_node`                  | **Static NEFF I/O catalog** — `variable_name` (`input0`, …), `shape`, `size` (bytes), `type` (`IN`/`OUT`/`WEIGHT`). Fed to `NeffTensorCatalog` / tensor mapper. See [TENSOR_MAPPER_AUDIT.md](../docs/TENSOR_MAPPER_AUDIT.md). |
| `dma`                        | **Primary ingest source** — hundreds of thousands of DMA records            |
| `layer_summary`              | Per-kernel/layer timing (`start`, `end`, `name`, FLOPs) → kernel boundaries |
| `annotation`                 | Tensor Viewer warnings, e.g. `load_to_sbuf_dma_count`, `tensor_name`        |
| `profile_info`               | Links back to `ntff_filename` and NEFF id                                   |
| `summary` / `summary_groups` | Aggregated counters (`hbm_read_bytes`, `sbuf_read_bytes`, …)                |


### DMA record shape (abbreviated)

```json
{
  "variable": "identity_19733_sg0000",
  "transfer_size": 65536,
  "source": [["WEIGHT"]],
  "dest": ["SB"],
  "timestamp": 434995,
  "duration": 889,
  "subgraph": "sg00"
}
```

**Timestamp unit:** microseconds relative to device profile start (converted to ns in ingest).

### `neff_node` record shape (abbreviated)

Each row is one **compiled graph slot** (whole-tensor geometry at the NEFF boundary — not kernel tile size, not DMA chunk size):

```json
{
  "variable_name": "input36",
  "shape": "[2048 512]",
  "size": "2097152",
  "type": "IN"
}
```

**Viewing:** Neuron Explorer Tensor Viewer (`neuron-explorer view -d "$PROFILE"`), raw `jq` on device JSON, or `NeffTensorCatalog` for mapped semantic names — see [docs/TENSOR_MAPPER_AUDIT.md](../docs/TENSOR_MAPPER_AUDIT.md) §1 Tier C.

**Ingest usage:** `build_catalog(device)` → mapper assigns semantic names/categories → trace `tensors[]` static bytes; unattributed decode DMA split by catalog category ratios (`hbm_traffic_*`).

**Common routes in the example (nc0):**


| Route        | Count (approx.) | dmsim mapping                |
| ------------ | --------------- | ---------------------------- |
| `VIRTUAL→SB` | 190k            | HBM-like → SBUF read         |
| `REMOTE→SB`  | 170k            | HBM / collective → SBUF read |
| `WEIGHT→SB`  | 10k             | Weight → SBUF read           |
| `SB→VIRTUAL` | 31k             | SBUF → HBM write             |


**Llama / decode caveat:** Many captures label ~99% of DMA rows as `source=unknown`, `dest=unknown`, `variable=unknown` with `queue_type=software_dynamic` (plus `hardware_dynamic` / `instruction`). This reflects **dynamic DMA (DGE)** — runtime-resolved addresses that Explorer often cannot bind to NEFF tensor symbols. Byte totals in `summary.hbm_read_bytes` remain correct. `dmsim ingest` treats those queues as HBM→SBUF reads and splits unattributed bytes across synthetic `hbm_traffic_{category}` tensors using NEFF catalog size ratios. Re-exporting with Neuron Explorer 2.29.1–2.30.x does not change attribution on existing NTFFs for this workload (2.29.1 fixed UI display only). See [LLAMA32_PROFILING_AND_DMSIM.md](LLAMA32_PROFILING_AND_DMSIM.md#why-decode-dma-rows-are-unknown).


## Recommended workflow

1. On Trainium: capture with Neuron Explorer (NEFF+NTFF as today).
2. **Export JSON** into a single directory (same structure as `neuron_profile_json_4-19`).
3. Upload that directory to the repo or S3.
4. Run `dmsim ingest` → normalized trace → `dmsim run` / `dmsim compare`.

## Multiple NEFF / cores

The example has **two model hashes** (`1014347842275474`, `124050204400345`) × **4 NeuronCores**. Select one with:

```bash
dmsim ingest --profile-dir ... --nc 2 --model-key 124050204400345 ...
```

List cores: `python -c "from dmsim.trace.neuron_json_ingest import list_neuron_cores, discover_profile_dir; print(list_neuron_cores(discover_profile_dir('data/traces/neuron_profile_json_4-19')))"`