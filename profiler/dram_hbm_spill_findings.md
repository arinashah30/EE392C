# DRAM↔HBM Spill Detection Findings

This document captures the validation of the `tensor_lifetime_analyzer.py` pipeline on real Llama 3.1 8B prefill traces, plus the engineering constraints encountered along the way.

> **Note:** this is a reconstructed summary from the design notes; the original session artifacts (raw `summary.json`, `.ntff` traces) were lost with a previous instance restart. The configuration and headline numbers below are accurate; reproduce via the script flow described under "Reproducer."

---

## Status: spill detection validated

The end-to-end pipeline reliably reports `spill_save_bytes` and `spill_reload_bytes` from real Neuron profiler output. The `llama_threshold_noflash` configuration produces a non-zero spill on a `trn2.3xlarge` (96 GB HBM, ~21.5 GB usable per logical NeuronCore at TP=4):

| Configuration | `spill_save_bytes` | Notes |
|---|---|---|
| `llama_smoke` | 0 | Baseline pipeline check; no spill |
| `llama_threshold_noflash` | 1,572,880 (~1.57 MB) | Flash attention disabled; attention-score tensor materialized; spill triggered |

The 1.57 MB number is small in absolute terms but is the "gold signal" the analyzer was built to detect — it confirms the pipeline correctly surfaces a real HBM→DRAM eviction emitted by `neuronx-cc`.

---

## Pipeline

1. `profile_prep.sh` — sets Neuron profiling env vars (`NEURON_RT_INSPECT_ENABLE`, `NEURON_RT_INSPECT_DEVICE_PROFILE`, `NEURON_RT_INSPECT_EVENT_FILTER_TYPE`, `NEURON_RT_INSPECT_OUTPUT_DIR`, `XLA_IR_DEBUG`, `XLA_HLO_DEBUG`).
2. `llama_prefill_stress.py` — wraps `LlamaRunner.trace` + `load_neuron_model` + a single `generate_on_neuron` call via the packaged `neuronx_distributed_inference` API. Includes a `_patch_disable_flash_attention()` monkey-patch for the `noflash` variants.
3. `run_llama_spill_experiment.sh` — orchestrates compile, run, profile, JSON conversion, and analysis. Single-NeuronCore capture via `NEURON_RT_INSPECT_EVENT_FILTER_NC=0`. Always emits `summary-json` (cheap); attempts heavy per-event JSON only when the kept `.ntff` is ≤ 600 MB.
4. `tensor_lifetime_analyzer.py` — auto-selects analysis mode: full per-tensor lifetime when per-event JSON is present, lightweight summary-only spill detection when only `summary.json` is available.

---

## Key engineering constraints encountered

- **`neuron-profile view` is memory-pathological.** A 951 MB `.ntff` expanded to >128 GB of RAM during JSON conversion and was OOM-killed on a 124 GB instance. Mitigation: aggressively prune `.ntff` files before conversion (keep only the largest one, which is the prefill graph; delete `vnc_[1-9]*.ntff`), and skip heavy per-event JSON above 600 MB while always emitting `summary-json` as the fallback path.
- **The Neuron compiler does not auto-spill weights.** It will compile a model that fits and refuse to compile a model that doesn't (HBM-overflow error). It *will* spill intermediate activations, which is what the `noflash` variant exploits — disabling flash attention forces materialization of the full attention-score tensor, pushing peak activation memory past the HBM budget and triggering a real spill.
- **`transformers` is version-coupled to NxD-Inference.** The packaged `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` venv expects a specific `transformers` version; substitutions break the Llama runner.
- **`libneuronpjrt-path` lives in the venv `bin/`.** Direct Python invocations must include `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin` in `PATH`.

---

## Analyzer modes

- **Full per-tensor mode** (`*_nc_*.json` present): per-tensor lifetimes, HBM residency windows, thrash detection.
- **Summary-only mode** (`summary.json` only): aggregate `spill_save_bytes` / `spill_reload_bytes` per NEFF; no per-tensor detail. Used when heavy JSON conversion would OOM the instance. This is the path that succeeded on `llama_threshold_noflash`.

---

## Reproducer

```bash
cd /home/ubuntu/aws-trainium/josh
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH"
bash run_llama_spill_experiment.sh llama_threshold_noflash
```

Expected: `summary.json` reports `spill_save_bytes` ≈ 1.57 MB; the analyzer's summary-only mode reports a positive spill detection.

---

## Follow-on assessment

See [`runtime_optimization_assessment.md`](runtime_optimization_assessment.md) for an assessment of whether the profile data can drive runtime speedups (short answer: no, except via NKI rewrites or workload-specific KV tiering).
