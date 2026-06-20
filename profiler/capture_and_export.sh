#!/usr/bin/env bash
#
# Profile Llama-3.2-1B-Instruct inference on trn2.3xlarge (4 NeuronCores)
# and export Neuron Explorer JSON for dmsim ingest.
#
# Defaults keep weights and compiled artifacts on /dev/shm (tmpfs).
# Inference uses LNC=1 + TP=4 so all 4 physical cores participate.
#
# Prereqs:
#   1. HF checkpoint at MODEL_PATH (or meta-llama/Llama-3.2-1B-Instruct in hub cache).
#   2. NXDI compiled model at TRACED (COMPILE=1 once; uses neuronx_distributed_inference).
#
# Usage:
#   ./capture_and_export.sh
#   ENABLE_DGE_NOTIFS=0 ./capture_and_export.sh   # disable if NRT overflow / timeout
#   PROMPT="..." ./capture_and_export.sh
#   COMPILE=1 ./capture_and_export.sh          # compile to /dev/shm then profile
#   SKIP_RUN=1 ./capture_and_export.sh       # re-export from existing NTFFs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_llama32_1b_trn2.py}"

# HF weights on tmpfs (download once, e.g. huggingface-cli download ... --local-dir ...)
MODEL_PATH="${MODEL_PATH:-/dev/shm/Llama-3.2-1B-Instruct}"
TRACED="${TRACED:-/dev/shm/traced_model/Llama-3.2-1B-Instruct-nxdi-lnc1-tp4-b1-ctx128-seq256}"
PROMPT="${PROMPT:-The capital of France is}"
OUT="${OUT:-$TRACED/profile}"

# trn2.3xlarge: 4 physical NeuronCores; LNC=1 => 4 logical; TP=4 shards across all.
LNC="${LNC:-1}"
TP_DEGREE="${TP_DEGREE:-4}"

if [[ "$LNC" != "1" || "$TP_DEGREE" != "4" ]]; then
    echo "[warn] LNC=$LNC TP_DEGREE=$TP_DEGREE — defaults LNC=1 TP=4 use all 4 cores on trn2.3xlarge" >&2
fi

mkdir -p "$(dirname "$TRACED")" "$OUT"

# shellcheck disable=SC1091
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:${PATH:-}"

# Must be set before any neuronx / torch_neuronx import (the Python runner also sets this).
export NEURON_LOGICAL_NC_CONFIG="$LNC"

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
export NEURON_RT_INSPECT_OUTPUT_DIR="$OUT"
# Do not set NEURON_RT_INSPECT_EVENT_FILTER_NC — capture all NeuronCores (0–3).

# Richer dynamic-DMA metadata (tensor names / routes in dma[]). On by default for ingest;
# may cause NRT_EXEC_SW_NQ_OVERFLOW on some runs — set ENABLE_DGE_NOTIFS=0 to disable.
if [[ "${ENABLE_DGE_NOTIFS:-1}" == "1" ]]; then
    export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1
fi

# Prefer tmpfs for HF hub cache when downloading via the runner.
export HF_HOME="${HF_HOME:-/dev/shm/huggingface}"

echo "=== capture_and_export (Llama-3.2-1B-Instruct) ==="
echo "  MODEL_PATH = $MODEL_PATH"
echo "  TRACED     = $TRACED"
echo "  PROMPT     = $PROMPT"
echo "  OUT        = $OUT"
echo "  LNC        = $LNC  (logical cores = $((4 / LNC)) on trn2.3xlarge)"
echo "  TP_DEGREE  = $TP_DEGREE"
echo "  DGE_NOTIFS = ${ENABLE_DGE_NOTIFS:-1}"
echo

if [[ "${COMPILE:-0}" == "1" ]]; then
    echo "[compile] compiling traced model (LNC=$LNC TP=$TP_DEGREE) ..."
    python "$RUNNER" compile \
        --model-path "$MODEL_PATH" \
        --compiled-model-path "$TRACED/" \
        --lnc "$LNC" \
        --tp-degree "$TP_DEGREE" \
        --clean-cache
    echo
elif [[ ! -d "$TRACED" ]]; then
    echo "[error] TRACED=$TRACED does not exist. Compile first, e.g.:" >&2
    echo "  COMPILE=1 $0" >&2
    echo "  # or:" >&2
    echo "  python $RUNNER compile --model-path \"$MODEL_PATH\" \\" >&2
    echo "      --compiled-model-path \"$TRACED/\" --lnc $LNC --tp-degree $TP_DEGREE --clean-cache" >&2
    exit 1
fi

if [[ "${SKIP_RUN:-0}" != "1" ]]; then
    echo "[1/2] running inference with profiler enabled (all NeuronCores) ..."
    python "$RUNNER" run \
        --model-path "$MODEL_PATH" \
        --compiled-model-path "$TRACED/" \
        --lnc "$LNC" \
        --tp-degree "$TP_DEGREE" \
        --prompt "$PROMPT"
else
    echo "[1/2] SKIP_RUN=1 — reusing existing NTFFs in $OUT"
fi

SESSION_DIR=$(find "$OUT" -mindepth 2 -maxdepth 3 -type d -name "[0-9]*" \
    -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')
if [[ -z "$SESSION_DIR" || ! -d "$SESSION_DIR" ]]; then
    echo "[error] could not locate inner session dir under $OUT" >&2
    exit 1
fi

NTFF_COUNT=$(find "$SESSION_DIR" -maxdepth 1 -name '*_vnc_*.ntff' 2>/dev/null | wc -l)
echo
echo "      session-dir = $SESSION_DIR"
echo "      ntff files  = $NTFF_COUNT (expect 4 for TP=4 on trn2.3xlarge)"
if [[ "$NTFF_COUNT" -lt 4 ]]; then
    echo "[warn] fewer than 4 per-core NTFFs — check LNC=$LNC TP_DEGREE=$TP_DEGREE" >&2
fi

echo
echo "[2/2] exporting Neuron Explorer JSON (system + per-NeuronCore device profiles) ..."
neuron-explorer view \
    -d "$OUT" \
    --output-format json \
    --disable-ui \
    --output-file "$OUT/profile.json" \
    --json-pretty-print

NC_JSON_COUNT=$(find "$OUT" -maxdepth 1 -name '*_nc_*_model_*.json' 2>/dev/null | wc -l)
echo "      per-core JSON files = $NC_JSON_COUNT (expect 4 for TP=4)"
if [[ "$NC_JSON_COUNT" -lt 4 ]]; then
    echo "[warn] fewer than 4 per-core JSON exports — check NTFF count and LNC/TP settings" >&2
fi

echo
echo "[done] artifacts under $OUT:"
echo "  NTFFs:       $SESSION_DIR/*_vnc_*.ntff"
echo "  System JSON: $OUT/profile.json"
echo "  Device JSON: $OUT/*_nc_*_model_*.json"
echo
echo "Ingest with dmsim:"
echo "  dmsim ingest --profile-dir $OUT --output data/traces/llama32_1b_ingested.json"
