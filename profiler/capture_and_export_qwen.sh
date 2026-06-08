#!/usr/bin/env bash
#
# Profile Qwen1.5-MoE-A2.7B on trn2.3xlarge (4 NeuronCores) and export Neuron
# Explorer JSON for dmsim ingest. Same prompt / shape defaults as Llama capture.
#
# Prereqs:
#   1. HF checkpoint at MODEL_PATH (default /dev/shm/Qwen1.5-MoE-A2.7B).
#   2. Vendored NxD inference code in profiler/nxd_inference/ (no external tree).
#   3. Compiled model at TRACED (COMPILE=1 once).
#
# Usage:
#   ./capture_and_export_qwen.sh
#   ENABLE_DGE_NOTIFS=1 ./capture_and_export_qwen.sh   # recommended for DMA attribution
#   COMPILE=1 ./capture_and_export_qwen.sh
#   PROMPT="..." ./capture_and_export_qwen.sh
#   SKIP_RUN=1 ./capture_and_export_qwen.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_qwen_moe_trn2.py}"

SHM_BASE="/dev/shm"
MODEL_PATH="${MODEL_PATH:-/dev/shm/Qwen1.5-MoE-A2.7B}"
TRACED="${TRACED:-/dev/shm/traced_model/Qwen1.5-MoE-A2.7B-lnc1-tp4-b1-p128-s256-rtopk_softmax}"
PROMPT="${PROMPT:-The capital of France is}"
OUT="${OUT:-$TRACED/profile}"

if [[ "${ALLOW_DISK_ARTIFACTS:-0}" != "1" ]]; then
    for _path in "$TRACED" "$OUT"; do
        if [[ "$_path" != "${SHM_BASE}"* ]]; then
            echo "[error] TRACED/OUT must be under ${SHM_BASE}/ (got $_path)" >&2
            exit 1
        fi
    done
fi

mkdir -p \
    "${SHM_BASE}/traced_model" \
    "${SHM_BASE}/tmp" \
    "${SHM_BASE}/huggingface" \
    "${SHM_BASE}/torch" \
    "${SHM_BASE}/xla_cache"

export TMPDIR="${TMPDIR:-${SHM_BASE}/tmp}"
export HF_HOME="${HF_HOME:-${SHM_BASE}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${SHM_BASE}/torch}"
export XLA_CACHE_DIR="${XLA_CACHE_DIR:-${SHM_BASE}/xla_cache}"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$MODEL_PATH}"

LNC="${LNC:-1}"
TP_DEGREE="${TP_DEGREE:-4}"
ROUTER_KERNEL="${ROUTER_KERNEL:-topk_softmax}"
export BASE_COMPILE_WORK_DIR="${BASE_COMPILE_WORK_DIR:-${SHM_BASE}/nxd_model_compile-lnc${LNC}-tp${TP_DEGREE}}"

if [[ "$LNC" != "1" || "$TP_DEGREE" != "4" ]]; then
    echo "[warn] LNC=$LNC TP_DEGREE=$TP_DEGREE — defaults LNC=1 TP=4 use all 4 cores on trn2.3xlarge" >&2
fi

mkdir -p "$OUT"

# shellcheck disable=SC1091
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:${PATH:-}"

export NEURON_LOGICAL_NC_CONFIG="$LNC"

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
export NEURON_RT_INSPECT_OUTPUT_DIR="$OUT"

# Richer dynamic-DMA metadata (tensor names / routes in dma[]). Recommended for Qwen
# re-capture after unattributed profiles; may cause NRT_EXEC_SW_NQ_OVERFLOW on some runs.
if [[ "${ENABLE_DGE_NOTIFS:-1}" == "1" ]]; then
    export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1
fi

echo "=== capture_and_export_qwen (Qwen1.5-MoE-A2.7B) ==="
echo "  SHM compile   = $BASE_COMPILE_WORK_DIR"
echo "  TMPDIR        = $TMPDIR"
echo "  MODEL_PATH     = $MODEL_PATH"
echo "  TRACED         = $TRACED"
echo "  PROMPT         = $PROMPT"
echo "  OUT            = $OUT"
echo "  LNC            = $LNC"
echo "  TP_DEGREE      = $TP_DEGREE"
echo "  ROUTER_KERNEL  = $ROUTER_KERNEL"
echo "  DGE_NOTIFS     = ${ENABLE_DGE_NOTIFS:-1}"
echo

_traced_ready() {
    [[ -f "${TRACED%/}/neuron_config.json" ]] && [[ -f "${TRACED%/}/model.pt" ]]
}

if [[ "${COMPILE:-0}" == "1" ]]; then
    echo "[compile] clearing stale NEFF cache and traced dir ..."
    rm -rf "$BASE_COMPILE_WORK_DIR" "$TRACED"
    echo "[compile] tracing + compiling (LNC=$LNC TP=$TP_DEGREE) ..."
    python "$RUNNER" compile \
        --model-path "$MODEL_PATH" \
        --traced-model-path "$TRACED/" \
        --lnc "$LNC" \
        --tp-degree "$TP_DEGREE" \
        --router-kernel "$ROUTER_KERNEL"
    echo
elif ! _traced_ready; then
    echo "[error] No compiled model at TRACED=$TRACED" >&2
    echo "  (need neuron_config.json and model.pt; found only profile/ or empty dir)" >&2
    echo "  First compile (slow, once per LNC/TP/shape):" >&2
    echo "    COMPILE=1 $0" >&2
    exit 1
fi

if [[ "${SKIP_RUN:-0}" != "1" ]]; then
    echo "[1/2] running inference with profiler enabled (all NeuronCores) ..."
    python "$RUNNER" run \
        --model-path "$MODEL_PATH" \
        --traced-model-path "$TRACED/" \
        --lnc "$LNC" \
        --tp-degree "$TP_DEGREE" \
        --router-kernel "$ROUTER_KERNEL" \
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

NC_JSON_COUNT=$(find "$OUT" -maxdepth 1 -name '*_nc_*_session_*.json' 2>/dev/null | wc -l)
echo "      per-core JSON files = $NC_JSON_COUNT (expect 4 for TP=4)"
if [[ "$NC_JSON_COUNT" -lt 4 ]]; then
    echo "[warn] fewer than 4 per-core JSON exports — check NTFF count and LNC/TP settings" >&2
fi

echo
echo "[done] artifacts under $OUT:"
echo "  NTFFs:       $SESSION_DIR/*_vnc_*.ntff"
echo "  System JSON: $OUT/profile.json"
echo "  Device JSON: $OUT/*_nc_*_session_*.json"
echo
echo "Ingest with dmsim:"
echo "  python3 -m dmsim.cli ingest \\"
echo "    --profile-dir $OUT \\"
echo "    --model-key <NEFF_id_from_profile_name> \\"
echo "    --min-transfer-bytes 1 \\"
echo "    --skip-unattributed-dma \\"
echo "    --no-aggregate-dma \\"
echo "    --max-access-events 0 \\"
echo "    --output data/traces/qwen1_5_moe_trn2_3_4core_min1_no_unknown.json"
echo
echo "Find model-key:"
echo "  grep -o '\"profile_name\":\"[^\"]*\"' \"$OUT\"/*_nc_0*.json | head -1"
