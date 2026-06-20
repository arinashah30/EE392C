#!/usr/bin/env bash
# All memory techs: LtRAM (rram, feram) and StRAM (edram_1t1c, edram_3t) at 10–75%, best/worst spill.
# Runs both Llama and Qwen decode traces (DGE-ingested, tensor-mapper v2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

LLAMA_TRACE="${LLAMA_TRACE:-data/traces/llama32_1b_decode_4core_dge_kv.json}"
QWEN_TRACE="${QWEN_TRACE:-data/traces/qwen1_5_moe_decode_4core_dge_v2.json}"

echo "=== Llama 3.2 1B decode — all-tech sweep ==="
echo "    trace: ${LLAMA_TRACE}"
python3 scripts/run_memory_sweep.py --tier all-tech \
  --trace "${LLAMA_TRACE}" \
  --out-dir data/traces/memory_sweeps \
  "$@"

echo ""
echo "=== Qwen1.5-MoE-A2.7B decode — all-tech sweep ==="
echo "    trace: ${QWEN_TRACE}"
python3 scripts/run_memory_sweep.py --tier all-tech \
  --trace "${QWEN_TRACE}" \
  --out-dir data/traces/memory_sweeps_qwen \
  "$@"
