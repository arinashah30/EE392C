#!/usr/bin/env bash
# LtRAM-only: 10/25/50/75% HBM area → LtRAM, best/worst spill. Default tech rram.
# Default trace: Llama DGE decode. Override with LLAMA_TRACE / QWEN_TRACE + --out-dir for Qwen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi
TECH="${1:-rram}"
shift || true
TRACE="${LLAMA_TRACE:-data/traces/llama32_1b_decode_4core_dge_kv.json}"
exec python3 scripts/run_memory_sweep.py --tier ltram --ltram-tech "$TECH" --trace "$TRACE" "$@"
