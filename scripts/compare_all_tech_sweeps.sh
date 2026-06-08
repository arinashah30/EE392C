#!/usr/bin/env bash
# All memory techs: LtRAM (rram, feram) and StRAM (edram_1t1c, edram_3t) at 10–75%, best/worst spill.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi
exec python3 scripts/run_memory_sweep.py --tier all-tech "$@"
