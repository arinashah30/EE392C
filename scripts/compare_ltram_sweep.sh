#!/usr/bin/env bash
# LtRAM-only: 10/25/50/75% HBM area → LtRAM, best/worst spill. Default tech rram.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi
TECH="${1:-rram}"
shift || true
exec python3 scripts/run_memory_sweep.py --tier ltram --ltram-tech "$TECH" "$@"
