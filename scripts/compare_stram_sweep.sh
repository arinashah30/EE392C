#!/usr/bin/env bash
# StRAM-only: 10/25/50/75% SBUF area → StRAM, best/worst spill. Default tech edram_1t1c.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi
TECH="${1:-edram_1t1c}"
shift || true
exec python3 scripts/run_memory_sweep.py --tier stram --stram-tech "$TECH" "$@"
