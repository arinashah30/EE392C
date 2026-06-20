#!/usr/bin/env bash
# Milestone 4 tiered hierarchy compares (4 iso-area configs × Llama + Qwen decode traces).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

LLAMA_TRACE="${LLAMA_TRACE:-data/traces/llama32_1b_decode_4core_dge_kv.json}"
QWEN_TRACE="${QWEN_TRACE:-data/traces/qwen1_5_moe_decode_4core_dge_v2.json}"
BASELINE_H="${BASELINE_H:-configs/hierarchy/trainium2_baseline.yaml}"
BASELINE_P="${BASELINE_P:-configs/policies/baseline_hbm.yaml}"
CANDIDATE_P="${CANDIDATE_P:-configs/policies/decode_tiered.yaml}"

CONFIGS=(
  "50sbuf_25hbm:configs/hierarchy/trainium2_diff_mem_50sbuf_25hbm.yaml"
  "25sbuf_25hbm:configs/hierarchy/trainium2_diff_mem_25sbuf_25hbm.yaml"
  "50sbuf_10hbm:configs/hierarchy/trainium2_diff_mem_50sbuf_10hbm.yaml"
  "25sbuf_50hbm:configs/hierarchy/trainium2_diff_mem_25sbuf_50hbm.yaml"
)

run_model() {
  local model="$1"
  local trace="$2"
  local out_dir="$3"
  local prefix="$4"
  mkdir -p "$out_dir"
  echo ""
  echo "=== M4 compares — ${model} (${trace}) ==="
  for entry in "${CONFIGS[@]}"; do
    local slug="${entry%%:*}"
    local hier="${entry#*:}"
    local out="${out_dir}/${prefix}_${slug}_tiered.json"
    echo "  ${slug} -> ${out}"
    PYTHONPATH=src python3 -m dmsim.cli compare \
      --trace "$trace" \
      --baseline-hierarchy "$BASELINE_H" \
      --candidate-hierarchy "$hier" \
      --baseline-policy "$BASELINE_P" \
      --candidate-policy "$CANDIDATE_P" \
      --output "$out"
  done
}

run_model "Llama 3.2 1B" "$LLAMA_TRACE" "results" "m4_llama"
run_model "Qwen1.5-MoE-A2.7B" "$QWEN_TRACE" "results/m5_qwen" "m4"

echo ""
echo "Done. M4 JSONs under results/ and results/m5_qwen/"
