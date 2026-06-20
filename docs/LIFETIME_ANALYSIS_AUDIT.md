# Lifetime Analysis Audit — `dmsim.trace.lifetime_analysis`

Audit of [`src/dmsim/trace/lifetime_analysis.py`](../src/dmsim/trace/lifetime_analysis.py) against the dmsim trace schema, simulator SBUF/kernel semantics, and live Llama/Qwen decode traces.

| Audit | Date | Scope |
|-------|------|-------|
| **Initial audit** | **20 June 2026** | Code review, synthetic reproducers, live 4-core DGE traces |

**Related:** [TENSOR_MAPPER_AUDIT.md](TENSOR_MAPPER_AUDIT.md), [sim/README.md](../src/dmsim/sim/README.md) (kernel wipe semantics), [schema.py](../src/dmsim/trace/schema.py).

**Verdict:** Core **session lifetime** and **SBUF stint** logic are sound for current ingested traces. **P1 fixed (20 June 2026):** chip-wide `kernel_end` kernel bookkeeping. **CLI + tests added.** Residual: StRAM direct-read SBUF gap (P2, documented).

---

## Changelog

| Date | Change |
|------|--------|
| 20 June 2026 | Initial audit |
| 20 June 2026 | P1 fix (`_kernel_end_cores`), single-pass event scan, median unification, `dmsim.cli lifetime`, `tests/test_lifetime_analysis.py` |

## 1. What the module does

| Metric | Definition | Used for |
|--------|------------|----------|
| **Session lifetime** | `last_access_t − first_access_t` over all `AccessEvent`s (read/write, any `target_level`) | Tiering intuition (KV vs activation vs weight) |
| **SBUF max stint** | Longest continuous residency after `read→sbuf` until per-core `kernel_end` wipe | Scratch vs near-memory placement |
| **Lifetime bins** | `point / micro / short / medium / long / persistent` | Aggregate histograms |
| **kernel_span_count** | Distinct `kernel_id`s active on the access core at touch time | Secondary; not used in summaries |

Entry point: `analyze_trace_lifetimes(trace) → LifetimeAnalysisResult`  
Serialization: `result_to_dict(result)`

**Not the same as:** `profiler/visualize_tensor_analysis.py` / `profiler/dram_hbm_spill_findings.md` — those describe an older Neuron-profiler spill pipeline (`tensor_lifetime_analyzer.py`, not in this repo).

---

## 2. Alignment with simulator semantics

### 2.1 Kernel-end SBUF wipe ✅

[`_compute_sbuf_stints`](../src/dmsim/trace/lifetime_analysis.py) matches [`engine._handle_kernel_boundary`](../src/dmsim/sim/engine.py):

- Install on `read` with `target_level == "sbuf"`.
- Clear all occupants on `kernel_end`.
- `kernel_end.core_id is None` → wipe **all cores** that have SBUF occupants (same as `_kernel_wipe_cores` when buffers exist).

Synthetic check: two-core trace with `core_id=None` `kernel_end` → both tensors get 50 ns stints.

### 2.2 StRAM direct-read gap ⚠️ (P2)

The engine **does not** install into SBUF for `_is_direct_stram_read` (tensor homed in StRAM, resident at home, read target SBUF). Lifetime analysis **always** counts `read→sbuf` as SBUF residency because it has no placement/home context.

Impact: trace-only analysis is correct for **baseline HBM** traces. After StRAM/LtRAM placement, SBUF stint stats may **overstate** scratch residency unless filtered or passed homes.

### 2.3 Access mix on live Llama trace

| op | target | count |
|----|--------|------:|
| read | sbuf | 462,052 |
| write | hbm | 134,396 |
| write | sbuf | 4,224 |

Session lifetime correctly includes HBM writebacks (KV spill path). SBUF stints correctly ignore `write→sbuf` (install is read-driven, matching engine).

---

## 3. Live validation (4-core DGE traces)

Traces: `data/traces/llama32_1b_decode_4core_dge_kv.json`, `data/traces/qwen1_5_moe_decode_4core_dge_v2.json`

| Check | Llama | Qwen |
|-------|------:|-----:|
| Tensors with access events | 1,236 | 3,002 |
| `lifetime == last − first` mismatches | 0 | 0 |
| `access_count` mismatches | 0 | 0 |
| `unknown` category | 0 | 0 |
| `kernel_end` with `core_id=None` | 0 | 0 |
| Trace span (ms) | 3,034 | 6,903 |
| SBUF stint tensors | 968 | 2,610 |

**Category session medians (multi-access only, ms):**

| Category | Llama | Qwen |
|----------|------:|-----:|
| kv_cache | 4.7 | 1.2 |
| weight | 6.8 | 12.2 |
| activation | 0.36 | 0.60 |

Ordering matches decode intuition: activations shortest, KV/weights span multiple layer kernels.

**Performance note:** ~25 s per 600k-event trace — full double pass over `parsed_events()` (main loop + SBUF pass). Consider single-pass merge.

---

## 4. Issues found

### P1 — `kernel_span_count` leaks when `kernel_end.core_id` is missing ✅ Fixed

**Location:** `analyze_trace_lifetimes`, kernel bookkeeping loop

**Was:** On `kernel_end`, `core` defaulted to `0` when `core_id is None`, so only `active_kernels[0]` was updated.

**Fix:** `_kernel_end_cores()` removes matching `kernel_id` from every core in `active_kernels` when `core_id is None` (mirrors `engine._kernel_wipe_cores`).

**Impact today:** None on current 4-core ingested traces (all `kernel_end` events carry `core_id`). Covered by `test_kernel_span_null_core_id_end`.

---

### P2 — SBUF stints ignore StRAM direct-read exception

See §2.2. Fix options: (a) document as trace-level upper bound; (b) accept optional `homes: dict[str,str]` and skip stint install when `_is_direct_stram_read` would apply.

---

### P3 — Summary median uses different method than category stats ✅ Fixed

Summary `lifetime_median_ms` / `session_lifetime_median_ms` now use `_median()` like category stats. Rank-based p10–p99 unchanged.

---

### P3 — No tests, no CLI integration ✅ Fixed

- `tests/test_lifetime_analysis.py` — 7 tests (bins, SBUF stints, kernel_span, empty trace, category filter, JSON)
- `python3 -m dmsim.cli lifetime --trace … [--output …]`

`result_to_dict` still emits both `session_by_category` and `by_category` (duplicate keys for backward compat).

---

### P4 — Naming / ecosystem confusion

| Artifact | Role |
|----------|------|
| `src/dmsim/trace/lifetime_analysis.py` | dmsim trace post-processor (this audit) |
| `profiler/visualize_tensor_analysis.py` | Legacy viz for `tensor_lifetime_analysis.json` (hot/warm/cold, roofline) |
| `profiler/dram_hbm_spill_findings.md` | External spill-detection pipeline notes |

Recommend a CLI hook: `python3 -m dmsim.cli lifetime --trace … --output results/lifetime.json`. **Implemented.**

---

## 5. Bin classifier spot-check

`classify_lifetime(lifetime_ns, trace_span_ns)`:

| lifetime | trace span | bin |
|----------|------------|-----|
| 0 | any | point |
| >0, <1 ms | any | micro |
| ≥50% of span (span ≥1 ms) | short trace | persistent |
| ≥1 s | any | persistent |

The **≥50% trace span → persistent** rule is intentional for single-decode-step captures where KV/weights touch most kernels.

---

## 6. Recommendations (priority order)

1. ~~**P1 fix**~~ — done.
2. ~~**Add tests**~~ — done.
3. ~~**CLI**~~ — done.
4. ~~**Single-pass**~~ — merged access + SBUF loops (one `parsed_events()` scan).
5. **P2 doc or homes arg** — clarify SBUF stint vs post-placement engine behavior.
6. ~~**Unify median**~~ — done.

---

## 7. Residual / acceptable

- Single-access tensors included in bin counts but excluded from category median stats (by design: `access_count > 1` filter).
- `kernel_start` ignored for lifetime (only `kernel_end` matters) — matches engine.
- Session lifetime is **trace touch span**, not hardware retention or allocator lifetime.
- No `hidden` category tensors on current traces (category comes from mapper).

---

## 8. Reproduce audit

```bash
cd /home/ubuntu/EE392C
PYTHONPATH=src python3 -c "
from pathlib import Path
from dmsim.trace.schema import load_trace
from dmsim.trace.lifetime_analysis import analyze_trace_lifetimes, result_to_dict
import json

for name in [
    'data/traces/llama32_1b_decode_4core_dge_kv.json',
    'data/traces/qwen1_5_moe_decode_4core_dge_v2.json',
]:
    r = analyze_trace_lifetimes(load_trace(Path(name)))
    print(name, r.summary.get('tensor_count'), r.summary.get('lifetime_median_ms'))
"

PYTHONPATH=src python3 -m dmsim.cli lifetime \
  --trace data/traces/llama32_1b_decode_4core_dge_kv.json \
  --output results/lifetime_llama.json

PYTHONPATH=src python3 -m pytest tests/test_lifetime_analysis.py -q
```
