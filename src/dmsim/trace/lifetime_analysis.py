"""Per-tensor lifetime analysis from dmsim trace access events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from dmsim.trace.schema import AccessEvent, KernelBoundaryEvent, Trace, TensorRecord


class LifetimeBin(str, Enum):
    """Lifetime categories for memory-placement intuition."""

    POINT = "point"  # single timestamp (lifetime == 0)
    MICRO = "micro"  # (0, 1 ms)
    SHORT = "short"  # [1 ms, 10 ms)
    MEDIUM = "medium"  # [10 ms, 100 ms) — typical KV window
    LONG = "long"  # [100 ms, 1 s)
    PERSISTENT = "persistent"  # >= 1 s or >= 50% of trace span


# Upper bounds in nanoseconds for fixed-width bins (exclusive except point).
_BIN_UPPER_NS: dict[LifetimeBin, float] = {
    LifetimeBin.POINT: 0.0,
    LifetimeBin.MICRO: 1e6,
    LifetimeBin.SHORT: 10e6,
    LifetimeBin.MEDIUM: 100e6,
    LifetimeBin.LONG: 1e9,
}

_BIN_ORDER = [
    LifetimeBin.POINT,
    LifetimeBin.MICRO,
    LifetimeBin.SHORT,
    LifetimeBin.MEDIUM,
    LifetimeBin.LONG,
    LifetimeBin.PERSISTENT,
]

_BIN_LABELS = {
    LifetimeBin.POINT: "Point (0 ns)",
    LifetimeBin.MICRO: "Micro (<1 ms)",
    LifetimeBin.SHORT: "Short (1–10 ms)",
    LifetimeBin.MEDIUM: "Medium (10–100 ms)",
    LifetimeBin.LONG: "Long (100 ms–1 s)",
    LifetimeBin.PERSISTENT: "Persistent (≥1 s or ≥50% trace)",
}


def classify_lifetime(lifetime_ns: float, trace_span_ns: float) -> LifetimeBin:
    """Assign a tensor to a lifetime bin."""
    if lifetime_ns <= 0:
        return LifetimeBin.POINT
    if (
        trace_span_ns >= 1e6
        and lifetime_ns >= 0.5 * trace_span_ns
    ):
        return LifetimeBin.PERSISTENT
    if lifetime_ns >= _BIN_UPPER_NS[LifetimeBin.LONG]:
        return LifetimeBin.PERSISTENT
    if lifetime_ns >= _BIN_UPPER_NS[LifetimeBin.MEDIUM]:
        return LifetimeBin.LONG
    if lifetime_ns >= _BIN_UPPER_NS[LifetimeBin.SHORT]:
        return LifetimeBin.MEDIUM
    if lifetime_ns >= _BIN_UPPER_NS[LifetimeBin.MICRO]:
        return LifetimeBin.SHORT
    return LifetimeBin.MICRO


@dataclass
class TensorLifetimeRecord:
    tensor_id: str
    name: str
    category: str
    bytes: int
    core_id: int | None
    first_t_ns: float
    last_t_ns: float
    lifetime_ns: float  # session lifetime: last − first DMA touch
    access_count: int
    kernel_span_count: int
    bin: LifetimeBin
    sbuf_total_ns: float = 0.0
    sbuf_max_stint_ns: float = 0.0
    sbuf_stint_count: int = 0

    @property
    def session_lifetime_ns(self) -> float:
        return self.lifetime_ns


@dataclass
class LifetimeBinStats:
    bin: LifetimeBin
    label: str
    tensor_count: int = 0
    total_bytes: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


@dataclass
class CategoryLifetimeStats:
    category: str
    tensor_count: int
    multi_access_count: int
    single_access_count: int
    min_lifetime_ms: float | None
    max_lifetime_ms: float | None
    median_lifetime_ms: float | None


@dataclass
class LifetimeAnalysisResult:
    trace_workload: str
    trace_span_ns: float
    first_event_t_ns: float
    last_event_t_ns: float
    tensors: list[TensorLifetimeRecord]
    bin_stats: list[LifetimeBinStats]
    category_stats: list[CategoryLifetimeStats]
    sbuf_category_stats: list[CategoryLifetimeStats]
    summary: dict[str, float | int]

    @property
    def trace_span_ms(self) -> float:
        return self.trace_span_ns / 1e6


def analyze_trace_lifetimes(trace: Trace) -> LifetimeAnalysisResult:
    """Compute per-tensor lifetimes and bin assignments from a trace."""
    tensor_map = trace.tensor_map()
    first_t: dict[str, float] = {}
    last_t: dict[str, float] = {}
    access_count: dict[str, int] = defaultdict(int)
    tensor_kernels: dict[str, set[int]] = defaultdict(set)
    active_kernels: dict[int, list[tuple[float, int]]] = defaultdict(list)

    event_times: list[float] = []

    for event in trace.parsed_events():
        event_times.append(event.t_ns)
        if isinstance(event, KernelBoundaryEvent):
            core = event.core_id if event.core_id is not None else 0
            if event.type == "kernel_start":
                active_kernels[core].append((event.t_ns, event.kernel_id))
            elif event.type == "kernel_end":
                active_kernels[core] = [
                    (t, kid) for t, kid in active_kernels[core] if kid != event.kernel_id
                ]
        elif isinstance(event, AccessEvent):
            tid = event.tensor_id
            access_count[tid] += 1
            if tid not in first_t:
                first_t[tid] = event.t_ns
                last_t[tid] = event.t_ns
            else:
                first_t[tid] = min(first_t[tid], event.t_ns)
                last_t[tid] = max(last_t[tid], event.t_ns)
            for _, kid in active_kernels.get(event.core_id, []):
                tensor_kernels[tid].add(kid)

    if not first_t:
        return LifetimeAnalysisResult(
            trace_workload=trace.metadata.workload,
            trace_span_ns=0.0,
            first_event_t_ns=min(event_times) if event_times else 0.0,
            last_event_t_ns=max(event_times) if event_times else 0.0,
            tensors=[],
            bin_stats=[LifetimeBinStats(bin=b, label=_BIN_LABELS[b]) for b in _BIN_ORDER],
            category_stats=[],
            sbuf_category_stats=[],
            summary={},
        )

    first_event_t_ns = min(event_times) if event_times else min(first_t.values())
    last_event_t_ns = max(event_times) if event_times else max(last_t.values())
    trace_span_ns = last_event_t_ns - first_event_t_ns

    sbuf_stints = _compute_sbuf_stints(trace, last_event_t_ns)

    records: list[TensorLifetimeRecord] = []
    for tid in sorted(first_t):
        tensor: TensorRecord | None = tensor_map.get(tid)
        lifetime_ns = last_t[tid] - first_t[tid]
        bin_id = classify_lifetime(lifetime_ns, trace_span_ns)
        stint_list = sbuf_stints.get(tid, [])
        sbuf_total = sum(stint_list)
        records.append(
            TensorLifetimeRecord(
                tensor_id=tid,
                name=tensor.name if tensor else tid,
                category=tensor.category.value if tensor else "unknown",
                bytes=tensor.bytes if tensor else 0,
                core_id=tensor.core_id if tensor else None,
                first_t_ns=first_t[tid],
                last_t_ns=last_t[tid],
                lifetime_ns=lifetime_ns,
                access_count=access_count[tid],
                kernel_span_count=len(tensor_kernels[tid]),
                bin=bin_id,
                sbuf_total_ns=sbuf_total,
                sbuf_max_stint_ns=max(stint_list) if stint_list else 0.0,
                sbuf_stint_count=len(stint_list),
            )
        )

    bin_stats = _aggregate_bin_stats(records)
    category_stats = _build_session_category_stats(records)
    sbuf_category_stats = _build_sbuf_category_stats(records)
    summary = _build_summary(records, trace_span_ns)

    return LifetimeAnalysisResult(
        trace_workload=trace.metadata.workload,
        trace_span_ns=trace_span_ns,
        first_event_t_ns=first_event_t_ns,
        last_event_t_ns=last_event_t_ns,
        tensors=records,
        bin_stats=bin_stats,
        category_stats=category_stats,
        sbuf_category_stats=sbuf_category_stats,
        summary=summary,
    )


def _compute_sbuf_stints(trace: Trace, last_event_t_ns: float) -> dict[str, list[float]]:
    """
    Per-tensor SBUF residency stints (install on read→sbuf until kernel_end wipe).

    Mirrors dmsim SBUF scratch behavior: data loaded into SBUF stays until the
    NeuronCore's kernel_end clears fast buffers.
    """
    default_target = "sbuf"
    in_sbuf: dict[int, dict[str, float]] = defaultdict(dict)
    stints: dict[str, list[float]] = defaultdict(list)

    for event in trace.parsed_events():
        if isinstance(event, KernelBoundaryEvent):
            if event.type != "kernel_end":
                continue
            cores = (
                [event.core_id]
                if event.core_id is not None
                else list(in_sbuf.keys()) or [0]
            )
            for core in cores:
                for tid, start_t in list(in_sbuf.get(core, {}).items()):
                    stints[tid].append(event.t_ns - start_t)
                in_sbuf[core] = {}
            continue

        if not isinstance(event, AccessEvent):
            continue

        target = event.target_level or default_target
        if event.op != "read" or target != default_target:
            continue

        core = event.core_id
        tid = event.tensor_id
        if tid not in in_sbuf[core]:
            in_sbuf[core][tid] = event.t_ns

    for core, occupants in in_sbuf.items():
        for tid, start_t in occupants.items():
            stints[tid].append(last_event_t_ns - start_t)

    return dict(stints)


def _aggregate_bin_stats(records: list[TensorLifetimeRecord]) -> list[LifetimeBinStats]:
    stats_map: dict[LifetimeBin, LifetimeBinStats] = {
        b: LifetimeBinStats(bin=b, label=_BIN_LABELS[b]) for b in _BIN_ORDER
    }
    for record in records:
        stats = stats_map[record.bin]
        stats.tensor_count += 1
        stats.total_bytes += record.bytes
        stats.by_category[record.category] = stats.by_category.get(record.category, 0) + 1
    return [stats_map[b] for b in _BIN_ORDER]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _multi_access_records(records: list[TensorLifetimeRecord]) -> list[TensorLifetimeRecord]:
    return [r for r in records if r.access_count > 1]


def _build_session_category_stats(
    records: list[TensorLifetimeRecord],
) -> list[CategoryLifetimeStats]:
    return _build_category_stats(
        records,
        value_ns=lambda r: r.lifetime_ns,
        include=lambda r: r.access_count > 1,
    )


def _build_sbuf_category_stats(
    records: list[TensorLifetimeRecord],
) -> list[CategoryLifetimeStats]:
    return _build_category_stats(
        records,
        value_ns=lambda r: r.sbuf_max_stint_ns,
        include=lambda r: r.sbuf_stint_count > 0,
    )


def _build_category_stats(
    records: list[TensorLifetimeRecord],
    *,
    value_ns,
    include,
) -> list[CategoryLifetimeStats]:
    by_category: dict[str, list[TensorLifetimeRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    category_order = ["weight", "kv_cache", "hidden", "activation", "other", "unknown"]
    ordered_cats = [c for c in category_order if c in by_category]
    ordered_cats.extend(sorted(c for c in by_category if c not in category_order))

    stats: list[CategoryLifetimeStats] = []
    for category in ordered_cats:
        group = by_category[category]
        included = [r for r in group if include(r)]
        lifetimes = [value_ns(r) for r in included]
        stats.append(
            CategoryLifetimeStats(
                category=category,
                tensor_count=len(group),
                multi_access_count=len(included),
                single_access_count=len(group) - len(included),
                min_lifetime_ms=min(lifetimes) / 1e6 if lifetimes else None,
                max_lifetime_ms=max(lifetimes) / 1e6 if lifetimes else None,
                median_lifetime_ms=_median(lifetimes) / 1e6 if lifetimes else None,
            )
        )
    return stats


def _build_summary(
    records: list[TensorLifetimeRecord],
    trace_span_ns: float,
) -> dict[str, float | int]:
    multi_records = _multi_access_records(records)
    lifetimes = [r.lifetime_ns for r in multi_records]
    if not records:
        return {}

    sorted_lt = sorted(lifetimes)
    n = len(sorted_lt)

    def percentile(p: float) -> float | None:
        if not sorted_lt:
            return None
        idx = min(int(p / 100 * n), n - 1)
        return sorted_lt[idx] / 1e6

    total_bytes = sum(r.bytes for r in records)
    single_access_count = sum(1 for r in records if r.access_count == 1)
    summary: dict[str, float | int] = {
        "tensor_count": len(records),
        "multi_access_tensors": len(multi_records),
        "single_access_tensors": single_access_count,
        "trace_span_ms": trace_span_ns / 1e6,
        "total_tensor_bytes": total_bytes,
    }
    if not sorted_lt:
        return summary

    summary.update(
        {
            "session_lifetime_min_ms": sorted_lt[0] / 1e6,
            "session_lifetime_max_ms": sorted_lt[-1] / 1e6,
            "session_lifetime_median_ms": percentile(50),
            "lifetime_min_ms": sorted_lt[0] / 1e6,
            "lifetime_max_ms": sorted_lt[-1] / 1e6,
            "lifetime_median_ms": percentile(50),
            "lifetime_mean_ms": sum(lifetimes) / n / 1e6,
            "lifetime_p10_ms": percentile(10),
            "lifetime_p25_ms": percentile(25),
            "lifetime_p75_ms": percentile(75),
            "lifetime_p90_ms": percentile(90),
            "lifetime_p99_ms": percentile(99),
        }
    )

    sbuf_records = [r for r in records if r.sbuf_stint_count > 0]
    sbuf_max_stints = sorted(r.sbuf_max_stint_ns for r in sbuf_records)
    if sbuf_max_stints:
        summary.update(
            {
                "sbuf_tensors": len(sbuf_records),
                "sbuf_max_stint_min_ms": sbuf_max_stints[0] / 1e6,
                "sbuf_max_stint_max_ms": sbuf_max_stints[-1] / 1e6,
                "sbuf_max_stint_median_ms": _median(sbuf_max_stints) / 1e6,
            }
        )
    return summary


def result_to_dict(result: LifetimeAnalysisResult) -> dict:
    """Serialize analysis result to JSON-friendly dict."""
    return {
        "trace_workload": result.trace_workload,
        "trace_span_ms": result.trace_span_ms,
        "first_event_t_ms": result.first_event_t_ns / 1e6,
        "last_event_t_ms": result.last_event_t_ns / 1e6,
        "summary": result.summary,
        "session_by_category": [
            {
                "category": stats.category,
                "tensor_count": stats.tensor_count,
                "included_count": stats.multi_access_count,
                "excluded_count": stats.single_access_count,
                "min_lifetime_ms": stats.min_lifetime_ms,
                "max_lifetime_ms": stats.max_lifetime_ms,
                "median_lifetime_ms": stats.median_lifetime_ms,
            }
            for stats in result.category_stats
        ],
        "sbuf_by_category": [
            {
                "category": stats.category,
                "tensor_count": stats.tensor_count,
                "included_count": stats.multi_access_count,
                "min_max_stint_ms": stats.min_lifetime_ms,
                "max_max_stint_ms": stats.max_lifetime_ms,
                "median_max_stint_ms": stats.median_lifetime_ms,
            }
            for stats in result.sbuf_category_stats
        ],
        "by_category": [
            {
                "category": stats.category,
                "tensor_count": stats.tensor_count,
                "multi_access_count": stats.multi_access_count,
                "single_access_count": stats.single_access_count,
                "min_lifetime_ms": stats.min_lifetime_ms,
                "max_lifetime_ms": stats.max_lifetime_ms,
                "median_lifetime_ms": stats.median_lifetime_ms,
            }
            for stats in result.category_stats
        ],
        "bins": [
            {
                "bin": stats.bin.value,
                "label": stats.label,
                "tensor_count": stats.tensor_count,
                "total_bytes": stats.total_bytes,
                "by_category": stats.by_category,
            }
            for stats in result.bin_stats
        ],
        "tensors": [
            {
                "tensor_id": r.tensor_id,
                "name": r.name,
                "category": r.category,
                "bytes": r.bytes,
                "core_id": r.core_id,
                "first_t_ms": r.first_t_ns / 1e6,
                "last_t_ms": r.last_t_ns / 1e6,
                "lifetime_ms": r.lifetime_ns / 1e6,
                "session_lifetime_ms": r.lifetime_ns / 1e6,
                "sbuf_total_ms": r.sbuf_total_ns / 1e6,
                "sbuf_max_stint_ms": r.sbuf_max_stint_ns / 1e6,
                "sbuf_stint_count": r.sbuf_stint_count,
                "lifetime_fraction_of_trace": (
                    r.lifetime_ns / result.trace_span_ns if result.trace_span_ns else 0.0
                ),
                "access_count": r.access_count,
                "kernel_span_count": r.kernel_span_count,
                "bin": r.bin.value,
            }
            for r in result.tensors
        ],
    }
