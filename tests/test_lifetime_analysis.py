"""Tests for per-tensor lifetime analysis."""

from __future__ import annotations

import json

from dmsim.trace.lifetime_analysis import (
    LifetimeBin,
    analyze_trace_lifetimes,
    classify_lifetime,
    result_to_dict,
)
from dmsim.trace.schema import TensorCategory, TensorRecord, Trace, TraceMetadata


def _trace(workload: str, tensors: list[TensorRecord], events: list[dict]) -> Trace:
    return Trace(metadata=TraceMetadata(workload=workload), tensors=tensors, events=events)


def test_classify_lifetime_bins() -> None:
    assert classify_lifetime(0, 1e9) == LifetimeBin.POINT
    assert classify_lifetime(500_000, 1e9) == LifetimeBin.MICRO
    assert classify_lifetime(5e6, 1e9) == LifetimeBin.SHORT
    assert classify_lifetime(50e6, 1e9) == LifetimeBin.MEDIUM
    assert classify_lifetime(400e6, 1e9) == LifetimeBin.LONG
    assert classify_lifetime(1e9, 2e9) == LifetimeBin.PERSISTENT
    assert classify_lifetime(50e6, 100e6) == LifetimeBin.PERSISTENT


def test_sbuf_multi_stint() -> None:
    trace = _trace(
        "synthetic",
        [TensorRecord(id="a", name="a", bytes=1024, category=TensorCategory.ACTIVATION)],
        [
            {"type": "access", "t_ns": 10, "tensor_id": "a", "op": "read", "bytes": 1024, "core_id": 0},
            {"type": "kernel_end", "t_ns": 100, "kernel_id": 0, "core_id": 0},
            {"type": "access", "t_ns": 200, "tensor_id": "a", "op": "read", "bytes": 1024, "core_id": 0},
            {"type": "kernel_end", "t_ns": 300, "kernel_id": 1, "core_id": 0},
        ],
    )
    result = analyze_trace_lifetimes(trace)
    rec = result.tensors[0]
    assert rec.sbuf_stint_count == 2
    assert rec.sbuf_max_stint_ns == 100.0
    assert rec.sbuf_total_ns == 190.0


def test_sbuf_chip_wide_kernel_end_wipe() -> None:
    trace = _trace(
        "multi",
        [
            TensorRecord(id="a", name="a", bytes=1, category=TensorCategory.ACTIVATION, core_id=0),
            TensorRecord(id="b", name="b", bytes=1, category=TensorCategory.ACTIVATION, core_id=1),
        ],
        [
            {"type": "access", "t_ns": 0, "tensor_id": "a", "op": "read", "bytes": 1, "core_id": 0},
            {"type": "access", "t_ns": 0, "tensor_id": "b", "op": "read", "bytes": 1, "core_id": 1},
            {"type": "kernel_end", "t_ns": 50, "kernel_id": 0},
        ],
    )
    result = analyze_trace_lifetimes(trace)
    by_id = {r.tensor_id: r for r in result.tensors}
    assert by_id["a"].sbuf_max_stint_ns == 50.0
    assert by_id["b"].sbuf_max_stint_ns == 50.0


def test_kernel_span_null_core_id_end() -> None:
    """kernel_end without core_id must remove the kernel from every active core."""
    trace = _trace(
        "kspan",
        [TensorRecord(id="x", name="x", bytes=1, category=TensorCategory.ACTIVATION, core_id=1)],
        [
            {"type": "kernel_start", "t_ns": 0, "kernel_id": 7, "core_id": 1},
            {"type": "kernel_end", "t_ns": 100, "kernel_id": 7},
            {"type": "kernel_start", "t_ns": 200, "kernel_id": 8, "core_id": 1},
            {"type": "access", "t_ns": 210, "tensor_id": "x", "op": "read", "bytes": 1, "core_id": 1},
            {"type": "kernel_end", "t_ns": 300, "kernel_id": 8, "core_id": 1},
        ],
    )
    result = analyze_trace_lifetimes(trace)
    assert result.tensors[0].kernel_span_count == 1


def test_empty_trace() -> None:
    trace = _trace("empty", [], [{"type": "kernel_start", "t_ns": 0, "kernel_id": 0}])
    result = analyze_trace_lifetimes(trace)
    assert result.tensors == []
    assert result.summary == {}


def test_category_median_excludes_single_access() -> None:
    trace = _trace(
        "single",
        [
            TensorRecord(id="a", name="a", bytes=1, category=TensorCategory.ACTIVATION),
            TensorRecord(id="b", name="b", bytes=1, category=TensorCategory.ACTIVATION),
        ],
        [
            {"type": "access", "t_ns": 0, "tensor_id": "a", "op": "read", "bytes": 1, "core_id": 0},
            {"type": "access", "t_ns": 0, "tensor_id": "b", "op": "read", "bytes": 1, "core_id": 0},
            {"type": "access", "t_ns": 1_000_000, "tensor_id": "b", "op": "read", "bytes": 1, "core_id": 0},
        ],
    )
    result = analyze_trace_lifetimes(trace)
    activation = next(s for s in result.category_stats if s.category == "activation")
    assert activation.tensor_count == 2
    assert activation.single_access_count == 1
    assert activation.multi_access_count == 1
    assert activation.median_lifetime_ms == 1.0


def test_result_to_dict_serializes_summary() -> None:
    trace = _trace(
        "json",
        [TensorRecord(id="a", name="a", bytes=1, category=TensorCategory.WEIGHT)],
        [
            {"type": "access", "t_ns": 0, "tensor_id": "a", "op": "read", "bytes": 1, "core_id": 0},
            {"type": "access", "t_ns": 2_000_000, "tensor_id": "a", "op": "read", "bytes": 1, "core_id": 0},
            {"type": "kernel_end", "t_ns": 10_000_000, "kernel_id": 0, "core_id": 0},
        ],
    )
    payload = result_to_dict(analyze_trace_lifetimes(trace))
    json.dumps(payload)
    assert payload["summary"]["tensor_count"] == 1
    assert payload["tensors"][0]["bin"] == "short"
