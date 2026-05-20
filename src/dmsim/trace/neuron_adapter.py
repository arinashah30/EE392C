"""
Convert Neuron Explorer artifacts (NEFF + NTFF) into normalized dmsim traces.

Phase 1: export a normalized JSON trace from your pipeline (see README).
Phase 2: implement NTFF parsing here when you pin a Neuron SDK version with
         stable protobuf definitions for device-level per-NeuronCore traces.
"""

from __future__ import annotations

import json
from pathlib import Path

from dmsim.trace.classify import classify_tensor
from dmsim.trace.schema import (
    AccessEvent,
    KernelBoundaryEvent,
    TensorCategory,
    TensorRecord,
    Trace,
    TraceMetadata,
)


def load_normalized_trace(path: Path) -> Trace:
    from dmsim.trace.schema import load_trace

    return load_trace(path)


def ingest_ntff(
    ntff_path: Path,
    neff_path: Path | None = None,
    *,
    neuron_core_id: int = 0,
    chip_id: int = 0,
) -> Trace:
    """
    Parse NTFF + optional NEFF into a Trace.

    Raises NotImplementedError until NTFF protobuf schema is wired in.
    Use `dmsim export-trace` workflow or manual JSON in the meantime.
    """
    if not ntff_path.exists():
        raise FileNotFoundError(ntff_path)
    raise NotImplementedError(
        "NTFF ingestion is not implemented yet. "
        f"Found ntff={ntff_path}, neff={neff_path}. "
        "Produce normalized JSON (see data/traces/synthetic_decode.json) "
        "or extend this module with your Neuron SDK NTFF bindings."
    )


def write_trace(trace: Trace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(trace.model_dump(mode="json"), handle, indent=2)


def build_trace_from_neuron_tensor_table(
    rows: list[dict],
    *,
    workload: str,
    neuron_core_id: int,
    access_events: list[dict],
    kernel_events: list[dict] | None = None,
) -> Trace:
    """
    Helper for ad-hoc exports from Neuron Explorer tensor tables.

    Each row should include: name, size_bytes, and optional category override.
    access_events: dicts compatible with AccessEvent.
    """
    tensors: list[TensorRecord] = []
    for row in rows:
        name = row["name"]
        category = row.get("category")
        if category:
            cat = TensorCategory(category)
        else:
            cat = classify_tensor(name)
        tensors.append(
            TensorRecord(
                id=row.get("id", name),
                name=name,
                bytes=int(row["size_bytes"]),
                category=cat,
                core_id=row.get("core_id"),
            )
        )

    events: list[dict] = []
    for event in access_events:
        events.append(AccessEvent.model_validate(event).model_dump())
    for event in kernel_events or []:
        events.append(KernelBoundaryEvent.model_validate(event).model_dump())

    return Trace(
        metadata=TraceMetadata(
            workload=workload,
            source="neuron_export",
            neuron_core_id=neuron_core_id,
        ),
        tensors=tensors,
        events=events,
    )
