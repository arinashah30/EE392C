from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class TensorCategory(str, Enum):
    WEIGHT = "weight"
    KV_CACHE = "kv_cache"
    HIDDEN = "hidden"
    ACTIVATION = "activation"
    OTHER = "other"


class TensorRecord(BaseModel):
    id: str
    name: str = ""
    bytes: int
    category: TensorCategory = TensorCategory.OTHER
    core_id: int | None = None


class TraceEventType(str, Enum):
    ACCESS = "access"
    KERNEL_START = "kernel_start"
    KERNEL_END = "kernel_end"


class AccessEvent(BaseModel):
    type: Literal["access"] = "access"
    t_ns: float = 0
    tensor_id: str
    op: Literal["read", "write"] = "read"
    bytes: int
    target_level: str = "sbuf"
    core_id: int = 0


class KernelBoundaryEvent(BaseModel):
    type: Literal["kernel_start", "kernel_end"]
    t_ns: float = 0
    kernel_id: int
    core_id: int | None = None


TraceEvent = AccessEvent | KernelBoundaryEvent


class TraceMetadata(BaseModel):
    workload: str = ""
    source: str = ""
    neuron_core_id: int | None = None
    neuron_core_ids: list[int] = Field(default_factory=list)
    chip_id: int = 0

    @property
    def num_neuron_cores(self) -> int:
        if self.neuron_core_ids:
            return len(self.neuron_core_ids)
        if self.neuron_core_id is not None:
            return 1
        return 1


class Trace(BaseModel):
    version: int = 1
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)
    tensors: list[TensorRecord]
    events: list[dict]

    def parsed_events(self) -> list[TraceEvent]:
        parsed: list[TraceEvent] = []
        for raw in self.events:
            event_type = raw.get("type")
            if event_type == "access":
                parsed.append(AccessEvent.model_validate(raw))
            elif event_type in ("kernel_start", "kernel_end"):
                parsed.append(KernelBoundaryEvent.model_validate(raw))
            else:
                raise ValueError(f"unknown trace event type: {event_type}")
        return sorted(parsed, key=lambda event: event.t_ns)

    def tensor_map(self) -> dict[str, TensorRecord]:
        return {tensor.id: tensor for tensor in self.tensors}

    def access_counts(self) -> dict[str, int]:
        """Count trace access events per tensor_id (for placement spill ordering)."""
        counts: dict[str, int] = {}
        for raw in self.events:
            if raw.get("type") != "access":
                continue
            tensor_id = raw.get("tensor_id")
            if not tensor_id:
                continue
            counts[tensor_id] = counts.get(tensor_id, 0) + 1
        return counts


def load_trace(path: Path) -> Trace:
    with path.open() as handle:
        data = json.load(handle)
    return Trace.model_validate(data)
