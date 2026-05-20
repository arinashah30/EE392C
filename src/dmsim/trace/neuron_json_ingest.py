"""
Ingest Neuron Explorer JSON profile exports into normalized dmsim traces.

Expected layout (same as Neuron Explorer directory export):
  profile_dir/
    profile.json                          # system trace + metadata
    i-<instance>_pid_<pid>_nc_<N>_session_0.json   # per-NeuronCore device trace

The raw NEFF/NTFF bundle is not required when JSON exports are present.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
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
from dmsim.trace.neuron_adapter import write_trace

# DMA endpoint labels used in Neuron Explorer JSON (device trace)
_HBM_LIKE = frozenset({"VIRTUAL", "REMOTE", "DRAM", "HBM"})
_SBUF_LIKE = frozenset({"SB", "SBUF"})


@dataclass
class IngestOptions:
    neuron_core_id: int = 0
    model_key: str | None = None
    min_transfer_bytes: int = 64
    aggregate_dma: bool = True
    include_system_tensor_events: bool = False
    kernel_from_layers: bool = True
    max_access_events: int | None = 50_000


@dataclass(frozen=True)
class _AggKey:
    variable: str
    route: str
    op: str


@dataclass
class _AggBucket:
    bytes: int = 0
    count: int = 0
    first_ts: int = 0
    last_ts: int = 0


def discover_profile_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "profile.json").exists():
        return path
    if path.name == "profile.json":
        return path.parent
    raise FileNotFoundError(f"No profile.json under {path}")


def list_neuron_cores(profile_dir: Path) -> list[int]:
    profile = _load_profile(profile_dir)
    cores: set[int] = set()
    for entry in profile.get("device_profile_list", []):
        if "nc_id" in entry:
            cores.add(int(entry["nc_id"]))
    return sorted(cores)


def resolve_device_json(profile_dir: Path, neuron_core_id: int, model_key: str | None) -> Path:
    profile = _load_profile(profile_dir)
    candidates: list[Path] = []
    seen: set[Path] = set()
    for entry in profile.get("device_profile_list", []):
        if int(entry.get("nc_id", -1)) != neuron_core_id:
            continue
        name = entry.get("device_profile_name", "")
        path = profile_dir / f"{name}.json"
        if path.exists() and path not in seen:
            candidates.append(path)
            seen.add(path)
    if not candidates:
        raise FileNotFoundError(
            f"No per-core JSON for nc_id={neuron_core_id} in {profile_dir}"
        )
    if model_key:
        matched = [path for path in candidates if _device_json_matches_model(path, model_key)]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            return matched[0]
        raise FileNotFoundError(
            f"model_key={model_key!r} not found in profile_info of {[p.name for p in candidates]}"
        )
    return candidates[0]


def _device_json_matches_model(path: Path, model_key: str) -> bool:
    """Match NEFF id inside profile_info / ntff_filename without loading full device JSON."""
    import mmap

    needle = model_key.encode("utf-8")
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.find(needle) != -1


def ingest_neuron_json_profile(
    profile_dir: Path,
    *,
    options: IngestOptions | None = None,
) -> Trace:
    opts = options or IngestOptions()
    profile_dir = discover_profile_dir(profile_dir)
    profile = _load_profile(profile_dir)
    device_path = resolve_device_json(profile_dir, opts.neuron_core_id, opts.model_key)
    device = _load_json(device_path)

    meta = _system_metadata(profile)
    tensors, accesses = _build_from_device_dma(device, opts)
    if opts.include_system_tensor_events:
        _merge_system_tensor_events(
            profile,
            opts.neuron_core_id,
            meta["first_ts_ns"],
            tensors,
            accesses,
            opts,
        )

    kernel_events = []
    if opts.kernel_from_layers:
        kernel_events = _kernel_events_from_layers(device, meta["first_ts_ns"])

    events: list[dict] = [e.model_dump() for e in kernel_events]
    events.extend(a.model_dump() for a in sorted(accesses, key=lambda event: event.t_ns))
    if opts.max_access_events and len(events) > opts.max_access_events:
        events = _downsample_events(events, opts.max_access_events)

    workload = device_path.stem
    return Trace(
        metadata=TraceMetadata(
            workload=workload,
            source=f"neuron_json:{profile_dir.name}",
            neuron_core_id=opts.neuron_core_id,
        ),
        tensors=list(tensors.values()),
        events=events,
    )


def ingest_and_write(
    profile_dir: Path,
    output_path: Path,
    *,
    options: IngestOptions | None = None,
) -> Trace:
    trace = ingest_neuron_json_profile(profile_dir, options=options)
    write_trace(trace, output_path)
    return trace


def _load_profile(profile_dir: Path) -> dict:
    return _load_json(profile_dir / "profile.json")


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _system_metadata(profile: dict) -> dict:
    meta_list = profile.get("system_profile_metadata") or []
    if meta_list:
        return meta_list[0]
    return {"first_ts_ns": 0, "last_ts_ns": 0}


def _flat_location(loc) -> str:
    if not loc:
        return "unknown"
    if isinstance(loc, list):
        if loc and isinstance(loc[0], list):
            return ",".join(_flat_location(x) for x in loc)
        return ",".join(str(x) for x in loc)
    return str(loc)


def _dma_route(dma: dict) -> tuple[str, str]:
    return _flat_location(dma.get("source")), _flat_location(dma.get("dest"))


def _classify_variable(name: str) -> TensorCategory:
    lowered = name.lower()
    if any(token in lowered for token in ("weight", "kernel", "param")):
        return TensorCategory.WEIGHT
    if any(token in lowered for token in ("kv", "cache", "k_proj", "v_proj")):
        return TensorCategory.KV_CACHE
    if any(token in lowered for token in ("hidden", "mlp", "ffn")):
        return TensorCategory.HIDDEN
    if "input" in lowered or "output" in lowered:
        return TensorCategory.ACTIVATION
    return classify_tensor(name)


def _build_from_device_dma(device: dict, opts: IngestOptions) -> tuple[dict[str, TensorRecord], list[AccessEvent]]:
    tensors: dict[str, TensorRecord] = {}
    accesses: list[AccessEvent] = []
    buckets: dict[_AggKey, _AggBucket] = defaultdict(_AggBucket)

    for dma in device.get("dma", []):
        transfer = int(dma.get("transfer_size") or dma.get("read_size") or 0)
        if transfer < opts.min_transfer_bytes:
            continue
        src, dst = _dma_route(dma)
        op_target = _map_dma_to_access(src, dst)
        if op_target is None:
            continue
        op, target_level = op_target
        variable = str(dma.get("variable") or dma.get("tensor_name") or "unknown")
        tensor_id = _tensor_id(variable)
        tensors[tensor_id] = TensorRecord(
            id=tensor_id,
            name=variable,
            bytes=max(tensors.get(tensor_id, TensorRecord(id=tensor_id, name=variable, bytes=0)).bytes, transfer),
            category=_classify_variable(variable),
        )
        ts = int(dma.get("timestamp") or 0)
        route = f"{src}->{dst}"
        if opts.aggregate_dma:
            key = _AggKey(variable=variable, route=route, op=op)
            bucket = buckets[key]
            bucket.bytes += transfer
            bucket.count += 1
            if bucket.count == 1:
                bucket.first_ts = ts
            bucket.last_ts = ts
        else:
            accesses.append(
                AccessEvent(
                    t_ns=_device_ts_to_ns(ts),
                    tensor_id=tensor_id,
                    op=op,
                    bytes=transfer,
                    target_level=target_level,
                    core_id=opts.neuron_core_id,
                )
            )

    if opts.aggregate_dma:
        for key, bucket in buckets.items():
            tensor_id = _tensor_id(key.variable)
            tensors[tensor_id] = TensorRecord(
                id=tensor_id,
                name=key.variable,
                bytes=max(tensors.get(tensor_id, TensorRecord(id=tensor_id, name=key.variable, bytes=0)).bytes, bucket.bytes),
                category=_classify_variable(key.variable),
            )
            src, dst = key.route.split("->", 1)
            mapped = _map_dma_to_access(src, dst)
            target_level = mapped[1] if mapped else "sbuf"
            accesses.append(
                AccessEvent(
                    t_ns=_device_ts_to_ns(bucket.first_ts),
                    tensor_id=tensor_id,
                    op=key.op,  # type: ignore[arg-type]
                    bytes=bucket.bytes,
                    target_level=target_level,
                    core_id=opts.neuron_core_id,
                )
            )

    for ann in device.get("annotation") or []:
        if "load_to_sbuf" not in json.dumps(ann):
            continue
        name = str(ann.get("tensor_name") or ann.get("name") or "unknown")
        tensor_id = _tensor_id(name)
        size = int(ann.get("tensor_size_bytes") or ann.get("load_to_sbuf_total_size_bytes") or 0)
        tensors[tensor_id] = TensorRecord(
            id=tensor_id,
            name=name,
            bytes=max(tensors.get(tensor_id, TensorRecord(id=tensor_id, name=name, bytes=0)).bytes, size),
            category=_classify_variable(name),
        )

    return tensors, accesses


def _map_dma_to_access(src: str, dst: str) -> tuple[str, str] | None:
    src_u, dst_u = src.upper(), dst.upper()
    src_tokens = set(re.split(r"[,/]", src_u))
    dst_tokens = set(re.split(r"[,/]", dst_u))

    if src_tokens & _SBUF_LIKE and dst_tokens & _HBM_LIKE:
        return "write", "hbm"
    if src_tokens & _HBM_LIKE and dst_tokens & _SBUF_LIKE:
        return "read", "sbuf"
    if "WEIGHT" in src_tokens and dst_tokens & _SBUF_LIKE:
        return "read", "sbuf"
    if "INPUT" in src_tokens and dst_tokens & _SBUF_LIKE:
        return "read", "sbuf"
    if src_tokens & _SBUF_LIKE and "OUTPUT" in dst_tokens:
        return "write", "sbuf"
    return None


def _merge_system_tensor_events(
    profile: dict,
    neuron_core_id: int,
    first_ts_ns: int,
    tensors: dict[str, TensorRecord],
    accesses: list[AccessEvent],
    opts: IngestOptions,
) -> None:
    for event in profile.get("trace_event") or []:
        if int(event.get("nc_idx", -1)) != neuron_core_id:
            continue
        name = event.get("name", "")
        if name not in ("nrt_tensor_read", "nrt_tensor_write", "dmem_buf_copyin", "dmem_buf_copyout"):
            continue
        size = int(event.get("size") or 0)
        if size < opts.min_transfer_bytes:
            continue
        ts = int(event.get("timestamp") or 0) - int(first_ts_ns or 0)
        tid = str(event.get("tensor_id", event.get("id", "runtime")))
        tensor_id = _tensor_id(f"rt_{tid}")
        tensors[tensor_id] = TensorRecord(
            id=tensor_id,
            name=f"runtime_tensor_{tid}",
            bytes=max(tensors.get(tensor_id, TensorRecord(id=tensor_id, name=tensor_id, bytes=0)).bytes, size),
            category=TensorCategory.OTHER,
        )
        if name in ("nrt_tensor_read", "dmem_buf_copyin"):
            op, target = "read", "sbuf"
        else:
            op, target = "write", "sbuf"
        accesses.append(
            AccessEvent(
                t_ns=float(max(ts, 0)),
                tensor_id=tensor_id,
                op=op,
                bytes=size,
                target_level=target,
                core_id=neuron_core_id,
            )
        )


def _kernel_events_from_layers(device: dict, first_ts_ns: int) -> list[KernelBoundaryEvent]:
    layers = device.get("layer_summary") or []
    if not layers:
        return []
    events: list[KernelBoundaryEvent] = []
    for idx, layer in enumerate(layers):
        start = int(layer.get("start") or 0)
        end = int(layer.get("end") or start)
        events.append(
            KernelBoundaryEvent(type="kernel_start", t_ns=_device_ts_to_ns(start), kernel_id=idx)
        )
        events.append(
            KernelBoundaryEvent(type="kernel_end", t_ns=_device_ts_to_ns(end), kernel_id=idx)
        )
    return events


def _device_ts_to_ns(timestamp_us: int) -> float:
    """Device trace timestamps are in microseconds relative to profile start."""
    return float(timestamp_us) * 1_000.0


def _tensor_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
    return slug or "tensor"


def _downsample_events(events: list[dict], limit: int) -> list[dict]:
    kernels = [event for event in events if event.get("type") != "access"]
    accesses = [event for event in events if event.get("type") == "access"]
    if len(accesses) <= limit - len(kernels):
        return kernels + accesses
    step = max(1, len(accesses) // (limit - len(kernels)))
    sampled = accesses[::step][: limit - len(kernels)]
    return kernels + sampled
