from pathlib import Path

import pytest

from collections import Counter

from dmsim.trace.neuron_json_ingest import (
    IngestOptions,
    discover_profile_dir,
    ingest_all_neuron_cores,
    ingest_neuron_json_profile,
    list_neuron_cores,
    merge_traces,
)
from dmsim.trace.schema import TensorCategory

ROOT = Path(__file__).resolve().parents[1]
JSON_PROFILE = ROOT / "data/traces/neuron_profile_json_4-19"


@pytest.mark.skipif(not JSON_PROFILE.exists(), reason="example profile not present")
def test_discover_and_list_cores() -> None:
    profile_dir = discover_profile_dir(JSON_PROFILE)
    cores = list_neuron_cores(profile_dir)
    assert 0 in cores


@pytest.mark.skipif(not JSON_PROFILE.exists(), reason="example profile not present")
def test_ingest_produces_trace() -> None:
    trace = ingest_neuron_json_profile(
        JSON_PROFILE,
        options=IngestOptions(
            neuron_core_id=0,
            model_key="124050204400345_vnc",
            min_transfer_bytes=4096,
            max_access_events=5000,
        ),
    )
    assert len(trace.tensors) > 0
    assert len(trace.events) > 0
    assert any(event.get("type") == "access" for event in trace.events)


@pytest.mark.skipif(not JSON_PROFILE.exists(), reason="example profile not present")
def test_ingest_all_cores_merges() -> None:
    trace = ingest_all_neuron_cores(
        JSON_PROFILE,
        options=IngestOptions(
            model_key="124050204400345",
            min_transfer_bytes=4096,
            max_access_events=20_000,
        ),
    )
    assert trace.metadata.neuron_core_ids == [0, 1, 2, 3]
    assert all(t.id.startswith("nc") for t in trace.tensors)
    core_ids = {event.get("core_id") for event in trace.events if event.get("type") == "access"}
    assert core_ids <= {0, 1, 2, 3}


@pytest.mark.skipif(not JSON_PROFILE.exists(), reason="example profile not present")
def test_ingest_classifies_weights_and_kv() -> None:
    trace = ingest_neuron_json_profile(
        JSON_PROFILE,
        options=IngestOptions(
            neuron_core_id=0,
            model_key="124050204400345",
            min_transfer_bytes=4096,
            max_access_events=5000,
        ),
    )
    cats = Counter(tensor.category for tensor in trace.tensors)
    assert cats[TensorCategory.WEIGHT] > 50
    assert cats[TensorCategory.KV_CACHE] > 20
    assert any("cache_k" in tensor.name for tensor in trace.tensors)
    assert any("attention.wq" in tensor.name for tensor in trace.tensors)
