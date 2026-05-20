from pathlib import Path

import pytest

from dmsim.trace.neuron_json_ingest import (
    IngestOptions,
    discover_profile_dir,
    ingest_neuron_json_profile,
    list_neuron_cores,
)

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
