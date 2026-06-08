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


def test_ingest_unattributed_dynamic_dma() -> None:
    """Decode-style profiles leave source/dest unknown on software_dynamic HBM DMA."""
    device = {
        "neff_node": [
            {"variable_name": "input100", "shape": "[128 16 2 64]", "size": "524288", "type": "IN"},
            {"variable_name": "input180", "shape": "[32064 2048]", "size": "65536000", "type": "IN"},
        ],
        "dma": [
            {
                "variable": "unknown",
                "transfer_size": 8192,
                "source": [["unknown"]],
                "dest": ["unknown"],
                "queue_type": "software_dynamic",
                "timestamp": 1000,
            },
            {
                "variable": "unknown",
                "transfer_size": 8192,
                "source": [["unknown"]],
                "dest": ["unknown"],
                "queue_type": "software_dynamic",
                "timestamp": 2000,
            },
            {
                "variable": "input1",
                "transfer_size": 1024,
                "source": [["INPUT"]],
                "dest": ["SB"],
                "queue_type": "input",
                "timestamp": 3000,
            },
        ],
        "layer_summary": [],
    }
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        profile_dir = Path(tmp)
        (profile_dir / "profile.json").write_text(
            json.dumps(
                {
                    "system_profile_metadata": [{"first_ts_ns": 0, "last_ts_ns": 1}],
                    "device_profile_list": [
                        {
                            "nc_id": 0,
                            "device_profile_name": "device_nc0",
                        }
                    ],
                }
            )
        )
        (profile_dir / "device_nc0.json").write_text(json.dumps(device))

        trace = ingest_neuron_json_profile(
            profile_dir,
            options=IngestOptions(
                neuron_core_id=0,
                min_transfer_bytes=64,
                aggregate_dma=True,
                kernel_from_layers=False,
                max_access_events=None,
            ),
        )

    access_events = [event for event in trace.events if event.get("type") == "access"]
    assert len(access_events) >= 2
    cats = {tensor.id: tensor.category for tensor in trace.tensors}
    access_cats = Counter(cats[event["tensor_id"]] for event in access_events)
    assert access_cats[TensorCategory.WEIGHT] > 0 or access_cats[TensorCategory.OTHER] > 0
    assert any(event["tensor_id"].startswith("hbm_traffic_") for event in access_events)


def test_ingest_skip_unattributed_dma() -> None:
    """Unknown dynamic DMA can be omitted instead of synthetic hbm_traffic_*."""
    device = {
        "neff_node": [
            {"variable_name": "input180", "shape": "[32064 2048]", "size": "65536000", "type": "IN"},
        ],
        "dma": [
            {
                "variable": "unknown",
                "transfer_size": 8192,
                "source": [["unknown"]],
                "dest": ["unknown"],
                "queue_type": "software_dynamic",
                "timestamp": 1000,
            },
            {
                "variable": "input1",
                "transfer_size": 1024,
                "source": [["INPUT"]],
                "dest": ["SB"],
                "queue_type": "input",
                "timestamp": 3000,
            },
        ],
        "layer_summary": [],
    }
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        profile_dir = Path(tmp)
        (profile_dir / "profile.json").write_text(
            json.dumps(
                {
                    "system_profile_metadata": [{"first_ts_ns": 0, "last_ts_ns": 1}],
                    "device_profile_list": [{"nc_id": 0, "device_profile_name": "device_nc0"}],
                }
            )
        )
        (profile_dir / "device_nc0.json").write_text(json.dumps(device))

        trace = ingest_neuron_json_profile(
            profile_dir,
            options=IngestOptions(
                neuron_core_id=0,
                kernel_from_layers=False,
                skip_unattributed_dma=True,
            ),
        )

    access_events = [event for event in trace.events if event.get("type") == "access"]
    assert not any(event["tensor_id"].startswith("hbm_traffic_") for event in access_events)
    assert len(access_events) == 1
