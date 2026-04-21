import pytest

from json2vec.processors.base import PROCESSORS, register
from json2vec.structs.enums import Stage, Strata
from json2vec.structs.experiment import Session


def _processor_name() -> str:
    if PROCESSORS:
        return next(iter(PROCESSORS))

    def _dataset_test_processor(observation: dict):
        return observation

    _dataset_test_processor.__name__ = "__dataset_test_processor"
    register(_dataset_test_processor)
    return _dataset_test_processor.__name__


def _structure_payload() -> dict:
    return {
        "name": "demo",
        "type": "structure",
        "batch_size": 2,
        "dropout": 0.1,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "context",
            "context_size": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "identifier",
                    "type": "entity",
                    "query": "[*].id",
                }
            ],
        },
    }


def _session_payload() -> dict:
    return {
        "name": "session",
        "task": Stage.fit.value,
        "dataset": {
            "root": "/tmp/dataset",
            "sample_rate": 1.0,
            "file_buffer_size": 32,
            "observation_buffer_size": 64,
            "processor": _processor_name(),
            "kwargs": {},
            "suffix": "ndjson",
            "patterns": {strata.value: ".*" for strata in Strata},
        },
        "structure": _structure_payload(),
        "trainer": {"min_epochs": 1, "max_epochs": 3},
        "learning_rate": 1e-3,
    }


def test_dataset_rejects_unregistered_processor():
    payload = _session_payload()
    payload["dataset"]["processor"] = "__missing_processor"

    with pytest.raises(ValueError, match="you haven't registered processor"):
        Session.model_validate(payload)


def test_session_patch_applies_override_without_mutating_original():
    session = Session.model_validate(_session_payload())
    patched = session.patch(override=[{"op": "replace", "path": "/dataset/sample_rate", "value": 0.5}], in_place=False)

    assert session.dataset.sample_rate == 1.0
    assert patched.dataset.sample_rate == 0.5


def test_dataset_root_allows_none_for_processor_driven_mode():
    payload = _session_payload()
    payload["dataset"]["root"] = None

    session = Session.model_validate(payload)
    assert session.dataset.root is None
