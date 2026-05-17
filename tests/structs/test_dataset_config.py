import pytest

from json2vec.data.datasets import Dataset
from json2vec.processors.base import PROCESSORS, shim
from json2vec.structs.enums import Strata


def _processor_name() -> str:
    if PROCESSORS:
        return next(iter(PROCESSORS))

    def _dataset_test_processor(observation: dict):
        return observation

    _dataset_test_processor.__name__ = "__dataset_test_processor"
    shim(yields=False)(_dataset_test_processor)
    return _dataset_test_processor.__name__


def _dataset_payload() -> dict:
    return {
        "root": "/tmp/dataset",
        "sample_rate": 1.0,
        "file_buffer_size": 32,
        "observation_buffer_size": 64,
        "processor": _processor_name(),
        "kwargs": {},
        "suffix": "ndjson",
        "patterns": {strata.value: ".*" for strata in Strata},
    }


def test_dataset_rejects_unregistered_processor():
    payload = _dataset_payload()
    payload["processor"] = "__missing_processor"

    with pytest.raises(ValueError, match="you haven't registered processor"):
        Dataset.model_validate(payload)


def test_dataset_root_allows_none_for_processor_driven_mode():
    payload = _dataset_payload()
    payload["root"] = None

    dataset = Dataset.model_validate(payload)
    assert dataset.root is None
