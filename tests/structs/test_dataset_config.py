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


def test_dataset_accepts_registered_processor_callable():
    def _dataset_callable_processor(observation: dict):
        return observation

    _dataset_callable_processor.__name__ = "__dataset_callable_processor"
    processor = shim(yields=False)(_dataset_callable_processor)

    try:
        payload = _dataset_payload()
        payload["processor"] = processor

        dataset = Dataset.model_validate(payload)

        assert dataset.processor == "__dataset_callable_processor"
    finally:
        PROCESSORS.pop("__dataset_callable_processor", None)


def test_dataset_accepts_configured_processor_callable():
    def _unregistered_dataset_callable_processor(observation: dict):
        return observation

    payload = _dataset_payload()
    payload["processor"] = _unregistered_dataset_callable_processor

    dataset = Dataset.model_validate(payload)

    assert dataset.processor is _unregistered_dataset_callable_processor


def test_dataset_root_allows_none_for_processor_driven_mode():
    payload = {
        "root": None,
        "processor": _processor_name(),
        "kwargs": {},
    }

    dataset = Dataset.model_validate(payload)
    assert dataset.root is None
    assert dataset.suffix is None
    assert dataset.patterns is None


def test_dataset_requires_suffix_when_root_is_configured():
    payload = _dataset_payload()
    payload.pop("suffix")

    with pytest.raises(ValueError, match="suffix is required when root is specified"):
        Dataset.model_validate(payload)


def test_dataset_warns_when_root_has_no_patterns():
    payload = _dataset_payload()
    payload.pop("patterns")

    with pytest.warns(UserWarning, match="all strata will read the same files"):
        dataset = Dataset.model_validate(payload)

    assert dataset.patterns is None


def test_dataset_rejects_dataloader_configuration():
    payload = _dataset_payload()
    payload["file_buffer_size"] = 32

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Dataset.model_validate(payload)
