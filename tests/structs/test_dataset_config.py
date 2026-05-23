import pytest

from json2vec.data.datasets import Dataset
from json2vec.preprocessors.base import PREPROCESSORS, preprocess
from json2vec.structs.enums import Strata


def _preprocessor_name() -> str:
    if PREPROCESSORS:
        return next(iter(PREPROCESSORS))

    def _dataset_test_preprocessor(observation: dict):
        return observation

    _dataset_test_preprocessor.__name__ = "__dataset_test_preprocessor"
    preprocess(yields=False)(_dataset_test_preprocessor)
    return _dataset_test_preprocessor.__name__


def _dataset_payload() -> dict:
    return {
        "root": "/tmp/dataset",
        "preprocessor": _preprocessor_name(),
        "kwargs": {},
        "suffix": "ndjson",
        "patterns": {strata.value: ".*" for strata in Strata},
    }


def test_dataset_rejects_unregistered_preprocessor():
    payload = _dataset_payload()
    payload["preprocessor"] = "__missing_preprocessor"

    with pytest.raises(ValueError, match="you haven't registered preprocessor"):
        Dataset.model_validate(payload)


def test_dataset_accepts_registered_preprocessor_callable():
    def _dataset_callable_preprocessor(observation: dict):
        return observation

    _dataset_callable_preprocessor.__name__ = "__dataset_callable_preprocessor"
    preprocessor = preprocess(yields=False)(_dataset_callable_preprocessor)

    try:
        payload = _dataset_payload()
        payload["preprocessor"] = preprocessor

        dataset = Dataset.model_validate(payload)

        assert dataset.preprocessor == "__dataset_callable_preprocessor"
    finally:
        PREPROCESSORS.pop("__dataset_callable_preprocessor", None)


def test_dataset_accepts_configured_preprocessor_callable():
    def _unregistered_dataset_callable_preprocessor(observation: dict):
        return observation

    payload = _dataset_payload()
    payload["preprocessor"] = _unregistered_dataset_callable_preprocessor

    dataset = Dataset.model_validate(payload)

    assert dataset.preprocessor is _unregistered_dataset_callable_preprocessor


def test_dataset_root_allows_none_for_preprocessor_driven_mode():
    payload = {
        "root": None,
        "preprocessor": _preprocessor_name(),
        "kwargs": {},
    }

    dataset = Dataset.model_validate(payload)
    assert dataset.root is None
    assert dataset.suffix is None
    assert dataset.patterns is None


def test_dataset_preprocessor_is_optional():
    dataset = Dataset.model_validate({"root": None})

    assert dataset.preprocessor is None


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
