import pytest
from beartype.roar import BeartypeCallHintParamViolation

import json2vec as j2v
from json2vec.data.datasets.base import PreprocessorConfig
from json2vec.data.datasets.streaming import StreamingDataModule
from json2vec.preprocessors.base import PREPROCESSORS, preprocess
from json2vec.structs.enums import Suffix
from json2vec.structs.experiment import Hyperparameters


def _preprocessor_name() -> str:
    if PREPROCESSORS:
        return next(iter(PREPROCESSORS))

    def _dataset_test_preprocessor(observation: dict):
        return observation

    _dataset_test_preprocessor.__name__ = "__dataset_test_preprocessor"
    preprocess(yields=False)(_dataset_test_preprocessor)
    return _dataset_test_preprocessor.__name__


def _hyperparameters():
    return Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "record",
                "type": "array",
                "max_length": 1,
                "fields": [],
            },
        }
    )


def _model():
    return j2v.Model.from_schema(
        j2v.Category("id", max_vocab_size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=1,
    )


def test_preprocessor_normalization_rejects_unregistered_name():
    with pytest.raises(ValueError, match="you haven't registered preprocessor"):
        PreprocessorConfig.normalize("__missing_preprocessor")


def test_preprocessor_normalization_accepts_registered_callable():
    def _dataset_callable_preprocessor(observation: dict):
        return observation

    _dataset_callable_preprocessor.__name__ = "__dataset_callable_preprocessor"
    preprocessor = preprocess(yields=False)(_dataset_callable_preprocessor)

    try:
        assert PreprocessorConfig.normalize(preprocessor) == "__dataset_callable_preprocessor"
    finally:
        PREPROCESSORS.pop("__dataset_callable_preprocessor", None)


def test_preprocessor_normalization_accepts_configured_callable():
    def _unregistered_dataset_callable_preprocessor(observation: dict):
        return observation

    assert (
        PreprocessorConfig.normalize(_unregistered_dataset_callable_preprocessor)
        is _unregistered_dataset_callable_preprocessor
    )


def test_preprocessor_normalization_is_optional():
    assert PreprocessorConfig.normalize(None) is None
    assert PreprocessorConfig.normalize(_preprocessor_name()) == _preprocessor_name()


def test_streaming_datamodule_rejects_uncompiled_split_pattern():
    with pytest.raises(BeartypeCallHintParamViolation):
        StreamingDataModule(
            model=_model(),
            root="/tmp/json2vec-test",
            suffix=Suffix.ndjson,
            train=r".*",
        )
