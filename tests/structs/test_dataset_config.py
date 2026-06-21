import pytest

import json2vec as jv
from json2vec.data.datasets.base import PreprocessorConfig
from json2vec.data.datasets.streaming import StreamingDataModule
from json2vec.structs.enums import Suffix
from json2vec.structs.experiment import Schema


def _schema():
    return Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "record",
                "type": "branch",
                "length": 1,
                "fields": [],
            },
        }
    )


def _model():
    return jv.Model.from_tree(
        jv.Category("id", size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=1,
    )


def test_preprocessor_normalization_rejects_string_names():
    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor object or None"):
        PreprocessorConfig.normalize("__missing_preprocessor")


def test_preprocessor_normalization_accepts_processor_object():
    @jv.preprocess
    def _dataset_callable_preprocessor(observation: dict):
        return jv.Observation(observation)

    assert PreprocessorConfig.normalize(_dataset_callable_preprocessor) is _dataset_callable_preprocessor


def test_preprocessor_normalization_rejects_raw_callable():
    def _unregistered_dataset_callable_preprocessor(observation: dict):
        return observation

    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor object or None"):
        PreprocessorConfig.normalize(_unregistered_dataset_callable_preprocessor)


def test_preprocessor_normalization_is_optional():
    assert PreprocessorConfig.normalize(None) is None


def test_streaming_datamodule_accepts_raw_split_pattern():
    module = StreamingDataModule(
        model=_model(),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        train=r".*",
    )

    assert module.train is not None
    assert module.train.pattern == r".*"
