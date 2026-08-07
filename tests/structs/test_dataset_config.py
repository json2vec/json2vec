import pytest

import relflow as rf
from relflow.data.datasets.base import PreprocessorConfig
from relflow.data.datasets.streaming import StreamingDataModule
from relflow.structs.enums import Suffix
from relflow.structs.experiment import Schema


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
    return rf.Model(
        rf.Category("id", size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=1,
    )


def test_preprocessor_normalization_rejects_string_names():
    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor object or None"):
        PreprocessorConfig.normalize("__missing_preprocessor")


def test_preprocessor_normalization_accepts_processor_object():
    @rf.preprocess
    def _dataset_callable_preprocessor(observation: dict):
        return rf.Observation(observation)

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
        root="/tmp/relflow-test",
        suffix=Suffix.ndjson,
        train=r".*",
    )

    assert module.train is not None
    assert module.train.pattern == r".*"
