import json2vec


def test_common_resources_are_available_from_package_root():
    assert json2vec.Model.__name__ == "Model"
    assert json2vec.AttentionMode.mha == "mha"
    assert not hasattr(json2vec, "Dataset")
    assert json2vec.PolarsDataModule.__name__ == "PolarsDataModule"
    assert json2vec.StreamingDataModule.__name__ == "StreamingDataModule"
    assert json2vec.Hyperparameters.__name__ == "Hyperparameters"
    assert json2vec.Address("root", "label") == "root/label"
    assert json2vec.Array.__name__ == "Array"
    assert json2vec.where("type").name == "type"
    assert json2vec.preprocess.__name__ == "preprocess"
    assert json2vec.Preprocessor.__name__ == "Preprocessor"
    assert json2vec.OptimizerConfig is not None
    assert json2vec.SchedulerConfig is not None
    assert json2vec.RollbackCheckpoint.__name__ == "RollbackCheckpoint"
    assert json2vec.SchemaField is not None
    assert json2vec.Category.model_fields["type"].default == "category"
    assert json2vec.Number.model_fields["type"].default == "number"
    assert json2vec.Set.model_fields["type"].default == "set"
    assert json2vec.VocabularySyncCallback.__name__ == "VocabularySyncCallback"
    assert "number" in json2vec.TENSORFIELDS
    assert "set" in json2vec.TENSORFIELDS
    assert isinstance(json2vec.PREPROCESSORS, dict)
