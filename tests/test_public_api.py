import json2vec


def test_common_resources_are_available_from_package_root():
    assert json2vec.Model.__name__ == "Model"
    assert json2vec.AttentionMode.mha == "mha"
    assert not hasattr(json2vec, "Dataset")
    assert json2vec.CustomDataModule.__name__ == "CustomDataModule"
    assert json2vec.PolarsDataModule.__name__ == "PolarsDataModule"
    assert json2vec.StreamingDataModule.__name__ == "StreamingDataModule"
    assert json2vec.Schema.__name__ == "Schema"
    assert json2vec.Address("root", "label") == "root/label"
    assert json2vec.Branch.__name__ == "Branch"
    assert json2vec.where("type").name == "type"
    assert json2vec.preprocess.__name__ == "preprocess"
    assert json2vec.postprocess.__name__ == "postprocess"
    assert json2vec.Preprocessor.__name__ == "Preprocessor"
    assert json2vec.PreprocessorProvider.strata == "strata"
    assert json2vec.Observation.__name__ == "Observation"
    assert json2vec.Observation({"id": 1}).data == {"id": 1}
    assert not hasattr(json2vec, "observe")
    assert json2vec.OptimizerConfig is not None
    assert json2vec.SchedulerConfig is not None
    assert json2vec.RollbackCheckpoint.__name__ == "RollbackCheckpoint"
    assert json2vec.Writer.__name__ == "Writer"
    assert json2vec.Postprocessor is not None
    assert json2vec.PostprocessorProvider.metadata == "metadata"
    assert json2vec.Deployment.__name__ == "Deployment"
    assert json2vec.Accelerator.cpu == "cpu"
    assert json2vec.JSONBackend.orjson == "orjson"
    assert json2vec.Input is not None
    assert json2vec.ModelSource is not None
    assert json2vec.UpdateOperation is not None
    assert json2vec.SchemaField is not None
    assert json2vec.Category.model_fields["type"].default == "category"
    assert json2vec.Number.model_fields["type"].default == "number"
    assert json2vec.Set.model_fields["type"].default == "set"
    assert json2vec.Overflow.tail == "tail"
    assert json2vec.VocabularySyncCallback.__name__ == "VocabularySyncCallback"
    assert "number" in json2vec.TENSORFIELDS
    assert "set" in json2vec.TENSORFIELDS
