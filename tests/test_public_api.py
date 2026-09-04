from typing import get_type_hints

import relflow


def test_common_resources_are_available_from_package_root():
    assert relflow.Model.__name__ == "Model"
    assert relflow.AttentionMode.mha == "mha"
    assert not hasattr(relflow, "Dataset")
    assert relflow.ArrowDataModule.__name__ == "ArrowDataModule"
    assert relflow.CustomDataModule.__name__ == "CustomDataModule"
    assert relflow.PolarsDataModule.__name__ == "PolarsDataModule"
    assert not hasattr(relflow, "StreamingDataModule")
    assert relflow.SyntheticDataModule.__name__ == "SyntheticDataModule"
    assert relflow.Schema.__name__ == "Schema"
    assert relflow.Address("root", "label") == "root/label"
    assert relflow.Branch.__name__ == "Branch"
    assert relflow.Mask.model_fields["rate"].default is None
    assert relflow.where("type").name == "type"
    assert relflow.preprocess.__name__ == "preprocess"
    assert relflow.postprocess.__name__ == "postprocess"
    assert relflow.Preprocessor.__name__ == "Preprocessor"
    assert relflow.PreprocessorProvider.strata == "strata"
    assert relflow.Batch.__name__ == "Batch"
    assert relflow.Context().state is None
    assert relflow.Context().salt == 0
    assert not hasattr(relflow, "Observation")
    assert not hasattr(relflow, "observe")
    assert relflow.OptimizerConfig is not None
    assert relflow.SchedulerConfig is not None
    assert relflow.RollbackCheckpoint.__name__ == "RollbackCheckpoint"
    assert relflow.Writer.__name__ == "Writer"
    assert relflow.Postprocessor is not None
    assert not hasattr(relflow, "PostprocessorProvider")
    assert relflow.Deployment.__name__ == "Deployment"
    assert relflow.Accelerator.cpu == "cpu"
    assert relflow.JSONBackend.orjson == "orjson"
    assert relflow.Input is not None
    assert relflow.ModelSource is not None
    assert relflow.UpdateOperation is not None
    assert relflow.SchemaField is not None
    assert relflow.Category.model_fields["type"].default == "category"
    assert relflow.Boolean.model_fields["type"].default == "boolean"
    assert relflow.Number.model_fields["type"].default == "number"
    assert relflow.Set.model_fields["type"].default == "set"
    assert relflow.Hash.model_fields["type"].default == "hash"
    assert not hasattr(relflow, "Entity")
    assert not hasattr(relflow, "StaticEntity")
    assert relflow.Overflow.tail == "tail"
    assert relflow.VocabularySyncCallback.__name__ == "VocabularySyncCallback"
    assert "number" in relflow.TENSORFIELDS
    assert "boolean" in relflow.TENSORFIELDS
    assert "set" in relflow.TENSORFIELDS
    assert "hash" in relflow.TENSORFIELDS
    assert "entity" not in relflow.TENSORFIELDS
    assert "static_entity" not in relflow.TENSORFIELDS
    assert "hashable" not in relflow.TENSORFIELDS


def test_data_module_constructor_annotations_resolve_at_runtime():
    for module in (
        relflow.ArrowDataModule,
        relflow.CustomDataModule,
        relflow.PolarsDataModule,
        relflow.SyntheticDataModule,
    ):
        hints = get_type_hints(module.__init__)
        assert hints["model"] is relflow.Model

    writer_hints = get_type_hints(relflow.Writer.write_on_batch_end)
    assert writer_hints["output"] is relflow.Batch
