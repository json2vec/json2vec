import json2vec


def test_common_resources_are_available_from_package_root():
    assert json2vec.Architecture.__name__ == "JSON2Vec"
    assert json2vec.Dataset.__name__ == "Dataset"
    assert json2vec.PolarsDataModule.__name__ == "PolarsDataModule"
    assert json2vec.StreamingDataModule.__name__ == "StreamingDataModule"
    assert json2vec.Hyperparameters.__name__ == "Hyperparameters"
    assert json2vec.Address("root", "label") == "root/label"
    assert json2vec.Array.__name__ == "Array"
    assert json2vec.Column("code", "category").type == "category"
    assert json2vec.where("type").name == "type"
    assert json2vec.schema.__name__ == "schema"
    assert json2vec.default_processor.__name__ == "default"
    assert json2vec.Category.model_fields["type"].default == "category"
    assert json2vec.Number.model_fields["type"].default == "number"
    assert json2vec.Set.model_fields["type"].default == "set"
    assert json2vec.VocabularySyncCallback.__name__ == "VocabularySyncCallback"
    assert "number" in json2vec.TENSORFIELDS
    assert "set" in json2vec.TENSORFIELDS
    assert "default" in json2vec.PROCESSORS
