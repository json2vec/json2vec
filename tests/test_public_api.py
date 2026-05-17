import json2vec


def test_common_resources_are_available_from_package_root():
    assert json2vec.JSON2Vec.__name__ == "JSON2Vec"
    assert json2vec.Dataset.__name__ == "Dataset"
    assert json2vec.DefaultDataModule.__name__ == "DefaultDataModule"
    assert json2vec.Hyperparameters.__name__ == "Hyperparameters"
    assert json2vec.Array.__name__ == "Array"
    assert "number" in json2vec.TENSORFIELDS
    assert "set" in json2vec.TENSORFIELDS
    assert "default" in json2vec.PROCESSORS
