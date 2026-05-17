from json2vec.structs.experiment import Hyperparameters


def _payload() -> dict:
    return {
        "dropout": 0.2,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "description": "root array docs",
            "dropout": 0.1,
            "max_length": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "branch",
                    "type": "array",
                    "description": "branch docs",
                    "max_length": 4,
                    "n_outputs": 1,
                    "fields": [
                        {
                            "name": "category_leaf",
                            "type": "category",
                            "description": "category docs",
                            "query": "[*].code",
                        }
                    ],
                }
            ],
        },
    }

def test_hyperparameters_derives_arrays_requests_and_shapes():
    structure = Hyperparameters.model_validate(_payload())

    assert "root" in structure.arrays
    assert "root/branch" in structure.arrays
    assert "root/branch/category_leaf" in structure.requests
    assert structure.shapes["root/branch/category_leaf"] == (2, 4)


def test_hyperparameters_depthwise_contains_array_levels():
    structure = Hyperparameters.model_validate(_payload())
    assert structure.depthwise == [["root"], ["root/branch"]]


def test_hyperparameters_string_representation_contains_tree_nodes():
    structure = Hyperparameters.model_validate(_payload())
    rendered = str(structure)
    assert "hyperparameters (hyperparameters)" in rendered
    assert "root (array)" in rendered
    assert "category_leaf (category)" in rendered


def test_hyperparameters_preserves_field_and_array_descriptions():
    structure = Hyperparameters.model_validate(_payload())
    assert structure.arrays["root"].description == "root array docs"
    assert structure.arrays["root/branch"].description == "branch docs"
    assert structure.requests["root/branch/category_leaf"].description == "category docs"


def test_hyperparameters_resolves_array_dropout_from_nearest_parent():
    structure = Hyperparameters.model_validate(_payload())

    assert structure.resolved_dropout("root") == 0.1
    assert structure.resolved_dropout("root/branch") == 0.1
    assert structure.resolved_dropout("root/branch/category_leaf") == 0.1


def test_hyperparameters_resolves_missing_dropout_to_zero():
    payload = _payload()
    payload.pop("dropout")
    payload["fields"].pop("dropout")
    structure = Hyperparameters.model_validate(payload)

    assert structure.resolved_dropout("root/branch") == 0.0
