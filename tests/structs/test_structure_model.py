import pytest

from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.structure import Array
from json2vec.tensorfields.extensions.category import Request as Category


def _payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "description": "root array docs",
            "dropout": 0.1,
            "max_length": 2,
            "fields": [
                {
                    "name": "branch",
                    "type": "array",
                    "description": "branch docs",
                    "max_length": 4,
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


def test_array_accepts_positional_children():
    array = Array(
        Category(name="category_leaf", query="[*].code"),
        name="branch",
    )

    assert array.fields[0].name == "category_leaf"


def test_array_rejects_positional_and_keyword_children():
    with pytest.raises(TypeError, match="both positionally and by keyword"):
        Array(
            Category(name="category_leaf", query="[*].code"),
            name="branch",
            fields=[],
        )


def test_hyperparameters_derives_arrays_requests_and_shapes():
    structure = Hyperparameters.model_validate(_payload())

    assert "root" in structure.arrays
    assert "root/branch" in structure.arrays
    assert "root/branch/category_leaf" in structure.requests
    assert structure.shapes["root/branch/category_leaf"] == (2, 4)


def test_hyperparameters_converts_leaf_instances_nested_in_arrays():
    structure = Hyperparameters(
        d_model=16,
        fields={
            "name": "root",
            "type": "array",
            "fields": [
                {
                    "name": "branch",
                    "type": "array",
                    "fields": [
                        Category(name="category_leaf", query="[*].code"),
                    ],
                }
            ],
        },
    )

    request = structure.requests["root/branch/category_leaf"]
    assert request.max_vocab_size == 10_000


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


def test_hyperparameters_uses_direct_array_dropout():
    structure = Hyperparameters.model_validate(_payload())

    assert structure.arrays["root"].dropout == 0.1
    assert structure.arrays["root/branch"].dropout is None
    assert structure.requests["root/branch/category_leaf"].dropout is None


def test_hyperparameters_allows_missing_dropout():
    payload = _payload()
    payload["fields"].pop("dropout")
    structure = Hyperparameters.model_validate(payload)

    assert structure.arrays["root"].dropout is None
    assert structure.arrays["root/branch"].dropout is None


def test_hyperparameters_preserves_direct_field_dropout():
    payload = _payload()
    payload["fields"]["fields"][0]["fields"][0]["dropout"] = 0.4

    structure = Hyperparameters.model_validate(payload)

    assert structure.requests["root/branch/category_leaf"].dropout == 0.4


def test_hyperparameters_preserves_direct_mask_and_target_rates():
    payload = _payload()
    payload["fields"]["p_mask"] = 0.2
    payload["fields"]["p_prune"] = 0.1
    payload["fields"]["fields"][0]["p_mask"] = 0.3
    payload["fields"]["fields"][0]["p_prune"] = 0.4
    payload["fields"]["fields"][0]["fields"][0]["p_mask"] = 0.5

    structure = Hyperparameters.model_validate(payload)

    assert structure.arrays["root"].p_mask == 0.2
    assert structure.arrays["root/branch"].p_mask == 0.3
    assert structure.requests["root/branch/category_leaf"].p_mask == 0.5
    assert structure.requests["root/branch/category_leaf"].p_prune is None


def test_hyperparameters_allows_missing_mask_and_target_rates():
    structure = Hyperparameters.model_validate(_payload())

    assert structure.requests["root/branch/category_leaf"].p_mask is None
    assert structure.requests["root/branch/category_leaf"].p_prune is None


def test_inactive_leaf_nodes_are_kept_in_tree_but_removed_from_runtime_maps():
    payload = _payload()
    payload["fields"]["fields"][0]["fields"][0]["active"] = False
    payload["fields"]["fields"][0]["fields"][0]["p_prune"] = 1.0
    payload["fields"]["fields"][0]["fields"][0]["embed"] = True

    structure = Hyperparameters.model_validate(payload)
    inactive = structure.select(lambda node: getattr(node, "name", None) == "category_leaf")[0]

    assert inactive.active is False
    assert inactive.address == "root/branch/category_leaf"
    assert "root/branch/category_leaf" in structure.requests
    assert "root/branch/category_leaf" not in structure.active_requests
    assert structure.shapes["root/branch/category_leaf"] == (2, 4)
    assert structure.target == []
    assert structure.embed == []
