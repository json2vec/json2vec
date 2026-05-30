import pydantic
import pytest

from json2vec.structs.tree import Address, Leaf, Node


class AddressPayload(pydantic.BaseModel):
    address: Address


def test_address_can_be_initialized_from_path_parts():
    address = Address("record", "label")

    assert address == "record/label"
    assert isinstance(address, str)


def test_address_accepts_slash_delimited_or_path_parts():
    assert Address("record/label") == Address("record", "label")
    assert Address("record/metrics/sepal_length") == Address("record", "metrics", "sepal_length")


def test_address_can_be_pydantic_coerced_from_string():
    payload = AddressPayload.model_validate({"address": "record/label"})

    assert payload.address == Address("record", "label")
    assert isinstance(payload.address, Address)


def test_node_rejects_invalid_name_characters():
    with pytest.raises(ValueError, match="name may contain only letters"):
        Node.model_validate({"name": "bad name", "type": "node", "n_heads": 4})


def test_node_requires_even_n_heads():
    with pytest.raises(ValueError, match="n_heads must be even"):
        Node.model_validate({"name": "ok_name", "type": "node", "n_heads": 3})


def test_leaf_requires_non_empty_query():
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "   "})


def test_leaf_rejects_invalid_jmespath():
    with pytest.raises(ValueError, match="invalid jmespath query"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "["})


def test_leaf_can_omit_query_until_bound_to_hyperparameters():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4})

    assert leaf.query is None


def test_leaf_defaults_to_not_embedded():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "[*].leaf"})

    assert leaf.embed is False


def test_node_prune_rate_can_be_null_for_inheritance():
    node = Node.model_validate({"name": "ok_name", "type": "node", "n_heads": 4, "p_prune": None})

    assert node.p_prune is None


def test_node_target_true_sets_prune_rate():
    node = Node.model_validate({"name": "label", "type": "node", "n_heads": 4, "target": True})

    assert node.target is True
    assert node.p_prune == 1.0
    assert not node.model_extra or "target" not in node.model_extra


def test_node_target_false_is_input_only_noop():
    node = Node.model_validate({"name": "label", "type": "node", "n_heads": 4, "target": False})

    assert node.target is False
    assert node.p_prune is None
    assert not node.model_extra or "target" not in node.model_extra


def test_node_target_property_updates_prune_rate():
    node = Node.model_validate({"name": "label", "type": "node", "n_heads": 4, "p_prune": 0.25})

    node.target = True
    assert node.p_prune == 1.0

    node.target = False
    assert node.p_prune is None


def test_node_target_rejects_conflicting_prune_rate():
    with pytest.raises(ValueError, match="target=True is shorthand for p_prune=1.0"):
        Node.model_validate({"name": "label", "type": "node", "n_heads": 4, "target": True, "p_prune": 0.5})


def test_node_target_false_rejects_conflicting_prune_rate():
    with pytest.raises(ValueError, match="target=False is shorthand for p_prune=None"):
        Node.model_validate({"name": "label", "type": "node", "n_heads": 4, "target": False, "p_prune": 0.5})


def test_node_target_requires_boolean():
    with pytest.raises(ValueError, match="target must be a boolean"):
        Node.model_validate({"name": "label", "type": "node", "n_heads": 4, "target": "yes"})


def test_node_description_trims_and_accepts_optional_metadata():
    node = Node.model_validate({"name": "ok_name", "type": "node", "description": "  docs here  ", "n_heads": 4})
    assert node.description == "docs here"


def test_node_description_empty_string_becomes_none():
    node = Node.model_validate({"name": "ok_name", "type": "node", "description": "   ", "n_heads": 4})
    assert node.description is None
