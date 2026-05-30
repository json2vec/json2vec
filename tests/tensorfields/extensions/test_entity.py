import pytest
import torch

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.tensorfields.extensions.entity import TensorField


def _structure_payload(*, max_length: int = 2, topk: list[int] | None = None) -> dict:
    field: dict = {
        "name": "identifier",
        "type": "entity",
        "query": "[*].id",
    }
    if topk is not None:
        field["topk"] = topk

    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "dropout": 0.1,
            "max_length": max_length,
            "fields": [field],
        },
    }


def test_entity_shape_validation_happens_during_pydantic_loading():
    Hyperparameters.model_validate(_structure_payload())

    with pytest.raises(ValueError, match="requires at least 2 elements per observation"):
        Hyperparameters.model_validate(_structure_payload(max_length=1))


def test_entity_topk_validation_rejects_one():
    with pytest.raises(ValueError, match="must not be 1"):
        Hyperparameters.model_validate(_structure_payload(topk=[1]))


def test_entity_topk_validation_allows_batch_size_dependent_values():
    Hyperparameters.model_validate(_structure_payload(topk=[4]))


def test_entity_tensorfield_uses_batch_local_unique_ids():
    structure = Hyperparameters.model_validate(_structure_payload())
    hyperparameters = structure

    values = [
        ["alice", "bob"],
        ["alice", "carol"],
    ]

    field = TensorField.new(
        values=values,
        address="root/identifier",
        hyperparameters=hyperparameters,
        strata=Strata.train,
    )

    unique_values = {token.item() for token in field.content.reshape(-1)}

    # unique entities: alice, bob, carol -> 3 local IDs
    assert unique_values == {0, 1, 2}
    assert torch.all(field.state == Tokens.valued.value)


def test_entity_tensorfield_separates_state_and_content():
    structure = Hyperparameters.model_validate(_structure_payload())
    hyperparameters = structure

    field = TensorField.new(
        values=[["alice", None], ["alice"]],
        address="root/identifier",
        hyperparameters=hyperparameters,
        strata=Strata.train,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [Tokens.valued.value, Tokens.null.value],
                [Tokens.valued.value, Tokens.padded.value],
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [0, 0],
                [0, 0],
            ],
            dtype=torch.int64,
        ),
    )


def test_entity_tensorfield_rejects_unhashable_values():
    structure = Hyperparameters.model_validate(_structure_payload())
    hyperparameters = structure

    values = [
        [[1, 2], "ok"],
        ["x", "y"],
    ]

    with pytest.raises(ValueError, match="only accepts hashable scalar values"):
        TensorField.new(
            values=values,
            address="root/identifier",
            hyperparameters=hyperparameters,
            strata=Strata.train,
        )


def test_entity_mask_preserves_targets_before_replacement():
    structure = Hyperparameters.model_validate(_structure_payload())
    hyperparameters = structure
    values = [
        ["a", "b"],
        ["c", "d"],
    ]

    field = TensorField.new(
        values=values,
        address="root/identifier",
        hyperparameters=hyperparameters,
        strata=Strata.train,
    )
    original_state = field.state.clone()
    original_content = field.content.clone()
    field.mask(1.0)

    assert torch.equal(field.targets[TensorKey.state], original_state)
    assert torch.equal(field.targets[TensorKey.content], original_content)
    assert torch.all(field.state == Tokens.masked.value)
    assert torch.all(field.content == 0)
