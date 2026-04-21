from types import SimpleNamespace

import pytest
import torch

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.structure import Structure
from json2vec.tensorfields.extensions.entity import TensorField


def _structure_payload(*, context_size: int = 2, topk: list[int] | None = None) -> dict:
    field: dict = {
        "name": "identifier",
        "type": "entity",
        "query": "[*].id",
    }
    if topk is not None:
        field["topk"] = topk

    return {
        "name": "demo",
        "type": "structure",
        "batch_size": 2,
        "dropout": 0.1,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "context",
            "context_size": context_size,
            "n_outputs": 1,
            "fields": [field],
        },
    }


def test_entity_shape_validation_happens_during_pydantic_loading():
    Structure.model_validate(_structure_payload())

    with pytest.raises(ValueError, match="requires at least 2 elements per observation"):
        Structure.model_validate(_structure_payload(context_size=1))


def test_entity_topk_validation_rejects_one():
    with pytest.raises(ValueError, match="must not be 1"):
        Structure.model_validate(_structure_payload(topk=[1]))


def test_entity_topk_validation_rejects_values_above_max_local_classes():
    # For shape (batch=2, context_size=2), max local classes are 2*2=4.
    with pytest.raises(ValueError, match="max local entity classes"):
        Structure.model_validate(_structure_payload(topk=[4]))


def test_entity_tensorfield_uses_batch_local_unique_ids():
    structure = Structure.model_validate(_structure_payload())
    session = SimpleNamespace(structure=structure)

    values = [
        ["alice", "bob"],
        ["alice", "carol"],
    ]

    field = TensorField.new(
        values=values,
        address="root/identifier",
        session=session,
        strata=Strata.train,
        state=None,
    )

    unique_values = {token.item() for token in field.content.reshape(-1)}

    # unique entities: alice, bob, carol -> 3 local IDs
    assert unique_values == {0, 1, 2}
    assert torch.all(field.state == Tokens.valued.value)


def test_entity_tensorfield_separates_state_and_content():
    structure = Structure.model_validate(_structure_payload())
    session = SimpleNamespace(structure=structure)

    field = TensorField.new(
        values=[["alice", None], ["alice"]],
        address="root/identifier",
        session=session,
        strata=Strata.train,
        state=None,
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
    structure = Structure.model_validate(_structure_payload())
    session = SimpleNamespace(structure=structure)

    values = [
        [[1, 2], "ok"],
        ["x", "y"],
    ]

    with pytest.raises(ValueError, match="only accepts hashable scalar values"):
        TensorField.new(
            values=values,
            address="root/identifier",
            session=session,
            strata=Strata.train,
            state=None,
        )


def test_entity_mask_preserves_targets_before_replacement():
    structure = Structure.model_validate(_structure_payload())
    session = SimpleNamespace(structure=structure)
    values = [
        ["a", "b"],
        ["c", "d"],
    ]

    field = TensorField.new(
        values=values,
        address="root/identifier",
        session=session,
        strata=Strata.train,
        state=None,
    )
    original_state = field.state.clone()
    original_content = field.content.clone()
    field.mask(1.0)

    assert torch.equal(field.targets[TensorKey.state], original_state)
    assert torch.equal(field.targets[TensorKey.content], original_content)
    assert torch.all(field.state == Tokens.masked.value)
    assert torch.all(field.content == 0)
