import pytest
import torch

import json2vec as jv
from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Schema
from json2vec.tensorfields.extensions.entity import TensorField

ADDRESS = "root/items/identifier"


def _structure_payload(*, length: int = 2, topk: list[int] | None = None) -> dict:
    field: dict = {
        "name": "identifier",
        "type": "entity",
        "query": "[*].items[*].id",
    }
    if topk is not None:
        field["topk"] = topk

    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": length,
                    "fields": [field],
                }
            ],
        },
    }


def test_entity_shape_validation_happens_during_pydantic_loading():
    Schema.model_validate(_structure_payload())

    with pytest.raises(ValueError, match="requires at least 2 elements per observation"):
        Schema.model_validate(_structure_payload(length=1))


def test_entity_topk_validation_rejects_one():
    with pytest.raises(ValueError, match="must not be 1"):
        Schema.model_validate(_structure_payload(topk=[1]))


def test_entity_topk_validation_rejects_values_at_or_above_slot_count():
    with pytest.raises(ValueError, match="less than the entity slot count"):
        Schema.model_validate(_structure_payload(length=4, topk=[4]))


def test_entity_topk_validation_allows_per_observation_values():
    Schema.model_validate(_structure_payload(length=4, topk=[2, 3]))


def test_entity_tensorfield_uses_observation_local_unique_ids():
    structure = Schema.model_validate(_structure_payload())
    schema = structure

    values = [
        [["alice", "bob"]],
        [["alice", "carol"]],
    ]

    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    unique_values = {token.item() for token in field.content.reshape(-1)}

    assert unique_values == {0, 1}
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[0, 1]],
                [[0, 1]],
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.all(field.state == Tokens.valued.value)


def test_entity_tensorfield_separates_state_and_content():
    structure = Schema.model_validate(_structure_payload())
    schema = structure

    field = TensorField.new(
        values=[[["alice", None]], [["alice"]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [[Tokens.valued.value, Tokens.null.value]],
                [[Tokens.valued.value, Tokens.padded.value]],
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[0, 0]],
                [[0, 0]],
            ],
            dtype=torch.int64,
        ),
    )


def test_entity_tensorfield_rejects_unhashable_values():
    structure = Schema.model_validate(_structure_payload())
    schema = structure

    values = [
        [[[1, 2], "ok"]],
        [["x", "y"]],
    ]

    with pytest.raises(ValueError, match="only accepts hashable scalar values"):
        TensorField.new(
            values=values,
            address=ADDRESS,
            schema=schema,
            strata=Strata.train,
        )


def test_entity_mask_preserves_targets_before_replacement():
    structure = Schema.model_validate(_structure_payload())
    schema = structure
    values = [
        [["a", "b"]],
        [["c", "d"]],
    ]

    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    original_state = field.state.clone()
    original_content = field.content.clone()
    field.mask(1.0)

    assert torch.equal(field.targets[TensorKey.state], original_state)
    assert torch.equal(field.targets[TensorKey.content], original_content)
    assert torch.all(field.state == Tokens.masked.value)
    assert torch.all(field.content == 0)


def test_entity_embedder_accepts_independent_observation_local_ids():
    model = jv.Model(
        jv.Branch(jv.Entity("id"), name="items", length=2),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )

    inputs = model.encode(
        [
            {"items": [{"id": "a"}, {"id": "b"}]},
            {"items": [{"id": "c"}, {"id": "d"}]},
        ],
        strata=Strata.train,
        mask=False,
    )

    assert torch.equal(
        inputs["record/items/id"].content,
        torch.tensor(
            [
                [[0, 1]],
                [[0, 1]],
            ],
            dtype=torch.int64,
        ),
    )
    model(inputs, strata=Strata.train)


def test_entity_training_loss_consumes_decoder_slot_logits_directly():
    model = jv.Model(
        jv.Branch(jv.Entity("id"), name="items", length=3),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    address = "record/items/id"
    inputs = model.encode(
        [
            {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
            {"items": [{"id": "d"}, {"id": "e"}, {"id": "f"}]},
        ],
        strata=Strata.train,
        mask=False,
    )
    field = inputs[address]
    field.hide(torch.ones_like(field.state, dtype=torch.bool))

    predictions = model(inputs, strata=Strata.train)
    prediction = next(prediction for prediction in predictions if prediction.address == address)
    content = prediction.payload[TensorKey.content]

    assert content.shape[-1] == 3
    assert content[..., 0].numel() == field.content.numel()
    output = model.training_step(inputs, batch_idx=0)
    assert torch.isfinite(output["loss"])
