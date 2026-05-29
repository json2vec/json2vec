from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import json2vec as j2v
from json2vec.architecture.contracts import ForwardContractError
from json2vec.data.iterables import encode
from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS


def _model(*fields: j2v.SchemaField) -> j2v.Model:
    return j2v.Model.from_schema(
        *fields,
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        max_length=1,
        attention="none",
    )


def _inputs(model: j2v.Model, batch: list[list[dict]] | None = None, strata: Strata = Strata.train) -> TensorDict:
    return encode(
        batch=batch
        or [
            [{"color": "red", "amount": 1.0, "label": "warm"}],
            [{"color": "blue", "amount": 2.0, "label": "cool"}],
        ],
        hyperparameters=model.hyperparameters,
        strata=strata,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )


def test_forward_contract_rejects_missing_active_field() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)

    del inputs[Address("record/color")]

    with pytest.raises(ForwardContractError, match="missing active request"):
        model(inputs, strata=Strata.train)


def test_forward_contract_requires_strata() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)

    with pytest.raises(TypeError, match="strata"):
        model(inputs)  # ty: ignore[missing-argument]


def test_forward_contract_rejects_unknown_extra_field() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    inputs[Address("record/extra")] = inputs[Address("record/color")].clone()

    with pytest.raises(ForwardContractError, match="unknown address"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_inactive_request_field() -> None:
    model = _model(
        j2v.Category(name="color", max_vocab_size=16),
        j2v.Category(name="ignored", active=False, max_vocab_size=16),
    )
    inputs = _inputs(model)
    tensorfield = TENSORFIELDS["category"].TensorField.empty(
        batch_size=2,
        address=Address("record/ignored"),
        hyperparameters=model.hyperparameters,
    )
    inputs[Address("record/ignored")] = tensorfield

    with pytest.raises(ForwardContractError, match="inactive request"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_array_address_field() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    inputs[Address("record")] = inputs[Address("record/color")].clone()

    with pytest.raises(ForwardContractError, match="array address"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_tensorfield_class() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16), j2v.Number(name="amount"))
    inputs = _inputs(model)
    inputs[Address("record/color")] = inputs[Address("record/amount")].clone()

    with pytest.raises(TypeError, match="must use tensorfield class"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_state_shape() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    inputs[Address("record/color")].state = inputs[Address("record/color")].state[:, :0]

    with pytest.raises(ForwardContractError, match="state must have shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_state_dtype() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    inputs[Address("record/color")].state = inputs[Address("record/color")].state.to(torch.float32)

    with pytest.raises(TypeError, match="state must use an integer dtype"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_content_without_state_shape_prefix() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    inputs[Address("record/color")].content = inputs[Address("record/color")].content[:, :0]

    with pytest.raises(ForwardContractError, match="content.*state shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_masked_non_trainable_input() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    field = inputs[Address("record/color")]
    field.state[0, 0] = Tokens.masked.value

    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train)


def test_forward_contract_allows_masked_non_trainable_predict_input() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model, strata=Strata.predict)
    field = inputs[Address("record/color")]
    field.state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.predict)


def test_forward_contract_rejects_trainable_input_without_targets() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)
    field = inputs[Address("record/color")]
    field.state[0, 0] = Tokens.masked.value
    field.trainable[0, 0] = True

    with pytest.raises(ForwardContractError, match=r"lacks targets\[state\]"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_target_leakage() -> None:
    model = _model(
        j2v.Category(name="color", max_vocab_size=16),
        j2v.Category(name="label", target=True, max_vocab_size=16),
    )
    inputs = _inputs(model)
    inputs[Address("record/label")].state[0, 0] = Tokens.valued.value

    with pytest.raises(ForwardContractError, match="must not contain visible input state"):
        model(inputs, strata=Strata.train)


def test_forward_contract_allows_predict_target_placeholder() -> None:
    model = _model(
        j2v.Category(name="color", max_vocab_size=16),
        j2v.Category(name="label", target=True, max_vocab_size=16),
    )
    inputs = _inputs(
        model,
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ],
        strata=Strata.predict,
    )

    predictions = model(inputs, strata=Strata.predict)

    assert any(prediction.address == Address("record/label") for prediction in predictions)


def test_forward_contract_rejects_predict_placeholder_in_train_strata() -> None:
    model = _model(
        j2v.Category(name="color", max_vocab_size=16),
        j2v.Category(name="label", target=True, max_vocab_size=16),
    )
    inputs = _inputs(
        model,
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ],
        strata=Strata.predict,
    )
    del inputs[TensorKey.metadata]

    with pytest.raises(ForwardContractError, match="must have trainable positions in train strata"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_metadata_outside_predict_strata() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model, strata=Strata.predict)

    with pytest.raises(ForwardContractError, match="metadata outside predict strata"):
        model(inputs, strata=Strata.train)


def test_forward_contract_uses_deterministic_backoff_schedule() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("record/color")].state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.train)
    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train)


def test_forward_contract_runs_when_batch_signature_changes() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("record/color")].content = inputs[Address("record/color")].content[:, :0]

    with pytest.raises(ForwardContractError, match="content.*state shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_runs_when_dataloader_index_changes() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train, dataloader_idx=0)

    inputs[Address("record/color")].state[0, 0] = Tokens.masked.value

    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train, dataloader_idx=1)


def test_forward_contract_resets_after_schema_mutation() -> None:
    model = _model(j2v.Category(name="color", max_vocab_size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("record/color")].state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.train)
    model.reset(j2v.where("name") == "color")

    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train)
