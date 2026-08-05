from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.contracts import ForwardContractError, _same_device
from relflow.data.iterables import encode
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS


def _model(*fields: rf.SchemaField) -> rf.Model:
    return rf.Model(
        *fields,
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        attention="none",
    )


def _inputs(model: rf.Model, batch: list[list[dict]] | None = None, strata: Strata = Strata.train) -> TensorDict:
    return encode(
        batch=batch
        or [
            [{"color": "red", "amount": 1.0, "label": "warm"}],
            [{"color": "blue", "amount": 2.0, "label": "cool"}],
        ],
        schema=model.schema,
        strata=strata,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )


def test_forward_contract_canonicalizes_default_accelerator_device_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _same_device(torch.device("mps"), torch.device("mps:0"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert _same_device(torch.device("cuda"), torch.device("cuda:0"))
    assert not _same_device(torch.device("cuda"), torch.device("cuda:1"))


def test_forward_contract_rejects_missing_active_field() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)

    del inputs[Address("record/color")]

    with pytest.raises(ForwardContractError, match="missing active request"):
        model(inputs, strata=Strata.train)


def test_forward_contract_requires_strata() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)

    with pytest.raises(TypeError, match="strata"):
        model(inputs)  # ty: ignore[missing-argument]


def test_forward_contract_rejects_unknown_extra_field() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    inputs[Address("record/extra")] = inputs[Address("record/color")].clone()

    with pytest.raises(ForwardContractError, match="unknown address"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_inactive_request_field() -> None:
    model = _model(
        rf.Category(name="color", size=16),
        rf.Category(name="ignored", active=False, size=16),
    )
    inputs = _inputs(model)
    tensorfield = TENSORFIELDS["category"].TensorField.empty(
        batch_size=2,
        address=Address("record/ignored"),
        schema=model.schema,
    )
    inputs[Address("record/ignored")] = tensorfield

    with pytest.raises(ForwardContractError, match="inactive request"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_branch_address_field() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    inputs[Address("record")] = inputs[Address("record/color")].clone()

    with pytest.raises(ForwardContractError, match="branch address"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_tensorfield_class() -> None:
    model = _model(rf.Category(name="color", size=16), rf.Number(name="amount"))
    inputs = _inputs(model)
    inputs[Address("record/color")] = inputs[Address("record/amount")].clone()

    with pytest.raises(TypeError, match="must use tensorfield class"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_state_shape() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    inputs[Address("record/color")].state = inputs[Address("record/color")].state[:, :0]

    with pytest.raises(ForwardContractError, match="state must have shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_state_dtype() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    inputs[Address("record/color")].state = inputs[Address("record/color")].state.to(torch.float32)

    with pytest.raises(TypeError, match="state must use an integer dtype"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_content_without_state_shape_prefix() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    inputs[Address("record/color")].content = inputs[Address("record/color")].content[:, :0]

    with pytest.raises(ForwardContractError, match="content.*state shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_masked_non_trainable_input() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    field = inputs[Address("record/color")]
    field.state[0, 0] = Tokens.masked.value

    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train)


def test_forward_contract_allows_masked_non_trainable_predict_input() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model, strata=Strata.predict)
    field = inputs[Address("record/color")]
    field.state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.predict)


def test_forward_contract_rejects_trainable_input_without_targets() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)
    field = inputs[Address("record/color")]
    field.state[0, 0] = Tokens.masked.value
    field.trainable[0, 0] = True

    with pytest.raises(ForwardContractError, match=r"lacks targets\[state\]"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_target_leakage() -> None:
    model = _model(
        rf.Category(name="color", size=16),
        rf.Category(name="label", target=True, size=16),
    )
    inputs = _inputs(model)
    inputs[Address("record/label")].state[0, 0] = Tokens.valued.value

    with pytest.raises(ForwardContractError, match="must not contain visible input state"):
        model(inputs, strata=Strata.train)


def test_forward_contract_allows_predict_target_placeholder() -> None:
    model = _model(
        rf.Category(name="color", size=16),
        rf.Category(name="label", target=True, size=16),
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
        rf.Category(name="color", size=16),
        rf.Category(name="label", target=True, size=16),
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
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model, strata=Strata.predict)

    with pytest.raises(ForwardContractError, match="metadata outside predict strata"):
        model(inputs, strata=Strata.train)


def test_forward_contract_uses_deterministic_backoff_schedule() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("record/color")].state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.train)
    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train)


def test_forward_contract_runs_when_batch_signature_changes() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("record/color")].content = inputs[Address("record/color")].content[:, :0]

    with pytest.raises(ForwardContractError, match="content.*state shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_runs_when_dataloader_index_changes() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train, dataloader_idx=0)

    inputs[Address("record/color")].state[0, 0] = Tokens.masked.value

    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train, dataloader_idx=1)


def test_forward_contract_resets_after_schema_mutation() -> None:
    model = _model(rf.Category(name="color", size=16))
    inputs = _inputs(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("record/color")].state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.train)
    model.reset(rf.where("name") == "color")

    with pytest.raises(ForwardContractError, match="masked state where trainable is false"):
        model(inputs, strata=Strata.train)
