from types import SimpleNamespace

import torch
from tensordict import TensorDict

from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.dateparts import Decoder, Embedder, TensorField, loss

ADDRESS = "root/items/created_at"


def _structure_payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.0,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [
                        {
                            "name": "created_at",
                            "type": "dateparts",
                            "query": "[*].items[*].created_at",
                            "dateparts": ["day_of_week", "month_of_year"],
                        }
                    ],
                }
            ],
        },
    }


def _values() -> list:
    return [
        [["2024-01-01", "2024-06-15"]],
        [["2025-02-03"]],
    ]


def test_dateparts_embedder_and_decoder_preserve_nested_shapes():
    schema = Schema.model_validate(_structure_payload())
    field = TensorField.new(values=_values(), address=ADDRESS, schema=schema, strata=Strata.train)

    parcel = Embedder(schema=schema, address=ADDRESS)(field)
    prediction = Decoder(schema=schema, address=ADDRESS)([parcel])

    assert parcel.payload.shape == (2, 1, 2, 16)
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    for datepart in schema.requests[ADDRESS].dateparts:
        assert prediction.payload[TensorKey.content][datepart].shape == (2, 2, 2)


def test_dateparts_loss_keeps_state_content_and_mask_slots_aligned():
    schema = Schema.model_validate(_structure_payload())
    field = TensorField.new(values=_values(), address=ADDRESS, schema=schema, strata=Strata.train)
    field.hide(field.state.eq(Tokens.valued.value))

    state_logits = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state_logits.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: field.targets[TensorKey.content].clone(),
            },
            batch_size=[2],
        ),
    )
    module = SimpleNamespace(schema=schema, track=lambda _names, value: value)

    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    assert torch.isclose(output, torch.zeros_like(output), atol=1e-6)
