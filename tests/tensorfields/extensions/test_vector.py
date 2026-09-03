from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict

from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.base import TENSORFIELDS
from relflow.tensorfields.extensions.vector import Decoder, Embedder, TensorField, loss, write
from relflow.tensorfields.extensions.vector import output as output_type
from tests.arrow import batch as arrow_batch

ADDRESS = "root/items/embedding"


def _structure_payload(*, n_dim: int = 3, objective: str = "l2") -> dict:
    field: dict = {
        "name": "embedding",
        "type": "vector",
        "n_dim": n_dim,
        "objective": objective,
    }
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
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def _values() -> list:
    return [
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        [[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
    ]


def _new_tensorfield(*, values: list, schema: Schema, strata: Strata) -> TensorField:
    batch = arrow_batch([{"items": [{"embedding": value} for value in root]} for (root,) in values])
    field = coalesce(batch, schema=schema, strata=strata)[ADDRESS]
    field = replace(field, values=TENSORFIELDS["vector"].prepare(field.values, address=ADDRESS))
    return TensorField.new(field=field, address=ADDRESS, schema=schema, strata=strata)


def test_vector_request_is_available_in_structure():
    structure = Schema.model_validate(_structure_payload())
    request = structure.requests[ADDRESS]
    assert request.type == "vector"
    assert request.n_dim == 3


def test_vector_request_rejects_non_positive_n_dim():
    with pytest.raises(ValueError, match="greater than 0"):
        Schema.model_validate(_structure_payload(n_dim=0))


def test_vector_tensorfield_new_rejects_wrong_embedding_length():
    structure = Schema.model_validate(_structure_payload(n_dim=3))
    schema = structure
    bad_values = [
        [[[0.1, 0.2], [0.3, 0.4, 0.5]]],
        [[[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]],
    ]

    with pytest.raises(ValueError, match="expects every value to have length 3"):
        _new_tensorfield(
            values=bad_values,
            schema=schema,
            strata=Strata.train,
        )


def test_vector_embedder_and_decoder_shapes():
    structure = Schema.model_validate(_structure_payload(n_dim=3))
    schema = structure

    field = _new_tensorfield(
        values=_values(),
        schema=schema,
        strata=Strata.train,
    )

    embedder = Embedder(schema=structure, address=ADDRESS)
    parcel = embedder(field)
    assert parcel.payload.shape == (2, 1, 2, 16)

    decoder = Decoder(schema=structure, address=ADDRESS)
    prediction = decoder([parcel])
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 3)


class _DummyModule:
    def __init__(self, structure: Schema):
        self.schema = structure
        self.logged: list[tuple[tuple[str, ...], float]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_vector_loss_uses_selected_objective(objective: str, expected: float):
    structure = Schema.model_validate(_structure_payload(objective=objective))
    schema = structure

    field = _new_tensorfield(
        values=_values(),
        schema=schema,
        strata=Strata.train,
    )
    field.mask(1.0)

    state_logits = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state_logits.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
    prediction_tensor = field.targets[TensorKey.content] + 2.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: prediction_tensor,
            },
            batch_size=[2],
        ),
    )

    module = _DummyModule(structure)
    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    assert torch.isclose(output, torch.tensor(expected, dtype=output.dtype))


def test_vector_write_returns_content_payload():
    structure = Schema.model_validate(_structure_payload())
    state_logits = torch.full((2, 2, len(Tokens)), -50.0)
    state_logits[0, :, Tokens.valued.value] = 50.0
    state_logits[1, :, Tokens.null.value] = 50.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.ones(2, 2, 3),
            },
            batch_size=[2],
        ),
    )

    module = _DummyModule(structure)
    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)
    assert output.type == datatype
    content = output.field(TensorKey.content.name)
    assert len(content) == 4
    assert content.to_pylist() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
