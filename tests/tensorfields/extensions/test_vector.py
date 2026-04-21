from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey
from json2vec.structs.packages import Prediction
from json2vec.structs.structure import Structure
from json2vec.tensorfields.extensions.vector import Decoder, Embedder, TensorField, loss, write


def _structure_payload(*, n_dim: int = 3, objective: str = "l2") -> dict:
    return {
        "name": "demo",
        "type": "structure",
        "batch_size": 2,
        "dropout": 0.1,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "context",
            "context_size": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "query": "[*].embedding",
                    "n_dim": n_dim,
                    "objective": objective,
                }
            ],
        },
    }


def _values() -> list:
    return [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
    ]


def _session(structure: Structure):
    return SimpleNamespace(structure=structure)


def test_vector_request_is_available_in_structure():
    structure = Structure.model_validate(_structure_payload())
    request = structure.requests["root/embedding"]
    assert request.type == "vector"
    assert request.n_dim == 3


def test_vector_request_rejects_non_positive_n_dim():
    with pytest.raises(ValueError, match="greater than 0"):
        Structure.model_validate(_structure_payload(n_dim=0))


def test_vector_tensorfield_new_rejects_wrong_embedding_length():
    structure = Structure.model_validate(_structure_payload(n_dim=3))
    session = _session(structure)
    bad_values = [
        [[0.1, 0.2], [0.3, 0.4, 0.5]],
        [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
    ]

    with pytest.raises(ValueError, match="expects embeddings with length 3"):
        TensorField.new(
            values=bad_values,
            address="root/embedding",
            session=session,
            strata=Strata.train,
            state=None,
        )


def test_vector_embedder_and_decoder_shapes():
    structure = Structure.model_validate(_structure_payload(n_dim=3))
    session = _session(structure)

    field = TensorField.new(
        values=_values(),
        address="root/embedding",
        session=session,
        strata=Strata.train,
        state=None,
    )

    embedder = Embedder(structure=structure, address="root/embedding")
    parcel = embedder(field)
    assert parcel.payload.shape == (2, 2, 16)

    decoder = Decoder(structure=structure, address="root/embedding")
    prediction = decoder([parcel])
    assert prediction.payload[TensorKey.content].shape == (2, 2, 3)


class _DummyModule:
    def __init__(self, structure: Structure):
        self.session = SimpleNamespace(structure=structure)
        self.logged: list[tuple[tuple[str, ...], float]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_vector_loss_uses_selected_objective(objective: str, expected: float):
    structure = Structure.model_validate(_structure_payload(objective=objective))
    session = _session(structure)

    field = TensorField.new(
        values=_values(),
        address="root/embedding",
        session=session,
        strata=Strata.train,
        state=None,
    )
    field.mask(1.0)

    prediction_tensor = field.targets[TensorKey.content] + 2.0
    prediction = Prediction(
        address="root/embedding",
        payload=TensorDict({TensorKey.content: prediction_tensor}, batch_size=[2]),
    )

    module = _DummyModule(structure)
    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    assert torch.isclose(output, torch.tensor(expected, dtype=output.dtype))


def test_vector_write_returns_content_payload():
    structure = Structure.model_validate(_structure_payload())
    prediction = Prediction(
        address="root/embedding",
        payload=TensorDict({TensorKey.content: torch.zeros(2, 2, 3)}, batch_size=[2]),
    )

    output = write(module=_DummyModule(structure), prediction=prediction)
    assert TensorKey.content.name in output
    assert output[TensorKey.content.name].shape == (2, 2, 3)
