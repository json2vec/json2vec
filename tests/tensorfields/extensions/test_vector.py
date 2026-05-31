import pytest
import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.packages import Prediction
from json2vec.tensorfields.extensions.vector import Decoder, Embedder, TensorField, loss, write


def _structure_payload(*, n_dim: int = 3, objective: str = "l2") -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "dropout": 0.1,
            "max_length": 2,
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


def test_vector_request_is_available_in_structure():
    structure = Hyperparameters.model_validate(_structure_payload())
    request = structure.requests["root/embedding"]
    assert request.type == "vector"
    assert request.n_dim == 3


def test_vector_request_rejects_non_positive_n_dim():
    with pytest.raises(ValueError, match="greater than 0"):
        Hyperparameters.model_validate(_structure_payload(n_dim=0))


def test_vector_tensorfield_new_rejects_wrong_embedding_length():
    structure = Hyperparameters.model_validate(_structure_payload(n_dim=3))
    hyperparameters = structure
    bad_values = [
        [[0.1, 0.2], [0.3, 0.4, 0.5]],
        [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
    ]

    with pytest.raises(ValueError, match="expects embeddings with length 3"):
        TensorField.new(
            values=bad_values,
            address="root/embedding",
            hyperparameters=hyperparameters,
            strata=Strata.train,
        )


def test_vector_embedder_and_decoder_shapes():
    structure = Hyperparameters.model_validate(_structure_payload(n_dim=3))
    hyperparameters = structure

    field = TensorField.new(
        values=_values(),
        address="root/embedding",
        hyperparameters=hyperparameters,
        strata=Strata.train,
    )

    embedder = Embedder(hyperparameters=structure, address="root/embedding")
    parcel = embedder(field)
    assert parcel.payload.shape == (2, 2, 16)

    decoder = Decoder(hyperparameters=structure, address="root/embedding")
    prediction = decoder([parcel])
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 3)


class _DummyModule:
    def __init__(self, structure: Hyperparameters):
        self.hyperparameters = structure
        self.logged: list[tuple[tuple[str, ...], float]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_vector_loss_uses_selected_objective(objective: str, expected: float):
    structure = Hyperparameters.model_validate(_structure_payload(objective=objective))
    hyperparameters = structure

    field = TensorField.new(
        values=_values(),
        address="root/embedding",
        hyperparameters=hyperparameters,
        strata=Strata.train,
    )
    field.mask(1.0)

    state_logits = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state_logits.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
    prediction_tensor = field.targets[TensorKey.content] + 2.0
    prediction = Prediction(
        address="root/embedding",
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
    structure = Hyperparameters.model_validate(_structure_payload())
    state_logits = torch.full((2, 2, len(Tokens)), -50.0)
    state_logits[0, :, Tokens.valued.value] = 50.0
    state_logits[1, :, Tokens.null.value] = 50.0
    prediction = Prediction(
        address="root/embedding",
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.ones(2, 2, 3),
            },
            batch_size=[2],
        ),
    )

    output = write(module=_DummyModule(structure), prediction=prediction)
    assert TensorKey.state.name in output
    assert set(output[TensorKey.state.name].keys()) == set(Tokens.__members__.keys())
    assert TensorKey.content.name in output
    assert output[TensorKey.content.name].shape == (2, 2, 3)
    assert output[TensorKey.content.name][0].sum() == 6.0
    assert output[TensorKey.content.name][1].sum() == 0.0
