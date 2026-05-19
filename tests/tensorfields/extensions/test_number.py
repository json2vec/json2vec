from types import SimpleNamespace

import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.packages import Prediction
from json2vec.tensorfields.extensions.number import Decoder, Embedder, TensorField, loss, write


def _structure_payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "dropout": 0.1,
            "max_length": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "amount",
                    "type": "number",
                    "query": "[*].amount",
                }
            ],
        },
    }


class _TrackingModule:
    def __init__(self, hyperparameters: Hyperparameters, embedder: Embedder, decoder: Decoder):
        self.hyperparameters = hyperparameters
        self.nodes = {"root/amount": SimpleNamespace(embedder=embedder, decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_number_loss_does_not_mutate_counter():
    structure = Hyperparameters.model_validate(_structure_payload())
    hyperparameters = structure

    field = TensorField.new(
        values=[[1.0, None], [2.0]],
        address="root/amount",
        hyperparameters=hyperparameters,
        strata=Strata.train,
        state=None,
    )
    field.mask(1.0)

    embedder = Embedder(hyperparameters=structure, address="root/amount")
    decoder = Decoder(hyperparameters=structure, address="root/amount")
    module = _TrackingModule(hyperparameters=structure, embedder=embedder, decoder=decoder)

    prediction = Prediction(
        address="root/amount",
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape, 1),
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(decoder.counter.counts, expected_counts)


def test_number_write_emits_state_probability_map():
    structure = Hyperparameters.model_validate(_structure_payload())
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.null.value] = 10.0
    prediction = Prediction(
        address="root/amount",
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.tensor([[[1.5]], [[2.5]]]),
            },
            batch_size=[2],
        ),
    )

    output = write(module=SimpleNamespace(hyperparameters=structure), prediction=prediction)
    state_payload = output[TensorKey.state.name]

    assert set(state_payload.keys()) == set(Tokens.__members__.keys())
    assert all(probabilities.shape == (2, 1) for probabilities in state_payload.values())
    assert state_payload[Tokens.valued.name][0, 0] > 0.99
    assert state_payload[Tokens.null.name][1, 0] > 0.99
    assert output[TensorKey.content.name].shape == (2, 1, 1)
