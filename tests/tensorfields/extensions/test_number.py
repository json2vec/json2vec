from types import SimpleNamespace

import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.packages import Prediction
from json2vec.structs.structure import Structure
from json2vec.tensorfields.extensions.number import Decoder, Embedder, TensorField, loss, write


def _structure_payload() -> dict:
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
                    "name": "amount",
                    "type": "number",
                    "query": "[*].amount",
                }
            ],
        },
    }


def _session(structure: Structure):
    return SimpleNamespace(structure=structure)


class _TrackingModule:
    def __init__(self, structure: Structure, embedder: Embedder, decoder: Decoder):
        self.session = SimpleNamespace(structure=structure)
        self.nodes = {"root/amount": SimpleNamespace(embedder=embedder, decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_number_loss_updates_state_counter():
    structure = Structure.model_validate(_structure_payload())
    session = _session(structure)

    field = TensorField.new(
        values=[[1.0, None], [2.0]],
        address="root/amount",
        session=session,
        strata=Strata.train,
        state=None,
    )
    field.mask(1.0)

    embedder = Embedder(structure=structure, address="root/amount")
    decoder = Decoder(structure=structure, address="root/amount")
    module = _TrackingModule(structure=structure, embedder=embedder, decoder=decoder)

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

    state_targets = field.targets[TensorKey.state]
    expected_counts = torch.ones(len(Tokens), dtype=torch.int64)
    expected_counts += torch.bincount(state_targets.reshape(-1), minlength=len(Tokens))
    assert torch.equal(decoder.counter.counts, expected_counts)


def test_number_write_emits_state_probability_map():
    structure = Structure.model_validate(_structure_payload())
    prediction = Prediction(
        address="root/amount",
        payload=TensorDict(
            {
                TensorKey.state: torch.tensor(
                    [
                        [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                        [[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]],
                    ]
                ),
                TensorKey.content: torch.tensor([[[1.5]], [[2.5]]]),
            },
            batch_size=[2],
        ),
    )

    output = write(module=SimpleNamespace(session=SimpleNamespace(structure=structure)), prediction=prediction)
    state_payload = output[TensorKey.state.name]

    assert set(state_payload.keys()) == set(Tokens.__members__.keys())
    assert all(probabilities.shape == (2, 1) for probabilities in state_payload.values())
    assert state_payload[Tokens.valued.name][0, 0] > 0.99
    assert state_payload[Tokens.null.name][1, 0] > 0.99
    assert output[TensorKey.content.name].shape == (2, 1, 1)
