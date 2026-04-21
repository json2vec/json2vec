from types import SimpleNamespace

import polars as pl
import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.packages import Prediction
from json2vec.structs.structure import Structure
from json2vec.tensorfields.extensions.category import Decoder, TensorField, loss, write


def _structure_payload(*, topk: list[int] | None = None) -> dict:
    field: dict = {
        "name": "category",
        "type": "category",
        "query": "[*].label",
        "max_vocab_size": 8,
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
            "context_size": 2,
            "n_outputs": 1,
            "fields": [field],
        },
    }


def _session(structure: Structure):
    return SimpleNamespace(structure=structure)


class _DummyState:
    def __init__(self):
        self.vocab: list[str] = []

    def __call__(self, word: str, update: bool = True) -> int:
        if not update:
            return self.vocab.index(word)

        if word not in self.vocab:
            self.vocab.append(word)

        return self.vocab.index(word)

    def __len__(self) -> int:
        return len(self.vocab)


def test_category_tensorfield_separates_state_and_content():
    structure = Structure.model_validate(_structure_payload())
    session = _session(structure)
    state = _DummyState()

    field = TensorField.new(
        values=[["ALPHA", None], ["BETA"]],
        address="root/category",
        session=session,
        strata=Strata.train,
        state=state,
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
                [1, 0],
            ],
            dtype=torch.int64,
        ),
    )


class _DummyVocab:
    def snapshot(self) -> list[str]:
        return ["ALPHA", "BETA", "GAMMA", "DELTA", "EPS"]


class _DummyEmbedder:
    def __init__(self):
        self.vocab = _DummyVocab()


class _DummyNode:
    def __init__(self):
        self.embedder = _DummyEmbedder()


class _DummyModule:
    def __init__(self):
        self.nodes = {"root/category": _DummyNode()}
        self.session = SimpleNamespace(
            structure=SimpleNamespace(
                requests={"root/category": SimpleNamespace(topk=[2, 3, 5, 10])}
            )
        )


def test_category_write_emits_state_and_content_payloads():
    module = _DummyModule()
    state_logits = torch.tensor(
        [
            [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]],
        ]
    )
    content_logits = torch.tensor(
        [
            [[0.1, 0.9, 0.2, 0.3, 0.4]],
            [[0.1, 0.2, 0.8, 0.3, 0.4]],
        ]
    )
    prediction = Prediction(
        address="root/category",
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_logits,
            },
            batch_size=[2],
        ),
    )

    output = write(module=module, prediction=prediction)
    state_payload = output[TensorKey.state.name]
    content_payload = output[TensorKey.content.name]
    topk_payload = content_payload[TensorKey.topk.name]

    assert set(state_payload.keys()) == set(Tokens.__members__.keys())
    assert all(probabilities.shape == (2, 1) for probabilities in state_payload.values())
    assert state_payload[Tokens.valued.name][0, 0] > 0.99
    assert state_payload[Tokens.padded.name][1, 0] > 0.99

    assert content_payload["value"].tolist() == [["BETA"], ["GAMMA"]]
    assert content_payload[TensorKey.probability.name].shape == (2, 1)

    assert len(topk_payload) == 2
    assert len(topk_payload[0][0]) == 5
    assert len(topk_payload[1][0]) == 5

    for row in topk_payload:
        assert set(row[0][0].keys()) == {"label", "probability"}

    frame = pl.DataFrame({"state": state_payload, "content": content_payload})
    assert isinstance(frame.schema["state"], pl.Struct)
    assert isinstance(frame.schema["content"], pl.Struct)


class _TrackingModule:
    def __init__(self, structure: Structure, decoder: Decoder):
        self.session = SimpleNamespace(structure=structure)
        self.nodes = {"root/category": SimpleNamespace(decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_category_loss_updates_state_and_content_counters():
    structure = Structure.model_validate(_structure_payload())
    session = _session(structure)
    state = _DummyState()

    field = TensorField.new(
        values=[["ALPHA", None], ["BETA"]],
        address="root/category",
        session=session,
        strata=Strata.train,
        state=state,
    )
    field.mask(1.0)

    decoder = Decoder(structure=structure, address="root/category")
    module = _TrackingModule(structure=structure, decoder=decoder)

    prediction = Prediction(
        address="root/category",
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(
                    *field.content.shape,
                    structure.requests["root/category"].max_vocab_size,
                ),
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    state_targets = field.targets[TensorKey.state]
    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    expected_state_counts += torch.bincount(state_targets.reshape(-1), minlength=len(Tokens))
    assert torch.equal(decoder.counters[TensorKey.state.name].counts, expected_state_counts)

    content_targets = field.targets[TensorKey.content]
    valued = state_targets.eq(Tokens.valued.value)
    expected_content_counts = torch.ones(
        structure.requests["root/category"].max_vocab_size,
        dtype=torch.int64,
    )
    expected_content_counts += torch.bincount(
        content_targets.masked_select(valued).reshape(-1),
        minlength=structure.requests["root/category"].max_vocab_size,
    )
    assert torch.equal(decoder.counters[TensorKey.content.name].counts, expected_content_counts)
