from threading import Lock
from types import SimpleNamespace

import polars as pl
import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.packages import Prediction
from json2vec.tensorfields.extensions.category import (
    UNAVAILABLE_LABEL,
    Decoder,
    TensorField,
    Vocabulary,
    loss,
    write,
)


def _structure_payload(*, topk: list[int] | None = None, p_unavailable: float | None = None) -> dict:
    field: dict = {
        "name": "category",
        "type": "category",
        "query": "[*].label",
        "max_vocab_size": 8,
    }
    if topk is not None:
        field["topk"] = topk
    if p_unavailable is not None:
        field["p_unavailable"] = p_unavailable

    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "dropout": 0.1,
            "max_length": 2,
            "n_outputs": 1,
            "fields": [field],
        },
    }


class _DummyState:
    def __init__(self, max_vocab_size: int = 8):
        self.vocab: list[str] = []
        self.max_vocab_size = max_vocab_size

    def __call__(self, word: str, update: bool = True) -> int:
        if word is None:
            return None

        if word in self.vocab:
            return self.vocab.index(word)

        if not update:
            return self.max_vocab_size

        if len(self.vocab) >= self.max_vocab_size:
            return self.max_vocab_size

        self.vocab.append(word)
        return self.vocab.index(word)

    def __len__(self) -> int:
        return len(self.vocab)


def test_category_vocabulary_refreshes_stale_validation_snapshot():
    master: list[str] = []
    lock = Lock()
    proposals: list[str] = []
    proposal_lock = Lock()
    validation_state = Vocabulary(
        master=master,
        lock=lock,
        proposals=proposals,
        proposal_lock=proposal_lock,
        max_vocab_size=8,
    )
    training_state = Vocabulary(
        master=master,
        lock=lock,
        proposals=proposals,
        proposal_lock=proposal_lock,
        max_vocab_size=8,
    )

    assert training_state("ALPHA", update=True) == 0
    assert validation_state("ALPHA", update=False) == 0
    assert len(validation_state) == 1


def test_category_vocabulary_nonzero_rank_proposes_unseen_tokens():
    master: list[str] = []
    proposals: list[str] = []
    state = Vocabulary(
        master=master,
        lock=Lock(),
        proposals=proposals,
        proposal_lock=Lock(),
        max_vocab_size=8,
    )
    state.configure_distributed(global_rank=1, world_size=2)

    assert state("ALPHA", update=True) == state.unavailable_index
    assert master == []
    assert proposals == ["ALPHA"]


def test_category_tensorfield_separates_state_and_content():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=0.0))
    hyperparameters = structure
    state = _DummyState()

    field = TensorField.new(
        values=[["ALPHA", None], ["BETA"]],
        address="root/category",
        hyperparameters=hyperparameters,
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


def test_category_tensorfield_marks_oov_as_unavailable_without_changing_state():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=0.0))
    hyperparameters = structure
    state = _DummyState(max_vocab_size=structure.requests["root/category"].max_vocab_size)

    TensorField.new(
        values=[["ALPHA"]],
        address="root/category",
        hyperparameters=hyperparameters,
        strata=Strata.train,
        state=state,
    )

    field = TensorField.new(
        values=[["OMEGA"]],
        address="root/category",
        hyperparameters=hyperparameters,
        strata=Strata.validate,
        state=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor([[Tokens.valued.value, Tokens.padded.value]], dtype=torch.int64),
    )
    assert torch.equal(
        field.content,
        torch.tensor([[structure.requests["root/category"].max_vocab_size, 0]], dtype=torch.int64),
    )


def test_category_tensorfield_can_simulate_unavailable_during_training():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=1.0))
    hyperparameters = structure
    state = _DummyState(max_vocab_size=structure.requests["root/category"].max_vocab_size)

    field = TensorField.new(
        values=[["ALPHA", None], ["BETA"]],
        address="root/category",
        hyperparameters=hyperparameters,
        strata=Strata.train,
        state=state,
    )

    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [structure.requests["root/category"].max_vocab_size, 0],
                [structure.requests["root/category"].max_vocab_size, 0],
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
        self.hyperparameters = SimpleNamespace(
            requests={"root/category": SimpleNamespace(topk=[2, 3, 5, 10], max_vocab_size=8)}
        )


def test_category_write_emits_state_and_content_payloads():
    module = _DummyModule()
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0
    content_logits = torch.tensor(
        [
            [[0.1, 0.9, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, -1.0]],
            [[0.1, 0.2, 0.8, 0.3, 0.4, 0.0, 0.0, 0.0, 1.2]],
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
    assert all(candidate["label"] != UNAVAILABLE_LABEL for row in topk_payload for candidate in row[0])

    for row in topk_payload:
        assert set(row[0][0].keys()) == {"label", "probability"}

    frame = pl.DataFrame({"state": state_payload, "content": content_payload})
    assert isinstance(frame.schema["state"], pl.Struct)
    assert isinstance(frame.schema["content"], pl.Struct)


def test_category_write_excludes_unavailable_when_it_has_highest_logit():
    module = _DummyModule()
    state_logits = torch.zeros(1, 1, len(Tokens))
    content_logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 100.0]]])
    prediction = Prediction(
        address="root/category",
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_logits,
            },
            batch_size=[1],
        ),
    )

    output = write(module=module, prediction=prediction)
    content_payload = output[TensorKey.content.name]

    assert content_payload[TensorKey.value.name].tolist() == [["EPS"]]
    assert all(
        candidate["label"] != UNAVAILABLE_LABEL
        for row in content_payload[TensorKey.topk.name]
        for candidate in row[0]
    )


class _TrackingModule:
    def __init__(self, hyperparameters: Hyperparameters, decoder: Decoder):
        self.hyperparameters = hyperparameters
        self.nodes = {"root/category": SimpleNamespace(decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_category_loss_updates_state_and_content_counters():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=0.0))
    hyperparameters = structure
    state = _DummyState()

    field = TensorField.new(
        values=[["ALPHA", None], ["BETA"]],
        address="root/category",
        hyperparameters=hyperparameters,
        strata=Strata.train,
        state=state,
    )
    field.mask(1.0)

    decoder = Decoder(hyperparameters=structure, address="root/category")
    module = _TrackingModule(hyperparameters=structure, decoder=decoder)

    prediction = Prediction(
        address="root/category",
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(
                    *field.content.shape,
                    structure.requests["root/category"].max_vocab_size + 1,
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
        structure.requests["root/category"].max_vocab_size + 1,
        dtype=torch.int64,
    )
    expected_content_counts += torch.bincount(
        content_targets.masked_select(valued).reshape(-1),
        minlength=structure.requests["root/category"].max_vocab_size + 1,
    )
    assert torch.equal(decoder.counters[TensorKey.content.name].counts, expected_content_counts)
