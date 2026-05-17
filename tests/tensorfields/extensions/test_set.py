import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.packages import Prediction
from json2vec.tensorfields.extensions.set import Decoder, TensorField, loss, write


def _structure_payload(*, p_unavailable: float | None = None) -> dict:
    field: dict = {
        "name": "tags",
        "type": "set",
        "query": "[*].tags",
        "max_vocab_size": 8,
    }
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


def test_set_request_is_available_in_hyperparameters():
    structure = Hyperparameters.model_validate(_structure_payload())
    request = structure.requests["root/tags"]

    assert request.type == "set"
    assert request.max_vocab_size == 8


def test_set_tensorfield_encodes_multi_hot_content():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=0.0))
    state = _DummyState()

    field = TensorField.new(
        values=[[["ALPHA", "BETA"], []], [["BETA"]]],
        address="root/tags",
        hyperparameters=structure,
        strata=Strata.train,
        state=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [Tokens.valued.value, Tokens.valued.value],
                [Tokens.valued.value, Tokens.padded.value],
            ],
            dtype=torch.int64,
        ),
    )
    assert field.content.shape == (2, 2, structure.requests["root/tags"].max_vocab_size + 1)
    assert field.content[0, 0, 0] == 1.0
    assert field.content[0, 0, 1] == 1.0
    assert field.content[0, 1].sum() == 0.0
    assert field.content[1, 0, 1] == 1.0


def test_set_tensorfield_marks_oov_as_unavailable_without_changing_state():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=0.0))
    state = _DummyState(max_vocab_size=structure.requests["root/tags"].max_vocab_size)

    TensorField.new(
        values=[[["ALPHA"]]],
        address="root/tags",
        hyperparameters=structure,
        strata=Strata.train,
        state=state,
    )

    field = TensorField.new(
        values=[[["OMEGA"]]],
        address="root/tags",
        hyperparameters=structure,
        strata=Strata.validate,
        state=state,
    )

    unavailable = structure.requests["root/tags"].max_vocab_size
    assert field.state[0, 0] == Tokens.valued.value
    assert field.content[0, 0, unavailable] == 1.0
    assert field.content[0, 0, :unavailable].sum() == 0.0


class _DummyVocab:
    def snapshot(self) -> list[str]:
        return ["ALPHA", "BETA"]


class _DummyEmbedder:
    def __init__(self):
        self.vocab = _DummyVocab()


class _DummyNode:
    def __init__(self, decoder: Decoder | None = None):
        self.embedder = _DummyEmbedder()
        self.decoder = decoder


class _DummyModule:
    def __init__(self, hyperparameters=None, decoder: Decoder | None = None):
        self.nodes = {"root/tags": _DummyNode(decoder=decoder)}
        self.hyperparameters = hyperparameters

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_set_write_emits_probability_for_each_known_vocab_item():
    module = _DummyModule()
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0
    content_logits = torch.tensor(
        [
            [[0.0, 2.0, -2.0]],
            [[1.0, -1.0, 3.0]],
        ]
    )
    prediction = Prediction(
        address="root/tags",
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

    assert set(state_payload.keys()) == set(Tokens.__members__.keys())
    assert set(content_payload.keys()) == {"ALPHA", "BETA"}
    assert content_payload["ALPHA"].shape == (2, 1)
    assert content_payload["BETA"][0, 0] > content_payload["ALPHA"][0, 0]


def test_set_loss_updates_state_counter():
    structure = Hyperparameters.model_validate(_structure_payload(p_unavailable=0.0))
    state = _DummyState()
    field = TensorField.new(
        values=[[["ALPHA", "BETA"], []], [["BETA"]]],
        address="root/tags",
        hyperparameters=structure,
        strata=Strata.train,
        state=state,
    )
    field.mask(1.0)

    decoder = Decoder(hyperparameters=structure, address="root/tags")
    module = _DummyModule(hyperparameters=structure, decoder=decoder)
    prediction = Prediction(
        address="root/tags",
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape),
            },
            batch_size=field.batch_size,
        ),
    )

    result = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    state_targets = field.targets[TensorKey.state]
    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    expected_state_counts += torch.bincount(state_targets.reshape(-1), minlength=len(Tokens))
    assert torch.equal(decoder.counters[TensorKey.state.name].counts, expected_state_counts)
    assert torch.isfinite(result)
