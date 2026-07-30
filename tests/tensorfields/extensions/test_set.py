import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Schema
from json2vec.structs.packages import Prediction
from json2vec.tensorfields.extensions.set import Decoder, Embedder, TensorField, loss, write
from json2vec.tensorfields.shared.vocabulary import OnlineVocabularyModel

ADDRESS = "root/items/tags"


def _structure_payload(
    *,
    p_unavailable: float | None = None,
    threshold: float | None = None,
) -> dict:
    field: dict = {
        "name": "tags",
        "type": "set",
        "query": "[*].items[*].tags",
        "size": 8,
    }
    if p_unavailable is not None:
        field["p_unavailable"] = p_unavailable
    if threshold is not None:
        field["threshold"] = threshold

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


def _state(size: int = 8):
    return OnlineVocabularyModel(size=size).state


def test_set_request_is_available_in_schema():
    structure = Schema.model_validate(_structure_payload())
    request = structure.requests[ADDRESS]

    assert request.type == "set"
    assert request.size == 8
    assert request.threshold is None


def test_set_request_accepts_threshold():
    structure = Schema.model_validate(_structure_payload(threshold=0.8))
    assert structure.requests[ADDRESS].threshold == 0.8


def test_set_tensorfield_encodes_multi_hot_content():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state()

    field = TensorField.new(
        values=[[[["ALPHA", "BETA"], []]], [[["BETA"]]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [[Tokens.valued.value, Tokens.valued.value]],
                [[Tokens.valued.value, Tokens.padded.value]],
            ],
            dtype=torch.int64,
        ),
    )
    assert field.content.shape == (2, 1, 2, structure.requests[ADDRESS].size)
    assert field.content[0, 0, 0, 0] == 1.0
    assert field.content[0, 0, 0, 1] == 1.0
    assert field.content[0, 0, 1].sum() == 0.0
    assert field.content[1, 0, 0, 1] == 1.0


def test_set_tensorfield_reserves_real_vocabulary_in_batch():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    vocabulary = OnlineVocabularyModel(size=structure.requests[ADDRESS].size)

    TensorField.new(
        values=[[[["ALPHA", "BETA"], ["ALPHA"]]], [[["BETA"]]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=vocabulary.state,
    )

    assert vocabulary.snapshot() == ["ALPHA", "BETA"]


def test_set_tensorfield_zeros_oov_content_without_changing_state():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state(size=structure.requests[ADDRESS].size)

    TensorField.new(
        values=[[[["ALPHA"]]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    field = TensorField.new(
        values=[[[["OMEGA"]]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.validate,
        interprocess_encoding_context=state,
    )

    assert field.state[0, 0, 0] == Tokens.valued.value
    assert field.content[0, 0, 0].sum() == 0.0


def test_set_tensorfield_simulated_unavailable_zeros_content():
    structure = Schema.model_validate(_structure_payload(p_unavailable=1.0))
    state = _state(size=structure.requests[ADDRESS].size)

    field = TensorField.new(
        values=[[[["ALPHA", "BETA"]]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    assert field.content.shape[-1] == structure.requests[ADDRESS].size
    assert field.content[0, 0, 0].sum() == 0.0


class _DummyVocab:
    def snapshot(self) -> list[str]:
        return ["ALPHA", "BETA"]


class _DummyEmbedder:
    def __init__(self):
        self.vocab = _DummyVocab()


class _DummyNode:
    def __init__(self, embedder=None, decoder: Decoder | None = None):
        self.embedder = embedder or _DummyEmbedder()
        self.decoder = decoder


class _DummyModule:
    def __init__(self, schema=None, embedder=None, decoder: Decoder | None = None):
        self.nodes = {ADDRESS: _DummyNode(embedder=embedder, decoder=decoder)}
        self.schema = schema

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_set_write_emits_probability_for_each_known_vocab_item():
    module = _DummyModule(schema=Schema.model_validate(_structure_payload()))
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
        address=ADDRESS,
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


def test_set_write_filters_content_when_threshold_is_configured():
    module = _DummyModule(schema=Schema.model_validate(_structure_payload(threshold=0.75)))
    state_logits = torch.zeros(2, 1, len(Tokens))
    content_logits = torch.tensor(
        [
            [[0.0, 2.0, -2.0]],
            [[2.0, -2.0, 0.0]],
        ]
    )
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_logits,
            },
            batch_size=[2],
        ),
    )

    output = write(module=module, prediction=prediction)
    expected_probability = torch.sigmoid(torch.tensor(2.0)).item()

    assert output[TensorKey.content.name] == [
        [{"BETA": expected_probability}],
        [{"ALPHA": expected_probability}],
    ]


def test_set_loss_does_not_mutate_counter():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state()
    field = TensorField.new(
        values=[[[["ALPHA", "BETA"], []]], [[["BETA"]]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field.mask(1.0)

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _DummyModule(schema=structure, embedder=embedder, decoder=decoder)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape),
            },
            batch_size=field.batch_size,
        ),
    )

    result = torch.stack(loss(module=module, prediction=prediction, batch=field, strata=Strata.train)).sum()

    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counters[TensorKey.state.name].counts, expected_state_counts)
    assert torch.isfinite(result)
