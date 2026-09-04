from types import SimpleNamespace

import torch
from tensordict import TensorDict

from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.set import Decoder, Embedder, TensorField, loss, write
from relflow.tensorfields.extensions.set import output as output_type
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel
from tests.arrow import batch as arrow_batch
from tests.tensorfields.helpers import tensorize

ADDRESS = "root/items/tags"


def _structure_payload(
    *,
    p_unavailable: float | None = None,
    threshold: float | None = None,
    mask: bool | Mask = False,
) -> dict:
    field: dict = {
        "name": "tags",
        "type": "set",
        "size": 8,
        "mask": mask,
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


def _new_tensorfield(
    *,
    values: list,
    schema: Schema,
    strata: Strata,
    state,
) -> TensorField:
    batch = arrow_batch([{"items": [{"tags": value} for value in root]} for (root,) in values])
    projection = coalesce(batch, schema=schema, strata=strata)[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["set"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(state=state),
    )


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

    field = _new_tensorfield(
        values=[[[["ALPHA", "BETA"], []]], [[["BETA"]]]],
        schema=structure,
        strata=Strata.train,
        state=state,
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


def test_set_nested_mask_string_is_an_ordinary_label():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state()

    field = _new_tensorfield(
        values=[[[["<MASK>", "ALPHA"]]]],
        schema=structure,
        strata=Strata.train,
        state=state,
    )

    assert state.vocab == ["<MASK>", "ALPHA"]
    assert field.state.tolist() == [[[Tokens.valued.value, Tokens.padded.value]]]
    assert field.content[0, 0, 0, :2].tolist() == [1.0, 1.0]


def test_set_tensorfield_reserves_real_vocabulary_in_batch():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    vocabulary = OnlineVocabularyModel(size=structure.requests[ADDRESS].size)

    _new_tensorfield(
        values=[[[["ALPHA", "BETA"], ["ALPHA"]]], [[["BETA"]]]],
        schema=structure,
        strata=Strata.train,
        state=vocabulary.state,
    )

    assert vocabulary.snapshot() == ["ALPHA", "BETA"]


def test_set_tensorfield_zeros_oov_content_without_changing_state():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state(size=structure.requests[ADDRESS].size)

    _new_tensorfield(
        values=[[[["ALPHA"]]]],
        schema=structure,
        strata=Strata.train,
        state=state,
    )

    field = _new_tensorfield(
        values=[[[["OMEGA"]]]],
        schema=structure,
        strata=Strata.validate,
        state=state,
    )

    assert field.state[0, 0, 0] == Tokens.valued.value
    assert field.content[0, 0, 0].sum() == 0.0


def test_set_tensorfield_simulated_unavailable_zeros_content():
    structure = Schema.model_validate(_structure_payload(p_unavailable=1.0))
    state = _state(size=structure.requests[ADDRESS].size)

    field = _new_tensorfield(
        values=[[[["ALPHA", "BETA"]]]],
        schema=structure,
        strata=Strata.train,
        state=state,
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


def test_set_write_emits_candidates_for_each_known_vocab_item():
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

    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)
    content = output.field(TensorKey.content.name).to_pylist()

    assert output.type == datatype
    assert [[candidate[TensorKey.value.name] for candidate in row] for row in content] == [
        ["ALPHA", "BETA"],
        ["ALPHA", "BETA"],
    ]
    assert content[0][1][TensorKey.probability.name] > content[0][0][TensorKey.probability.name]


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

    output = write(module=module, prediction=prediction, datatype=output_type(module, ADDRESS))
    expected_probability = torch.sigmoid(torch.tensor(2.0)).item()
    content = output.field(TensorKey.content.name).to_pylist()

    assert content == [
        [{TensorKey.value.name: "BETA", TensorKey.probability.name: expected_probability}],
        [{TensorKey.value.name: "ALPHA", TensorKey.probability.name: expected_probability}],
    ]


def test_set_empty_vocabulary_keeps_declared_schema():
    structure = Schema.model_validate(_structure_payload())
    module = _DummyModule(schema=structure, embedder=SimpleNamespace(vocab=OnlineVocabularyModel(size=8)))
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(1, 1, len(Tokens)),
                TensorKey.content: torch.zeros(1, 1, 8),
            },
            batch_size=[1],
        ),
    )

    datatype = output_type(module, ADDRESS)
    written = write(module=module, prediction=prediction, datatype=datatype)

    assert written.type == datatype
    assert written.field(TensorKey.content.name).to_pylist() == [[]]


def test_set_loss_does_not_mutate_counter():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0, mask=Mask(reconstruct=True)))
    state = _state()
    field = _new_tensorfield(
        values=[[[["ALPHA", "BETA"], []]], [[["BETA"]]]],
        schema=structure,
        strata=Strata.train,
        state=state,
    )
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

    result = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counters[TensorKey.state.name].counts, expected_state_counts)
    assert torch.isfinite(result)
