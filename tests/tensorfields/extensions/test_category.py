from types import SimpleNamespace
from typing import Any

import polars as pl
import torch
from tensordict import TensorDict

from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.category import (
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState

ADDRESS = "root/items/category"


def _structure_payload(*, topk: list[int] | None = None, p_unavailable: float | None = None) -> dict:
    field: dict = {
        "name": "category",
        "type": "category",
        "size": 8,
    }
    if topk is not None:
        field["topk"] = topk
    if p_unavailable is not None:
        field["p_unavailable"] = p_unavailable

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


def _tensorfield(
    rows: list[list[Any]],
    *,
    schema: Schema,
    strata: Strata,
    interprocess_encoding_context: VocabularyState,
) -> TensorField:
    batch = [[{"items": [{"category": value} for value in row]}] for row in rows]
    field = coalesce(batch, schema=schema, strata=strata)[ADDRESS]
    return TensorField.new(
        field=field,
        address=ADDRESS,
        schema=schema,
        strata=strata,
        interprocess_encoding_context=interprocess_encoding_context,
    )


def test_category_vocabulary_refreshes_stale_validation_snapshot():
    vocabulary = OnlineVocabularyModel(size=8)
    validation_state = vocabulary.state
    training_state = vocabulary.state

    training_state.reserve("ALPHA", learn=True)
    validation_state.reserve("ALPHA", learn=False)

    assert training_state.encode("ALPHA") == 0
    assert validation_state.encode("ALPHA") == 0
    assert len(validation_state) == 1


def test_category_vocabulary_nonzero_rank_proposes_unseen_tokens():
    vocabulary = OnlineVocabularyModel(size=8)
    state = vocabulary.state
    state.configure_distributed(global_rank=1, world_size=2)

    state.reserve("ALPHA", learn=True)

    assert state.encode("ALPHA") == state.unavailable_index
    assert vocabulary.snapshot() == []
    assert list(vocabulary.proposals) == ["ALPHA"]


def test_category_vocabulary_reserves_nested_tokens_in_batch():
    vocabulary = OnlineVocabularyModel(size=8)
    state = vocabulary.state

    state.reserve([[["ALPHA", None, "BETA"], ["ALPHA"]]], learn=True)

    assert vocabulary.snapshot() == ["ALPHA", "BETA"]
    assert state.encode("ALPHA") == 0
    assert state.encode("BETA") == 1


def test_category_vocabulary_batch_proposals_are_unique_per_call():
    vocabulary = OnlineVocabularyModel(size=8)
    state = vocabulary.state
    state.configure_distributed(global_rank=1, world_size=2)

    state.reserve([["ALPHA", "ALPHA"], ["BETA"]], learn=True)

    assert state.encode("ALPHA") == state.unavailable_index
    assert vocabulary.snapshot() == []
    assert list(vocabulary.proposals) == ["ALPHA", "BETA"]


def test_category_tensorfield_separates_state_and_content():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    schema = structure
    state = _state()

    field = _tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [[Tokens.valued.value, Tokens.null.value]],
                [[Tokens.valued.value, Tokens.padded.value]],
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[0, 0]],
                [[1, 0]],
            ],
            dtype=torch.int64,
        ),
    )


def test_category_tensorfield_marks_oov_as_unavailable_without_changing_state():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    schema = structure
    state = _state(size=structure.requests[ADDRESS].size)

    _tensorfield(
        rows=[["ALPHA"]],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    field = _tensorfield(
        rows=[["OMEGA"]],
        schema=schema,
        strata=Strata.validate,
        interprocess_encoding_context=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor([[[Tokens.valued.value, Tokens.padded.value]]], dtype=torch.int64),
    )
    assert torch.equal(
        field.content,
        torch.tensor([[[structure.requests[ADDRESS].size, 0]]], dtype=torch.int64),
    )


def test_category_tensorfield_can_simulate_unavailable_during_training():
    structure = Schema.model_validate(_structure_payload(p_unavailable=1.0))
    schema = structure
    state = _state(size=structure.requests[ADDRESS].size)

    field = _tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[structure.requests[ADDRESS].size, 0]],
                [[structure.requests[ADDRESS].size, 0]],
            ],
            dtype=torch.int64,
        ),
    )


def test_category_embedder_and_decoder_use_real_vocab_width():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    request = structure.requests[ADDRESS]
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)

    assert embedder.embeddings[TensorKey.content.name].num_embeddings == request.size
    assert embedder.counters[TensorKey.content.name].size == request.size
    assert decoder.linears[TensorKey.content.name].out_features == request.size


def test_category_embedder_zeroes_unavailable_and_non_valued_content_contributions():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)
    unavailable = structure.requests[ADDRESS].size
    field = TensorField(
        state=torch.tensor(
            [
                [
                    Tokens.valued.value,
                    Tokens.valued.value,
                    Tokens.null.value,
                    Tokens.padded.value,
                    Tokens.masked.value,
                    Tokens.other.value,
                ]
            ],
            dtype=torch.int64,
        ),
        content=torch.tensor([[0, unavailable, 0, 0, 0, 0]], dtype=torch.int64),
        trainable=torch.zeros((1, 6), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    output = embedder(field).payload
    expected = embedder.embeddings[TensorKey.state.name](field.state)
    expected[:, 0] += embedder.embeddings[TensorKey.content.name](field.content[:, 0])

    assert torch.allclose(output, expected)


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
        self.nodes = {ADDRESS: _DummyNode()}
        self.schema = SimpleNamespace(requests={ADDRESS: SimpleNamespace(topk=[2, 3, 5, 10], size=8)})


def test_category_write_emits_state_and_content_payloads():
    module = _DummyModule()
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0
    content_logits = torch.tensor(
        [
            [[0.1, 0.9, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]],
            [[0.1, 0.2, 0.8, 0.3, 0.4, 0.0, 0.0, 0.0]],
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


def test_category_write_ignores_logits_beyond_vocabulary_snapshot():
    module = _DummyModule()
    state_logits = torch.zeros(1, 1, len(Tokens))
    content_logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 100.0]]])
    prediction = Prediction(
        address=ADDRESS,
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


class _TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.tracked = {}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.tracked[names] = value
        return value


def test_category_loss_does_not_mutate_counters():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    schema = structure
    state = _state()

    field = _tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field.mask(1.0)

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(schema=structure, embedder=embedder, decoder=decoder)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(
                    *field.content.shape,
                    structure.requests[ADDRESS].size,
                ),
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counters[TensorKey.state.name].counts, expected_state_counts)

    expected_content_counts = torch.ones(
        structure.requests[ADDRESS].size,
        dtype=torch.int64,
    )
    assert torch.equal(embedder.counters[TensorKey.content.name].counts, expected_content_counts)


def test_category_loss_uses_uniform_target_for_unavailable_content():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state(size=structure.requests[ADDRESS].size)

    _tensorfield(
        rows=[["ALPHA"]],
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field = _tensorfield(
        rows=[["OMEGA"]],
        schema=structure,
        strata=Strata.validate,
        interprocess_encoding_context=state,
    )
    field.target(1.0)

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(schema=structure, embedder=embedder, decoder=decoder)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape, structure.requests[ADDRESS].size),
            },
            batch_size=field.batch_size,
        ),
    )

    result = loss(module=module, prediction=prediction, batch=field, strata=Strata.validate)

    assert torch.isfinite(result)
    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.validate, "vocabulary", "size")],
        torch.tensor(0.0),
    )
