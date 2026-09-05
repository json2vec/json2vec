from types import SimpleNamespace

import polars as pl
import torch
from tensordict import TensorDict

from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.category import (
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel

ADDRESS = "root/items/category"


def _structure_payload(*, topk: list[int] | None = None, p_unavailable: float | None = None) -> dict:
    field: dict = {
        "name": "category",
        "type": "category",
        "query": "[*].items[*].label",
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

    field = TensorField.new(
        values=[[["ALPHA", None]], [["BETA"]]],
        address=ADDRESS,
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

    TensorField.new(
        values=[[["ALPHA"]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    field = TensorField.new(
        values=[[["OMEGA"]]],
        address=ADDRESS,
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

    field = TensorField.new(
        values=[[["ALPHA", None]], [["BETA"]]],
        address=ADDRESS,
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
    assert decoder.linears[TensorKey.content.name].out_features == structure.d_model


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


def test_category_embedder_only_adds_content_for_available_valued_tokens():
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
        content=torch.tensor([[2, unavailable, 3, 4, 5, 6]], dtype=torch.int64),
        trainable=torch.zeros((1, 6), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    output = embedder(field).payload
    state_embedding = embedder.embeddings[TensorKey.state.name](field.state)
    content_contribution = output - state_embedding
    expected = torch.zeros_like(content_contribution)
    expected[0, 0] = torch.nn.functional.normalize(
        embedder.embeddings[TensorKey.content.name].weight[2],
        dim=-1,
    )

    assert torch.allclose(content_contribution, expected, atol=1e-6)
    assert torch.allclose(content_contribution[0, 0].norm(), torch.tensor(1.0))

    output.sum().backward()
    content_gradient = embedder.embeddings[TensorKey.content.name].weight.grad
    state_gradient = embedder.embeddings[TensorKey.state.name].weight.grad
    assert content_gradient is not None
    assert state_gradient is not None
    assert content_gradient[2].abs().sum() > 0
    assert torch.count_nonzero(content_gradient[:2]) == 0
    assert torch.count_nonzero(content_gradient[3:]) == 0
    assert torch.all(state_gradient.abs().sum(dim=-1) > 0)


def test_category_embedder_does_not_renormalize_content_table_at_forward(monkeypatch):
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)
    field = TensorField(
        state=torch.tensor([[Tokens.valued.value, Tokens.null.value]], dtype=torch.int64),
        content=torch.tensor([[1, 0]], dtype=torch.int64),
        trainable=torch.zeros((1, 2), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )
    content_weight = embedder.embeddings[TensorKey.content.name].weight
    content_id = id(content_weight)
    normalized_ids: list[int] = []
    normalize = torch.nn.functional.normalize

    def track_normalize(inputs: torch.Tensor, *args, **kwargs):
        normalized_ids.append(id(inputs))
        return normalize(inputs, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "normalize", track_normalize)

    embedder(field)

    assert content_id not in normalized_ids


def test_category_embedder_content_directions_are_unit_at_init():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)

    norms = embedder.content_directions().norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_category_normalize_content_directions_restores_unit_rows():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)

    with torch.no_grad():
        embedder.embeddings[TensorKey.content.name].weight.mul_(3.0)

    pre_norms = embedder.content_directions().norm(dim=-1)
    assert torch.allclose(pre_norms, torch.full_like(pre_norms, 3.0), atol=1e-6)

    embedder.normalize_content_directions()

    post_norms = embedder.content_directions().norm(dim=-1)
    assert torch.allclose(post_norms, torch.ones_like(post_norms), atol=1e-6)


def test_category_cosface_is_finite_for_zero_float16_query_and_gradient():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS).half()
    query = torch.zeros(2, structure.d_model, dtype=torch.float16, requires_grad=True)

    cosine = embedder.content_cosine(query)
    targets = torch.tensor([0, 1])
    one_hot = torch.nn.functional.one_hot(targets, num_classes=embedder.size)
    logits = structure.requests[ADDRESS].scale * (cosine - embedder.margin * one_hot)
    content_loss = torch.nn.functional.cross_entropy(logits, targets)
    content_loss.backward()

    assert torch.isfinite(cosine).all()
    assert torch.count_nonzero(cosine) == 0
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert embedder.embeddings[TensorKey.content.name].weight.grad is not None
    assert torch.isfinite(embedder.embeddings[TensorKey.content.name].weight.grad).all()


class _DummyVocab:
    def snapshot(self) -> list[str]:
        return ["ALPHA", "BETA", "GAMMA", "DELTA", "EPS"]


def _dummy_write_module(*, d_model: int = 4, scale: float = 30.0, topk: list[int] | None = None):
    payload = _structure_payload(topk=topk or [2, 3, 5])
    payload["d_model"] = d_model
    payload["fields"]["fields"][0]["fields"][0]["size"] = d_model
    structure = Schema.model_validate(payload)
    structure.requests[ADDRESS].scale = scale

    embedder = Embedder(schema=structure, address=ADDRESS)
    for token in _DummyVocab().snapshot():
        embedder.vocab.state.reserve(token, learn=True)

    with torch.no_grad():
        weight = embedder.embeddings[TensorKey.content.name].weight
        weight.zero_()
        for index in range(len(_DummyVocab().snapshot())):
            weight[index, index] = 1.0

    module = SimpleNamespace(
        nodes={ADDRESS: SimpleNamespace(embedder=embedder)},
        schema=structure,
    )
    return module, embedder, d_model


def _basis_query(shape: tuple[int, ...], index: int, d_model: int) -> torch.Tensor:
    query = torch.zeros(*shape, d_model)
    query[..., index] = 1.0
    return query


def test_category_write_emits_state_and_content_payloads():
    module, _embedder, d_model = _dummy_write_module(d_model=8, topk=[2, 3, 5])
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0
    content_query = torch.stack([_basis_query((1,), 1, d_model), _basis_query((1,), 2, d_model)], dim=0)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_query,
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
    module, _embedder, d_model = _dummy_write_module(d_model=8, topk=[2, 3, 5])
    query = _basis_query((1, 1), 4, d_model)
    state_logits = torch.zeros(1, 1, len(Tokens))
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: query,
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

    field = TensorField.new(
        values=[[["ALPHA", None]], [["BETA"]]],
        address=ADDRESS,
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
                    structure.d_model,
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

    TensorField.new(
        values=[[["ALPHA"]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field = TensorField.new(
        values=[[["OMEGA"]]],
        address=ADDRESS,
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
                TensorKey.content: torch.zeros(*field.content.shape, structure.d_model),
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
    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.validate, Metric.loss, TensorKey.content)],
        torch.log(torch.tensor(float(structure.requests[ADDRESS].size))),
    )


def test_category_loss_scores_only_valued_targets_and_matches_cosface_reference(monkeypatch):
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state(size=structure.requests[ADDRESS].size)
    field = TensorField.new(
        values=[[["ALPHA", None]], [["BETA"]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field.mask(1.0)
    valued = field.trainable & field.targets[TensorKey.state].eq(Tokens.valued.value)
    valued_indices = valued.reshape(-1).nonzero().flatten()
    field.targets[TensorKey.content].reshape(-1)[valued_indices[-1]] = structure.requests[ADDRESS].size

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(schema=structure, embedder=embedder, decoder=decoder)
    torch.manual_seed(0)
    content_query = torch.randn(*field.content.shape, structure.d_model, requires_grad=True)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: content_query,
            },
            batch_size=field.batch_size,
        ),
    )

    state_targets = field.targets[TensorKey.state]
    valued = field.trainable & state_targets.eq(Tokens.valued.value)
    valued_query = content_query[valued]
    valued_targets = field.targets[TensorKey.content][valued]
    cosine = embedder.content_cosine(valued_query)
    available = valued_targets.lt(structure.requests[ADDRESS].size)
    unavailable = valued_targets.eq(structure.requests[ADDRESS].size)
    cos_available = cosine[available]
    tgt_available = valued_targets[available]
    one_hot = torch.nn.functional.one_hot(
        tgt_available,
        num_classes=structure.requests[ADDRESS].size,
    )
    reference_logits = structure.requests[ADDRESS].scale * (cos_available - embedder.margin * one_hot)
    reference_available_loss = torch.nn.functional.cross_entropy(
        input=reference_logits,
        target=tgt_available,
        weight=embedder.counters[TensorKey.content.name].weight,
        reduction="none",
    ).sum()
    reference_unavailable_loss = (
        -torch.nn.functional.log_softmax(
            structure.requests[ADDRESS].scale * cosine[unavailable],
            dim=1,
        )
        .mean(dim=1)
        .sum()
    )
    reference_loss = (reference_available_loss + reference_unavailable_loss) / valued_targets.numel()
    reference_query_gradient, reference_weight_gradient = torch.autograd.grad(
        reference_loss,
        (
            content_query,
            embedder.embeddings[TensorKey.content.name].weight,
        ),
        retain_graph=True,
    )

    query_shapes: list[tuple[int, ...]] = []
    content_cosine = embedder.content_cosine

    def track_content_cosine(query: torch.Tensor, indices: torch.Tensor | None = None):
        query_shapes.append(tuple(query.shape))
        return content_cosine(query=query, indices=indices)

    monkeypatch.setattr(embedder, "content_cosine", track_content_cosine)

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    optimized_loss = module.tracked[(ADDRESS, Strata.train, Metric.loss, TensorKey.content)]
    optimized_query_gradient, optimized_weight_gradient = torch.autograd.grad(
        optimized_loss,
        (
            content_query,
            embedder.embeddings[TensorKey.content.name].weight,
        ),
    )

    assert query_shapes == [(int(valued.sum()), structure.d_model)]
    assert torch.allclose(optimized_loss, reference_loss)
    assert torch.allclose(optimized_query_gradient, reference_query_gradient)
    assert torch.allclose(optimized_weight_gradient, reference_weight_gradient)
    assert torch.count_nonzero(optimized_query_gradient[~valued]) == 0
    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.train, "embedding", "valued_state_norm")],
        embedder.embeddings[TensorKey.state.name].weight[Tokens.valued.value].norm(),
    )


def test_category_loss_reuses_nested_rankings_and_excludes_unavailable_from_metrics():
    payload = _structure_payload(topk=[2, 3], p_unavailable=0.0)
    payload["d_model"] = 8
    structure = Schema.model_validate(payload)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(schema=structure, embedder=embedder, decoder=decoder)
    unavailable = structure.requests[ADDRESS].size
    with torch.no_grad():
        embedder.embeddings[TensorKey.content.name].weight.copy_(torch.eye(8))

    field = TensorField(
        state=torch.full((1, 3), Tokens.masked.value, dtype=torch.int64),
        content=torch.zeros((1, 3), dtype=torch.int64),
        trainable=torch.ones((1, 3), dtype=torch.bool),
        targets=TensorDict(
            {
                TensorKey.state: torch.full((1, 3), Tokens.valued.value, dtype=torch.int64),
                TensorKey.content: torch.tensor([[0, 1, unavailable]], dtype=torch.int64),
            }
        ),
        batch_size=1,
    )
    content_query = torch.tensor(
        [
            [
                [0.3, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(1, 3, len(Tokens)),
                TensorKey.content: content_query,
            },
            batch_size=1,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.train, Metric.accuracy, TensorKey.content)],
        torch.tensor(0.5),
    )
    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.train, Metric.accuracy, "top2")],
        torch.tensor(0.5),
    )
    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.train, Metric.accuracy, "top3")],
        torch.tensor(1.0),
    )


def test_category_content_loss_stays_finite_and_bounded_for_large_vocabulary():
    payload = _structure_payload(p_unavailable=0.0)
    payload["fields"]["fields"][0]["fields"][0]["size"] = 4096
    structure = Schema.model_validate(payload)
    d_model = structure.d_model
    scale = structure.requests[ADDRESS].scale
    state = _state(size=structure.requests[ADDRESS].size)

    field = TensorField.new(
        values=[[["ALPHA", "BETA"]], [["GAMMA", "DELTA"]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field.mask(1.0)

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(schema=structure, embedder=embedder, decoder=decoder)

    torch.manual_seed(0)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.randn(*field.content.shape, d_model),
            },
            batch_size=field.batch_size,
        ),
    )

    result = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    assert torch.isfinite(result)
    tracked_content = module.tracked[(ADDRESS, Strata.train, Metric.loss, TensorKey.content)]
    assert torch.isfinite(tracked_content)
    assert tracked_content.item() <= 2.0 * scale + float(torch.log(torch.tensor(4096.0))) + 5.0
