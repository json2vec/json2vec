"""Tests for Category runtime state accessors."""

from __future__ import annotations

import pytest

import relflow as rf
from relflow.structs.enums import TensorKey
from relflow.tensorfields.shared.counter import CounterUpdateCallback

ADDRESS = rf.Address("record/category")


def _model(*, embed: bool = False) -> rf.Model:
    return rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        attention="none",
        embed=embed,
        category=rf.Category(size=8, p_unavailable=0.0),
    )


def _learn(model: rf.Model, *values: str) -> None:
    model.encode(
        [{"category": value} for value in values],
        strata=rf.Strata.train,
        mask=False,
    )


def _observe(model: rf.Model, *values: str | None) -> None:
    inputs = model.encode(
        [{"category": value} for value in values],
        strata=rf.Strata.train,
        mask=False,
    )
    CounterUpdateCallback().on_train_batch_start(
        trainer=None,
        pl_module=model,
        batch=inputs,
        batch_idx=0,
    )


def test_category_vocabulary_returns_immutable_model_snapshot() -> None:
    model = _model()
    _learn(model, "ALPHA", "BETA")

    snapshot = rf.Category.vocabulary(model, ADDRESS)

    assert snapshot == ("ALPHA", "BETA")
    assert isinstance(snapshot, tuple)

    _learn(model, "GAMMA")

    # A returned snapshot must not be a live view of the model's vocabulary.
    assert snapshot == ("ALPHA", "BETA")
    assert rf.Category.vocabulary(model, "record/category") == (
        "ALPHA",
        "BETA",
        "GAMMA",
    )


def test_category_vocabulary_reads_latest_encoding_context_snapshot() -> None:
    model = _model()
    encoding_context = model.interprocess_encoding_context

    # This context was created before the model vocabulary grew. The binding
    # reads its shared authoritative vocabulary rather than its local index.
    _learn(model, "ALPHA", "BETA")

    assert rf.Category.vocabulary(encoding_context, ADDRESS) == ("ALPHA", "BETA")


def test_category_vocabulary_reads_same_length_context_replacement() -> None:
    model = _model()
    _learn(model, "ALPHA")
    encoding_context = model.interprocess_encoding_context

    model.nodes[ADDRESS].embedder.vocab.load_snapshot(["BETA"])

    assert rf.Category.vocabulary(encoding_context, ADDRESS) == ("BETA",)


def test_category_vocabulary_is_available_to_inference_preprocessor() -> None:
    model = _model(embed=True)
    _learn(model, "ALPHA", "BETA")
    seen: list[tuple[object, ...]] = []
    contexts: list[object] = []

    @rf.preprocess
    def keep_known_category(observation: dict, *, encoding_context):
        contexts.append(encoding_context)
        vocabulary = rf.Category.vocabulary(encoding_context, ADDRESS)
        seen.append(vocabulary)
        value = observation["raw_category"]
        return rf.Observation({"category": value if value in vocabulary else None})

    predictions = model.predict(
        [
            {"raw_category": "ALPHA"},
            {"raw_category": "UNKNOWN"},
        ],
        preprocess=keep_known_category,
    )

    assert seen == [("ALPHA", "BETA"), ("ALPHA", "BETA")]
    assert contexts[0] is contexts[1]
    assert rf.Address("record") in predictions
    assert len(predictions[rf.Address("record")]["embedding"]) == 2


@pytest.mark.parametrize("use_model", [True, False], ids=["model", "encoding-context"])
def test_category_vocabulary_raises_for_missing_address(use_model: bool) -> None:
    model = _model()
    source = model if use_model else model.interprocess_encoding_context

    with pytest.raises(KeyError, match="missing"):
        rf.Category.vocabulary(source, "record/missing")


def test_category_vocabulary_rejects_invalid_context_resource() -> None:
    encoding_context = {ADDRESS: object()}

    with pytest.raises(TypeError, match="VocabularyState"):
        rf.Category.vocabulary(encoding_context, ADDRESS)


def test_category_vocabulary_rejects_invalid_source() -> None:
    with pytest.raises(TypeError, match="Model or InterprocessEncodingContext"):
        rf.Category.vocabulary(object(), ADDRESS)


def test_category_vocabulary_rejects_non_category_model_field() -> None:
    model = rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
    )

    with pytest.raises(TypeError, match="not a Category field"):
        rf.Category.vocabulary(model, "record/amount")


def test_category_counts_returns_populated_training_counts() -> None:
    model = _model()
    _observe(model, "ALPHA", "BETA", "ALPHA", None)

    counts = rf.Category.counts(model, ADDRESS)

    assert counts == {"ALPHA": 2, "BETA": 1}
    assert len(counts) < model.schema.requests[ADDRESS].size

    counts["ALPHA"] = 100
    assert rf.Category.counts(model, ADDRESS) == {"ALPHA": 2, "BETA": 1}


def test_category_counts_subtracts_and_clamps_internal_prior() -> None:
    model = _model()
    _learn(model, "ALPHA")
    counter = model.nodes[ADDRESS].embedder.counters[TensorKey.content.name]
    counter.counts[0] = 0

    assert rf.Category.counts(model, ADDRESS) == {"ALPHA": 0}


def test_category_counts_validates_model_and_address() -> None:
    model = _model()

    with pytest.raises(TypeError, match="must be a Model"):
        rf.Category.counts(model.interprocess_encoding_context, ADDRESS)
    with pytest.raises(KeyError, match="missing"):
        rf.Category.counts(model, "record/missing")

    wrong_model = rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
    )
    with pytest.raises(TypeError, match="not a Category field"):
        rf.Category.counts(wrong_model, "record/amount")
