"""Tests for Category runtime state accessors."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import relflow as rf
from relflow.structs.enums import TensorKey
from relflow.tensorfields.shared.counter import CounterUpdateCallback
from tests.arrow import table

ADDRESS = rf.Address("record/category")


def build(*, embed: bool = False) -> rf.Model:
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


def learn(model: rf.Model, *values: str) -> None:
    model.encode(
        table([{"category": value} for value in values]),
        strata=rf.Strata.train,
        mask=False,
    )


def observe(model: rf.Model, *values: str | None) -> None:
    inputs = model.encode(
        table([{"category": value} for value in values]),
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
    model = build()
    learn(model, "ALPHA", "BETA")

    snapshot = rf.Category.vocabulary(model, ADDRESS)

    assert snapshot == ("ALPHA", "BETA")
    assert isinstance(snapshot, tuple)

    learn(model, "GAMMA")

    # A returned snapshot must not be a live view of the model's vocabulary.
    assert snapshot == ("ALPHA", "BETA")
    assert rf.Category.vocabulary(model, "record/category") == (
        "ALPHA",
        "BETA",
        "GAMMA",
    )


def test_category_vocabulary_reads_latest_encoding_context_snapshot() -> None:
    model = build()
    encoding_context = model.interprocess_encoding_context

    # This context was created before the model vocabulary grew. The binding
    # reads its shared authoritative vocabulary rather than its local index.
    learn(model, "ALPHA", "BETA")

    assert rf.Category.vocabulary(encoding_context, ADDRESS) == ("ALPHA", "BETA")


def test_category_vocabulary_reads_same_length_context_replacement() -> None:
    model = build()
    learn(model, "ALPHA")
    encoding_context = model.interprocess_encoding_context

    model.nodes[ADDRESS].embedder.vocab.load_snapshot(["BETA"])

    assert rf.Category.vocabulary(encoding_context, ADDRESS) == ("BETA",)


def test_category_vocabulary_is_available_to_inference_preprocessor() -> None:
    model = build(embed=True)
    learn(model, "ALPHA", "BETA")
    seen: list[tuple[object, ...]] = []
    contexts: list[object] = []

    @rf.preprocess(requires=("raw_category",), produces=("category",))
    def keep_known_category(batch: rf.Batch, *, encoding_context):
        contexts.append(encoding_context)
        vocabulary = rf.Category.vocabulary(encoding_context, ADDRESS)
        seen.append(vocabulary)
        values = batch.data["raw_category"]
        known = pc.is_in(values, value_set=pa.array(vocabulary, type=values.type))
        category = pc.if_else(known, values, pa.scalar(None, type=values.type))
        return batch.replace(pa.table({"category": category}))

    predictions = model.predict(
        table(
            [
                {"raw_category": "ALPHA"},
                {"raw_category": "UNKNOWN"},
            ]
        ),
        preprocess=keep_known_category,
    )

    assert seen == [("ALPHA", "BETA")]
    assert len(contexts) == 1
    assert rf.Category.vocabulary(contexts[0], ADDRESS) == ("ALPHA", "BETA")
    root = predictions["predictions"].combine_chunks().field("record")
    assert len(root.field("embedding")) == 2


@pytest.mark.parametrize("use_model", [True, False], ids=["model", "encoding-context"])
def test_category_vocabulary_raises_for_missing_address(use_model: bool) -> None:
    model = build()
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
    model = build()
    observe(model, "ALPHA", "BETA", "ALPHA", None)

    counts = rf.Category.counts(model, ADDRESS)

    assert counts == {"ALPHA": 2, "BETA": 1}
    assert len(counts) < model.schema.requests[ADDRESS].size

    counts["ALPHA"] = 100
    assert rf.Category.counts(model, ADDRESS) == {"ALPHA": 2, "BETA": 1}


def test_category_counts_subtracts_and_clamps_internal_prior() -> None:
    model = build()
    learn(model, "ALPHA")
    counter = model.nodes[ADDRESS].embedder.counters[TensorKey.content.name]
    counter.counts[0] = 0

    assert rf.Category.counts(model, ADDRESS) == {"ALPHA": 0}


def test_category_counts_validates_model_and_address() -> None:
    model = build()

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
