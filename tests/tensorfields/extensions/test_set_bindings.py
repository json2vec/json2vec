"""Tests for Set runtime vocabulary access."""

from __future__ import annotations

import pytest

import relflow as rf
from tests.arrow import table

ADDRESS = rf.Address("record/tags")


def build() -> rf.Model:
    return rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        attention="none",
        tags=rf.Set(size=8, p_unavailable=0.0),
    )


def learn(model: rf.Model, *values: list[str]) -> None:
    model.encode(
        table([{"tags": value} for value in values]),
        strata=rf.Strata.train,
    )


def test_set_vocabulary_returns_immutable_model_snapshot() -> None:
    model = build()
    learn(model, ["ALPHA", "BETA"])

    snapshot = rf.Set.vocabulary(model, ADDRESS)

    assert snapshot == ("ALPHA", "BETA")
    assert isinstance(snapshot, tuple)

    learn(model, ["GAMMA"])

    assert snapshot == ("ALPHA", "BETA")
    assert rf.Set.vocabulary(model, "record/tags") == ("ALPHA", "BETA", "GAMMA")


def test_set_vocabulary_reads_latest_encoding_context_snapshot() -> None:
    model = build()
    encoding_context = model.interprocess_encoding_context

    learn(model, ["ALPHA", "BETA"])

    assert rf.Set.vocabulary(encoding_context, ADDRESS) == ("ALPHA", "BETA")


def test_set_vocabulary_reads_same_length_context_replacement() -> None:
    model = build()
    learn(model, ["ALPHA"])
    encoding_context = model.interprocess_encoding_context

    model.nodes[ADDRESS].embedder.vocab.load_snapshot(["BETA"])

    assert rf.Set.vocabulary(encoding_context, ADDRESS) == ("BETA",)


@pytest.mark.parametrize("use_model", [True, False], ids=["model", "encoding-context"])
def test_set_vocabulary_raises_for_missing_address(use_model: bool) -> None:
    model = build()
    source = model if use_model else model.interprocess_encoding_context

    with pytest.raises(KeyError, match="missing"):
        rf.Set.vocabulary(source, "record/missing")


def test_set_vocabulary_rejects_invalid_context_resource_and_source() -> None:
    with pytest.raises(TypeError, match="VocabularyState"):
        rf.Set.vocabulary({ADDRESS: object()}, ADDRESS)

    with pytest.raises(TypeError, match="Model or InterprocessEncodingContext"):
        rf.Set.vocabulary(object(), ADDRESS)


def test_set_vocabulary_rejects_non_set_model_field() -> None:
    model = rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
    )

    with pytest.raises(TypeError, match="not a Set field"):
        rf.Set.vocabulary(model, "record/amount")
