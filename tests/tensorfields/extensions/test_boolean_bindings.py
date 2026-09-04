"""Tests for Boolean runtime count access."""

from __future__ import annotations

import pytest

import relflow as rf
from relflow.structs.enums import TensorKey
from tests.arrow import table

ADDRESS = rf.Address("record/flag")


def build() -> rf.Model:
    return rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=4,
        attention="none",
        flag=rf.Boolean,
    )


def observe(model: rf.Model, *values: bool | None) -> None:
    model.encode(
        table([{"flag": value} for value in values]),
        strata=rf.Strata.train,
    )


def test_boolean_counts_returns_defensive_training_count_snapshot() -> None:
    model = build()
    observe(model, False, True, False, None)

    counts = rf.Boolean.counts(model, ADDRESS)

    assert counts == {False: 2, True: 1}

    counts[False] = 100
    assert rf.Boolean.counts(model, ADDRESS) == {False: 2, True: 1}


def test_boolean_counts_subtracts_and_clamps_internal_prior() -> None:
    model = build()
    assert rf.Boolean.counts(model, ADDRESS) == {False: 0, True: 0}

    counter = model.nodes[ADDRESS].embedder.counters[TensorKey.content.name]
    counter.counts.zero_()

    assert rf.Boolean.counts(model, ADDRESS) == {False: 0, True: 0}


def test_boolean_counts_validates_model_and_address() -> None:
    model = build()

    with pytest.raises(TypeError, match="must be a Model"):
        rf.Boolean.counts({}, ADDRESS)
    with pytest.raises(KeyError, match="missing"):
        rf.Boolean.counts(model, "record/missing")

    wrong_model = rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        category=rf.Category(size=4),
    )
    with pytest.raises(TypeError, match="not a Boolean field"):
        rf.Boolean.counts(wrong_model, "record/category")
