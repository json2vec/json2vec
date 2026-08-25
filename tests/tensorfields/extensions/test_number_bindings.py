"""Tests for Number runtime normalization access."""

from __future__ import annotations

import math

import pytest
import torch

import relflow as rf

ADDRESS = rf.Address("record/amount")


def _model(*, alpha: float | None = None) -> rf.Model:
    return rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        attention="none",
        amount=rf.Number(alpha=alpha),
    )


def test_number_normalization_returns_cumulative_snapshot() -> None:
    model = _model()
    normalizer = model.nodes[ADDRESS].embedder.normalizer
    normalizer.update(torch.tensor([1.0, 3.0]))

    snapshot = rf.Number.normalization(model, ADDRESS)

    assert snapshot == pytest.approx(
        {
            "mean": 2.0,
            "variance": 1.0,
            "std": math.sqrt(1.0 + normalizer.epsilon),
            "count": 2,
            "alpha": None,
        }
    )
    assert isinstance(snapshot["mean"], float)
    assert isinstance(snapshot["variance"], float)
    assert isinstance(snapshot["std"], float)
    assert isinstance(snapshot["count"], int)


def test_number_normalization_reports_ema_configuration_without_count() -> None:
    model = _model(alpha=0.25)
    normalizer = model.nodes[ADDRESS].embedder.normalizer
    normalizer.update(torch.tensor([2.0, 4.0]))

    snapshot = rf.Number.normalization(model, "record/amount")

    assert snapshot["mean"] == pytest.approx(0.75)
    assert snapshot["variance"] == pytest.approx(1.0)
    assert snapshot["std"] == pytest.approx(math.sqrt(1.0 + normalizer.epsilon))
    assert snapshot["count"] is None
    assert snapshot["alpha"] == 0.25


def test_number_normalization_count_remains_exact_above_float32_limit() -> None:
    model = _model()
    normalizer = model.nodes[ADDRESS].embedder.normalizer
    normalizer.count.fill_(2**24)

    normalizer.update(torch.tensor([1.0]))

    assert rf.Number.normalization(model, ADDRESS)["count"] == 2**24 + 1


def test_number_normalization_rejects_invalid_model_and_address() -> None:
    with pytest.raises(TypeError, match="model must be a Model"):
        rf.Number.normalization(object(), ADDRESS)

    model = _model()
    with pytest.raises(KeyError, match="missing"):
        rf.Number.normalization(model, "record/missing")


def test_number_normalization_rejects_non_number_model_field() -> None:
    model = rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        category=rf.Category(size=8),
    )

    with pytest.raises(TypeError, match="not a Number field"):
        rf.Number.normalization(model, "record/category")
