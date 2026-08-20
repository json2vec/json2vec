from __future__ import annotations

from typing import Any

import pytest
import torch

from relflow.metrics.extensions.regression import MAE, RMSE


class TrackingModule:
    def __init__(self):
        self.calls: list[tuple[tuple[Any, ...], torch.Tensor]] = []

    def track(self, names: tuple[Any, ...], /, value: torch.Tensor) -> torch.Tensor:
        self.calls.append((names, value))
        return value


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (MAE(), 5 / 3),
        (RMSE(), 3**0.5),
    ],
)
def test_regression_metrics_ignore_nontrainable_extreme_errors(metric, expected: float):
    module = TrackingModule()
    predictions = torch.tensor([1.0, 3.0, -1.0, 1_000_000.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0, -1_000_000.0])
    trainable = torch.tensor([True, True, True, False])

    result = metric(
        module,
        predictions,
        targets,
        trainable,
        address="record/amount",
        strata="test",
        scope=("content",),
    )

    assert result is None
    assert len(module.calls) == 1
    key, value = module.calls[0]
    assert key == ("record/amount", "test", str(metric), "content")
    assert value.item() == pytest.approx(expected)
    assert value.ndim == 0
    assert not value.requires_grad
