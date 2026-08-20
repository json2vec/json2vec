from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from relflow.metrics.extensions.dateparts import AngularMAE


class TrackingModule:
    def __init__(self):
        self.calls: list[tuple[tuple[Any, ...], torch.Tensor]] = []

    def track(self, names: tuple[Any, ...], /, value: torch.Tensor) -> torch.Tensor:
        self.calls.append((names, value))
        return value


def test_angular_mae_ignores_nontrainable_wrong_pairs():
    module = TrackingModule()
    metric = AngularMAE()
    predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    trainable = torch.tensor([True, True, False])

    result = metric(
        module,
        predictions,
        targets,
        trainable,
        address="record/created_at",
        strata="validate",
        scope=("content", "hour_of_day"),
    )

    assert result is None
    assert len(module.calls) == 1
    key, value = module.calls[0]
    assert key == ("record/created_at", "validate", "mae", "content", "hour_of_day")
    assert value.item() == pytest.approx(math.pi / 4)
    assert value.ndim == 0
    assert not value.requires_grad
