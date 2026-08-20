from __future__ import annotations

from typing import Any

import pydantic
import pytest
import torch

from relflow.metrics.extensions.classification import Accuracy, Precision, Recall, Specificity


class TrackingModule:
    def __init__(self):
        self.calls: list[tuple[tuple[Any, ...], torch.Tensor]] = []

    def track(self, names: tuple[Any, ...], /, value: torch.Tensor) -> torch.Tensor:
        self.calls.append((names, value))
        return value


def invoke(
    metric,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    trainable: torch.Tensor,
) -> tuple[object, TrackingModule]:
    module = TrackingModule()
    result = metric(
        module,
        predictions,
        targets,
        trainable,
        address="record/label",
        strata="validate",
        scope=("content",),
    )
    return result, module


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (Accuracy(), 0.5),
        (Precision(), 0.5),
        (Recall(), 0.5),
        (Specificity(), 0.5),
    ],
)
def test_classification_metrics_ignore_nontrainable_wrong_decisions(metric, expected: float):
    predictions = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
    targets = torch.tensor([1, 0, 1, 0, 0, 1])
    trainable = torch.tensor([True, True, True, True, False, False])

    result, module = invoke(metric, predictions, targets, trainable)

    assert result is None
    assert len(module.calls) == 1
    key, value = module.calls[0]
    assert key == ("record/label", "validate", str(metric), "content")
    assert value.ndim == 0
    assert value.item() == pytest.approx(expected)
    assert not value.requires_grad


def test_accuracy_uses_the_same_reduction_for_prepared_multiclass_ids():
    _, module = invoke(
        Accuracy(),
        torch.tensor([0, 1, 1]),
        torch.tensor([0, 2, 1]),
        torch.tensor([True, True, True]),
    )

    assert module.calls[0][1].item() == pytest.approx(2 / 3)


def test_accuracy_positional_threshold_is_named_and_equivalent_to_keyword():
    positional = Accuracy(0.75)
    keyword = Accuracy(threshold=0.75)

    assert positional == keyword
    assert positional.name == "accuracy@{threshold:.2f}"
    assert str(positional) == "accuracy@0.75"

    with pytest.raises(TypeError, match="both positionally and by keyword"):
        Accuracy(0.5, threshold=0.75)


@pytest.mark.parametrize("metric", [Accuracy, Precision, Recall, Specificity])
@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_classification_metrics_validate_threshold(metric, threshold: float):
    with pytest.raises(pydantic.ValidationError):
        metric(threshold=threshold)
