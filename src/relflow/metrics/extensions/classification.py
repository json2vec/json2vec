"""Stateless metrics for prepared classification decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
import torch

from ..base import Metric, Trait, register

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.enums import Strata
    from relflow.structs.tree import Address

_MISSING = object()
_Threshold = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


@register(Trait.classification)
class Accuracy(Metric):
    """Accuracy of prepared, same-shaped classification decisions."""

    type: Literal["accuracy"] = "accuracy"
    name: str = "accuracy@{threshold:.2f}"
    threshold: _Threshold = 0.5

    def __init__(self, threshold: float | object = _MISSING, /, **data: Any):
        if threshold is not _MISSING:
            if "threshold" in data:
                raise TypeError("threshold was provided both positionally and by keyword")
            data["threshold"] = threshold
        super().__init__(**data)

    @torch.no_grad()
    def __call__(
        self,
        module: Model,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Address,
        strata: Strata,
        scope: tuple[str, ...],
    ) -> None:
        value = predictions.eq(targets).masked_select(trainable).float().mean()
        module.track((address, strata, str(self), *scope), value=value)


@register("boolean")
class Precision(Metric):
    """Precision of prepared binary decisions."""

    type: Literal["precision"] = "precision"
    name: str = "precision@{threshold:.2f}"
    threshold: _Threshold = 0.5

    @torch.no_grad()
    def __call__(
        self,
        module: Model,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Address,
        strata: Strata,
        scope: tuple[str, ...],
    ) -> None:
        predicted = predictions.bool().masked_select(trainable)
        actual = targets.bool().masked_select(trainable)
        true_positive = predicted.logical_and(actual).sum()
        value = true_positive.float() / predicted.sum().clamp_min(1)
        module.track((address, strata, str(self), *scope), value=value)


@register("boolean")
class Recall(Metric):
    """Recall of prepared binary decisions."""

    type: Literal["recall"] = "recall"
    name: str = "recall@{threshold:.2f}"
    threshold: _Threshold = 0.5

    @torch.no_grad()
    def __call__(
        self,
        module: Model,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Address,
        strata: Strata,
        scope: tuple[str, ...],
    ) -> None:
        predicted = predictions.bool().masked_select(trainable)
        actual = targets.bool().masked_select(trainable)
        true_positive = predicted.logical_and(actual).sum()
        value = true_positive.float() / actual.sum().clamp_min(1)
        module.track((address, strata, str(self), *scope), value=value)


@register("boolean")
class Specificity(Metric):
    """Specificity of prepared binary decisions."""

    type: Literal["specificity"] = "specificity"
    name: str = "specificity@{threshold:.2f}"
    threshold: _Threshold = 0.5

    @torch.no_grad()
    def __call__(
        self,
        module: Model,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Address,
        strata: Strata,
        scope: tuple[str, ...],
    ) -> None:
        predicted = predictions.bool().masked_select(trainable)
        actual_negative = targets.bool().masked_select(trainable).logical_not()
        true_negative = predicted.logical_not().logical_and(actual_negative).sum()
        value = true_negative.float() / actual_negative.sum().clamp_min(1)
        module.track((address, strata, str(self), *scope), value=value)


__all__ = ["Accuracy", "Precision", "Recall", "Specificity"]
