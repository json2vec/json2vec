"""Stateless regression metrics."""

from __future__ import annotations

from typing import Any, Literal

import torch

from ..base import Metric, Trait, register


@register(Trait.regression)
class MAE(Metric):
    """Mean absolute error for one prepared prediction/target pair."""

    type: Literal["mae"] = "mae"
    name: str = "mae"

    @torch.no_grad()
    def __call__(
        self,
        module: Any,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Any,
        strata: Any,
        scope: tuple[str, ...],
    ) -> None:
        value = predictions.sub(targets).abs().masked_select(trainable).mean()
        module.track((address, strata, str(self), *scope), value=value)


@register(Trait.regression)
class RMSE(Metric):
    """Root mean squared error for one prepared prediction/target pair."""

    type: Literal["rmse"] = "rmse"
    name: str = "rmse"

    @torch.no_grad()
    def __call__(
        self,
        module: Any,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Any,
        strata: Any,
        scope: tuple[str, ...],
    ) -> None:
        value = predictions.sub(targets).square().masked_select(trainable).mean().sqrt()
        module.track((address, strata, str(self), *scope), value=value)


__all__ = ["MAE", "RMSE"]
