"""Stateless metrics for cyclic date-part representations."""

from __future__ import annotations

from typing import Any, Literal

import torch

from ..base import Metric, Trait, register


@register(Trait.cyclic, "dateparts")
class AngularMAE(Metric):
    """Mean unsigned angular error, in radians, between sin/cos pairs."""

    type: Literal["angular_mae"] = "angular_mae"
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
        cosine = (predictions * targets).sum(dim=-1)
        value = cosine.clamp(min=-1.0, max=1.0).arccos().masked_select(trainable).mean()
        module.track((address, strata, str(self), *scope), value=value)


__all__ = ["AngularMAE"]
