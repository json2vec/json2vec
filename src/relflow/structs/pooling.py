"""Frozen schema configuration for memory pooling."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic

StrictPositiveInt: TypeAlias = Annotated[pydantic.StrictInt, pydantic.Field(gt=0)]
PoolDropout: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]


class Mean(pydantic.BaseModel):
    """Arithmetic-mean pooling configuration."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["mean"] = "mean"
    width: StrictPositiveInt | None = None


class Attention(pydantic.BaseModel):
    """Learned-query attention pooling configuration."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["attention"] = "attention"
    width: StrictPositiveInt | None = None
    n_heads: StrictPositiveInt | None = None
    n_layers: StrictPositiveInt = 1
    dropout: PoolDropout | None = None


class Convolution(pydantic.BaseModel):
    """Residual convolution pooling configuration."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["convolution"] = "convolution"
    width: StrictPositiveInt | None = None
    kernel_size: StrictPositiveInt = 3
    n_layers: StrictPositiveInt = 1
    dropout: PoolDropout | None = None

    @pydantic.field_validator("kernel_size")
    @classmethod
    def check_odd_kernel_size(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return value


PoolingConfig: TypeAlias = Annotated[
    Mean | Attention | Convolution,
    pydantic.Field(discriminator="type"),
]


__all__ = [
    "Attention",
    "Convolution",
    "Mean",
    "PoolingConfig",
    "StrictPositiveInt",
]
