"""Shared configuration and tensor operation for affine jitter."""

from typing import Annotated

import pydantic
import torch


class Jitter(pydantic.BaseModel):
    """Configure bounded affine noise for a continuous tensor representation.

    Consumers with separate raw and normalized representations use
    ``normalize`` to select the boundary. Consumers with one continuous
    representation apply noise at that boundary.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    add: Annotated[
        float,
        pydantic.Field(ge=0.0, description="Maximum triangular additive offset."),
    ] = 0.0
    multiply: Annotated[
        float,
        pydantic.Field(ge=0.0, description="Maximum triangular deviation from an identity multiplier."),
    ] = 0.0
    normalize: Annotated[
        pydantic.StrictBool,
        pydantic.Field(description="Prefer the consumer's normalized representation when available."),
    ] = True

    @pydantic.field_validator("add", "multiply", mode="before")
    @classmethod
    def validate_amount(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Jitter.add and Jitter.multiply must be numbers, not booleans")
        return value

    def apply(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply independent triangular ``mx + b`` noise to selected values.

        The caller owns training-mode gating and normalization placement.
        """

        if not self.add and not self.multiply:
            return inputs

        values = inputs.masked_select(mask)
        if self.multiply:
            scale = torch.rand_like(values).sub(torch.rand_like(values)).mul(self.multiply).add(1.0)
            values = values.mul(scale)
        if self.add:
            offset = torch.rand_like(values).sub(torch.rand_like(values)).mul(self.add)
            values = values.add(offset)
        return inputs.masked_scatter(mask, values)

    def __str__(self) -> str:
        return repr(self)


__all__ = ["Jitter"]
