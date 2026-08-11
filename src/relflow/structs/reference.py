"""Exact schema references and their optional named-axis reductions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import pydantic

from relflow.structs.pooling import Attention, Convolution, Mean, PoolingConfig, StrictPositiveInt
from relflow.structs.tree import Address


class AxisName(str):
    """An unresolved short Branch-axis name used while constructing a schema."""

    def __new__(cls, value: str) -> "AxisName":
        if not isinstance(value, str):
            raise TypeError("axis names must be strings")
        if not value:
            raise ValueError("axis names must be non-empty")
        if "/" in value:
            raise ValueError("short axis names cannot contain '/'; use Address for a full axis address")
        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


_MISSING = object()


class AxisResize(pydantic.BaseModel):
    """Resize one named schema axis to a positive retained extent."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    address: AxisName | Address
    size: StrictPositiveInt

    def __init__(self, address: str | Address | AxisName | object = _MISSING, size: int | object = _MISSING, /, **data):
        if address is not _MISSING:
            if "address" in data:
                raise TypeError("address was provided both positionally and by keyword")
            data["address"] = address
        if size is not _MISSING:
            if "size" in data:
                raise TypeError("size was provided both positionally and by keyword")
            data["size"] = size
        super().__init__(**data)

    @pydantic.field_validator("address", mode="before")
    @classmethod
    def normalize_address(cls, value: Any) -> Address | AxisName:
        if isinstance(value, Address):
            if not value:
                raise ValueError("axis addresses must be non-empty")
            return value
        if isinstance(value, AxisName):
            return value
        if not isinstance(value, str):
            raise TypeError("axis addresses must be strings or Address values")
        if not value:
            raise ValueError("axis addresses must be non-empty")
        return Address(value) if "/" in value else AxisName(value)


ReductionName: TypeAlias = Literal["sum", "min", "max", "prod"]
Reducer: TypeAlias = PoolingConfig | ReductionName


def _normalize_size(value: Any) -> int | None:
    if value is False:
        return None
    if value is True:
        return 1
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("Reduce axis targets must be booleans or positive integers")
    if value <= 0:
        raise ValueError("Reduce axis targets must be positive")
    return value


class Reduce(pydantic.BaseModel):
    """Block-reduce named schema dimensions before routing a Reference."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    reducer: Reducer = pydantic.Field(default_factory=Mean)
    axes: tuple[AxisResize, ...] = ()

    def __init__(
        self,
        reducer: Reducer | Literal["mean"] | object = _MISSING,
        /,
        *,
        axes: Mapping[str | Address | AxisName, bool | int]
        | Sequence[AxisResize | Mapping[str, Any]]
        | object = _MISSING,
        **targets: bool | int,
    ):
        data: dict[str, Any] = {}

        if reducer is _MISSING and "reducer" in targets:
            reducer = targets.pop("reducer")
        if reducer is not _MISSING:
            data["reducer"] = reducer

        if axes is _MISSING and "axes" in targets:
            axes = targets.pop("axes")

        entries: list[AxisResize | Mapping[str, Any]] = []
        if axes is not _MISSING:
            if isinstance(axes, Mapping):
                entries.extend(
                    AxisResize(address, size)
                    for address, target in axes.items()
                    if (size := _normalize_size(target)) is not None
                )
            elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
                entries.extend(axes)
            else:
                raise TypeError("Reduce axes must be a mapping or sequence of AxisResize values")

        entries.extend(
            AxisResize(AxisName(name), size)
            for name, target in targets.items()
            if (size := _normalize_size(target)) is not None
        )
        data["axes"] = tuple(entries)
        super().__init__(**data)

    @pydantic.field_validator("reducer", mode="before")
    @classmethod
    def normalize_reducer(cls, value: Any) -> Any:
        return Mean() if value == "mean" else value

    @pydantic.model_validator(mode="after")
    def check_axes_and_reducer(self):
        seen: set[tuple[type, str]] = set()
        for axis in self.axes:
            key = (type(axis.address), str(axis.address))
            if key in seen:
                raise ValueError(f"duplicate Reduce axis: {axis.address}")
            seen.add(key)

        if isinstance(self.reducer, (Mean, Attention, Convolution)) and self.reducer.width not in (None, 1):
            raise ValueError("Reference reducers must have width=None or width=1")
        if isinstance(self.reducer, Convolution) and len(self.axes) > 1:
            raise ValueError("Convolution can reduce exactly one named axis")
        return self


class Reference(pydantic.BaseModel):
    """An exact pointer from one Branch to another schema node."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    address: Address
    graft: pydantic.StrictBool = False
    reduce: Reduce | None = None

    def __init__(self, address: str | Address | object = _MISSING, /, **data):
        if address is not _MISSING:
            if "address" in data:
                raise TypeError("address was provided both positionally and by keyword")
            data["address"] = address
        super().__init__(**data)

    @pydantic.field_validator("address", mode="before")
    @classmethod
    def normalize_address(cls, value: Any) -> Address:
        if not isinstance(value, str):
            raise TypeError("Reference address must be a string or Address")
        if not value:
            raise ValueError("Reference address must be non-empty")
        return Address(value)

    @pydantic.field_validator("reduce", mode="before")
    @classmethod
    def normalize_reduce(cls, value: Any) -> Reduce | None:
        if value is None:
            return None
        reduction = value if isinstance(value, Reduce) else Reduce.model_validate(value)
        return reduction if reduction.axes else None


__all__ = [
    "AxisName",
    "AxisResize",
    "Reduce",
    "Reducer",
    "Reference",
]
