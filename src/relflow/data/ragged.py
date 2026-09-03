"""Coalesce schema-shaped inputs into canonical Arrow-backed fields."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np
import pyarrow as pa

from relflow.data.arrow import Batch
from relflow.data.query import Projection, query
from relflow.structs.enums import Overflow, Strata, Tokens
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


def member(name: str) -> str:
    """Render a schema name as one native query member."""

    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) else f"[{json.dumps(name)}]"


def append(expression: str, selector: str) -> str:
    """Append one node-relative selector to an accumulated path."""

    if not expression:
        return selector
    return f"{expression}{selector}" if selector.startswith("[") else f"{expression}.{selector}"


def expression(request: Any, root: Branch) -> str:
    """Compose one leaf's node-relative selectors into an executable path."""

    result = root.query or ""
    for node in request.path:
        if node is root or node is request or getattr(node, "type", None) != "branch":
            continue
        result = append(result, node.query or member(node.name))
        result = f"{result}[*]"

    return append(result, request.query or member(request.name))


def extract(
    selected: Projection,
    *,
    address: Address,
    leaf_axis: int,
) -> tuple[ak.Array, ak.Array]:
    """Lower an Arrow projection to Awkward values and model-state slots."""

    try:
        values = ak.from_arrow(selected.values)[:, np.newaxis]
        present = ak.from_arrow(selected.present)[:, np.newaxis]
        if present.ndim > leaf_axis + 1:
            available = ~ak.is_none(present, axis=leaf_axis)
        else:
            available = ak.fill_none(present, False, axis=leaf_axis)
        null = ak.is_none(values, axis=leaf_axis)
        states = ak.where(
            available,
            ak.where(null, Tokens.null.value, Tokens.valued.value),
            Tokens.padded.value,
        )

        return values, ak.values_astype(states, np.int8)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise TypeError(f"Arrow query result for address '{address}' is not Awkward-compatible") from error


def regularize(
    values: ak.Array,
    states: ak.Array,
    *,
    shape: tuple[int, ...],
    overflows: tuple[Overflow, ...],
    address: Address,
) -> tuple[ak.Array, ak.Array]:
    if len(shape) != len(overflows):
        raise ValueError(f"overflows length must match shape rank: expected {len(shape)}, got {len(overflows)}")

    for axis, (capacity, overflow) in enumerate(zip(shape, overflows, strict=True)):
        # Padding an outer axis creates option-valued parent slots. Before
        # descending, turn those absent parents into empty lists so the next
        # declared model axis can be regularized normally.
        if axis > 0:
            values = ak.fill_none(values, [], axis=axis - 1)
            states = ak.fill_none(states, [], axis=axis - 1)

        overflow = Overflow(overflow)
        if overflow == Overflow.error:
            try:
                exceeds = ak.num(states, axis=axis) > capacity
            except ak.errors.AxisError:
                exceeds = False
            overflowed = bool(ak.any(exceeds, axis=None)) if isinstance(exceeds, ak.Array) else bool(exceeds)
            if overflowed:
                context = (
                    "batch dimension 0" if axis == 0 else "root node dimension 1" if axis == 1 else f"dimension {axis}"
                )
                raise ValueError(f"branch overflow at {context} for {address}: capacity is {capacity}")

        start, stop = (-capacity, None) if overflow == Overflow.tail else (0, capacity)
        slices = (*((slice(None),) * axis), slice(start, stop))
        values = values[slices]
        states = states[slices]

        values = ak.pad_none(values, capacity, axis=axis, clip=True)
        states = ak.pad_none(states, capacity, axis=axis, clip=True)

    return values, states


@dataclass(frozen=True, slots=True)
class RaggedField:
    """Arrow-backed retained values and their dense model geometry."""

    values: pa.Array | pa.ChunkedArray
    state: pa.Array | pa.ChunkedArray
    placement: pa.Array | pa.ChunkedArray
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, (pa.Array, pa.ChunkedArray)):
            raise TypeError(f"RaggedField.values must be an Arrow array, got {type(self.values).__name__}")
        if not isinstance(self.state, (pa.Array, pa.ChunkedArray)) or self.state.type != pa.int8():
            datatype = getattr(self.state, "type", type(self.state).__name__)
            raise TypeError(f"RaggedField.state must be an int8 Arrow array, got {datatype}")
        if not isinstance(self.placement, (pa.Array, pa.ChunkedArray)) or self.placement.type != pa.int64():
            datatype = getattr(self.placement, "type", type(self.placement).__name__)
            raise TypeError(f"RaggedField.placement must be an int64 Arrow array, got {datatype}")
        if self.state.null_count or self.placement.null_count:
            raise ValueError("RaggedField state and placement cannot contain nulls")
        if self.values.null_count:
            raise ValueError("RaggedField values cannot contain nulls")
        if not self.shape or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in self.shape
        ):
            raise ValueError("RaggedField.shape must contain non-negative integer dimensions")
        if len(self.state) != math.prod(self.shape):
            raise ValueError(
                f"RaggedField state length must equal shape product {math.prod(self.shape)}, got {len(self.state)}"
            )
        if len(self.values) != len(self.placement):
            raise ValueError(
                f"RaggedField values/placement length mismatch: {len(self.values)} values, "
                f"{len(self.placement)} placements"
            )
        states = self.state.to_numpy(zero_copy_only=False)
        allowed = np.asarray([token.value for token in Tokens], dtype=np.int8)
        if np.any(~np.isin(states, allowed)):
            raise ValueError("RaggedField state contains an unknown token")

        positions = self.placement.to_numpy(zero_copy_only=False)
        if len(positions) and (np.any(positions < 0) or np.any(positions >= len(self.state))):
            raise ValueError("RaggedField placement contains an out-of-range dense position")
        if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError("RaggedField placement must be strictly increasing")
        expected = np.flatnonzero(states == Tokens.valued.value)
        if not np.array_equal(positions, expected):
            raise ValueError("RaggedField placement must contain every valued state position exactly once")

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def dense(self) -> np.ndarray:
        """Return model state as one dense int64 NumPy view."""

        values = self.state.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        return values.reshape(self.shape)

    def place(
        self,
        encoded: np.ndarray,
        *,
        fill: Any,
        value_shape: tuple[int, ...] = (),
    ) -> np.ndarray:
        """Scatter encoded retained values into this field's dense geometry."""
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in value_shape):
            raise ValueError("value_shape entries must be non-negative integers")

        array = np.asarray(encoded)
        expected = (len(self.values), *value_shape)
        if array.shape != expected:
            raise ValueError(f"encoded values must have shape {expected}, got {array.shape}")

        placed = np.full((*self.shape, *value_shape), fill, dtype=array.dtype)
        flat_shape = (len(self.state), *value_shape)
        placement = self.placement.to_numpy(zero_copy_only=False)
        placed.reshape(flat_shape)[placement] = array
        return placed


def coalesce(
    values: Batch,
    schema: Schema,
    strata: Strata | str,
) -> dict[Address, RaggedField]:
    """Prepare every encoded field before datatype codecs mutate shared state."""
    if not isinstance(values, Batch):
        raise TypeError(f"coalesce values must be an Arrow Batch, got {type(values).__name__}")

    strata = Strata.normalize(strata)
    predict_targets = set(schema.target) if strata == Strata.predict else set()
    addresses = tuple(address for address in schema.active_requests if address not in predict_targets)
    paths = {address: expression(schema.requests[address], schema.fields) for address in addresses}
    batch_size = len(values)

    if not addresses:
        return {}

    fields: dict[Address, RaggedField] = {}
    for address in addresses:
        shape = (batch_size, *schema.shapes[address])
        selected = query(
            values.data,
            paths[address],
            address=address,
        )
        try:
            projected_values, projected_states = extract(
                selected,
                address=address,
                leaf_axis=len(shape) - 1,
            )
        except (TypeError, IndexError, ak.errors.AxisError) as error:
            raise ValueError(
                f"Arrow query for address '{address}' does not match its schema shape: "
                f"{paths[address]!r} must produce {len(shape)} list axes"
            ) from error

        try:
            dense_values, dense_states = regularize(
                projected_values,
                projected_states,
                shape=shape,
                overflows=schema.overflows(address),
                address=address,
            )
            state = np.asarray(
                ak.to_numpy(ak.fill_none(dense_states, Tokens.padded.value, axis=len(shape) - 1)),
                dtype=np.int64,
            )
            for _ in range(len(shape) - 1):
                dense_values = ak.flatten(dense_values, axis=1)
        except (IndexError, ak.errors.AxisError) as error:
            raise ValueError(
                f"Arrow query for address '{address}' does not match its schema shape: "
                f"{paths[address]!r} must produce {len(shape)} list axes"
            ) from error

        if state.shape != shape:
            raise ValueError(f"coalesced state for address '{address}' must have shape {shape}, got {state.shape}")
        placement = np.flatnonzero(state.ravel() == Tokens.valued.value).astype(np.int64, copy=False)
        retained = ak.to_arrow(dense_values[placement], extensionarray=False)
        if not isinstance(retained, pa.Array):
            raise TypeError(f"coalesced values for address '{address}' did not produce an Arrow array")
        fields[address] = RaggedField(
            values=retained,
            state=pa.array(state.reshape(-1), type=pa.int8()),
            placement=pa.array(placement, type=pa.int64()),
            shape=shape,
        )

    return fields


__all__ = ["RaggedField"]
