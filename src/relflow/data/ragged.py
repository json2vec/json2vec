"""Coalesce schema-shaped inputs into canonical Awkward-backed fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import awkward as ak
import jmespath
import numpy as np
from jmespath.exceptions import JMESPathError

from relflow.structs.enums import Overflow, Strata, Tokens
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.data.datasets.base import EncodedBatch
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch
    from relflow.tensorfields.base import Plugin, ValueTypeObservation, ValueTypeObservations

MASK_LITERAL = "<MASK>"
MaskLiteral: TypeAlias = Literal["<MASK>"]

_SEQUENCE_TYPES = (list, tuple, np.ndarray)
_AWKWARD_CONVERSION_ERRORS = (TypeError, ValueError, RuntimeError, OverflowError)


@cache
def compile(expression: str) -> jmespath.parser.ParsedResult:
    try:
        return jmespath.compile(f"[*]{expression}")
    except JMESPathError as error:
        raise ValueError(f"invalid JMESPath query {expression!r}: {error}") from error


def sequence(value: Any) -> bool:
    return isinstance(value, _SEQUENCE_TYPES) and not (isinstance(value, np.ndarray) and value.ndim == 0)


def prepare(
    value: Any,
    *,
    address: Address,
    plugin: Plugin,
) -> tuple[Any, int, ValueTypeObservations]:
    if type(value) in (str, np.str_) and str(value) == MASK_LITERAL:
        return None, Tokens.masked.value, None

    prepared, observed = plugin.prepare(value, address=address)
    state = Tokens.null.value if prepared is None else Tokens.valued.value
    return prepared, state, observed


def normalize(
    record: Any,
    branch: Branch,
    direct_branches: frozenset[Address],
    plugins: Mapping[Address, Plugin],
    observed_types: dict[Address, set[ValueTypeObservation]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(record, Mapping):
        record = {}

    values: dict[str, Any] = {}
    states: dict[str, Any] = {}
    for child in branch.fields:
        name = child.name
        if child.type == "branch":
            if child.address not in direct_branches:
                values[name] = []
                states[name] = []
                continue

            raw = record.get(name)
            if raw is None:
                values[name] = []
                states[name] = []
                continue

            if isinstance(raw, Mapping):
                if child.length != 1:
                    raise TypeError(
                        f"branch at address '{child.address}' expects a sequence, got a mapping; "
                        "mapping shorthand is only valid for length=1"
                    )
                items = [raw]
            elif sequence(raw):
                items = raw
            else:
                raise TypeError(f"branch at address '{child.address}' expects a sequence, got {type(raw).__name__}")

            child_values: list[dict[str, Any]] = []
            child_states: list[dict[str, Any]] = []
            for item_index, item in enumerate(items):
                if item is not None and not isinstance(item, Mapping):
                    raise TypeError(
                        f"branch at address '{child.address}' item {item_index} must be a mapping or null, "
                        f"got {type(item).__name__}"
                    )
                item_values, item_states = normalize(
                    item,
                    child,
                    direct_branches,
                    plugins,
                    observed_types,
                )
                child_values.append(item_values)
                child_states.append(item_states)
            values[name] = child_values
            states[name] = child_states
            continue

        # Explicit queries read from ``source`` and never enter the direct
        # Awkward layout. This also prevents a same-named raw key from
        # affecting the queried field's validation or inferred Awkward Form.
        if child.query is not None or child.address not in plugins:
            states[name] = Tokens.padded.value
            continue

        if name not in record:
            states[name] = Tokens.padded.value
            continue

        prepared, state, observed = prepare(
            record[name],
            address=child.address,
            plugin=plugins[child.address],
        )
        if observed:
            observed_types.setdefault(child.address, set()).update(observed)

        # A literal is routing state, not a value. Store it as an option value
        # so it cannot influence Awkward's type promotion.
        values[name] = prepared
        states[name] = state

    return values, states


def diagnose(
    value: Any,
    branch: Branch,
    addresses: tuple[Address, ...],
    direct_branches: frozenset[Address],
) -> None:
    if not isinstance(value, Mapping):
        return

    for child in branch.fields:
        name = child.name
        if name not in value:
            continue

        raw = value[name]
        if child.type == "branch":
            if child.address not in direct_branches or raw is None:
                continue
            items = [raw] if isinstance(raw, Mapping) else raw
            if sequence(items):
                for item in items:
                    diagnose(item, child, addresses, direct_branches)
            continue

        if child.query is not None or child.address not in addresses:
            continue

        try:
            ak.Array([raw])
        except _AWKWARD_CONVERSION_ERRORS as error:
            raise TypeError(
                f"field at address '{child.address}' contains an unsupported {type(raw).__name__}; "
                "normalize it in a preprocessor"
            ) from error


def project(
    values: ak.Array,
    states: ak.Array,
    fields: tuple[str, ...],
    *,
    address: Address,
) -> tuple[ak.Array, ak.Array]:
    for index, field in enumerate(fields):
        value_fields = ak.fields(values)
        state_fields = ak.fields(states)

        if field in state_fields:
            state = states[field]
        elif not state_fields and len(ak.flatten(states, axis=None)) == 0:
            state = states
        else:
            raise KeyError(f"cannot project field {field!r} for address '{address}'")

        if field in value_fields:
            value = values[field]
        elif field in state_fields:
            # A leaf absent from every source record has no Awkward record
            # field. Its schema-owned state is padded, so materialize an
            # option-valued projection. Missing intermediate branches stay
            # empty and acquire their declared geometry during regularization.
            available = state != Tokens.padded.value
            value = ak.mask(state, available) if index == len(fields) - 1 else state
        elif not value_fields and len(ak.flatten(values, axis=None)) == 0:
            value = values
        else:
            raise KeyError(f"cannot project field {field!r} for address '{address}'")

        values, states = value, state

    return values, states


def query(
    source: EncodedBatch,
    expression: str,
    *,
    address: Address,
    leaf_axis: int,
    plugin: Plugin,
) -> tuple[ak.Array, ak.Array]:
    def walk(value: Any, depth: int) -> tuple[Any, Any, ValueTypeObservations]:
        if depth == 0:
            return prepare(value, address=address, plugin=plugin)
        if value is None:
            return None, None, None
        if hasattr(value, "__next__"):
            raise TypeError(
                f"query result for address '{address}' contains a one-shot iterator; "
                "materialize it as a list in a preprocessor"
            )
        if not isinstance(value, list):
            return value, Tokens.padded.value, None

        cleaned: list[Any] = []
        states: list[Any] = []
        observed: set[ValueTypeObservation] = set()
        for child in value:
            child_value, child_state, child_observed = walk(child, depth - 1)
            cleaned.append(child_value)
            states.append(child_state)
            if child_observed:
                observed.update(child_observed)
        return cleaned, states, observed or None

    try:
        result = compile(expression).search(source)
    except _AWKWARD_CONVERSION_ERRORS + (JMESPathError,) as error:
        raise TypeError(f"JMESPath query failed for address '{address}': {expression!r}") from error

    normalized, state_slots, observed = walk(result, leaf_axis + 1)
    plugin.validate(observed, address=address)

    try:
        values = ak.Array(normalized)
        states = ak.values_astype(ak.Array(state_slots), np.int8)
    except _AWKWARD_CONVERSION_ERRORS + (JMESPathError,) as error:
        raise TypeError(
            f"JMESPath query for address '{address}' did not produce an Awkward-compatible "
            f"schema-shaped result: {expression!r}"
        ) from error

    return values, states


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
    """Dense field geometry plus flat retained Awkward values."""

    state: np.ndarray
    values: ak.Array
    placement: np.ndarray

    def __post_init__(self) -> None:
        if self.state.dtype != np.dtype(np.int64):
            raise TypeError(f"RaggedField.state must use int64, got {self.state.dtype}")
        if self.placement.dtype != np.dtype(np.int64) or self.placement.ndim != 1:
            raise TypeError("RaggedField.placement must be a one-dimensional int64 array")
        if len(self.values) != len(self.placement):
            raise ValueError(
                f"RaggedField values/placement length mismatch: {len(self.values)} values, "
                f"{len(self.placement)} placements"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.state.shape

    @property
    def batch_size(self) -> int:
        return self.shape[0]

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
        flat_shape = (self.state.size, *value_shape)
        placed.reshape(flat_shape)[self.placement] = array
        return placed


def coalesce(
    values: EncodedBatch,
    schema: Schema,
    strata: Strata | str,
) -> dict[Address, RaggedField]:
    """Prepare every encoded field before datatype codecs mutate shared state."""
    from relflow.tensorfields.base import TENSORFIELDS

    if not isinstance(values, list):
        raise TypeError(f"coalesce values must be an encoded batch list, got {type(values).__name__}")

    strata = Strata.normalize(strata)
    predict_targets = set(schema.target) if strata == Strata.predict else set()
    addresses = tuple(address for address in schema.active_requests if address not in predict_targets)
    plugins = {address: TENSORFIELDS[schema.requests[address].type] for address in addresses}
    direct_branches = frozenset(
        node.address
        for address in addresses
        if schema.requests[address].query is None
        for node in schema.requests[address].path
        if node.type == "branch"
    )
    observed_types: dict[Address, set[ValueTypeObservation]] = {}
    direct_values: list[list[dict[str, Any]]] = []
    direct_states: list[list[dict[str, Any]]] = []

    for batch_index, roots in enumerate(values):
        if not isinstance(roots, list) or len(roots) != 1:
            length = len(roots) if isinstance(roots, list) else None
            detail = type(roots).__name__ if length is None else str(length)
            raise ValueError(
                "coalesce requires exactly one generated-root record per batch item; "
                f"batch item {batch_index} provided {detail}"
            )
        if not isinstance(roots[0], Mapping):
            raise TypeError(
                f"coalesce root record at batch item {batch_index} must be a mapping, got {type(roots[0]).__name__}"
            )

        root_values, root_states = normalize(
            roots[0],
            schema.fields,
            direct_branches,
            plugins,
            observed_types,
        )
        direct_values.append([root_values])
        direct_states.append([root_states])

    for address, observed in observed_types.items():
        plugins[address].validate(observed, address=address)
    if not addresses:
        return {}

    try:
        layout = ak.Array(direct_values)
        states = ak.values_astype(ak.Array(direct_states), np.int8)
    except _AWKWARD_CONVERSION_ERRORS as error:
        for roots in values:
            diagnose(roots[0], schema.fields, addresses, direct_branches)
        raise TypeError(
            "coalesce requires directly bound values to be Awkward-compatible; "
            f"normalize the modeled value in a preprocessor: {error}"
        ) from error

    fields: dict[Address, RaggedField] = {}
    root_name = schema.fields.name
    for address in addresses:
        request = schema.requests[address]
        shape = (len(values), *schema.shapes[address])
        parts = tuple(str(address).split("/"))
        if parts and parts[0] == root_name:
            parts = parts[1:]

        if request.query is None:
            projected_values, projected_states = project(layout, states, parts, address=address)
        else:
            projected_values, projected_states = query(
                values,
                request.query,
                address=address,
                leaf_axis=len(shape) - 1,
                plugin=plugins[address],
            )

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
            if request.query is None:
                raise
            raise ValueError(
                f"JMESPath query for address '{address}' does not match its schema shape: "
                f"{request.query!r} must produce {len(shape)} list axes"
            ) from error

        if state.shape != shape:
            raise ValueError(f"coalesced state for address '{address}' must have shape {shape}, got {state.shape}")
        if np.any(state == Tokens.masked.value) and strata != Strata.predict:
            raise ValueError(f"{MASK_LITERAL!r} at address '{address}' is only valid during predict strata")

        placement = np.flatnonzero(state.ravel() == Tokens.valued.value).astype(np.int64, copy=False)
        fields[address] = RaggedField(
            state=state,
            values=dense_values[placement],
            placement=placement,
        )

    return fields


__all__ = ["MASK_LITERAL", "MaskLiteral", "RaggedField"]
