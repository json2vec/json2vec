"""Canonical Awkward-backed preprocessing structures.

``RaggedBatch`` owns the schema-shaped nested layout for one encoded batch.
``RaggedField`` projects one modeled leaf, applies overflow, and exposes the
retained values separately from their dense destinations.
"""

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
    from relflow.tensorfields.base import Plugin, ValueTypeObservation, ValueTypeObservations

MASK_LITERAL = "<MASK>"
MaskLiteral: TypeAlias = Literal["<MASK>"]

_SEQUENCE_TYPES = (list, tuple, np.ndarray)
_AWKWARD_CONVERSION_ERRORS = (TypeError, ValueError, RuntimeError, OverflowError)
_MASK_LITERAL_TYPES = frozenset({str, np.str_})
_DIRECT_CONTAINER_TYPES = frozenset({list, tuple, dict})


@cache
def _compile_query(expression: str) -> jmespath.parser.ParsedResult:
    """Compile an observation-relative query for a complete encoded batch."""
    try:
        return jmespath.compile(f"[*]{expression}")
    except JMESPathError as error:
        raise ValueError(f"invalid JMESPath query {expression!r}: {error}") from error


def _is_sequence(value: Any) -> bool:
    return isinstance(value, _SEQUENCE_TYPES) and not (isinstance(value, np.ndarray) and value.ndim == 0)


def _empty_provenance_record(branch: Any) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for child in branch.fields:
        if child.type == "branch":
            provenance[child.name] = []
        else:
            provenance[child.name] = 0
    return provenance


def _direct_branch_addresses(branch: Any) -> frozenset[Address]:
    """Return branches that contain an active directly bound leaf."""
    addresses: set[Address] = set()

    def visit(node: Any) -> bool:
        has_direct_leaf = False
        for child in node.fields:
            if child.type == "branch":
                if visit(child):
                    addresses.add(child.address)
                    has_direct_leaf = True
            elif child.active and child.query is None:
                has_direct_leaf = True
        return has_direct_leaf

    visit(branch)
    return frozenset(addresses)


def _is_mask_literal(value: Any) -> bool:
    value_type = type(value)
    return value_type in _MASK_LITERAL_TYPES and str(value) == MASK_LITERAL


def _contains_one_shot_iterator(value: Any) -> bool:
    value_type = type(value)
    if hasattr(value, "__next__"):
        return True
    if value_type is dict:
        return any(_contains_one_shot_iterator(child) for child in value.values())
    if value_type is list or value_type is tuple:
        return any(_contains_one_shot_iterator(child) for child in value)
    if value_type is np.ndarray and value.dtype == np.dtype(object):
        return any(_contains_one_shot_iterator(child) for child in value.flat)
    if isinstance(value, (list, tuple)):
        return any(_contains_one_shot_iterator(child) for child in value)
    if isinstance(value, np.ndarray) and value.dtype == np.dtype(object):
        return any(_contains_one_shot_iterator(child) for child in value.flat)
    if isinstance(value, Mapping):
        return any(_contains_one_shot_iterator(child) for child in value.values())
    return False


def _reject_one_shot_iterators(value: Any, *, context: str) -> None:
    if _contains_one_shot_iterator(value):
        raise TypeError(
            f"RaggedBatch.new does not accept a one-shot iterator in {context}; "
            "materialize it as a list in a preprocessor"
        )


def _validate_value_types(
    observed_types: Mapping[Address, set[ValueTypeObservation]],
    plugins: Mapping[Address, Plugin],
) -> None:
    for address, observed in observed_types.items():
        plugins[address].validate_value_types(observed, address=address)


def _normalize_record(
    value: Any,
    branch: Any,
    direct_branches: frozenset[Address],
    plugins: Mapping[Address, Plugin],
    observed_types: dict[Address, set[ValueTypeObservation]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize modeled values and their source provenance in one pass."""
    if not isinstance(value, Mapping):
        return {}, _empty_provenance_record(branch)

    normalized: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for child in branch.fields:
        name = child.name
        if child.type == "branch":
            if child.address not in direct_branches:
                normalized[name] = []
                provenance[name] = []
                continue

            raw = value.get(name)
            if raw is None:
                normalized[name] = []
                provenance[name] = []
                continue

            if isinstance(raw, Mapping):
                if child.length != 1:
                    raise TypeError(
                        f"branch at address '{child.address}' expects a sequence, got a mapping; "
                        "mapping shorthand is only valid for length=1"
                    )
                items = [raw]
            elif _is_sequence(raw):
                items = raw
            else:
                raise TypeError(f"branch at address '{child.address}' expects a sequence, got {type(raw).__name__}")

            child_values: list[dict[str, Any]] = []
            child_provenance: list[dict[str, Any]] = []
            for item_index, item in enumerate(items):
                if item is not None and not isinstance(item, Mapping):
                    raise TypeError(
                        f"branch at address '{child.address}' item {item_index} must be a mapping or null, "
                        f"got {type(item).__name__}"
                    )
                item_values, item_provenance = _normalize_record(
                    item,
                    child,
                    direct_branches,
                    plugins,
                    observed_types,
                )
                child_values.append(item_values)
                child_provenance.append(item_provenance)
            normalized[name] = child_values
            provenance[name] = child_provenance
            continue

        # Explicit queries read from ``source`` and never enter the direct
        # schema carrier. This also prevents a same-named raw key from
        # affecting the queried field's validation or inferred Awkward Form.
        if child.query is not None or not child.active:
            provenance[name] = 0
            continue

        if name not in value:
            provenance[name] = 0
            continue

        raw = value[name]
        raw_type = type(raw)
        if raw_type in _DIRECT_CONTAINER_TYPES or (raw_type is np.ndarray and raw.dtype == np.dtype(object)):
            _reject_one_shot_iterators(raw, context=f"field at address '{child.address}'")
        elif (
            hasattr(raw, "__next__")
            or isinstance(raw, (list, tuple, Mapping))
            or (isinstance(raw, np.ndarray) and raw.dtype == np.dtype(object))
        ):
            _reject_one_shot_iterators(raw, context=f"field at address '{child.address}'")

        literal = raw_type in _MASK_LITERAL_TYPES and str(raw) == MASK_LITERAL
        if literal:
            prepared = None
        else:
            prepared, observed = plugins[child.address].prepare_value(raw, address=child.address)
            if observed:
                observed_types.setdefault(child.address, set()).update(observed)

        # A literal is routing provenance, not a value. Store it as an option
        # value so it cannot influence Awkward's type promotion; the parallel
        # provenance carrier restores ``Tokens.masked`` after regularization.
        normalized[name] = prepared
        provenance[name] = 2 if literal else 1

    return normalized, provenance


def _prepare_query_values(
    value: Any,
    *,
    depth: int,
    address: Address,
    plugin: Plugin,
) -> tuple[Any, Any, ValueTypeObservations]:
    """Prepare query leaves and remove whole-leaf literals before Awkward conversion."""
    if depth == 0:
        literal = _is_mask_literal(value)
        if literal:
            return None, 2, None
        prepared, observed = plugin.prepare_value(value, address=address)
        return prepared, 1, observed
    if isinstance(value, list):
        cleaned: list[Any] = []
        provenance: list[Any] = []
        observed: set[ValueTypeObservation] = set()
        for child in value:
            child_value, child_provenance, child_observed = _prepare_query_values(
                child,
                depth=depth - 1,
                address=address,
                plugin=plugin,
            )
            cleaned.append(child_value)
            provenance.append(child_provenance)
            if child_observed:
                observed.update(child_observed)
        return cleaned, provenance, observed or None
    if value is None:
        return None, None, None
    return value, 0, None


def _diagnose_modeled_values(value: Any, branch: Any) -> None:
    """Add a schema address to an Awkward conversion failure."""
    if not isinstance(value, Mapping):
        return

    for child in branch.fields:
        name = child.name
        if name not in value:
            continue

        raw = value[name]
        if child.type == "branch":
            if raw is None:
                continue
            items = [raw] if isinstance(raw, Mapping) else raw
            if _is_sequence(items):
                for item in items:
                    _diagnose_modeled_values(item, child)
            continue

        if child.query is not None or not child.active:
            continue

        try:
            ak.Array([raw])
        except _AWKWARD_CONVERSION_ERRORS as error:
            raise TypeError(
                f"field at address '{child.address}' contains an unsupported {type(raw).__name__}; "
                "normalize it in a preprocessor"
            ) from error


def _project(
    values: ak.Array,
    provenance: ak.Array,
    fields: tuple[str, ...],
    *,
    address: Address,
) -> tuple[ak.Array, ak.Array]:
    """Project values and source-slot provenance through one schema path."""
    projected_values = values
    projected_provenance = provenance
    for index, field in enumerate(fields):
        value_fields = ak.fields(projected_values)
        provenance_fields = ak.fields(projected_provenance)

        if field in provenance_fields:
            next_provenance = projected_provenance[field]
        elif not provenance_fields and len(ak.flatten(projected_provenance, axis=None)) == 0:
            next_provenance = projected_provenance
        else:
            raise KeyError(f"cannot project field {field!r} for address '{address}'")

        if field in value_fields:
            next_values = projected_values[field]
        elif field in provenance_fields:
            # A leaf absent from every source record has no Awkward record
            # field. Its schema-owned provenance is zero, so materialize an
            # option-valued projection. Missing intermediate branches stay
            # empty and acquire their declared geometry during regularization.
            available = next_provenance != 0
            next_values = ak.mask(next_provenance, available) if index == len(fields) - 1 else next_provenance
        elif not value_fields and len(ak.flatten(projected_values, axis=None)) == 0:
            next_values = projected_values
        else:
            raise KeyError(f"cannot project field {field!r} for address '{address}'")

        projected_values = next_values
        projected_provenance = next_provenance

    return projected_values, projected_provenance


def _query_project(
    source: EncodedBatch,
    expression: str,
    *,
    address: Address,
    leaf_axis: int,
    plugin: Plugin,
) -> tuple[ak.Array, ak.Array]:
    """Evaluate an opt-in JMESPath query and recover its returned slots."""
    try:
        result = _compile_query(expression).search(source)
    except _AWKWARD_CONVERSION_ERRORS + (JMESPathError,) as error:
        raise TypeError(f"JMESPath query failed for address '{address}': {expression!r}") from error

    _reject_one_shot_iterators(result, context=f"query result for address '{address}'")
    normalized, provenance_slots, observed = _prepare_query_values(
        result,
        depth=leaf_axis + 1,
        address=address,
        plugin=plugin,
    )
    plugin.validate_value_types(observed, address=address)

    try:
        values = ak.Array(normalized)
        provenance = ak.values_astype(ak.Array(provenance_slots), np.int8)
    except _AWKWARD_CONVERSION_ERRORS + (JMESPathError,) as error:
        raise TypeError(
            f"JMESPath query for address '{address}' did not produce an Awkward-compatible "
            f"schema-shaped result: {expression!r}"
        ) from error

    return values, provenance


def _slice_axis(array: ak.Array, *, axis: int, start: int | None, stop: int | None) -> ak.Array:
    slices = (*((slice(None),) * axis), slice(start, stop))
    return array[slices]


def _has_overflow(array: ak.Array, *, axis: int, capacity: int) -> bool:
    try:
        counts = ak.num(array, axis=axis)
    except ak.errors.AxisError:
        return False
    exceeds = counts > capacity
    if isinstance(exceeds, ak.Array):
        return bool(ak.any(exceeds, axis=None))
    return bool(exceeds)


def _regularize(
    array: ak.Array,
    *,
    shape: tuple[int, ...],
    overflows: tuple[Overflow, ...],
    address: Address,
    validate_overflow: bool = True,
) -> ak.Array:
    if len(shape) != len(overflows):
        raise ValueError(f"overflows length must match shape rank: expected {len(shape)}, got {len(overflows)}")

    regular = array
    for axis, (capacity, overflow) in enumerate(zip(shape, overflows, strict=True)):
        # Padding an outer axis creates option-valued parent slots. Before
        # descending, turn those absent parents into empty lists so the next
        # declared model axis can be regularized normally.
        if axis > 0:
            regular = ak.fill_none(regular, [], axis=axis - 1)

        overflow = Overflow(overflow)
        if validate_overflow and overflow == Overflow.error and _has_overflow(regular, axis=axis, capacity=capacity):
            context = (
                "batch dimension 0" if axis == 0 else "root node dimension 1" if axis == 1 else f"dimension {axis}"
            )
            raise ValueError(f"branch overflow at {context} for {address}: capacity is {capacity}")
        if overflow == Overflow.tail:
            regular = _slice_axis(regular, axis=axis, start=-capacity, stop=None)
        else:
            regular = _slice_axis(regular, axis=axis, start=0, stop=capacity)

        regular = ak.pad_none(regular, capacity, axis=axis, clip=True)

    return regular


@dataclass(frozen=True, slots=True)
class RaggedBatch:
    """One schema-aware, batch-by-singleton-root Awkward carrier."""

    layout: ak.Array
    provenance: ak.Array
    schema: Schema
    source: EncodedBatch

    @classmethod
    def new(cls, values: EncodedBatch, *, schema: Schema) -> RaggedBatch:
        from relflow.tensorfields.base import TENSORFIELDS

        if not isinstance(values, list):
            raise TypeError(f"RaggedBatch.new values must be an encoded batch list, got {type(values).__name__}")

        normalized_values: list[list[dict[str, Any]]] = []
        normalized_provenance: list[list[dict[str, Any]]] = []
        plugins = {address: TENSORFIELDS[request.type] for address, request in schema.requests.items()}
        observed_types: dict[Address, set[ValueTypeObservation]] = {}
        direct_branches = _direct_branch_addresses(schema.fields)
        for batch_index, roots in enumerate(values):
            if not isinstance(roots, list) or len(roots) != 1:
                length = len(roots) if isinstance(roots, list) else None
                detail = type(roots).__name__ if length is None else str(length)
                raise ValueError(
                    "RaggedBatch.new requires exactly one generated-root record per batch item; "
                    f"batch item {batch_index} provided {detail}"
                )
            if not isinstance(roots[0], Mapping):
                raise TypeError(
                    f"RaggedBatch.new root record at batch item {batch_index} must be a mapping, "
                    f"got {type(roots[0]).__name__}"
                )

            root_values, root_provenance = _normalize_record(
                roots[0],
                schema.fields,
                direct_branches,
                plugins,
                observed_types,
            )
            normalized_values.append([root_values])
            normalized_provenance.append([root_provenance])

        _validate_value_types(observed_types, plugins)

        try:
            layout = ak.Array(normalized_values)
            provenance = ak.values_astype(ak.Array(normalized_provenance), np.int8)
        except _AWKWARD_CONVERSION_ERRORS as error:
            for roots in values:
                _diagnose_modeled_values(roots[0], schema.fields)
            raise TypeError(
                "RaggedBatch.new requires directly bound values to be Awkward-compatible; "
                f"normalize the modeled value in a preprocessor: {error}"
            ) from error

        return cls(layout=layout, provenance=provenance, schema=schema, source=values)

    @property
    def batch_size(self) -> int:
        return len(self.layout)


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

    @classmethod
    def new(
        cls,
        batch: RaggedBatch,
        *,
        address: Address | str,
        strata: Strata | str,
    ) -> RaggedField:
        from relflow.tensorfields.base import TENSORFIELDS

        address = Address(str(address))
        strata = Strata.normalize(strata)
        if address not in batch.schema.requests:
            raise KeyError(f"no request at address '{address}'")

        root_name = batch.schema.fields.name
        parts = tuple(str(address).split("/"))
        if parts and parts[0] == root_name:
            parts = parts[1:]

        shape = (batch.batch_size, *batch.schema.shapes[address])
        overflows = batch.schema.overflows(address)
        request = batch.schema.requests[address]
        plugin = TENSORFIELDS[request.type]
        if request.query is None:
            projected_values, projected_provenance = _project(
                batch.layout,
                batch.provenance,
                parts,
                address=address,
            )
        else:
            projected_values, projected_provenance = _query_project(
                batch.source,
                request.query,
                address=address,
                leaf_axis=len(shape) - 1,
                plugin=plugin,
            )
        try:
            dense_provenance = _regularize(
                projected_provenance,
                shape=shape,
                overflows=overflows,
                address=address,
            )
            dense_values = _regularize(
                projected_values,
                shape=shape,
                overflows=overflows,
                address=address,
                validate_overflow=False,
            )
        except (IndexError, ak.errors.AxisError) as error:
            if request.query is None:
                raise
            raise ValueError(
                f"JMESPath query for address '{address}' does not match its schema shape: "
                f"{request.query!r} must produce {len(shape)} list axes"
            ) from error

        provenance = np.asarray(
            ak.to_numpy(ak.fill_none(dense_provenance, 0, axis=len(shape) - 1)),
            dtype=np.int8,
        )
        present = provenance != 0
        literal = provenance == 2
        null = np.asarray(ak.to_numpy(ak.is_none(dense_values, axis=len(shape) - 1)), dtype=np.bool_)

        state = np.full(shape, Tokens.padded.value, dtype=np.int64)
        state[present & ~literal & null] = Tokens.null.value
        state[present & ~literal & ~null] = Tokens.valued.value
        state[literal] = Tokens.masked.value

        candidate_placement = np.flatnonzero(present.ravel() & ~literal.ravel() & ~null.ravel()).astype(
            np.int64,
            copy=False,
        )
        flattened_values = dense_values
        for _ in range(len(shape) - 1):
            flattened_values = ak.flatten(flattened_values, axis=1)
        candidates = flattened_values[candidate_placement]

        if literal.any() and strata != Strata.predict:
            raise ValueError(f"{MASK_LITERAL!r} at address '{address}' is only valid during predict strata")

        return cls(
            state=state,
            values=candidates,
            placement=candidate_placement,
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


__all__ = ["MASK_LITERAL", "MaskLiteral", "RaggedBatch", "RaggedField"]
