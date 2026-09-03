"""Coalesce schema-shaped inputs into canonical Arrow-backed fields."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from relflow.data.arrow import Batch, variants
from relflow.data.query import listed, parts, query
from relflow.structs.enums import Overflow, Strata, Tokens
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


def member(name: str) -> str:
    """Render a schema name as one native query member."""

    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) else f"[{json.dumps(name)}]"


def boolean(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Lower an Arrow validity vector to a dense Boolean array."""

    if not values.null_count:
        return values.to_numpy(zero_copy_only=False).astype(bool, copy=False)
    return pc.fill_null(values, False).to_numpy(zero_copy_only=False).astype(bool, copy=False)


def dictionary(datatype: pa.DataType) -> bool:
    """Return whether an Arrow type contains dictionary-encoded storage."""

    if pa.types.is_dictionary(datatype):
        return True
    if isinstance(datatype, pa.ExtensionType):
        return dictionary(datatype.storage_type)
    if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
        return dictionary(datatype.value_type)
    if pa.types.is_struct(datatype) or pa.types.is_union(datatype):
        return any(dictionary(field.type) for field in datatype)
    if pa.types.is_map(datatype):
        return dictionary(datatype.key_type) or dictionary(datatype.item_type)
    return False


def compact(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Compact ordinary chunks without combining independent dictionaries."""

    if not isinstance(values, pa.ChunkedArray):
        return values
    if values.num_chunks == 1:
        return values.chunk(0)
    if values.num_chunks and not dictionary(values.type):
        return values.combine_chunks()
    return values


def retain(values: pa.Array | pa.ChunkedArray, selected: np.ndarray) -> pa.Array | pa.ChunkedArray:
    """Filter Arrow values without invoking kernels on an invalid zero-chunk container."""

    if selected.all():
        return compact(values)
    if not selected.any():
        return compact(values.slice(0, 0))
    return compact(pc.filter(values, pa.array(selected, type=pa.bool_())))


def valid(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Return logical validity, including the selected children of Arrow unions."""

    if isinstance(values, pa.ChunkedArray):
        if not values.chunks:
            return np.empty(0, dtype=bool)
        return np.concatenate([valid(chunk) for chunk in values.chunks])
    if isinstance(values, pa.ExtensionArray):
        return valid(values.storage)
    if pa.types.is_dictionary(values.type):
        present = boolean(pc.is_valid(values.indices))
        positions = pc.fill_null(values.indices, 0).to_numpy(zero_copy_only=False)
        result = present.copy()
        result[present] &= valid(values.dictionary)[positions[present]]
        return result
    if not pa.types.is_union(values.type) and not values.null_count:
        return np.ones(len(values), dtype=bool)
    if not pa.types.is_union(values.type):
        return boolean(pc.is_valid(values))

    codes, offsets = variants(values)
    result = np.zeros(len(values), dtype=bool)
    for index, code in enumerate(values.type.type_codes):
        selected = codes == code
        child = valid(values.field(index))
        result[selected] = child[offsets[selected]] if offsets is not None else child[selected]
    return result


@dataclass(frozen=True, slots=True)
class Layout:
    """Structurally present records and their positions in dense model geometry."""

    records: pa.Array | pa.ChunkedArray
    placement: np.ndarray
    shape: tuple[int, ...]


def root(values: pa.Table, branch: Branch) -> Layout:
    """Build the singleton root layout without copying unmodeled columns."""

    records = compact(values.to_struct_array())
    if branch.query is None:
        return Layout(
            records=records,
            placement=np.arange(len(values), dtype=np.int64),
            shape=(len(values), 1),
        )

    selected = query(records, branch.query, address=str(branch.address))
    if selected.present.type != pa.bool_():
        raise ValueError(f"root query {branch.query!r} must select one object per observation")

    present = boolean(selected.present)
    retained = present & valid(selected.values)
    return Layout(
        records=retain(selected.values, retained),
        placement=np.flatnonzero(retained).astype(np.int64, copy=False),
        shape=(len(values), 1),
    )


def descend(layout: Layout, branch: Branch, address: Address) -> Layout:
    """Lower one repeated branch and retain its dense child positions."""

    expression = branch.query or member(branch.name)
    selected = query(layout.records, expression, address=str(branch.address))
    if selected.present.type != pa.bool_():
        raise ValueError(f"branch query {expression!r} must select one list per parent")
    if not listed(selected.values.type):
        raise ValueError(f"branch query {expression!r} expected a list, got {selected.values.type}")

    capacity = branch.length
    overflow = Overflow(branch.overflow)
    shape = (*layout.shape, capacity)
    present = boolean(selected.present)
    source = compact(selected.values)
    if isinstance(source, pa.ChunkedArray) and source.num_chunks == 0:
        return Layout(
            records=pa.chunked_array([], type=source.type.value_type),
            placement=np.empty(0, dtype=np.int64),
            shape=shape,
        )
    chunks = source.chunks if isinstance(source, pa.ChunkedArray) else [source]

    record_chunks: list[pa.Array] = []
    placement_chunks: list[np.ndarray] = []
    offset = 0
    for values in chunks:
        parent_placement = layout.placement[offset : offset + len(values)]
        available = present[offset : offset + len(values)]
        offset += len(values)

        list_offsets, children, valid_list = parts(values)
        lengths = np.diff(list_offsets)
        available = available & valid_list
        if overflow == Overflow.error and np.any(available & (lengths > capacity)):
            axis = len(layout.shape)
            raise ValueError(f"branch overflow at dimension {axis} for {address}: capacity is {capacity}")

        counts = np.where(available, np.minimum(lengths, capacity), 0)
        total = int(counts.sum())
        if total == 0:
            record_chunks.append(children.slice(0, 0))
            placement_chunks.append(np.empty(0, dtype=np.int64))
            continue

        parents = np.repeat(np.arange(len(values), dtype=np.int64), counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        slots = np.arange(total, dtype=np.int64) - starts
        starts = list_offsets[:-1] + (lengths - counts if overflow == Overflow.tail else 0)
        indices = starts[parents] + slots
        if int(indices[-1]) - int(indices[0]) + 1 == total:
            records = children.slice(int(indices[0]), total)
        else:
            records = pc.take(children, pa.array(indices, type=pa.int64()))
        valid_record = valid(records)
        placement = parent_placement[parents] * capacity + slots
        if valid_record.all():
            record_chunks.append(records)
            placement_chunks.append(placement)
        else:
            record_chunks.append(pc.filter(records, pa.array(valid_record)))
            placement_chunks.append(placement[valid_record])

    records = (
        record_chunks[0] if len(record_chunks) == 1 else pa.chunked_array(record_chunks, type=record_chunks[0].type)
    )
    return Layout(
        records=records,
        placement=np.concatenate(placement_chunks),
        shape=shape,
    )


def project(layout: Layout, request: Any) -> RaggedField:
    """Project one leaf while preserving plugin-owned Arrow values unchanged."""

    expression = request.query or member(request.name)
    selected = query(layout.records, expression, address=str(request.address))
    present = (
        boolean(selected.present) if selected.present.type == pa.bool_() else boolean(pc.is_valid(selected.present))
    )
    available = valid(selected.values)
    valued = present & available

    state = np.full(math.prod(layout.shape), Tokens.padded.value, dtype=np.int8)
    state[layout.placement[present]] = Tokens.null.value
    state[layout.placement[valued]] = Tokens.valued.value
    placement = layout.placement[valued]
    return RaggedField(
        values=retain(selected.values, valued),
        state=pa.array(state, type=pa.int8()),
        placement=pa.array(placement, type=pa.int64()),
        shape=layout.shape,
    )


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
        if not valid(self.values).all():
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
        if len(states) and (states.min() < Tokens.valued.value or states.max() > Tokens.other.value):
            raise ValueError("RaggedField state contains an unknown token")

        positions = self.placement.to_numpy(zero_copy_only=False)
        if len(positions) and (np.any(positions < 0) or np.any(positions >= len(self.state))):
            raise ValueError("RaggedField placement contains an out-of-range dense position")
        if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError("RaggedField placement must be strictly increasing")
        valued = states == Tokens.valued.value
        if len(positions) != np.count_nonzero(valued) or not valued[positions].all():
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
    if not addresses:
        return {}

    descendants: dict[Address, list[Address]] = {}
    for address in addresses:
        for node in schema.requests[address].path:
            if getattr(node, "type", None) == "branch":
                descendants.setdefault(node.address, []).append(address)

    fields: dict[Address, RaggedField] = {}

    def visit(branch: Branch, layout: Layout) -> None:
        for child in branch.fields:
            if getattr(child, "type", None) == "branch":
                active = descendants.get(child.address, [])
                if active:
                    visit(child, descend(layout, child, active[0]))
            elif child.address in schema.active_requests and child.address not in predict_targets:
                fields[child.address] = project(layout, child)

    visit(schema.fields, root(values.data, schema.fields))

    return fields


__all__ = ["RaggedField"]
