"""Coalesce schema-shaped inputs into canonical Arrow-backed fields."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from relflow.data.arrow import Batch, matrix, mix, variants
from relflow.data.query import listed, parts, query
from relflow.structs.enums import Overflow, Strata, Tokens
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch
    from relflow.structs.tree import Mask


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


@dataclass(frozen=True, slots=True)
class Decision:
    """One policy selection in its owner's fixed coordinate geometry."""

    mask: Mask
    selected: np.ndarray
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Projection:
    """Pristine Arrow values plus resolved model-input and objective routing."""

    pristine: RaggedField
    visible: pa.Array
    present: pa.Array
    trainable: pa.Array
    inferred: pa.Array
    vacant: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.vacant, bool):
            raise TypeError(f"Projection.vacant must be a boolean, got {type(self.vacant).__name__}")
        if self.vacant and (len(self.pristine.values) or not pa.types.is_null(self.pristine.values.type)):
            raise ValueError("a vacant Projection must carry one zero-length Arrow NullArray")

        size = math.prod(self.pristine.shape)
        for name in ("visible", "present", "trainable", "inferred"):
            values = getattr(self, name)
            if not isinstance(values, pa.Array) or values.type != pa.bool_():
                datatype = getattr(values, "type", type(values).__name__)
                raise TypeError(f"Projection.{name} must be a Boolean Arrow array, got {datatype}")
            if values.null_count:
                raise ValueError(f"Projection.{name} cannot contain nulls")
            if len(values) != size:
                raise ValueError(f"Projection.{name} length must equal shape product {size}, got {len(values)}")

    def split(
        self,
        values: pa.Array | pa.ChunkedArray,
    ) -> tuple[RaggedField, RaggedField]:
        """Split one prepared pristine value array into input and target fields."""

        if not isinstance(values, (pa.Array, pa.ChunkedArray)):
            raise TypeError(f"prepared projection values must be an Arrow array, got {type(values).__name__}")
        if not self.vacant and len(values) != len(self.pristine.values):
            raise ValueError(
                "prepared projection values must preserve pristine length "
                f"{len(self.pristine.values)}, got {len(values)}"
            )
        if not valid(values).all():
            raise ValueError("prepared projection values cannot contain nulls")

        state = self.pristine.state.to_numpy(zero_copy_only=False).astype(np.int8, copy=True)
        visible = boolean(self.visible)
        present = boolean(self.present)
        trainable = boolean(self.trainable)
        placement = self.pristine.placement.to_numpy(zero_copy_only=False)

        input_state = state.copy()
        input_state[~present] = Tokens.padded.value
        input_state[present & ~visible] = Tokens.masked.value
        input_values = visible[placement]
        input_placement = placement[input_values]

        target_state = np.full(len(state), Tokens.padded.value, dtype=np.int8)
        target_state[trainable] = state[trainable]
        target_values = trainable[placement]
        target_placement = placement[target_values]

        if self.vacant and (input_values.any() or target_values.any()):
            raise ValueError("vacant projection values cannot satisfy visible input or target positions")

        return (
            RaggedField(
                values=retain(values, input_values),
                state=pa.array(input_state, type=pa.int8()),
                placement=pa.array(input_placement, type=pa.int64()),
                shape=self.pristine.shape,
            ),
            RaggedField(
                values=retain(values, target_values),
                state=pa.array(target_state, type=pa.int8()),
                placement=pa.array(target_placement, type=pa.int64()),
                shape=self.pristine.shape,
            ),
        )


def root(values: pa.Table, branch: Branch, *, size: int | None = None) -> Layout:
    """Build the singleton root layout without copying unmodeled columns."""

    rows = values.num_rows if values.num_columns else (0 if size is None else size)
    records = (
        compact(values.to_struct_array())
        if values.num_columns
        else pa.StructArray.from_buffers(pa.struct([]), rows, [None])
    )
    if branch.query is None:
        return Layout(
            records=records,
            placement=np.arange(rows, dtype=np.int64),
            shape=(rows, 1),
        )

    selected = query(records, branch.query, address=str(branch.address))
    if selected.present.type != pa.bool_():
        raise ValueError(f"root query {branch.query!r} must select one object per observation")

    present = boolean(selected.present)
    retained = present & valid(selected.values)
    return Layout(
        records=retain(selected.values, retained),
        placement=np.flatnonzero(retained).astype(np.int64, copy=False),
        shape=(rows, 1),
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


def vacancy(layout: Layout) -> RaggedField:
    """Represent a source-less leaf without inventing a plugin datatype."""

    return RaggedField(
        values=pa.nulls(0),
        state=pa.array(np.full(math.prod(layout.shape), Tokens.padded.value, dtype=np.int8)),
        placement=pa.array([], type=pa.int64()),
        shape=layout.shape,
    )


def pristine(layout: Layout, request: Any) -> RaggedField:
    """Project one untouched leaf while preserving plugin-owned Arrow values."""

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


def active(mask: Mask, strata: Strata) -> bool:
    """Return whether one normalized policy applies in a stratum."""

    if strata == Strata.train:
        return True
    if bool(mask.dropout):
        return False
    return strata != Strata.predict or mask.rate is None


def eligible(layout: Layout, mask: Mask, *, address: Address) -> np.ndarray:
    """Resolve one policy query against its owner's retained records."""

    if mask.query is None:
        return np.ones(len(layout.records), dtype=bool)

    selected = query(layout.records, mask.query, address=str(address))
    if len(selected.values) != len(layout.records):
        raise ValueError(
            f"mask query {mask.query!r} at '{address}' returned {len(selected.values)} values for "
            f"{len(layout.records)} owner records"
        )
    if selected.values.type != pa.bool_():
        raise TypeError(
            f"mask query {mask.query!r} at '{address}' must return Boolean values, got {selected.values.type}"
        )
    if selected.present.type != pa.bool_():
        raise ValueError(f"mask query {mask.query!r} at '{address}' must return one scalar Boolean per owner record")
    structural = boolean(selected.present)
    if not structural.all():
        missing = int((~structural).sum())
        raise ValueError(
            f"mask query {mask.query!r} at '{address}' is structurally absent for {missing} owner record(s); "
            "fill missing selector values explicitly"
        )
    if selected.values.null_count:
        raise ValueError(
            f"mask query {mask.query!r} at '{address}' contains {selected.values.null_count} null value(s); "
            "fill null selector values explicitly"
        )
    return boolean(selected.values)


def scores(
    batch: Batch,
    layout: Layout,
    *,
    address: Address,
    mask: Mask,
    seed: int,
    epoch: int,
    strata: Strata,
) -> np.ndarray:
    """Hash owner identity and nested slot into stable policy scores."""

    if not len(layout.records):
        return np.empty(0, dtype=np.uint64)

    row_size = math.prod(layout.shape[1:])
    rows = layout.placement // row_size
    slots = layout.placement % row_size
    identities = matrix(pc.struct_field(batch.identity, "instance"))[rows]
    effective_epoch = epoch if strata == Strata.train else 0
    selection = json.dumps(
        (mask.query, float(mask.rate).hex() if mask.rate is not None else None),
        separators=(",", ":"),
    )
    payload = f"relflow-mask-v1:{seed}:{strata}:{effective_epoch}:{address}:{selection}".encode()
    salt = np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    result = np.full(len(layout.records), salt, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for column in range(identities.shape[1]):
            lane = np.uint64((0x9E3779B97F4A7C15 * (column + 1)) & ((1 << 64) - 1))
            result = mix(result ^ mix(identities[:, column] + lane))
        result = mix(result ^ mix(slots.astype(np.uint64, copy=False) + salt))
    return result


def select(
    batch: Batch,
    layout: Layout,
    mask: Mask,
    *,
    address: Address,
    seed: int,
    epoch: int,
    strata: Strata,
) -> np.ndarray:
    """Resolve one literal, query-backed, or uniformly sampled policy."""

    chosen = eligible(layout, mask, address=address)
    if mask.rate is not None:
        rate = float(mask.rate)
        if rate <= 0.0:
            chosen = np.zeros_like(chosen)
        elif rate < 1.0:
            threshold = np.uint64(int(rate * np.iinfo(np.uint64).max))
            chosen &= (
                scores(
                    batch,
                    layout,
                    address=address,
                    mask=mask,
                    seed=seed,
                    epoch=epoch,
                    strata=strata,
                )
                <= threshold
            )

    dense = np.zeros(math.prod(layout.shape), dtype=bool)
    dense[layout.placement] = chosen
    return dense


def resolve(
    batch: Batch,
    layout: Layout,
    node: Any,
    *,
    seed: int,
    epoch: int,
    strata: Strata,
) -> tuple[Decision, ...]:
    """Resolve every active policy once in one node's owner geometry."""

    return tuple(
        Decision(
            mask=mask,
            selected=select(
                batch,
                layout,
                mask,
                address=node.address,
                seed=seed,
                epoch=epoch,
                strata=strata,
            ),
            shape=layout.shape,
        )
        for mask in node.mask
        if active(mask, strata)
    )


def broadcast(decision: Decision, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast one ancestor decision through fixed descendant geometry."""

    if shape[: len(decision.shape)] != decision.shape:
        raise ValueError(f"mask owner shape {decision.shape} is not a prefix of descendant shape {shape}")
    return np.repeat(decision.selected, math.prod(shape[len(decision.shape) :]))


def project(
    layout: Layout,
    request: Any,
    decisions: tuple[Decision, ...],
    *,
    strata: Strata,
) -> Projection:
    """Build pristine values and resolved input/objective routing for one leaf."""

    allow_vacancy = strata == Strata.predict and any(
        decision.mask.reconstruct and decision.mask.query is None and decision.mask.rate is None
        for decision in decisions
    )
    source_less = bool(
        allow_vacancy
        and request.query is None
        and pa.types.is_struct(layout.records.type)
        and not layout.records.type.get_all_field_indices(request.name)
    )
    field = vacancy(layout) if source_less else pristine(layout, request)
    state = field.state.to_numpy(zero_copy_only=False)
    observed = state != Tokens.padded.value
    owner = np.zeros(len(state), dtype=bool)
    owner[layout.placement] = True
    requested = np.zeros(len(state), dtype=bool)
    skipped = np.zeros(len(state), dtype=bool)
    masked = np.zeros(len(state), dtype=bool)

    for decision in decisions:
        selected = broadcast(decision, field.shape) & owner
        if decision.mask.reconstruct:
            requested |= selected
        if decision.mask.skip:
            skipped |= selected
        else:
            masked |= selected

    predicting = strata == Strata.predict
    modeled = observed.copy()
    if predicting:
        modeled |= requested
    skipped &= modeled
    masked &= modeled & ~skipped
    trainable = np.zeros(len(state), dtype=bool) if predicting else observed & requested
    inferred = requested if predicting else np.zeros(len(state), dtype=bool)
    present = modeled & ~skipped
    visible = observed & ~masked & ~skipped
    return Projection(
        pristine=field,
        visible=pa.array(visible, type=pa.bool_()),
        present=pa.array(present, type=pa.bool_()),
        trainable=pa.array(trainable, type=pa.bool_()),
        inferred=pa.array(inferred, type=pa.bool_()),
        vacant=source_less,
    )


def coalesce(
    values: Batch,
    schema: Schema,
    strata: Strata | str,
    *,
    seed: int = 0,
    epoch: int = 0,
) -> dict[Address, Projection]:
    """Resolve schema geometry and masks before datatype conversion."""
    if not isinstance(values, Batch):
        raise TypeError(f"coalesce values must be an Arrow Batch, got {type(values).__name__}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"coalesce seed must be an integer, got {type(seed).__name__}")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("coalesce epoch must be a non-negative integer")

    strata = Strata.normalize(strata)
    addresses = tuple(schema.active_requests)
    if not addresses:
        return {}

    descendants: dict[Address, list[Address]] = {}
    for address in addresses:
        for node in schema.requests[address].path:
            if getattr(node, "type", None) == "branch":
                descendants.setdefault(node.address, []).append(address)

    fields: dict[Address, Projection] = {}

    def visit(branch: Branch, layout: Layout, inherited: tuple[Decision, ...]) -> None:
        for child in branch.fields:
            if getattr(child, "type", None) == "branch":
                active = descendants.get(child.address, [])
                if active:
                    child_layout = descend(layout, child, active[0])
                    decisions = resolve(
                        values,
                        child_layout,
                        child,
                        seed=seed,
                        epoch=epoch,
                        strata=strata,
                    )
                    visit(child, child_layout, (*inherited, *decisions))
            elif child.address in schema.active_requests:
                decisions = resolve(
                    values,
                    layout,
                    child,
                    seed=seed,
                    epoch=epoch,
                    strata=strata,
                )
                fields[child.address] = project(layout, child, (*inherited, *decisions), strata=strata)

    layout = root(values.data, schema.fields, size=len(values))
    decisions = resolve(values, layout, schema.fields, seed=seed, epoch=epoch, strata=strata)
    visit(schema.fields, layout, decisions)

    return fields


__all__ = ["Projection", "RaggedField"]
