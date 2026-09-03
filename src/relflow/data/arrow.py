"""Arrow-native data and identity carried through the RelFlow pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from tensordict import TensorDict

IDENTITY = pa.struct(
    [
        pa.field("logical", pa.binary(32), nullable=False),
        pa.field("instance", pa.binary(32), nullable=False),
        pa.field("order", pa.large_binary(), nullable=False),
    ]
)


def mappings(
    values: Sequence[Mapping[str, Any]],
    *,
    schema: pa.Schema | None = None,
    context: str = "Python records",
) -> list[dict[str, Any]]:
    """Align bounded Python mappings before their one Arrow conversion."""

    if any(not isinstance(value, Mapping) for value in values):
        raise TypeError(f"{context} must contain only mappings")
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            if key not in seen:
                seen.add(key)
                keys.append(key)

    if schema is not None:
        if len(set(schema.names)) != len(schema.names):
            raise ValueError(f"{context} Arrow schema must have unique field names")
        extra = [key for key in keys if key not in schema.names]
        if extra:
            raise TypeError(f"{context} introduced field(s) {extra}; provide a complete arrow_schema")
        keys = list(schema.names)
    elif values and not keys:
        raise ValueError(f"{context} cannot infer an Arrow schema from mappings without fields")

    return [{key: value.get(key) for key in keys} for value in values]


def integers(values: pa.Array | pa.ChunkedArray, *, name: str) -> pa.Array:
    """Validate and normalize an Arrow integer vector."""

    if not isinstance(values, (pa.Array, pa.ChunkedArray)):
        raise TypeError(f"Batch {name} must be an Arrow Array or ChunkedArray, got {type(values).__name__}")
    if not pa.types.is_integer(values.type):
        raise TypeError(f"Batch {name} must have an integer Arrow type, got {values.type}")
    if values.null_count:
        raise ValueError(f"Batch {name} cannot contain nulls")
    try:
        cast = pc.cast(values, pa.int64(), safe=True)
    except pa.ArrowException as error:
        raise ValueError(f"Batch {name} must fit in signed 64-bit indices") from error
    return cast.combine_chunks() if isinstance(cast, pa.ChunkedArray) else cast


def matrix(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """View non-null 32-byte Arrow values as four native unsigned words."""

    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if not pa.types.is_fixed_size_binary(array.type) or array.type.byte_width != 32:
        raise TypeError(f"lineage hashes must use fixed_size_binary[32], got {array.type}")
    if not len(array):
        return np.empty((0, 4), dtype=np.uint64)

    buffer = array.buffers()[1]
    raw = np.frombuffer(buffer, dtype=np.uint8, count=len(array) * 32, offset=array.offset * 32)
    return raw.reshape(len(array), 4, 8).copy().view(">u8").reshape(len(array), 4).astype(np.uint64)


def mix(values: np.ndarray) -> np.ndarray:
    """Avalanche unsigned words with vectorized SplitMix64 rounds."""

    result = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        result ^= result >> np.uint64(30)
        result *= np.uint64(0xBF58476D1CE4E5B9)
        result ^= result >> np.uint64(27)
        result *= np.uint64(0x94D049BB133111EB)
        result ^= result >> np.uint64(31)
    return result


def binary(values: np.ndarray) -> pa.FixedSizeBinaryArray:
    """Pack four unsigned words into Arrow's canonical lineage hash type."""

    wire = np.asarray(values, dtype=np.uint64).astype(">u8").view(np.uint8).reshape(-1, 32).copy()
    return pa.FixedSizeBinaryArray.from_buffers(pa.binary(32), len(wire), [None, pa.py_buffer(wire)])


def derive(values: pa.Array | pa.ChunkedArray, ordinals: np.ndarray) -> pa.FixedSizeBinaryArray:
    """Derive stable child hashes from parent hashes and output ordinals."""

    words = matrix(values)
    positions = np.asarray(ordinals, dtype=np.uint64)
    if len(words) != len(positions):
        raise ValueError("lineage parent and ordinal lengths must match")

    result = np.empty_like(words)
    with np.errstate(over="ignore"):
        for column in range(4):
            seed = np.uint64((0x9E3779B97F4A7C15 * (column + 1)) & ((1 << 64) - 1))
            salt = mix(positions + seed)
            result[:, column] = mix(words[:, column] ^ salt)
    return binary(result)


def collapse(
    values: pa.Array | pa.ChunkedArray,
    parents: np.ndarray,
    groups: np.ndarray,
    ordinals: np.ndarray,
    lengths: np.ndarray,
) -> pa.FixedSizeBinaryArray:
    """Derive ordered group hashes from flattened parent lineage."""

    words = matrix(values)[parents]
    result = np.empty((len(lengths), 4), dtype=np.uint64)
    sizes = np.asarray(lengths, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for column in range(4):
            seed = np.uint64((0x9E3779B97F4A7C15 * (column + 1)) & ((1 << 64) - 1))
            result[:, column] = mix(sizes ^ seed)
            salt = mix(np.asarray(ordinals, dtype=np.uint64) + seed)
            contribution = mix(words[:, column] ^ salt)
            np.bitwise_xor.at(result[:, column], groups, contribution)
            result[:, column] = mix(result[:, column] ^ sizes)
    return binary(result)


def ordinal(values: np.ndarray) -> pa.Array:
    """Encode non-negative ordinals as lexicographically sortable bytes."""

    wire = np.asarray(values, dtype=np.uint64).astype(">u8").view(np.uint8).reshape(-1, 8).copy()
    return pa.FixedSizeBinaryArray.from_buffers(pa.binary(8), len(wire), [None, pa.py_buffer(wire)])


def join(prefix: pa.Array | pa.ChunkedArray, marker: bytes, suffix: pa.Array) -> pa.Array | pa.ChunkedArray:
    """Append a tagged fixed-width suffix to Arrow binary order values."""

    tagged = pc.binary_join_element_wise(
        pc.cast(prefix, pa.large_binary()),
        pa.repeat(pa.scalar(marker, type=pa.large_binary()), len(prefix)),
        pa.scalar(b"", type=pa.large_binary()),
    )
    return pc.binary_join_element_wise(
        tagged,
        pc.cast(suffix, pa.large_binary()),
        pa.scalar(b"", type=pa.large_binary()),
    )


def pack(
    logical: pa.Array | pa.ChunkedArray,
    instance: pa.Array | pa.ChunkedArray,
    order: pa.Array | pa.ChunkedArray,
) -> pa.StructArray:
    """Build a non-chunked identity struct from aligned lineage fields."""

    arrays = [
        item.combine_chunks() if isinstance(item, pa.ChunkedArray) else item for item in (logical, instance, order)
    ]
    return pa.StructArray.from_arrays(arrays, fields=list(IDENTITY))


@dataclass(frozen=True, slots=True)
class Batch:
    """An Arrow table and the stable identity of each logical row.

    ``identity`` is deliberately separate from ``data`` so framework lineage
    cannot accidentally become model input. Its length defines the batch
    length, including when ``data`` has no columns and Arrow therefore cannot
    retain a positive row count on its own.

    Use :meth:`slice`, :meth:`take`, or :meth:`filter` whenever rows change so
    the payload and its identity stay aligned. Use :meth:`replace` for a
    same-row column transformation.
    """

    data: pa.Table
    identity: pa.Array | pa.ChunkedArray

    def __post_init__(self) -> None:
        if not isinstance(self.data, pa.Table):
            raise TypeError(f"Batch data must be a pyarrow.Table, got {type(self.data).__name__}")

        if not isinstance(self.identity, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                f"Batch identity must be a pyarrow.Array or pyarrow.ChunkedArray, got {type(self.identity).__name__}"
            )

        if self.identity.type != IDENTITY:
            raise TypeError(f"Batch identity must have type {IDENTITY}, got {self.identity.type}")

        if self.identity.null_count:
            raise ValueError("Batch identity cannot contain null rows")

        for field in IDENTITY:
            if pc.struct_field(self.identity, field.name).null_count:
                raise ValueError(f"Batch identity field {field.name!r} cannot contain nulls")

        if self.data.num_columns and self.data.num_rows != len(self.identity):
            raise ValueError(f"Batch data has {self.data.num_rows} rows but identity has {len(self.identity)} rows")

    def __len__(self) -> int:
        """Return the logical row count, including for zero-column data."""

        return len(self.identity)

    def slice(self, offset: int = 0, length: int | None = None) -> Batch:
        """Return a zero-copy contiguous view of rows and their identity."""

        if length is None:
            return Batch(data=self.data.slice(offset), identity=self.identity.slice(offset))

        return Batch(data=self.data.slice(offset, length), identity=self.identity.slice(offset, length))

    def take(self, indices: pa.Array | pa.ChunkedArray) -> Batch:
        """Select or reorder rows with an integer Arrow index array."""

        indices = integers(indices, name="take indices")

        identity = self.identity.take(indices)
        data = self.data.take(indices) if self.data.num_columns else self.data
        return Batch(data=data, identity=identity)

    def order(self, indices: pa.Array | pa.ChunkedArray) -> Batch:
        """Deliberately reorder every row and establish a new stable order."""

        indices = integers(indices, name="order indices")
        positions = indices.to_numpy(zero_copy_only=False)
        if len(indices) != len(self) or len(np.unique(positions)) != len(self):
            raise ValueError("Batch order indices must be a permutation of every row")
        if np.any(positions < 0) or np.any(positions >= len(self)):
            raise IndexError(f"Batch order indices must be between 0 and {len(self) - 1}")

        selected = self.identity.take(indices)
        logical = pc.struct_field(selected, "logical")
        instance = pc.struct_field(selected, "instance")
        previous = pc.struct_field(selected, "order")
        if len(selected):
            anchor = previous[0]
            prefix = pa.repeat(pa.scalar(anchor.as_py() + b"\x00R", type=pa.large_binary()), len(selected))
            ordered = pc.binary_join_element_wise(
                prefix,
                pc.cast(ordinal(np.arange(len(selected), dtype=np.uint64)), pa.large_binary()),
                pa.scalar(b"", type=pa.large_binary()),
            )
        else:
            ordered = pa.array([], type=pa.large_binary())

        data = self.data.take(indices) if self.data.num_columns else self.data
        return Batch(data=data, identity=pack(logical, instance, ordered))

    def expand(
        self,
        data: pa.Table,
        parents: pa.Array | pa.ChunkedArray,
        ordinals: pa.Array | pa.ChunkedArray,
    ) -> Batch:
        """Create one or more stable child observations from parent rows."""

        if not isinstance(data, pa.Table):
            raise TypeError(f"Batch expand data must be a pyarrow.Table, got {type(data).__name__}")
        parents = integers(parents, name="expand parents")
        ordinals = integers(ordinals, name="expand ordinals")
        if len(parents) != len(ordinals):
            raise ValueError(f"Batch expand has {len(parents)} parent indices but {len(ordinals)} output ordinals")
        if data.num_columns and data.num_rows != len(parents):
            raise ValueError(f"Batch expand data has {data.num_rows} rows but lineage has {len(parents)} rows")

        parent_values = parents.to_numpy(zero_copy_only=False)
        ordinal_values = ordinals.to_numpy(zero_copy_only=False)
        if np.any(parent_values < 0) or np.any(parent_values >= len(self)):
            raise IndexError(f"Batch expand parents must be between 0 and {len(self) - 1}")
        if np.any(ordinal_values < 0):
            raise ValueError("Batch expand ordinals must be non-negative")
        if len(parents):
            permutation = np.lexsort((ordinal_values, parent_values))
            ordered_parents = parent_values[permutation]
            ordered_ordinals = ordinal_values[permutation]
            duplicate = (ordered_parents[1:] == ordered_parents[:-1]) & (ordered_ordinals[1:] == ordered_ordinals[:-1])
            if np.any(duplicate):
                raise ValueError("Batch expand ordinals must be unique within each parent row")

        selected = self.identity.take(parents)
        logical = derive(pc.struct_field(selected, "logical"), ordinal_values)
        instance = derive(pc.struct_field(selected, "instance"), ordinal_values)
        order = join(pc.struct_field(selected, "order"), b"\x00E", ordinal(ordinal_values))
        return Batch(data=data, identity=pack(logical, instance, order))

    def explode(self, column: str, *, name: str | None = None) -> Batch:
        """Create one observation per item of a top-level Arrow list column."""

        if not isinstance(column, str) or not column:
            raise ValueError("Batch explode column must be a non-empty string")
        output = column if name is None else name
        if not isinstance(output, str) or not output:
            raise ValueError("Batch explode name must be a non-empty string")

        indices = self.data.schema.get_all_field_indices(column)
        if not indices:
            raise KeyError(f"Batch explode column {column!r} does not exist")
        if len(indices) > 1:
            raise ValueError(f"Batch explode column {column!r} is ambiguous")
        index = indices[0]
        values = self.data.column(index)
        if not (
            pa.types.is_list(values.type)
            or pa.types.is_large_list(values.type)
            or pa.types.is_fixed_size_list(values.type)
        ):
            raise TypeError(f"Batch explode column {column!r} must have an Arrow list type, got {values.type}")
        if output != column and output in self.data.column_names:
            raise ValueError(f"Batch explode output column {output!r} already exists")

        parents = pc.list_parent_indices(values)
        flattened = pc.list_flatten(values)
        repeated = self.data.take(parents)
        if output == column:
            data = repeated.set_column(index, output, flattened)
        else:
            data = repeated.remove_column(index).append_column(output, flattened)

        lengths = pc.fill_null(pc.list_value_length(values), 0)
        counts = lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        total = int(counts.sum())
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        ordinals = np.arange(total, dtype=np.int64) - starts
        return self.expand(data, parents, pa.array(ordinals, type=pa.int64()))

    def group(self, data: pa.Table, parents: pa.Array | pa.ChunkedArray) -> Batch:
        """Create stable observations from ordered groups of parent rows."""

        if not isinstance(data, pa.Table):
            raise TypeError(f"Batch group data must be a pyarrow.Table, got {type(data).__name__}")
        if not isinstance(parents, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                f"Batch group parents must be an Arrow list Array or ChunkedArray, got {type(parents).__name__}"
            )
        if not (
            pa.types.is_list(parents.type)
            or pa.types.is_large_list(parents.type)
            or pa.types.is_fixed_size_list(parents.type)
        ) or not pa.types.is_integer(parents.type.value_type):
            raise TypeError(f"Batch group parents must have a list-of-integer Arrow type, got {parents.type}")
        if parents.null_count:
            raise ValueError("Batch group parents cannot contain null groups")
        if data.num_columns and data.num_rows != len(parents):
            raise ValueError(f"Batch group data has {data.num_rows} rows but lineage has {len(parents)} rows")

        lengths = pc.list_value_length(parents)
        counts = lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        if np.any(counts == 0):
            raise ValueError("Batch group parents cannot contain empty groups")
        flat = integers(pc.list_flatten(parents), name="group parent indices")
        parent_values = flat.to_numpy(zero_copy_only=False)
        if np.any(parent_values < 0) or np.any(parent_values >= len(self)):
            raise IndexError(f"Batch group parents must be between 0 and {len(self) - 1}")

        groups = np.repeat(np.arange(len(parents), dtype=np.int64), counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        positions = np.arange(len(flat), dtype=np.int64) - starts
        if len(flat):
            permutation = np.lexsort((parent_values, groups))
            ordered_groups = groups[permutation]
            ordered_parents = parent_values[permutation]
            duplicate = (ordered_groups[1:] == ordered_groups[:-1]) & (ordered_parents[1:] == ordered_parents[:-1])
            if np.any(duplicate):
                raise ValueError("Batch group cannot repeat a parent within one group")

        logical_values = pc.struct_field(self.identity, "logical")
        instance_values = pc.struct_field(self.identity, "instance")
        logical = collapse(logical_values, parent_values, groups, positions, counts)
        instance = collapse(instance_values, parent_values, groups, positions, counts)
        if pc.count_distinct(logical).as_py() != len(logical):
            raise ValueError("Batch group produced duplicate logical identities")

        first = np.cumsum(counts) - counts
        first_parents = pa.array(parent_values[first], type=pa.int64())
        parent_order = pc.take(pc.struct_field(self.identity, "order"), first_parents)
        order = join(parent_order, b"\x00G", logical)
        return Batch(data=data, identity=pack(logical, instance, order))

    def filter(self, mask: pa.Array | pa.ChunkedArray) -> Batch:
        """Keep rows selected by an Arrow boolean mask.

        Arrow's standard filter semantics apply: null mask entries are dropped.
        """

        if not isinstance(mask, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                f"Batch filter mask must be a pyarrow.Array or pyarrow.ChunkedArray, got {type(mask).__name__}"
            )
        if not pa.types.is_boolean(mask.type):
            raise TypeError(f"Batch filter mask must have boolean Arrow type, got {mask.type}")
        if len(mask) != len(self):
            raise ValueError(f"Batch filter mask has {len(mask)} rows but identity has {len(self)} rows")

        identity = pc.filter(self.identity, mask)
        data = self.data.filter(mask) if self.data.num_columns else self.data
        return Batch(data=data, identity=identity)

    def replace(self, data: pa.Table) -> Batch:
        """Replace payload columns while preserving row identity and order."""

        return Batch(data=data, identity=self.identity)


@dataclass(frozen=True, slots=True)
class Encoded:
    """Internal tensor payload paired with its untouched Arrow source batch."""

    tensors: TensorDict
    source: Batch
    retain: tuple[str, ...] | Literal["*"] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, TensorDict):
            raise TypeError(f"Encoded tensors must be a TensorDict, got {type(self.tensors).__name__}")
        if not isinstance(self.source, Batch):
            raise TypeError(f"Encoded source must be an rf.Batch, got {type(self.source).__name__}")
        if self.retain != "*" and (
            not isinstance(self.retain, tuple)
            or any(not isinstance(name, str) or not name for name in self.retain)
            or len(set(self.retain)) != len(self.retain)
        ):
            raise ValueError("Encoded retain must be '*', or a tuple of unique non-empty column names")


__all__ = ["Batch"]
