"""Small node-relative query language compiled to Arrow operations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

Atom: TypeAlias = str | int | bool


@dataclass(frozen=True, slots=True)
class Member:
    """A named struct member."""

    name: str
    segment: str


@dataclass(frozen=True, slots=True)
class Literal:
    """A bracket literal resolved from its parent Arrow type during binding."""

    value: Atom
    segment: str


@dataclass(frozen=True, slots=True)
class Traverse:
    """Traversal over exactly one list axis."""

    segment: str = "[*]"


@dataclass(frozen=True, slots=True)
class Index:
    """A single list position."""

    value: int
    segment: str


@dataclass(frozen=True, slots=True)
class Slice:
    """A half-open list slice with optional Python-style bounds."""

    start: int | None
    stop: int | None
    segment: str


@dataclass(frozen=True, slots=True)
class Lookup:
    """An exact literal map-key lookup."""

    value: Atom
    segment: str


Parsed: TypeAlias = Member | Literal | Traverse | Slice
Bound: TypeAlias = Member | Traverse | Index | Slice | Lookup


@dataclass(frozen=True, slots=True)
class Query:
    """A syntax-checked, Arrow-type-neutral query."""

    text: str
    steps: tuple[Parsed, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    """A query bound to one Arrow input type."""

    query: Query
    steps: tuple[Bound, ...]
    input: pa.DataType
    output: pa.DataType
    presence: pa.DataType


@dataclass(frozen=True, slots=True)
class Projection:
    """Selected Arrow values and an aligned structural-presence bitmap."""

    values: pa.Array | pa.ChunkedArray
    present: pa.Array | pa.ChunkedArray


class Parser:
    """Strict parser for RelFlow's structural path grammar."""

    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def error(self, reason: str, position: int | None = None) -> ValueError:
        offset = self.position if position is None else position
        segment = self.text[offset:] or "<end>"
        return ValueError(f"invalid query {self.text!r} at {segment!r}: {reason}")

    def identifier(self) -> Member:
        start = self.position
        if self.position >= len(self.text) or not self.text[self.position].isalpha():
            raise self.error("expected an identifier beginning with a letter")

        self.position += 1
        while self.position < len(self.text):
            character = self.text[self.position]
            if not (character.isalpha() or character.isdigit() or character == "_"):
                break
            self.position += 1

        name = self.text[start : self.position]
        return Member(name=name, segment=name)

    def bracket(self) -> Literal | Traverse | Slice:
        start = self.position
        self.position += 1
        if self.position >= len(self.text):
            raise self.error("unterminated bracket", start)

        if self.text.startswith("*]", self.position):
            self.position += 2
            return Traverse()

        if self.text[self.position] == '"':
            try:
                value, consumed = json.JSONDecoder().raw_decode(self.text[self.position :])
            except json.JSONDecodeError as error:
                raise self.error("invalid JSON string literal", start) from error
            if not isinstance(value, str):
                raise self.error("quoted bracket values must be strings", start)

            self.position += consumed
            if self.position >= len(self.text) or self.text[self.position] != "]":
                raise self.error("expected ']' after the string literal", start)
            self.position += 1
            return Literal(value=value, segment=self.text[start : self.position])

        end = self.text.find("]", self.position)
        if end < 0:
            raise self.error("unterminated bracket", start)

        content = self.text[self.position : end]
        self.position = end + 1
        segment = self.text[start : self.position]

        if not content:
            raise self.error("flattening is not supported; use a preprocessor", start)
        if content.startswith("?"):
            raise self.error("filters are not supported; use a preprocessor", start)
        if ":" in content:
            bounds = content.split(":")
            if len(bounds) != 2:
                raise self.error("slice steps are not supported", start)
            if not all(not bound or re.fullmatch(r"-?(0|[1-9][0-9]*)", bound) for bound in bounds):
                raise self.error("slice bounds must be integers", start)
            return Slice(
                start=int(bounds[0]) if bounds[0] else None,
                stop=int(bounds[1]) if bounds[1] else None,
                segment=segment,
            )
        if content == "true":
            return Literal(value=True, segment=segment)
        if content == "false":
            return Literal(value=False, segment=segment)
        if re.fullmatch(r"-?(0|[1-9][0-9]*)", content):
            return Literal(value=int(content), segment=segment)
        if content.startswith("'"):
            raise self.error("string literals must use JSON double quotes", start)
        if "," in content:
            raise self.error("multiselects are not supported; use a preprocessor", start)
        raise self.error("unsupported bracket expression", start)

    def parse(self) -> Query:
        if not self.text or not self.text.strip():
            raise self.error("query must be a non-empty string", 0)
        if self.text != self.text.strip():
            raise self.error("whitespace is not allowed outside quoted field names", 0)
        if self.text.startswith("[*]"):
            raise self.error("queries are observation-relative and must not begin with '[*]'", 0)
        if self.text.startswith("$"):
            raise self.error("queries are observation-relative and must not begin with '$'", 0)

        if self.text.startswith("["):
            first = self.bracket()
            if not isinstance(first, Literal) or not isinstance(first.value, str):
                raise self.error("a query must begin with a field name", 0)
            steps: list[Parsed] = [first]
        else:
            steps = [self.identifier()]

        while self.position < len(self.text):
            character = self.text[self.position]
            if character == ".":
                if self.text.startswith("..", self.position):
                    raise self.error("recursive descent is not supported; use a preprocessor")
                self.position += 1
                if self.position < len(self.text) and self.text[self.position] == "*":
                    raise self.error("object wildcards are not supported; name the field explicitly")
                steps.append(self.identifier())
                continue
            if character == "[":
                steps.append(self.bracket())
                continue
            if character == "|":
                raise self.error("pipes are not supported; use a preprocessor")
            if character == "(":
                raise self.error("functions are not supported; use a preprocessor")
            if character in "<>=!&":
                raise self.error("expressions are not supported; use a preprocessor")
            raise self.error("expected '.' or '['")

        return Query(text=self.text, steps=tuple(steps))


@cache
def compile(expression: str) -> Query:
    """Parse one observation-relative structural query."""

    if not isinstance(expression, str):
        raise TypeError(f"query must be a string, got {type(expression).__name__}")
    return Parser(expression).parse()


def listed(datatype: pa.DataType) -> bool:
    """Return whether an Arrow type owns one traversable list axis."""

    return pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype)


def nest(container: pa.DataType, child: pa.DataType) -> pa.DataType:
    """Replace a list container's value type while preserving its list kind."""

    value = pa.field(
        container.value_field.name,
        child,
        nullable=container.value_field.nullable,
        metadata=container.value_field.metadata,
    )
    if pa.types.is_large_list(container):
        return pa.large_list(value)
    if pa.types.is_fixed_size_list(container):
        return pa.list_(value, container.list_size)
    return pa.list_(value)


def bounds(length: int, start: int | None, stop: int | None) -> tuple[int, int]:
    """Resolve static Python slice bounds for a fixed-size list type."""

    lower, upper, _ = slice(start, stop).indices(length)
    return lower, max(lower, upper)


def sliced(datatype: pa.DataType, start: int | None, stop: int | None) -> pa.DataType:
    """Return the stable Arrow type produced by a list slice."""

    if not pa.types.is_fixed_size_list(datatype):
        return datatype

    lower, upper = bounds(datatype.list_size, start, stop)
    size = upper - lower
    if size == 0:
        return pa.list_(datatype.value_field)
    return pa.list_(datatype.value_field, size)


def context(query: Query, segment: str, address: str | None, reason: str) -> ValueError:
    """Build a query error with expression, segment, and model address context."""

    location = f" for address {address!r}" if address is not None else ""
    return ValueError(f"query {query.text!r}{location} cannot apply segment {segment!r}: {reason}")


@cache
def bind(query: Query | str, source: pa.Schema | pa.DataType, *, address: str | None = None) -> Plan:
    """Resolve neutral bracket operations against an Arrow schema."""

    parsed = compile(query) if isinstance(query, str) else query
    datatype = pa.struct(list(source)) if isinstance(source, pa.Schema) else source
    input_type = datatype
    steps: list[Bound] = []
    wrappers: list[pa.DataType] = []

    for step in parsed.steps:
        if isinstance(step, Member):
            if not pa.types.is_struct(datatype):
                raise context(parsed, step.segment, address, f"expected a struct, got {datatype}")
            indices = datatype.get_all_field_indices(step.name)
            if not indices:
                raise context(parsed, step.segment, address, f"field {step.name!r} is absent from {datatype}")
            if len(indices) > 1:
                raise context(parsed, step.segment, address, f"field {step.name!r} is ambiguous in {datatype}")
            datatype = datatype.field(indices[0]).type
            steps.append(step)
            continue

        if isinstance(step, Traverse):
            if not listed(datatype):
                raise context(parsed, step.segment, address, f"expected a list, got {datatype}")
            wrappers.append(datatype)
            datatype = datatype.value_type
            steps.append(step)
            continue

        if isinstance(step, Slice):
            if not listed(datatype):
                raise context(parsed, step.segment, address, f"expected a list, got {datatype}")
            datatype = sliced(datatype, step.start, step.stop)
            steps.append(step)
            continue

        if pa.types.is_struct(datatype) and isinstance(step.value, str):
            indices = datatype.get_all_field_indices(step.value)
            if not indices:
                raise context(parsed, step.segment, address, f"field {step.value!r} is absent from {datatype}")
            if len(indices) > 1:
                raise context(parsed, step.segment, address, f"field {step.value!r} is ambiguous in {datatype}")
            datatype = datatype.field(indices[0]).type
            steps.append(Member(name=step.value, segment=step.segment))
            continue

        if listed(datatype) and isinstance(step.value, int) and not isinstance(step.value, bool):
            datatype = datatype.value_type
            steps.append(Index(value=step.value, segment=step.segment))
            continue

        if pa.types.is_map(datatype):
            key_type = datatype.key_type
            compatible = (
                isinstance(step.value, bool)
                and pa.types.is_boolean(key_type)
                or isinstance(step.value, int)
                and not isinstance(step.value, bool)
                and pa.types.is_integer(key_type)
                or isinstance(step.value, str)
                and (pa.types.is_string(key_type) or pa.types.is_large_string(key_type))
            )
            if not compatible:
                raise context(
                    parsed,
                    step.segment,
                    address,
                    f"literal {step.value!r} is not an exact {key_type} map key",
                )
            try:
                pa.scalar(step.value, type=key_type)
            except (pa.ArrowException, OverflowError, TypeError, ValueError) as error:
                raise context(
                    parsed,
                    step.segment,
                    address,
                    f"literal {step.value!r} is not an exact {key_type} map key",
                ) from error
            datatype = datatype.item_type
            steps.append(Lookup(value=step.value, segment=step.segment))
            continue

        raise context(parsed, step.segment, address, f"bracket literal is invalid for {datatype}")

    output = datatype
    presence = pa.bool_()
    for wrapper in reversed(wrappers):
        output = nest(wrapper, output)
        presence = nest(wrapper, presence)

    return Plan(query=parsed, steps=tuple(steps), input=input_type, output=output, presence=presence)


def parts(values: pa.Array) -> tuple[np.ndarray, pa.Array, np.ndarray]:
    """Return list offsets, the underlying child array, and parent validity."""

    valid = pc.is_valid(values).to_numpy(zero_copy_only=False)
    if pa.types.is_fixed_size_list(values.type):
        start = values.offset * values.type.list_size
        stop = start + len(values) * values.type.list_size
        offsets = np.arange(start, stop + 1, values.type.list_size, dtype=np.int64)
        return offsets, values.values, valid

    offsets = values.offsets.to_numpy(zero_copy_only=False)
    return offsets, values.values, valid


def wrap(container: pa.Array, child: pa.Array, mask: pa.Array) -> pa.Array:
    """Rebuild one list container around transformed child values."""

    datatype = nest(container.type, child.type)
    if pa.types.is_fixed_size_list(container.type):
        return pa.FixedSizeListArray.from_arrays(child, type=datatype, mask=mask)

    offsets = container.offsets
    base = offsets[0].as_py()
    normalized = pc.subtract(offsets, pa.scalar(base, type=offsets.type))
    if pa.types.is_large_list(container.type):
        return pa.LargeListArray.from_arrays(normalized, child, type=datatype, mask=mask)
    return pa.ListArray.from_arrays(normalized, child, type=datatype, mask=mask)


def index(values: pa.Array, present: pa.Array, position: int) -> Projection:
    """Safely select one position from every list without bounds failures."""

    offsets, children, valid_parent = parts(values)
    lengths = np.diff(offsets)
    positions = lengths + position if position < 0 else np.full(len(values), position, dtype=np.int64)
    valid = valid_parent & (positions >= 0) & (positions < lengths)
    absolute = offsets[:-1] + np.maximum(positions, 0)
    indices = pa.array(absolute, mask=~valid, type=pa.int64())
    selected = pc.take(children, indices)
    structural = pc.and_(present, pa.array(valid, type=pa.bool_()))
    return Projection(values=selected, present=structural)


def subslice(values: pa.Array, present: pa.Array, start: int | None, stop: int | None) -> pa.Array:
    """Apply Python half-open slice bounds independently to every list."""

    offsets, children, valid = parts(values)
    lengths = np.diff(offsets)

    if start is None:
        lower = np.zeros(len(values), dtype=np.int64)
    elif start < 0:
        lower = np.maximum(lengths + start, 0)
    else:
        lower = np.minimum(start, lengths)

    if stop is None:
        upper = lengths.copy()
    elif stop < 0:
        upper = np.maximum(lengths + stop, 0)
    else:
        upper = np.minimum(stop, lengths)

    counts = np.maximum(upper - lower, 0)
    if pa.types.is_fixed_size_list(values.type):
        fixed_lower, fixed_upper = bounds(values.type.list_size, start, stop)
        counts.fill(fixed_upper - fixed_lower)
        lower.fill(fixed_lower)

    total = int(counts.sum())
    if total:
        parents = np.repeat(np.arange(len(values), dtype=np.int64), counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        positions = offsets[:-1][parents] + lower[parents] + np.arange(total, dtype=np.int64) - starts
        selected = pc.take(children, pa.array(positions, type=pa.int64()))
    else:
        selected = children.slice(0, 0)

    mask = pc.invert(pc.and_(present, pa.array(valid, type=pa.bool_())))
    datatype = sliced(values.type, start, stop)
    if pa.types.is_fixed_size_list(datatype):
        return pa.FixedSizeListArray.from_arrays(selected, type=datatype, mask=mask)

    result_offsets = pa.array(np.concatenate(([0], np.cumsum(counts))), type=pa.int64())
    if pa.types.is_large_list(datatype):
        return pa.LargeListArray.from_arrays(result_offsets, selected, type=datatype, mask=mask)
    return pa.ListArray.from_arrays(pc.cast(result_offsets, pa.int32()), selected, type=datatype, mask=mask)


def lookup(values: pa.Array, present: pa.Array, step: Lookup, plan: Plan, address: str | None) -> Projection:
    """Select one exact map key and reject duplicate matches."""

    matches = pc.map_lookup(values, step.value, occurrence="all")
    counts = pc.fill_null(pc.list_value_length(matches), 0)
    duplicated = pc.any(pc.greater(counts, 1)).as_py()
    if duplicated:
        raise context(plan.query, step.segment, address, f"map contains duplicate key {step.value!r}")
    return index(matches, present, 0)


def select(
    values: pa.Array,
    present: pa.Array,
    steps: tuple[Bound, ...],
    plan: Plan,
    address: str | None,
) -> Projection:
    """Execute bound steps against one non-chunked Arrow array."""

    if not steps:
        return Projection(values=values, present=present)

    step, *remaining = steps
    tail = tuple(remaining)
    if isinstance(step, Member):
        structural = pc.and_(present, pc.is_valid(values))
        child = pc.struct_field(values, step.name)
        return select(child, structural, tail, plan, address)
    if isinstance(step, Index):
        selected = index(values, present, step.value)
        return select(selected.values, selected.present, tail, plan, address)
    if isinstance(step, Slice):
        child = subslice(values, present, step.start, step.stop)
        return select(child, present, tail, plan, address)
    if isinstance(step, Lookup):
        selected = lookup(values, present, step, plan, address)
        return select(selected.values, selected.present, tail, plan, address)

    offsets, children, valid = parts(values)
    lengths = np.diff(offsets)
    parents = pa.array(np.repeat(np.arange(len(values), dtype=np.int64), lengths), type=pa.int64())
    child_present = pc.and_(pc.take(present, parents), pc.take(pa.array(valid), parents))
    start = int(offsets[0])
    stop = int(offsets[-1])
    selected = select(children.slice(start, stop - start), child_present, tail, plan, address)
    mask = pc.invert(pc.and_(present, pa.array(valid, type=pa.bool_())))
    return Projection(
        values=wrap(values, selected.values, mask),
        present=wrap(values, selected.present, mask),
    )


def query(
    source: pa.Table | pa.Array | pa.ChunkedArray,
    expression: Query | Plan | str,
    *,
    address: str | None = None,
) -> Projection:
    """Bind and execute a structural query without Python row materialization."""

    if isinstance(source, pa.Table):
        datatype = pa.struct(list(source.schema))
        values: pa.Array | pa.ChunkedArray = source.to_struct_array()
    elif isinstance(source, (pa.Array, pa.ChunkedArray)):
        datatype = source.type
        values = source
    else:
        raise TypeError(f"query source must be an Arrow Table, Array, or ChunkedArray, got {type(source).__name__}")

    if isinstance(expression, Plan):
        plan = expression
        if plan.input != datatype:
            raise TypeError(f"query plan expects Arrow input {plan.input}, got {datatype}")
    else:
        plan = bind(expression, datatype, address=address)

    if isinstance(values, pa.ChunkedArray):
        selected = [
            select(chunk, pa.repeat(pa.scalar(True), len(chunk)), plan.steps, plan, address) for chunk in values.chunks
        ]
        return Projection(
            values=pa.chunked_array([item.values for item in selected], type=plan.output),
            present=pa.chunked_array([item.present for item in selected], type=plan.presence),
        )

    selected = select(values, pa.repeat(pa.scalar(True), len(values)), plan.steps, plan, address)
    if selected.values.type != plan.output or selected.present.type != plan.presence:
        location = f" for address {address!r}" if address is not None else ""
        raise TypeError(
            f"query {plan.query.text!r}{location} produced Arrow types "
            f"({selected.values.type}, {selected.present.type}); expected ({plan.output}, {plan.presence})"
        )
    return selected


__all__ = ["Plan", "Projection", "Query", "bind", "compile", "query"]
