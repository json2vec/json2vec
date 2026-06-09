import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TypeAlias

import numpy as np
from beartype import beartype

from json2vec.structs.enums import Overflow, Strata, Tokens
from json2vec.structs.tree import Address

MASK_LITERAL = "<MASK>"
MaskLiteral: TypeAlias = Literal["<MASK>"]


def apply(
    values: Any,
    function: Callable[..., Any],
    /,
    *args: Any,
    leaf_depth: int | None = None,
    **kwargs: Any,
) -> Any:
    """Apply a function recursively to nested list leaves.

    When ``leaf_depth`` is set, the function is applied exactly at that depth;
    higher-level non-list values are preserved so downstream padding can mark
    them as incomplete.
    """
    if leaf_depth is not None and leaf_depth < 0:
        raise ValueError("leaf_depth must be >= 0")

    def walk(node: Any, depth: int) -> Any:
        if leaf_depth is None:
            if isinstance(node, list):
                return [walk(item, depth + 1) for item in node]

            if node is None:
                return None

            return function(node, *args, **kwargs)

        if depth == leaf_depth:
            if node is None:
                return None

            return function(node, *args, **kwargs)

        if isinstance(node, list):
            return [walk(item, depth + 1) for item in node]

        return node

    return walk(values, depth=0)


def contains_mask_literal(value: Any) -> bool:
    """Return whether a JSON-like value contains the predict-time mask sentinel."""
    if isinstance(value, str):
        return value == MASK_LITERAL

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return bool(value.item() == MASK_LITERAL)
        return any(contains_mask_literal(item) for item in value.tolist())

    if isinstance(value, (list, tuple)):
        return any(contains_mask_literal(item) for item in value)

    return False


def extract_mask_literals(
    values: Any,
    *,
    strata: Strata,
    address: Address,
    leaf_depth: int,
) -> tuple[Any, Any]:
    """Replace whole-leaf ``"<MASK>"`` values with ``None`` plus boolean flags.

    The sentinel is only accepted in predict strata. Structured leaf values
    such as vectors and sets may use ``"<MASK>"`` as the whole field value, but
    not as one item inside the vector or set payload.
    """
    if leaf_depth < 0:
        raise ValueError("leaf_depth must be >= 0")

    found = False

    def walk(node: Any, depth: int) -> tuple[Any, Any]:
        nonlocal found

        if depth == leaf_depth:
            if isinstance(node, str) and node == MASK_LITERAL:
                found = True
                return None, True

            if contains_mask_literal(node):
                raise ValueError(
                    f"input for address '{address}' may only use {MASK_LITERAL!r} as the whole field value"
                )

            return node, False

        if isinstance(node, list):
            cleaned: list[Any] = []
            flags: list[Any] = []
            for item in node:
                child_value, child_flag = walk(item, depth + 1)
                cleaned.append(child_value)
                flags.append(child_flag)
            return cleaned, flags

        if isinstance(node, tuple):
            cleaned_items: list[Any] = []
            flag_items: list[Any] = []
            for item in node:
                child_value, child_flag = walk(item, depth + 1)
                cleaned_items.append(child_value)
                flag_items.append(child_flag)
            return tuple(cleaned_items), tuple(flag_items)

        if contains_mask_literal(node):
            raise ValueError(
                f"input for address '{address}' may only use {MASK_LITERAL!r} at the configured field depth"
            )

        return node, False

    cleaned_values, mask_flags = walk(values, depth=0)
    if found and strata != Strata.predict:
        raise ValueError(f"{MASK_LITERAL!r} is only valid during predict strata for address '{address}'")

    return cleaned_values, mask_flags


def _iter_leaf_nodes(
    nested: Any,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    overflows: tuple[Overflow, ...],
    address: Address | str | None = None,
):
    ndim = len(shape)
    stack: list[tuple[Any, int, int]] = [(nested, 0, 0)]

    while stack:
        node, depth, base = stack.pop()

        if depth == ndim:
            yield base, node
            continue

        if not isinstance(node, (list, tuple, np.ndarray)):
            continue

        if isinstance(node, np.ndarray) and node.ndim == 0:
            continue

        length = len(node)
        max_length = shape[depth]
        policy = overflows[depth]

        if policy == Overflow.error and length > max_length:
            provided_shape = (*shape[:depth], length, *shape[depth + 1 :])
            if depth == 0:
                context = "batch dimension 0"
            elif depth == 1:
                context = "root node dimension 1"
            else:
                context = f"dimension {depth}"
            if address is not None:
                context = f"{context} for {Address(str(address))}"
            raise ValueError(f"array overflow at {context}: expected shape {shape}, provided shape {provided_shape}")

        limit = min(length, max_length)
        if policy == Overflow.tail:
            source_start = length - limit
            indices = [(source_start + destination, destination) for destination in range(limit)]
        else:
            indices = [(destination, destination) for destination in range(limit)]

        step = strides[depth]
        for source_index, destination_index in reversed(indices):
            stack.append((node[source_index], depth + 1, base + (destination_index * step)))


def _fill_python(
    nested: Any,
    flat_values: np.ndarray,
    flat_flags: np.ndarray,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    overflows: tuple[Overflow, ...],
    address: Address | str | None,
    encode: Callable[[Any], Any] | None,
) -> None:
    for flat_index, node in _iter_leaf_nodes(
        nested=nested,
        shape=shape,
        strides=strides,
        overflows=overflows,
        address=address,
    ):
        if node is None:
            flat_flags[flat_index] = Tokens.null.value
        else:
            flat_values[flat_index] = encode(node) if encode is not None else node
            flat_flags[flat_index] = Tokens.valued.value


@beartype
def pad(
    nested: Any,
    shape: tuple[int, ...],
    dtype: type | str = object,
    pad_value: Any = None,
    overflows: tuple[Overflow, ...] | None = None,
    address: Address | str | None = None,
    value_shape: tuple[int, ...] = (),
    encode: Callable[[Any], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_dtype = np.dtype(dtype)
    values = np.full((*shape, *value_shape), pad_value, dtype=resolved_dtype)
    flags = np.full(shape, Tokens.padded.value, dtype=np.int8)

    ndim = len(shape)
    resolved_overflows = (Overflow.head,) * ndim if overflows is None else tuple(Overflow(item) for item in overflows)
    if len(resolved_overflows) != ndim:
        raise ValueError(f"overflows length must match shape rank: expected {ndim}, got {len(resolved_overflows)}")

    if ndim == 0:
        if nested is None:
            flags[...] = Tokens.null.value
        else:
            values[...] = encode(nested) if encode is not None else nested
            flags[...] = Tokens.valued.value
        return values, flags

    strides = [1] * ndim
    for depth in range(ndim - 2, -1, -1):
        strides[depth] = strides[depth + 1] * shape[depth + 1]
    stride_tuple = tuple(strides)

    flat_values = values.reshape(-1, *value_shape)
    flat_flags = flags.reshape(-1)

    _fill_python(
        nested=nested,
        flat_values=flat_values,
        flat_flags=flat_flags,
        shape=shape,
        strides=stride_tuple,
        overflows=resolved_overflows,
        address=address,
        encode=encode,
    )

    return values, flags


@beartype
class Pipeline:
    def __init__(self, **arguments):
        self.arguments: dict[str, Any] = arguments
        self.steps: list[Callable] = []

    def __or__(self, function: Callable) -> "Pipeline":
        required = [name for name in inspect.signature(function).parameters.keys()]

        available = set(required) & set(self.arguments.keys())

        self.steps.append(partial(function, **{arg: self.arguments[arg] for arg in available}))

        return self

    def __repr__(self):
        return f"Pipeline(steps={len(self.steps)}, arguments={self.arguments!r})"

    def __iter__(self):
        stream = self.steps[0]()

        for step in self.steps[1:]:
            stream = step(stream)

        return iter(stream)
