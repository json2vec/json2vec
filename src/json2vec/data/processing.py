import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from beartype import beartype

from json2vec.structs.enums import Tokens


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


def _iter_leaf_nodes(
    nested: Any,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
):
    ndim = len(shape)
    stack: list[tuple[Any, int, int]] = [(nested, 0, 0)]

    while stack:
        node, depth, base = stack.pop()

        if depth == ndim:
            yield base, node
            continue

        if not isinstance(node, list):
            continue

        limit = min(len(node), shape[depth])
        step = strides[depth]
        for index in range(limit - 1, -1, -1):
            stack.append((node[index], depth + 1, base + (index * step)))


def _fill_python(
    nested: Any,
    flat_values: np.ndarray,
    flat_flags: np.ndarray,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> None:
    for flat_index, node in _iter_leaf_nodes(nested=nested, shape=shape, strides=strides):
        if node is None:
            flat_flags[flat_index] = Tokens.null.value
        else:
            flat_values[flat_index] = node
            flat_flags[flat_index] = Tokens.valued.value


@beartype
def pad(
    nested: Any, shape: tuple[int, ...], dtype: type | str = object, pad_value: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    resolved_dtype = np.dtype(dtype)
    values = np.full(shape, pad_value, dtype=resolved_dtype)
    flags = np.full(shape, Tokens.padded.value, dtype=np.int8)

    ndim = len(shape)
    if ndim == 0:
        if nested is None:
            flags[...] = Tokens.null.value
        else:
            values[...] = nested
            flags[...] = Tokens.valued.value
        return values, flags

    strides = [1] * ndim
    for depth in range(ndim - 2, -1, -1):
        strides[depth] = strides[depth + 1] * shape[depth + 1]
    stride_tuple = tuple(strides)

    flat_values = values.reshape(-1)
    flat_flags = flags.reshape(-1)

    _fill_python(
        nested=nested,
        flat_values=flat_values,
        flat_flags=flat_flags,
        shape=shape,
        strides=stride_tuple,
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
        return f"Pipeline({repr(self.source)}, {repr(self.arguments)})"

    def __iter__(self):
        stream = self.steps[0]()

        for step in self.steps[1:]:
            stream = step(stream)

        return iter(stream)
