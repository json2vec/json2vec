"""Composable schema node selectors and selection cache models."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import pydantic

from json2vec.structs.structure import Array
from json2vec.structs.tree import Leaf, Node

SelectionKey: TypeAlias = tuple[Any, ...]
SchemaField: TypeAlias = Array | Leaf


class SelectionCacheEntry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: SelectionKey
    predicate: Callable[[Node], bool]
    include_root: bool
    nodes: tuple[Node, ...]


class NodePredicate(pydantic.BaseModel):
    """Composable predicate used to select schema nodes."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    func: Callable[[Node], bool]
    key: SelectionKey
    cacheable: bool = True

    @classmethod
    def from_callable(cls, key: str | tuple[Any, ...], func: Callable[[Node], bool]) -> "NodePredicate":
        cache_key = key if isinstance(key, tuple) else ("callable", key)
        return cls(func=func, key=cache_key)

    @classmethod
    def from_selector(cls, value: "NodeSelector") -> "NodePredicate":
        if isinstance(value, cls):
            return value

        if isinstance(value, NodeAttribute):
            return cls(
                func=lambda node: _has_model_attribute(node, value.name) and value.get(node) is True,
                key=("truthy", value.name),
            )

        if not callable(value):
            raise TypeError("node predicates must be where(...) expressions or callables")

        return cls(
            func=value,
            key=("callable", id(value)),
            cacheable=True,
        )

    def __call__(self, node: Node) -> bool:
        return self.func(node)

    def __and__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> "NodePredicate":
        predicate = NodePredicate.from_selector(other)
        return NodePredicate(
            func=lambda node: self(node) and predicate(node),
            key=("and", (self.key, predicate.key)),
            cacheable=self.cacheable and predicate.cacheable,
        )

    def __or__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> "NodePredicate":
        predicate = NodePredicate.from_selector(other)
        return NodePredicate(
            func=lambda node: self(node) or predicate(node),
            key=("or", (self.key, predicate.key)),
            cacheable=self.cacheable and predicate.cacheable,
        )

    def __invert__(self) -> "NodePredicate":
        return NodePredicate(
            func=lambda node: not self(node),
            key=("not", self.key),
            cacheable=self.cacheable,
        )


def _cache_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def predicate(key: str | tuple[Any, ...], func: Callable[[Node], bool]) -> NodePredicate:
    """Create a cacheable node predicate from a callable."""
    return NodePredicate.from_callable(key=key, func=func)


_QUERYABLE_BUILTINS = frozenset(
    {
        "address",
        "parent",
        "children",
        "ancestors",
        "descendants",
        "target",
    }
)


class NodeAttribute(pydantic.BaseModel):
    """Queryable schema node attribute returned by `where(...)`."""

    model_config = pydantic.ConfigDict(frozen=True)

    name: str = pydantic.Field(
        description=(
            "Queryable node attribute. Built-ins include name, type, address, parent, "
            "children, ancestors, descendants, and target. Pydantic fields and "
            "extra metadata fields are also queryable."
        )
    )

    @classmethod
    def named(cls, name: str) -> "NodeAttribute":
        return cls(name=name)

    def get(self, node: Node, default: Any = None) -> Any:
        if self.name == "address":
            return str(node.address)
        if self.name == "parent":
            parent = getattr(node, "parent", None)
            return None if parent is None or not getattr(parent, "address", None) else str(parent.address)
        if self.name == "children":
            return tuple(str(child.address) for child in getattr(node, "children", ()))
        if self.name == "ancestors":
            return tuple(str(parent.address) for parent in getattr(node, "ancestors", ()) if parent.address)
        if self.name == "descendants":
            return tuple(str(child.address) for child in getattr(node, "descendants", ()))
        if self.name == "target":
            return isinstance(node, Leaf) and node.active and getattr(node, "p_prune", None) == 1.0

        extra = getattr(node, "model_extra", None) or {}
        if self.name in extra:
            return extra[self.name]

        return getattr(node, self.name, default)

    def exists(self) -> NodePredicate:
        return NodePredicate(
            func=lambda node: _has_model_attribute(node, self.name),
            key=("exists", self.name),
        )

    def __and__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> NodePredicate:
        return NodePredicate.from_selector(self) & other

    def __or__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> NodePredicate:
        return NodePredicate.from_selector(self) | other

    def __invert__(self) -> NodePredicate:
        return NodePredicate(
            func=lambda node: not bool(self.get(node, False)),
            key=("not_truthy", self.name),
        )

    def __bool__(self) -> bool:
        raise TypeError("Use ~where(...) for negated predicates; Python 'not where(...)' cannot build a predicate")

    def is_in(self, values: Iterable[Any]) -> NodePredicate:
        cached_values = tuple(values)
        return NodePredicate(
            func=lambda node: self.get(node) in cached_values,
            key=(
                "is_in",
                self.name,
                tuple(sorted((_cache_value(value) for value in cached_values), key=repr)),
            ),
        )

    def matches(self, pattern: str | re.Pattern[str]) -> NodePredicate:
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        return NodePredicate(
            func=lambda node: regex.search(str(self.get(node, ""))) is not None,
            key=("matches", self.name, regex.pattern),
        )

    def contains(self, value: Any) -> NodePredicate:
        return NodePredicate(
            func=lambda node: value in (self.get(node) or ()),
            key=("contains", self.name, _cache_value(value)),
        )

    def is_null(self) -> NodePredicate:
        return NodePredicate(
            func=lambda node: self.get(node) is None,
            key=("is_null", self.name),
        )

    def is_not_null(self) -> NodePredicate:
        return NodePredicate(
            func=lambda node: self.get(node) is not None,
            key=("is_not_null", self.name),
        )

    def __eq__(self, other: Any) -> NodePredicate:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return NodePredicate(
            func=lambda node: self.get(node) == other,
            key=("eq", self.name, _cache_value(other)),
        )

    def __ne__(self, other: Any) -> NodePredicate:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return NodePredicate(
            func=lambda node: self.get(node) != other,
            key=("ne", self.name, _cache_value(other)),
        )


def where(name: str) -> NodeAttribute:
    """Start a schema predicate against a node attribute."""
    return NodeAttribute.named(name)


NodeSelector: TypeAlias = NodePredicate | NodeAttribute | Callable[[Node], bool]
ExtendArg: TypeAlias = NodeSelector | SchemaField


def _has_model_attribute(node: Node, name: str) -> bool:
    if name in _QUERYABLE_BUILTINS:
        return True

    fields = getattr(type(node), "model_fields", {})
    extra = getattr(node, "model_extra", None) or {}
    return name in fields or name in extra or hasattr(node, name)
