"""Composable schema node selectors and selection cache models."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from itertools import islice
from typing import Any, TypeAlias

import pydantic

from relflow.rich import strip_control_sequences
from relflow.structs.structure import Branch
from relflow.structs.tree import Leaf, Node

SelectionKey: TypeAlias = tuple[Any, ...]
SchemaField: TypeAlias = Branch | Leaf

_EXPRESSION_COLLECTION_LIMIT = 8
_EXPRESSION_DEPTH_LIMIT = 4
_EXPRESSION_STRING_LIMIT = 80
_EXPRESSION_TOTAL_LIMIT = 320


def _bounded_expression_text(value: str, *, limit: int = _EXPRESSION_STRING_LIMIT) -> str:
    value = " ".join(strip_control_sequences(value).split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


def _expression_value(value: Any, *, depth: int = 0) -> str:
    """Return a deterministic, bounded value representation for selectors."""

    if isinstance(value, Enum):
        return _expression_value(value.value, depth=depth)
    if isinstance(value, str):
        rendered = repr(value)
        if len(rendered) <= _EXPRESSION_STRING_LIMIT:
            return rendered

        keep = min(len(value), _EXPRESSION_STRING_LIMIT // 2)
        rendered = repr(f"{value[:keep]}…")
        while len(rendered) > _EXPRESSION_STRING_LIMIT and keep:
            keep -= 1
            rendered = repr(f"{value[:keep]}…")
        return rendered
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)

    if depth >= _EXPRESSION_DEPTH_LIMIT and isinstance(value, (Mapping, list, tuple, set, frozenset)):
        type_name = _bounded_expression_text(type(value).__name__)
        return f"<{type_name} items={len(value)}>"

    if isinstance(value, Mapping):
        parts: list[str] = []
        for index, (key, item) in enumerate(value.items()):
            if index >= _EXPRESSION_COLLECTION_LIMIT:
                break
            parts.append(f"{_expression_value(key, depth=depth + 1)}: {_expression_value(item, depth=depth + 1)}")
        omitted = len(value) - len(parts)
        if omitted:
            parts.append(f"… +{omitted}")
        return "{" + ", ".join(parts) + "}"

    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_expression_value(item, depth=depth + 1) for item in islice(value, _EXPRESSION_COLLECTION_LIMIT)]
        if isinstance(value, (set, frozenset)):
            items.sort()
        omitted = len(value) - len(items)
        if omitted:
            items.append(f"… +{omitted}")

        if isinstance(value, tuple):
            body = ", ".join(items)
            if len(items) == 1 and not omitted:
                body += ","
            return f"({body})"
        if isinstance(value, (set, frozenset)):
            return "{" + ", ".join(items) + "}"
        return "[" + ", ".join(items) + "]"

    return f"<{_bounded_expression_text(type(value).__name__)}>"


def _callable_label(func: Callable[[Node], bool]) -> str:
    name = getattr(func, "__name__", type(func).__name__)
    return "callable" if name == "<lambda>" else name


def _predicate_expression(
    key: SelectionKey,
    *,
    func: Callable[[Node], bool],
    depth: int = 0,
) -> str:
    if depth >= _EXPRESSION_DEPTH_LIMIT:
        return "…"
    if not key:
        return "predicate()"

    operation = key[0]
    if operation == "all" and len(key) == 1:
        return "all_nodes()"
    if operation == "callable" and len(key) == 2:
        label = _callable_label(func) if isinstance(key[1], int) else str(key[1])
        return f"predicate({_expression_value(label)})"
    if operation in {"truthy", "not_truthy", "exists", "is_null", "is_not_null"} and len(key) == 2:
        attribute = f"where({_expression_value(key[1])})"
        return {
            "truthy": attribute,
            "not_truthy": f"~{attribute}",
            "exists": f"{attribute}.exists()",
            "is_null": f"{attribute}.is_null()",
            "is_not_null": f"{attribute}.is_not_null()",
        }[operation]
    if operation in {"eq", "ne", "contains", "matches", "is_in"} and len(key) == 3:
        attribute = f"where({_expression_value(key[1])})"
        value = _expression_value(key[2])
        return {
            "eq": f"{attribute} == {value}",
            "ne": f"{attribute} != {value}",
            "contains": f"{attribute}.contains({value})",
            "matches": f"{attribute}.matches({value})",
            "is_in": f"{attribute}.is_in({value})",
        }[operation]
    if operation == "not" and len(key) == 2 and isinstance(key[1], tuple):
        return f"~({_predicate_expression(key[1], func=func, depth=depth + 1)})"
    if operation in {"and", "or"} and len(key) == 2 and isinstance(key[1], tuple):
        children = key[1]
        bounded_children = tuple(islice(children, _EXPRESSION_COLLECTION_LIMIT))
        expressions = [
            _predicate_expression(child, func=func, depth=depth + 1)
            for child in bounded_children
            if isinstance(child, tuple)
        ]
        omitted = len(children) - len(bounded_children)
        if omitted:
            expressions.append(f"… (+{omitted})")
        operator = " & " if operation == "and" else " | "
        return operator.join(f"({expression})" for expression in expressions)

    return f"predicate(key={_expression_value(key)})"


class SelectionCacheEntry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True, hide_input_in_errors=True)

    key: SelectionKey
    predicate: Callable[[Node], bool]
    include_root: bool
    nodes: tuple[Node, ...]


class NodePredicate(pydantic.BaseModel):
    """Composable predicate used to select schema nodes."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True, hide_input_in_errors=True)

    func: Callable[[Node], bool]
    key: SelectionKey
    cacheable: bool = True

    @property
    def expression(self) -> str:
        expression = _predicate_expression(self.key, func=self.func)
        return _bounded_expression_text(expression, limit=_EXPRESSION_TOTAL_LIMIT)

    def __repr__(self) -> str:
        return self.expression

    def __str__(self) -> str:
        return self.expression

    def __rich_repr__(self):
        yield "expression", self.expression

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

    model_config = pydantic.ConfigDict(frozen=True, hide_input_in_errors=True)

    name: str = pydantic.Field(
        description=(
            "Queryable node attribute. Built-ins include name, type, address, parent, "
            "children, ancestors, descendants, and target. Pydantic fields and "
            "extra metadata fields are also queryable."
        )
    )

    @property
    def expression(self) -> str:
        return f"where({_expression_value(self.name)})"

    def __repr__(self) -> str:
        return self.expression

    def __str__(self) -> str:
        return self.expression

    def __rich_repr__(self):
        yield "expression", self.expression

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
            return isinstance(node, Leaf) and node.active and node.target

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
