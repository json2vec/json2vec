from __future__ import annotations

import functools
import json
import logging
import re
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

import pydantic
from anytree import LevelOrderGroupIter, PreOrderIter, RenderTree

from json2vec.structs.structure import Array, RequestTypes
from json2vec.structs.tree import Address, Leaf, Node

logger = logging.getLogger("json2vec.hyperparameters")
SelectionKey: TypeAlias = tuple[Any, ...]
SchemaField: TypeAlias = Array | Leaf
_MISSING = object()


class SelectionCacheEntry(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: SelectionKey
    predicate: Callable[[Node], bool]
    include_root: bool
    nodes: tuple[Node, ...]


class NodeSelection(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    owner: Any = pydantic.Field(exclude=True)
    key: SelectionKey | None = None
    nodes: tuple[Node, ...]

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def to_list(self) -> list[Node]:
        return list(self.nodes)

    def set(
        self,
        strict: bool = True,
        allow_extra: bool = False,
        validate: bool = True,
        **values: Any,
    ) -> "Hyperparameters":
        return self.owner._set_nodes(
            self.nodes,
            strict=strict,
            allow_extra=allow_extra,
            validate=validate,
            **values,
        )


class MutationChange(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    node: str
    field: str
    old: Any
    new: Any
    action: Literal["set", "restore"]


class MutationResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    operation_id: str
    parent_operation_id: str | None = None
    action: Literal["set", "restore"] = "set"
    matched: int
    updated: int
    skipped: int = 0
    changes: tuple[MutationChange, ...] = ()


class NodePredicate(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    func: Callable[[Node], bool]
    key: SelectionKey
    cacheable: bool = True

    def __call__(self, node: Node) -> bool:
        return self.func(node)

    def __and__(self, other: "NodePredicate | Callable[[Node], bool]") -> "NodePredicate":
        predicate = _normalize_predicate(other)
        return NodePredicate(
            func=lambda node: self(node) and predicate(node),
            key=("and", (self.key, predicate.key)),
            cacheable=self.cacheable and predicate.cacheable,
        )

    def __or__(self, other: "NodePredicate | Callable[[Node], bool]") -> "NodePredicate":
        predicate = _normalize_predicate(other)
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
    cache_key = key if isinstance(key, tuple) else ("callable", key)
    return NodePredicate(func=func, key=cache_key)


_QUERYABLE_BUILTINS = frozenset(
    {
        "address",
        "parent",
        "children",
        "ancestors",
        "descendants",
    }
)


class NodeAttribute(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    name: str = pydantic.Field(
        description=(
            "Queryable node attribute. Built-ins include name, type, address, parent, "
            "children, ancestors, and descendants. Pydantic fields and extra metadata "
            "fields are also queryable."
        )
    )

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

        extra = getattr(node, "model_extra", None) or {}
        if self.name in extra:
            return extra[self.name]

        return getattr(node, self.name, default)

    def exists(self) -> NodePredicate:
        return NodePredicate(
            func=lambda node: _has_model_attribute(node, self.name),
            key=("exists", self.name),
        )

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

    def __eq__(self, other: Any) -> NodePredicate:  # type: ignore[override]
        return NodePredicate(
            func=lambda node: self.get(node) == other,
            key=("eq", self.name, _cache_value(other)),
        )

    def __ne__(self, other: Any) -> NodePredicate:  # type: ignore[override]
        return NodePredicate(
            func=lambda node: self.get(node) != other,
            key=("ne", self.name, _cache_value(other)),
        )


def where(name: str) -> NodeAttribute:
    return NodeAttribute(name=name)


def _has_model_attribute(node: Node, name: str) -> bool:
    if name in _QUERYABLE_BUILTINS:
        return True

    fields = getattr(type(node), "model_fields", {})
    extra = getattr(node, "model_extra", None) or {}
    return name in fields or name in extra or hasattr(node, name)


def _allows_extra_attributes(node: Node) -> bool:
    model_config = getattr(type(node), "model_config", {})
    return model_config.get("extra") == "allow"


def _validate_node_update(node: Node, values: Mapping[str, Any]) -> None:
    payload = node.model_dump(mode="python", round_trip=True)
    payload.update(values)
    type(node).model_validate(payload)


def _normalize_predicate(value: NodePredicate | Callable[[Node], bool]) -> NodePredicate:
    if isinstance(value, NodePredicate):
        return value

    return NodePredicate(
        func=value,
        key=("callable", id(value)),
        cacheable=True,
    )


def _combined(predicates: tuple[NodePredicate | Callable[[Node], bool], ...]) -> NodePredicate:
    if not predicates:
        return NodePredicate(func=lambda node: True, key=("all",))

    normalized = tuple(_normalize_predicate(item) for item in predicates)
    return NodePredicate(
        func=lambda node: all(item(node) for item in normalized),
        key=("and", tuple(item.key for item in normalized)),
        cacheable=all(item.cacheable for item in normalized),
    )


def _log_mutation(result: MutationResult) -> None:
    logger.info(
        "json2vec.hyperparameters.%s",
        result.action,
        extra={"mutation": result.model_dump(mode="python", exclude={"changes"})},
    )
    for change in result.changes:
        logger.debug(
            "json2vec.hyperparameters.change",
            extra={"mutation_change": change.model_dump(mode="python")},
        )


def sanitize_node_name(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return sanitized or "field"


def jmespath_member(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return value
    return json.dumps(value)


def _request_from_leaf(leaf: Leaf) -> RequestTypes:
    from json2vec.tensorfields.base import TENSORFIELDS

    request_cls = getattr(TENSORFIELDS[leaf.type], "Request")
    return request_cls.model_validate(leaf.model_dump(mode="python", round_trip=True))


def _schema_query(array_path: tuple[str, ...], source: str) -> str:
    selectors = "".join(f".{jmespath_member(array)}[*]" for array in array_path)
    return f"[*]{selectors}.{jmespath_member(source)}"


def _normalize_schema_node(node: SchemaField, *, array_path: tuple[str, ...] = ()) -> Array | RequestTypes:
    if isinstance(node, Leaf):
        source = node.name
        node_name = sanitize_node_name(source)
        updates: dict[str, Any] = {}

        if node_name != source:
            updates["name"] = node_name
            if node.description is None:
                updates["description"] = source

        if node.query is None:
            updates["query"] = _schema_query(array_path, source)

        return _request_from_leaf(node.model_copy(update=updates))

    if isinstance(node, Array):
        child_path = (*array_path, node.name)
        fields = [_normalize_schema_node(field, array_path=child_path) for field in node.fields]
        payload = node.model_dump(mode="python", round_trip=True, exclude={"fields"})
        return Array(*fields, **payload)

    raise TypeError("schema fields must be Array, Leaf, or concrete request instances")


def _materialize_raw_leaves(array: Array) -> Array:
    fields: list[Array | RequestTypes] = []
    for field in list(array.fields):
        field.parent = None

        if isinstance(field, Array):
            fields.append(_materialize_raw_leaves(field))
            continue

        if type(field) is Leaf:
            fields.append(_request_from_leaf(field))
            continue

        fields.append(field)

    array.fields = fields
    for field in array.fields:
        field.parent = array

    return array


class Hyperparameters(Node):
    model_config = pydantic.ConfigDict(extra="forbid")

    name: Literal["hyperparameters"] = pydantic.Field(default="hyperparameters", exclude=True)
    type: Literal["hyperparameters"] = pydantic.Field(default="hyperparameters", exclude=True)
    description: Literal[None] = pydantic.Field(default=None, exclude=True)
    d_model: Annotated[int, pydantic.Field(gt=0, default=128)]
    fields: Array

    embed: ClassVar[None] = None
    p_prune: ClassVar[None] = None
    dropout: ClassVar[None] = None
    p_mask: ClassVar[None] = None

    _selection_cache: dict[SelectionKey, SelectionCacheEntry] = pydantic.PrivateAttr(default_factory=dict)
    _mutation_history: list[MutationResult] = pydantic.PrivateAttr(default_factory=list)

    @classmethod
    def schema(
        cls,
        *field_args: SchemaField,
        d_model: int,
        n_layers: int,
        n_heads: int,
        fields: Sequence[SchemaField] | None = None,
        root: str = "record",
    ) -> Self:
        normalized = [*(fields or ()), *field_args]
        if not normalized:
            raise ValueError("schema requires at least one field")

        seen_sources: set[str] = set()
        root_fields: list[Array | RequestTypes] = []

        for field in normalized:
            if not isinstance(field, (Array, Leaf)):
                raise TypeError("schema fields must be Array, Leaf, or concrete request instances")

            source = field.name
            if source in seen_sources:
                raise ValueError(f"duplicate schema source field: {source}")
            seen_sources.add(source)

            root_fields.append(_normalize_schema_node(field))

        array = Array(
            name=root,
            n_layers=n_layers,
            n_heads=n_heads,
            n_outputs=1,
            max_length=1,
            fields=root_fields,
        )
        return cls(d_model=d_model, fields=array)

    def model_post_init(self, __context):
        self.fields = _materialize_raw_leaves(self.fields)
        self.fields.parent: Self = self
        for request in self.requests.values():
            request.post_bind_validate()

    @property
    def target(self) -> list[Address]:
        role = NodePredicate(
            func=lambda node: getattr(node, "p_prune", 0.0) == 1.0,
            key=("role", "target"),
        )
        return [Address(str(node.address)) for node in self.select(role).to_list()]

    @property
    def embed(self) -> list[Address]:  # noqa: F811
        role = NodePredicate(
            func=lambda node: getattr(node, "embed", False) is True,
            key=("role", "embed"),
        )
        return [Address(str(node.address)) for node in self.select(role).to_list()]

    @property
    def last_mutation(self) -> MutationResult | None:
        return self._mutation_history[-1] if self._mutation_history else None

    @property
    def mutation_history(self) -> tuple[MutationResult, ...]:
        return tuple(self._mutation_history)

    @functools.cached_property
    def arrays(self) -> dict[Address, Array]:
        return {node.address: node for node in self.descendants if isinstance(node, Array)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if not isinstance(node, Array)}

    @functools.cached_property
    def shapes(self) -> dict[Address, tuple[int, ...]]:
        return {request.address: request.shape for request in self.requests.values()}

    @functools.cached_property
    def depthwise(self) -> list[list[Address]]:
        out: list[list[Address]] = []
        for depth in LevelOrderGroupIter(self.fields):
            arrays = [node.address for node in depth if isinstance(node, Array)]
            if arrays:
                out.append(arrays)

        return out

    def _walk_nodes(self, include_root: bool = True) -> Iterator[Node]:
        for node in PreOrderIter(self.fields):
            if node is self.fields and not include_root:
                continue
            yield node

    def _record_mutation(self, result: MutationResult) -> None:
        self._mutation_history.append(result)
        _log_mutation(result)

    def _clear_tree_caches(self) -> None:
        for name in ("arrays", "requests", "shapes", "depthwise"):
            self.__dict__.pop(name, None)

        for node in PreOrderIter(self.fields):
            for name in ("address", "heritage", "shape"):
                node.__dict__.pop(name, None)

    def clear_selection_cache(self) -> None:
        self._selection_cache.clear()

    def refresh_selection_cache(self) -> None:
        self._selection_cache = {
            key: entry.model_copy(
                update={
                    "nodes": tuple(
                        node
                        for node in self._walk_nodes(include_root=entry.include_root)
                        if entry.predicate(node)
                    )
                }
            )
            for key, entry in self._selection_cache.items()
        }

    def nodes(
        self,
        *predicates: NodePredicate | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
    ) -> Iterator[Node]:
        yield from self.select(*predicates, include_root=include_root, use_cache=use_cache)

    def select(
        self,
        *predicates: NodePredicate | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
    ) -> NodeSelection:
        combined = _combined(predicates)
        key = ("select", include_root, combined.key)

        if use_cache and combined.cacheable and key in self._selection_cache:
            return NodeSelection(owner=self, key=key, nodes=self._selection_cache[key].nodes)

        nodes = tuple(
            node
            for node in self._walk_nodes(include_root=include_root)
            if combined(node)
        )

        if use_cache and combined.cacheable:
            self._selection_cache[key] = SelectionCacheEntry(
                key=key,
                predicate=combined,
                include_root=include_root,
                nodes=nodes,
            )

        return NodeSelection(owner=self, key=key, nodes=nodes)

    def set(
        self,
        *predicates: NodePredicate | Callable[[Node], bool],
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> Self:
        nodes = tuple(self.nodes(*predicates, include_root=include_root))
        return self._set_nodes(
            nodes,
            strict=strict,
            allow_extra=allow_extra,
            validate=validate,
            **values,
        )

    def _set_nodes(
        self,
        nodes: Sequence[Node],
        *,
        strict: bool = True,
        allow_extra: bool = False,
        validate: bool = True,
        action: Literal["set", "restore"] = "set",
        parent_operation_id: str | None = None,
        **values: Any,
    ) -> Self:
        if not values:
            raise ValueError("set requires at least one field value")

        changes: list[MutationChange] = []
        matched = updated = skipped = 0

        for node in nodes:
            matched += 1
            missing = [
                name
                for name in values
                if not _has_model_attribute(node, name)
                and not (allow_extra and _allows_extra_attributes(node))
            ]
            if missing and strict:
                label = str(node.address) or node.name
                raise AttributeError(f"{label} has no attribute(s): {missing}")

            applicable_values = {
                name: value
                for name, value in values.items()
                if _has_model_attribute(node, name)
                or (allow_extra and _allows_extra_attributes(node))
            }
            skipped += len(values) - len(applicable_values)

            if validate and applicable_values:
                _validate_node_update(node, applicable_values)

            changed = False
            for name, value in applicable_values.items():
                old = getattr(node, name, _MISSING)
                setattr(node, name, value)
                if name in getattr(type(node), "model_fields", {}):
                    node.model_fields_set.add(name)
                if old != value:
                    changes.append(
                        MutationChange(
                            node=str(node.address),
                            field=name,
                            old=None if old is _MISSING else old,
                            new=value,
                            action=action,
                        )
                    )
                    changed = True

            updated += int(changed)

        result = MutationResult(
            operation_id=uuid.uuid4().hex,
            parent_operation_id=parent_operation_id,
            action=action,
            matched=matched,
            updated=updated,
            skipped=skipped,
            changes=tuple(changes),
        )
        self._clear_tree_caches()
        self.refresh_selection_cache()
        self._record_mutation(result)
        return self

    @contextmanager
    def override(
        self,
        *predicates: NodePredicate | Callable[[Node], bool],
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> Iterator[MutationResult]:
        nodes = tuple(self.nodes(*predicates, include_root=include_root))
        snapshot = [
            (node, name, getattr(node, name, _MISSING), name in getattr(node, "model_fields_set", set()))
            for node in nodes
            for name in values
            if _has_model_attribute(node, name) or (allow_extra and _allows_extra_attributes(node))
        ]

        self._set_nodes(
            nodes,
            strict=strict,
            allow_extra=allow_extra,
            validate=validate,
            **values,
        )
        result = self.last_mutation
        assert result is not None

        try:
            yield result
        finally:
            restore_changes: list[MutationChange] = []
            for node, name, original, was_set in snapshot:
                current = getattr(node, name, _MISSING)
                if original is _MISSING:
                    if current is _MISSING:
                        continue
                    delattr(node, name)
                    restored = None
                else:
                    setattr(node, name, original)
                    if name in getattr(type(node), "model_fields", {}):
                        if was_set:
                            node.model_fields_set.add(name)
                        else:
                            node.model_fields_set.discard(name)
                    restored = original

                if current != original:
                    restore_changes.append(
                        MutationChange(
                            node=str(node.address),
                            field=name,
                            old=None if current is _MISSING else current,
                            new=restored,
                            action="restore",
                        )
                    )

            restore_result = MutationResult(
                operation_id=uuid.uuid4().hex,
                parent_operation_id=result.operation_id,
                action="restore",
                matched=len(nodes),
                updated=len({change.node for change in restore_changes}),
                changes=tuple(restore_changes),
            )
            self._clear_tree_caches()
            self.refresh_selection_cache()
            self._record_mutation(restore_result)

    def resolved_dropout(self, address: Address | str) -> float:
        return self._resolved_rate(address, "dropout")

    def _node_at(self, address: Address | str) -> Node:
        key = Address(str(address))
        if key in self.arrays:
            return self.arrays[key]

        if key in self.requests:
            return self.requests[key]

        raise ValueError(f"address '{address}' not found in hyperparameters")

    def _resolved_rate(self, address: Address | str, name: Literal["dropout", "p_mask", "p_prune"]) -> float:
        node: Node | None = self._node_at(address)

        while node is not None:
            rate = getattr(node, name, None)
            if rate is not None:
                return float(rate)
            node = getattr(node, "parent", None)

        return 0.0

    def resolved_p_mask(self, address: Address | str) -> float:
        return self._resolved_rate(address, "p_mask")

    def resolved_p_prune(self, address: Address | str) -> float:
        return self._resolved_rate(address, "p_prune")

    def __str__(self) -> str:
        lines: list[str] = []
        for pre, _, node in RenderTree(self):
            lines.append(f"{pre}{node.name} ({node.type})")

        return "\n".join(lines)


def schema(
    *field_args: SchemaField,
    d_model: int,
    n_layers: int,
    n_heads: int,
    fields: Sequence[SchemaField] | None = None,
    root: str = "record",
) -> Hyperparameters:
    return Hyperparameters.schema(
        *field_args,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        fields=fields,
        root=root,
    )
