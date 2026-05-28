"""Schema hyperparameters, node predicates, and mutation helpers."""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

import pydantic
from anytree import LevelOrderGroupIter, PreOrderIter, RenderTree

from json2vec.structs.enums import AttentionMode
from json2vec.structs.structure import Array, RequestTypes
from json2vec.structs.tree import Address, Leaf, Node, PruneRate, Rate

SelectionKey: TypeAlias = tuple[Any, ...]
SchemaField: TypeAlias = Array | Leaf
_MISSING = object()


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
    """Start a schema predicate against a node attribute.

    Example:
        ```python
        model.update(where("type") == "number", p_mask=0.10)
        model.update(where("name") == "label", target=True)
        ```
    """
    return NodeAttribute.named(name)


NodeSelector: TypeAlias = NodePredicate | NodeAttribute | Callable[[Node], bool]
ExtendArg: TypeAlias = NodeSelector | SchemaField


def _has_model_attribute(node: Node, name: str) -> bool:
    if name in _QUERYABLE_BUILTINS:
        return True

    fields = getattr(type(node), "model_fields", {})
    extra = getattr(node, "model_extra", None) or {}
    return name in fields or name in extra or hasattr(node, name)


class Hyperparameters(Node):
    """Serializable schema and training metadata used to build a `Model`."""

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

    @classmethod
    def update_values(cls, values: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(values)
        target = normalized.pop("target", None)

        if target is None:
            return normalized

        if not isinstance(target, bool):
            raise ValueError("target must be a boolean")

        if target:
            if normalized.get("p_prune") not in (None, 1.0):
                raise ValueError("target=True is shorthand for p_prune=1.0")
            normalized["p_prune"] = 1.0
        else:
            if normalized.get("p_prune") is not None:
                raise ValueError("target=False is shorthand for p_prune=None")
            normalized["p_prune"] = None

        return normalized

    @classmethod
    def jmespath_member(cls, value: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            return value
        return json.dumps(value)

    @classmethod
    def query_for_source(cls, array_path: tuple[str, ...], source: str) -> str:
        """Infer a request-level query for a leaf source field.

        The encoder prepends the outer batch selector during search. Inferred
        queries therefore start at the processed-observation level: `[*].amount`,
        not `[*][*].amount`.
        """
        selectors = "".join(f".{cls.jmespath_member(array)}[*]" for array in array_path)
        return f"[*]{selectors}.{cls.jmespath_member(source)}"

    @classmethod
    def request_from_leaf(cls, leaf: Leaf) -> RequestTypes:
        from json2vec.tensorfields.base import TENSORFIELDS

        request_cls = getattr(TENSORFIELDS[leaf.type], "Request")
        return request_cls.model_validate(leaf.model_dump(mode="python", round_trip=True))

    @classmethod
    def from_schema_node(cls, node: SchemaField, *, array_path: tuple[str, ...] = ()) -> Array | RequestTypes:
        if isinstance(node, Leaf):
            source = node.name
            node_name = Node.sanitize_name(source)
            updates: dict[str, Any] = {}

            if node_name != source:
                updates["name"] = node_name
                if node.description is None:
                    updates["description"] = source

            if node.query is None:
                updates["query"] = cls.query_for_source(array_path, source)

            return cls.request_from_leaf(node.model_copy(update=updates))

        if isinstance(node, Array):
            child_path = (*array_path, node.name)
            fields = [cls.from_schema_node(field, array_path=child_path) for field in node.fields]
            payload = node.model_dump(mode="python", round_trip=True, exclude={"fields"})
            return Array(*fields, **payload)

        raise TypeError("schema fields must be Array, Leaf, or concrete request instances")

    @classmethod
    def from_schema(
        cls,
        *field_args: SchemaField,
        d_model: int,
        n_layers: int,
        n_heads: int,
        fields: Sequence[SchemaField] | None = None,
        root: str = "record",
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        max_length: Annotated[int, pydantic.Field(gt=0)] = 1,
        n_outputs: Annotated[int, pydantic.Field(gt=0)] = 1,
        n_linear: Annotated[int, pydantic.Field(gt=0)] = 1,
        dropout: Rate | None = None,
        p_mask: Rate | None = None,
        p_prune: PruneRate | None = None,
    ) -> Self:
        """Build hyperparameters from schema fields."""
        normalized = [*(fields or ()), *field_args]
        if not normalized:
            raise ValueError("from_schema requires at least one field")

        seen_sources: set[str] = set()
        root_fields: list[Array | RequestTypes] = []

        for field in normalized:
            if not isinstance(field, (Array, Leaf)):
                raise TypeError("schema fields must be Array, Leaf, or concrete request instances")

            source = field.name
            if source in seen_sources:
                raise ValueError(f"duplicate schema source field: {source}")
            seen_sources.add(source)

            root_fields.append(cls.from_schema_node(field))

        array = Array(
            name=root,
            description=description,
            embed=embed,
            attention=attention,
            n_layers=n_layers,
            n_heads=n_heads,
            n_outputs=n_outputs,
            n_linear=n_linear,
            max_length=max_length,
            dropout=dropout,
            p_mask=p_mask,
            p_prune=p_prune,
            fields=root_fields,
        )
        return cls(d_model=d_model, fields=array)

    def model_post_init(self, __context):
        def materialize(array: Array) -> Array:
            fields: list[Array | RequestTypes] = []
            for field in list(array.fields):
                field.parent = None

                if isinstance(field, Array):
                    fields.append(materialize(field))
                elif type(field) is Leaf:
                    fields.append(self.request_from_leaf(field))
                else:
                    fields.append(field)

            array.fields = fields
            for field in array.fields:
                field.parent = array

            return array

        self.fields = materialize(self.fields)
        self.fields.parent: Self = self
        for request in self.requests.values():
            request.post_bind_validate()

    @property
    def target(self) -> list[Address]:
        role = NodePredicate(
            func=lambda node: isinstance(node, Leaf) and node.active and getattr(node, "p_prune", 0.0) == 1.0,
            key=("role", "target"),
        )
        return [Address(str(node.address)) for node in self.select(role)]

    @property
    def embed(self) -> list[Address]:  # noqa: F811
        role = NodePredicate(
            func=lambda node: getattr(node, "embed", False) is True and (not isinstance(node, Leaf) or node.active),
            key=("role", "embed"),
        )
        return [Address(str(node.address)) for node in self.select(role)]

    @functools.cached_property
    def arrays(self) -> dict[Address, Array]:
        return {node.address: node for node in self.descendants if isinstance(node, Array)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if isinstance(node, Leaf)}

    @functools.cached_property
    def active_requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.requests.values() if node.active}

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

    def _clear_tree_caches(self) -> None:
        for name in ("arrays", "requests", "active_requests", "shapes", "depthwise"):
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
                        for node in PreOrderIter(self.fields)
                        if (entry.include_root or node is not self.fields)
                        if entry.predicate(node)
                    )
                }
            )
            for key, entry in self._selection_cache.items()
        }

    def select(
        self,
        *predicates: NodeSelector,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        if predicates:
            normalized = tuple(NodePredicate.from_selector(item) for item in predicates)
            combined = NodePredicate(
                func=lambda node: all(item(node) for item in normalized),
                key=("and", tuple(item.key for item in normalized)),
                cacheable=all(item.cacheable for item in normalized),
            )
        else:
            combined = NodePredicate(func=lambda node: True, key=("all",))

        key = ("select", include_root, combined.key)

        if use_cache and combined.cacheable and key in self._selection_cache:
            return list(self._selection_cache[key].nodes)

        nodes = tuple(
            node for node in PreOrderIter(self.fields) if (include_root or node is not self.fields) if combined(node)
        )

        if use_cache and combined.cacheable:
            self._selection_cache[key] = SelectionCacheEntry(
                key=key,
                predicate=combined,
                include_root=include_root,
                nodes=nodes,
            )

        return list(nodes)

    def update(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> None:
        """Mutate matching schema nodes.

        `target=True` is normalized to `p_prune=1.0`; `target=False` clears the
        target prune rate.
        """
        values = self.update_values(values)
        if not values:
            raise ValueError("update requires at least one field value")

        nodes = self.select(*predicates, include_root=include_root)
        for node in nodes:
            can_apply_extra = allow_extra and getattr(type(node), "model_config", {}).get("extra") == "allow"
            missing = [name for name in values if not _has_model_attribute(node, name) and not can_apply_extra]
            if missing and strict:
                label = str(node.address) or node.name
                raise AttributeError(f"{label} has no attribute(s): {missing}")

            applicable_values = {
                name: value for name, value in values.items() if _has_model_attribute(node, name) or can_apply_extra
            }

            if validate and applicable_values:
                payload = node.model_dump(mode="python", round_trip=True)
                payload.update(applicable_values)
                type(node).model_validate(payload)

            for name, value in applicable_values.items():
                setattr(node, name, value)
                if name in getattr(type(node), "model_fields", {}):
                    node.model_fields_set.add(name)

        self._clear_tree_caches()
        self.refresh_selection_cache()

    def extend(
        self,
        *args: ExtendArg,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        """Append new schema fields under the single array selected by predicates."""
        predicates: list[NodeSelector] = []
        fields: list[SchemaField] = []
        reading_fields = False

        for item in args:
            if isinstance(item, (Array, Leaf)):
                reading_fields = True
                fields.append(item)
                continue

            if reading_fields:
                raise TypeError("extend predicates must come before new schema fields")

            predicates.append(item)

        if not fields:
            raise ValueError("extend requires at least one schema field")

        candidates = [
            node
            for node in self.select(*predicates, include_root=include_root, use_cache=use_cache)
            if isinstance(node, Array)
        ]

        if len(candidates) != 1:
            raise ValueError(f"extend requires exactly one matching array node, found {len(candidates)}")

        parent = candidates[0]
        array_path = tuple(node.name for node in parent.path[2:] if isinstance(node, Array))
        new_fields = [self.from_schema_node(field, array_path=array_path) for field in fields]
        existing_names = {field.name for field in parent.fields}
        duplicate_names = sorted({field.name for field in new_fields if field.name in existing_names})
        duplicate_names.extend(
            sorted(
                {
                    field.name
                    for index, field in enumerate(new_fields)
                    if any(other.name == field.name for other in new_fields[index + 1 :])
                }
            )
        )
        if duplicate_names:
            raise ValueError(f"duplicate field name(s): {sorted(set(duplicate_names))}")

        original_fields = list(parent.fields)
        try:
            parent.fields.extend(new_fields)
            for field in new_fields:
                field.parent = parent

            self._clear_tree_caches()
            for field in new_fields:
                requests = (
                    [field]
                    if isinstance(field, Leaf)
                    else [
                        descendant for descendant in getattr(field, "descendants", ()) if isinstance(descendant, Leaf)
                    ]
                )
                for request in requests:
                    request.post_bind_validate()
        except Exception:
            parent.fields = original_fields
            for field in new_fields:
                field.parent = None
            self._clear_tree_caches()
            self.refresh_selection_cache()
            raise

        self.refresh_selection_cache()

    def delete(
        self,
        *predicates: NodeSelector,
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        """Permanently remove selected schema nodes from the tree."""
        if not predicates:
            raise ValueError("delete requires at least one predicate")

        selected = self.select(*predicates, include_root=include_root, use_cache=use_cache)
        if not selected:
            raise ValueError("delete matched no nodes")
        if self.fields in selected:
            raise ValueError("delete cannot remove the root array")

        selected_ids = {id(node) for node in selected}
        roots = [
            node
            for node in selected
            if not any(
                id(ancestor) in selected_ids for ancestor in getattr(node, "ancestors", ()) if ancestor is not self
            )
        ]
        removed_by_id = {id(node): node for node in roots}
        for node in roots:
            removed_by_id.update({id(descendant): descendant for descendant in getattr(node, "descendants", ())})
        removed_addresses = {node.address for node in removed_by_id.values()}

        remaining_request_addresses = {address for address in self.requests if address not in removed_addresses}
        if not remaining_request_addresses:
            raise ValueError("delete would remove every request")

        remaining_array_addresses = {address for address in self.arrays if address not in removed_addresses}
        for address in remaining_array_addresses:
            prefix = f"{address}/"
            if not any(str(request_address).startswith(prefix) for request_address in remaining_request_addresses):
                raise ValueError(f"delete would leave array '{address}' without request descendants")

        for node in roots:
            parent = node.parent
            if not isinstance(parent, Array):
                raise ValueError(f"delete cannot remove '{node.address}' because it has no array parent")
            parent.fields = [field for field in parent.fields if field is not node]
            node.parent = None

        self._clear_tree_caches()
        self.refresh_selection_cache()

    @contextmanager
    def override(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> Iterator[None]:
        nodes = self.select(*predicates, include_root=include_root)
        normalized_values = self.update_values(values)
        snapshot = [
            (node, name, getattr(node, name, _MISSING), name in getattr(node, "model_fields_set", set()))
            for node in nodes
            for name in normalized_values
            if _has_model_attribute(node, name)
            or (allow_extra and getattr(type(node), "model_config", {}).get("extra") == "allow")
        ]

        self.update(
            *predicates,
            strict=strict,
            allow_extra=allow_extra,
            include_root=include_root,
            validate=validate,
            **normalized_values,
        )

        try:
            yield
        finally:
            for node, name, original, was_set in snapshot:
                if original is _MISSING:
                    if getattr(node, name, _MISSING) is _MISSING:
                        continue
                    delattr(node, name)
                else:
                    setattr(node, name, original)
                    if name in getattr(type(node), "model_fields", {}):
                        if was_set:
                            node.model_fields_set.add(name)
                        else:
                            node.model_fields_set.discard(name)

            self._clear_tree_caches()
            self.refresh_selection_cache()

    def __str__(self) -> str:
        lines: list[str] = []
        for pre, _, node in RenderTree(self):
            lines.append(f"{pre}{node.name} ({node.type})")

        return "\n".join(lines)
