"""Serializable schema, node predicates, and mutation helpers."""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

import pydantic
from anytree import LevelOrderGroupIter, PreOrderIter
from rich.text import Text

from relflow.structs.enums import AttentionMode, Component, Overflow, Strata
from relflow.structs.selectors import (
    ExtendArg,
    NodeAttribute,
    NodePredicate,
    NodeSelector,
    SchemaField,
    SelectionCacheEntry,
    SelectionKey,
    _has_model_attribute,
    predicate,
    where,
)
from relflow.structs.structure import Branch, Mask, RequestTypes
from relflow.structs.tree import Address, Leaf, MaskInput, Node, Rate, Selection

__all__ = [
    "ExtendArg",
    "Schema",
    "NodeAttribute",
    "NodePredicate",
    "NodeSelector",
    "SchemaField",
    "SelectionCacheEntry",
    "SelectionKey",
    "TreeFieldInput",
    "predicate",
    "where",
]

TreeFieldInput: TypeAlias = SchemaField | type[Leaf]

_MISSING = object()


def bind_tree_field(source: str | None, value: TreeFieldInput) -> SchemaField:
    """Materialize and bind a tree field from a keyword source name."""
    if isinstance(value, type) and issubclass(value, Leaf):
        value = value()

    if not isinstance(value, (Branch, Leaf)):
        label = f" '{source}'" if source is not None else ""
        raise TypeError(f"tree field{label} must be a Branch, Leaf, or Leaf class")

    if source is None:
        if value.name is None:
            raise ValueError("tree field is unnamed; pass it as a keyword or provide a name")
        return value

    if value.name is None:
        value.name = source
        value.model_fields_set.add("name")
        value.check_node_name()
        return value

    if value.name != source:
        raise ValueError(f"tree keyword '{source}' cannot bind field named '{value.name}'")

    return value


class Schema(Node):
    """Serializable schema and training metadata used to build a `Model`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: Literal["schema"] = pydantic.Field(default="schema", exclude=True)
    type: Literal["schema"] = pydantic.Field(default="schema", exclude=True)
    description: Literal[None] = pydantic.Field(default=None, exclude=True)
    d_model: Annotated[int, pydantic.Field(gt=0, default=128)]
    fields: Branch

    embed: ClassVar[None] = None
    dropout: ClassVar[None] = None  # ty:ignore[invalid-attribute-override]

    _selection_cache: dict[SelectionKey, SelectionCacheEntry] = pydantic.PrivateAttr(default_factory=dict)

    @classmethod
    def update_values(cls, values: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(values)
        removed = sorted({"masks", "p_mask", "p_prune", "target"} & normalized.keys())
        if removed:
            raise ValueError(f"removed node field(s): {removed}; use mask")
        return normalized

    @pydantic.model_validator(mode="before")
    @classmethod
    def restore_masks(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        values = dict(data)

        def restore_policy(value: Mapping[str, Any]) -> Mask:
            return Mask.model_validate(dict(value))

        def restore(node: Any) -> Any:
            if not isinstance(node, Mapping):
                return node
            payload = dict(node)
            if "mask" in payload:
                value = payload["mask"]
                if isinstance(value, Mapping):
                    payload["mask"] = restore_policy(value)
                elif isinstance(value, (list, tuple)):
                    payload["mask"] = tuple(
                        restore_policy(policy) if isinstance(policy, Mapping) else policy for policy in value
                    )
            if isinstance(payload.get("fields"), (list, tuple)):
                payload["fields"] = [restore(field) for field in payload["fields"]]
            return payload

        if "fields" in values:
            values["fields"] = restore(values["fields"])
        return values

    @classmethod
    def request_from_leaf(cls, leaf: Leaf) -> RequestTypes:
        from relflow.tensorfields.base import TENSORFIELDS

        request_cls = getattr(TENSORFIELDS[leaf.type], "Request")
        payload = leaf.model_dump(mode="python", round_trip=True, exclude={"mask"})
        return request_cls.model_validate({**payload, "mask": leaf.mask})

    @classmethod
    def from_tree_node(cls, node: SchemaField) -> Branch | RequestTypes:
        if isinstance(node, Leaf):
            if node.name is None:
                raise ValueError("tree field is unnamed; pass it as a keyword or provide a name")
            source = node.name
            node_name = Node.sanitize_name(source)
            updates: dict[str, Any] = {}

            if node_name != source:
                updates["name"] = node_name
                if node.description is None:
                    updates["description"] = source

            return cls.request_from_leaf(node.model_copy(update=updates))

        if isinstance(node, Branch):
            if node.name is None:
                raise ValueError("tree field is unnamed; pass it as a keyword or provide a name")
            fields = [cls.from_tree_node(field) for field in node.fields]
            payload = node.model_dump(mode="python", round_trip=True, exclude={"fields", "mask"})
            return Branch(*fields, mask=node.mask, **payload)

        raise TypeError("tree fields must be Branch, Leaf, or concrete request instances")

    @classmethod
    def from_tree(
        cls,
        *field_args: TreeFieldInput,
        d_model: int,
        n_layers: int,
        n_heads: int,
        fields: Sequence[TreeFieldInput] | None = None,
        name: str = "record",
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        n_linear: Annotated[int, pydantic.Field(gt=0)] = 1,
        dropout: Rate | None = None,
        mask: MaskInput = False,
        **field_kwargs: TreeFieldInput,
    ) -> Self:
        """Build schema from tree fields."""
        normalized = [
            *(bind_tree_field(None, field) for field in (fields or ())),
            *(bind_tree_field(None, field) for field in field_args),
            *(bind_tree_field(source, field) for source, field in field_kwargs.items()),
        ]
        if not normalized:
            raise ValueError("from_tree requires at least one field")

        seen_sources: set[str] = set()
        root_fields: list[Branch | RequestTypes] = []

        for field in normalized:
            if not isinstance(field, (Branch, Leaf)):
                raise TypeError("tree fields must be Branch, Leaf, or concrete request instances")
            if field.name is None:
                raise ValueError("tree field is unnamed; pass it as a keyword or provide a name")

            source = field.name
            if source in seen_sources:
                raise ValueError(f"duplicate schema source field: {source}")
            seen_sources.add(source)

            root_fields.append(cls.from_tree_node(field))

        branch = Branch(
            name=name,
            query=query,
            description=description,
            embed=embed,
            attention=attention,
            n_layers=n_layers,
            n_heads=n_heads,
            n_linear=n_linear,
            length=1,
            overflow=Overflow.error,
            dropout=dropout,
            mask=mask,
            fields=root_fields,
        )
        return cls(d_model=d_model, fields=branch)

    def model_post_init(self, __context):
        def materialize(branch: Branch) -> Branch:
            fields: list[Branch | RequestTypes] = []
            for field in list(branch.fields):
                field.parent = None

                if isinstance(field, Branch):
                    fields.append(materialize(field))
                elif type(field) is Leaf:
                    fields.append(self.request_from_leaf(field))
                else:
                    fields.append(field)

            branch.fields = fields
            for field in branch.fields:
                field.parent = branch

            return branch

        self.fields = materialize(self.fields)
        self.fields.length = 1
        self.fields.overflow = Overflow.error
        self.fields.parent: Self = self
        self._post_bind_validate()

    @property
    def reconstruct(self) -> list[Address]:
        role = NodePredicate(
            func=lambda node: (
                isinstance(node, Leaf)
                and node.active
                and any(policy.reconstruct for owner in node.path for policy in getattr(owner, "mask", ()))
            ),
            key=("role", "reconstruct"),
        )
        return [Address(str(node.address)) for node in self.select(role)]

    @property
    def objectives(self) -> list[Address]:
        return self.reconstruct

    @property
    def decodes(self) -> list[Address]:
        role = NodePredicate(
            func=lambda node: (
                isinstance(node, Leaf)
                and node.active
                and any(
                    policy.reconstruct and policy.rate is None
                    for owner in node.path
                    for policy in getattr(owner, "mask", ())
                )
            ),
            key=("role", "decodes"),
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
    def branches(self) -> dict[Address, Branch]:
        return {node.address: node for node in self.descendants if isinstance(node, Branch)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if isinstance(node, Leaf)}

    @functools.cached_property
    def active_requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.requests.values() if node.active}

    @functools.cached_property
    def shapes(self) -> dict[Address, tuple[int, ...]]:
        return {request.address: request.shape for request in self.requests.values()}

    def overflows(self, address: Address) -> tuple[Overflow, ...]:
        return (Overflow.error, *self.requests[Address(str(address))].overflows)

    def masks_for(self, address: Address) -> tuple[tuple[Address, Mask], ...]:
        request = self.requests[Address(str(address))]
        return tuple(
            (node.address, policy) for node in request.path if isinstance(node, (Branch, Leaf)) for policy in node.mask
        )

    def forward_for(self, strata: Strata | str) -> list[Address]:
        selected = self.decodes if Strata.normalize(strata) is Strata.predict else self.objectives
        addresses = {*selected, *self.embed}
        return [node.address for node in PreOrderIter(self.fields) if node.address in addresses]

    @functools.cached_property
    def depthwise(self) -> list[list[Address]]:
        out: list[list[Address]] = []
        for depth in LevelOrderGroupIter(self.fields):
            branches = [node.address for node in depth if isinstance(node, Branch)]
            if branches:
                out.append(branches)

        return out

    def _clear_tree_caches(self) -> None:
        for name in ("branches", "requests", "active_requests", "shapes", "depthwise"):
            self.__dict__.pop(name, None)

        for node in PreOrderIter(self.fields):
            for name in ("address", "heritage", "shape", "overflows"):
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

    def _post_bind_validate(self) -> None:
        for branch in self.branches.values():
            branch.post_bind_validate()

        for request in self.requests.values():
            request.post_bind_validate()

        self.validate_capabilities()

    def validate_capabilities(self) -> None:
        """Validate effective leaf roles against registered extension components."""

        from relflow.tensorfields.base import TENSORFIELDS

        for address, request in self.active_requests.items():
            owners = [
                owner.address
                for owner in request.path
                if any(policy.reconstruct for policy in getattr(owner, "mask", ()))
            ]
            required: list[Component] = []
            if owners:
                required.extend((Component.Decoder, Component.loss))
            elif request.embed:
                required.append(Component.Decoder)

            extension = TENSORFIELDS[request.type]
            missing = [component.value for component in required if component not in extension.components]
            if not missing:
                continue

            names = ", ".join(missing)
            if owners:
                sources = ", ".join(repr(str(owner)) for owner in owners)
                raise ValueError(
                    f"reconstruction mask at {sources} reaches leaf '{address}', but extension "
                    f"'{extension.name}' is missing required component(s): {names}; register both Decoder and loss"
                )
            raise ValueError(
                f"embedded leaf '{address}' uses extension '{extension.name}', which is missing required "
                f"component: {names}; register a Decoder or set embed=False"
            )

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
            return Selection(self._selection_cache[key].nodes)

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

        return Selection(nodes)

    def update(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: Any,
    ) -> None:
        """Mutate matching schema nodes."""
        values = self.update_values(values)
        if not values:
            raise ValueError("update requires at least one field value")

        nodes = self.select(*predicates, include_root=include_root, use_cache=use_cache)
        snapshots: list[tuple[Node, str, Any, bool]] = []
        try:
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
                    payload = node.model_dump(mode="python", round_trip=True, exclude={"mask"})
                    payload["mask"] = node.mask
                    payload.update(applicable_values)
                    validated = type(node).model_validate(payload)
                    applicable_values = {name: getattr(validated, name) for name in applicable_values}

                for name, value in applicable_values.items():
                    snapshots.append(
                        (
                            node,
                            name,
                            getattr(node, name, _MISSING),
                            name in getattr(node, "model_fields_set", set()),
                        )
                    )
                    setattr(node, name, value)
                    if name in getattr(type(node), "model_fields", {}):
                        node.model_fields_set.add(name)

            self._clear_tree_caches()
            self._post_bind_validate()
        except Exception:
            for node, name, original, was_set in reversed(snapshots):
                if original is _MISSING:
                    if hasattr(node, name):
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
            raise

        self.refresh_selection_cache()

    def extend(
        self,
        *args: ExtendArg,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        """Append new schema fields under the single branch selected by predicates."""
        predicates: list[NodeSelector] = []
        fields: list[SchemaField] = []
        reading_fields = False

        for item in args:
            if isinstance(item, (Branch, Leaf)):
                reading_fields = True
                fields.append(item)
                continue

            if reading_fields:
                raise TypeError("extend predicates must come before new tree fields")

            predicates.append(item)

        if not fields:
            raise ValueError("extend requires at least one schema field")

        candidates = [
            node
            for node in self.select(*predicates, include_root=include_root, use_cache=use_cache)
            if isinstance(node, Branch)
        ]

        if len(candidates) != 1:
            raise ValueError(f"extend requires exactly one matching branch node, found {len(candidates)}")

        parent = candidates[0]
        new_fields = [self.from_tree_node(field) for field in fields]
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
            self._post_bind_validate()
        except Exception:
            parent.fields = original_fields
            for field in new_fields:
                field.parent = None
            self._clear_tree_caches()
            self._post_bind_validate()
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
            raise ValueError("delete cannot remove the root branch")

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

        remaining_branch_addresses = {address for address in self.branches if address not in removed_addresses}
        for address in remaining_branch_addresses:
            prefix = f"{address}/"
            if not any(str(request_address).startswith(prefix) for request_address in remaining_request_addresses):
                raise ValueError(f"delete would leave branch '{address}' without request descendants")

        for node in roots:
            parent = node.parent
            if not isinstance(parent, Branch):
                raise ValueError(f"delete cannot remove '{node.address}' because it has no branch parent")
            parent.fields = [field for field in parent.fields if field is not node]
            node.parent = None

        self._clear_tree_caches()
        self._post_bind_validate()
        self.refresh_selection_cache()

    @contextmanager
    def override(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: Any,
    ) -> Iterator[None]:
        nodes = self.select(*predicates, include_root=include_root, use_cache=use_cache)
        normalized_values = self.update_values(values)
        snapshot = [
            (
                node,
                name,
                getattr(node, name, _MISSING),
                name in getattr(node, "model_fields_set", set()),
            )
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
            use_cache=use_cache,
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
            self._post_bind_validate()
            self.refresh_selection_cache()

    def __rich_console__(self, console, options):
        heading = Text()
        heading.append(self.name, style=self.RICH_NAME_STYLE)
        heading.append(" ")
        heading.append(f"[{self.type}]", style=self.RICH_TYPE_STYLE)
        for name, value in (
            ("d_model", self.d_model),
            ("branches", len(self.branches)),
            ("fields", len(self.active_requests)),
            ("reconstruct", len(self.reconstruct)),
            ("embeds", len(self.embed)),
        ):
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        yield heading

        lines = list(self.fields.__rich_console__(console, options))
        if not lines:
            return
        first = Text()
        first.append("`-- ", style=self.RICH_TREE_STYLE)
        if isinstance(lines[0], Text):
            first.append_text(lines[0])
        else:
            first.append(str(lines[0]))
        yield first
        for line in lines[1:]:
            nested = Text()
            nested.append("    ", style=self.RICH_TREE_STYLE)
            if isinstance(line, Text):
                nested.append_text(line)
            else:
                nested.append(str(line))
            yield nested
