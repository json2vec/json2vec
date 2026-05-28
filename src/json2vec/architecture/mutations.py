"""Model-facing schema mutation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from json2vec.architecture.graph import ModelGraph
from json2vec.structs.experiment import NodeAttribute, NodePredicate, SchemaField
from json2vec.structs.structure import Array
from json2vec.structs.tree import Leaf, Node

if TYPE_CHECKING:
    from json2vec.architecture.root import Model

_MISSING = object()


@dataclass(frozen=True)
class AttributeChange:
    node: Node
    name: str
    original: Any
    definition_attribute: bool


class SchemaEditor:
    """Coordinate schema mutations with runtime graph rebuilds."""

    def __init__(self, module: "Model") -> None:
        self.module = module

    def select(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        return self.module.hyperparameters.select(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
        )

    def update(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: Any,
    ) -> None:
        self.module._assert_mutation_allowed("update")
        values = self.module.hyperparameters.update_values(values)
        changes = self._attribute_changes(
            values=values,
            predicates=predicates,
            allow_extra=allow_extra,
            include_root=include_root,
            use_cache=use_cache,
        )
        self.module.hyperparameters.update(
            *predicates,
            strict=strict,
            allow_extra=allow_extra,
            include_root=include_root,
            validate=validate,
            use_cache=use_cache,
            **values,
        )
        ModelGraph.rebuild(self.module)
        self.module._reset_contracts()
        self._log_attribute_changes("update", changes)

    def extend(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.module._assert_mutation_allowed("extend")
        parent, field_count = self._extend_target(*args, include_root=include_root, use_cache=use_cache)
        self.module.hyperparameters.extend(*args, include_root=include_root, use_cache=use_cache)
        ModelGraph.rebuild(self.module)
        self.module._reset_contracts()
        for field in parent.fields[-field_count:]:
            self._log_node_mutation(
                action="extend",
                message="extended schema node",
                node=field,
                parent=parent,
            )

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.module._assert_mutation_allowed("delete")
        roots = self._delete_roots(*predicates, include_root=include_root, use_cache=use_cache)
        self.module.hyperparameters.delete(*predicates, include_root=include_root, use_cache=use_cache)
        ModelGraph.rebuild(self.module)
        self.module._reset_contracts()
        for node in roots:
            self._log_node_mutation(
                action="delete",
                message="deleted schema node",
                node=node,
                descendants=len(getattr(node, "descendants", ())),
            )

    def reset(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
        descendants: bool = False,
    ) -> None:
        self.module._assert_mutation_allowed("reset")
        selected = self.module.hyperparameters.select(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
        )
        if not selected:
            raise ValueError("reset matched no nodes")

        nodes = self._runtime_reset_nodes(selected, descendants=descendants)
        ModelGraph.reset_selected(self.module, selected, descendants=descendants)
        self.module._reset_contracts()
        for node in nodes:
            self._log_node_mutation(
                action="reset",
                message="reset runtime node",
                node=node,
                descendants=descendants,
            )

    @contextmanager
    def override(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: Any,
    ) -> Iterator[None]:
        self.module._assert_mutation_allowed("override")
        values = self.module.hyperparameters.update_values(values)
        changes = self._attribute_changes(
            values=values,
            predicates=predicates,
            allow_extra=allow_extra,
            include_root=include_root,
            use_cache=use_cache,
        )
        entered = False
        try:
            with self.module.hyperparameters.override(
                *predicates,
                strict=strict,
                allow_extra=allow_extra,
                include_root=include_root,
                validate=validate,
                use_cache=use_cache,
                **values,
            ):
                entered = True
                ModelGraph.rebuild(self.module)
                self.module._reset_contracts()
                self._log_attribute_changes("override", changes)
                yield
        finally:
            ModelGraph.rebuild(self.module)
            self.module._reset_contracts()
            if entered:
                self._log_attribute_changes("override_restore", changes, restored=True)

    def _attribute_changes(
        self,
        *,
        values: dict[str, Any],
        predicates: tuple[NodePredicate | NodeAttribute | Callable[[Node], bool], ...],
        allow_extra: bool,
        include_root: bool,
        use_cache: bool,
    ) -> list[AttributeChange]:
        nodes = self.module.hyperparameters.select(*predicates, include_root=include_root, use_cache=use_cache)
        changes: list[AttributeChange] = []
        for node in nodes:
            can_apply_extra = allow_extra and getattr(type(node), "model_config", {}).get("extra") == "allow"
            for name in values:
                if not (_has_node_attribute(node, name) or can_apply_extra):
                    continue

                changes.append(
                    AttributeChange(
                        node=node,
                        name=name,
                        original=getattr(node, name, _MISSING),
                        definition_attribute=_is_definition_attribute(node, name),
                    )
                )

        return changes

    def _extend_target(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool,
        use_cache: bool,
    ) -> tuple[Array, int]:
        predicates: list[NodePredicate | NodeAttribute | Callable[[Node], bool]] = []
        field_count = 0
        reading_fields = False

        for item in args:
            if isinstance(item, (Array, Leaf)):
                reading_fields = True
                field_count += 1
                continue

            if reading_fields:
                raise TypeError("extend predicates must come before new schema fields")

            predicates.append(item)

        if field_count == 0:
            raise ValueError("extend requires at least one schema field")

        candidates = [
            node
            for node in self.module.hyperparameters.select(*predicates, include_root=include_root, use_cache=use_cache)
            if isinstance(node, Array)
        ]
        if len(candidates) != 1:
            raise ValueError(f"extend requires exactly one matching array node, found {len(candidates)}")

        return candidates[0], field_count

    def _delete_roots(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool,
        use_cache: bool,
    ) -> list[Node]:
        selected = self.module.hyperparameters.select(*predicates, include_root=include_root, use_cache=use_cache)
        selected_ids = {id(node) for node in selected}
        return [
            node
            for node in selected
            if not any(
                id(ancestor) in selected_ids
                for ancestor in getattr(node, "ancestors", ())
                if ancestor is not self.module.hyperparameters
            )
        ]

    def _runtime_reset_nodes(self, selected: list[Node], *, descendants: bool) -> list[Node]:
        nodes: dict[str, Node] = {}
        for node in selected:
            if node.address in self.module.nodes:
                nodes[str(node.address)] = node

            if descendants:
                for descendant in getattr(node, "descendants", ()):
                    if descendant.address in self.module.nodes:
                        nodes[str(descendant.address)] = descendant

        return list(nodes.values())

    def _log_attribute_changes(self, action: str, changes: list[AttributeChange], *, restored: bool = False) -> None:
        for change in changes:
            value = change.original if restored else getattr(change.node, change.name, _MISSING)
            logger.bind(
                component="schema_mutation",
                action=action,
                address=str(change.node.address),
                node_type=change.node.type,
                attribute=change.name,
                definition_attribute=change.definition_attribute,
                value=_format_log_value(value),
                previous_value=_format_log_value(change.original),
            ).info("restored schema node attribute" if restored else "mutated schema node attribute")

    def _log_node_mutation(self, *, action: str, message: str, node: Node, **kwargs: Any) -> None:
        extra = {key: str(value.address) if isinstance(value, Node) else value for key, value in kwargs.items()}
        logger.bind(
            component="schema_mutation",
            action=action,
            address=str(node.address),
            node_type=node.type,
            attribute=None,
            definition_attribute=None,
            **extra,
        ).info(message)


def _has_node_attribute(node: Node, name: str) -> bool:
    fields = getattr(type(node), "model_fields", {})
    extra = getattr(node, "model_extra", None) or {}
    return name in fields or name in extra or hasattr(node, name)


def _is_definition_attribute(node: Node, name: str) -> bool:
    return name in getattr(type(node), "model_fields", {})


def _format_log_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"

    text = repr(value)
    return text if len(text) <= 160 else f"{text[:157]}..."
