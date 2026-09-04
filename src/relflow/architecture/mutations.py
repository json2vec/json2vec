"""Model-facing schema mutation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from functools import partialmethod, wraps
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import pydantic
import torch
from lightning.pytorch import Callback
from loguru import logger

from relflow.architecture.graph import ModelGraph
from relflow.structs.enums import Strata
from relflow.structs.experiment import NodeAttribute, NodePredicate, SchemaField
from relflow.structs.structure import Branch
from relflow.structs.tree import Leaf, Node

if TYPE_CHECKING:
    from relflow.architecture.root import Model

_MISSING = object()


def immutable(name: str | Strata) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            locks = self.locks
            locks[name] += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                if locks[name] <= 1:
                    locks.pop(name, None)
                else:
                    locks[name] -= 1

        return wrapped

    return decorator


class MutationLockCallback(Callback):
    """Prevent runtime schema mutations while Lightning owns an active loop."""

    locks: tuple[Strata, ...] = (Strata.train, Strata.validate, Strata.test, Strata.predict)

    def on_loop_start(self, trainer: lit.Trainer, pl_module: "Model", strata: Strata) -> None:
        if strata == Strata.predict:
            pl_module.output_plans.clear()
        pl_module.locks[strata] += 1

    def on_loop_end(self, trainer: lit.Trainer, pl_module: "Model", strata: Strata) -> None:
        locks = pl_module.locks
        if locks[strata] <= 1:
            locks.pop(strata, None)
        else:
            locks[strata] -= 1

    def on_exception(
        self,
        trainer: lit.Trainer,
        pl_module: "Model",
        exception: BaseException,
    ) -> None:  # ty:ignore[invalid-method-override]
        for lock in self.locks:
            pl_module.locks.pop(lock, None)

    on_train_start = partialmethod(on_loop_start, strata=Strata.train)
    on_train_end = partialmethod(on_loop_end, strata=Strata.train)
    on_validation_start = partialmethod(on_loop_start, strata=Strata.validate)
    on_validation_end = partialmethod(on_loop_end, strata=Strata.validate)
    on_test_start = partialmethod(on_loop_start, strata=Strata.test)
    on_test_end = partialmethod(on_loop_end, strata=Strata.test)
    on_predict_start = partialmethod(on_loop_start, strata=Strata.predict)
    on_predict_end = partialmethod(on_loop_end, strata=Strata.predict)


class RuntimePlacementCallback(Callback):
    """Move late-created modules onto the Lightning module's active device."""

    def on_loop_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        device = getattr(pl_module, "device", None)
        if isinstance(device, torch.device):
            pl_module.to(device=device)

    on_train_start = partialmethod(on_loop_start, strata=Strata.train)
    on_validation_start = partialmethod(on_loop_start, strata=Strata.validate)
    on_test_start = partialmethod(on_loop_start, strata=Strata.test)
    on_predict_start = partialmethod(on_loop_start, strata=Strata.predict)


class AttributeChange(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    node: Node
    name: str
    original: Any
    definition_attribute: bool
    address: str
    node_name: str
    node_type: str
    changed: Any = _MISSING
    changed_address: Any = _MISSING


class SchemaEditor:
    """Coordinate schema mutations with runtime graph rebuilds."""

    def __init__(self, module: "Model") -> None:
        self.module = module

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Restore the schema and runtime graph when a mutation cannot rebuild."""

        schema = deepcopy(self.module.schema)
        nodes = self.module.nodes
        example = self.module.example_input_array
        try:
            yield
        except Exception:
            self.module.schema = schema
            self.module.nodes = nodes
            self.module.example_input_array = example
            raise

    def assert_mutation_allowed(self, action: str) -> None:
        active = tuple(name for name, count in self.module.locks.items() if count > 0)
        if active:
            labels = ", ".join(active)
            raise RuntimeError(f"model.{action}(...) cannot run while the model is in an active loop: {labels}")

    def select(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        return self.module.schema.select(
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
        self.assert_mutation_allowed("update")
        values = self.module.schema.update_values(values)
        changes = self.attribute_changes(
            values=values,
            predicates=predicates,
            allow_extra=allow_extra,
            include_root=include_root,
            use_cache=use_cache,
        )
        with self.transaction():
            self.module.schema.update(
                *predicates,
                strict=strict,
                allow_extra=allow_extra,
                include_root=include_root,
                validate=validate,
                use_cache=use_cache,
                **values,
            )
            ModelGraph.rebuild(self.module)
        self.module.reset_contracts()
        self.log_attribute_changes("update", changes)

    def extend(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.assert_mutation_allowed("extend")
        parent, field_count = self.extend_target(*args, include_root=include_root, use_cache=use_cache)
        with self.transaction():
            self.module.schema.extend(*args, include_root=include_root, use_cache=use_cache)
            ModelGraph.rebuild(self.module)
        self.module.reset_contracts()
        for field in parent.fields[-field_count:]:
            self.log_node_mutation(
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
        self.assert_mutation_allowed("delete")
        roots = self.delete_roots(*predicates, include_root=include_root, use_cache=use_cache)
        with self.transaction():
            self.module.schema.delete(*predicates, include_root=include_root, use_cache=use_cache)
            ModelGraph.rebuild(self.module)
        self.module.reset_contracts()
        for node in roots:
            self.log_node_mutation(
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
        self.assert_mutation_allowed("reset")
        selected = self.module.schema.select(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
        )
        if not selected:
            raise ValueError("reset matched no nodes")

        nodes = self.runtime_reset_nodes(selected, descendants=descendants)
        ModelGraph.reset_selected(self.module, selected, descendants=descendants)
        self.module.reset_contracts()
        for node in nodes:
            self.log_node_mutation(
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
        self.assert_mutation_allowed("override")
        values = self.module.schema.update_values(values)
        changes = self.attribute_changes(
            values=values,
            predicates=predicates,
            allow_extra=allow_extra,
            include_root=include_root,
            use_cache=use_cache,
        )
        entered = False
        try:
            with self.module.schema.override(
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
                self.module.reset_contracts()
                self.log_attribute_changes("override", changes)
                yield
        finally:
            ModelGraph.rebuild(self.module)
            self.module.reset_contracts()
            if entered:
                self.log_attribute_changes("override_restore", changes, restored=True)

    def attribute_changes(
        self,
        *,
        values: dict[str, Any],
        predicates: tuple[NodePredicate | NodeAttribute | Callable[[Node], bool], ...],
        allow_extra: bool,
        include_root: bool,
        use_cache: bool,
    ) -> list[AttributeChange]:
        nodes = self.module.schema.select(*predicates, include_root=include_root, use_cache=use_cache)
        changes: list[AttributeChange] = []
        for node in nodes:
            can_apply_extra = allow_extra and getattr(type(node), "model_config", {}).get("extra") == "allow"
            for name in values:
                if not (has_node_attribute(node, name) or can_apply_extra):
                    continue

                changes.append(
                    AttributeChange(
                        node=node,
                        name=name,
                        original=getattr(node, name, _MISSING),
                        definition_attribute=is_definition_attribute(node, name),
                        address=str(node.address),
                        node_name=node.name,
                        node_type=node.type,
                    )
                )

        return changes

    def extend_target(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool,
        use_cache: bool,
    ) -> tuple[Branch, int]:
        predicates: list[NodePredicate | NodeAttribute | Callable[[Node], bool]] = []
        field_count = 0
        reading_fields = False

        for item in args:
            if isinstance(item, (Branch, Leaf)):
                reading_fields = True
                field_count += 1
                continue

            if reading_fields:
                raise TypeError("extend predicates must come before new tree fields")

            predicates.append(item)

        if field_count == 0:
            raise ValueError("extend requires at least one tree field")

        candidates = [
            node
            for node in self.module.schema.select(*predicates, include_root=include_root, use_cache=use_cache)
            if isinstance(node, Branch)
        ]
        if len(candidates) != 1:
            raise ValueError(f"extend requires exactly one matching branch node, found {len(candidates)}")

        return candidates[0], field_count

    def delete_roots(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool,
        use_cache: bool,
    ) -> list[Node]:
        selected = self.module.schema.select(*predicates, include_root=include_root, use_cache=use_cache)
        selected_ids = {id(node) for node in selected}
        return [
            node
            for node in selected
            if not any(
                id(ancestor) in selected_ids
                for ancestor in getattr(node, "ancestors", ())
                if ancestor is not self.module.schema
            )
        ]

    def runtime_reset_nodes(self, selected: list[Node], *, descendants: bool) -> list[Node]:
        nodes: dict[str, Node] = {}
        for node in selected:
            if node.address in self.module.nodes:
                nodes[str(node.address)] = node

            if descendants:
                for descendant in getattr(node, "descendants", ()):
                    if descendant.address in self.module.nodes:
                        nodes[str(descendant.address)] = descendant

        return list(nodes.values())

    def log_attribute_changes(self, action: str, changes: list[AttributeChange], *, restored: bool = False) -> None:
        for change in changes:
            current_address = str(change.node.address)
            value = change.original if restored else getattr(change.node, change.name, _MISSING)
            if not restored:
                change.changed = value
                change.changed_address = current_address
            previous_value = change.changed if restored else change.original
            previous_address = change.changed_address if restored else change.address
            if previous_address is _MISSING:
                previous_address = change.address
            address_context = (
                current_address if previous_address == current_address else f"{previous_address} -> {current_address}"
            )
            value_text = format_log_value(value)
            previous_value_text = format_log_value(previous_value)
            logger.bind(
                component="schema_mutation",
                action=action,
                address=current_address,
                previous_address=previous_address,
                node_name=change.node.name,
                previous_node_name=change.node_name,
                node_type=change.node_type,
                attribute=change.name,
                definition_attribute=change.definition_attribute,
                value=value_text,
                previous_value=previous_value_text,
                change=f"{change.name}: {previous_value_text} -> {value_text}",
            ).info(
                "{} {}: {} {} -> {}",
                "restored" if restored else "mutated",
                address_context,
                change.name,
                previous_value_text,
                value_text,
            )

    def log_node_mutation(self, *, action: str, message: str, node: Node, **kwargs: Any) -> None:
        extra = {key: str(value.address) if isinstance(value, Node) else value for key, value in kwargs.items()}
        context = format_node_log_context(node, extra)
        logger.bind(
            component="schema_mutation",
            action=action,
            address=str(node.address),
            node_type=node.type,
            node_name=node.name,
            attribute=None,
            definition_attribute=None,
            **extra,
        ).info("{} {}", message, context)


def has_node_attribute(node: Node, name: str) -> bool:
    fields = getattr(type(node), "model_fields", {})
    extra = getattr(node, "model_extra", None) or {}
    return name in fields or name in extra or hasattr(node, name)


def is_definition_attribute(node: Node, name: str) -> bool:
    return name in getattr(type(node), "model_fields", {})


def format_log_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"

    text = repr(value)
    return text if len(text) <= 160 else f"{text[:157]}..."


def format_node_log_context(node: Node, extra: dict[str, Any]) -> str:
    parts = [str(node.address)]
    if parent := extra.get("parent"):
        parts.append(f"under {parent}")
    if "descendants" in extra:
        parts.append(f"descendants={extra['descendants']}")

    return " ".join(parts)
