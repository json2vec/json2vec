"""Model-facing schema mutation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partialmethod, wraps
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import torch
from lightning.pytorch import Callback

from relflow.architecture.graph import ModelGraph
from relflow.structs.enums import Strata
from relflow.structs.experiment import NodeAttribute, NodePredicate, SchemaField
from relflow.structs.tree import Node

if TYPE_CHECKING:
    from relflow.architecture.root import Model


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

    def _on_loop_start(self, trainer: lit.Trainer, pl_module: "Model", strata: Strata) -> None:
        pl_module.locks[strata] += 1

    def _on_loop_end(self, trainer: lit.Trainer, pl_module: "Model", strata: Strata) -> None:
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

    on_train_start = partialmethod(_on_loop_start, strata=Strata.train)
    on_train_end = partialmethod(_on_loop_end, strata=Strata.train)
    on_validation_start = partialmethod(_on_loop_start, strata=Strata.validate)
    on_validation_end = partialmethod(_on_loop_end, strata=Strata.validate)
    on_test_start = partialmethod(_on_loop_start, strata=Strata.test)
    on_test_end = partialmethod(_on_loop_end, strata=Strata.test)
    on_predict_start = partialmethod(_on_loop_start, strata=Strata.predict)
    on_predict_end = partialmethod(_on_loop_end, strata=Strata.predict)


class RuntimePlacementCallback(Callback):
    """Move late-created modules onto the Lightning module's active device."""

    def _on_loop_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        device = getattr(pl_module, "device", None)
        if isinstance(device, torch.device):
            pl_module.to(device=device)

    on_train_start = partialmethod(_on_loop_start, strata=Strata.train)
    on_validation_start = partialmethod(_on_loop_start, strata=Strata.validate)
    on_test_start = partialmethod(_on_loop_start, strata=Strata.test)
    on_predict_start = partialmethod(_on_loop_start, strata=Strata.predict)


class SchemaEditor:
    """Coordinate schema mutations with runtime graph rebuilds."""

    def __init__(self, module: "Model") -> None:
        self.module = module

    def _assert_mutation_allowed(self, action: str) -> None:
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
        self._assert_mutation_allowed("update")
        values = self.module.schema.update_values(values)
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
        self.module._reset_contracts()

    def extend(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        self._assert_mutation_allowed("extend")
        self.module.schema.extend(*args, include_root=include_root, use_cache=use_cache)
        ModelGraph.rebuild(self.module)
        self.module._reset_contracts()

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        self._assert_mutation_allowed("delete")
        self.module.schema.delete(*predicates, include_root=include_root, use_cache=use_cache)
        ModelGraph.rebuild(self.module)
        self.module._reset_contracts()

    def reset(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
        descendants: bool = False,
    ) -> None:
        self._assert_mutation_allowed("reset")
        selected = self.module.schema.select(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
        )
        if not selected:
            raise ValueError("reset matched no nodes")

        ModelGraph.reset_selected(self.module, selected, descendants=descendants)
        self.module._reset_contracts()

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
        self._assert_mutation_allowed("override")
        values = self.module.schema.update_values(values)
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
                ModelGraph.rebuild(self.module)
                self.module._reset_contracts()
                yield
        finally:
            ModelGraph.rebuild(self.module)
            self.module._reset_contracts()
