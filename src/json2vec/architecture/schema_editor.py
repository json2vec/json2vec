"""Model-facing schema mutation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from json2vec.architecture.graph import ModelGraph
from json2vec.structs.experiment import NodeAttribute, NodePredicate, SchemaField
from json2vec.structs.tree import Node

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


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

    def extend(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.module._assert_mutation_allowed("extend")
        self.module.hyperparameters.extend(*args, include_root=include_root, use_cache=use_cache)
        ModelGraph.rebuild(self.module)

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.module._assert_mutation_allowed("delete")
        self.module.hyperparameters.delete(*predicates, include_root=include_root, use_cache=use_cache)
        ModelGraph.rebuild(self.module)

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

        ModelGraph.reset_selected(self.module, selected, descendants=descendants)

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
                ModelGraph.rebuild(self.module)
                yield
        finally:
            ModelGraph.rebuild(self.module)
