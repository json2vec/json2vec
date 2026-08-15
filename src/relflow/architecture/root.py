"""Public Lightning model facade for `relflow` schemas."""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partialmethod
from pathlib import Path
from typing import Any, Self, cast

import lightning.pytorch as lit
import torch
from beartype import beartype
from lightning.pytorch import Callback
from loguru import logger
from rich.text import Text
from tensordict import TensorDict
from torchmetrics import Metric as TorchMetric

from relflow.architecture.checkpoint import CheckpointState, RollbackCheckpoint
from relflow.architecture.contracts import ContractScheduler
from relflow.architecture.graph import ModelGraph
from relflow.architecture.mutations import (
    MutationLockCallback,
    RuntimePlacementCallback,
    SchemaEditor,
    immutable,
)
from relflow.architecture.runtime import ModelRuntime, Postprocessor, Preprocessor, step
from relflow.data.datasets.base import EncodedBatch, EncodedInput
from relflow.logging.throughput import ThroughputLogger
from relflow.structs.enums import AttentionMode, Strata
from relflow.structs.experiment import (
    NodeAttribute,
    NodePredicate,
    Schema,
    SchemaField,
    TreeFieldInput,
)
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address, Node, Rate, Renderable
from relflow.tensorfields.base import TENSORFIELDS, Plugin, TensorFieldBase

OptimizerConfig = torch.optim.Optimizer | Callable[["Model"], torch.optim.Optimizer]
SchedulerConfig = Any | Callable[["Model", torch.optim.Optimizer], Any]

__all__ = [
    "Model",
    "MutationLockCallback",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
]


class Model(lit.LightningModule, Renderable):
    """Neural model generated from a `relflow` schema tree.

    `Model` owns the schema tree, tensorfield embedders, branch
    encoders, decoders, and convenience methods for prediction, checkpointing,
    schema display and mutation.

    Example:
        ```python
        import relflow as rf

        model = rf.Model(
            rf.Category("segment", size=32),
            rf.Category("label", target=True, size=4),
            d_model=16,
            n_layers=1,
            n_heads=4,
            batch_size=8,
            embed=True,
        )
        ```
    """

    @classmethod
    def from_tree(
        cls,
        *field_args: TreeFieldInput,
        d_model: int,
        n_layers: int,
        n_heads: int,
        batch_size: int = 1,
        fields: Sequence[TreeFieldInput] | None = None,
        name: str = "record",
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        n_linear: int = 1,
        dropout: Rate | None = None,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        **field_kwargs: TreeFieldInput,
    ) -> Self:
        """Compatibility wrapper for constructing a model from tree fields.

        New code should call ``Model(...)`` directly with these same arguments.

        Args:
            *field_args: Field constructors such as `Category`, `Number`, or
                nested `Branch` nodes.
            d_model: Shared model width.
            n_layers: Number of encoder layers on generated branch nodes.
            n_heads: Even attention-head count used by the generated root. It
                must divide `d_model` and leave at least two dimensions per
                head.
            batch_size: Batch size used by data modules, examples, and mocked
                Lightning example inputs.
            fields: Optional sequence form of `field_args`.
            name: Root branch name. Defaults to `record`.
            description: Optional description on the generated root branch.
            embed: Configure the generated root branch as an embedding output.
            attention: Attention mode for the generated root branch.
            n_linear: Learned-query cross-attention pooling block count on the
                generated root branch. Each block also contains a feed-forward
                network.
            dropout: Optional dropout rate on the generated root branch.
            optimizer: Optimizer instance or factory used by Lightning training.
            scheduler: Optional scheduler config or factory.

        Returns:
            A compiled `Model` with modules built for the schema.
        """
        return cls(
            *field_args,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            batch_size=batch_size,
            fields=fields,
            name=name,
            description=description,
            embed=embed,
            attention=attention,
            n_linear=n_linear,
            dropout=dropout,
            optimizer=optimizer,
            scheduler=scheduler,
            **field_kwargs,
        )

    @beartype
    def __init__(
        self,
        *field_args: TreeFieldInput | Schema,
        schema: Schema | None = None,
        d_model: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        batch_size: int = 1,
        fields: Sequence[TreeFieldInput] | None = None,
        name: str = "record",
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        n_linear: int = 1,
        dropout: Rate | None = None,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        **field_kwargs: Any,
    ):
        """Build a model from tree fields, or from an existing ``schema``.

        The public constructor accepts the same field and root-architecture
        options as :meth:`from_tree`. Passing ``schema=...`` is retained for
        checkpoint loading and lower-level integrations.
        """
        if field_args and isinstance(field_args[0], Schema):
            if len(field_args) != 1 or schema is not None:
                raise TypeError("a positional Schema cannot be combined with other fields or schema=")
            schema = field_args[0]
            field_args = ()

        if schema is not None:
            if field_args or fields is not None or field_kwargs:
                raise TypeError("schema cannot be combined with tree fields")
            if d_model is not None or n_layers is not None or n_heads is not None:
                raise TypeError("schema cannot be combined with d_model, n_layers, or n_heads")
        else:
            required = {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}
            missing = [key for key, value in required.items() if value is None]
            if missing:
                names = ", ".join(missing)
                raise TypeError(f"Model requires {names} when constructed from tree fields")

            schema = Schema.from_tree(
                *cast(tuple[TreeFieldInput, ...], field_args),
                d_model=cast(int, d_model),
                n_layers=cast(int, n_layers),
                n_heads=cast(int, n_heads),
                fields=fields,
                name=name,
                description=description,
                embed=embed,
                attention=attention,
                n_linear=n_linear,
                dropout=dropout,
                **field_kwargs,
            )

        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.schema: Schema = schema
        self.batch_size: int = batch_size
        self.optimizer: OptimizerConfig | None = optimizer
        self.scheduler: SchedulerConfig | None = scheduler
        self.locks: Counter[str | Strata] = Counter()
        self.nodes: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._schema_editor: SchemaEditor = SchemaEditor(self)
        self._contract_generation: int = 0
        self._contract_scheduler: ContractScheduler = ContractScheduler()

        self._build()

        logger.bind(
            component="model",
            batch_size=self.batch_size,
            requests=len(self.schema.active_requests),
            branches=len(self.schema.branches),
            embeds=len(self.schema.embed),
        ).info("initialized Model module")

    def _build(self) -> None:
        ModelGraph.install(self)

    def _rebuild(self) -> None:
        ModelGraph.rebuild(self)
        self._reset_contracts()

    def _reset_contracts(self) -> None:
        self._contract_generation += 1
        self._contract_scheduler.reset()

    def __rich_console__(self, console, options):
        parameters = sum(parameter.numel() for parameter in self.parameters())
        heading = Text()
        heading.append(type(self).__name__, style=self.RICH_NAME_STYLE)
        heading.append(" ")
        heading.append("[model]", style=self.RICH_TYPE_STYLE)
        for name, value in (
            ("batch_size", self.batch_size),
            ("d_model", self.schema.d_model),
            ("parameters", f"{parameters:,}"),
            ("branches", len(self.schema.branches)),
            ("fields", len(self.schema.active_requests)),
            ("targets", len(self.schema.target)),
            ("embeds", len(self.schema.embed)),
        ):
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        yield heading

        lines = list(self.schema.fields.__rich_console__(console, options))
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

    def select(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        """Return schema nodes that satisfy every predicate."""
        return self._schema_editor.select(*predicates, include_root=include_root, use_cache=use_cache)

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
        """Mutate selected schema nodes and rebuild compatible modules.

        `target=True` is shorthand for `p_prune=1.0`; `target=False` clears
        target behavior by setting `p_prune=0.0`.

        Args:
            *predicates: Predicates used to select nodes.
            strict: Raise when a selected node cannot accept one of `values`.
            allow_extra: Permit updates to extra metadata fields on models that
                allow unknown fields.
            include_root: Include the root node in predicate matching.
            validate: Validate each node after applying candidate values.
            use_cache: Permit cached selector results. Mutations default this to
                `False` so updates always evaluate against current schema state.
            **values: Schema attributes to update.
        """
        self._schema_editor.update(
            *predicates,
            strict=strict,
            allow_extra=allow_extra,
            include_root=include_root,
            validate=validate,
            use_cache=use_cache,
            **values,
        )

    def extend(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        """Append new schema fields under one selected branch node and rebuild modules."""
        self._schema_editor.extend(*args, include_root=include_root, use_cache=use_cache)

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        """Permanently remove selected schema nodes and rebuild modules."""
        self._schema_editor.delete(*predicates, include_root=include_root, use_cache=use_cache)

    def reset(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
        descendants: bool = False,
    ) -> None:
        """Reinitialize selected runtime node modules while preserving schema values."""
        self._schema_editor.reset(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
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
        """Temporarily mutate selected schema nodes and keep runtime modules synchronized."""
        with self._schema_editor.override(
            *predicates,
            strict=strict,
            allow_extra=allow_extra,
            include_root=include_root,
            validate=validate,
            use_cache=use_cache,
            **values,
        ):
            yield

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []
        factories: set[Any] = set()
        trainer = getattr(self, "_trainer", None)
        attached_callback_types = {type(callback) for callback in getattr(trainer, "callbacks", ())}

        if RuntimePlacementCallback not in attached_callback_types:
            callbacks.append(RuntimePlacementCallback())
        if MutationLockCallback not in attached_callback_types:
            callbacks.append(MutationLockCallback())
        if ThroughputLogger not in attached_callback_types:
            callbacks.append(ThroughputLogger())

        for request in self.schema.active_requests.values():
            plugin: Plugin = TENSORFIELDS[request.type]
            for factory in plugin.callback_factories:
                if factory in factories:
                    continue

                factories.add(factory)
                callback = factory()
                if type(callback) not in attached_callback_types:
                    callbacks.append(callback)

        # Callbacks may perform distributed work, so register them in a
        # deterministic order on every rank. Use class paths instead of Python's
        # salted hash or schema traversal order.
        callbacks.sort(
            key=lambda callback: (
                type(callback).__module__,
                type(callback).__qualname__,
            )
        )

        return callbacks

    def track(self, names: tuple[str, ...], /, value: torch.Tensor | TorchMetric) -> torch.Tensor | TorchMetric:
        def groupname(names: tuple[str, ...]) -> str:
            assert len(names) > 1

            group, *keys = tuple(map(lambda x: x.replace("/", ".").lower(), names))
            key = ".".join(list(keys))

            return f"{group}/{key}"

        # Scalar metrics are emitted from data-dependent branches, so DDP ranks cannot
        # safely synchronize every scalar log call as a collective. Stateful
        # TorchMetrics are updated/logged on every rank and can aggregate their state.
        stateful = isinstance(value, TorchMetric)
        self.log(
            name=groupname(names),
            value=value.detach() if isinstance(value, torch.Tensor) else value,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=not stateful,
            batch_size=self.batch_size,
        )

        return value

    @property
    def interprocess_encoding_context(self) -> dict[Address, Any]:
        return {
            Address(str(address)): node.embedder.interprocess_encoding_context
            for address, node in self.nodes.items()
            if hasattr(node, "embedder") and hasattr(node.embedder, "interprocess_encoding_context")
        }

    @beartype
    def save(self, pathname: str | Path) -> str | Path:
        """Save model weights and schema to a checkpoint."""
        CheckpointState.save(self, pathname)

        return pathname

    @immutable("forward")
    @beartype
    def forward(
        self,
        inputs: TensorDict[Address, TensorFieldBase],
        *,
        strata: Strata | str,
        dataloader_idx: int = 0,
    ) -> list[Prediction]:
        return ModelRuntime.forward(self, inputs, strata=strata, dataloader_idx=dataloader_idx)

    @beartype
    def configure_optimizers(self):
        if self.optimizer is None:
            raise ValueError("optimizer must be passed to Model before fitting")

        if isinstance(self.optimizer, torch.optim.Optimizer):
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self)

        scheduler = self.scheduler(self, optimizer) if callable(self.scheduler) else self.scheduler

        if scheduler is None:
            return optimizer

        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def on_save_checkpoint(self, checkpoint):
        CheckpointState.dump(self, checkpoint)

    def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore this model in place from a `relflow` checkpoint dictionary."""
        CheckpointState.restore(self, checkpoint)

    @classmethod
    def load(cls, checkpoint: str | Path) -> Self:
        """Load a `Model` checkpoint written by `Model.save(...)`."""
        return cast(Self, CheckpointState.load(cls, checkpoint))

    from_checkpoint = load

    def write(self, predictions: list[Prediction]) -> dict[Address, dict[str, Any]]:
        return ModelRuntime.write(self, predictions)

    @immutable("inference")
    def encode(
        self,
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: Preprocessor | None = None,
        strata: Strata | str = Strata.predict,
        mask: bool = True,
    ) -> EncodedInput:
        """Return encoded tensorfield inputs for raw or processed observations."""
        return ModelRuntime.encode(
            self,
            batch=batch,
            preprocess=preprocess,
            strata=strata,
            mask=mask,
        )

    @immutable("inference")
    def predict(
        self,
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: Preprocessor | None = None,
        postprocess: Postprocessor | None = None,
    ) -> dict[Address, dict[str, Any]]:
        return ModelRuntime.predict(
            self,
            batch=batch,
            preprocess=preprocess,
            postprocess=postprocess,
        )

    training_step = partialmethod(step, strata=Strata.train)
    validation_step = partialmethod(step, strata=Strata.validate)
    test_step = partialmethod(step, strata=Strata.test)
    predict_step = partialmethod(step, strata=Strata.predict)
