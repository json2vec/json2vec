"""Public Lightning model facade for `json2vec` schemas."""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import lightning.pytorch as lit
import torch
from beartype import beartype
from lightning.pytorch import Callback
from loguru import logger
from rich.text import Text
from tensordict import TensorDict

from json2vec.architecture.checkpoint import CheckpointState, RollbackCheckpoint
from json2vec.architecture.contracts import ContractScheduler
from json2vec.architecture.graph import ModelGraph
from json2vec.architecture.mutations import (
    MutationLockCallback,
    RuntimePlacementCallback,
    SchemaEditor,
    immutable,
)
from json2vec.architecture.runtime import ModelRuntime, Postprocessor, Preprocessor, step
from json2vec.data.datasets.base import EncodedBatch, EncodedInput
from json2vec.logging.throughput import ThroughputLogger
from json2vec.structs.enums import AttentionMode, Strata
from json2vec.structs.experiment import (
    Hyperparameters,
    NodeAttribute,
    NodePredicate,
    SchemaField,
)
from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address, Node, Rate, Renderable
from json2vec.tensorfields.base import TENSORFIELDS, Plugin, TensorFieldBase

if TYPE_CHECKING:
    from json2vec.helpers.inference import InferenceConfig

OptimizerConfig = torch.optim.Optimizer | Callable[["Model"], torch.optim.Optimizer]
SchedulerConfig = Any | Callable[["Model", torch.optim.Optimizer], Any]

__all__ = [
    "Model",
    "MutationLockCallback",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
]


class Model(lit.LightningModule, Renderable):
    """Neural model generated from a `json2vec` schema tree.

    `Model` owns the schema hyperparameters, tensorfield embedders, array
    encoders, decoders, and convenience methods for prediction, checkpointing,
    schema display and mutation.

    Example:
        ```python
        import json2vec as j2v

        model = j2v.Model.from_schema(
            j2v.Category("segment", max_vocab_size=32),
            j2v.Category("label", target=True, max_vocab_size=4),
            d_model=16,
            n_layers=1,
            n_heads=4,
            batch_size=8,
            embed=True,
        )
        ```
    """

    @classmethod
    def from_schema(
        cls,
        *field_args: SchemaField,
        d_model: int,
        n_layers: int,
        n_heads: int,
        batch_size: int = 1,
        fields: Sequence[SchemaField] | None = None,
        name: str = "record",
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        n_linear: int = 1,
        dropout: Rate | None = None,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
    ) -> Self:
        """Build a model directly from schema fields.

        Args:
            *field_args: Field constructors such as `Category`, `Number`, or
                nested `Array` nodes.
            d_model: Shared model width.
            n_layers: Number of encoder layers on generated array nodes.
            n_heads: Attention heads used by generated nodes.
            batch_size: Batch size used by data modules, examples, and mocked
                Lightning input arrays.
            fields: Optional sequence form of `field_args`.
            name: Root array name. Defaults to `record`.
            description: Optional description on the generated root array.
            embed: Configure the generated root array as an embedding output.
            attention: Attention mode for the generated root array.
            n_linear: Feed-forward block count on the generated root array.
            dropout: Optional dropout rate on the generated root array.
            optimizer: Optimizer instance or factory used by Lightning training.
            scheduler: Optional scheduler config or factory.

        Returns:
            A compiled `Model` with modules built for the schema.
        """
        hyperparameters = Hyperparameters.from_schema(
            *field_args,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            fields=fields,
            name=name,
            description=description,
            embed=embed,
            attention=attention,
            n_linear=n_linear,
            dropout=dropout,
        )
        return cls(
            hyperparameters=hyperparameters,
            batch_size=batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    @classmethod
    def from_records(
        cls,
        records: Any,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        batch_size: int = 1,
        name: str = "record",
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        n_linear: int = 1,
        dropout: Rate | None = None,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        infer: "InferenceConfig | None" = None,
        explain: bool = False,
        **infer_overrides: Any,
    ) -> Self:
        """Build a model from sample records by inferring the schema.

        Convenience wrapper that runs :func:`json2vec.infer_schema` over the
        records and feeds the inferred field constructors to
        :meth:`from_schema`. The inferred schema is a best-effort starting
        point; correct any guesses afterwards with ``model.update(...)``.

        Args:
            records: A sequence of dict-like records, an iterable of them, or a
                frame exposing ``.to_dicts()`` (e.g. a Polars ``DataFrame``).
            d_model: Shared model width.
            n_layers: Number of encoder layers on generated array nodes.
            n_heads: Attention heads used by generated nodes.
            batch_size: Batch size used by data modules and mocked input arrays.
            name: Root array name. Defaults to ``record``.
            description: Optional description on the generated root array.
            embed: Configure the generated root array as an embedding output.
            attention: Attention mode for the generated root array.
            n_linear: Feed-forward block count on the generated root array.
            dropout: Optional dropout rate on the generated root array.
            optimizer: Optimizer instance or factory used by Lightning training.
            scheduler: Optional scheduler config or factory.
            infer: Full :class:`~json2vec.InferenceConfig`. When omitted a
                default config is built from ``infer_overrides``.
            explain: When ``True``, print the inferred type of each column.
            **infer_overrides: Convenience overrides for individual
                :class:`~json2vec.InferenceConfig` fields.

        Returns:
            A compiled `Model` built for the inferred schema.
        """
        from json2vec.helpers.inference import infer_schema

        fields = infer_schema(records, config=infer, explain=explain, **infer_overrides)
        return cls.from_schema(
            *fields,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            batch_size=batch_size,
            name=name,
            description=description,
            embed=embed,
            attention=attention,
            n_linear=n_linear,
            dropout=dropout,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    @beartype
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        *,
        batch_size: int = 1,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.hyperparameters: Hyperparameters = hyperparameters
        self.batch_size: int = batch_size
        self.optimizer: OptimizerConfig | None = optimizer
        self.scheduler: SchedulerConfig | None = scheduler
        self.locks: Counter[str | Strata] = Counter()
        self.nodes: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.schema: SchemaEditor = SchemaEditor(self)
        self._contract_generation: int = 0
        self._contract_scheduler: ContractScheduler = ContractScheduler()

        self._build()

        logger.bind(
            component="model",
            batch_size=self.batch_size,
            requests=len(self.hyperparameters.active_requests),
            arrays=len(self.hyperparameters.arrays),
            embeds=len(self.hyperparameters.embed),
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
            ("d_model", self.hyperparameters.d_model),
            ("parameters", f"{parameters:,}"),
            ("arrays", len(self.hyperparameters.arrays)),
            ("fields", len(self.hyperparameters.active_requests)),
            ("targets", len(self.hyperparameters.target)),
            ("embeds", len(self.hyperparameters.embed)),
        ):
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        yield heading

        lines = list(self.hyperparameters.fields.__rich_console__(console, options))
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
        return self.schema.select(*predicates, include_root=include_root, use_cache=use_cache)

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
        self.schema.update(
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
        """Append new schema fields under one selected array node and rebuild modules."""
        self.schema.extend(*args, include_root=include_root, use_cache=use_cache)

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        """Permanently remove selected schema nodes and rebuild modules."""
        self.schema.delete(*predicates, include_root=include_root, use_cache=use_cache)

    def reset(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
        descendants: bool = False,
    ) -> None:
        """Reinitialize selected runtime node modules while preserving schema values."""
        self.schema.reset(
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
        with self.schema.override(
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

        for request in self.hyperparameters.active_requests.values():
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

    def track(self, names: tuple[str, ...], /, value: torch.Tensor) -> torch.Tensor:
        def groupname(names: tuple[str, ...]) -> str:
            assert len(names) > 1

            group, *keys = tuple(map(lambda x: x.replace("/", ":").lower(), names))
            key = ":".join(list(keys))

            return f"{group}/{key}"

        # These metrics are emitted from data-dependent branches, so DDP ranks cannot
        # safely synchronize every log call as a collective. rank_zero_only keeps
        # Lightning from running a sync while still marking the metric as handled.
        self.log(
            name=groupname(names),
            value=value.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
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
        """Save model weights and schema hyperparameters to a checkpoint."""
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
        """Restore this model in place from a `json2vec` checkpoint dictionary."""
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
        """Return typed predictions and configured embeddings for a raw or encoded batch."""
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
