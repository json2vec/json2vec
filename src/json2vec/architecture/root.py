"""Lightning model assembly and runtime helpers for JSON2Vec schemas."""

import uuid
import weakref
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from functools import partialmethod
from pathlib import Path
from typing import Any, Literal, NotRequired, Self, TypeAlias, TypedDict, cast

import lightning.pytorch as lit
import torch
from beartype import beartype
from lightning.pytorch import Callback
from loguru import logger
from tensordict import TensorDict

from json2vec.architecture.encoder import ArrayEncoder
from json2vec.architecture.node import NodeModule
from json2vec.data.datasets.base import EncodedBatch
from json2vec.data.iterables import encode, mock
from json2vec.structs.enums import Metric, Strata, TensorKey
from json2vec.structs.experiment import (
    Hyperparameters,
    MutationChange,
    MutationResult,
    NodeAttribute,
    NodePredicate,
    SchemaField,
)
from json2vec.structs.packages import Embedding, Parcel, Prediction
from json2vec.structs.tree import Address, Node, PruneRate, Rate
from json2vec.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)


class Output(TypedDict):
    loss: NotRequired[torch.Tensor]
    predictions: NotRequired[list[Prediction]]


OptimizerConfig = torch.optim.Optimizer | Callable[["Model"], torch.optim.Optimizer]
SchedulerConfig = Any | Callable[["Model", torch.optim.Optimizer], Any]
PreprocessFn: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
Postprocessor: TypeAlias = Callable[
    [dict[str, Any], dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]],
    tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]] | None,
]


class MutationLockCallback(Callback):
    """Prevent runtime schema mutations while Lightning owns an active loop."""

    locks: tuple[Strata, ...] = (Strata.train, Strata.validate, Strata.test, Strata.predict)

    def _on_loop_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        enter = getattr(pl_module, "_enter_mutation_lock", None)
        if callable(enter):
            enter(strata)

    def _on_loop_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        exit_ = getattr(pl_module, "_exit_mutation_lock", None)
        if callable(exit_):
            exit_(strata)

    on_train_start = partialmethod(_on_loop_start, strata=Strata.train)
    on_train_end = partialmethod(_on_loop_end, strata=Strata.train)
    on_validation_start = partialmethod(_on_loop_start, strata=Strata.validate)
    on_validation_end = partialmethod(_on_loop_end, strata=Strata.validate)
    on_test_start = partialmethod(_on_loop_start, strata=Strata.test)
    on_test_end = partialmethod(_on_loop_end, strata=Strata.test)
    on_predict_start = partialmethod(_on_loop_start, strata=Strata.predict)
    on_predict_end = partialmethod(_on_loop_end, strata=Strata.predict)

    def on_exception(self, trainer: lit.Trainer, pl_module: lit.LightningModule, exception: BaseException) -> None:
        release = getattr(pl_module, "_release_mutation_locks", None)
        if callable(release):
            release(self.locks)


@beartype
def step(
    module: "Model",
    batch: TensorDict[Address, TensorFieldBase],
    batch_idx: int,
    strata: Strata,
) -> Output:
    predictions: list[Prediction] = module.forward(batch)

    if strata == Strata.predict:
        return Output(predictions=predictions)

    losses: list[torch.Tensor] = []

    for prediction in predictions:

        if isinstance(prediction, Embedding):
            continue

        address: Address = prediction.address
        request: RequestBase = module.hyperparameters.requests[address]
        extension: Plugin = TENSORFIELDS[request.type]
        loss_fn = cast(Callable[..., torch.Tensor], getattr(extension, "loss"))

        loss: torch.Tensor = loss_fn(module=module, prediction=prediction, batch=batch[address], strata=strata)
        losses.append(loss * torch.tensor(request.weight))

    if len(losses) == 0:
        # under idealistic circumstances this would never happen.
        # but with small mask rates, batch sizes, and flat input data it is possible
        logger.warning("no trainable fields in batch, returning zero loss")
        loss: torch.Tensor = torch.tensor(0.0, device=batch.device, requires_grad=True)
        return Output(loss=loss)

    loss: torch.Tensor = module.track((Metric.loss, strata), value=torch.stack(losses).sum())

    return Output(loss=loss)


class Model(lit.LightningModule):
    """Neural model generated from a JSON2Vec schema tree.

    `Model` owns the schema hyperparameters, tensorfield embedders, array
    encoders, decoders, and convenience methods for prediction, embedding,
    checkpointing, plotting, and schema mutation.

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
        root: str = "record",
        description: str | None = None,
        embed: bool = False,
        attention: Literal["mha", "gqa", "mqa", "none"] = "mha",
        max_length: int = 1,
        n_outputs: int = 1,
        n_linear: int = 1,
        dropout: Rate | None = None,
        p_mask: Rate | None = None,
        p_prune: PruneRate | None = None,
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
            root: Root array name. Defaults to `record`.
            description: Optional description on the generated root array.
            embed: Configure the generated root array as an embedding output.
            attention: Attention mode for the generated root array.
            max_length: Maximum number of records per observation at the root.
            n_outputs: Number of pooled outputs emitted by the generated root array.
            n_linear: Feed-forward block count on the generated root array.
            dropout: Optional dropout rate on the generated root array.
            p_mask: Optional mask rate on the generated root array.
            p_prune: Optional prune rate on the generated root array.
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
            root=root,
            description=description,
            embed=embed,
            attention=attention,
            max_length=max_length,
            n_outputs=n_outputs,
            n_linear=n_linear,
            dropout=dropout,
            p_mask=p_mask,
            p_prune=p_prune,
        )
        return cls(
            hyperparameters=hyperparameters,
            batch_size=batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Self:
        """Alias for `from_schema(...)`."""
        return cls.from_schema(*args, **kwargs)

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
        self._mutation_locks: Counter[str] = Counter()
        self._data_modules: weakref.WeakSet[Any] = weakref.WeakSet()

        self._build()

        logger.bind(
            component="model",
            batch_size=self.batch_size,
            requests=len(self.hyperparameters.active_requests),
            arrays=len(self.hyperparameters.arrays),
            embeds=len(self.hyperparameters.embed),
        ).info("initialized Model module")

    def _build(self) -> None:
        self.nodes: torch.nn.ModuleDict[str, NodeModule] = torch.nn.ModuleDict()

        for address in self.hyperparameters.requests | self.hyperparameters.arrays:
            self.nodes[address] = NodeModule(
                hyperparameters=self.hyperparameters,
                address=address,
                batch_size=self.batch_size,
            )

        self.example_input_array = mock(hyperparameters=self.hyperparameters, batch_size=self.batch_size)

    def _runtime_placement(self) -> tuple[torch.device | None, torch.dtype | None]:
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for tensor in (*self.parameters(), *self.buffers()):
            if device is None:
                device = tensor.device
            if dtype is None and tensor.is_floating_point():
                dtype = tensor.dtype
            if device is not None and dtype is not None:
                break

        return device, dtype

    def _apply_runtime_placement(
        self,
        module: torch.nn.Module,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        if device is None and dtype is None:
            return
        if device is None:
            module.to(dtype=dtype)
            return
        if dtype is None:
            module.to(device=device)
            return

        module.to(device=device, dtype=dtype)

    def _rebuild(self) -> None:
        device, dtype = self._runtime_placement()
        previous = {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for name, value in self.state_dict().items()
        }
        self._build()
        self._apply_runtime_placement(self, device=device, dtype=dtype)
        current = self.state_dict()
        compatible = {}
        for name, value in previous.items():
            if name not in current:
                continue

            current_value = current[name]
            if isinstance(current_value, torch.Tensor) and isinstance(value, torch.Tensor):
                if current_value.shape != value.shape:
                    continue
            elif type(current_value) is not type(value):
                continue

            compatible[name] = value

        self.load_state_dict(compatible, strict=False)
        self._refresh_data_modules()

    def _register_data_module(self, datamodule: Any) -> None:
        self._data_modules.add(datamodule)

    def _refresh_data_modules(self) -> None:
        context = self.interprocess_encoding_context
        for datamodule in list(self._data_modules):
            setter = getattr(datamodule, "_set_interprocess_encoding_context", None)
            if callable(setter):
                setter(context)
            else:
                setattr(datamodule, "interprocess_encoding_context", context)

    def select(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        """Return schema nodes that satisfy every predicate."""
        return self.hyperparameters.select(
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
        **values: Any,
    ) -> None:
        """Mutate selected schema nodes and rebuild compatible modules.

        `target=True` is shorthand for `p_prune=1.0`; `target=False` clears
        target behavior by setting `p_prune=None`.

        Args:
            *predicates: Predicates used to select nodes.
            strict: Raise when a selected node cannot accept one of `values`.
            allow_extra: Permit updates to extra metadata fields on models that
                allow unknown fields.
            include_root: Include the root node in predicate matching.
            validate: Validate each node after applying candidate values.
            **values: Schema attributes to update.
        """
        self._assert_mutation_allowed("update")
        self.hyperparameters.update(
            *predicates,
            strict=strict,
            allow_extra=allow_extra,
            include_root=include_root,
            validate=validate,
            **values,
        )
        self._rebuild()

    def extend(
        self,
        *args: NodePredicate | NodeAttribute | Callable[[Node], bool] | SchemaField,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> None:
        """Append new schema fields under one selected array node and rebuild modules."""
        self._assert_mutation_allowed("extend")
        self.hyperparameters.extend(*args, include_root=include_root, use_cache=use_cache)
        self._rebuild()

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        """Permanently remove selected schema nodes and rebuild modules."""
        self._assert_mutation_allowed("delete")
        self.hyperparameters.delete(*predicates, include_root=include_root, use_cache=use_cache)
        self._rebuild()

    def reset(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        use_cache: bool = True,
        descendants: bool = False,
    ) -> None:
        """Reinitialize selected runtime node modules while preserving schema values."""
        self._assert_mutation_allowed("reset")
        selected = self.hyperparameters.select(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
        )
        if not selected:
            raise ValueError("reset matched no nodes")

        selected_by_address: dict[Address, Node] = {}
        for node in selected:
            if node.address in self.nodes:
                selected_by_address[Address(str(node.address))] = node

            if descendants:
                for descendant in getattr(node, "descendants", ()):
                    if descendant.address in self.nodes:
                        selected_by_address[Address(str(descendant.address))] = descendant

        if not selected_by_address:
            raise ValueError("reset matched no runtime nodes")

        changes: list[MutationChange] = []
        device, dtype = self._runtime_placement()
        for address, node in selected_by_address.items():
            module = self.nodes[address]
            state_keys = tuple(module.state_dict().keys())
            parameter_count = sum(parameter.numel() for parameter in module.parameters())
            replacement = NodeModule(
                hyperparameters=self.hyperparameters,
                address=address,
                batch_size=self.batch_size,
            )
            self._apply_runtime_placement(replacement, device=device, dtype=dtype)
            self.nodes[address] = replacement
            changes.append(
                MutationChange(
                    node=str(node.address),
                    field="state",
                    old={"parameter_count": parameter_count, "state_keys": state_keys},
                    new=None,
                    action="reset",
                )
            )

        self.example_input_array = mock(hyperparameters=self.hyperparameters, batch_size=self.batch_size)
        self._refresh_data_modules()
        result = MutationResult(
            operation_id=uuid.uuid4().hex,
            action="reset",
            matched=len(selected),
            updated=len(selected_by_address),
            changes=tuple(changes),
        )
        self.hyperparameters._record_mutation(result)

    @contextmanager
    def override(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> Iterator[MutationResult]:
        """Temporarily mutate selected schema nodes and keep runtime modules synchronized."""
        self._assert_mutation_allowed("override")
        try:
            with self.hyperparameters.override(
                *predicates,
                strict=strict,
                allow_extra=allow_extra,
                include_root=include_root,
                validate=validate,
                **values,
            ) as result:
                self._rebuild()
                yield result
        finally:
            self._rebuild()

    def _assert_mutation_allowed(self, action: str) -> None:
        active = tuple(name for name, count in self._mutation_locks.items() if count > 0)
        if active:
            labels = ", ".join(active)
            raise RuntimeError(f"model.{action}(...) cannot run while the model is in an active loop: {labels}")

    def _enter_mutation_lock(self, name: str) -> None:
        self._mutation_locks[name] += 1

    def _exit_mutation_lock(self, name: str) -> None:
        if self._mutation_locks[name] <= 1:
            self._mutation_locks.pop(name, None)
            return

        self._mutation_locks[name] -= 1

    def _release_mutation_locks(self, names: Sequence[str]) -> None:
        for name in names:
            self._mutation_locks.pop(name, None)

    @contextmanager
    def _mutation_lock(self, name: str) -> Iterator[None]:
        self._enter_mutation_lock(name)
        try:
            yield
        finally:
            self._exit_mutation_lock(name)

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []
        factories: set[Any] = set()
        trainer = getattr(self, "_trainer", None)
        attached_callback_types = {
            type(callback)
            for callback in getattr(trainer, "callbacks", ())
        }

        if MutationLockCallback not in attached_callback_types:
            callbacks.append(MutationLockCallback())

        for request in self.hyperparameters.active_requests.values():
            plugin: Plugin = TENSORFIELDS[request.type]
            for factory in plugin.callback_factories:
                if factory in factories:
                    continue

                factories.add(factory)
                callback = factory()
                if type(callback) not in attached_callback_types:
                    callbacks.append(callback)

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
            value=value,
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
            address: node.embedder.interprocess_encoding_context
            for address, node in self.nodes.items()
            if hasattr(node, "embedder") and hasattr(node.embedder, "interprocess_encoding_context")
        }

    def plot(
        self,
        address: Address | str | None = None,
        detail: bool = False,
        out: str | Path | None = None,
        mode: str = "schema",
    ) -> None:
        """Print a Rich model visualization.

        Args:
            address: Optional subtree address to render.
            detail: Include tensorfield-specific detail sections.
            out: Optional output path for the rendered console text.
            mode: Plot mode. Supported values are `schema`, `state`, `flow`,
                and `debug`.
        """
        from json2vec.architecture.plot import plot

        return plot(module=self, address=address, detail=detail, out=out, mode=mode)

    @beartype
    def save(self, pathname: str | Path) -> None:
        """Save model weights and schema hyperparameters to a checkpoint."""
        path = Path(pathname)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {"state_dict": self.state_dict()}
        self.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, path)

    @beartype
    def forward(self, inputs: TensorDict[Address, TensorFieldBase]) -> list[Prediction]:
        with self._mutation_lock("forward"):
            return self._forward(inputs)

    def _forward(self, inputs: TensorDict[Address, TensorFieldBase]) -> list[Prediction]:
        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []

        for address in self.hyperparameters.active_requests.keys():
            tensorfield: TensorFieldBase = inputs[address]
            if address in self.hyperparameters.target:
                continue

            embedder: EmbedderBase = self.nodes[address].embedder
            embedding: Parcel = embedder(tensorfield)
            processed[embedding.destination].append(embedding)
            outgoing[embedding.origin] = embedding

            if address in self.hyperparameters.embed:
                predictions.append(Embedding.from_parcel(embedding))

        # DAG traversal from leaves to root
        for depth in reversed(self.hyperparameters.depthwise):
            # these are order-independent within the same depth
            for address in depth:

                if len(processed[address]) == 0:
                    continue

                encoder: ArrayEncoder = self.nodes[address].encoder
                encoding: Parcel = encoder(processed[address])
                processed[encoding.destination].append(encoding)
                outgoing[encoding.origin] = encoding

                if address in self.hyperparameters.embed:
                    predictions.append(Embedding.from_parcel(encoding))

        for address in self.hyperparameters.active_requests.keys():

            if (torch.any(inputs[address].trainable)) or (address in self.hyperparameters.target):

                heritage: list[Address] = self.hyperparameters.requests[address].heritage
                parcels: list[Parcel] = [
                    outgoing[address] for address in heritage
                    if address not in self.hyperparameters.target and address in outgoing.keys()
                ]

                decoder: DecoderBase = self.nodes[address].decoder
                predictions.append(decoder(parcels))

        return predictions

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
        checkpoint["hyperparameters"] = self.hyperparameters.model_dump(mode="python")
        checkpoint["batch_size"] = self.batch_size

    @classmethod
    def load(cls, checkpoint: str | Path) -> Self:
        """Load a `Model` checkpoint written by `Model.save(...)`."""
        path = Path(checkpoint)
        logger.bind(component="model_factory", checkpoint=str(path)).info("loading Model from checkpoint")
        state = torch.load(path, weights_only=False, map_location="cpu")
        if "hyperparameters" not in state:
            raise ValueError("missing hyperparameters in checkpoint")

        model: "Model" = cls(
            hyperparameters=Hyperparameters.model_validate(state["hyperparameters"]),
            batch_size=state["batch_size"],
        )
        model.load_state_dict(state_dict=state["state_dict"])
        logger.bind(component="model_factory", checkpoint=str(path)).info("restored model state from checkpoint")

        return model

    def write(self, predictions: list[Prediction]) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:

        supervised: dict[Address, dict[str, Any]] = {}
        embeddings: dict[Address, dict[str, Any]] = {}

        for prediction in predictions:

            if isinstance(prediction, Embedding):

                embeddings[prediction.address] = Prediction.serialize(
                    Prediction.squeeze(Embedding.write(prediction), preserve_first_dimension=True)
                )

                continue

            request: RequestBase = self.hyperparameters.requests[prediction.address]

            extension: Plugin = TENSORFIELDS[request.type]
            write_fn = cast(Callable[..., dict[TensorKey, Any] | None], getattr(extension, "write"))

            scribed: dict[TensorKey, Any] | None = write_fn(module=self, prediction=prediction)

            if scribed is not None:
                supervised[prediction.address] = Prediction.serialize(
                    Prediction.squeeze(scribed, preserve_first_dimension=True)
                )

        return supervised, embeddings

    def evaluate(
        self,
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: PreprocessFn | None = None,
        postprocess: Postprocessor | None = None,
    ) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:
        """Run prediction and embedding for encoded or raw observations.

        If `preprocess` is omitted, raw records are encoded unchanged.
        """
        with self._mutation_lock("inference"):
            return self._evaluate(batch=batch, preprocess=preprocess, postprocess=postprocess)

    def _evaluate(
        self,
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: PreprocessFn | None = None,
        postprocess: Postprocessor | None = None,
    ) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:
        was_training = self.training
        raw_batch = batch

        if preprocess is not None:
            observations: EncodedBatch = []
            for request in batch:
                observation = preprocess(request)
                if not isinstance(observation, dict):
                    raise TypeError(f"preprocessor must return a dict object, got {type(observation).__name__}")

                observations.append([observation])

            batch = observations

        inputs = encode(
            batch=batch,
            hyperparameters=self.hyperparameters,
            strata=Strata.predict,
            interprocess_encoding_context=self.interprocess_encoding_context,
        )

        self.eval()
        try:
            with torch.inference_mode():
                predictions = self(inputs)
        finally:
            if was_training:
                self.train()

        supervised, embeddings = self.write(predictions)

        if postprocess is not None:
            context = {
                "batch": raw_batch,
                "observations": batch,
                "input": inputs,
                "metadata": inputs["metadata"],
            }
            processed = postprocess(context, supervised, embeddings)

            if processed is not None:
                supervised, embeddings = processed

        return supervised, embeddings

    def predict(
        self,
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: PreprocessFn | None = None,
        postprocess: Postprocessor | None = None,
    ) -> dict[Address, dict[str, Any]]:
        """Return typed predictions for a raw or encoded batch."""
        supervised, _ = self.evaluate(
            batch=batch,
            preprocess=preprocess,
            postprocess=postprocess,
        )
        return supervised

    def embed(
        self,
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: PreprocessFn | None = None,
        postprocess: Postprocessor | None = None,
    ) -> dict[Address, dict[str, Any]]:
        """Return configured embeddings for a raw or encoded batch."""
        _, embeddings = self.evaluate(
            batch=batch,
            preprocess=preprocess,
            postprocess=postprocess,
        )
        return embeddings

    training_step = partialmethod(step, strata=Strata.train)
    validation_step = partialmethod(step, strata=Strata.validate)
    test_step = partialmethod(step, strata=Strata.test)
    predict_step = partialmethod(step, strata=Strata.predict)
