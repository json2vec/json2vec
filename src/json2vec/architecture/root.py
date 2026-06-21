"""Public Lightning model facade for `json2vec` schemas."""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partialmethod, wraps
from pathlib import Path
from typing import Any, Literal, Self, cast

import lightning.pytorch as lit
import torch
from beartype import beartype
from lightning.pytorch import Callback, strategies
from loguru import logger
from rich.text import Text
from tensordict import TensorDict
from torchmetrics import Metric as TorchMetric
from torchjd.aggregation import UPGrad
from torchjd.autojac import Incidence, build_incidence, jac_to_grad, jd_backward

from json2vec.architecture.checkpoint import CheckpointState, RollbackCheckpoint
from json2vec.architecture.contracts import ContractScheduler
from json2vec.architecture.graph import ModelGraph
from json2vec.architecture.mutations import SchemaEditor
from json2vec.architecture.runtime import ModelRuntime, Postprocessor, Preprocessor, step
from json2vec.data.datasets.base import EncodedBatch, EncodedInput
from json2vec.distributed import is_distributed, mean_all_reduce_grads
from json2vec.logging.throughput import ThroughputLogger
from json2vec.structs.enums import AttentionMode, Metric, Strata
from json2vec.structs.experiment import (
    NodeAttribute,
    NodePredicate,
    Schema,
    SchemaField,
    TreeFieldInput,
)
from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address, Node, Rate, Renderable
from json2vec.tensorfields.base import TENSORFIELDS, Plugin, TensorFieldBase

OptimizerConfig = torch.optim.Optimizer | Callable[["Model"], torch.optim.Optimizer]
SchedulerConfig = Any | Callable[["Model", torch.optim.Optimizer], Any]
DistributedJDMode = Literal["auto", "manual_allreduce", "off"]


def _ddp_bypass_incompatible_strategies() -> tuple[type, ...]:
    names = ("FSDPStrategy", "DeepSpeedStrategy", "ModelParallelStrategy", "XLAFSDPStrategy")
    return tuple(cls for name in names if isinstance(cls := getattr(strategies, name, None), type))


_DDP_BYPASS_INCOMPATIBLE_STRATEGIES: tuple[type, ...] = _ddp_bypass_incompatible_strategies()

__all__ = [
    "Model",
    "MutationLockCallback",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
]


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

    def on_exception(self, trainer: lit.Trainer, pl_module: "Model", exception: BaseException) -> None:  # ty:ignore[invalid-method-override]
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


class _BypassDDPWrappingCallback(Callback):
    """Bypass Lightning's DDP wrapping so TorchJD's vmap'd autograd is unblocked.

    ``DistributedDataParallel`` installs C++ post-hooks on every parameter's
    ``AccumulateGrad`` node; those hooks call ``.data_ptr()`` on the gradient
    and fail when ``torchjd``'s autogram engine assembles Jacobians under
    ``torch.vmap`` (TorchJD #154 / PyTorch #138422). ``Model.training_step``
    performs the equivalent mean gradient all-reduce manually via
    ``mean_all_reduce_grads``.
    """

    def setup(self, trainer: lit.Trainer, pl_module: lit.LightningModule, stage: str) -> None:
        from lightning.pytorch.strategies import DDPStrategy

        strategy = trainer.strategy
        if isinstance(strategy, _DDP_BYPASS_INCOMPATIBLE_STRATEGIES):
            name = type(strategy).__name__
            raise RuntimeError(
                f"json2vec Model with distributed_jd != 'off' is incompatible with {name}; "
                "use a default DDPStrategy or pass distributed_jd='off'."
            )
        if not isinstance(strategy, DDPStrategy):
            return
        if getattr(strategy, "_ddp_comm_hook", None) is not None:
            raise RuntimeError(
                "DDPStrategy.ddp_comm_hook is wired into the C++ reducer that the json2vec "
                "Jacobian-descent bypass replaces; drop the hook or pass distributed_jd='off'."
            )

        strategy_name = type(strategy).__name__
        original_configure_ddp = strategy.configure_ddp

        def _bypass_configure_ddp() -> None:
            logger.bind(
                component="ddp",
                strategy=strategy_name,
                stage=stage,
            ).info("bypassing DDP wrapping for Jacobian descent (manual gradient all-reduce)")
            if is_distributed():
                # DDP normally broadcasts rank-0 parameters in its constructor;
                # mirror that here so ranks start from identical state.
                from lightning.pytorch.overrides.distributed import _sync_module_states

                _sync_module_states(strategy.model)

        strategy.configure_ddp = _bypass_configure_ddp  # type: ignore[method-assign]
        self._original_configure_ddp = original_configure_ddp
        self._patched_strategy = strategy

    def teardown(self, trainer: lit.Trainer, pl_module: lit.LightningModule, stage: str) -> None:
        strategy = getattr(self, "_patched_strategy", None)
        original = getattr(self, "_original_configure_ddp", None)
        if strategy is not None and original is not None:
            strategy.configure_ddp = original  # type: ignore[method-assign]
        self._patched_strategy = None
        self._original_configure_ddp = None


class Model(lit.LightningModule, Renderable):
    """Neural model generated from a `json2vec` schema tree.

    `Model` owns the schema tree, tensorfield embedders, branch
    encoders, decoders, and convenience methods for prediction, checkpointing,
    schema display and mutation.

    Example:
        ```python
        import json2vec as jv

        model = jv.Model(
            jv.Category("segment", size=32),
            jv.Category("label", target=True, size=4),
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
        distributed_jd: DistributedJDMode = "off",
        **field_kwargs: TreeFieldInput,
    ) -> Self:
        """Compatibility wrapper for constructing a model from tree fields.

        New code should call ``Model(...)`` directly with these same arguments.

        Args:
            *field_args: Field constructors such as `Category`, `Number`, or
                nested `Branch` nodes.
            d_model: Shared model width.
            n_layers: Number of encoder layers on generated branch nodes.
            n_heads: Attention heads used by generated nodes.
            batch_size: Batch size used by data modules, examples, and mocked
                Lightning example inputs.
            fields: Optional sequence form of `field_args`.
            name: Root branch name. Defaults to `record`.
            description: Optional description on the generated root branch.
            embed: Configure the generated root branch as an embedding output.
            attention: Attention mode for the generated root branch.
            n_linear: Feed-forward block count on the generated root branch.
            dropout: Optional dropout rate on the generated root branch.
            optimizer: Optimizer instance or factory used by Lightning training.
            scheduler: Optional scheduler config or factory.
            distributed_jd: Multi-rank Jacobian-descent mode.
                ``\"auto\"`` / ``\"manual_allreduce\"`` bypass Lightning's DDP
                wrapping and manually all-reduce gradients (required because
                TorchJD is incompatible with the DDP reducer). ``\"off\"`` disables
                JD and falls back to ``loss.sum()`` backward with stock DDP / FSDP.

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
            distributed_jd=distributed_jd,
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
        distributed_jd: DistributedJDMode = "auto",
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
        self.distributed_jd: DistributedJDMode = distributed_jd
        self.automatic_optimization: bool = False
        self.locks: Counter[str | Strata] = Counter()
        self.nodes: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._schema_editor: SchemaEditor = SchemaEditor(self)
        self._contract_generation: int = 0
        self._contract_scheduler: ContractScheduler = ContractScheduler()
        self._jd_aggregation = None
        self.incidence_matrix = None

        self._build()
        self._build_jd_components()

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

    def _build_jd_components(self) -> None:
        if self.distributed_jd != "off":
            self._jd_aggregation = UPGrad()

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
        if self.distributed_jd != "off" and _BypassDDPWrappingCallback not in attached_callback_types:
            callbacks.append(_BypassDDPWrappingCallback())

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

            group, *keys = tuple(map(lambda x: x.replace("/", ":").lower(), names))
            key = ":".join(list(keys))

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
    def shared_submodules(self) -> list[Any]:
        embedders = [node.embedder for node in self.nodes.values() if hasattr(node, "embedder")]
        encoders = [node.encoder for node in self.nodes.values() if hasattr(node, "encoder")]
        return embedders + encoders

    @property
    def shared_params(self) -> list[Any]:
        params = []
        for module in self.shared_submodules:
            params.extend([x for x in module.parameters() if x.requires_grad])
        return params

    @property
    def interprocess_encoding_context(self) -> dict[Address, Any]:
        return {
            Address(str(address)): node.embedder.interprocess_encoding_context
            for address, node in self.nodes.items()
            if hasattr(node, "embedder") and hasattr(node.embedder, "interprocess_encoding_context")
        }

    @beartype
    def save(self, pathname: str | Path) -> str | Path:
        """Save model weights and schema schema to a checkpoint."""
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
        predictions, _ = ModelRuntime.forward(self, inputs, strata=strata, dataloader_idx=dataloader_idx)
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

    def on_fit_start(self):
        super().on_fit_start()
        self._build_jd_components()
        self.incidence_matrix: Incidence | None = None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        trainer = getattr(self, "_trainer", None)
        accumulate = getattr(trainer, "accumulate_grad_batches", 1)
        is_accumulation_boundary = (batch_idx + 1) % accumulate == 0

        output = step(self, batch, batch_idx, strata=Strata.train)
        if "losses" in output:
            no_losses = False
            losses = output["losses"]
        elif "loss" in output:
            no_losses = True
            losses = [output["loss"]]
        else:
            raise RuntimeError("training step produced no loss")

        loss_vec = torch.stack(losses)

        if self.distributed_jd == "off" or no_losses:
            self.manual_backward(loss_vec.sum())
        else:
            # addresses: list[Address] = output["addresses"]
            # root_embedding: torch.Tensor = output["root_embedding"]
            # shared_param_ids: set[int] = {id(parameter) for parameter in self.shared_params}
            # tasks_params: list[list[torch.nn.Parameter]] = [
            #     [
            #         parameter
            #         for parameter in self.nodes[address].decoder.parameters()
            #         if parameter.requires_grad and id(parameter) not in shared_param_ids
            #     ]
            #     for address in addresses
            # ]
            all_parameters = list(self.parameters())
            if self.incidence_matrix is None:
                self.incidence_matrix = build_incidence(losses, all_parameters)
            incident_parameters = [p for p in all_parameters if p in self.incidence_matrix]
            jd_backward(
                losses,
                params=incident_parameters,
                incidence=self.incidence_matrix,
                epsilon=0.01,
                parallel_chunk_size=None,
            )
            has_jd_hooks = any(getattr(module, "_forward_hooks", None) for module in self._jd_aggregation.modules())
            jac_to_grad(
                [p for p in incident_parameters if getattr(p, "jac", None) is not None],
                self._jd_aggregation,
                optimize_gramian_computation=not has_jd_hooks,
            )

        self.track((Metric.loss, Strata.train), value=loss_vec.detach().sum())

        if is_accumulation_boundary:
            if self.distributed_jd != "off":
                mean_all_reduce_grads(self)
            opt.step()
            opt.zero_grad()
            if sch is not None:
                sch.step()

        return {"loss": loss_vec}

    validation_step = partialmethod(step, strata=Strata.validate)
    test_step = partialmethod(step, strata=Strata.test)
    predict_step = partialmethod(step, strata=Strata.predict)
