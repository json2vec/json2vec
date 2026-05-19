from collections import defaultdict
from collections.abc import Callable
from functools import cache, partialmethod
from pathlib import Path
from typing import Any, NotRequired, Self, TypedDict, cast
from urllib.parse import urlparse

import lightning.pytorch as lit
import pyarrow.fs as pafs
import torch
from beartype import beartype
from lightning.pytorch import Callback
from loguru import logger
from tensordict import TensorDict

from json2vec.architecture.encoder import ArrayEncoder
from json2vec.architecture.node import NodeModule
from json2vec.data.datasets import EncodedBatch, encode, mock
from json2vec.structs.enums import Metric, Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.packages import Embedding, Parcel, Prediction
from json2vec.structs.tree import Address
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
    predictions: list[Prediction]


OptimizerConfig = torch.optim.Optimizer | Callable[["JSON2Vec"], torch.optim.Optimizer]
SchedulerConfig = Any | Callable[["JSON2Vec", torch.optim.Optimizer], Any]


@beartype
def step(
    module: "JSON2Vec",
    batch: TensorDict[Address, TensorFieldBase],
    batch_idx: int,
    strata: Strata,
) -> Output:
    update_counters(module=module, batch=batch, strata=strata)
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
        return Output(loss=loss, predictions=[])

    loss: torch.Tensor = module.track((Metric.loss, strata), value=torch.stack(losses).sum())

    return Output(loss=loss, predictions=predictions)


@cache
def groupname(names: tuple[str, ...]) -> str:
    assert len(names) > 1

    group, *keys = tuple(map(lambda x: x.replace("/", ":").lower(), names))

    key: str = ":".join(list(keys))

    return f"{group}/{key}"


def _counter_value(field: TensorFieldBase, key: TensorKey) -> torch.Tensor | None:
    targets = getattr(field, "targets", None)
    if targets is not None and key in targets.keys():
        return targets[key]

    value = getattr(field, key.name, None)
    if isinstance(value, torch.Tensor):
        return value

    return None


@torch.no_grad()
def update_counters(
    module: "JSON2Vec",
    batch: TensorDict[Address, TensorFieldBase],
    strata: Strata,
) -> None:
    if strata != Strata.train:
        return

    for address in module.hyperparameters.requests:
        field = batch[address]
        decoder = module.nodes[address].decoder
        state = _counter_value(field=field, key=TensorKey.state)

        if state is not None and hasattr(decoder, "counter"):
            decoder.counter(state)

        counters = getattr(decoder, "counters", None)
        if counters is None:
            continue

        if state is not None and TensorKey.state.name in counters:
            counters[TensorKey.state.name](state)

        if state is None or TensorKey.content.name not in counters:
            continue

        content = _counter_value(field=field, key=TensorKey.content)
        if content is None or content.shape != state.shape:
            continue

        values = content.masked_select(state.eq(Tokens.valued.value))
        if values.numel() > 0:
            counters[TensorKey.content.name](values)


class JSON2Vec(lit.LightningModule):
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

        self.nodes: torch.nn.ModuleDict[str, NodeModule] = torch.nn.ModuleDict()

        for address in self.hyperparameters.requests | self.hyperparameters.arrays:
            self.nodes[address] = NodeModule(
                hyperparameters=self.hyperparameters,
                address=address,
                batch_size=self.batch_size,
            )

        self.example_input_array = mock(hyperparameters=self.hyperparameters, batch_size=self.batch_size)

        logger.bind(
            component="model",
            batch_size=self.batch_size,
            requests=len(self.hyperparameters.requests),
            arrays=len(self.hyperparameters.arrays),
            embeds=len(self.hyperparameters.embed),
        ).info("initialized JSON2Vec module")

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []
        factories: set[Any] = set()

        for request in self.hyperparameters.requests.values():
            plugin: Plugin = TENSORFIELDS[request.type]
            for factory in plugin.callback_factories:
                if factory in factories:
                    continue

                factories.add(factory)
                callbacks.append(factory())

        return callbacks

    def track(self, names: tuple[str, ...], /, value: torch.Tensor) -> torch.Tensor:
        self.log(
            name=groupname(names),
            value=value,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return value

    @property
    def state(self) -> dict[Address, Any]:
        return {
            address: node.embedder.state
            for address, node in self.nodes.items()
            if hasattr(node, "embedder") and hasattr(node.embedder, "state")
        }

    def plot(
        self,
        address: Address | str | None = None,
        detail: bool = False,
        out: str | Path | None = None,
    ) -> str:
        from json2vec.architecture.plot import plot

        return plot(module=self, address=address, detail=detail, out=out)

    @beartype
    def forward(self, inputs: TensorDict[Address, TensorFieldBase]) -> list[Prediction]:
        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []

        for address in self.hyperparameters.requests.keys():
            if address in self.hyperparameters.target:
                continue

            tensorfield: TensorFieldBase = inputs[address]
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

        for address in self.hyperparameters.requests.keys():

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
            raise ValueError("optimizer must be passed to JSON2Vec before fitting")

        if isinstance(self.optimizer, torch.optim.Optimizer):
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self)

        scheduler = self.scheduler(self, optimizer) if callable(self.scheduler) else self.scheduler

        if scheduler is None:
            return optimizer

        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def on_load_checkpoint(self, checkpoint):
        logger.bind(component="checkpoint").info("loading hyperparameters from checkpoint payload")
        if getattr(self, "hyperparameters", None) is None:
            self.hyperparameters = self._hyperparameters_from_checkpoint(checkpoint)

        if getattr(self, "hyperparameters", None) is None:
            raise ValueError("missing hyperparameters in checkpoint and constructor")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyperparameters"] = self.hyperparameters.model_dump(mode="python")

    @classmethod
    def _hyperparameters_from_checkpoint(cls, checkpoint: dict[str, Any]) -> Hyperparameters:
        if "hyperparameters" in checkpoint:
            return Hyperparameters.model_validate(cls._migrate_hyperparameters_payload(checkpoint["hyperparameters"]))

        if "hyper_parameters" in checkpoint:
            payload = checkpoint["hyper_parameters"]
            if isinstance(payload, dict) and "hyperparameters" in payload:
                return Hyperparameters.model_validate(cls._migrate_hyperparameters_payload(payload["hyperparameters"]))

            return Hyperparameters.model_validate(cls._migrate_hyperparameters_payload(payload))

        if "session" in checkpoint:
            payload = checkpoint["session"]
            if isinstance(payload, dict) and isinstance(payload.get("structure"), dict):
                migrated = dict(payload["structure"])
                migrated.pop("name", None)
                migrated.pop("type", None)
                for key in ("target", "embed", "p_target", "p_mask"):
                    if key in payload:
                        migrated[key] = payload[key]
                if "target" not in migrated and "pruned" in payload:
                    migrated["target"] = payload["pruned"]
                if "p_target" not in migrated and "p_prune" in payload:
                    migrated["p_target"] = payload["p_prune"]
                return Hyperparameters.model_validate(cls._migrate_hyperparameters_payload(migrated))

            return Hyperparameters.model_validate(cls._migrate_hyperparameters_payload(payload))

        raise ValueError("missing hyperparameters in checkpoint")

    @staticmethod
    def _migrate_hyperparameters_payload(payload: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(payload)
        if "target" not in migrated and "pruned" in migrated:
            migrated["target"] = migrated.pop("pruned")
        if "p_target" not in migrated and "p_prune" in migrated:
            migrated["p_target"] = migrated.pop("p_prune")
        else:
            migrated.pop("p_prune", None)

        fields = migrated.get("fields")
        if isinstance(fields, dict):
            fields = dict(fields)
            for key in ("dropout", "p_mask", "p_target"):
                if key in migrated:
                    value = migrated.pop(key)
                    if fields.get(key) is None:
                        fields[key] = value
            migrated["fields"] = fields

        def migrate_array(node: Any) -> Any:
            if not isinstance(node, dict):
                return node

            node = dict(node)
            if node.get("type") == "context":
                node["type"] = "array"
            if "max_length" not in node and "context_size" in node:
                node["max_length"] = node.pop("context_size")
            if "fields" in node:
                node["fields"] = [migrate_array(child) for child in node["fields"]]
            return node

        if "fields" in migrated:
            migrated["fields"] = migrate_array(migrated["fields"])

        return migrated

    @classmethod
    def _load_checkpoint(cls, checkpoint: str) -> dict[str, Any]:
        parsed = urlparse(checkpoint)
        if parsed.scheme == "s3":
            fs = pafs.S3FileSystem()  # type: ignore[attr-defined]
            path = f"{parsed.netloc}{parsed.path}"
            with fs.open_input_file(path) as handle:
                return torch.load(handle, weights_only=False, map_location="cpu")

        return torch.load(checkpoint, weights_only=False, map_location="cpu")

    def _load_compatible_state_dict(self, state_dict: dict[str, Any]) -> None:
        current = self.state_dict()
        compatible: dict[str, Any] = {}
        skipped: list[str] = []

        for key, value in state_dict.items():
            if key not in current:
                skipped.append(key)
                continue

            current_value = current[key]
            if isinstance(current_value, torch.Tensor) and isinstance(value, torch.Tensor):
                if current_value.shape != value.shape:
                    skipped.append(key)
                    continue
            elif type(current_value) is not type(value):
                skipped.append(key)
                continue

            compatible[key] = value

        self.load_state_dict(state_dict=compatible, strict=False)

        if skipped:
            logger.bind(
                component="checkpoint",
                skipped=len(skipped),
                keys=skipped,
            ).warning("skipped checkpoint parameters with incompatible shapes")

    @classmethod
    def get_or_create(
        cls,
        hyperparameters: Hyperparameters | None = None,
        checkpoint: str | None = None,
        *,
        batch_size: int = 1,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
    ) -> Self:

        if checkpoint is None:
            logger.bind(component="model_factory").info("creating new JSON2Vec model")
            if hyperparameters is None:
                raise ValueError("hyperparameters are required when checkpoint is not provided")

            model: "JSON2Vec" = cls(
                hyperparameters=hyperparameters,
                batch_size=batch_size,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            return model

        else:
            logger.bind(component="model_factory", checkpoint=checkpoint).info("loading JSON2Vec model from checkpoint")
            state = cls._load_checkpoint(checkpoint)

            model: "JSON2Vec" = cls(
                hyperparameters=hyperparameters or cls._hyperparameters_from_checkpoint(state),
                batch_size=batch_size,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            model._load_compatible_state_dict(state_dict=state["state_dict"])
            logger.bind(component="model_factory", checkpoint=checkpoint).info("restored model state from checkpoint")

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
        batch: EncodedBatch,
    ) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:
        was_training = self.training
        inputs = encode(
            batch=batch,
            hyperparameters=self.hyperparameters,
            strata=Strata.predict,
            state=self.state,
        )

        self.eval()
        try:
            with torch.inference_mode():
                predictions = self(inputs)
        finally:
            if was_training:
                self.train()

        return self.write(predictions)

    def predict(self, batch: EncodedBatch) -> dict[Address, dict[str, Any]]:
        supervised, _ = self.evaluate(batch=batch)
        return supervised

    def embed(self, batch: EncodedBatch) -> dict[Address, dict[str, Any]]:
        _, embeddings = self.evaluate(batch=batch)
        return embeddings

    training_step = partialmethod(step, strata=Strata.train)
    validation_step = partialmethod(step, strata=Strata.validate)
    test_step = partialmethod(step, strata=Strata.test)
    predict_step = partialmethod(step, strata=Strata.predict)
