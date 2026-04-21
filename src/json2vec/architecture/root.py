import math
import traceback
from collections import defaultdict
from functools import cache, partialmethod, wraps
from typing import Any, NotRequired, Self, TypedDict

import lightning.pytorch as lit
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict

from json2vec.architecture.encoder import ContextEncoder
from json2vec.architecture.node import NodeModule
from json2vec.data.datasets import dataloader, mock
from json2vec.structs.enums import Metric, Strata, TensorKey
from json2vec.structs.experiment import Session
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


@beartype
def step(
    module: "JSON2Vec",
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
        request: RequestBase = module.session.structure.requests[address]
        extension: Plugin = TENSORFIELDS[request.type]

        loss: torch.Tensor = extension.loss(module=module, prediction=prediction, batch=batch[address], strata=strata)
        losses.append(loss * torch.tensor(request.weight))

    if len(losses) == 0:
        # under idealistic circumstances this would never happen.
        # but with small mask rates, batch sizes, and flat input data it is possible
        logger.warning("no trainable fields in batch, returning zero loss")
        loss: torch.Tensor = torch.tensor(0.0, device=batch.device, requires_grad=True)
        return Output(loss=loss, predictions=[])

    loss: torch.Tensor = module.track((Metric.loss, strata), value=torch.stack(losses).sum())

    return Output(loss=loss, predictions=predictions)


def compile(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        try:
            import thunder

            model = thunder.compile(model)
            logger.info("successfully compiled module with thunder")
        except Exception:
            traceback.print_exc()
            logger.info("[thunder] Returning uncompiled model instead.")
        return model

    return wrapper


@cache
def groupname(names: tuple[str, ...]) -> str:
    assert len(names) > 1

    group, *keys = tuple(map(lambda x: x.replace("/", ":").lower(), names))

    key: str = ":".join(list(keys))

    return f"{group}/{key}"


class JSON2Vec(lit.LightningModule):
    @beartype
    def __init__(self, session: Session):

        super().__init__()

        self.session: Session = session

        self.nodes: torch.nn.ModuleDict[str, NodeModule] = torch.nn.ModuleDict()

        for address in self.session.structure.requests | self.session.structure.contexts:
            self.nodes[address] = NodeModule(structure=self.session.structure, address=address)

        self.example_input_array = mock(structure=session.structure)

        logger.bind(
            component="model",
            session=self.session.name,
            structure=self.session.structure.name,
            requests=len(self.session.structure.requests),
            contexts=len(self.session.structure.contexts),
            outputs=len(self.session.output),
        ).info("initialized JSON2Vec module")

    def track(self, names: tuple[str, ...], /, value: torch.Tensor) -> torch.Tensor:
        self.log(
            name=groupname(names),
            value=value,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.session.structure.batch_size,
        )

        return value

    @property
    def state(self) -> dict[Address, Any]:
        return {
            address: node.embedder.state
            for address, node in self.nodes.items()
            if hasattr(node, "embedder") and hasattr(node.embedder, "state")
        }

    @beartype
    def forward(self, inputs: TensorDict[Address, TensorFieldBase]) -> list[Prediction]:
        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []

        for address in self.session.structure.requests.keys():
            if address in self.session.pruned:
                continue

            tensorfield: TensorFieldBase = inputs[address]
            embedder: EmbedderBase = self.nodes[address].embedder
            embedding: Parcel = embedder(tensorfield)
            processed[embedding.destination].append(embedding)
            outgoing[embedding.origin] = embedding

            if address in self.session.output:
                predictions.append(Embedding.from_parcel(embedding))

        # DAG traversal from leaves to root
        for depth in reversed(self.session.structure.depthwise):
            # these are order-independent within the same depth
            for address in depth:

                if len(processed[address]) == 0:
                    continue

                encoder: ContextEncoder = self.nodes[address].encoder
                encoding: Parcel = encoder(processed[address])
                processed[encoding.destination].append(encoding)
                outgoing[encoding.origin] = encoding

                if address in self.session.output:
                    predictions.append(Embedding.from_parcel(encoding))

        for address in self.session.structure.requests.keys():

            if (torch.any(inputs[address].trainable)) or (address in self.session.pruned):

                heritage: list[Address] = self.session.structure.requests[address].heritage
                parcels: list[Parcel] = [
                    outgoing[address] for address in heritage
                    if address not in self.session.pruned and address in outgoing.keys()
                ]

                decoder: DecoderBase = self.nodes[address].decoder
                predictions.append(decoder(parcels))

        return predictions

    @beartype
    def configure_optimizers(self):

        if self.session.learning_rate is None:
            raise ValueError("learning_rate must be defined for optimizer configuration")

        class GroupedParameter(TypedDict):
            params: list[torch.nn.Parameter]
            weight_decay: float

        params: dict[str, GroupedParameter] = dict(
            with_decay = GroupedParameter(params=[], weight_decay=self.session.weight_decay),
            no_decay = GroupedParameter(params=[], weight_decay=0.0),
        )

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            if name.endswith(".bias") or parameter.ndim <= 1 or "norm" in name.lower():
                params["no_decay"]["params"].append(parameter)
            else:
                params["with_decay"]["params"].append(parameter)


        optimizer = torch.optim.AdamW(params=list(params.values()), lr=self.session.learning_rate, betas=(0.9, 0.95))
        trainable_parameters: int = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        logger.bind(
            component="optimizer",
            session=self.session.name,
            learning_rate=self.session.learning_rate,
            weight_decay=self.session.weight_decay,
            trainable_parameters=trainable_parameters,
            warmup_ratio=self.session.warmup_ratio,
            min_lr_ratio=self.session.min_lr_ratio,
        ).info("configured AdamW optimizer")

        total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)

        if total <= 0:
            return optimizer

        warmup = max(1, int(total * self.session.warmup_ratio))
        min_lr_ratio = self.session.min_lr_ratio

        def schedule(step: int) -> float:

            if step < warmup:
                return float(step + 1) / float(warmup)

            ratio = float(step - warmup) / float(max(1, total - warmup))

            progress = min(1.0, ratio)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

        return dict(
            optimizer= optimizer,
            lr_scheduler=dict(
                scheduler = scheduler,
                interval = "step",
                frequency = 1,
            ),
        )

    def on_save_checkpoint(self, checkpoint):
        logger.bind(component="checkpoint", session=self.session.name).info("serializing session")
        checkpoint["session"] = self.session.model_dump()

    def on_load_checkpoint(self, checkpoint):
        logger.bind(component="checkpoint").info("loading session from checkpoint payload")
        if "session" in checkpoint and getattr(self, "session", None) is None:
            self.session = Session.model_validate(checkpoint["session"])

        if getattr(self, "session", None) is None:
            raise ValueError("missing session in checkpoint and constructor")

    @classmethod
    def get_or_create(
        cls,
        session: Session|None = None,
        checkpoint: str | None = None,
    ) -> Self:

        if checkpoint is None:
            logger.bind(component="model_factory").info("creating new JSON2Vec model")
            if session is None:
                raise ValueError("session is required when checkpoint is not provided")

            model: "JSON2Vec" = cls(session=session)

            return model

        else:
            logger.bind(component="model_factory", checkpoint=checkpoint).info("loading JSON2Vec model from checkpoint")
            state = torch.load(checkpoint, weights_only=False, map_location="cpu")

            model: "JSON2Vec" = cls(
                session=session or Session.model_validate(state["session"]),
            )

            model.load_state_dict(state_dict=state["state_dict"], strict=False)
            logger.bind(component="model_factory", checkpoint=checkpoint).info("restored model state from checkpoint")

            return model

    def write(self, predictions: list[Prediction]) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:

        supervised: dict[Address, dict[str, Any]] = {}
        embeddings: dict[Address, dict[str, Any]] = {}

        for prediction in predictions:

            if isinstance(prediction, Embedding):

                embeddings[prediction.address] = Embedding.write(prediction)

                continue

            request: RequestBase = self.session.structure.requests[prediction.address]

            extension: Plugin = TENSORFIELDS[request.type]

            scribed: dict[TensorKey, Any]|None = extension.write(module=self, prediction=prediction)

            if scribed is not None:
                supervised[prediction.address] = Prediction.serialize(scribed)



        return supervised, embeddings

    training_step = partialmethod(step, strata=Strata.train)
    validation_step = partialmethod(step, strata=Strata.validate)
    test_step = partialmethod(step, strata=Strata.test)
    predict_step = partialmethod(step, strata=Strata.predict)

    train_dataloader = partialmethod(dataloader, strata=Strata.train)
    val_dataloader = partialmethod(dataloader, strata=Strata.validate)
    test_dataloader = partialmethod(dataloader, strata=Strata.test)
    predict_dataloader = partialmethod(dataloader, strata=Strata.predict)
