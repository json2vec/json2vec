"""Forward, loss, writing, and inference runtime for JSON2Vec models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, TypeAlias, TypedDict, cast

import torch
from loguru import logger
from tensordict import TensorDict

from json2vec.architecture.encoder import ArrayEncoder
from json2vec.architecture.node import NodeModule
from json2vec.data.datasets.base import EncodedBatch
from json2vec.structs.enums import Metric, Strata, TensorKey
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

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


class Output(TypedDict):
    loss: NotRequired[torch.Tensor]
    predictions: NotRequired[list[Prediction]]


PreprocessFn: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
Postprocessor: TypeAlias = Callable[
    [dict[str, Any], dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]],
    tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]] | None,
]


@dataclass(frozen=True)
class EvaluationResult:
    """Typed inference result before public convenience methods split it."""

    predictions: dict[Address, dict[str, Any]]
    embeddings: dict[Address, dict[str, Any]]

    def as_tuple(self) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:
        return self.predictions, self.embeddings


class ModelRuntime:
    """Own runtime behavior that depends on an already-built model graph."""

    @staticmethod
    def forward(module: "Model", inputs: TensorDict[Address, TensorFieldBase]) -> list[Prediction]:
        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []

        for address in module.hyperparameters.active_requests.keys():
            tensorfield: TensorFieldBase = inputs[address]
            if address in module.hyperparameters.target:
                continue

            node_module = cast(NodeModule, module.nodes[address])
            embedder: EmbedderBase = node_module.embedder
            embedding: Parcel = embedder(tensorfield)
            if embedding.destination is None:
                raise ValueError(f"parcel from '{embedding.origin}' has no destination")
            processed[embedding.destination].append(embedding)
            outgoing[embedding.origin] = embedding

            if address in module.hyperparameters.embed:
                predictions.append(Embedding.from_parcel(embedding))

        for depth in reversed(module.hyperparameters.depthwise):
            for address in depth:
                if len(processed[address]) == 0:
                    continue

                node_module = cast(NodeModule, module.nodes[address])
                encoder: ArrayEncoder = node_module.encoder
                encoding: Parcel = encoder(processed[address])
                if encoding.destination is None:
                    raise ValueError(f"parcel from '{encoding.origin}' has no destination")
                processed[encoding.destination].append(encoding)
                outgoing[encoding.origin] = encoding

                if address in module.hyperparameters.embed:
                    predictions.append(Embedding.from_parcel(encoding))

        for address in module.hyperparameters.active_requests.keys():
            if (torch.any(inputs[address].trainable)) or (address in module.hyperparameters.target):
                heritage: list[Address] = module.hyperparameters.requests[address].heritage
                parcels: list[Parcel] = [
                    outgoing[address]
                    for address in heritage
                    if address not in module.hyperparameters.target and address in outgoing.keys()
                ]

                node_module = cast(NodeModule, module.nodes[address])
                decoder: DecoderBase = node_module.decoder
                predictions.append(decoder(parcels))

        return predictions

    @staticmethod
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
            logger.warning("no trainable fields in batch, returning zero loss")
            loss: torch.Tensor = torch.tensor(0.0, device=batch.device, requires_grad=True)
            return Output(loss=loss)

        loss: torch.Tensor = module.track((Metric.loss, strata), value=torch.stack(losses).sum())
        return Output(loss=loss)

    @staticmethod
    def write(
        module: "Model",
        predictions: list[Prediction],
    ) -> tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]]:
        supervised: dict[Address, dict[str, Any]] = {}
        embeddings: dict[Address, dict[str, Any]] = {}

        for prediction in predictions:
            if isinstance(prediction, Embedding):
                embeddings[prediction.address] = Prediction.serialize(
                    Prediction.squeeze(Embedding.write(prediction), preserve_first_dimension=True)
                )
                continue

            request: RequestBase = module.hyperparameters.requests[prediction.address]
            extension: Plugin = TENSORFIELDS[request.type]
            write_fn = cast(Callable[..., dict[TensorKey, Any] | None], getattr(extension, "write"))

            scribed: dict[TensorKey, Any] | None = write_fn(module=module, prediction=prediction)
            if scribed is not None:
                supervised[prediction.address] = Prediction.serialize(
                    Prediction.squeeze(scribed, preserve_first_dimension=True)
                )

        return supervised, embeddings

    @staticmethod
    def evaluate(
        module: "Model",
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: PreprocessFn | None = None,
        postprocess: Postprocessor | None = None,
    ) -> EvaluationResult:
        from json2vec.data.iterables import encode

        was_training = module.training
        raw_batch = batch

        if preprocess is not None:
            observations: EncodedBatch = []
            for request in cast(list[dict[str, Any]], batch):
                observation = preprocess(request)
                if not isinstance(observation, dict):
                    raise TypeError(f"preprocessor must return a dict object, got {type(observation).__name__}")

                observations.append([observation])

            batch = observations
        elif batch and isinstance(batch[0], dict):
            batch = [[request] for request in cast(list[dict[str, Any]], batch)]

        inputs = encode(
            batch=cast(EncodedBatch, batch),
            hyperparameters=module.hyperparameters,
            strata=Strata.predict,
            interprocess_encoding_context=module.interprocess_encoding_context,
        )

        module.eval()
        try:
            with torch.inference_mode():
                raw_predictions = module(inputs)
        finally:
            if was_training:
                module.train()

        supervised, embeddings = module.write(raw_predictions)

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

        return EvaluationResult(predictions=supervised, embeddings=embeddings)


step = ModelRuntime.step
