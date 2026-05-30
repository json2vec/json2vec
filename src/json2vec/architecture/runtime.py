"""Forward, loss, writing, and inference runtime for JSON2Vec models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NotRequired, TypeAlias, TypedDict, cast

import torch
from loguru import logger
from tensordict import TensorDict

from json2vec.architecture.contracts import sanitize
from json2vec.architecture.encoder import ArrayEncoder
from json2vec.architecture.node import NodeModule
from json2vec.data.datasets.base import EncodedBatch, EncodedInput
from json2vec.data.iterables import encode
from json2vec.structs.enums import Metric, Strata, TensorKey
from json2vec.structs.packages import Parcel, Prediction
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


Preprocessor: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]
Postprocessor: TypeAlias = Callable[
    [dict[str, Any], dict[Address, dict[str, Any]]],
    dict[Address, dict[str, Any]] | None,
]


class ModelRuntime:
    """Own runtime behavior that depends on an already-built model graph."""

    @staticmethod
    def forward(
        module: "Model",
        inputs: TensorDict[Address, TensorFieldBase],
        *,
        strata: Strata | str,
        dataloader_idx: int = 0,
    ) -> list[Prediction]:
        sanitize(module, inputs, strata=strata, dataloader_idx=dataloader_idx)

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
                    predictions.append(
                        Prediction(
                            address=encoding.origin,
                            payload=TensorDict(
                                {TensorKey.embedding: encoding.payload},
                                batch_size=encoding.payload.shape[0],
                            ),
                            batch_size=encoding.payload.shape[0],
                        )
                    )

        for address in module.hyperparameters.active_requests.keys():
            if (
                torch.any(inputs[address].trainable)
                or (address in module.hyperparameters.target)
                or (address in module.hyperparameters.embed)
            ):
                heritage: list[Address] = module.hyperparameters.requests[address].heritage
                parcels: list[Parcel] = [
                    outgoing[address]
                    for address in heritage
                    if address not in module.hyperparameters.target and address in outgoing.keys()
                ]

                node_module = cast(NodeModule, module.nodes[address])
                decoder: DecoderBase = node_module.decoder
                predictions.append(decoder(parcels, embed=address in module.hyperparameters.embed))

        return predictions

    @staticmethod
    def step(
        module: "Model",
        batch: TensorDict[Address, TensorFieldBase],
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Strata,
    ) -> Output:
        predictions: list[Prediction] = module.forward(batch, strata=strata, dataloader_idx=dataloader_idx)

        if strata == Strata.predict:
            return Output(predictions=predictions)

        losses: list[torch.Tensor] = []

        for prediction in predictions:
            if prediction.address not in module.hyperparameters.requests:
                continue

            if set(prediction.payload.keys()) <= {TensorKey.embedding}:
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
    ) -> dict[Address, dict[str, Any]]:
        outputs: dict[Address, dict[str, Any]] = {}

        for prediction in predictions:
            scribed: dict[Any, Any] = {}

            if prediction.address in module.hyperparameters.requests:
                request: RequestBase = module.hyperparameters.requests[prediction.address]
                extension: Plugin = TENSORFIELDS[request.type]
                write_fn = cast(Callable[..., dict[TensorKey, Any] | None], getattr(extension, "write"))

                written: dict[TensorKey, Any] | None = write_fn(module=module, prediction=prediction)
                if written is not None:
                    scribed.update(written)

            if TensorKey.embedding in prediction.payload.keys():
                values = prediction.payload[TensorKey.embedding].detach().float()
                embedding = torch.nn.functional.normalize(values, p=2, dim=-1, eps=1e-12)
                scribed[TensorKey.embedding.name] = embedding.cpu().tolist()

            if scribed:
                outputs[prediction.address] = Prediction.serialize(
                    Prediction.squeeze(scribed, preserve_first_dimension=True)
                )

        return outputs

    @staticmethod
    def encode(
        module: "Model",
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: Preprocessor | None = None,
        strata: Strata | str = Strata.predict,
    ) -> EncodedInput:
        strata = Strata.normalize(strata)

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

        return encode(
            batch=cast(EncodedBatch, batch),
            hyperparameters=module.hyperparameters,
            strata=strata,
            interprocess_encoding_context=module.interprocess_encoding_context,
        )

    @staticmethod
    def predict(
        module: "Model",
        batch: EncodedBatch | list[dict[str, Any]],
        preprocess: Preprocessor | None = None,
        postprocess: Postprocessor | None = None,
    ) -> dict[Address, dict[str, Any]]:
        was_training = module.training
        raw_batch = batch
        inputs = ModelRuntime.encode(module=module, batch=batch, preprocess=preprocess, strata=Strata.predict)

        module.eval()
        try:
            with torch.inference_mode():
                raw_predictions = module(inputs, strata=Strata.predict)
        finally:
            if was_training:
                module.train()

        predictions = module.write(raw_predictions)

        if postprocess is not None:
            context = {
                "batch": raw_batch,
                "observations": inputs[TensorKey.metadata],
                "input": inputs,
                TensorKey.metadata: inputs[TensorKey.metadata],
            }
            processed = postprocess(context, predictions)

            if processed is not None:
                predictions = processed

        return predictions


step = ModelRuntime.step
