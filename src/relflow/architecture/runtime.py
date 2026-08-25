"""Forward, loss, writing, and inference runtime for `relflow` models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import torch
from loguru import logger
from tensordict import TensorDict

from relflow.architecture.contracts import sanitize
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.node import NodeModule
from relflow.data.datasets.base import EncodedBatch, EncodedInput
from relflow.data.iterables import encode as encode_batch
from relflow.data.iterables import mask as apply_mask
from relflow.data.processors import Postprocessor, Preprocessor
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class Output(TypedDict):
    loss: NotRequired[torch.Tensor]
    predictions: NotRequired[list[Prediction]]


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

        for address in module.schema.active_requests.keys():
            tensorfield: TensorFieldBase = inputs[address]
            if address in module.schema.target:
                continue

            node_module = cast(NodeModule, module.nodes[address])
            embedder: EmbedderBase = node_module.embedder
            embedding: Parcel = embedder(tensorfield)
            if embedding.destination is None:
                raise ValueError(f"parcel from '{embedding.origin}' has no destination")
            processed[embedding.destination].append(embedding)
            outgoing[embedding.origin] = embedding

        for depth in reversed(module.schema.depthwise):
            for address in depth:
                if len(processed[address]) == 0:
                    continue

                node_module = cast(NodeModule, module.nodes[address])
                encoder: BranchEncoder = node_module.encoder
                encoding: Parcel = encoder(processed[address])
                if encoding.destination is None:
                    raise ValueError(f"parcel from '{encoding.origin}' has no destination")
                processed[encoding.destination].append(encoding)
                outgoing[encoding.origin] = encoding

                if address in module.schema.embed:
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

        for address in module.schema.active_requests.keys():
            has_masked_input = inputs[address].state.eq(Tokens.masked.value).any()
            if (
                torch.any(inputs[address].trainable)
                or (strata == Strata.predict and has_masked_input)
                or (address in module.schema.target)
                or (address in module.schema.embed)
            ):
                heritage: list[Address] = module.schema.requests[address].heritage
                parcels: list[Parcel] = [
                    outgoing[address]
                    for address in heritage
                    if address not in module.schema.target and address in outgoing.keys()
                ]

                node_module = cast(NodeModule, module.nodes[address])
                decoder: DecoderBase = node_module.decoder
                prediction = decoder(parcels, embed=address in module.schema.embed)
                if Strata.normalize(strata) == Strata.predict:
                    prediction.payload[TensorKey.inferred] = inputs[address].state.eq(Tokens.masked.value)
                else:
                    prediction.payload[TensorKey.inferred] = inputs[address].trainable.bool()
                predictions.append(prediction)

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
            if prediction.address not in module.schema.requests:
                continue

            if set(prediction.payload.keys()) <= {TensorKey.embedding, TensorKey.inferred}:
                continue

            address: Address = prediction.address
            request: RequestBase = module.schema.requests[address]
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

            if prediction.address in module.schema.requests:
                request: RequestBase = module.schema.requests[prediction.address]
                extension: Plugin = TENSORFIELDS[request.type]
                write_fn = cast(Callable[..., dict[TensorKey, Any] | None], getattr(extension, "write"))

                written: dict[TensorKey, Any] | None = write_fn(module=module, prediction=prediction)
                if written is not None:
                    scribed.update(written)

            has_decoded_output = TensorKey.state.name in scribed or TensorKey.content.name in scribed
            if has_decoded_output and TensorKey.inferred in prediction.payload.keys():
                scribed[TensorKey.inferred.name] = prediction.payload[TensorKey.inferred].detach().cpu()

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
        mask: bool = True,
    ) -> EncodedInput:
        strata = Strata.normalize(strata)
        resolved_preprocessor = Preprocessor.normalize(preprocess)
        encoding_context = module.interprocess_encoding_context

        if resolved_preprocessor is not None:
            observations: EncodedBatch = []
            for request in cast(list[dict[str, Any]], batch):
                observations.extend(
                    resolved_preprocessor.outputs(
                        request,
                        strata=strata,
                        schema=module.schema,
                        encoding_context=encoding_context,
                    )
                )

            batch = observations
        elif batch and isinstance(batch[0], dict):
            batch = [[request] for request in cast(list[dict[str, Any]], batch)]

        inputs = encode_batch(
            batch=cast(EncodedBatch, batch),
            schema=module.schema,
            strata=strata,
            interprocess_encoding_context=encoding_context,
            defer_target_masking=True,
        )
        if mask:
            return next(apply_mask([inputs], module.schema, strata=strata))

        return inputs

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

        resolved_postprocessor = Postprocessor.normalize(postprocess)
        if resolved_postprocessor is not None:
            processed = resolved_postprocessor.run(
                predictions,
                available={
                    "batch": raw_batch,
                    "observations": inputs[TensorKey.metadata],
                    "input": inputs,
                    "metadata": inputs[TensorKey.metadata],
                },
            )

            if processed is not None:
                predictions = dict(processed)

        return predictions


step = ModelRuntime.step
