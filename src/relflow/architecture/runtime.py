"""Forward, loss, writing, and inference runtime for `relflow` models."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import torch
from einops import rearrange, reduce
from loguru import logger
from tensordict import TensorDict

from relflow.architecture.contracts import sanitize
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.execution import CompiledExecutionGraph, InputPlan
from relflow.architecture.node import NodeModule
from relflow.data.datasets.base import EncodedBatch, EncodedInput
from relflow.data.iterables import encode as encode_batch
from relflow.data.iterables import mask as apply_mask
from relflow.data.processors import Postprocessor, Preprocessor
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.pooling import Mean
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
        strata = Strata.normalize(strata)

        execution: CompiledExecutionGraph = module.execution_graph
        memory: dict[Address, torch.Tensor] = {}
        summary: dict[Address, torch.Tensor] = {}
        predictions: list[Prediction] = []

        for address in module.schema.active_requests.keys():
            tensorfield: TensorFieldBase = inputs[address]
            if address in module.schema.target:
                continue

            node_module = cast(NodeModule, module.nodes[address])
            embedder: EmbedderBase = node_module.embedder
            embedding: Parcel = embedder(tensorfield)
            memory[address] = embedding.payload
            summary[address] = embedding.payload

        for address in execution.encoder_order:
            if address not in execution.active_branches:
                continue

            node_module = cast(NodeModule, module.nodes[address])
            encoder: BranchEncoder = node_module.encoder
            payloads: list[torch.Tensor] = []
            for plan in execution.branch_inputs[address]:
                payload = ModelRuntime._activation(plan, memory=memory, summary=summary)
                if plan.reference_id is not None:
                    payload = ModelRuntime._reference_payload(
                        module=module,
                        node_module=node_module,
                        plan=plan,
                        payload=payload,
                    )
                payloads.append(payload)

            encoding = encoder(payloads)
            memory[address] = encoding.memory
            summary[address] = encoding.summary

            if address in module.schema.embed:
                predictions.append(
                    Prediction(
                        address=address,
                        payload=TensorDict(
                            {TensorKey.embedding: encoding.summary},
                            batch_size=encoding.summary.shape[0],
                        ),
                        batch_size=encoding.summary.shape[0],
                    )
                )

        for address in module.schema.active_requests.keys():
            field = inputs[address]
            if strata == Strata.predict:
                inferred = field.state.eq(Tokens.masked.value)
                should_decode = bool(inferred.any()) or address in module.schema.embed
            else:
                inferred = field.trainable.bool()
                should_decode = bool(inferred.any())
            if not should_decode:
                continue

            parcels: list[Parcel] = []
            for plan in execution.decoder_contexts[address]:
                payload = ModelRuntime._activation(plan, memory=memory, summary=summary)
                parcels.append(
                    Parcel(
                        payload=payload,
                        origin=plan.address,
                        destination=address,
                        batch_size=payload.shape[0],
                    )
                )

            node_module = cast(NodeModule, module.nodes[address])
            decoder: DecoderBase = node_module.decoder
            prediction = decoder(parcels, embed=address in module.schema.embed)
            prediction.payload[TensorKey.inferred] = inferred
            predictions.append(prediction)

        return predictions

    @staticmethod
    def _activation(
        plan: InputPlan,
        *,
        memory: dict[Address, torch.Tensor],
        summary: dict[Address, torch.Tensor],
    ) -> torch.Tensor:
        values = memory if plan.view == "memory" else summary
        try:
            return values[plan.address]
        except KeyError as error:
            raise RuntimeError(f"compiled {plan.view} for '{plan.address}' was unavailable during forward") from error

    @staticmethod
    def _reference_payload(
        *,
        module: "Model",
        node_module: NodeModule,
        plan: InputPlan,
        payload: torch.Tensor,
    ) -> torch.Tensor:
        if plan.reference_id is None:
            return payload

        reference = module.execution_graph.references[plan.reference_id]
        if reference.declaration.reduce is not None:
            payload = ModelRuntime._reduce_reference(
                module=module,
                node_module=node_module,
                reference_id=plan.reference_id,
                payload=payload,
            )

        consumer = module.schema.branches[reference.consumer]
        outer = [ancestor.length for ancestor in consumer.ancestors if getattr(ancestor, "type", None) == "branch"]
        actual = list(payload.shape[1 : 1 + len(outer)])
        if actual != outer:
            raise ValueError(
                f"reference '{plan.address}' cannot align with branch '{reference.consumer}': "
                f"expected outer coordinates {outer}, got {actual}"
            )

        batch = payload.shape[0]
        channel = payload.shape[-1]
        return payload.reshape(batch, *outer, -1, channel)

    @staticmethod
    def _reduce_reference(
        *,
        module: "Model",
        node_module: NodeModule,
        reference_id: tuple[Address, int],
        payload: torch.Tensor,
    ) -> torch.Tensor:
        plan = module.execution_graph.references[reference_id]
        reduction = plan.declaration.reduce
        if reduction is None or not plan.axes:
            return payload

        middle_count = payload.ndim - 2
        by_position = {axis.position: axis for axis in plan.axes}
        input_axes: list[str] = []
        output_axes: list[str] = []
        reduced_axes: list[str] = []
        sizes: dict[str, int] = {}

        for position in range(middle_count):
            name = f"d{position}"
            axis = by_position.get(position)
            if axis is None:
                input_axes.append(name)
                output_axes.append(name)
                sizes[name] = payload.shape[position + 1]
                continue

            output_name = f"{name}_out"
            reduce_name = f"{name}_reduce"
            input_axes.append(f"({output_name} {reduce_name})")
            output_axes.append(output_name)
            reduced_axes.append(reduce_name)
            sizes[output_name] = axis.size
            sizes[reduce_name] = axis.extent // axis.size

        input_pattern = " ".join(("batch", *input_axes, "channel"))
        reducer = reduction.reducer
        if isinstance(reducer, Mean) or isinstance(reducer, str):
            name = "mean" if isinstance(reducer, Mean) else reducer
            output_pattern = " ".join(("batch", *output_axes, "channel"))
            return reduce(
                payload,
                f"{input_pattern} -> {output_pattern}",
                reduction=name,
                **sizes,
            )

        kept = output_axes
        leading = "batch" if not kept else f"(batch {' '.join(kept)})"
        tokens = reduced_axes[0] if len(reduced_axes) == 1 else f"({' '.join(reduced_axes)})"
        folded = rearrange(
            payload,
            f"{input_pattern} -> {leading} {tokens} channel",
            **sizes,
        )
        module_reducer = node_module.reference_reducers[str(reference_id[1])]
        aggregated = module_reducer(folded).squeeze(-2)
        if not kept:
            return aggregated

        restore_sizes = {"batch": payload.shape[0], **{name: sizes[name] for name in kept}}
        return rearrange(
            aggregated,
            f"(batch {' '.join(kept)}) channel -> batch {' '.join(kept)} channel",
            **restore_sizes,
        )

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

        if resolved_preprocessor is not None:
            observations: EncodedBatch = []
            for request in cast(list[dict[str, Any]], batch):
                observations.extend(
                    resolved_preprocessor.outputs(
                        request,
                        strata=strata,
                        schema=module.schema,
                        encoding_context=module.interprocess_encoding_context,
                    )
                )

            batch = observations
        elif batch and isinstance(batch[0], dict):
            batch = [[request] for request in cast(list[dict[str, Any]], batch)]

        inputs = encode_batch(
            batch=cast(EncodedBatch, batch),
            schema=module.schema,
            strata=strata,
            interprocess_encoding_context=module.interprocess_encoding_context,
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
