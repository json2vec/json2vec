# ty: ignore[unknown-argument]
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import array, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


boolean: Plugin = Plugin(name="boolean", types=(bool,))
boolean.callback(CounterUpdateCallback)
BOOLEAN_VALUES = (-1.0, 0.0, 1.0)
Threshold = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]
Thresholds = Annotated[list[Threshold], pydantic.Field(min_length=1)]


@boolean.register
class Request(RequestBase):
    """Boolean scalar tensorfield request without a vocabulary."""

    type: Literal["boolean"] = "boolean"
    threshold: Threshold | Thresholds = 0.5

    @classmethod
    def counts(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[bool, int]:
        """Return observed training counts for false and true values."""
        from relflow.architecture.root import Model

        if not isinstance(model, Model):
            raise TypeError(f"Boolean.counts model must be a Model, got {type(model).__name__}")

        address = Address(str(address))
        if address not in model.nodes:
            raise KeyError(f"no field at address {str(address)!r}")

        embedder = getattr(model.nodes[address], "embedder", None)
        if not isinstance(embedder, Embedder):
            raise TypeError(f"address {str(address)!r} is not a Boolean field (got {type(embedder).__name__})")

        counts = embedder.counters[TensorKey.content.name].counts.detach().cpu().sub(1).clamp_min(0).tolist()
        return {False: int(counts[0]), True: int(counts[1])}

    @pydantic.field_validator("threshold", mode="after")
    @classmethod
    def deduplicate_thresholds(cls, value: float | list[float]) -> float | list[float]:
        if isinstance(value, list):
            return list(dict.fromkeys(value))
        return value

    @property
    def thresholds(self) -> tuple[float, ...]:
        if isinstance(self.threshold, list):
            return tuple(self.threshold)
        return (self.threshold,)


class BooleanCounter(Counter):
    """Count fixed Boolean content values as false/true classes."""

    @torch.no_grad()
    def observe(self, values: torch.Tensor) -> torch.Tensor:
        return super().observe(values.gt(0).to(dtype=torch.int64))


@boolean.register
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Count complete pristine Boolean state and content."""

    if not learn:
        return None
    values = pc.cast(field.values, pa.int64(), safe=True)
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    content = torch.from_numpy(values.to_numpy(zero_copy_only=False).copy())
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(content, 2),
        },
        batch_size=[],
    )


@boolean.register
@tensorclass
class TensorField(TensorFieldBase):
    content: torch.Tensor
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorFieldBase:
        def encode(field: RaggedField) -> torch.Tensor:
            encoded = pc.if_else(field.values, pa.scalar(1.0, pa.float32()), pa.scalar(-1.0, pa.float32()))
            if isinstance(encoded, pa.ChunkedArray):
                encoded = encoded.combine_chunks()
            return torch.from_numpy(field.place(encoded.to_numpy(zero_copy_only=False), fill=0.0))

        state = torch.from_numpy(input.dense)
        target_state = torch.from_numpy(target.dense)
        return cls(
            content=encode(input),
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: target_state,
                    TensorKey.content: encode(target),
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@boolean.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        request: Request = schema.requests[address]
        self.origin = address
        self.destination = request.parent.address
        self.state = torch.nn.Embedding(len(Tokens), schema.d_model)
        values = torch.tensor(BOOLEAN_VALUES, dtype=torch.float32).unsqueeze(-1)
        self.register_buffer("content", values.expand(len(BOOLEAN_VALUES), schema.d_model).clone())
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
                TensorKey.content.name: BooleanCounter(address=address, size=2),
            }
        )

        self.content: torch.Tensor

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        content = self.content[inputs.content.to(dtype=torch.int64).add(1)]
        embeddings = self.state(inputs.state) + content

        return Parcel(
            payload=embeddings,
            present=torch.ones(inputs.state.shape[0], dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=inputs.batch_size[0],
        )


@boolean.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine Boolean counts to model-owned resources."""

    if strata != Strata.train:
        raise ValueError(f"boolean learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = module.nodes[address].embedder
    embedder.counters[TensorKey.state.name].learn(observation[TensorKey.state])
    embedder.counters[TensorKey.content.name].learn(observation[TensorKey.content])


@boolean.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        request: Request = schema.requests[address]
        self.state = torch.nn.Linear(schema.d_model, len(Tokens))
        self.content = torch.nn.Linear(schema.d_model, 1)
        self.thresholds = request.thresholds
        self.metrics = torch.nn.ModuleDict(
            {
                f"{strata.value}_metrics": torch.nn.ModuleDict(
                    {
                        Metric.auc.value: BinaryAUROC(),
                        **{
                            f"threshold_{index}": torch.nn.ModuleDict(
                                {
                                    Metric.accuracy.value: BinaryAccuracy(threshold=threshold),
                                    Metric.precision.value: BinaryPrecision(threshold=threshold),
                                    Metric.recall.value: BinaryRecall(threshold=threshold),
                                    Metric.specificity.value: BinarySpecificity(threshold=threshold),
                                }
                            )
                            for index, threshold in enumerate(self.thresholds)
                        },
                    }
                )
                for strata in (Strata.train, Strata.validate, Strata.test)
            }
        )

    def content_metrics(self, strata: Strata) -> Iterator[tuple[str, TorchMetric]]:
        metrics = cast(torch.nn.ModuleDict, self.metrics[f"{strata.value}_metrics"])
        yield Metric.auc.value, cast(TorchMetric, metrics[Metric.auc.value])
        for index, threshold in enumerate(self.thresholds):
            threshold_metrics = cast(torch.nn.ModuleDict, metrics[f"threshold_{index}"])
            for name, metric in threshold_metrics.items():
                yield f"{name}@{threshold}", cast(TorchMetric, metric)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            {
                TensorKey.state: self.state(pooled),
                TensorKey.content: self.content(pooled),
            }
        )


@boolean.register
def loss(module: Model, prediction: Prediction, batch: TensorFieldBase, strata: Strata) -> torch.Tensor:
    address = prediction.address
    embedder: Embedder = module.nodes[address].embedder
    trainable = batch.trainable.reshape(-1)
    state_targets = batch.targets[TensorKey.state].reshape(-1)
    state_logits = prediction.payload[TensorKey.state].reshape(state_targets.numel(), -1)

    total = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=torch.nn.functional.cross_entropy(
            state_logits[trainable],
            state_targets[trainable],
            weight=cast(Counter, embedder.counters[TensorKey.state.name]).weight,
        ),
    )
    module.track(
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_logits[trainable].argmax(dim=-1).eq(state_targets[trainable]).float().mean(),
    )

    decoder: Decoder = module.nodes[address].decoder
    metrics = tuple(decoder.content_metrics(strata))
    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        # Every rank must log the same stateful metrics even when this rank's
        # batch has no valued targets. The metrics remain unchanged here.
        for metric_name, metric in metrics:
            module.track((address, strata, metric_name, TensorKey.content), value=metric)
        return total

    logits = prediction.payload[TensorKey.content].reshape(-1)[valued]
    targets = batch.targets[TensorKey.content].reshape(-1)[valued].gt(0).long()
    total += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=(
            torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
            * cast(BooleanCounter, embedder.counters[TensorKey.content.name]).weight[targets]
        ).mean(),
    )

    probabilities = logits.sigmoid()
    for metric_name, metric in metrics:
        metric.update(probabilities, targets)
        module.track((address, strata, metric_name, TensorKey.content), value=metric)
    return total


@boolean.register
def output(module: Model, address: Address) -> pa.StructType:
    content = pa.struct([pa.field(TensorKey.probability.name, pa.float32(), nullable=False)])
    return pa.struct([pa.field(TensorKey.content.name, content, nullable=False)])


@boolean.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content_type = datatype.field(TensorKey.content.name).type
    probabilities = array(prediction.payload[TensorKey.content].sigmoid(), pa.float32())
    content = struct({TensorKey.probability.name: probabilities}, content_type)
    return struct({TensorKey.content.name: content}, datatype)
