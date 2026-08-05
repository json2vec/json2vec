# ty: ignore[unknown-argument]
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
)

from relflow.data.nested import extract_mask_literals, pad
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    apply_mask_policies,
)
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


boolean: Plugin = Plugin(name="boolean")
boolean.callback(CounterUpdateCallback)
BOOLEAN_VALUES = (-1.0, 0.0, 1.0)


def _encode(value: Any) -> float:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Boolean values must be bool or None, received {type(value).__name__}")
    return 1.0 if bool(value) else -1.0


@boolean.register
class Request(RequestBase):
    """Boolean scalar tensorfield request without a vocabulary."""

    type: Literal["boolean"] = "boolean"
    threshold: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.5)] = 0.5


class BooleanCounter(Counter):
    """Count fixed Boolean content values as false/true classes."""

    @torch.no_grad()
    def observe(self, values: torch.Tensor) -> torch.Tensor:
        return super().observe(values.gt(0).to(dtype=torch.int64))


@boolean.register
@tensorclass
class TensorField(TensorFieldBase):
    content: torch.Tensor
    state: torch.Tensor
    trainable: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    def new(
        cls,
        values: list,
        address: Address,
        schema: Schema,
        strata: Strata,
    ) -> TensorFieldBase:
        leading_shape = (len(values), *schema.shapes[address])
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )
        data, states = pad(
            nested=values,
            shape=leading_shape,
            dtype=np.float32,
            pad_value=0.0,
            overflows=schema.overflows(address),
            address=address,
            encode=_encode,
        )
        literal_data, _ = pad(
            nested=literal_masks,
            shape=leading_shape,
            dtype=bool,
            pad_value=False,
            overflows=schema.overflows(address),
            address=address,
        )

        literal_mask = torch.tensor(literal_data, dtype=torch.bool)
        content = torch.tensor(data, dtype=torch.float32).masked_fill(literal_mask, 0.0)
        state = torch.tensor(states, dtype=torch.int64).masked_fill(literal_mask, Tokens.masked.value)
        return cls(
            content=content,
            state=state,
            trainable=torch.zeros_like(state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        if cache_targets and TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()
        if cache_targets and TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_fill(selected, Tokens.masked.value)
        self.content = self.content.masked_fill(selected, 0.0)
        if trainable:
            self.trainable |= selected

    def mask(self, p_mask: float = 0.0, **kwargs: Any):
        apply_mask_policies(self, p_mask=p_mask, **kwargs)

    def target(self, p_prune: float = 1.0):
        apply_mask_policies(self, p_prune=p_prune)

    @classmethod
    def empty(cls, batch_size: int, address: Address, schema: Schema):
        shape = (batch_size, *schema.shapes[address])
        state = torch.full(shape, Tokens.masked.value, dtype=torch.int64)
        return cls(
            content=torch.zeros(shape, dtype=torch.float32),
            state=state,
            trainable=torch.zeros_like(state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
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
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        content = self.content[inputs.content.to(dtype=torch.int64).add(1)]
        embeddings = self.state(inputs.state) + content

        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=inputs.batch_size[0],
        )


@boolean.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        request: Request = schema.requests[address]
        self.state = torch.nn.Linear(schema.d_model, len(Tokens))
        self.content = torch.nn.Linear(schema.d_model, 1)
        self.metrics = torch.nn.ModuleDict(
            {
                f"{strata.value}_metrics": torch.nn.ModuleDict(
                    {
                        Metric.accuracy.value: BinaryAccuracy(threshold=request.threshold),
                        Metric.precision.value: BinaryPrecision(threshold=request.threshold),
                        Metric.recall.value: BinaryRecall(threshold=request.threshold),
                        Metric.auc.value: BinaryAUROC(),
                    }
                )
                for strata in (Strata.train, Strata.validate, Strata.test)
            }
        )

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
    metrics = decoder.metrics[f"{strata.value}_metrics"]
    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        # Every rank must log the same stateful metrics even when this rank's
        # batch has no valued targets. The metrics remain unchanged here.
        for metric_name, metric in metrics.items():
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
    for metric_name, metric in metrics.items():
        metric.update(probabilities, targets)
        module.track((address, strata, metric_name, TensorKey.content), value=metric)
    return total


@boolean.register
def write(module: Model, prediction: Prediction):
    request: Request = module.schema.requests[prediction.address]
    state_logits = prediction.payload[TensorKey.state]
    state_distribution = state_logits.softmax(dim=-1).detach().float().cpu().numpy()
    state_payload = {token.name: state_distribution[..., token.value] for token in Tokens}

    probabilities = prediction.payload[TensorKey.content].sigmoid().squeeze(-1).detach().float().cpu().numpy()
    return {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: {
            TensorKey.value.name: probabilities >= request.threshold,
            TensorKey.probability.name: probabilities,
        },
    }
