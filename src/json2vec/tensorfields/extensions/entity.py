# ty: ignore[unknown-argument]
from __future__ import annotations

import math
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.data.processing import extract_mask_literals, pad
from json2vec.structs.enums import Metric, Strata, TensorKey, Tokens
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    apply_mask_policies,
)

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Schema


entity: Plugin = Plugin(name="entity")


def _local_reindex(data: np.ndarray, states: np.ndarray) -> np.ndarray:
    tokens = np.zeros_like(states, dtype=np.int64)

    for observation_index in range(data.shape[0]):
        vocab: dict[Hashable, int] = {}
        flat_values = data[observation_index].reshape(-1)
        flat_states = states[observation_index].reshape(-1)
        flat_tokens = tokens[observation_index].reshape(-1)

        for index, state in enumerate(flat_states):
            if state != Tokens.valued.value:
                continue

            value: Any = flat_values[index]
            if not isinstance(value, Hashable):
                raise TypeError(f"entity values must be hashable, got {type(value).__name__}")

            local_id = vocab.setdefault(value, len(vocab))
            flat_tokens[index] = local_id

    return tokens


@entity.register
class Request(RequestBase):
    """Per-observation entity tensorfield request for local identity matching."""

    type: Literal["entity"] = "entity"
    topk: list[int] | None = None

    @pydantic.model_validator(mode="after")
    def check_topk(self):
        if self.topk is None:
            self.topk = []

        for topk in self.topk:
            if not isinstance(topk, int):
                raise ValueError("topk values must be integers")

            if topk <= 0:
                raise ValueError("topk values must be positive")

            if topk == 1:
                raise ValueError("topk values must not be 1")

        return self

    def post_bind_validate(self):
        max_slots: int = math.prod(self.shape)
        if max_slots <= 1:
            raise ValueError(
                f"entity field at '{self.address}' requires at least 2 elements per observation, "
                f"but configured count is {max_slots}"
            )

        for topk in self.topk or []:
            if topk >= max_slots:
                raise ValueError("topk values must be less than the entity slot count")


@entity.register
@tensorclass
class TensorField(TensorFieldBase):
    state: torch.Tensor
    content: torch.Tensor
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
        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )

        data, states = pad(
            nested=values,
            shape=leading_shape,
            dtype=object,
            pad_value=None,
            overflows=schema.overflows(address),
            address=address,
        )
        literal_data, _ = pad(
            nested=literal_masks,
            shape=leading_shape,
            dtype=bool,
            pad_value=False,
            overflows=schema.overflows(address),
            address=address,
        )

        try:
            tokens = _local_reindex(data=data, states=states)
        except TypeError as error:
            raise ValueError(f"entity field at '{address}' only accepts hashable scalar values") from error

        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = torch.tensor(states, dtype=torch.int64).masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.tensor(tokens, dtype=torch.int64)
        content = content.masked_fill(literal_mask_tensor, 0)

        return cls(
            state=state_tensor,
            content=content,
            trainable=torch.zeros_like(input=state_tensor, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        mask_token = torch.full_like(input=self.state, fill_value=Tokens.masked.value)

        if cache_targets and TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if cache_targets and TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_scatter(selected, mask_token)
        self.content = self.content.masked_fill(selected, 0)

        if trainable:
            self.trainable |= selected

    def mask(self, p_mask: float = 0.0, **kwargs: Any):
        apply_mask_policies(self, p_mask=p_mask, **kwargs)

    def target(self, p_prune: float = 1.0):
        apply_mask_policies(self, p_prune=p_prune)

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        schema: Schema,
    ):
        shape: tuple[int, ...] = (batch_size, *schema.shapes[address])

        state = torch.full(shape, Tokens.masked)
        content = torch.zeros(shape, dtype=torch.int64)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@entity.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.max_slots: int = math.prod(schema.shapes[address])
        self.origin: Address = address
        self.destination: Address = schema.requests[address].parent.address
        self.n_embeddings: int = self.max_slots + len(Tokens)

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.content.name: torch.nn.Embedding(
                    num_embeddings=self.max_slots,
                    embedding_dim=schema.d_model,
                ),
            }
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if valued.any() and (content.masked_select(valued) >= self.max_slots).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds bucket space of {self.max_slots}")

        safe_content = content.masked_fill(~valued, 0)
        embeddings: torch.Tensor = (
            self.embeddings[TensorKey.state.name](state)
            + self.embeddings[TensorKey.content.name](safe_content) * valued.unsqueeze(-1)
        ).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@entity.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.max_slots: int = math.prod(schema.shapes[address])
        self.state_linear = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))
        self.projection = torch.nn.Linear(in_features=schema.d_model, out_features=self.max_slots)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.state_linear(pooled),
                TensorKey.content: self.projection(pooled),
            }
        )


@entity.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    N: int = batch.targets[TensorKey.state].numel()
    trainable = batch.trainable.reshape(N)
    state_inputs = prediction.payload[TensorKey.state].reshape(N, -1)
    state_targets = batch.targets[TensorKey.state].reshape(N)

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    inputs = prediction.payload[TensorKey.content].reshape(N, -1)
    targets = batch.targets[TensorKey.content].reshape(N)
    n_content_tokens = inputs.shape[-1]
    invalid = valued & targets.ge(n_content_tokens)
    if invalid.any():
        raise ValueError(f"Token in address {prediction.address} exceeds entity slot count")

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=(
            torch.nn.functional.cross_entropy(
                input=inputs,
                target=targets,
                reduction="none",
            )
            .masked_select(valued)
            .mean()
        ),
    )

    for topk in module.schema.requests[prediction.address].topk:
        if topk >= inputs.shape[1]:
            continue

        module.track(
            (prediction.address, strata, Metric.accuracy, f"top{topk}"),
            value=(
                inputs.topk(k=topk, dim=1)
                .indices.eq(targets.unsqueeze(1))
                .any(dim=1)
                .masked_select(valued)
                .float()
                .mean()
            ),
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=inputs.argmax(dim=1).eq(targets).masked_select(valued).float().mean(),
    )

    return loss


@entity.register
def write(module: Model, prediction: Prediction):
    return None
