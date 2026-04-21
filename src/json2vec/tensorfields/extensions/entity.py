from __future__ import annotations

import math
from collections.abc import Hashable
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.data.processing import pad
from json2vec.structs.enums import Metric, Strata, TensorKey, Tokens
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Session, Structure


entity: Plugin = Plugin(name="entity")


def _local_reindex(data: np.ndarray, states: np.ndarray) -> np.ndarray:
    vocab: dict[Hashable, int] = {}
    tokens = np.zeros_like(states, dtype=np.int64)
    flat_values = data.reshape(-1)
    flat_states = states.reshape(-1)
    flat_tokens = tokens.reshape(-1)

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
    type: Literal["entity"]
    topk: Annotated[list[int], pydantic.Field(default_factory=list)]

    @pydantic.model_validator(mode="after")
    def check_topk(self):
        for topk in self.topk:
            if not isinstance(topk, int):
                raise ValueError("topk values must be integers")

            if topk <= 0:
                raise ValueError("topk values must be positive")

            if topk == 1:
                raise ValueError("topk values must not be 1")

        return self

    def post_bind_validate(self):
        root = self.path[0]
        per_observation_count: int = math.prod(self.shape)
        if per_observation_count <= 1:
            raise ValueError(
                f"entity field at '{self.address}' requires at least 2 elements per observation, "
                f"but configured count is {per_observation_count}"
            )

        max_classes = root.batch_size * per_observation_count
        for topk in self.topk:
            if topk >= max_classes:
                raise ValueError(
                    f"topk values must be less than max local entity classes ({max_classes}) for '{self.address}'"
                )


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
        session: Session,
        strata: Strata,
        state: Any,
    ) -> TensorFieldBase:

        context_shape: tuple[int, ...] = session.structure.shapes[address]

        data, states = pad(
            nested=values,
            shape=(len(values), *context_shape),
            dtype=object,
            pad_value=None,
        )

        try:
            tokens = _local_reindex(data=data, states=states)
        except TypeError as error:
            raise ValueError(f"entity field at '{address}' only accepts hashable scalar values") from error

        state_tensor = torch.tensor(states, dtype=torch.int64)
        content = torch.tensor(tokens, dtype=torch.int64)

        return cls(
            state=state_tensor,
            content=content,
            trainable=torch.zeros_like(input=state_tensor, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def mask(self, p_mask: float):
        mask_token = torch.full_like(input=self.state, fill_value=Tokens.masked.value)
        is_masked = torch.rand_like(input=self.state, dtype=torch.float).lt(other=p_mask)

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_scatter(is_masked, mask_token)
        self.content = self.content.masked_scatter(is_masked, torch.zeros_like(input=self.content))

        self.trainable |= is_masked

    def prune(self, p_prune: float = 1.0):
        prune_tokens = torch.full_like(input=self.state, fill_value=Tokens.pruned)

        is_pruned = (
            torch.rand(self.state.size(0), *([1] * (len(self.state.shape) - 1)), device=self.state.device)
            .lt(p_prune)
            .expand_as(self.state)
        )

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_scatter(is_pruned, prune_tokens)
        self.content = self.content.masked_scatter(is_pruned, torch.zeros_like(input=self.content))

        self.trainable |= is_pruned

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        structure: Structure,
    ):
        shape: tuple[int, ...] = (batch_size, *structure.shapes[address])

        state = torch.full(shape, Tokens.pruned)
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
    def __init__(self, structure: Structure, address: Address):
        super().__init__(structure=structure, address=address)

        self.max_slots: int = structure.batch_size * math.prod(structure.shapes[address])
        self.origin: Address = address
        self.destination: Address = structure.requests[address].parent.address
        self.n_embeddings: int = self.max_slots + len(Tokens)

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=structure.d_model,
                ),
                TensorKey.content.name: torch.nn.Embedding(
                    num_embeddings=self.max_slots,
                    embedding_dim=structure.d_model,
                ),
            }
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: tuple[int, ...]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if valued.any() and (content.masked_select(valued) >= self.max_slots).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds bucket space of {self.max_slots}")

        safe_content = content.masked_fill(~valued, 0)
        embeddings: torch.Tensor = (
            self.embeddings[TensorKey.state.name](state) +
            self.embeddings[TensorKey.content.name](safe_content) * valued.unsqueeze(-1)
        ).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@entity.register
class Decoder(DecoderBase):
    def __init__(self, structure: Structure, address: Address):
        super().__init__(structure=structure, address=address)

        self.state_linear = torch.nn.Linear(in_features=structure.d_model, out_features=len(Tokens))
        self.projection = torch.nn.Linear(in_features=structure.d_model, out_features=structure.d_model)

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
    module: JSON2Vec,
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
        )
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    features = prediction.payload[TensorKey.content].reshape(N, -1)
    targets = batch.targets[TensorKey.content].reshape(N)

    max_index = int(targets.masked_select(valued).max().item()) + 1
    codebook = module.nodes[prediction.address].embedder.embeddings[TensorKey.content.name].weight[:max_index]
    inputs = torch.matmul(features, codebook.transpose(0, 1))

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
        )
    )

    for topk in module.session.structure.requests[prediction.address].topk:
        if topk >= inputs.shape[1]:
            continue

        module.track(
            (prediction.address, strata, Metric.accuracy, f"top{topk}"),
            value=(
                inputs
                .topk(k=topk, dim=1)
                .indices.eq(targets.unsqueeze(1))
                .any(dim=1)
                .masked_select(valued).float().mean()
            )
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=inputs.argmax(dim=1).eq(targets).masked_select(valued).float().mean(),
    )

    return loss


@entity.register
def write(module: JSON2Vec, prediction: Prediction):
    return None
