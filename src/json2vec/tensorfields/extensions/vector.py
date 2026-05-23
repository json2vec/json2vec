from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.data.processing import apply, pad
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
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Hyperparameters


vector: Plugin = Plugin(name="vector")


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"


@vector.register
class Request(RequestBase):
    type: Literal["vector"] = "vector"
    n_dim: Annotated[int, pydantic.Field(gt=0)]
    objective: Objective = Objective.l2


def coerce(value: Any, *, n_dim: int, address: Address) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"vector field at '{address}' expects 1D embeddings, got array with ndim={value.ndim}")
        raw: list[Any] = value.tolist()
    elif isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"vector field at '{address}' expects 1D embeddings, got tensor with ndim={value.ndim}")
        raw = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(
            f"vector field at '{address}' expects embeddings as list/tuple/1D tensor/1D ndarray, got {type(value).__name__}"
        )

    if len(raw) != n_dim:
        raise ValueError(f"vector field at '{address}' expects embeddings with length {n_dim}, got {len(raw)}")

    try:
        return np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"vector field at '{address}' contains non-numeric embedding values") from error


@vector.register
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
        hyperparameters: Hyperparameters,
        strata: Strata,
    ) -> TensorFieldBase:
        array_shape: tuple[int, ...] = hyperparameters.shapes[address]
        request: Request = hyperparameters.requests[address]

        leading_shape: tuple[int, ...] = (len(values), *array_shape)

        coerced = apply(
            values,
            coerce,
            n_dim=request.n_dim,
            address=address,
            leaf_depth=len(leading_shape),
        )

        data, state = pad(
            nested=coerced,
            shape=leading_shape,
            dtype=object,
            pad_value=None,
        )

        content = np.zeros((*leading_shape, request.n_dim), dtype=np.float32)
        valued = state == Tokens.valued.value

        if valued.any():
            vectors: list[np.ndarray] = data[valued].tolist()
            content[valued] = np.stack(vectors, axis=0)

        return cls(
            content=torch.tensor(content, dtype=torch.float32),
            state=torch.tensor(state, dtype=torch.int64),
            trainable=torch.zeros(leading_shape, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def mask(self, p_mask: float):
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked)
        is_masked: torch.Tensor = torch.rand_like(input=self.state, dtype=torch.float).lt(other=p_mask)
        expanded = is_masked.unsqueeze(-1).expand_as(self.content)

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()
        self.state = self.state.masked_scatter(is_masked, mask_token)

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()
        self.content = self.content.masked_scatter(expanded, torch.zeros_like(input=self.content))

        self.trainable |= is_masked

    def target(self, p_prune: float = 1.0):
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked)
        is_targeted = (
            torch.rand(self.state.size(0), *([1] * (len(self.state.shape) - 1)), device=self.state.device)
            .lt(p_prune)
            .expand_as(self.state)
        )
        expanded = is_targeted.unsqueeze(-1).expand_as(self.content)

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()
        self.state = self.state.masked_scatter(is_targeted, mask_token)

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()
        self.content = self.content.masked_scatter(expanded, torch.zeros_like(input=self.content))

        self.trainable |= is_targeted

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        hyperparameters: Hyperparameters,
    ):
        request: Request = hyperparameters.requests[address]
        leading_shape: tuple[int, ...] = (batch_size, *hyperparameters.shapes[address])
        state = torch.full(leading_shape, Tokens.masked)
        content = torch.zeros((*leading_shape, request.n_dim), dtype=torch.float32)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@vector.register
class Embedder(EmbedderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=hyperparameters.d_model,
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=request.n_dim, out_features=hyperparameters.d_model),
            torch.nn.GELU(),
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        state = inputs.state.reshape(D)
        content = inputs.content.reshape(D, -1)

        projected = self.linear(content)
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(projected + embeddings).reshape(N, *dims, -1),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@vector.register
class Decoder(DecoderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]

        self.linear = torch.nn.Linear(
            in_features=hyperparameters.d_model,
            out_features=request.n_dim,
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.content: self.linear(pooled),
            }
        )


def _objective_loss(inputs: torch.Tensor, targets: torch.Tensor, objective: Objective) -> torch.Tensor:
    if objective == Objective.l1:
        return torch.nn.functional.l1_loss(input=inputs, target=targets, reduction="none").mean(dim=1)

    return torch.nn.functional.mse_loss(input=inputs, target=targets, reduction="none").mean(dim=1)


@vector.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = module.hyperparameters.requests[address]

    trainable = batch.trainable.reshape(-1)
    inputs = prediction.payload[TensorKey.content].reshape(-1, request.n_dim)
    targets = batch.targets[TensorKey.content].reshape(-1, request.n_dim)
    diff = inputs.subtract(targets)

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=_objective_loss(inputs=inputs, targets=targets, objective=request.objective).masked_select(trainable).mean(),
    )

    module.track(
        (address, strata, Metric.mae, TensorKey.content),
        value=diff.absolute().mean(dim=1).masked_select(trainable).mean(),
    )

    module.track(
        (address, strata, Metric.rmse, TensorKey.content),
        value=diff.square().mean(dim=1).sqrt().masked_select(trainable).mean(),
    )

    return loss


@vector.register
def write(module: Model, prediction: Prediction):
    return {
        TensorKey.content.name: prediction.payload[TensorKey.content].detach().float().cpu().numpy(),
    }
