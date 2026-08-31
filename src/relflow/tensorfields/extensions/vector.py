# ty: ignore[unknown-argument]
from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Annotated, Any, Literal

import awkward as ak
import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
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

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


vector: Plugin = Plugin(name="vector", types=(int | float,))


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self == Objective.l1:
            return torch.nn.functional.l1_loss(input=inputs, target=targets, reduction="none").mean(dim=1)

        return torch.nn.functional.mse_loss(input=inputs, target=targets, reduction="none").mean(dim=1)


@vector.register
class Request(RequestBase):
    """Fixed-width numeric vector tensorfield request."""

    type: Literal["vector"] = "vector"
    n_dim: Annotated[int, pydantic.Field(gt=0)]
    objective: Objective = Objective.l2


def coerce(value: Any, *, n_dim: int, address: Address) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"vector field at '{address}' expects 1D embeddings, got array with ndim={value.ndim}")
        raw: list[Any] = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(
            f"vector field at '{address}' expects embeddings as list/tuple/1D ndarray, got {type(value).__name__}"
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
        field: RaggedField,
        address: Address,
        schema: Schema,
        strata: Strata,
    ) -> TensorFieldBase:
        request: Request = schema.requests[address]
        vectors = [coerce(value, n_dim=request.n_dim, address=address) for value in ak.to_list(field.values)]
        encoded = np.stack(vectors, axis=0) if vectors else np.empty((0, request.n_dim), dtype=np.float32)
        content = field.place(encoded, fill=0.0, value_shape=(request.n_dim,))
        state = torch.from_numpy(field.state)

        return cls(
            content=torch.from_numpy(content),
            state=state,
            trainable=torch.zeros(field.shape, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=field.batch_size,
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked)
        expanded = selected.unsqueeze(-1).expand_as(self.content)

        if cache_targets and TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()
        self.state = self.state.masked_scatter(selected, mask_token)

        if cache_targets and TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()
        self.content = self.content.masked_scatter(expanded, torch.zeros_like(input=self.content))

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
        request: Request = schema.requests[address]
        leading_shape: tuple[int, ...] = (batch_size, *schema.shapes[address])
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
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=request.n_dim, out_features=schema.d_model),
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
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]

        self.classification = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )
        self.regression = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=request.n_dim,
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.regression(pooled),
            }
        )


@vector.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = module.schema.requests[address]

    trainable = batch.trainable.reshape(-1)
    state_targets = batch.targets[TensorKey.state].reshape(-1)
    state_inputs = prediction.payload[TensorKey.state].reshape(-1, len(Tokens))

    output: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
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
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return output

    inputs = prediction.payload[TensorKey.content].reshape(-1, request.n_dim)
    targets = batch.targets[TensorKey.content].reshape(-1, request.n_dim)
    diff = inputs.subtract(targets)

    output += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=request.objective.loss(inputs=inputs, targets=targets).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.mae, TensorKey.content),
        value=diff.absolute().mean(dim=1).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.rmse, TensorKey.content),
        value=diff.square().mean(dim=1).sqrt().masked_select(valued).mean(),
    )

    return output


@vector.register
def write(module: Model, prediction: Prediction):
    content: np.ndarray = prediction.payload[TensorKey.content].detach().float().cpu().numpy()
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    tokens: np.ndarray = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    non_valued = state_logits.argmax(dim=-1).ne(Tokens.valued.value).detach().cpu().numpy()
    content = content.copy()
    content[non_valued] = 0.0

    return {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: content,
    }
