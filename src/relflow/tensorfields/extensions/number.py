# ty: ignore[unknown-argument]
from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.distributed import all_reduce_sum
from relflow.helpers import Jitter
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import array, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


number: Extension = Extension(name="number", types=(int | float,))
number.callback(CounterUpdateCallback)

FOURIER_SAFE_MAX_ANGLE = float(1e4)


class Objective(enum.StrEnum):
    mae = "mae"
    mse = "mse"
    huber = "huber"

    def loss(self, *, input: torch.Tensor, target: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        match self:
            case Objective.mae:
                return torch.nn.functional.l1_loss(input=input, target=target, reduction=reduction)
            case Objective.mse:
                return torch.nn.functional.mse_loss(input=input, target=target, reduction=reduction)
            case Objective.huber:
                return torch.nn.functional.huber_loss(input=input, target=target, reduction=reduction)


@number.register
class Request(RequestBase):
    """Numeric scalar tensorfield request."""

    type: Literal["number"] = "number"
    jitter: Jitter = pydantic.Field(default_factory=Jitter)
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    alpha: Annotated[float | None, pydantic.Field(gt=0.0, lt=1.0, default=None)] = None
    objective: Objective = Objective.mae

    @classmethod
    def normalization(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[str, float | int | None]:
        """Return a CPU-native snapshot of one Number field's normalizer."""
        from relflow.architecture.root import Model

        if not isinstance(model, Model):
            raise TypeError(f"Number.normalization model must be a Model, got {type(model).__name__}")

        address = Address(str(address))
        if address not in model.nodes:
            raise KeyError(f"no field at address {str(address)!r}")

        embedder = getattr(model.nodes[address], "embedder", None)
        if not isinstance(embedder, Embedder):
            raise TypeError(f"address {str(address)!r} is not a Number field (got {type(embedder).__name__})")

        normalizer = embedder.normalizer
        mean = float(normalizer.mean.detach().cpu().item())
        variance = float(normalizer.var.detach().cpu().item())
        count = None if normalizer.alpha is not None else int(normalizer.count.detach().cpu().item())

        return {
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance + normalizer.epsilon),
            "count": count,
            "alpha": normalizer.alpha,
        }


@number.register
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Summarize complete pristine numeric state and content."""

    if not learn:
        return None
    values = pc.cast(field.values, pa.float64(), safe=True)
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    content = torch.from_numpy(values.to_numpy(zero_copy_only=False).copy())
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: moments(content),
        },
        batch_size=[],
    )


@number.register
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
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            if (
                pa.types.is_list(values.type)
                or pa.types.is_large_list(values.type)
                or pa.types.is_fixed_size_list(values.type)
            ):
                raise ValueError(f"number field at '{address}' expects scalar Arrow values, got {values.type}")

            encoded = pc.cast(values, pa.float64(), safe=True)
            if isinstance(encoded, pa.ChunkedArray):
                encoded = encoded.combine_chunks()
            data = field.place(encoded.to_numpy(zero_copy_only=False), fill=0.0)
            return torch.from_numpy(np.nan_to_num(data, nan=0.0)).to(dtype=torch.float32)

        state = torch.from_numpy(input.dense)

        return cls(
            content=encode(input),
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: encode(target),
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


def moments(values: torch.Tensor) -> torch.Tensor:
    """Reduce finite numeric values to count, sum, and squared sum."""

    finite = values.reshape(-1).to(dtype=torch.float64)
    finite = finite.masked_select(torch.isfinite(finite))
    if not finite.numel():
        return torch.zeros(3, dtype=torch.float64, device=values.device)
    return torch.stack(
        (
            finite.new_tensor(finite.numel()),
            finite.sum(),
            finite.square().sum(),
        )
    )


class GlobalOnlineNormalizer(torch.nn.Module):
    def __init__(self, alpha: float | None = None, epsilon: float = 1e-5):
        super().__init__()

        self.epsilon: float = epsilon
        self.alpha: float | None = alpha

        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.zeros(1, dtype=torch.int64))

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        self.merge(moments(values))

    @torch.no_grad()
    def learn(self, observation: torch.Tensor) -> None:
        """Synchronize and apply one pristine numeric observation."""

        if not isinstance(observation, torch.Tensor):
            raise TypeError(f"normalizer observation must be a tensor, got {type(observation).__name__}")
        if tuple(observation.shape) != (3,):
            raise ValueError(f"normalizer observation must have shape (3,), got {tuple(observation.shape)}")
        self.merge(all_reduce_sum(observation.to(device=self.mean.device, dtype=torch.float64).clone()))

    @torch.no_grad()
    def merge(self, observation: torch.Tensor) -> None:
        """Merge count, sum, and squared sum into running statistics."""

        if tuple(observation.shape) != (3,):
            raise ValueError(f"normalizer observation must have shape (3,), got {tuple(observation.shape)}")
        if not torch.isfinite(observation).all():
            raise ValueError("normalizer observation must contain only finite values")
        if observation[0] < 0:
            raise ValueError("normalizer observation count cannot be negative")

        batch_count = observation[0].to(device=self.count.device, dtype=self.count.dtype)
        if not batch_count:
            return

        batch_mean = observation[1].div(observation[0]).to(device=self.mean.device, dtype=self.mean.dtype)
        batch_var = (
            observation[2].div(observation[0]).sub(observation[1].div(observation[0]).square()).clamp_min(0.0)
        ).to(device=self.var.device, dtype=self.var.dtype)

        if self.alpha is not None:
            alpha: float = self.alpha
            new_mean = (1 - alpha) * self.mean + alpha * batch_mean
            new_var = (1 - alpha) * self.var + alpha * batch_var

            # Commit updates
            self.mean = new_mean
            self.var = new_var

            return

        old_count = self.count
        new_count = old_count + batch_count

        delta = batch_mean - self.mean

        # New mean
        new_mean = self.mean + delta * (batch_count / new_count)

        # Variance update
        m_a = self.var * old_count
        m_b = batch_var * batch_count
        m_c = delta.pow(2) * old_count * batch_count / new_count
        new_var = (m_a + m_b + m_c) / new_count

        # Commit
        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, update=True) -> torch.Tensor:
        finite_mask = mask & torch.isfinite(inputs)
        if self.training and update:
            self.update(inputs[finite_mask])

        std = torch.sqrt(self.var + self.epsilon)
        out = inputs.clone()
        out[finite_mask] = (inputs[finite_mask] - self.mean) / std

        return out


@number.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address

        self.embeddings = torch.nn.Embedding(num_embeddings=len(Tokens), embedding_dim=schema.d_model)
        self.counter = Counter(address=address, size=len(Tokens))

        n_bands = request.n_bands
        offset = request.offset

        weights = torch.logspace(start=-n_bands, end=offset, steps=n_bands + offset + 1, base=2)
        self.linear = torch.nn.Linear(2 * len(weights), schema.d_model)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))
        self.register_buffer("max_fourier_input", torch.tensor(FOURIER_SAFE_MAX_ANGLE) / self.weights.abs().max())

        self.jitter: Jitter = request.jitter
        self.weights: torch.Tensor
        self.max_fourier_input: torch.Tensor

        self.normalizer: GlobalOnlineNormalizer = GlobalOnlineNormalizer(alpha=request.alpha)

    def clamp(self, content: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bound = self.max_fourier_input.to(device=content.device, dtype=content.dtype)
        valued = state.eq(Tokens.valued)
        finite = torch.isfinite(content)
        out_of_range = content.abs().gt(bound)
        unsafe = ~finite | out_of_range

        if unsafe.any():
            unsafe_values = content[unsafe].detach()
            unsafe_abs = unsafe_values.abs()
            finite_abs = unsafe_abs[torch.isfinite(unsafe_abs)]
            if torch.isinf(unsafe_abs).any():
                max_abs = math.inf
            elif finite_abs.numel() > 0:
                max_abs = float(finite_abs.max().cpu().item())
            else:
                max_abs = math.nan

            logger.bind(
                component="tensorfield",
                field_type="number",
                address=str(self.origin),
                count=int(unsafe.sum().cpu().item()),
                valued_count=int((unsafe & valued).sum().cpu().item()),
                nonfinite_count=int((~torch.isfinite(unsafe_values)).sum().cpu().item()),
                max_abs_normalized=max_abs,
                bound=float(bound.detach().cpu().item()),
                safe_max_angle=FOURIER_SAFE_MAX_ANGLE,
            ).warning("number Fourier inputs exceed safe range; clamping normalized values")

        clamped = content.clamp(min=-bound, max=bound)
        return torch.where(torch.isnan(clamped), torch.zeros_like(clamped), clamped)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        state = inputs.state.reshape(D)
        content = inputs.content.reshape(D)
        eligible = state.eq(Tokens.valued) & torch.isfinite(content)

        if self.training and not self.jitter.normalize:
            content = self.jitter.apply(content, eligible)

        content = self.normalizer(inputs=content, mask=state.eq(Tokens.valued), update=False)

        if self.training and self.jitter.normalize:
            content = self.jitter.apply(content, eligible)

        content = self.clamp(content=content, state=state)

        # weight inputs with buffers of precision bands
        weighted = content.unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.cat([torch.sin(weighted), torch.cos(weighted)], dim=1)

        projection = torch.nn.functional.gelu(self.linear(fourier)).reshape(N, *dims, -1)

        embeddings = self.embeddings(state).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings + projection,
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@number.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine numeric counts and normalization moments."""

    if strata != Strata.train:
        raise ValueError(f"number learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = module.nodes[address].embedder
    embedder.counter.learn(observation[TensorKey.state])
    embedder.normalizer.learn(observation[TensorKey.content])


@number.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.classification = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))
        self.regression = torch.nn.Linear(in_features=schema.d_model, out_features=1)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.regression(pooled),
            }
        )


@number.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = module.schema.requests[prediction.address]

    embedder: Embedder = module.nodes[address].embedder
    normalizer: GlobalOnlineNormalizer = embedder.normalizer

    N: int = batch.targets[TensorKey.state].numel()

    trainable: torch.Tensor = batch.trainable.reshape(N)
    state_targets = batch.targets[TensorKey.state].reshape(N)

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=prediction.payload[TensorKey.state].reshape(N, -1),
                target=state_targets,
                weight=embedder.counter.weight,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    target: torch.Tensor = batch.targets[TensorKey.content].reshape(N)
    inputs: torch.Tensor = prediction.payload[TensorKey.content].reshape(N)
    diff: torch.Tensor = inputs.subtract(target)

    loss += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=request.objective.loss(
            input=diff / normalizer.var.sqrt().clamp_min(normalizer.epsilon),
            target=torch.zeros_like(diff),
            reduction="none",
        )
        .masked_select(trainable)
        .mean(),
    )

    module.track(
        (address, strata, Metric.mae, TensorKey.content),
        value=diff.absolute().masked_select(trainable).float().mean(),
    )

    module.track(
        (address, strata, Metric.rmse, TensorKey.content),
        value=diff.square().masked_select(trainable).float().mean().sqrt(),
    )

    return loss


@number.register
def output(module: Model, address: Address) -> pa.StructType:
    return pa.struct([pa.field(TensorKey.content.name, pa.float64(), nullable=False)])


@number.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content = array(prediction.payload[TensorKey.content], pa.float64())
    return struct({TensorKey.content.name: content}, datatype)
