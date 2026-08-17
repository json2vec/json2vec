# ty: ignore[unknown-argument]
from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Annotated, Any, Iterable, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.nested import extract_mask_literals, pad
from relflow.rich import console, record_incident
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


number: Plugin = Plugin(name="number")
number.callback(CounterUpdateCallback)

FOURIER_SAFE_MAX_ANGLE = float(1e4)


def jitter(inputs: torch.Tensor, jitter_amount: torch.Tensor) -> torch.Tensor:
    noise = torch.rand_like(inputs).sub(torch.rand_like(inputs)).mul(jitter_amount)
    return inputs.add(noise)


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
    jitter: Annotated[float, pydantic.Field(ge=0.0, default=0.0)] = 0.0
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    alpha: Annotated[float | None, pydantic.Field(gt=0.0, lt=1.0, default=None)] = None
    objective: Objective = Objective.mae


@number.register
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
        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )

        data, flags = pad(
            nested=values,
            shape=leading_shape,
            dtype=np.float64,
            pad_value=np.nan,
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

        cdf = np.nan_to_num(data, nan=0.0)
        content = torch.tensor(data=cdf, dtype=torch.float)
        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        content = content.masked_fill(literal_mask_tensor, 0.0)
        state = torch.tensor(data=flags, dtype=torch.int64).masked_fill(literal_mask_tensor, Tokens.masked.value)

        return cls(
            content=content,
            state=state,
            targets=TensorDict({}),
            trainable=torch.zeros_like(input=content, dtype=torch.bool),
            batch_size=len(values),
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked)

        if cache_targets and TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()
        self.state: torch.Tensor = self.state.masked_scatter(selected, mask_token)

        if cache_targets and TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()
        self.content: torch.Tensor = self.content.masked_scatter(
            selected, torch.full_like(input=self.content, fill_value=0.0)
        )

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
        content = torch.full(shape, 0.0)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


class GlobalOnlineNormalizer(torch.nn.Module):
    def __init__(self, alpha: float | None = None, epsilon: float = 1e-5):
        super().__init__()

        self.epsilon: float = epsilon
        self.alpha: float | None = alpha

        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        numel = values.numel()
        if numel == 0:
            return

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)

        if self.alpha is not None:
            alpha: float = self.alpha
            new_mean = (1 - alpha) * self.mean + alpha * batch_mean
            new_var = (1 - alpha) * self.var + alpha * batch_var

            # Commit updates
            self.mean = new_mean
            self.var = new_var

            return

        old_count = self.count
        new_count = old_count + numel

        delta = batch_mean - self.mean

        # New mean
        new_mean = self.mean + delta * (numel / new_count)

        # Variance update
        m_a = self.var * old_count
        m_b = batch_var * numel
        m_c = delta.pow(2) * old_count * numel / new_count
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
        self.register_buffer("jitter", torch.tensor(request.jitter))
        self.register_buffer("max_fourier_input", torch.tensor(FOURIER_SAFE_MAX_ANGLE) / self.weights.abs().max())

        self.jitter: torch.Tensor
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
            nonfinite = ~torch.isfinite(unsafe_values)
            severity = "nonfinite" if nonfinite.any() else "out-of-range"
            if record_incident("number-clamp", id(self), str(self.origin), severity).emit:
                unsafe_abs = unsafe_values.abs()
                finite_abs = unsafe_abs[torch.isfinite(unsafe_abs)]
                if torch.isinf(unsafe_abs).any():
                    max_abs = math.inf
                elif finite_abs.numel() > 0:
                    max_abs = float(finite_abs.max().cpu().item())
                else:
                    max_abs = math.nan

                console.log(
                    "[relflow.warning]number inputs exceed the safe Fourier range; clamping values[/]",
                    {
                        "address": self.origin,
                        "count": int(unsafe.sum().cpu().item()),
                        "valued_count": int((unsafe & valued).sum().cpu().item()),
                        "nonfinite_count": int(nonfinite.sum().cpu().item()),
                        "max_abs_normalized": max_abs,
                        "bound": float(bound.detach().cpu().item()),
                        "safe_max_angle": FOURIER_SAFE_MAX_ANGLE,
                    },
                )

        clamped = content.clamp(min=-bound, max=bound)
        return torch.where(torch.isnan(clamped), torch.zeros_like(clamped), clamped)

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        state = inputs.state.reshape(D)
        content = inputs.content.reshape(D)

        content = self.normalizer(inputs=content, mask=state.eq(Tokens.valued))

        if self.training:
            content = jitter(content, jitter_amount=self.jitter)

        content = self.clamp(content=content, state=state)

        # weight inputs with buffers of precision bands
        weighted = content.unsqueeze(dim=1).mul(self.weights)

        # apply sine and cosine functions to weighted inputs
        fourier = torch.cat([torch.sin(weighted), torch.cos(weighted)], dim=1)

        projection = torch.nn.functional.gelu(self.linear(fourier)).reshape(N, *dims, -1)

        embeddings = self.embeddings(state).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings + projection,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


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
def write(module: Model, prediction: Prediction):
    content: np.ndarray = prediction.payload[TensorKey.content].detach().double().cpu().numpy()
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    tokens: np.ndarray = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    output: dict[str, Iterable] = {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: content,
    }

    return output
