from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterable, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.architecture.plot import Pane
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
from json2vec.tensorfields.shared.counter import Counter, CounterUpdateCallback

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Hyperparameters


number: Plugin = Plugin(name="number")
number.callback(CounterUpdateCallback)


def jitter(inputs: torch.Tensor, jitter_amount: torch.Tensor) -> torch.Tensor:
    noise = torch.rand_like(inputs).sub(torch.rand_like(inputs)).mul(jitter_amount)
    return inputs.add(noise)


class Objective(enum.StrEnum):

    mae = "mae"
    mse = "mse"
    huber = "huber"

OBJECTIVES: dict[Objective, Any] = {
    Objective.mae: torch.nn.functional.l1_loss,
    Objective.mse: torch.nn.functional.mse_loss,
    Objective.huber: torch.nn.functional.huber_loss,
}

@number.register
class Request(RequestBase):
    """Numeric scalar tensorfield request."""

    type: Literal["number"] = "number"
    jitter: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)] = 0.0
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    alpha: Annotated[float|None, pydantic.Field(gt=0.0, lt=1.0, default=None)] = None
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
        hyperparameters: Hyperparameters,
        strata: Strata,
    ) -> TensorFieldBase:
        array_shape: tuple[int, ...] = hyperparameters.shapes[address]

        data, flags = pad(
            nested=values,
            shape=(len(values), *array_shape),
            dtype=np.float64,
            pad_value=np.nan,
        )

        cdf = np.nan_to_num(data, nan=0.0)
        content = torch.tensor(data=cdf, dtype=torch.float)

        return cls(
            content=content,
            state=torch.tensor(data=flags, dtype=torch.int64),
            targets=TensorDict({}),
            trainable=torch.zeros_like(input=content, dtype=torch.bool),
            batch_size=len(values),
        )

    def mask(self, p_mask: float):
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked)

        is_masked: torch.Tensor = torch.rand_like(input=self.content, dtype=torch.float).lt(other=p_mask)

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()
        self.state: torch.Tensor = self.state.masked_scatter(is_masked, mask_token)

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()
        self.content: torch.Tensor = self.content.masked_scatter(
            is_masked, torch.full_like(input=self.content, fill_value=0.0)
        )

        self.trainable |= is_masked

    def target(self, p_prune: float = 1.0):
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked)

        is_targeted = (
            torch.rand(self.state.size(0), *([1] * (len(self.state.shape) - 1)), device=self.state.device)
            .lt(p_prune)
            .expand_as(self.state)
        )

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        self.state: torch.Tensor = self.state.masked_scatter(is_targeted, mask_token)

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.content: torch.Tensor = self.content.masked_scatter(
            is_targeted, torch.full_like(input=self.content, fill_value=0.0)
        )

        self.trainable |= is_targeted

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        hyperparameters: Hyperparameters,
    ):
        shape: tuple[int, ...] = (batch_size, *hyperparameters.shapes[address])

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

    def __init__(self, alpha: float | None=None, epsilon: float = 1e-5):

        super().__init__()

        self.epsilon: float = epsilon
        self.alpha: float|None = alpha

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

        if self.training and update:
            self.update(inputs[mask])

        std = torch.sqrt(self.var + self.epsilon)
        out = inputs.clone()
        out[mask] = (inputs[mask] - self.mean) / std

        return out


@number.register
class Embedder(EmbedderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address

        self.embeddings = torch.nn.Embedding(num_embeddings=len(Tokens), embedding_dim=hyperparameters.d_model)
        self.counter = Counter(address=address, size=len(Tokens))

        n_bands = request.n_bands
        offset = request.offset

        weights = torch.logspace(start=-n_bands, end=offset, steps=n_bands + offset + 1, base=2)
        self.linear = torch.nn.Linear(2 * len(weights), hyperparameters.d_model)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))
        self.register_buffer("jitter", torch.tensor(request.jitter))

        self.jitter: torch.Tensor
        self.weights: torch.Tensor

        # # Define a safe maximum angle (radians) to avoid "soft overflow" / extreme inputs to sin/cos.
        # # This scalar controls the largest magnitude of the weighted value that will be passed to sin/cos.
        # # You can tune SAFE_MAX_ANGLE to be more or less restrictive.
        # SAFE_MAX_ANGLE = float(1e4)
        # # Compute the maximum absolute content value that will produce weighted inputs within SAFE_MAX_ANGLE.
        # max_content = SAFE_MAX_ANGLE / self.weights.abs().max()
        # # store as a buffer so it's moved with the module/device and dtype
        # self.register_buffer("max_fourier_input", max_content)
        # def fourier_input_bounds(self) -> tuple[float, float]:
        #     """Return (min, max) allowed raw content values that will be passed to the Fourier transform."""
        #     bound = float(self.max_fourier_input.item())
        #     return (-bound, bound)

        self.normalizer: GlobalOnlineNormalizer = GlobalOnlineNormalizer(alpha=request.alpha)


    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:

        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        state = inputs.state.reshape(D)
        content = inputs.content.reshape(D)

        content = self.normalizer(inputs=content, mask=state.eq(Tokens.valued))

        if self.training:
            content = jitter(content, jitter_amount=self.jitter)

        # # clamp content to the theoretical bounds determined by the weights to avoid excessively large
        # # angles passed into sin/cos (which can lead to numerical instability in practice)
        # content = content.clamp(min=-self.max_fourier_input, max=self.max_fourier_input)

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
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        self.classification = torch.nn.Linear(in_features=hyperparameters.d_model, out_features=len(Tokens))
        self.regression = torch.nn.Linear(in_features=hyperparameters.d_model, out_features=1)

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
    request: RequestBase = module.hyperparameters.requests[prediction.address]

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
        )
    )

    target: torch.Tensor = batch.targets[TensorKey.content].reshape(N)
    inputs: torch.Tensor = prediction.payload[TensorKey.content].reshape(N)
    diff: torch.Tensor = inputs.subtract(target)

    objective: Callable = OBJECTIVES[request.objective]

    loss += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=objective(
            input=diff / normalizer.var.sqrt().clamp_min(normalizer.epsilon),
            target=torch.zeros_like(diff),
            reduction="none",
        ).masked_select(trainable).mean(),
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
    state_payload = {
        token: state_distribution[..., index]
        for index, token in enumerate(tokens.tolist())
    }

    output: dict[str, Iterable] = {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: content,
    }

    return output


@number.register
def plot(
    module: Model,
    address: Address,
    branch: Pane,
    detail: bool,
):
    if not detail:
        return

    normalizer: GlobalOnlineNormalizer = module.nodes[address].embedder.normalizer
    branch.add_section(
        "state",
        {
            "mean": float(normalizer.mean.item()),
            "variance": float(normalizer.var.item()),
            "std_dev": float(torch.sqrt(normalizer.var + normalizer.epsilon).item()),
            "count": int(normalizer.count.item()),
        },
    )
