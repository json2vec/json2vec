from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.architecture.plot import Pane
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
from json2vec.tensorfields.extensions.category import (
    UNAVAILABLE_LABEL,
)
from json2vec.tensorfields.shared.counter import Counter
from json2vec.tensorfields.shared.vocabulary import OnlineVocabularyModel, Vocabulary, VocabularySyncCallback

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Hyperparameters

sets: Plugin = Plugin(name="set")
sets.callback(VocabularySyncCallback)


@sets.register
class Request(RequestBase):
    type: Literal["set"] = "set"
    max_vocab_size: Annotated[int, pydantic.Field(gt=0, default=10_000)] = 10_000
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01


def _items(value: Any) -> Iterable[Any]:
    if value is None:
        return ()

    if isinstance(value, str):
        return (value,)

    if isinstance(value, Iterable):
        return value

    return (value,)


def _encode_set(value: Any, state: Vocabulary, update: bool, n_tokens: int) -> np.ndarray:
    encoded = np.zeros(n_tokens, dtype=np.float32)

    for item in _items(value):
        if item is None:
            continue

        index = state(item, update=update)
        if index is not None:
            encoded[index] = 1.0

    return encoded


def _pad_sets(
    values: list,
    shape: tuple[int, ...],
    state: Vocabulary,
    update: bool,
    n_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    content = np.zeros((*shape, n_tokens), dtype=np.float32)
    flags = np.full(shape, Tokens.padded.value, dtype=np.int64)

    def walk(node: Any, depth: int, index: tuple[int, ...]) -> None:
        if depth == len(shape):
            if node is None:
                flags[index] = Tokens.null.value
                return

            flags[index] = Tokens.valued.value
            content[index] = _encode_set(value=node, state=state, update=update, n_tokens=n_tokens)
            return

        if not isinstance(node, list):
            return

        for position, child in enumerate(node[:shape[depth]]):
            walk(child, depth + 1, (*index, position))

    walk(values, 0, ())

    return content, flags


@sets.register
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
        hyperparameters: Hyperparameters,
        strata: Strata,
        state: Vocabulary,
    ) -> TensorFieldBase:
        request: Request = hyperparameters.requests[address]
        shape: tuple[int, ...] = (len(values), *hyperparameters.shapes[address])
        n_tokens: int = request.max_vocab_size + 1

        data, states = _pad_sets(
            values=values,
            shape=shape,
            state=state,
            update=(strata == Strata.train),
            n_tokens=n_tokens,
        )

        state_tensor = torch.tensor(states, dtype=torch.int64)
        content = torch.tensor(data=data, dtype=torch.float32)

        if strata == Strata.train and request.p_unavailable > 0.0:
            known = content[..., : request.max_vocab_size].bool()
            simulated = torch.rand_like(content[..., : request.max_vocab_size]).lt(request.p_unavailable) & known
            if simulated.any():
                content[..., : request.max_vocab_size] = content[..., : request.max_vocab_size].masked_fill(
                    simulated,
                    0.0,
                )
                content[..., request.max_vocab_size] = torch.maximum(
                    content[..., request.max_vocab_size],
                    simulated.any(dim=-1).to(dtype=content.dtype),
                )

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
        self.content = self.content.masked_fill(is_masked.unsqueeze(-1), 0.0)

        self.trainable |= is_masked

    def target(self, p_target: float = 1.0):
        mask_tokens = torch.full_like(input=self.state, fill_value=Tokens.masked.value)

        is_targeted = (
            torch.rand(self.state.size(0), *([1] * (len(self.state.shape) - 1)), device=self.state.device)
            .lt(p_target)
            .expand_as(self.state)
        )

        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_scatter(is_targeted, mask_tokens)
        self.content = self.content.masked_fill(is_targeted.unsqueeze(-1), 0.0)

        self.trainable |= is_targeted

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        hyperparameters: Hyperparameters,
    ):
        request: Request = hyperparameters.requests[address]
        shape: tuple[int, ...] = (batch_size, *hyperparameters.shapes[address])

        state = torch.full(shape, Tokens.masked)
        content = torch.zeros((*shape, request.max_vocab_size + 1), dtype=torch.float32)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@sets.register
class Embedder(EmbedderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.max_vocab_size: int = request.max_vocab_size

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel(max_vocab_size=request.max_vocab_size)
        self.n_content_tokens: int = request.max_vocab_size + 1

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=hyperparameters.d_model,
                ),
                TensorKey.content.name: torch.nn.Embedding(
                    num_embeddings=self.n_content_tokens,
                    embedding_dim=hyperparameters.d_model,
                ),
            }
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: list[int]

        N, *dims, n_tokens = inputs.content.shape
        if n_tokens != self.n_content_tokens:
            raise ValueError(f"Set in address {self.origin} has invalid vocabulary width")

        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1, n_tokens)
        valued = state.eq(Tokens.valued.value)

        weights = self.embeddings[TensorKey.content.name].weight
        counts = content.sum(dim=-1, keepdim=True).clamp_min(1.0)
        content_embedding = content.to(dtype=weights.dtype).matmul(weights) / counts

        embeddings: torch.Tensor = (
            self.embeddings[TensorKey.state.name](state) + content_embedding * valued.unsqueeze(-1)
        ).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def state(self) -> Vocabulary:
        return self.vocab.state


@sets.register
class Decoder(DecoderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=hyperparameters.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.content.name: torch.nn.Linear(
                    in_features=hyperparameters.d_model,
                    out_features=request.max_vocab_size + 1,
                ),
            }
        )

        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
            }
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.linears[TensorKey.state.name](pooled),
                TensorKey.content: self.linears[TensorKey.content.name](pooled),
            }
        )


@sets.register
def loss(
    module: JSON2Vec,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    decoder: Decoder = module.nodes[prediction.address].decoder
    N: int = batch.targets[TensorKey.state].numel()
    trainable = batch.trainable.reshape(N)

    state_inputs = prediction.payload[TensorKey.state].reshape(N, -1)
    state_targets = batch.targets[TensorKey.state].reshape(N)
    decoder.counters[TensorKey.state.name](batch.targets[TensorKey.state])

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                weight=decoder.counters[TensorKey.state.name].weight,
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

    content_inputs = prediction.payload[TensorKey.content].reshape(N, -1)
    content_targets = batch.targets[TensorKey.content].reshape(N, -1)

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=torch.nn.functional.binary_cross_entropy_with_logits(
            input=content_inputs.masked_select(valued.unsqueeze(1)).reshape(-1, content_inputs.shape[-1]),
            target=content_targets.masked_select(valued.unsqueeze(1)).reshape(-1, content_targets.shape[-1]),
        ),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=(
            content_inputs.sigmoid()
            .ge(0.5)
            .eq(content_targets.bool())
            .masked_select(valued.unsqueeze(1))
            .float()
            .mean()
        ),
    )

    return loss


@sets.register
def write(module: JSON2Vec, prediction: Prediction):
    node = module.nodes[prediction.address]
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    content_logits: torch.Tensor = prediction.payload[TensorKey.content]

    tokens = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {
        token: state_distribution[..., index]
        for index, token in enumerate(tokens.tolist())
    }

    vocab = node.embedder.vocab.snapshot()
    probabilities = content_logits[..., : len(vocab)].sigmoid().detach().float().cpu().numpy()
    content_payload = {
        str(label): probabilities[..., index]
        for index, label in enumerate(vocab)
    }

    return {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: content_payload,
    }


@sets.register
def plot(
    module: JSON2Vec,
    address: Address,
    branch: Pane,
    detail: bool,
):
    if not detail:
        return

    embedder: Embedder = module.nodes[address].embedder
    vocabulary = embedder.vocab.snapshot()
    branch.add_section(
        "state",
        {
            "vocabulary_size": len(vocabulary),
            "vocabulary": vocabulary,
            "unavailable_label": UNAVAILABLE_LABEL,
        },
    )
