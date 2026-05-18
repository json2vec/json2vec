from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.architecture.plot import Pane
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
from json2vec.tensorfields.shared.counter import Counter
from json2vec.tensorfields.shared.vocabulary import OnlineVocabularyModel, Vocabulary, VocabularySyncCallback

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Hyperparameters

category: Plugin = Plugin(name="category")
# This is a content-level fallback for OOV categories. The field is still present,
# so we keep state=`valued` and route the content into a reserved bucket instead of
# collapsing it into state=`null`.
UNAVAILABLE_LABEL = "<unavailable>"

category.callback(VocabularySyncCallback)

@category.register
class Request(RequestBase):
    type: Literal["category"] = "category"
    max_vocab_size: Annotated[int, pydantic.Field(gt=0, default=10_000)] = 10000
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01
    topk: list[int] | None = None

    @pydantic.model_validator(mode="after")
    def check_topk(self):

        if self.topk is None:
            self.topk = []

        # enforce uniqueness
        self.topk = sorted(set(self.topk))

        for topk in self.topk:
            if not isinstance(topk, int):
                raise ValueError("topk values must be integers")

            if topk <= 0:
                raise ValueError("topk values must be positive")

            if topk == 1:
                raise ValueError("topk values must not be 1")

            if topk >= self.max_vocab_size:
                raise ValueError("topk values must be less than max_vocab_size")

        return self

    

@category.register
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

        array_shape: tuple[int, ...] = hyperparameters.shapes[address]

        tokens = apply(values, partial(state, update=(strata == Strata.train)))

        if len(state) > (max_vocab_size := hyperparameters.requests[address].max_vocab_size):
            print(f"Vocab in address {address} exceeds max vocab size of {max_vocab_size}")

        data, states = pad(
            nested=tokens,
            shape=(len(values), *array_shape),
            dtype=np.int64,
            pad_value=0,
        )

        state_tensor = torch.tensor(states, dtype=torch.int64)
        content = torch.tensor(data=data, dtype=torch.int64)
        if strata == Strata.train:
            p_unavailable: float = hyperparameters.requests[address].p_unavailable
            unavailable_index: int = hyperparameters.requests[address].max_vocab_size

            if p_unavailable > 0.0:
                # Unavailable content never appears naturally during training, because the
                # train split is exactly where the vocabulary is built. We simulate a small
                # amount of OOV behavior here so the decoder learns to use that bucket.
                is_known = state_tensor.eq(Tokens.valued.value) & content.ne(unavailable_index)
                if is_known.any():
                    simulated = (
                        torch.rand_like(input=state_tensor, dtype=torch.float).lt(other=p_unavailable) & is_known
                    )
                    if simulated.any():
                        content = content.masked_fill(simulated, unavailable_index)

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
        self.content = self.content.masked_scatter(is_targeted, torch.zeros_like(input=self.content))

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
        content = torch.zeros(shape, dtype=torch.int64)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@category.register
class Embedder(EmbedderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.max_vocab_size: int = request.max_vocab_size

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel(max_vocab_size=request.max_vocab_size)
        # One extra slot is reserved for UNAVAILABLE_LABEL on top of the learned vocabulary.
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
        dims: tuple[int, ...]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if valued.any() and (content.masked_select(valued) >= self.n_content_tokens).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds max vocab size of {self.max_vocab_size}")

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

    @property
    def state(self) -> Vocabulary:
        return self.vocab.state



@category.register
class Decoder(DecoderBase):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: RequestBase = hyperparameters.requests[address]

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
                TensorKey.content.name: Counter(address=address, size=request.max_vocab_size + 1),
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


@category.register
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
        )
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    content_inputs = prediction.payload[TensorKey.content].reshape(N, -1)
    content_targets = batch.targets[TensorKey.content].reshape(N)
    content_counter_values = content_targets.masked_select(state_targets.eq(Tokens.valued.value))
    if content_counter_values.numel() > 0:
        decoder.counters[TensorKey.content.name](content_counter_values)

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=(
            torch.nn.functional.cross_entropy(
                input=content_inputs,
                target=content_targets,
                weight=decoder.counters[TensorKey.content.name].weight,
                reduction="none",
            )
            .masked_select(valued)
            .mean()
        )
    )

    for topk in module.hyperparameters.requests[prediction.address].topk:
        module.track(
            (prediction.address, strata, Metric.accuracy, f"top{topk}"),
            value=(
                content_inputs
                .topk(k=topk, dim=1)
                .indices.eq(content_targets.unsqueeze(1))
                .any(dim=1)
                .masked_select(valued).float().mean()
            )
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=content_inputs.argmax(dim=1).eq(content_targets).masked_select(valued).float().mean(),
    )

    return loss


@category.register
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

    vocab = np.array(node.embedder.vocab.snapshot(), dtype=object)
    labels = vocab
    content_shape = tuple(state_distribution.shape[:-1])
    content_labels = np.full(content_shape, None, dtype=object)
    content_probabilities = np.zeros(content_shape, dtype=np.float32)

    requested_ks: list[int] = module.hyperparameters.requests[prediction.address].topk
    max_requested_k: int = max(requested_ks, default=0)

    def _pack_candidates(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]] | list:
        if labels.ndim == 1:
            return [
                {"label": str(label), "probability": float(probability)}
                for label, probability in zip(labels.tolist(), probabilities.tolist())
            ]

        return [_pack_candidates(labels[index], probabilities[index]) for index in range(labels.shape[0])]

    def _empty_candidates(shape: tuple[int, ...]) -> list | None:
        if len(shape) == 0:
            return []

        return [_empty_candidates(shape[1:]) for _ in range(shape[0])]

    topk_payload: list | None = _empty_candidates(content_shape)
    if len(vocab) > 0:
        candidate_indices = torch.arange(len(vocab), device=content_logits.device, dtype=torch.int64)
        candidate_logits = content_logits.index_select(dim=-1, index=candidate_indices)
        log_norm = candidate_logits.logsumexp(dim=-1, keepdim=True)
        max_logits, max_indices = candidate_logits.max(dim=-1)
        content_probabilities = (max_logits - log_norm.squeeze(-1)).exp().detach().float().cpu().numpy()

        max_indices_np: np.ndarray = max_indices.detach().cpu().numpy().astype(np.int32)
        content_labels = labels[max_indices_np]

        if max_requested_k > 0:
            topk: int = min(max_requested_k, candidate_logits.shape[-1])
            topk_logits, topk_indices = candidate_logits.topk(k=topk, dim=-1)
            topk_probabilities = (topk_logits - log_norm).exp()

            topk_indices_np: np.ndarray = topk_indices.detach().cpu().numpy().astype(np.int32)
            topk_labels_np: np.ndarray = labels[topk_indices_np]
            topk_probabilities_np: np.ndarray = topk_probabilities.detach().float().cpu().numpy()
            topk_payload = _pack_candidates(
                labels=topk_labels_np,
                probabilities=topk_probabilities_np,
            )

    return {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: {
            TensorKey.value.name: content_labels,
            TensorKey.probability.name: content_probabilities,
            TensorKey.topk.name: topk_payload,
        },
    }


@category.register
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
