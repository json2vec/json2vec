# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pydantic
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict, tensorclass

from relflow.data.nested import apply, extract_mask_literals, pad
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
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState, VocabularySyncCallback

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema

category: Plugin = Plugin(name="category")

category.callback(VocabularySyncCallback, CounterUpdateCallback)


@category.register
class Request(RequestBase):
    """Categorical scalar tensorfield request backed by an online vocabulary."""

    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

    type: Literal["category"] = "category"
    capacity: Annotated[
        int,
        pydantic.Field(alias="size", serialization_alias="size", gt=0, default=1024),
    ] = 1024
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01
    topk: list[int] | None = None

    @property
    def size(self) -> int:
        return self.capacity

    @size.setter
    def size(self, value: int) -> None:
        self.capacity = value
        self.model_fields_set.add("capacity")

    @pydantic.model_validator(mode="before")
    @classmethod
    def reject_removed_options(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "max_vocab_size" in data:
            raise ValueError("max_vocab_size was removed; use size")

        return data

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

            if topk >= self.size:
                raise ValueError("topk values must be less than size")

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
        schema: Schema,
        strata: Strata,
        interprocess_encoding_context: VocabularyState,
    ) -> TensorFieldBase:
        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )
        learn = strata == Strata.train

        interprocess_encoding_context.reserve(values, learn=learn)
        tokens = apply(values, interprocess_encoding_context.encode)

        if len(interprocess_encoding_context) > (size := schema.requests[address].size):
            logger.bind(component="tensorfield", field_type="category", address=str(address)).warning(
                "vocabulary exceeds size={}", size
            )

        data, states = pad(
            nested=tokens,
            shape=leading_shape,
            dtype=np.int64,
            pad_value=0,
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

        state_tensor = torch.tensor(states, dtype=torch.int64)
        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = state_tensor.masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.tensor(data=data, dtype=torch.int64)
        content = content.masked_fill(literal_mask_tensor, 0)
        if strata == Strata.train:
            p_unavailable: float = schema.requests[address].p_unavailable
            unavailable_index: int = schema.requests[address].size

            if p_unavailable > 0.0:
                # Unavailable content never appears naturally during training, because the
                # train split is exactly where the vocabulary is built. We simulate a small
                # amount of OOV behavior so the content objective does not reward any real
                # class for valued inputs whose categorical content is unavailable.
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


@category.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.size: int = request.size

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel(size=request.size)

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.content.name: torch.nn.Embedding(
                    num_embeddings=self.size,
                    embedding_dim=schema.d_model,
                ),
            }
        )
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
                TensorKey.content.name: Counter(address=address, size=request.size),
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

        if valued.any() and (content.masked_select(valued) > self.size).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds vocabulary size of {self.size}")

        known = valued & content.lt(self.size)
        safe_content = content.masked_fill(~known, 0)
        content_embedding = self.embeddings[TensorKey.content.name](safe_content) * known.unsqueeze(-1)

        embeddings: torch.Tensor = (self.embeddings[TensorKey.state.name](state) + content_embedding).reshape(
            N, *dims, -1
        )

        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def interprocess_encoding_context(self) -> VocabularyState:
        return self.vocab.state


@category.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.content.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=request.size,
                ),
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
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    embedder: Embedder = module.nodes[prediction.address].embedder
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
                weight=cast(Counter, embedder.counters[TensorKey.state.name]).weight,
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
    module.track(
        (prediction.address, strata, "vocabulary", "size"),
        value=state_inputs.new_tensor(len(embedder.vocab.master), dtype=torch.float32),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    content_inputs = prediction.payload[TensorKey.content].reshape(N, -1)
    content_targets = batch.targets[TensorKey.content].reshape(N)
    n_content_tokens = content_inputs.shape[-1]
    invalid = valued & content_targets.gt(n_content_tokens)
    if invalid.any():
        raise ValueError(f"Token in address {prediction.address} exceeds vocabulary size")

    known = valued & content_targets.lt(n_content_tokens)
    unavailable = valued & content_targets.eq(n_content_tokens)

    content_loss_sum = content_inputs.new_zeros(())
    if known.any():
        known_losses = torch.nn.functional.cross_entropy(
            input=content_inputs[known],
            target=content_targets[known],
            weight=cast(Counter, embedder.counters[TensorKey.content.name]).weight,
            reduction="none",
        )
        content_loss_sum = content_loss_sum + known_losses.sum()

    if unavailable.any():
        unavailable_losses = -torch.nn.functional.log_softmax(content_inputs[unavailable], dim=1).mean(dim=1)
        content_loss_sum = content_loss_sum + unavailable_losses.sum()

    content_loss = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=content_loss_sum / valued.float().sum().clamp_min(1.0),
    )
    loss += content_loss

    if not known.any():
        return loss

    for topk in module.schema.requests[prediction.address].topk:
        module.track(
            (prediction.address, strata, Metric.accuracy, f"top{topk}"),
            value=(
                content_inputs.topk(k=topk, dim=1)
                .indices.eq(content_targets.unsqueeze(1))
                .any(dim=1)
                .masked_select(known)
                .float()
                .mean()
            ),
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=content_inputs.argmax(dim=1).eq(content_targets).masked_select(known).float().mean(),
    )

    return loss


@category.register
def write(module: Model, prediction: Prediction):
    node = module.nodes[prediction.address]
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    content_logits: torch.Tensor = prediction.payload[TensorKey.content]

    tokens = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    vocab = np.array(node.embedder.vocab.snapshot(), dtype=object)
    labels = vocab
    content_shape = tuple(state_distribution.shape[:-1])
    content_labels = np.full(content_shape, None, dtype=object)
    content_probabilities = np.zeros(content_shape, dtype=np.float32)

    requested_ks: list[int] = module.schema.requests[prediction.address].topk
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
