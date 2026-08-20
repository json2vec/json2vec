# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.nested import extract_mask_literals, pad
from relflow.metrics.base import Trait
from relflow.structs.enums import LogKey, Strata, TensorKey, Tokens
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

sets: Plugin = Plugin(name="set", traits=(Trait.classification,))
sets.callback(VocabularySyncCallback, CounterUpdateCallback)


@sets.register
class Request(RequestBase):
    """Multi-label set tensorfield request backed by an online vocabulary."""

    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

    type: Literal["set"] = "set"
    capacity: Annotated[
        int,
        pydantic.Field(alias="size", serialization_alias="size", gt=0, default=10_000),
    ] = 10_000
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01
    threshold: Annotated[float | None, pydantic.Field(ge=0.0, le=1.0, default=None)] = None

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


def _encode_set_content(value: Any, state: VocabularyState, n_tokens: int) -> np.ndarray:
    encoded = np.zeros(n_tokens, dtype=np.float32)

    if value is None:
        return encoded

    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Iterable):
        items = value
    else:
        items = (value,)

    for item in items:
        if item is None:
            continue

        index = state.encode(item)
        if index is not None and index < n_tokens:
            encoded[index] = 1.0

    return encoded


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
        schema: Schema,
        strata: Strata,
        interprocess_encoding_context: VocabularyState,
    ) -> TensorFieldBase:
        request: Request = schema.requests[address]
        shape: tuple[int, ...] = (len(values), *schema.shapes[address])
        n_tokens: int = request.size
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(shape),
        )
        learn = strata == Strata.train

        interprocess_encoding_context.reserve(values, learn=learn)

        data, states = pad(
            nested=values,
            shape=shape,
            dtype=np.float32,
            pad_value=0.0,
            overflows=schema.overflows(address),
            address=address,
            value_shape=(n_tokens,),
            encode=lambda value: _encode_set_content(
                value=value,
                state=interprocess_encoding_context,
                n_tokens=n_tokens,
            ),
        )
        literal_data, _ = pad(
            nested=literal_masks,
            shape=shape,
            dtype=bool,
            pad_value=False,
            overflows=schema.overflows(address),
            address=address,
        )

        state_tensor = torch.tensor(states, dtype=torch.int64)
        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = state_tensor.masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.tensor(data=data, dtype=torch.float32)
        content = content.masked_fill(literal_mask_tensor.unsqueeze(-1), 0.0)

        if strata == Strata.train and request.p_unavailable > 0.0:
            # Training learns vocabulary online, so known set labels rarely look OOV.
            # Simulate partial observation by randomly dropping positive labels.
            known = content.bool()
            simulated = torch.rand_like(content).lt(request.p_unavailable) & known
            if simulated.any():
                content = content.masked_fill(simulated, 0.0)

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
        self.content = self.content.masked_fill(selected.unsqueeze(-1), 0.0)

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
        shape: tuple[int, ...] = (batch_size, *schema.shapes[address])

        state = torch.full(shape, Tokens.masked)
        content = torch.zeros((*shape, request.size), dtype=torch.float32)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@sets.register
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
            }
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: list[int]

        N, *dims, n_tokens = inputs.content.shape
        if n_tokens != self.size:
            raise ValueError(f"Set in address {self.origin} has invalid vocabulary width")

        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1, n_tokens)
        valued = state.eq(Tokens.valued.value)

        weights = cast(torch.nn.Embedding, self.embeddings[TensorKey.content.name]).weight
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
    def interprocess_encoding_context(self) -> VocabularyState:
        return self.vocab.state


@sets.register
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


@sets.register
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
        (prediction.address, strata, LogKey.loss, TensorKey.state),
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
        (prediction.address, strata, LogKey.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    content_inputs = prediction.payload[TensorKey.content].reshape(N, -1)
    content_targets = batch.targets[TensorKey.content].reshape(N, -1)

    loss += module.track(
        (prediction.address, strata, LogKey.loss, TensorKey.content),
        value=torch.nn.functional.binary_cross_entropy_with_logits(
            input=content_inputs.masked_select(valued.unsqueeze(1)).reshape(-1, content_inputs.shape[-1]),
            target=content_targets.masked_select(valued.unsqueeze(1)).reshape(-1, content_targets.shape[-1]),
        ),
    )

    module.track(
        (prediction.address, strata, LogKey.accuracy, TensorKey.content),
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
def write(module: Model, prediction: Prediction):
    node = module.nodes[prediction.address]
    request: Request = module.schema.requests[prediction.address]
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    content_logits: torch.Tensor = prediction.payload[TensorKey.content]

    tokens = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    vocab = node.embedder.vocab.snapshot()
    probabilities = content_logits[..., : len(vocab)].sigmoid().detach().float().cpu().numpy()
    if request.threshold is None:
        content_payload = {str(label): probabilities[..., index] for index, label in enumerate(vocab)}
    else:
        labels = np.asarray(vocab, dtype=object)

        def pack_thresholded(values: np.ndarray) -> dict[str, float] | list:
            if values.ndim == 1:
                keep = values >= request.threshold
                return {
                    str(label): float(probability)
                    for label, probability in zip(labels[keep].tolist(), values[keep].tolist())
                }

            return [pack_thresholded(values[index]) for index in range(values.shape[0])]

        content_payload = pack_thresholded(probabilities)

    return {
        TensorKey.state.name: state_payload,
        TensorKey.content.name: content_payload,
    }
