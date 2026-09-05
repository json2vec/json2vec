# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pydantic
import torch
from beartype import beartype
from lightning.pytorch import Callback
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
    from lightning.pytorch import Trainer

    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.structs.experiment import Schema

category: Plugin = Plugin(name="category")

# Smaller clamps can overflow zero-query CosFace gradients in float16 at the default scale.
_NORMALIZE_EPS: float = 1e-3


def _default_margin(size: int, d_model: int) -> float:
    if size < 2:
        return 0.0

    if size <= d_model + 1:
        simplex_max = size / (size - 1)
        return min(0.5, 0.5 * simplex_max)

    return max(0.05, 0.35 * (d_model / size) ** 0.5)


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
    scale: Annotated[float, pydantic.Field(gt=0.0, default=30.0)] = 30.0
    margin: Annotated[float | None, pydantic.Field(ge=0.0, lt=1.0, default=None)] = None
    topk: list[int] | None = None

    @classmethod
    def vocabulary(
        cls,
        source: "Model | InterprocessEncodingContext",
        address: Address | str,
        /,
    ) -> tuple[Any, ...]:
        """Return an immutable snapshot of one categorical vocabulary."""
        from relflow.architecture.root import Model

        address = Address(str(address))
        if isinstance(source, Model):
            if address not in source.nodes:
                raise KeyError(f"no field at address {str(address)!r}")

            embedder = getattr(source.nodes[address], "embedder", None)
            if not isinstance(embedder, Embedder):
                raise TypeError(f"address {str(address)!r} is not a Category field (got {type(embedder).__name__})")
            with embedder.vocab.lock:
                return tuple(embedder.vocab.master)
        elif isinstance(source, Mapping):
            if address not in source:
                raise KeyError(f"no encoding context at address {str(address)!r}")

            state = source[address]
            if not isinstance(state, VocabularyState):
                raise TypeError(
                    f"encoding context at {str(address)!r} is not a VocabularyState (got {type(state).__name__})"
                )
        else:
            raise TypeError(
                "Category.vocabulary source must be a Model or InterprocessEncodingContext, "
                f"got {type(source).__name__}"
            )

        with state.lock:
            return tuple(state.master)

    @classmethod
    def counts(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[Any, int]:
        """Return observed training counts for the populated vocabulary."""
        from relflow.architecture.root import Model

        if not isinstance(model, Model):
            raise TypeError(f"Category.counts model must be a Model, got {type(model).__name__}")

        address = Address(str(address))
        if address not in model.nodes:
            raise KeyError(f"no field at address {str(address)!r}")

        embedder = getattr(model.nodes[address], "embedder", None)
        if not isinstance(embedder, Embedder):
            raise TypeError(f"address {str(address)!r} is not a Category field (got {type(embedder).__name__})")

        vocabulary = cls.vocabulary(model, address)
        counts = (
            embedder.counters[TensorKey.content.name]
            .counts[: len(vocabulary)]
            .detach()
            .cpu()
            .sub(1)
            .clamp_min(0)
            .tolist()
        )
        return {label: int(count) for label, count in zip(vocabulary, counts, strict=True)}

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
        self.margin: float = (
            float(request.margin)
            if request.margin is not None
            else _default_margin(size=request.size, d_model=schema.d_model)
        )

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
        with torch.no_grad():
            content_weight = self.embeddings[TensorKey.content.name].weight
            content_weight.copy_(torch.nn.functional.normalize(content_weight, dim=-1, eps=_NORMALIZE_EPS))

    def content_directions(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        embedding = self.embeddings[TensorKey.content.name]
        return embedding.weight if indices is None else embedding(indices)

    def content_cosine(self, query: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        query_hat = torch.nn.functional.normalize(query, dim=-1, eps=_NORMALIZE_EPS)
        return query_hat @ self.content_directions(indices).T

    @torch.no_grad()
    def normalize_content_directions(self) -> None:
        weight = self.embeddings[TensorKey.content.name].weight
        weight.copy_(torch.nn.functional.normalize(weight, dim=-1, eps=_NORMALIZE_EPS))

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if (content.masked_select(valued) > self.size).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds vocabulary size of {self.size}")

        available = valued & content.lt(self.size)
        safe_content = content.masked_fill(~available, 0)
        content_embedding = self.content_directions(safe_content) * available.unsqueeze(-1)

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

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.content.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=schema.d_model,
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
    trainable_state_inputs = state_inputs[trainable]
    trainable_state_targets = state_targets[trainable]

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=torch.nn.functional.cross_entropy(
            input=trainable_state_inputs,
            target=trainable_state_targets,
            weight=cast(Counter, embedder.counters[TensorKey.state.name]).weight,
            reduction="none",
        ).mean(),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=trainable_state_inputs.argmax(dim=1).eq(trainable_state_targets).float().mean(),
    )
    module.track(
        (prediction.address, strata, "vocabulary", "size"),
        value=state_inputs.new_tensor(len(embedder.vocab.master), dtype=torch.float32),
    )
    with torch.no_grad():
        valued_state_norm = embedder.embeddings[TensorKey.state.name].weight[Tokens.valued.value].norm()
    module.track(
        (prediction.address, strata, "embedding", "valued_state_norm"),
        value=valued_state_norm,
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    request: Request = cast(Request, module.schema.requests[prediction.address])
    vocab_size: int = embedder.size
    query = prediction.payload[TensorKey.content].reshape(N, -1)
    content_targets = batch.targets[TensorKey.content].reshape(N)
    valued_query = query[valued]
    valued_targets = content_targets[valued]

    if valued_targets.gt(vocab_size).any():
        raise ValueError(f"Token in address {prediction.address} exceeds vocabulary size")

    available = valued_targets.lt(vocab_size)
    unavailable = valued_targets.eq(vocab_size)

    cosine = embedder.content_cosine(valued_query)
    cos_available = cosine[available]
    tgt_available = valued_targets[available]

    content_loss_sum = query.new_zeros(())
    logits_available = request.scale * cos_available
    if embedder.margin > 0.0:
        margin = logits_available.new_full(
            (tgt_available.shape[0], 1),
            -request.scale * embedder.margin,
        )
        logits_available.scatter_add_(1, tgt_available.unsqueeze(1), margin)

    available_losses = torch.nn.functional.cross_entropy(
        input=logits_available,
        target=tgt_available,
        weight=cast(Counter, embedder.counters[TensorKey.content.name]).weight,
        reduction="none",
    )
    content_loss_sum = content_loss_sum + available_losses.sum()

    logits_unavailable = request.scale * cosine[unavailable]
    unavailable_losses = -torch.nn.functional.log_softmax(logits_unavailable, dim=1).mean(dim=1)
    content_loss_sum = content_loss_sum + unavailable_losses.sum()

    content_loss = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=content_loss_sum / valued_targets.numel(),
    )
    loss += content_loss

    if tgt_available.numel() == 0:
        return loss

    requested_ks: list[int] = request.topk
    top_indices = cos_available.topk(k=max(requested_ks), dim=1).indices if requested_ks else None
    for topk in requested_ks:
        assert top_indices is not None
        module.track(
            (prediction.address, strata, Metric.accuracy, f"top{topk}"),
            value=(top_indices[:, :topk].eq(tgt_available.unsqueeze(1)).any(dim=1).float().mean()),
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=cos_available.argmax(dim=1).eq(tgt_available).float().mean(),
    )

    return loss


@category.register
def write(module: Model, prediction: Prediction):
    node = module.nodes[prediction.address]
    request: Request = cast(Request, module.schema.requests[prediction.address])
    embedder: Embedder = node.embedder
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    content_query: torch.Tensor = prediction.payload[TensorKey.content]

    tokens = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    vocab = np.array(embedder.vocab.snapshot(), dtype=object)
    labels = vocab
    content_shape = tuple(state_distribution.shape[:-1])
    content_labels = np.full(content_shape, None, dtype=object)
    content_probabilities = np.zeros(content_shape, dtype=np.float32)

    requested_ks: list[int] = request.topk
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
        candidate_indices = torch.arange(len(vocab), device=content_query.device, dtype=torch.int64)
        candidate_logits = request.scale * embedder.content_cosine(content_query, indices=candidate_indices)
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


class ContentNormalizeCallback(Callback):
    """Renormalize categorical content directions at each train epoch end."""

    @torch.no_grad()
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: Model,
    ) -> None:  # ty:ignore[invalid-method-override]
        for node in pl_module.nodes.values():
            embedder = getattr(node, "embedder", None)
            if isinstance(embedder, Embedder):
                embedder.normalize_content_directions()


category.callback(VocabularySyncCallback, CounterUpdateCallback, ContentNormalizeCallback)
