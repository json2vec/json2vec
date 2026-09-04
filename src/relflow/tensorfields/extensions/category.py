# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
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
from relflow.tensorfields.output import array, labels, struct, variable
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState, VocabularySyncCallback

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.structs.experiment import Schema

category: Extension = Extension(name="category", types=(bool, int, float, str, bytes))

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
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Reserve and count the complete pristine categorical exposure."""

    if not learn:
        return None
    if not isinstance(state, VocabularyState):
        raise RuntimeError(f"category field at '{address}' requires a vocabulary encoding context")

    indices = state.indices(field.values, learn=True)
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(torch.from_numpy(indices.copy()), schema.requests[address].size),
        },
        batch_size=[],
    )


@category.register
@tensorclass
class TensorField(TensorFieldBase):
    state: torch.Tensor
    content: torch.Tensor
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
        state = context.state
        if state is not None and not isinstance(state, VocabularyState):
            raise TypeError(
                f"category field at '{address}' requires VocabularyState context, got {type(state).__name__}"
            )

        def encode(field: RaggedField) -> torch.Tensor:
            if not len(field.values):
                return torch.zeros(field.shape, dtype=torch.int64)
            if state is None:
                raise RuntimeError(f"category field at '{address}' requires a vocabulary encoding context")
            tokens = state.indices(field.values, learn=False)
            return torch.from_numpy(field.place(tokens, fill=0))

        content = encode(input)
        target_content = encode(target)

        if state is not None and len(state) > (size := schema.requests[address].size):
            logger.bind(component="tensorfield", field_type="category", address=str(address)).warning(
                "vocabulary exceeds size={}", size
            )

        state_tensor = torch.from_numpy(input.dense)
        target_state = torch.from_numpy(target.dense)
        if strata == Strata.train:
            p_unavailable: float = schema.requests[address].p_unavailable
            unavailable_index: int = schema.requests[address].size

            if p_unavailable > 0.0:
                # Unavailable content never appears naturally during training, because the
                # train split is exactly where the vocabulary is built. We simulate a small
                # amount of OOV behavior so the content objective does not reward any real
                # class for valued inputs whose categorical content is unavailable.
                def regularize(values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    known = state.eq(Tokens.valued.value) & values.ne(unavailable_index)
                    selected = torch.rand_like(state, dtype=torch.float).lt(p_unavailable) & known
                    return values.masked_fill(selected, unavailable_index)

                content = regularize(content, state_tensor)
                target_content = regularize(target_content, target_state)

        return cls(
            state=state_tensor,
            content=content,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: target_state,
                    TensorKey.content: target_content,
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
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
    def forward(self, inputs: TensorInput) -> Parcel:
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
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def context(self) -> VocabularyState:
        return self.vocab.state


@category.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine categorical counts to model-owned resources."""

    if strata != Strata.train:
        raise ValueError(f"category learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = module.nodes[address].embedder
    embedder.counters[TensorKey.state.name].learn(observation[TensorKey.state])
    embedder.counters[TensorKey.content.name].learn(observation[TensorKey.content])


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
def output(module: Model, address: Address) -> pa.StructType:
    candidate = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    content = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string()),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
            pa.field(TensorKey.topk.name, pa.list_(candidate), nullable=False),
        ]
    )
    return pa.struct([pa.field(TensorKey.content.name, content, nullable=False)])


@category.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content_type = datatype.field(TensorKey.content.name).type
    candidate_type = content_type.field(TensorKey.topk.name).type.value_type
    logits = prediction.payload[TensorKey.content]
    coordinates = logits.reshape(-1, logits.shape[-1])
    vocabulary = labels(module.nodes[prediction.address].embedder.vocab)
    size = len(vocabulary)
    if size > coordinates.shape[-1]:
        raise ValueError(
            f"category vocabulary at {prediction.address!s} has {size} values but prediction width is "
            f"{coordinates.shape[-1]}"
        )

    count = coordinates.shape[0]
    best_labels: pa.Array = pa.nulls(count, type=pa.large_string())
    best_probabilities = torch.zeros(count, dtype=torch.float32, device=logits.device)
    candidates = struct(
        {
            TensorKey.value.name: pa.array([], type=pa.large_string()),
            TensorKey.probability.name: pa.array([], type=pa.float32()),
        },
        candidate_type,
    )
    candidate_counts = torch.zeros(count, dtype=torch.int64)

    if size:
        probabilities = coordinates[:, :size].softmax(dim=-1)
        best_probabilities, best_indices = probabilities.max(dim=-1)
        best_labels = pc.take(vocabulary, array(best_indices, pa.int64()))

        request: Request = module.schema.requests[prediction.address]
        width = min(max(request.topk, default=0), size)
        if width:
            top_probabilities, top_indices = probabilities.topk(k=width, dim=-1)
            candidates = struct(
                {
                    TensorKey.value.name: pc.take(vocabulary, array(top_indices, pa.int64())),
                    TensorKey.probability.name: array(top_probabilities, pa.float32()),
                },
                candidate_type,
            )
            candidate_counts = torch.full((count,), width, dtype=torch.int64)

    content = struct(
        {
            TensorKey.value.name: best_labels,
            TensorKey.probability.name: array(best_probabilities, pa.float32()),
            TensorKey.topk.name: variable(candidates, candidate_counts),
        },
        content_type,
    )
    return struct({TensorKey.content.name: content}, datatype)
