# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Plugin,
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

sets: Plugin = Plugin(name="set", types=(bool, int, float, str, bytes))
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

    @classmethod
    def vocabulary(
        cls,
        source: "Model | InterprocessEncodingContext",
        address: Address | str,
        /,
    ) -> tuple[Any, ...]:
        """Return an immutable snapshot of one set vocabulary."""
        from relflow.architecture.root import Model

        address = Address(str(address))
        if isinstance(source, Model):
            if address not in source.nodes:
                raise KeyError(f"no field at address {str(address)!r}")

            embedder = getattr(source.nodes[address], "embedder", None)
            if not isinstance(embedder, Embedder):
                raise TypeError(f"address {str(address)!r} is not a Set field (got {type(embedder).__name__})")
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
                f"Set.vocabulary source must be a Model or InterprocessEncodingContext, got {type(source).__name__}"
            )

        with state.lock:
            return tuple(state.master)

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


@sets.register
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Reserve and count the complete pristine set exposure."""

    if not learn:
        return None
    if not isinstance(state, VocabularyState):
        raise RuntimeError(f"set field at '{address}' requires a vocabulary encoding context")

    values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type) or pa.types.is_fixed_size_list(values.type):
        values = pc.list_flatten(values)
    if values.null_count:
        values = pc.filter(values, pc.is_valid(values))
    indices = state.indices(values, learn=True)
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(torch.from_numpy(indices.copy()), schema.requests[address].size),
        },
        batch_size=[],
    )


@sets.register
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
        request: Request = schema.requests[address]
        n_tokens: int = request.size
        state = context.state
        if state is not None and not isinstance(state, VocabularyState):
            raise TypeError(f"set field at '{address}' requires VocabularyState context, got {type(state).__name__}")

        def encode(field: RaggedField) -> torch.Tensor:
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            encoded = np.zeros((len(values), n_tokens), dtype=np.float32)
            if not len(values):
                return torch.from_numpy(field.place(encoded, fill=0.0, value_shape=(n_tokens,)))
            if state is None:
                raise RuntimeError(f"set field at '{address}' requires a vocabulary encoding context")
            if (
                pa.types.is_list(values.type)
                or pa.types.is_large_list(values.type)
                or pa.types.is_fixed_size_list(values.type)
            ):
                parents = pc.list_parent_indices(values)
                flattened = pc.list_flatten(values)
            else:
                parents = pa.array(np.arange(len(values), dtype=np.int64))
                flattened = values
            if flattened.null_count:
                valid = pc.is_valid(flattened)
                flattened = pc.filter(flattened, valid)
                parents = pc.filter(parents, valid)
            indices = state.indices(flattened, learn=False)
            if len(indices):
                rows = parents.to_numpy(zero_copy_only=False)
                known = indices < n_tokens
                encoded[rows[known], indices[known]] = 1.0
            return torch.from_numpy(field.place(encoded, fill=0.0, value_shape=(n_tokens,)))

        state_tensor = torch.from_numpy(input.dense)
        content = encode(input)
        target_content = encode(target)

        if strata == Strata.train and request.p_unavailable > 0.0:
            # Training learns vocabulary online, so known set labels rarely look OOV.
            # Simulate partial observation by randomly dropping positive labels.
            def regularize(values: torch.Tensor) -> torch.Tensor:
                selected = torch.rand_like(values).lt(request.p_unavailable) & values.bool()
                return values.masked_fill(selected, 0.0)

            content = regularize(content)
            target_content = regularize(target_content)

        return cls(
            state=state_tensor,
            content=content,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: target_content,
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
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
                TensorKey.content.name: Counter(address=address, size=request.size),
            }
        )

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
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
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def context(self) -> VocabularyState:
        return self.vocab.state


@sets.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine set-state counts to the model resource."""

    if strata != Strata.train:
        raise ValueError(f"set learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = module.nodes[address].embedder
    embedder.counters[TensorKey.state.name].learn(observation[TensorKey.state])
    embedder.counters[TensorKey.content.name].learn(observation[TensorKey.content])


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
def output(module: Model, address: Address) -> pa.StructType:
    candidate = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    return pa.struct([pa.field(TensorKey.content.name, pa.list_(candidate), nullable=False)])


@sets.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content_type = datatype.field(TensorKey.content.name).type
    candidate_type = content_type.value_type
    logits = prediction.payload[TensorKey.content]
    coordinates = logits.reshape(-1, logits.shape[-1])
    vocabulary = labels(module.nodes[prediction.address].embedder.vocab)
    size = len(vocabulary)
    if size > coordinates.shape[-1]:
        raise ValueError(
            f"set vocabulary at {prediction.address!s} has {size} values but prediction width is "
            f"{coordinates.shape[-1]}"
        )

    probabilities = coordinates[:, :size].sigmoid()
    request: Request = module.schema.requests[prediction.address]
    if request.threshold is None:
        counts = torch.full((coordinates.shape[0],), size, dtype=torch.int64)
        indices = torch.arange(size, device=logits.device).expand(coordinates.shape[0], size).reshape(-1)
        selected = probabilities.reshape(-1)
    else:
        keep = probabilities.ge(request.threshold)
        counts = keep.sum(dim=-1, dtype=torch.int64)
        indices = keep.nonzero(as_tuple=False)[:, 1]
        selected = probabilities[keep]

    candidates = struct(
        {
            TensorKey.value.name: pc.take(vocabulary, array(indices, pa.int64())),
            TensorKey.probability.name: array(selected, pa.float32()),
        },
        candidate_type,
    )
    return struct({TensorKey.content.name: variable(candidates, counts)}, datatype)
