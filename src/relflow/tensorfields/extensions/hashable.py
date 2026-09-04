# ty: ignore[unknown-argument]
from __future__ import annotations

import math
from collections.abc import Hashable as HashableValue
from typing import TYPE_CHECKING, Annotated, Literal

import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from blake3 import blake3
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

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


hashable: Plugin = Plugin(name="hash", types=(bool, int, float, str, bytes))


_HASH_NORMALIZER: float = float(1 << 63)


@hashable.register
class Request(RequestBase):
    """Hash-based tensorfield with no learned identity vocabulary.

    Content is a static function of the input value: `n_hashes` independent
    deterministic hash lanes produce a fixed-length integer vector per slot.
    On the model device, those integers are normalized, expanded through the
    Fourier feature bands configured by `n_bands` and `offset`, and summed
    directly in model space.
    No learned content embedding or projection is created; a field-local state
    embedding is used only for non-valued slots.

    For reconstruction loss, each hash channel is quantized into `n_buckets`
    uniform bins over `[-1, 1)`, and a per-channel cross-entropy trains the
    decoder to identify the correct bucket. Effective identity fingerprint
    capacity is `n_buckets ** n_hashes`.

    During training and validation, the hash key is salted independently for
    each encoded batch, so the network cannot memorize persistent
    value-specific representations. Every `hash` field in a batch
    receives the same salt, preserving equality relationships within an
    observation and across fields. Test and predict use salt 0 for stable
    inference.
    """

    type: Literal["hash"] = "hash"
    n_hashes: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    n_buckets: Annotated[int, pydantic.Field(gt=1, default=4)] = 4


@hashable.register
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
        n_hashes: int = request.n_hashes

        key = context.salt.to_bytes(32, "big", signed=False)

        def encode(field: RaggedField) -> torch.Tensor:
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            if (
                pa.types.is_list(values.type)
                or pa.types.is_large_list(values.type)
                or pa.types.is_fixed_size_list(values.type)
                or pa.types.is_struct(values.type)
                or pa.types.is_map(values.type)
            ):
                raise ValueError(f"hash field at '{address}' expects scalar Arrow values, got {values.type}")
            if pa.types.is_dictionary(values.type):
                values = pc.dictionary_decode(values)
            unique = pc.unique(values)
            unique_hashes = np.empty((len(unique), n_hashes), dtype=np.int64)
            for index, value in enumerate(unique.to_pylist()):
                if not isinstance(value, HashableValue):
                    raise ValueError(
                        f"hash field at '{address}' only accepts MessagePack-compatible hashable scalar values"
                    )
                try:
                    payload = msgspec.msgpack.encode(value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        f"hash field at '{address}' only accepts MessagePack-compatible hashable scalar values"
                    ) from error
                digest = blake3(payload, key=key).digest(length=n_hashes * 8)
                unique_hashes[index] = np.frombuffer(digest, dtype=">i8").astype(np.int64)

            positions = pc.index_in(values, value_set=unique)
            if positions.null_count:
                raise RuntimeError(f"hash field at '{address}' failed to resolve an Arrow value")
            hashes = unique_hashes[positions.to_numpy(zero_copy_only=False)]
            return torch.from_numpy(field.place(hashes, fill=0, value_shape=(n_hashes,)))

        state_tensor = torch.from_numpy(input.dense)

        return cls(
            state=state_tensor,
            content=encode(input),
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


@hashable.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.n_hashes: int = request.n_hashes

        n_bands = request.n_bands
        offset = request.offset
        n_frequencies = (schema.d_model + 1) // 2
        weights = torch.logspace(start=-n_bands, end=offset, steps=n_frequencies, base=2).mul(math.pi)
        self.register_buffer("weights", weights.reshape(1, 1, -1))
        self.weights: torch.Tensor

        self.state_embeddings = torch.nn.Embedding(num_embeddings=len(Tokens), embedding_dim=schema.d_model)
        self.d_model = schema.d_model

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1, self.n_hashes)
        valued = state.eq(Tokens.valued.value)

        normalized = content.to(dtype=self.weights.dtype).div(_HASH_NORMALIZER)
        weighted = normalized.unsqueeze(-1).mul(self.weights)
        sinusoidal = torch.stack([torch.sin(weighted), torch.cos(weighted)], dim=-1)
        sinusoidal = sinusoidal.flatten(start_dim=-2)[..., : self.d_model]
        content_embeddings = sinusoidal.sum(dim=1)
        state_embeddings = self.state_embeddings(state)
        embeddings = torch.where(valued.unsqueeze(-1), content_embeddings, state_embeddings).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings,
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@hashable.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.n_hashes: int = request.n_hashes
        self.n_buckets: int = request.n_buckets
        self.state_linear = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))

        self.content_linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=request.n_hashes * request.n_buckets,
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.state_linear(pooled),
                TensorKey.content: self.content_linear(pooled),
            }
        )


@hashable.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
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

    request: Request = module.schema.requests[prediction.address]
    n_hashes: int = request.n_hashes
    n_buckets: int = request.n_buckets

    # Per-hash categorical over deterministic quantile buckets.
    inputs = prediction.payload[TensorKey.content].reshape(N * n_hashes, n_buckets)
    raw_targets = batch.targets[TensorKey.content].reshape(N, n_hashes)
    bucket_targets = (
        raw_targets.to(dtype=torch.get_default_dtype())
        .div(_HASH_NORMALIZER)
        .add(1.0)
        .mul(0.5 * n_buckets)
        .floor()
        .long()
        .clamp(min=0, max=n_buckets - 1)
        .reshape(N * n_hashes)
    )

    per_hash_ce = torch.nn.functional.cross_entropy(
        input=inputs,
        target=bucket_targets,
        reduction="none",
    )

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=(per_hash_ce.reshape(N, n_hashes).mean(dim=-1).masked_select(valued).mean()),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=(
            inputs.argmax(dim=-1)
            .eq(bucket_targets)
            .float()
            .reshape(N, n_hashes)
            .mean(dim=-1)
            .masked_select(valued)
            .mean()
        ),
    )

    return loss


@hashable.register
def output(module: Model, address: Address) -> None:
    return None


@hashable.register
def write(module: Model, prediction: Prediction, datatype: None) -> None:
    return None
