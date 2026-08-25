# ty: ignore[unknown-argument]
from __future__ import annotations

import math
from collections.abc import Hashable as HashableValue
from typing import TYPE_CHECKING, Annotated, Any, Literal

import msgspec
import numpy as np
import pydantic
import torch
from beartype import beartype
from blake3 import blake3
from tensordict import TensorDict, tensorclass

from relflow.data.nested import extract_mask_literals, pad
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

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


hashable: Plugin = Plugin(name="hash")


_HASH_NORMALIZER: float = float(1 << 63)
_STATIC_HASH_KEY = bytes(32)
_SPLITMIX_GAMMA = -7046029254386353131
_SPLITMIX_MIX_1 = -4658895280553007687
_SPLITMIX_MIX_2 = -7723592293110705685
_SALTED_STRATA = frozenset({Strata.train, Strata.validate})


def salt(
    module: Model | None,
    *,
    deterministic: bool,
    strata: Strata,
    n_hashes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve the shared train/validation salt for one optimizer step."""
    if deterministic or strata not in _SALTED_STRATA:
        return None

    global_step = 0 if module is None else int(module.global_step)
    if global_step < 0:
        raise ValueError("global_step must be non-negative")

    seed = torch.scalar_tensor(global_step + 1, dtype=torch.int64, device=device)
    mixed = torch.arange(n_hashes, dtype=torch.int64, device=seed.device)
    mixed = mixed.mul(_SPLITMIX_GAMMA).add(seed)
    shifted = torch.bitwise_and(torch.bitwise_right_shift(mixed, 30), (1 << 34) - 1)
    mixed = torch.bitwise_xor(mixed, shifted).mul(_SPLITMIX_MIX_1)
    shifted = torch.bitwise_and(torch.bitwise_right_shift(mixed, 27), (1 << 37) - 1)
    mixed = torch.bitwise_xor(mixed, shifted).mul(_SPLITMIX_MIX_2)
    shifted = torch.bitwise_and(torch.bitwise_right_shift(mixed, 31), (1 << 33) - 1)
    return torch.bitwise_xor(mixed, shifted)


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

    CPU tensorization always uses the same hash key. By default, training and
    validation hash the global optimizer step into one device-side salt per
    lane. Every salted `hash` field receives the same lane salts, preserving
    equality relationships within an observation and across fields. Set
    `deterministic=True` to disable this rotation for a field, which exposes a
    persistent identifier representation that can be powerful but may
    overfit. Test and predict always bypass salting for stable inference.
    """

    type: Literal["hash"] = "hash"
    n_hashes: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    n_buckets: Annotated[int, pydantic.Field(gt=1, default=4)] = 4
    deterministic: bool = False


@hashable.register
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
    ) -> TensorFieldBase:
        request: Request = schema.requests[address]
        n_hashes: int = request.n_hashes

        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )

        data, states = pad(
            nested=values,
            shape=leading_shape,
            dtype=object,
            pad_value=None,
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

        hashes = np.zeros((*data.shape, n_hashes), dtype=np.int64)
        flat_hashes = hashes.reshape(-1, n_hashes)
        for index, (value, state) in enumerate(zip(data.reshape(-1), states.reshape(-1), strict=True)):
            if state != Tokens.valued.value:
                continue
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
            digest = blake3(payload, key=_STATIC_HASH_KEY).digest(length=n_hashes * 8)
            flat_hashes[index] = np.frombuffer(digest, dtype=">i8").astype(np.int64)

        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = torch.tensor(states, dtype=torch.int64).masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.from_numpy(hashes).masked_fill(literal_mask_tensor.unsqueeze(-1), 0)

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
        self.content = self.content.masked_fill(selected.unsqueeze(-1).expand_as(self.content), 0.0)

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
        content = torch.zeros((*shape, request.n_hashes), dtype=torch.int64)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@hashable.register
class Embedder(EmbedderBase):
    def __init__(
        self,
        schema: Schema,
        address: Address,
        module: Model | None = None,
    ):
        super().__init__(schema=schema, address=address)
        # A normal assignment would register the parent as this child's
        # submodule and create a recursive PyTorch module graph.
        object.__setattr__(self, "_module", module)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.n_hashes: int = request.n_hashes
        self.deterministic: bool = request.deterministic

        n_bands = request.n_bands
        offset = request.offset
        n_frequencies = (schema.d_model + 1) // 2
        weights = torch.logspace(start=-n_bands, end=offset, steps=n_frequencies, base=2).mul(math.pi)
        self.register_buffer("weights", weights.reshape(1, 1, -1))
        self.weights: torch.Tensor

        self.state_embeddings = torch.nn.Embedding(num_embeddings=len(Tokens), embedding_dim=schema.d_model)
        self.d_model = schema.d_model

    @beartype
    def forward(
        self,
        inputs: TensorFieldBase,
        *,
        strata: Strata = Strata.predict,
    ) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1, self.n_hashes)
        valued = state.eq(Tokens.valued.value)

        salted = salt(
            self._module,
            deterministic=self.deterministic,
            strata=strata,
            n_hashes=self.n_hashes,
            device=content.device,
        )
        normalized = (
            (content if salted is None else torch.bitwise_xor(content, salted))
            .to(dtype=self.weights.dtype)
            .div(_HASH_NORMALIZER)
        )
        weighted = normalized.unsqueeze(-1).mul(self.weights)
        sinusoidal = torch.stack([torch.sin(weighted), torch.cos(weighted)], dim=-1)
        sinusoidal = sinusoidal.flatten(start_dim=-2)[..., : self.d_model]
        content_embeddings = sinusoidal.sum(dim=1)
        state_embeddings = self.state_embeddings(state)
        embeddings = torch.where(valued.unsqueeze(-1), content_embeddings, state_embeddings).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings,
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

    # Per-hash categorical over optimizer-step-salted quantile buckets.
    inputs = prediction.payload[TensorKey.content].reshape(N * n_hashes, n_buckets)
    raw_targets = batch.targets[TensorKey.content].reshape(N, n_hashes)
    salted = salt(
        module,
        deterministic=request.deterministic,
        strata=strata,
        n_hashes=n_hashes,
        device=raw_targets.device,
    )
    bucket_targets = (
        (raw_targets if salted is None else torch.bitwise_xor(raw_targets, salted))
        .to(dtype=torch.get_default_dtype())
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
def write(module: Model, prediction: Prediction):
    return None
