# ty: ignore[unknown-argument]
from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Hashable
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.data.nested import extract_mask_literals, pad
from json2vec.structs.enums import Metric, Strata, TensorKey, Tokens
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    apply_mask_policies,
)

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Schema


static_entity: Plugin = Plugin(name="static_entity")


_HASH_DIGEST_BYTES: int = 8
_HASH_NORMALIZER: float = float(1 << (_HASH_DIGEST_BYTES * 8 - 1))

_GROUP_STATE: dict[tuple[int, str], dict[str, Any]] = {}

_GROUP_CONFIG_FIELDS: tuple[str, ...] = ("n_hashes", "n_bands", "offset", "n_buckets")


def _group_state(schema: Schema, group: str) -> dict[str, Any]:
    """Registry of parameters shared across `static_entity` addresses in a group."""
    key = (id(schema), group)
    cached = _GROUP_STATE.get(key)
    if cached is not None:
        return cached

    addresses = sorted(
        (
            address
            for address, request in schema.active_requests.items()
            if request.type == "static_entity" and getattr(request, "group", None) == group
        ),
        key=str,
    )
    if not addresses:
        raise ValueError(f"no active static_entity fields configured for group '{group}'")

    canonical_request = schema.active_requests[addresses[0]]
    for address in addresses[1:]:
        request = schema.active_requests[address]
        for field_name in _GROUP_CONFIG_FIELDS:
            expected = getattr(canonical_request, field_name)
            actual = getattr(request, field_name)
            if expected != actual:
                raise ValueError(
                    f"static_entity group '{group}' has inconsistent {field_name}: "
                    f"{addresses[0]}={expected!r}, {address}={actual!r}"
                )

    state: dict[str, Any] = {
        "addresses": tuple(addresses),
        "canonical": addresses[0],
        "linear": None,
        "content_linear": None,
    }
    _GROUP_STATE[key] = state
    weakref.finalize(schema, _GROUP_STATE.pop, key, None)
    return state


def _canonical_bytes(value: Any) -> bytes:
    """Return a stable byte representation of a hashable scalar value."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bool):
        return b"\x01" if value else b"\x00"
    if isinstance(value, (int, float, str)):
        return repr(value).encode("utf-8")
    return repr(value).encode("utf-8")


def _hash_value(value: Any, seed: int) -> float:
    """Deterministic scalar hash in [-1, 1) seeded per hash function."""
    key = seed.to_bytes(_HASH_DIGEST_BYTES, "big")
    digest = hashlib.blake2b(_canonical_bytes(value), digest_size=_HASH_DIGEST_BYTES, key=key).digest()
    raw = int.from_bytes(digest, "big", signed=False)
    return (raw / _HASH_NORMALIZER) - 1.0


def _static_hash_content(
    data: np.ndarray,
    states: np.ndarray,
    n_hashes: int,
) -> np.ndarray:
    """Compute per-value hash vectors of shape (*data.shape, n_hashes)."""
    content = np.zeros((*data.shape, n_hashes), dtype=np.float32)
    flat_values = data.reshape(-1)
    flat_states = states.reshape(-1)
    flat_content = content.reshape(-1, n_hashes)

    for index, state in enumerate(flat_states):
        if state != Tokens.valued.value:
            continue

        value: Any = flat_values[index]
        if not isinstance(value, Hashable):
            raise TypeError(f"static_entity values must be hashable, got {type(value).__name__}")

        for seed in range(n_hashes):
            flat_content[index, seed] = _hash_value(value, seed)

    return content


def _bucketize(content: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """Uniformly quantize raw hash floats in [-1, 1) to bucket ids in [0, n_buckets)."""
    scaled = content.add(1.0).mul(0.5).mul(n_buckets)
    return scaled.floor().long().clamp(min=0, max=n_buckets - 1)


@static_entity.register
class Request(RequestBase):
    """Hash-based entity tensorfield with no learned identity vocabulary.

    Content is a static function of the input value: `n_hashes` independent
    deterministic hash functions produce a fixed-length feature vector per
    slot, which is then expanded through the same Fourier feature bands used
    by `Number` (see `n_bands`, `offset`). No embedding table is created;
    all identity information is a fixed factor of the input value.

    For reconstruction loss, each hash channel is quantized into `n_buckets`
    uniform bins over `[-1, 1)`, and a per-channel cross-entropy trains the
    decoder to identify the correct bucket. Effective identity fingerprint
    capacity is `n_buckets ** n_hashes`.

    Setting `group="<name>"` shares the encoder Fourier projection and the
    decoder content head across every `static_entity` address with the same
    group name, so the same input value produces identical embeddings and
    reconstruction targets at every address in the group. All grouped
    addresses must share `n_hashes`, `n_bands`, `offset`, and `n_buckets`.
    """

    type: Literal["static_entity"] = "static_entity"
    n_hashes: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    n_buckets: Annotated[int, pydantic.Field(gt=1, default=4)] = 4
    group: str | None = None


@static_entity.register
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

        try:
            content_np = _static_hash_content(data=data, states=states, n_hashes=n_hashes)
        except TypeError as error:
            raise ValueError(f"static_entity field at '{address}' only accepts hashable scalar values") from error

        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = torch.tensor(states, dtype=torch.int64).masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.tensor(content_np, dtype=torch.float)
        content = content.masked_fill(literal_mask_tensor.unsqueeze(-1), 0.0)

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
        content = torch.zeros((*shape, request.n_hashes), dtype=torch.float)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@static_entity.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.n_hashes: int = request.n_hashes

        n_bands = request.n_bands
        offset = request.offset
        weights = torch.logspace(start=-n_bands, end=offset, steps=n_bands + offset + 1, base=2).mul(math.pi)
        self.register_buffer("weights", weights.reshape(1, 1, -1))
        self.weights: torch.Tensor

        self.register_buffer("state_eye", torch.eye(len(Tokens)))
        self.state_eye: torch.Tensor

        n_features = self.n_hashes * 2 * weights.numel() + len(Tokens)

        group = request.group
        if group is not None:
            state = _group_state(schema, group)
            if state["linear"] is None:
                state["linear"] = torch.nn.Linear(n_features, schema.d_model)
            self.linear: torch.nn.Linear = state["linear"]
        else:
            self.linear = torch.nn.Linear(n_features, schema.d_model)

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1, self.n_hashes)

        # Interleave hashes with Fourier bands: (K, n_hashes, 1) * (1, 1, F) -> (K, n_hashes, F)
        weighted = content.unsqueeze(-1).mul(self.weights)
        fourier = torch.cat([torch.sin(weighted), torch.cos(weighted)], dim=-1)
        fourier = fourier.reshape(fourier.shape[0], -1)

        state_onehot = self.state_eye.index_select(dim=0, index=state)
        features = torch.cat([fourier, state_onehot], dim=-1)

        projection = torch.nn.functional.gelu(self.linear(features)).reshape(N, *dims, -1)

        return Parcel(
            payload=projection,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@static_entity.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.n_hashes: int = request.n_hashes
        self.n_buckets: int = request.n_buckets
        self.state_linear = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))

        out_features = request.n_hashes * request.n_buckets
        group = request.group
        if group is not None:
            state = _group_state(schema, group)
            if state["content_linear"] is None:
                state["content_linear"] = torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=out_features,
                )
            self.content_linear: torch.nn.Linear = state["content_linear"]
        else:
            self.content_linear = torch.nn.Linear(
                in_features=schema.d_model,
                out_features=out_features,
            )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.state_linear(pooled),
                TensorKey.content: self.content_linear(pooled),
            }
        )


@static_entity.register
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

    # Per-hash categorical over deterministic quantile buckets: entity-style CE per channel.
    inputs = prediction.payload[TensorKey.content].reshape(N * n_hashes, n_buckets)
    raw_targets = batch.targets[TensorKey.content].reshape(N, n_hashes)
    bucket_targets = _bucketize(raw_targets, n_buckets=n_buckets).reshape(N * n_hashes)

    per_hash_ce = torch.nn.functional.cross_entropy(
        input=inputs,
        target=bucket_targets,
        reduction="none",
    )

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=(
            per_hash_ce.reshape(N, n_hashes)
            .mean(dim=-1)
            .masked_select(valued)
            .mean()
        ),
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


@static_entity.register
def write(module: Model, prediction: Prediction):
    return None
