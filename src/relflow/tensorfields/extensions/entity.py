# ty: ignore[unknown-argument]
from __future__ import annotations

import math
import weakref
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
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


entity: Plugin = Plugin(name="entity")

_GROUP_STATE: dict[tuple[int, str], dict[str, Any]] = {}


def _group_state(schema: Schema, group: str) -> dict[str, Any]:
    key = (id(schema), group)
    cached = _GROUP_STATE.get(key)
    if cached is not None:
        return cached

    addresses = sorted(
        (
            address
            for address, request in schema.active_requests.items()
            if request.type == "entity" and getattr(request, "group", None) == group
        ),
        key=str,
    )
    if not addresses:
        raise ValueError(f"no active entity fields configured for group '{group}'")

    max_slots = sum(math.prod(schema.shapes[address]) for address in addresses)

    state: dict[str, Any] = {
        "addresses": tuple(addresses),
        "max_slots": max_slots,
        "canonical": addresses[0],
        "embedding": None,
        "projection": None,
    }
    _GROUP_STATE[key] = state
    weakref.finalize(schema, _GROUP_STATE.pop, key, None)
    return state


def _local_reindex(
    data: np.ndarray,
    states: np.ndarray,
    shared_vocab: list[dict[Hashable, int]] | None = None,
) -> np.ndarray:
    tokens = np.zeros_like(states, dtype=np.int64)

    for observation_index in range(data.shape[0]):
        vocab: dict[Hashable, int]
        if shared_vocab is not None:
            vocab = shared_vocab[observation_index]
        else:
            vocab = {}
        flat_values = data[observation_index].reshape(-1)
        flat_states = states[observation_index].reshape(-1)
        flat_tokens = tokens[observation_index].reshape(-1)

        for index, state in enumerate(flat_states):
            if state != Tokens.valued.value:
                continue

            value: Any = flat_values[index]
            if not isinstance(value, Hashable):
                raise TypeError(f"entity values must be hashable, got {type(value).__name__}")

            local_id = vocab.setdefault(value, len(vocab))
            flat_tokens[index] = local_id

    return tokens


# TODO try providing just the structural map as it's own branch so that the information and graphical traversals aren't lost in the upwards cross-attentions.


def build_shared_group_vocabs(
    batch: Any,
    schema: Schema,
    strata: Strata,
    target_addresses: set[Address] | frozenset[Address],
) -> dict[str, list[dict[Hashable, int]]]:
    """Pre-populate per-observation shared vocabs for every grouped entity.

    Called once per batch by the encode pipeline. Runs the same JMESPath and
    padding path each grouped ``TensorField.new`` would run, then unions the
    valued cells per observation into one dict per group.
    """
    from relflow.data.iterables import compile_query_extractor, query as _jmes_query  # noqa: I001

    groups: dict[str, list[Address]] = {}
    for address, request in schema.active_requests.items():
        if request.type != "entity":
            continue
        group = getattr(request, "group", None)
        if group is None:
            continue
        groups.setdefault(group, []).append(address)

    if not groups:
        return {}

    batch_size = len(batch)
    result: dict[str, list[dict[Hashable, int]]] = {}

    for group_name, addresses in groups.items():
        vocabs: list[dict[Hashable, int]] = [{} for _ in range(batch_size)]

        for address in sorted(addresses, key=str):
            if strata == Strata.predict and address in target_addresses:
                continue

            request = schema.active_requests[address]
            expression = request.query
            if expression is None:
                raise ValueError(f"request '{address}' must define query")

            extractor = compile_query_extractor(expression)
            values = extractor(batch) if extractor is not None else _jmes_query(expression).search(batch)

            leading_shape = (batch_size, *schema.shapes[address])
            cleaned, _ = extract_mask_literals(
                values,
                strata=strata,
                address=address,
                leaf_depth=len(leading_shape),
            )
            data, states = pad(
                nested=cleaned,
                shape=leading_shape,
                dtype=object,
                pad_value=None,
                overflows=schema.overflows(address),
                address=address,
            )

            for observation_index in range(batch_size):
                flat_values = data[observation_index].reshape(-1)
                flat_states = states[observation_index].reshape(-1)
                vocab = vocabs[observation_index]
                for index, state in enumerate(flat_states):
                    if state != Tokens.valued.value:
                        continue
                    value = flat_values[index]
                    if not isinstance(value, Hashable):
                        raise TypeError(f"entity values must be hashable, got {type(value).__name__}")
                    vocab.setdefault(value, len(vocab))

        result[group_name] = vocabs

    return result


@entity.register
class Request(RequestBase):
    """Per-observation entity tensorfield request for local identity matching."""

    type: Literal["entity"] = "entity"
    topk: list[int] | None = None
    group: str | None = None

    @pydantic.model_validator(mode="after")
    def check_topk(self):
        if self.topk is None:
            self.topk = []

        for topk in self.topk:
            if not isinstance(topk, int):
                raise ValueError("topk values must be integers")

            if topk <= 0:
                raise ValueError("topk values must be positive")

            if topk == 1:
                raise ValueError("topk values must not be 1")

        return self

    def post_bind_validate(self):
        max_slots: int = math.prod(self.shape)
        if self.group is None and max_slots <= 1:
            raise ValueError(
                f"entity field at '{self.address}' requires at least 2 elements per observation, "
                f"but configured count is {max_slots}"
            )

        for topk in self.topk or []:
            if topk >= max_slots and self.group is None:
                raise ValueError("topk values must be less than the entity slot count")


@entity.register
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
        shared_vocab: list[dict[Hashable, int]] | None = None,
    ) -> TensorFieldBase:
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
            tokens = _local_reindex(data=data, states=states, shared_vocab=shared_vocab)
        except TypeError as error:
            raise ValueError(f"entity field at '{address}' only accepts hashable scalar values") from error

        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = torch.tensor(states, dtype=torch.int64).masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.tensor(tokens, dtype=torch.int64)
        content = content.masked_fill(literal_mask_tensor, 0)

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


@entity.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.origin: Address = address
        self.destination: Address = schema.requests[address].parent.address

        group = getattr(schema.requests[address], "group", None)
        if group is not None:
            state = _group_state(schema, group)
            self.max_slots: int = state["max_slots"]
            if state["embedding"] is None:
                state["embedding"] = torch.nn.Embedding(
                    num_embeddings=self.max_slots,
                    embedding_dim=schema.d_model,
                )
            content_embedding: torch.nn.Embedding = state["embedding"]
        else:
            self.max_slots = math.prod(schema.shapes[address])
            content_embedding = torch.nn.Embedding(
                num_embeddings=self.max_slots,
                embedding_dim=schema.d_model,
            )

        self.n_embeddings: int = self.max_slots + len(Tokens)

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.content.name: content_embedding,
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

        if valued.any() and (content.masked_select(valued) >= self.max_slots).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds bucket space of {self.max_slots}")

        safe_content = content.masked_fill(~valued, 0)
        embeddings: torch.Tensor = (
            self.embeddings[TensorKey.state.name](state)
            + self.embeddings[TensorKey.content.name](safe_content) * valued.unsqueeze(-1)
        ).reshape(N, *dims, -1)

        return Parcel(
            payload=embeddings,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@entity.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        group = getattr(schema.requests[address], "group", None)
        if group is not None:
            state = _group_state(schema, group)
            self.max_slots: int = state["max_slots"]
            if state["projection"] is None:
                state["projection"] = torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=self.max_slots,
                )
            projection: torch.nn.Linear = state["projection"]
        else:
            self.max_slots = math.prod(schema.shapes[address])
            projection = torch.nn.Linear(in_features=schema.d_model, out_features=self.max_slots)

        self.state_linear = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))
        self.projection = projection

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.state_linear(pooled),
                TensorKey.content: self.projection(pooled),
            }
        )


@entity.register
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

    inputs = prediction.payload[TensorKey.content].reshape(N, -1)
    targets = batch.targets[TensorKey.content].reshape(N)
    n_content_tokens = inputs.shape[-1]
    invalid = valued & targets.ge(n_content_tokens)
    if invalid.any():
        raise ValueError(f"Token in address {prediction.address} exceeds entity slot count")

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.content),
        value=(
            torch.nn.functional.cross_entropy(
                input=inputs,
                target=targets,
                reduction="none",
            )
            .masked_select(valued)
            .mean()
        ),
    )

    for topk in module.schema.requests[prediction.address].topk:
        if topk >= inputs.shape[1]:
            continue

        module.track(
            (prediction.address, strata, Metric.accuracy, f"top{topk}"),
            value=(
                inputs.topk(k=topk, dim=1)
                .indices.eq(targets.unsqueeze(1))
                .any(dim=1)
                .masked_select(valued)
                .float()
                .mean()
            ),
        )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=inputs.argmax(dim=1).eq(targets).masked_select(valued).float().mean(),
    )

    return loss


@entity.register
def write(module: Model, prediction: Prediction):
    return None
