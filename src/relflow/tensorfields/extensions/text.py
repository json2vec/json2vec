# ty: ignore[unknown-argument]
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.helpers import Jitter
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
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


text: Extension = Extension(name="text", types=(str,))
text.callback(CounterUpdateCallback)

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
DEFAULT_TEXT_MODEL = "google/bert_uncased_L-2_H-128_A-2"


class Pooling(enum.StrEnum):
    cls = "cls"
    mean = "mean"
    pooler = "pooler"


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        if self == Objective.l1:
            diff = diff.absolute()
        else:
            diff = diff.square()

        return diff.mean(dim=1)


@dataclass
class CachedModel:
    _models: ClassVar[dict[str, "CachedModel"]] = {}
    _tokenizers: ClassVar[dict[str, Any]] = {}

    key: str
    model: Any
    hidden_size: int
    device: torch.device | None = None

    @classmethod
    def get_model(
        cls,
        key: str,
    ) -> "CachedModel":
        if key not in cls._models:
            try:
                from transformers import AutoModel  # ty:ignore[unresolved-import]
            except ImportError:
                raise ImportError("Text requires `transformers`; install `relflow[text]`.")

            model = AutoModel.from_pretrained(key)
            model.eval()
            model.requires_grad_(False)
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(f"text model '{key}' does not expose `config.hidden_size`")
            cls._models[key] = cls(key=key, model=model, hidden_size=int(hidden_size))

        return cls._models[key]

    @classmethod
    def get_tokenizer(cls, key: str):
        if key not in cls._tokenizers:
            try:
                from transformers import AutoTokenizer  # ty:ignore[unresolved-import]
            except ImportError:
                raise ImportError("Text requires `transformers`; install `relflow[text]`.")

            tokenizer = AutoTokenizer.from_pretrained(key)

            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.sep_token is not None:
                    tokenizer.pad_token = tokenizer.sep_token
                else:
                    raise ValueError(f"text model '{key}' tokenizer does not define a pad/eos/sep token")
            cls._tokenizers[key] = tokenizer

        return cls._tokenizers[key]

    def module_for(self, device: torch.device) -> Any:
        target = torch.device(device)
        if self.device != target:
            self.model.to(target)
            self.device = target

        self.model.eval()
        return self.model


@text.register
class Request(RequestBase):
    """Text tensorfield request encoded by a frozen Hugging Face model."""

    model_config = pydantic.ConfigDict(extra="allow", str_strip_whitespace=True)

    type: Literal["text"] = "text"
    model: Annotated[str, pydantic.Field(min_length=1)] = DEFAULT_TEXT_MODEL
    max_length: Annotated[int, pydantic.Field(gt=0, default=128)] = 128
    encoder_batch_size: Annotated[int, pydantic.Field(gt=0, default=32)] = 32
    encoder_pooling: Pooling = Pooling.cls
    objective: Objective = Objective.l2
    jitter: Jitter = pydantic.Field(default_factory=Jitter)


@text.register
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Count pristine text states without tokenizing hidden content."""

    if not learn:
        return None
    return TensorDict(
        {TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens))},
        batch_size=[],
    )


@text.register
@tensorclass
class TensorField(TensorFieldBase):
    content: TensorDict[str, torch.Tensor]
    state: torch.Tensor
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

        def encode(field: RaggedField) -> TensorDict[str, torch.Tensor]:
            values = field.values.to_pylist()
            if values:
                tokenizer = CachedModel.get_tokenizer(request.model)
                encoded = tokenizer(
                    values,
                    padding="max_length",
                    truncation=True,
                    max_length=request.max_length,
                    return_tensors="pt",
                )
                encoded_ids = encoded[INPUT_IDS].to(dtype=torch.int64).numpy()
                encoded_mask = encoded[ATTENTION_MASK].to(dtype=torch.int64).numpy()
            else:
                encoded_ids = np.empty((0, request.max_length), dtype=np.int64)
                encoded_mask = np.empty((0, request.max_length), dtype=np.int64)

            return TensorDict(
                {
                    INPUT_IDS: torch.from_numpy(field.place(encoded_ids, fill=0, value_shape=(request.max_length,))),
                    ATTENTION_MASK: torch.from_numpy(
                        field.place(encoded_mask, fill=0, value_shape=(request.max_length,))
                    ),
                },
                batch_size=field.shape,
            )

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


@text.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]

        self.origin: Address = address
        self.destination: Address = request.parent.address
        cached_model = CachedModel.get_model(request.model)
        self.__dict__["_cached_model"] = cached_model
        self.hidden_size: int = cached_model.hidden_size
        self.request = request
        self.jitter: Jitter = request.jitter
        self.padding_side: str = getattr(CachedModel.get_tokenizer(request.model), "padding_side", "right")

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )
        self.counter = Counter(address=address, size=len(Tokens))
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=schema.d_model),
            torch.nn.GELU(),
        )

    @beartype
    def encode(
        self,
        content: TensorDict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        N, *dims = state.shape
        D = math.prod((N, *dims))

        flat_state = state.reshape(D)
        flat_ids = content[INPUT_IDS].reshape(D, -1)
        flat_mask = content[ATTENTION_MASK].reshape(D, -1)

        embeddings = torch.zeros((D, self.hidden_size), device=flat_ids.device, dtype=torch.float32)
        valued = flat_state.eq(Tokens.valued.value)
        if not valued.any():
            return embeddings.reshape(N, *dims, self.hidden_size)

        valued_ids = flat_ids[valued]
        valued_mask = flat_mask[valued]
        model = self.__dict__["_cached_model"].module_for(flat_ids.device)

        restore_order: torch.Tensor | None = None
        starts = range(0, valued_ids.size(0), self.request.encoder_batch_size)
        stops = [min(start + self.request.encoder_batch_size, valued_ids.size(0)) for start in starts]
        if self.padding_side == "right":
            lengths = valued_mask.count_nonzero(dim=-1)
            restore_order = torch.argsort(lengths, stable=True)
            valued_ids = valued_ids.index_select(0, restore_order)
            valued_mask = valued_mask.index_select(0, restore_order)
            sorted_lengths = lengths.index_select(0, restore_order)

            # Copy every microbatch boundary length to the host in one
            # synchronization, rather than calling .item() before each encoder
            # invocation. A valued row with no attended tokens is forwarded at
            # full width to preserve the previous cls/pooler behavior.
            metadata_indexes = torch.as_tensor(
                [*starts, *(stop - 1 for stop in stops)],
                device=sorted_lengths.device,
            )
            metadata = sorted_lengths.index_select(0, metadata_indexes).tolist()
            boundary = len(stops)
            chunk_minimums = metadata[:boundary]
            chunk_maximums = metadata[boundary:]
            widths = [
                valued_ids.size(1) if minimum == 0 else maximum
                for minimum, maximum in zip(chunk_minimums, chunk_maximums, strict=True)
            ]
        else:
            # Removing leading padding changes absolute position ids for many
            # models. Keep the full width and original order for left-padded
            # tokenizers.
            widths = [valued_ids.size(1)] * len(stops)

        encoded: list[torch.Tensor] = []
        for start, stop, width in zip(starts, stops, widths, strict=True):
            batch_ids = valued_ids[start:stop, :width]
            batch_mask = valued_mask[start:stop, :width]

            with torch.inference_mode():
                outputs = model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                )

            if self.request.encoder_pooling == Pooling.pooler:
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    raise ValueError(f"text model '{self.request.model}' does not expose pooler_output")
            else:
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError(f"text model '{self.request.model}' does not expose last_hidden_state")

                if self.request.encoder_pooling == Pooling.cls:
                    pooled = hidden[:, 0]
                else:
                    mask = batch_mask.unsqueeze(-1).to(dtype=hidden.dtype)
                    denom = mask.sum(dim=1).clamp_min(1.0)
                    pooled = (hidden * mask).sum(dim=1) / denom

            encoded.append(pooled.to(dtype=torch.float32))

        batched_embeddings = torch.cat(encoded, dim=0)
        if restore_order is None:
            valued_embeddings = batched_embeddings
        else:
            valued_embeddings = torch.empty_like(batched_embeddings)
            valued_embeddings[restore_order] = batched_embeddings
        embeddings[valued] = valued_embeddings

        return embeddings.reshape(N, *dims, self.hidden_size)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        state = inputs.state.reshape(D)
        valued = state.eq(Tokens.valued.value).unsqueeze(-1)
        encoded = self.encode(content=inputs.content, state=inputs.state)
        encoded = encoded.reshape(D, self.hidden_size)
        if self.training:
            eligible = valued & torch.isfinite(encoded)
            encoded = self.jitter.apply(encoded, eligible)
        projected = self.linear(encoded) * valued
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(embeddings + projected).reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@text.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine text-state counts to the model resource."""

    if strata != Strata.train:
        raise ValueError(f"text learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = module.nodes[address].embedder
    embedder.counter.learn(observation[TensorKey.state])


@text.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        hidden_size = CachedModel.get_model(request.model).hidden_size
        self.classification = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )
        self.linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=hidden_size,
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.linear(pooled),
            }
        )


@text.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorField,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = module.schema.requests[address]
    embedder: Embedder = module.nodes[address].embedder

    trainable = batch.trainable.reshape(-1)
    state_targets = batch.targets[TensorKey.state].reshape(-1)
    state_inputs = prediction.payload[TensorKey.state].reshape(-1, len(Tokens))

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                weight=embedder.counter.weight,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    inputs = prediction.payload[TensorKey.content].reshape(-1, embedder.hidden_size)
    if TensorKey.embedding not in batch.targets.keys():
        batch.targets[TensorKey.embedding] = embedder.encode(
            content=batch.targets[TensorKey.content],
            state=batch.targets[TensorKey.state],
        )
    targets = batch.targets[TensorKey.embedding].reshape(-1, embedder.hidden_size)
    diff = inputs.subtract(targets)

    loss += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=request.objective.loss(diff).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.mae, TensorKey.content),
        value=diff.absolute().mean(dim=1).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.rmse, TensorKey.content),
        value=diff.square().mean(dim=1).sqrt().masked_select(valued).mean(),
    )

    return loss
