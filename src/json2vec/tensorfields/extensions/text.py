# ty: ignore[unknown-argument]
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, cast

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
from json2vec.tensorfields.shared.counter import Counter, CounterUpdateCallback

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Schema


text: Plugin = Plugin(name="text")
text.callback(CounterUpdateCallback)

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TEXT_TENSOR_KEYS = (INPUT_IDS, ATTENTION_MASK)
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
                raise ImportError("Text requires `transformers`; install `json2vec[text]`.")

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
                raise ImportError("Text requires `transformers`; install `json2vec[text]`.")

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


@text.register
@tensorclass
class TensorField(TensorFieldBase):
    content: TensorDict[str, torch.Tensor]
    state: torch.Tensor
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
        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )

        data, state = pad(
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

        token_ids = torch.zeros((*leading_shape, request.max_length), dtype=torch.int64)
        attention_mask = torch.zeros_like(token_ids)
        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = torch.tensor(state, dtype=torch.int64).masked_fill(literal_mask_tensor, Tokens.masked.value)

        valued = state == Tokens.valued.value
        if valued.any():
            invalid = next((value for value in data[valued].flat if not isinstance(value, str)), None)
            if invalid is not None:
                raise ValueError(f"text field at '{address}' expects string values, got {type(invalid).__name__}")

            tokenizer = CachedModel.get_tokenizer(request.model)
            encoded = tokenizer(
                data[valued].tolist(),
                padding="max_length",
                truncation=True,
                max_length=request.max_length,
                return_tensors="pt",
            )

            valued_index = torch.from_numpy(valued.astype(bool))
            token_ids[valued_index] = encoded[INPUT_IDS].to(dtype=torch.int64)
            attention_mask[valued_index] = encoded[ATTENTION_MASK].to(dtype=torch.int64)

        return cls(
            state=state_tensor,
            content=TensorDict(
                {
                    INPUT_IDS: token_ids,
                    ATTENTION_MASK: attention_mask,
                },
                batch_size=leading_shape,
            ),
            trainable=torch.zeros(leading_shape, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked.value)

        if cache_targets:
            if TensorKey.state not in self.targets.keys():
                self.targets[TensorKey.state] = self.state.clone()
            if TensorKey.content not in self.targets.keys():
                self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_scatter(selected, mask_token)
        expanded = selected.unsqueeze(-1).expand_as(self.content[INPUT_IDS])
        for key in TEXT_TENSOR_KEYS:
            self.content[key] = self.content[key].masked_scatter(
                expanded,
                torch.zeros_like(input=self.content[key]),
            )

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
        leading_shape: tuple[int, ...] = (batch_size, *schema.shapes[address])
        token_shape: tuple[int, ...] = (*leading_shape, request.max_length)
        state = torch.full(leading_shape, Tokens.masked, dtype=torch.int64)

        return cls(
            state=state,
            content=TensorDict(
                {
                    INPUT_IDS: torch.zeros(token_shape, dtype=torch.int64),
                    ATTENTION_MASK: torch.zeros(token_shape, dtype=torch.int64),
                },
                batch_size=leading_shape,
            ),
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
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
        encoded: list[torch.Tensor] = []
        for start in range(0, valued_ids.size(0), self.request.encoder_batch_size):
            stop = start + self.request.encoder_batch_size
            batch_mask = valued_mask[start:stop]

            with torch.inference_mode():
                outputs = model(
                    input_ids=valued_ids[start:stop],
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

        embeddings[valued] = torch.cat(encoded, dim=0)

        return embeddings.reshape(N, *dims, self.hidden_size)

    @beartype
    def target_embeddings(self, inputs: TensorField) -> torch.Tensor:
        if TensorKey.embedding not in inputs.targets.keys():
            # Targets hold the original tokenized text; masked inputs have these tensors zeroed.
            inputs.targets[TensorKey.embedding] = self.encode(
                content=inputs.targets[TensorKey.content],
                state=inputs.targets[TensorKey.state],
            )

        return inputs.targets[TensorKey.embedding]

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        if TensorKey.content in inputs.targets.keys() and TensorKey.state in inputs.targets.keys():
            self.target_embeddings(cast(TensorField, inputs))

        state = inputs.state.reshape(D)
        valued = state.eq(Tokens.valued.value).unsqueeze(-1)
        encoded = self.encode(content=inputs.content, state=inputs.state).reshape(D, self.hidden_size)
        projected = self.linear(encoded) * valued
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(embeddings + projected).reshape(N, *dims, -1),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


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
) -> list[torch.Tensor]:
    address: Address = prediction.address
    request: Request = module.schema.requests[address]
    embedder: Embedder = module.nodes[address].embedder

    trainable = batch.trainable.reshape(-1)
    state_targets = batch.targets[TensorKey.state].reshape(-1)
    state_inputs = prediction.payload[TensorKey.state].reshape(-1, len(Tokens))

    state_loss: torch.Tensor = module.track(
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
        return [state_loss]

    inputs = prediction.payload[TensorKey.content].reshape(-1, embedder.hidden_size)
    targets = embedder.target_embeddings(batch).reshape(-1, embedder.hidden_size)
    diff = inputs.subtract(targets)

    content_loss = module.track(
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

    return [state_loss, content_loss]


@text.register
def write(module: Model, prediction: Prediction):
    return None
