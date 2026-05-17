from __future__ import annotations

import enum
import math
import warnings
from functools import cache
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.architecture.counter import Counter
from json2vec.data.processing import apply, pad
from json2vec.structs.enums import Metric, Strata, TensorKey, Tokens
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Hyperparameters


text: Plugin = Plugin(name="text")

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TEXT_TENSOR_KEYS = (INPUT_IDS, ATTENTION_MASK)


class Pooling(enum.StrEnum):
    cls = "cls"
    mean = "mean"
    pooler = "pooler"


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"


def _import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        message = (
            "text tensorfield requires the optional dependency `transformers`. "
            "Install it explicitly before using `type: text` fields; it is not installed by default."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise ImportError(message) from error

    return AutoModel, AutoTokenizer


@cache
def _get_tokenizer(
    model_name: str,
    revision: str | None,
    local_files_only: bool,
):
    _, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            raise ValueError(f"text model '{model_name}' tokenizer does not define a pad/eos/sep token")

    return tokenizer


_HF_MODELS: dict[tuple[str, str | None, bool], Any] = {}


def _get_model(
    model_name: str,
    revision: str | None,
    local_files_only: bool,
):
    key = (model_name, revision, local_files_only)

    if key not in _HF_MODELS:
        AutoModel, _ = _import_transformers()
        model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        model.eval()
        model.requires_grad_(False)
        _HF_MODELS[key] = model

    return _HF_MODELS[key]


def _hidden_size(
    model_name: str,
    revision: str | None,
    local_files_only: bool,
) -> int:
    hidden_size = getattr(_get_model(model_name, revision, local_files_only).config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"text model '{model_name}' does not expose `config.hidden_size`")

    return int(hidden_size)


def _objective_loss(inputs: torch.Tensor, targets: torch.Tensor, objective: Objective) -> torch.Tensor:
    if objective == Objective.l1:
        return torch.nn.functional.l1_loss(input=inputs, target=targets, reduction="none").mean(dim=1)

    return torch.nn.functional.mse_loss(input=inputs, target=targets, reduction="none").mean(dim=1)


def coerce_text(value: Any, *, address: Address) -> str:
    if not isinstance(value, str):
        raise ValueError(f"text field at '{address}' expects string values, got {type(value).__name__}")

    return value


@text.register
class Request(RequestBase):
    type: Literal["text"] = "text"
    model_name: str
    max_length: Annotated[int, pydantic.Field(gt=0, default=128)] = 128
    encoder_batch_size: Annotated[int, pydantic.Field(gt=0, default=32)] = 32
    encoder_pooling: Pooling = Pooling.cls
    objective: Objective = Objective.l2
    revision: str | None = None
    local_files_only: bool = False

    @pydantic.field_validator("model_name", mode="before")
    @classmethod
    def normalize_model_name(cls, value: str):
        if not isinstance(value, str):
            raise ValueError("model_name must be a string")

        return value.strip()

    @pydantic.field_validator("revision", mode="before")
    @classmethod
    def normalize_revision(cls, value: str | None):
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("revision must be a string when provided")

        normalized = value.strip()
        return normalized or None

    @pydantic.model_validator(mode="after")
    def check_model_name(self):
        if not self.model_name:
            raise ValueError("model_name must be a non-empty string")

        return self


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
        hyperparameters: Hyperparameters,
        strata: Strata,
        state: Any,
    ) -> TensorFieldBase:
        request: Request = hyperparameters.requests[address]
        array_shape: tuple[int, ...] = hyperparameters.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)

        coerced = apply(
            values,
            coerce_text,
            address=address,
            leaf_depth=len(leading_shape),
        )

        data, state = pad(
            nested=coerced,
            shape=leading_shape,
            dtype=object,
            pad_value=None,
        )

        token_ids = torch.zeros((*leading_shape, request.max_length), dtype=torch.int64)
        attention_mask = torch.zeros_like(token_ids)
        state_tensor = torch.tensor(state, dtype=torch.int64)

        valued = state == Tokens.valued.value
        if valued.any():
            tokenizer = _get_tokenizer(
                request.model_name,
                request.revision,
                request.local_files_only,
            )
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

    def _cache_targets(self):
        if TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

    def _zero_content(self, selected: torch.Tensor):
        expanded = selected.unsqueeze(-1).expand_as(self.content[INPUT_IDS])

        for key in TEXT_TENSOR_KEYS:
            self.content[key] = self.content[key].masked_scatter(
                expanded,
                torch.zeros_like(input=self.content[key]),
            )

    def mask(self, p_mask: float):
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked.value)
        is_masked = torch.rand_like(input=self.state, dtype=torch.float).lt(other=p_mask)

        self._cache_targets()

        self.state = self.state.masked_scatter(is_masked, mask_token)
        self._zero_content(is_masked)

        self.trainable |= is_masked

    def target(self, p_target: float = 1.0):
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked.value)
        is_targeted = (
            torch.rand(self.state.size(0), *([1] * (len(self.state.shape) - 1)), device=self.state.device)
            .lt(p_target)
            .expand_as(self.state)
        )
        self._cache_targets()

        self.state = self.state.masked_scatter(is_targeted, mask_token)
        self._zero_content(is_targeted)

        self.trainable |= is_targeted

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        hyperparameters: Hyperparameters,
    ):
        request: Request = hyperparameters.requests[address]
        leading_shape: tuple[int, ...] = (batch_size, *hyperparameters.shapes[address])
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
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]

        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.hidden_size: int = _hidden_size(request.model_name, request.revision, request.local_files_only)
        self.request = request

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=hyperparameters.d_model,
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=hyperparameters.d_model),
            torch.nn.GELU(),
        )

        self.__dict__["_hf_model"] = _get_model(request.model_name, request.revision, request.local_files_only)
        self.__dict__["_hf_device"] = None

    @property
    def hf_model(self):
        return self.__dict__["_hf_model"]

    def _ensure_hf_model_device(self, device: torch.device) -> Any:
        current: torch.device | None = self.__dict__["_hf_device"]
        if current != device:
            self.hf_model.to(device)
            self.__dict__["_hf_device"] = device

        self.hf_model.eval()
        return self.hf_model

    def _pool(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.request.encoder_pooling == Pooling.pooler:
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise ValueError(f"text model '{self.request.model_name}' does not expose pooler_output")
            return pooled

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError(f"text model '{self.request.model_name}' does not expose last_hidden_state")

        if self.request.encoder_pooling == Pooling.cls:
            return hidden[:, 0]

        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def _encode_flat(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() == 0:
            return torch.zeros((0, self.hidden_size), device=token_ids.device, dtype=torch.float32)

        model = self._ensure_hf_model_device(token_ids.device)

        encoded: list[torch.Tensor] = []
        for start in range(0, token_ids.size(0), self.request.encoder_batch_size):
            stop = start + self.request.encoder_batch_size

            with torch.inference_mode():
                outputs = model(
                    input_ids=token_ids[start:stop],
                    attention_mask=attention_mask[start:stop],
                )

            encoded.append(self._pool(outputs, attention_mask[start:stop]).to(dtype=torch.float32))

        return torch.cat(encoded, dim=0)

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

        if valued.any():
            embeddings[valued] = self._encode_flat(
                token_ids=flat_ids[valued],
                attention_mask=flat_mask[valued],
            )

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
            self.target_embeddings(inputs)

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
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__(hyperparameters=hyperparameters, address=address)

        request: Request = hyperparameters.requests[address]
        hidden_size = _hidden_size(request.model_name, request.revision, request.local_files_only)
        self.classification = torch.nn.Linear(
            in_features=hyperparameters.d_model,
            out_features=len(Tokens),
        )
        self.linear = torch.nn.Linear(
            in_features=hyperparameters.d_model,
            out_features=hidden_size,
        )
        self.counter = Counter(address=address, size=len(Tokens))

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
    module: JSON2Vec,
    prediction: Prediction,
    batch: TensorField,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = module.hyperparameters.requests[address]
    embedder: Embedder = module.nodes[address].embedder
    decoder: Decoder = module.nodes[address].decoder

    trainable = batch.trainable.reshape(-1)
    state_targets = batch.targets[TensorKey.state].reshape(-1)
    state_inputs = prediction.payload[TensorKey.state].reshape(-1, len(Tokens))
    decoder.counter(batch.targets[TensorKey.state])

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                weight=decoder.counter.weight,
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
    targets = embedder.target_embeddings(batch).reshape(-1, embedder.hidden_size)
    diff = inputs.subtract(targets)

    loss += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=_objective_loss(inputs=inputs, targets=targets, objective=request.objective).masked_select(valued).mean(),
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


@text.register
def write(module: JSON2Vec, prediction: Prediction):
    pass
