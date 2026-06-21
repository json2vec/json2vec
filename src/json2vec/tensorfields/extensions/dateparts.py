# ty: ignore[invalid-argument-type,invalid-assignment,unknown-argument,unresolved-attribute]
from __future__ import annotations

import difflib
import enum
import math
import re
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal

import numpy as np
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from json2vec.data.nested import apply, extract_mask_literals, pad
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


class DatePart(enum.StrEnum):
    day_of_year = "day_of_year"
    week_of_year = "week_of_year"
    month_of_year = "month_of_year"
    day_of_month = "day_of_month"
    week_of_month = "week_of_month"
    day_of_week = "day_of_week"
    hour_of_day = "hour_of_day"
    minute_of_hour = "minute_of_hour"

    def register(self, depth: int):
        cls = type(self)

        # Lazy initialization
        if not hasattr(cls, "REGISTRY"):
            cls.REGISTRY: dict[DatePart, Callable[..., Any]] = {}

        if not hasattr(cls, "DEPTH"):
            cls.DEPTH: dict[DatePart, int] = {}

        def decorator(func: Callable[..., Any]):
            if self in cls.REGISTRY:
                raise ValueError(f"{self.name} already has a registered function.")

            cls.REGISTRY[self] = func
            cls.DEPTH[self] = depth

            return func

        return decorator

    def __call__(self, *args, **kwargs):
        func = getattr(type(self), "REGISTRY", {}).get(self)

        if func is None:
            raise RuntimeError(f"No function registered for {self.name}")

        return func(*args, **kwargs)

    @classmethod
    def depth(cls, datepart: DatePart) -> int:
        return cls.DEPTH[datepart]


def _normalize_datepart_key(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    return value.strip("_").casefold()


def _datepart_lookup() -> dict[str, DatePart]:
    lookup: dict[str, DatePart] = {}
    for datepart in DatePart:
        normalized = _normalize_datepart_key(datepart.value)
        lookup[normalized] = datepart
        lookup[normalized.replace("_", "")] = datepart

    return lookup


@DatePart.day_of_month.register(depth=31)
def _(arr: np.ndarray) -> np.ndarray:
    month_start = arr.astype("datetime64[M]")
    return (arr - month_start).astype("timedelta64[D]").astype(int) + 1


@DatePart.day_of_year.register(depth=366)
def _(arr: np.ndarray) -> np.ndarray:
    year_start = arr.astype("datetime64[Y]")
    return (arr - year_start).astype("timedelta64[D]").astype(int) + 1


@DatePart.month_of_year.register(depth=12)
def _(arr: np.ndarray) -> np.ndarray:
    return (arr.astype("datetime64[M]") - arr.astype("datetime64[Y]")).astype(int) + 1


@DatePart.week_of_year.register(depth=53)
def _(arr: np.ndarray) -> np.ndarray:
    year_start = arr.astype("datetime64[Y]")
    return ((arr.astype("datetime64[W]") - year_start.astype("datetime64[W]")).astype(int) + 1).astype(int)


@DatePart.day_of_week.register(depth=7)
def _(arr: np.ndarray) -> np.ndarray:
    return (arr.astype("datetime64[D]").astype(int) + 4) % 7


@DatePart.week_of_month.register(depth=6)
def _(arr: np.ndarray) -> np.ndarray:
    month_start = arr.astype("datetime64[M]")
    month_start_dow = (month_start.astype("datetime64[D]").astype(int) + 4) % 7
    day_offset = (arr - month_start).astype("timedelta64[D]").astype(int)
    return ((day_offset + month_start_dow) // 7) + 1


@DatePart.hour_of_day.register(depth=24)
def _(arr: np.ndarray) -> np.ndarray:
    day_start = arr.astype("datetime64[D]")
    return (arr - day_start).astype("timedelta64[h]").astype(int)


@DatePart.minute_of_hour.register(depth=60)
def _(arr: np.ndarray) -> np.ndarray:
    hour_start = arr.astype("datetime64[h]")
    return (arr - hour_start).astype("timedelta64[m]").astype(int)


dateparts: Plugin = Plugin(name="dateparts")


@dateparts.register
class Request(RequestBase):
    """Date/time tensorfield request that extracts configured calendar parts."""

    type: Literal["dateparts"] = "dateparts"
    dateparts: list[DatePart]
    pattern: Annotated[str | None, pydantic.Field(default=None)] = None

    @pydantic.field_validator("dateparts", mode="before", check_fields=False)
    @classmethod
    def _coerce_dateparts(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value

        lookup = _datepart_lookup()
        canonical = [datepart.value for datepart in DatePart]
        dateparts: list[DatePart] = []
        for item in value:
            if isinstance(item, DatePart):
                dateparts.append(item)
                continue

            if not isinstance(item, str):
                raise ValueError(f"datepart values must be strings, got {type(item).__name__}")

            key = _normalize_datepart_key(item)
            match = lookup.get(key) or lookup.get(key.replace("_", ""))
            if match is not None:
                dateparts.append(match)
                continue

            suggestions = difflib.get_close_matches(key, canonical, n=1)
            suggestion = f"; did you mean '{suggestions[0]}'?" if suggestions else ""
            raise ValueError(f"unknown datepart '{item}'{suggestion}")

        return dateparts

    @pydantic.field_validator("dateparts", check_fields=False)
    @classmethod
    def check_dateparts(cls, v):
        if not v:
            raise ValueError("dateparts cannot be empty")

        if not len(v) == len(set(v)):
            raise ValueError("dateparts must be unique")

        return v

    @pydantic.field_validator("pattern", check_fields=False)
    @classmethod
    def check_date_pattern(cls, v):
        if v is None:
            return v

        regex: re.Pattern = re.compile(r"^(?:%%| %(?:[aAwdbBmyYHIpMSfzZjUWcxXGuV])|[^%])+$", re.VERBOSE)

        if not bool(regex.fullmatch(v)):
            raise ValueError(f"{v} is not a valid format pattern")

        return v


@dateparts.register
@tensorclass
class TensorField(TensorFieldBase):
    state: torch.Tensor
    content: TensorDict[DatePart, torch.Tensor]
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
        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )

        request: RequestBase = schema.requests[address]

        if request.pattern is not None:
            values = apply(values, datetime.strptime, request.pattern)

        data, state = pad(
            nested=values,
            shape=leading_shape,
            dtype="datetime64[m]",
            pad_value=np.nan,
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

        state: torch.Tensor = torch.tensor(data=state, dtype=torch.int64)
        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state = state.masked_fill(literal_mask_tensor, Tokens.masked.value)

        dateparts: dict[DatePart, torch.Tensor] = {}

        for datepart in request.dateparts:
            dateparts[datepart] = (
                torch.tensor(datepart(data))
                .add(other=len(Tokens))
                .masked_scatter(mask=state != Tokens.valued.value, source=state)
            )

        content: TensorDict[DatePart, torch.Tensor] = TensorDict(dateparts)

        return cls(
            content=content,
            state=state,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        mask_token: torch.Tensor = torch.full_like(input=self.state, fill_value=Tokens.masked.value)

        if cache_targets and TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if cache_targets and TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state: torch.Tensor = self.state.masked_scatter(selected, mask_token)

        for datepart in self.content.keys():
            self.content[datepart] = self.content[datepart].masked_scatter(selected, mask_token)

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

        state: torch.Tensor = torch.full(shape, Tokens.masked)

        dateparts: dict[DatePart, torch.Tensor] = {}
        for datepart in schema.requests[address].dateparts:
            dateparts[datepart] = state.clone()

        return cls(
            state=state,
            content=TensorDict(dateparts),
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
        )


@dateparts.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )

        self.dateparts = torch.nn.ModuleDict()

        for datepart in request.dateparts:
            self.dateparts[datepart] = torch.nn.Embedding(
                num_embeddings=len(Tokens) + DatePart.depth(datepart) + 1,
                embedding_dim=schema.d_model,
            )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        embeddings: torch.Tensor = self.embeddings(inputs.state.reshape(D))

        for datepart in self.dateparts:
            embedder: torch.nn.Embedding = self.dateparts[datepart]
            embeddings += embedder(inputs.content[datepart].reshape(D))

        return Parcel(
            payload=embeddings.reshape(N, *dims, -1),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@dateparts.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )

        self.dateparts = torch.nn.ModuleDict()

        for datepart in schema.requests[address].dateparts:
            dim = len(Tokens) + DatePart.depth(datepart) + 1
            self.dateparts[datepart] = torch.nn.Linear(in_features=schema.d_model, out_features=dim)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        content: dict[DatePart, torch.Tensor] = {}
        for datepart in self.dateparts:
            content[datepart] = self.dateparts[datepart](pooled)

        return TensorDict(
            source={
                TensorKey.state: self.linear(pooled),
                TensorKey.content: TensorDict(content, batch_size=pooled.shape[0]),
            }
        )


@dateparts.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    numel: int = batch.targets[TensorKey.state].numel()

    trainable = batch.trainable.reshape(numel)

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=(inputs := prediction.payload[TensorKey.state].reshape(numel, -1)),
                target=(targets := batch.targets[TensorKey.state].reshape(numel)),
                reduction="none",
            )
            .masked_select(mask=trainable)
            .mean()
        ),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=inputs.argmax(dim=1).eq(targets).masked_select(trainable).float().mean(),
    )

    request: RequestBase = module.schema.requests[prediction.address]

    losses: list[torch.Tensor] = []

    for datepart in request.dateparts:
        losses.append(
            module.track(
                (prediction.address, strata, Metric.loss, TensorKey.content, datepart),
                value=(
                    torch.nn.functional.cross_entropy(
                        input=(inputs := prediction.payload[TensorKey.content][datepart].reshape(numel, -1)),
                        target=(targets := batch.targets[TensorKey.content][datepart].reshape(numel)),
                        reduction="none",
                    )
                    .masked_select(trainable)
                    .mean()
                ),
            )
        )

        module.track(
            (prediction.address, strata, Metric.accuracy, TensorKey.content, datepart),
            value=inputs.argmax(dim=1).eq(targets).masked_select(trainable).float().mean(),
        )

    loss += torch.stack(losses).mean()

    return loss


@dateparts.register
def write(module: Model, prediction: Prediction):
    return None
