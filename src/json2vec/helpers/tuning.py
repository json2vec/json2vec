"""Optuna tuning helpers backed by constrained Pydantic models."""

from __future__ import annotations

import enum
import warnings
from collections.abc import Callable
from typing import Annotated, Any, Self, TypeAlias, cast

import pydantic

try:
    import optuna
    from pydantic_optuna_bridge import optuna_config
except ModuleNotFoundError as error:
    if error.name in {"optuna", "pydantic_optuna_bridge"}:
        message = "json2vec.helpers.tuning requires the tuning extra; install with `pip install json2vec[tuning]`."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise ModuleNotFoundError(message) from error
    raise

import json2vec as j2v
from json2vec.architecture.root import Model, OptimizerConfig
from json2vec.helpers.optimizers import adamw

OptimizerTuningFactory: TypeAlias = Callable[..., OptimizerConfig]

__all__ = [
    "Array",
    "Attention",
    "Leaf",
    "Optimizer",
    "OptimizerTuningFactory",
    "Pooling",
    "Root",
    "Toggle",
    "tune",
]


class Attention(str, enum.Enum):
    mha = "mha"
    gqa = "gqa"
    mqa = "mqa"
    none = "none"


class Pooling(str, enum.Enum):
    query = "query"
    mean = "mean"


class Toggle(enum.Enum):
    false = False
    true = True


class _ModelWidth(enum.Enum):
    small = 32
    medium = 64
    large = 128


class _HeadCount(enum.Enum):
    two = 2
    four = 4
    eight = 8


class _Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(use_enum_values=True)

    @classmethod
    def from_trial(cls, trial: optuna.Trial, /, **overrides: Any) -> Self:
        from_optuna_trial = getattr(cls, "from_optuna_trial")
        return cast(Self, from_optuna_trial(_PrefixedTrial(trial, cls.__name__.lower()), **overrides))

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class _PrefixedTrial:
    def __init__(self, trial: optuna.Trial, prefix: str) -> None:
        self.trial = trial
        self.prefix = prefix

    def _name(self, name: str) -> str:
        return f"{self.prefix}.{name}"

    def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
        return self.trial.suggest_float(self._name(name), *args, **kwargs)

    def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
        return self.trial.suggest_int(self._name(name), *args, **kwargs)

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self.trial.suggest_categorical(self._name(name), choices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trial, name)


@optuna_config()
class Array(_Model):
    attention: Attention = Attention.mha
    n_layers: Annotated[int, pydantic.Field(ge=1, le=3)] = 1
    n_heads: _HeadCount = _HeadCount.four
    n_linear: Annotated[int, pydantic.Field(ge=1, le=2)] = 1
    dropout: Annotated[float, pydantic.Field(ge=0.0, le=0.1)] = 0.0


@optuna_config()
class Root(_Model):
    attention: Attention = Attention.mha
    n_layers: Annotated[int, pydantic.Field(ge=1, le=3)] = 1
    n_heads: _HeadCount = _HeadCount.four
    n_linear: Annotated[int, pydantic.Field(ge=1, le=2)] = 1
    dropout: Annotated[float, pydantic.Field(ge=0.0, le=0.1)] = 0.0
    d_model: _ModelWidth = _ModelWidth.medium


@optuna_config()
class Leaf(_Model):
    pooling: Pooling = Pooling.query
    n_heads: _HeadCount = _HeadCount.four
    n_linear: Annotated[int, pydantic.Field(ge=1, le=2)] = 1
    dropout: Annotated[float, pydantic.Field(ge=0.0, le=0.1)] = 0.0
    p_mask: Annotated[float, pydantic.Field(ge=0.0, le=0.15)] = 0.0
    p_prune: Annotated[float, pydantic.Field(ge=0.0, le=0.5)] = 0.0


@optuna_config(log_scale_fields={"learning_rate", "weight_decay", "eps"})
class Optimizer(_Model):
    learning_rate: Annotated[float, pydantic.Field(ge=1e-5, le=1e-2)] = 1e-3
    weight_decay: Annotated[float, pydantic.Field(ge=1e-6, le=1e-1)] = 0.01
    beta1: Annotated[float, pydantic.Field(ge=0.8, le=0.99)] = 0.9
    beta2: Annotated[float, pydantic.Field(ge=0.9, le=0.999)] = 0.95
    eps: Annotated[float, pydantic.Field(ge=1e-9, le=1e-7)] = 1e-8
    decay_bias: Toggle = Toggle.false
    decay_1d: Toggle = Toggle.false


def tune(
    params: j2v.Hyperparameters,
    trial: optuna.Trial,
    optimizer: OptimizerTuningFactory = adamw,
    *,
    batch_size: int = 1,
) -> Model:
    cloned = _copy(params)

    root = Root.from_trial(trial).as_dict()
    d_model = root.pop("d_model")
    cloned.d_model = d_model
    cloned.model_fields_set.add("d_model")
    cloned.update(j2v.where("is_root"), **root)

    cloned.update(j2v.where("type") == "array", include_root=False, **Array.from_trial(trial).as_dict())

    leaf = Leaf.from_trial(trial).as_dict()
    p_prune = leaf.pop("p_prune")
    cloned.update(j2v.where("type") != "array", include_root=False, **leaf)
    cloned.update((j2v.where("type") != "array") & ~j2v.where("target"), include_root=False, p_prune=p_prune)

    _validate_attention_shapes(cloned)

    return Model(
        hyperparameters=_copy(cloned),
        batch_size=batch_size,
        optimizer=optimizer(**Optimizer.from_trial(trial).as_dict()),
    )


def _copy(params: j2v.Hyperparameters) -> j2v.Hyperparameters:
    return j2v.Hyperparameters.model_validate(params.model_dump(mode="python", round_trip=True))


def _validate_attention_shapes(params: j2v.Hyperparameters) -> None:
    def validate(label: str, n_heads: int, *, require_divisible: bool) -> None:
        if n_heads <= 0 or n_heads % 2 != 0:
            raise ValueError(f"{label}.n_heads must be a positive even integer")
        if require_divisible and params.d_model % n_heads != 0:
            raise ValueError(f"{label}.n_heads={n_heads} must divide d_model={params.d_model}")

    root = params.fields
    validate("root", root.n_heads, require_divisible=root.attention != "none")
    for address, array in params.arrays.items():
        if array is root:
            continue
        validate(f"array.{address}", array.n_heads, require_divisible=array.attention != "none")
    for address, leaf in params.active_requests.items():
        if leaf.pooling == "query":
            validate(f"leaf.{address}", leaf.n_heads, require_divisible=True)
