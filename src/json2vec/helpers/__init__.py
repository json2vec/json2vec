"""Helper utilities kept outside the core `json2vec` namespace."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from json2vec.helpers import optimizers as optimizers
from json2vec.helpers.inference import InferenceConfig, infer_schema

if TYPE_CHECKING:
    from json2vec.helpers.tuning import (
        Array,
        Attention,
        Leaf,
        Optimizer,
        OptimizerTuningFactory,
        Pooling,
        Root,
        Toggle,
        tune,
    )

_TUNING_EXPORTS = {
    "Array",
    "Attention",
    "Leaf",
    "Optimizer",
    "OptimizerTuningFactory",
    "Pooling",
    "Root",
    "Toggle",
    "tune",
}
_TUNING_DEPENDENCIES = {"optuna", "pydantic_optuna_bridge"}


def _missing_tuning_extra(name: str, error: ModuleNotFoundError) -> None:
    message = f"json2vec.helpers.{name} requires the tuning extra; install with `pip install json2vec[tuning]`."
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    raise ModuleNotFoundError(message) from error


def __getattr__(name: str) -> Any:
    if name not in _TUNING_EXPORTS:
        raise AttributeError(f"module 'json2vec.helpers' has no attribute {name!r}")

    try:
        tuning = importlib.import_module("json2vec.helpers.tuning")
    except ModuleNotFoundError as error:
        if error.name in _TUNING_DEPENDENCIES:
            _missing_tuning_extra(name, error)
        raise

    value = getattr(tuning, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_TUNING_EXPORTS])


__all__ = [
    "Array",
    "Attention",
    "InferenceConfig",
    "Leaf",
    "Optimizer",
    "OptimizerTuningFactory",
    "Pooling",
    "Root",
    "Toggle",
    "infer_schema",
    "optimizers",
    "tune",
]
