"""Inference helpers for batch prediction and LitServe deployments."""

from typing import TYPE_CHECKING, Any

from json2vec.inference.callback import Postprocessor, Writer

if TYPE_CHECKING:
    from json2vec.inference.deployment import (
        API,
        Accelerator,
        BatchItem,
        Deployment,
        ErrorItem,
        Input,
        ModelSource,
        UpdateOperation,
    )

_DEPLOYMENT_EXPORTS = {
    "API",
    "Accelerator",
    "BatchItem",
    "Deployment",
    "ErrorItem",
    "Input",
    "ModelSource",
    "UpdateOperation",
}


def __getattr__(name: str) -> Any:
    if name not in _DEPLOYMENT_EXPORTS:
        raise AttributeError(f"module 'json2vec.inference' has no attribute {name!r}")

    try:
        from json2vec.inference import deployment
    except ModuleNotFoundError as error:
        if error.name in {"litserve", "pydantic_settings"}:
            raise ModuleNotFoundError(
                f"json2vec.inference.{name} requires the serving extra; install with `pip install json2vec[serving]`."
            ) from error
        raise

    value = getattr(deployment, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_DEPLOYMENT_EXPORTS])


__all__ = [
    "API",
    "Accelerator",
    "BatchItem",
    "Deployment",
    "ErrorItem",
    "Input",
    "ModelSource",
    "Postprocessor",
    "UpdateOperation",
    "Writer",
]
