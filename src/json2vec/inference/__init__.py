"""Inference helpers for batch prediction and realtime deployments."""

from typing import TYPE_CHECKING, Any

from json2vec.inference.callback import Postprocessor, Writer

if TYPE_CHECKING:
    from json2vec.inference.deployment import (
        Accelerator,
        Deployment,
        Input,
        JSONBackend,
        ModelSource,
        UpdateOperation,
    )

_DEPLOYMENT_EXPORTS = {
    "Accelerator",
    "Deployment",
    "Input",
    "JSONBackend",
    "ModelSource",
    "UpdateOperation",
}


def __getattr__(name: str) -> Any:
    if name not in _DEPLOYMENT_EXPORTS:
        raise AttributeError(f"module 'json2vec.inference' has no attribute {name!r}")

    try:
        from json2vec.inference import deployment
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "orjson", "pydantic_settings", "uvicorn"}:
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
    "Accelerator",
    "Deployment",
    "Input",
    "JSONBackend",
    "ModelSource",
    "Postprocessor",
    "UpdateOperation",
    "Writer",
]
