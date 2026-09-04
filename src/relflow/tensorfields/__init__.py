"""Tensorfield extension API and built-in implementations."""

from __future__ import annotations

from relflow.tensorfields import base as base
from relflow.tensorfields import extensions as extensions
from relflow.tensorfields.base import (
    TENSORFIELDS,
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import (
    array,
    embedding,
    fixed,
    inferred,
    labels,
    offsets,
    shape,
    state,
    struct,
    variable,
)

__all__ = [
    "TENSORFIELDS",
    "Context",
    "DecoderBase",
    "EmbedderBase",
    "Extension",
    "RequestBase",
    "TensorFieldBase",
    "TensorInput",
    "array",
    "base",
    "embedding",
    "extensions",
    "fixed",
    "inferred",
    "labels",
    "offsets",
    "shape",
    "state",
    "struct",
    "variable",
]
