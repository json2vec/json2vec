"""Tensorfield plugin extension API and built-in extension imports."""

from __future__ import annotations

from relflow.tensorfields import base as base
from relflow.tensorfields import extensions as extensions
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
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
    "DecoderBase",
    "EmbedderBase",
    "Plugin",
    "RequestBase",
    "TensorFieldBase",
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
