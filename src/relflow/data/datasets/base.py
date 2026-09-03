"""Shared tensor types used after Arrow coalescing."""

from typing import Any, TypeAlias

from tensordict import TensorDict

from relflow.structs.tree import Address
from relflow.tensorfields.base import TensorFieldBase

EncodedInput: TypeAlias = TensorDict[Address, TensorFieldBase]
InterprocessEncodingContext: TypeAlias = dict[Address, Any]

__all__ = ["EncodedInput", "InterprocessEncodingContext"]
