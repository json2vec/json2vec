"""Shared data module type aliases and helpers."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from functools import partial
from typing import Annotated, Any, TypeAlias, TypeVar

from beartype import beartype
from beartype.vale import Is
from tensordict import TensorDict
from torch.utils.data import get_worker_info

from json2vec.data.processors import Preprocessor
from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.structs.enums import Strata
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TensorFieldBase

T = TypeVar("T")
StrataMap: TypeAlias = Mapping[Strata | str, T]
NonNegativeInt: TypeAlias = Annotated[int, Is[lambda value: not isinstance(value, bool) and value >= 0]]
PositiveInt: TypeAlias = Annotated[int, Is[lambda value: not isinstance(value, bool) and value >= 1]]
SampleRate: TypeAlias = Annotated[int | float, Is[lambda value: not isinstance(value, bool) and 0.0 < value <= 1.0]]
RawObservation: TypeAlias = dict[str, Any]
ProcessedObservation: TypeAlias = list[RawObservation]
EncodedBatch: TypeAlias = list[ProcessedObservation]
EncodedInput: TypeAlias = TensorDict[Address, TensorFieldBase]
InterprocessEncodingContext: TypeAlias = dict[Address, Any]

# Encoded batches are `list[list[dict]]`: outer batch, then records emitted for
# one processed observation. Request queries are written relative to the inner
# list; the encoder prepends the outer batch selector before JMESPath search.


@beartype
class Pipeline:
    def __init__(self, **arguments: Any):
        self.arguments: dict[str, Any] = arguments
        self.steps: list[Callable[..., Any]] = []

    def __or__(self, function: Callable[..., Any]) -> "Pipeline":
        required = [name for name in inspect.signature(function).parameters.keys()]
        available = set(required) & set(self.arguments.keys())
        self.steps.append(partial(function, **{arg: self.arguments[arg] for arg in available}))
        return self

    def __repr__(self) -> str:
        return f"Pipeline(steps={len(self.steps)}, arguments={self.arguments!r})"

    def __iter__(self):
        stream = self.steps[0]()

        for step in self.steps[1:]:
            stream = step(stream)

        return iter(stream)


class PreprocessorConfig:
    Value: TypeAlias = Preprocessor | None

    @classmethod
    def normalize(cls, preprocessor: Value) -> Value:
        if preprocessor is None:
            return None

        if isinstance(preprocessor, Preprocessor):
            return preprocessor

        raise TypeError(f"preprocessor must be a Preprocessor object or None, got {type(preprocessor).__name__}")


@beartype
def sha256(string: str, bits: int = 64) -> int:
    if not (1 <= bits <= 256):
        raise ValueError("bits must be between 1 and 256")

    digest = hashlib.sha256(string.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") >> (256 - bits)


def _worker_identity(global_rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    if global_rank is None:
        global_rank = distributed_rank()
    if world_size is None:
        world_size = distributed_world_size()

    worker_info = get_worker_info()
    if worker_info is None:
        return global_rank, max(1, world_size)

    worker_count = max(1, worker_info.num_workers)
    return (global_rank * worker_count) + worker_info.id, max(1, world_size) * worker_count


def _is_assigned_to_worker(shard_key: str, worker_id: int, num_workers: int) -> bool:
    if num_workers <= 1:
        return True

    owner: int = sha256(shard_key) % num_workers
    return owner == worker_id


def share_interprocess_encoding_context(context: InterprocessEncodingContext) -> None:
    """Opt encoding context resources into multiprocessing-safe storage."""
    for field_context in context.values():
        share = getattr(field_context, "share", None)
        if callable(share):
            share()


def identity(data: Any) -> Any:
    return data
