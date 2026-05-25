"""Shared data module type aliases and helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Annotated, Any, TypeAlias, TypeVar

from beartype import beartype
from beartype.vale import Is
from tensordict import TensorDict
from torch.utils.data import get_worker_info

from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.preprocessors.base import PREPROCESSORS, Preprocessor
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


class PreprocessorConfig:
    Value: TypeAlias = str | Callable[..., Any] | Preprocessor | None

    @classmethod
    def normalize(cls, preprocessor: Value) -> Value:
        if preprocessor is None:
            return None

        if isinstance(preprocessor, str):
            if preprocessor not in PREPROCESSORS:
                raise ValueError(f"you haven't registered preprocessor {preprocessor}")
            return preprocessor

        if isinstance(preprocessor, Preprocessor):
            return preprocessor

        if callable(preprocessor):
            name = getattr(preprocessor, "__name__", None)
            if isinstance(name, str) and name in PREPROCESSORS:
                return name
            return preprocessor

        return preprocessor


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


def identity(data: Any) -> Any:
    return data
