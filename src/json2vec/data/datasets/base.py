"""Shared dataset configuration models and type aliases."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Mapping
from typing import Annotated, Any, TypeAlias, TypeVar, cast

import polars as pl
import pydantic
from beartype import beartype
from beartype.vale import Is
from tensordict import TensorDict
from torch.utils.data import get_worker_info

from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.preprocessors.base import PREPROCESSORS, Preprocessor
from json2vec.structs.enums import Strata, Suffix
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TensorFieldBase

T = TypeVar("T")
StrataMap: TypeAlias = Mapping[Strata | str, T]
DataFrameMap: TypeAlias = Mapping[Strata | str, pl.DataFrame]
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


class Dataset(pydantic.BaseModel):
    """Input dataset configuration for streaming or in-memory records.

    A dataset may point at a file root for streaming reads, or it may only carry
    an optional preprocessor for in-memory data modules. When `preprocessor` is
    `None`, observations pass through unchanged.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    root: str | None = None
    preprocessor: Annotated[str | Callable[..., Any] | Preprocessor | None, pydantic.Field(default=None)] = None
    kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    suffix: Suffix | None = None
    patterns: dict[Strata, str] | None = None

    @pydantic.field_validator("preprocessor", mode="before")
    @classmethod
    def normalize_preprocessor(cls, value: Any):
        if value is None or isinstance(value, str):
            return value

        if isinstance(value, Preprocessor):
            return value

        if callable(value):
            name = getattr(value, "__name__", None)
            if isinstance(name, str) and name in PREPROCESSORS:
                return name
            return value

        return value

    @pydantic.model_validator(mode="after")
    def check_dataset_configuration(self):
        if isinstance(self.preprocessor, str) and self.preprocessor not in PREPROCESSORS:
            raise ValueError(f"you haven't registered preprocessor {self.preprocessor}")

        if self.root is not None and self.suffix is None:
            raise ValueError("suffix is required when root is specified")

        if self.root is not None and self.patterns is None:
            warnings.warn(
                "dataset patterns are not configured; all strata will read the same files, "
                "so training data may be used for validation",
                UserWarning,
                stacklevel=2,
            )

        return self


@beartype
def _by_strata(value: T | StrataMap[T], *, default: T) -> dict[Strata, T]:
    if isinstance(value, Mapping):
        normalized = {strata: default for strata in Strata}
        mapped = cast(StrataMap[T], value)
        for key, item in mapped.items():
            normalized[Strata(str(key).lower())] = item
        return normalized

    item = cast(T, value)
    return {strata: item for strata in Strata}


def _dataframes_by_strata(dataframe: pl.DataFrame | DataFrameMap) -> dict[Strata, pl.DataFrame]:
    if not isinstance(dataframe, Mapping):
        return {strata: dataframe for strata in Strata}

    normalized: dict[Strata, pl.DataFrame] = {}
    for key, frame in dataframe.items():
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"dataframe for strata '{key}' must be a polars DataFrame")
        normalized[Strata(str(key).lower())] = frame

    if not normalized:
        raise ValueError("dataframe mapping must include at least one strata")

    return normalized


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
