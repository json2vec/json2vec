from __future__ import annotations

import hashlib
import inspect
import os
import random
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cache, partial
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, TypeVar, cast
from urllib.parse import urlparse

import jmespath
import lightning.pytorch as lit
import polars as pl
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pydantic
import torch
from beartype import beartype
from beartype.vale import Is
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from json2vec.data.processing import Pipeline
from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.processors.base import PROCESSORS, Processor, ProcessorMode
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix
from json2vec.structs.experiment import Hyperparameters

# import pyarrow.fs as fs
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec


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


class Dataset(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    root: str | None = None
    processor: Annotated[str | Callable[..., Any] | Processor | None, pydantic.Field(default="default")]
    kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    suffix: Suffix | None = None
    patterns: dict[Strata, str] | None = None

    @pydantic.field_validator("processor", mode="before")
    @classmethod
    def normalize_processor(cls, value: Any):
        if value is None or isinstance(value, str):
            return value

        if isinstance(value, Processor):
            return value

        if callable(value):
            name = getattr(value, "__name__", None)
            if isinstance(name, str) and name in PROCESSORS:
                return name
            return value

        return value

    @pydantic.model_validator(mode="after")
    def check_dataset_configuration(self):
        if isinstance(self.processor, str) and self.processor not in PROCESSORS:
            raise ValueError(f"you haven't registered processor {self.processor}")

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

    # Hash the string using SHA-256
    h: bytes = hashlib.sha256(string.encode("utf-8")).digest()

    # Convert hash to integer and truncate to desired number of bits
    return int.from_bytes(h, "big") >> (256 - bits)


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


@beartype
def fetch(
    dataset: Dataset,
    strata: Strata,
    sharding: ShardingStrategy,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[str]:
    if dataset.root is None:
        return

    pattern = None if dataset.patterns is None else dataset.patterns.get(strata)
    regex: re.Pattern[str] | None = None if pattern is None else re.compile(pattern)

    parsed = urlparse(dataset.root)

    if parsed.scheme == "s3":
        fs = pafs.S3FileSystem()  # type: ignore[attr-defined]
        path = f"{parsed.netloc}{parsed.path}"
        uri_prefix = "s3://"

    elif parsed.scheme in ("", "file"):
        fs = pafs.LocalFileSystem()
        path = parsed.path
        uri_prefix = ""

    else:
        raise ValueError(f"Unsupported scheme: {parsed.scheme or 'file'}")

    selector = pafs.FileSelector(path, recursive=True)

    worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)

    for info in fs.get_file_info(selector):
        if info.is_file:
            uri_path: str = f"{uri_prefix}{info.path}" if uri_prefix else info.path
            if regex is None or regex.search(uri_path):
                if sharding == ShardingStrategy.file:
                    if not _is_assigned_to_worker(
                        shard_key=f"file:{uri_path}",
                        worker_id=worker_id,
                        num_workers=num_workers,
                    ):
                        continue

                yield uri_path


@beartype
def observe(
    dataset: Dataset,
    strata: Strata,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    file_buffer_size: int,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[RawObservation]:
    if dataset.root is None:
        # Processor-driven mode: seed a single synthetic observation for one worker.
        worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)
        if _is_assigned_to_worker(
            shard_key="synthetic:seed",
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            yield {}
        return

    paths: Iterator[str] = fetch(
        dataset=dataset,
        strata=strata,
        sharding=sharding,
        global_rank=global_rank,
        world_size=world_size,
    )
    shuffled_paths: Iterator[str] = shuffle(
        paths,
        size=file_buffer_size,
        strata=strata,
    )
    yield from read(
        shuffled_paths,
        dataset=dataset,
        sharding=sharding,
        chunk_batch_size=chunk_batch_size,
        global_rank=global_rank,
        world_size=world_size,
    )


@beartype
def observe_polars(
    dataframe: pl.DataFrame,
    strata: Strata,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[RawObservation]:
    worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)

    if sharding == ShardingStrategy.file:
        if not _is_assigned_to_worker(
            shard_key="dataframe:0",
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            return

    if sharding == ShardingStrategy.record:
        for row_index, row in enumerate(dataframe.iter_rows(named=True)):
            if _is_assigned_to_worker(
                shard_key=f"dataframe:record:{row_index}",
                worker_id=worker_id,
                num_workers=num_workers,
            ):
                yield cast(RawObservation, row)
        return

    for chunk_index, offset in enumerate(range(0, dataframe.height, chunk_batch_size)):
        if sharding == ShardingStrategy.chunk:
            if not _is_assigned_to_worker(
                shard_key=f"dataframe:chunk:{chunk_index}",
                worker_id=worker_id,
                num_workers=num_workers,
            ):
                continue

        yield from cast(list[RawObservation], dataframe.slice(offset, chunk_batch_size).to_dicts())


@beartype
def read(
    pipe: Iterable[str],
    dataset: Dataset,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[RawObservation]:
    worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)

    match dataset.suffix:
        case Suffix.ndjson:
            import json

            for uri_path in pipe:
                record_index: int = 0

                with open(uri_path, "r") as file:
                    for line in file:
                        if not line.strip():
                            continue

                        if sharding == ShardingStrategy.chunk:
                            chunk_index = record_index // chunk_batch_size
                            if not _is_assigned_to_worker(
                                shard_key=f"chunk:{uri_path}:{chunk_index}",
                                worker_id=worker_id,
                                num_workers=num_workers,
                            ):
                                record_index += 1
                                continue

                        elif sharding == ShardingStrategy.record:
                            if not _is_assigned_to_worker(
                                shard_key=f"record:{uri_path}:{record_index}",
                                worker_id=worker_id,
                                num_workers=num_workers,
                            ):
                                record_index += 1
                                continue

                        record_index += 1
                        yield json.loads(line)

        case Suffix.feather | Suffix.parquet | Suffix.avro | Suffix.csv | Suffix.orc | Suffix.json:
            for uri_path in pipe:
                
                parsed = urlparse(uri_path)

                if parsed.scheme == "s3":
                    fs = pafs.S3FileSystem()  # type: ignore[attr-defined]
                    path = f"{parsed.netloc}{parsed.path}"

                elif parsed.scheme in ("", "file"):
                    fs = pafs.LocalFileSystem()
                    path = parsed.path

                else:
                    raise ValueError(f"Unsupported scheme: {parsed.scheme or 'file'}")

                bucket = parsed.netloc
                key = parsed.path.lstrip("/")  # remove leading slash

                # Create pyarrow S3 filesystem

                try:
                    arrow_dataset = ds.dataset(
                        f"{bucket}/{key}",
                        format=dataset.suffix.value,
                        filesystem=fs,
                    )

                    for chunk_index, batch in enumerate(arrow_dataset.to_batches(batch_size=chunk_batch_size)):
                        if sharding == ShardingStrategy.chunk:
                            if not _is_assigned_to_worker(
                                shard_key=f"chunk:{uri_path}:{chunk_index}",
                                worker_id=worker_id,
                                num_workers=num_workers,
                            ):
                                continue

                            rows: list[RawObservation] = batch.to_pylist()
                            yield from rows
                            continue

                        rows: list[RawObservation] = batch.to_pylist()

                        if sharding == ShardingStrategy.record:
                            for row_index, row in enumerate(rows):
                                if _is_assigned_to_worker(
                                    shard_key=f"record:{uri_path}:{chunk_index}:{row_index}",
                                    worker_id=worker_id,
                                    num_workers=num_workers,
                                ):
                                    yield row
                            continue

                        yield from rows
                except Exception:
                    print(f"Error reading {path}, skipping.")
                    continue

        case _:
            raise ValueError(f"Unsupported suffix: {dataset.suffix}")


@beartype
def process(
    pipe: Iterable[RawObservation],
    dataset: Dataset,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
) -> Iterator[ProcessedObservation]:

    if dataset.processor is None:
        for item in pipe:
            yield [item]

    else:
        if isinstance(dataset.processor, str):
            processor: Processor = PROCESSORS[dataset.processor]
        elif isinstance(dataset.processor, Processor):
            processor = dataset.processor
        else:
            processor = Processor(
                name=getattr(dataset.processor, "__name__", type(dataset.processor).__name__),
                func=dataset.processor,
                mode=ProcessorMode.transformation,
            )

        for item in pipe:
            yield from processor.outputs(
                item,
                **dataset.kwargs,
                strata=strata,
                interprocess_encoding_context=interprocess_encoding_context,
            )


@beartype
def batch(pipe: Iterable[T], batch_size: int) -> Iterator[list[T]]:

    batch: list[T] = []

    for item in pipe:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


@beartype
def sample(pipe: Iterable[T], sample_rate: float, strata: Strata) -> Iterator[T]:
    if strata == Strata.predict or sample_rate >= 1.0:
        yield from pipe
        return

    for item in pipe:
        if random.random() < sample_rate:
            yield item



@beartype
def shuffle(pipe: Iterable[T], size: int, strata: Strata) -> Iterator[T]:

    if strata == Strata.predict:
        yield from pipe
        return

    iterable = iter(pipe)
    buffer: list[T] = []
    exhausted: bool = False

    for _ in range(size):
        try:
            buffer.append(next(iterable))
        except StopIteration:
            exhausted = True
            break

    while buffer:
        idx = random.randrange(len(buffer))
        item = buffer[idx]

        if exhausted:
            buffer.pop(idx)
        else:
            try:
                buffer[idx] = next(iterable)
            except StopIteration:
                exhausted = True
                buffer.pop(idx)

        yield item


@beartype
@cache
def query(expression: str) -> jmespath.parser.ParsedResult:
    return jmespath.compile(expression=f"[*]{expression}")


def _contains_observed_value(result: Any) -> bool:
    stack = [result]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif item is None:
            continue
        elif isinstance(item, str) and item == "":
            continue
        else:
            return True

    return False


class JMESPathResolutionMonitor(pydantic.BaseModel):
    every: Annotated[int, pydantic.Field(gt=0)] = 1000

    _counts: Counter[Address] = pydantic.PrivateAttr(default_factory=Counter)

    def observe(self, *, address: Address, expression: str, result: Any) -> None:
        self._counts[address] += 1
        count = self._counts[address]

        if count % self.every != 0:
            return

        if _contains_observed_value(result):
            return

        raise ValueError(f"JMESPath query returned empty result for address '{address}': {expression}")


@cache
def _accepts_interprocess_encoding_context(TensorField: type[TensorFieldBase]) -> bool:
    return "interprocess_encoding_context" in inspect.signature(TensorField.new).parameters


def encode(
    batch: EncodedBatch,
    hyperparameters: Hyperparameters,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    jmespath_resolution_monitor: JMESPathResolutionMonitor | None = None,
) -> EncodedInput:

    out: dict[Address, TensorFieldBase] = {}
    target_addresses = set(hyperparameters.target)

    for address, request in hyperparameters.requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))

        if (strata == Strata.predict) & (address in target_addresses):
            # basically, if we are in inference mode, we should create empty values

            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                hyperparameters=hyperparameters,
            )

            continue

        expression = request.query
        if expression is None:
            raise ValueError(f"request '{address}' must define query")

        result: list = query(expression).search(batch)

        if jmespath_resolution_monitor is not None:
            jmespath_resolution_monitor.observe(address=address, expression=expression, result=result)

        kwargs: dict[str, Any] = dict(
            values=result,
            address=address,
            hyperparameters=hyperparameters,
            strata=strata,
        )
        if _accepts_interprocess_encoding_context(TensorField):
            kwargs["interprocess_encoding_context"] = interprocess_encoding_context.get(address)

        out[address] = TensorField.new(**kwargs)

        # otherwise, we should "target" them entirely during model training
        # but we still need to instantiate them for backpropagation
        if address in target_addresses:
            out[address].target(p_prune=1.0)

    inputs = cast(EncodedInput, TensorDict(source=cast(Any, out)))

    if strata == Strata.predict:
        inputs["metadata"] = batch

    return inputs


@beartype
def transform(
    pipe: Iterable[EncodedBatch],
    hyperparameters: Hyperparameters,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    jmespath_resolution_monitor: JMESPathResolutionMonitor | None = None,
) -> Iterator[EncodedInput]:
    for batch in pipe:

        yield encode(
            batch=batch,
            hyperparameters=hyperparameters,
            strata=strata,
            interprocess_encoding_context=interprocess_encoding_context,
            jmespath_resolution_monitor=jmespath_resolution_monitor,
        )


@beartype
def mask(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    for batch in pipe:
        for address in hyperparameters.requests.keys():
            p_mask = hyperparameters.resolved_p_mask(address)
            if p_mask <= 0.0:
                continue

            field: TensorFieldBase = batch[address]
            field.mask(p_mask=p_mask)

        yield batch


@beartype
def target(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    for batch in pipe:
        for address in hyperparameters.requests.keys():
            p_prune = hyperparameters.resolved_p_prune(address)
            if p_prune <= 0.0:
                continue

            field: TensorFieldBase = batch[address]
            field.target(p_prune=p_prune)

        yield batch


def identity(data: Any) -> Any:
    return data


class BatchDataset(IterableDataset):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataset: Dataset,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        sharding: ShardingStrategy,
        chunk_batch_size: int,
        file_buffer_size: int,
        observation_buffer_size: int,
        sample_rate: float,
        global_rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()

        self.hyperparameters: Hyperparameters = hyperparameters
        self.dataset: Dataset = dataset
        self.interprocess_encoding_context: InterprocessEncodingContext = interprocess_encoding_context
        self.global_rank: int = distributed_rank() if global_rank is None else global_rank
        self.world_size: int = distributed_world_size() if world_size is None else world_size
        self.batch_size: int = batch_size
        self.strata: Strata = strata
        self.sharding: ShardingStrategy = sharding
        self.chunk_batch_size: int = chunk_batch_size
        self.file_buffer_size: int = file_buffer_size
        self.observation_buffer_size: int = observation_buffer_size
        self.sample_rate: float = sample_rate

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        yield from (
            Pipeline(
                hyperparameters=self.hyperparameters,
                dataset=self.dataset,
                strata=self.strata,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=JMESPathResolutionMonitor(),
                sharding=self.sharding,
                chunk_batch_size=self.chunk_batch_size,
                file_buffer_size=self.file_buffer_size,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            | observe
            | process
            | sample
            | partial(shuffle, size=self.observation_buffer_size)
            | batch
            | transform
            | mask
            | target
        )


class PolarsBatchDataset(IterableDataset):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataframe: pl.DataFrame,
        dataset: Dataset,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        sharding: ShardingStrategy,
        chunk_batch_size: int,
        observation_buffer_size: int,
        sample_rate: float,
        global_rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()

        self.hyperparameters: Hyperparameters = hyperparameters
        self.dataframe: pl.DataFrame = dataframe
        self.dataset: Dataset = dataset
        self.interprocess_encoding_context: InterprocessEncodingContext = interprocess_encoding_context
        self.global_rank: int = distributed_rank() if global_rank is None else global_rank
        self.world_size: int = distributed_world_size() if world_size is None else world_size
        self.batch_size: int = batch_size
        self.strata: Strata = strata
        self.sharding: ShardingStrategy = sharding
        self.chunk_batch_size: int = chunk_batch_size
        self.observation_buffer_size: int = observation_buffer_size
        self.sample_rate: float = sample_rate

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        yield from (
            Pipeline(
                hyperparameters=self.hyperparameters,
                dataframe=self.dataframe,
                dataset=self.dataset,
                strata=self.strata,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=JMESPathResolutionMonitor(),
                sharding=self.sharding,
                chunk_batch_size=self.chunk_batch_size,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            | observe_polars
            | process
            | sample
            | partial(shuffle, size=self.observation_buffer_size)
            | batch
            | transform
            | mask
            | target
        )


def dataloader(
    hyperparameters: Hyperparameters,
    dataset: Dataset,
    interprocess_encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    num_workers: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    file_buffer_size: int,
    observation_buffer_size: int,
    sample_rate: float,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    workers: int = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers: bool = persistent_workers and workers > 0
    active_pin_memory: bool = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    global_rank = distributed_rank() if global_rank is None else global_rank
    world_size = distributed_world_size() if world_size is None else world_size

    return DataLoader(
        dataset=BatchDataset(
            hyperparameters=hyperparameters,
            dataset=dataset,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=batch_size,
            strata=strata,
            sharding=sharding,
            chunk_batch_size=chunk_batch_size,
            file_buffer_size=file_buffer_size,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
            global_rank=global_rank,
            world_size=world_size,
        ),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=workers,
        persistent_workers=active_persistent_workers,
        pin_memory=active_pin_memory,
    )


def polars_dataloader(
    hyperparameters: Hyperparameters,
    dataframe: pl.DataFrame,
    dataset: Dataset,
    interprocess_encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    num_workers: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    observation_buffer_size: int,
    sample_rate: float,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    workers: int = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers: bool = persistent_workers and workers > 0
    active_pin_memory: bool = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    global_rank = distributed_rank() if global_rank is None else global_rank
    world_size = distributed_world_size() if world_size is None else world_size

    return DataLoader(
        dataset=PolarsBatchDataset(
            hyperparameters=hyperparameters,
            dataframe=dataframe,
            dataset=dataset,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=batch_size,
            strata=strata,
            sharding=sharding,
            chunk_batch_size=chunk_batch_size,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
            global_rank=global_rank,
            world_size=world_size,
        ),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=workers,
        persistent_workers=active_persistent_workers,
        pin_memory=active_pin_memory,
    )


class StreamingDataModule(lit.LightningDataModule):
    @beartype
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataset: Dataset,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: PositiveInt,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.chunk,
        chunk_batch_size: PositiveInt | StrataMap[PositiveInt] = 4096,
        file_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
    ):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.interprocess_encoding_context = interprocess_encoding_context
        self.batch_size = batch_size
        self.num_workers = _by_strata(num_workers, default=None)
        self.persistent_workers = _by_strata(persistent_workers, default=True)
        self.pin_memory = _by_strata(pin_memory, default=True)
        _sharding = lambda value: value if isinstance(value, ShardingStrategy) else ShardingStrategy(value.strip().lower())  # noqa: E731
        self.sharding = {
            strata: _sharding(strategy)
            for strata, strategy in _by_strata(sharding, default=ShardingStrategy.chunk).items()
        }
        self.chunk_batch_size = _by_strata(chunk_batch_size, default=4096)
        self.file_buffer_size = _by_strata(file_buffer_size, default=1)
        self.observation_buffer_size = _by_strata(observation_buffer_size, default=1)
        self.sample_rate = {
            strata: float(rate)
            for strata, rate in _by_strata(sample_rate, default=1.0).items()
        }

    @classmethod
    def from_model(
        cls,
        model: JSON2Vec,
        dataset: Dataset,
        **kwargs: Any,
    ) -> "StreamingDataModule":
        return cls(
            hyperparameters=model.hyperparameters,
            dataset=dataset,
            interprocess_encoding_context=model.interprocess_encoding_context,
            batch_size=model.batch_size,
            **kwargs,
        )

    def dataloader(self, strata: Strata) -> DataLoader:
        trainer = getattr(self, "trainer", None)
        global_rank = getattr(trainer, "global_rank", None)
        world_size = getattr(trainer, "world_size", None)

        return dataloader(
            hyperparameters=self.hyperparameters,
            dataset=self.dataset,
            interprocess_encoding_context=self.interprocess_encoding_context,
            batch_size=self.batch_size,
            strata=strata,
            num_workers=self.num_workers[strata],
            persistent_workers=self.persistent_workers[strata],
            pin_memory=self.pin_memory[strata],
            sharding=self.sharding[strata],
            chunk_batch_size=self.chunk_batch_size[strata],
            file_buffer_size=self.file_buffer_size[strata],
            observation_buffer_size=self.observation_buffer_size[strata],
            sample_rate=self.sample_rate[strata],
            global_rank=global_rank,
            world_size=world_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.train)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.validate)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.test)

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.predict)


class PolarsDataModule(lit.LightningDataModule):
    @beartype
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataframe: pl.DataFrame | DataFrameMap,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: PositiveInt,
        dataset: Dataset | None = None,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.chunk,
        chunk_batch_size: PositiveInt | StrataMap[PositiveInt] = 4096,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
    ):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.dataframes = _dataframes_by_strata(dataframe)
        self.dataset = Dataset(root=None, processor="default") if dataset is None else dataset
        if self.dataset.root is not None:
            raise ValueError("PolarsDataModule dataset must not define root; pass processor configuration only")

        self.interprocess_encoding_context = interprocess_encoding_context
        self.batch_size = batch_size
        self.num_workers = _by_strata(num_workers, default=None)
        self.persistent_workers = _by_strata(persistent_workers, default=True)
        self.pin_memory = _by_strata(pin_memory, default=True)
        _sharding = lambda value: value if isinstance(value, ShardingStrategy) else ShardingStrategy(value.strip().lower())  # noqa: E731
        self.sharding = {
            strata: _sharding(strategy)
            for strata, strategy in _by_strata(sharding, default=ShardingStrategy.chunk).items()
        }
        self.chunk_batch_size = _by_strata(chunk_batch_size, default=4096)
        self.observation_buffer_size = _by_strata(observation_buffer_size, default=1)
        self.sample_rate = {
            strata: float(rate)
            for strata, rate in _by_strata(sample_rate, default=1.0).items()
        }

    @classmethod
    def from_model(
        cls,
        model: JSON2Vec,
        train: pl.DataFrame | None = None,
        validate: pl.DataFrame | None = None,
        test: pl.DataFrame | None = None,
        predict: pl.DataFrame | None = None,
        dataset: Dataset | None = None,
        processor: str | Callable[..., Any] | Processor | None = "default",
        dataframe: pl.DataFrame | DataFrameMap | None = None,
        **kwargs: Any,
    ) -> "PolarsDataModule":
        if dataframe is not None and any(frame is not None for frame in (train, validate, test, predict)):
            raise ValueError("pass either dataframe or named splits, not both")

        if dataframe is None:
            dataframes = {
                strata: frame
                for strata, frame in {
                    Strata.train: train,
                    Strata.validate: validate,
                    Strata.test: test,
                    Strata.predict: predict,
                }.items()
                if frame is not None
            }
            if not dataframes:
                raise ValueError("at least one dataframe split is required")
        else:
            dataframes = _dataframes_by_strata(dataframe)

        if dataset is None:
            dataset = Dataset(root=None, processor=processor)

        return cls(
            hyperparameters=model.hyperparameters,
            dataframe=dataframes,
            dataset=dataset,
            interprocess_encoding_context=model.interprocess_encoding_context,
            batch_size=model.batch_size,
            **kwargs,
        )

    def dataloader(self, strata: Strata) -> DataLoader:
        trainer = getattr(self, "trainer", None)
        global_rank = getattr(trainer, "global_rank", None)
        world_size = getattr(trainer, "world_size", None)
        if strata not in self.dataframes:
            raise ValueError(f"no dataframe configured for strata: {strata}")

        return polars_dataloader(
            hyperparameters=self.hyperparameters,
            dataframe=self.dataframes[strata],
            dataset=self.dataset,
            interprocess_encoding_context=self.interprocess_encoding_context,
            batch_size=self.batch_size,
            strata=strata,
            num_workers=self.num_workers[strata],
            persistent_workers=self.persistent_workers[strata],
            pin_memory=self.pin_memory[strata],
            sharding=self.sharding[strata],
            chunk_batch_size=self.chunk_batch_size[strata],
            observation_buffer_size=self.observation_buffer_size[strata],
            sample_rate=self.sample_rate[strata],
            global_rank=global_rank,
            world_size=world_size,
        )

    def _maybe_dataloader(self, strata: Strata) -> DataLoader | None:
        if strata not in self.dataframes:
            return None
        return self.dataloader(strata=strata)

    def train_dataloader(self) -> DataLoader | None:
        return self._maybe_dataloader(strata=Strata.train)

    def val_dataloader(self) -> DataLoader | None:
        return self._maybe_dataloader(strata=Strata.validate)

    def test_dataloader(self) -> DataLoader | None:
        return self._maybe_dataloader(strata=Strata.test)

    def predict_dataloader(self) -> DataLoader | None:
        return self._maybe_dataloader(strata=Strata.predict)


def mock(hyperparameters: Hyperparameters, batch_size: int) -> EncodedInput:

    out: dict[Address, TensorFieldBase] = {}

    for address, request in hyperparameters.requests.items():

        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))

        out[address] = TensorField.empty(
            batch_size=batch_size,
            address=address,
            hyperparameters=hyperparameters,
        )

    return cast(EncodedInput, TensorDict(source=cast(Any, out), batch_size=batch_size))
