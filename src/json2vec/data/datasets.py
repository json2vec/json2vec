from __future__ import annotations

import hashlib
import os
import random
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from functools import cache, partial
from typing import TYPE_CHECKING, Annotated, Any, Callable, TypeAlias, TypeVar, cast
from urllib.parse import urlparse

import jmespath
import lightning.pytorch as lit
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pydantic
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from json2vec.data.processing import Pipeline
from json2vec.processors.base import PROCESSORS, Processor
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix
from json2vec.structs.experiment import Hyperparameters

# import pyarrow.fs as fs
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec


T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
StrataMap: TypeAlias = Mapping[Strata | str, T]
RawObservation: TypeAlias = dict[str, Any]
ProcessedObservation: TypeAlias = list[RawObservation]
EncodedBatch: TypeAlias = list[ProcessedObservation]
EncodedInput: TypeAlias = TensorDict[Address, TensorFieldBase]


class Dataset(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    root: str | None = None
    processor: Annotated[str | None, pydantic.Field(default="default")]
    kwargs: dict[str, Any] = pydantic.Field(default_factory=dict)
    suffix: Suffix | None = None
    patterns: dict[Strata, str] | None = None

    @pydantic.field_validator("processor", mode="before")
    @classmethod
    def normalize_processor(cls, value: Any):
        if value is None or isinstance(value, str):
            return value

        if callable(value):
            return value.__name__

        return value

    @pydantic.model_validator(mode="after")
    def check_dataset_configuration(self):
        if self.processor is not None and self.processor not in PROCESSORS:
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


def _strata_key(value: Strata | str) -> Strata:
    if isinstance(value, Strata):
        return value

    return Strata(value.strip().lower())


def _by_strata(
    value: InputT | StrataMap[InputT],
    *,
    default: OutputT,
    coerce: Callable[[InputT], OutputT],
) -> dict[Strata, OutputT]:
    if isinstance(value, Mapping):
        normalized = {strata: default for strata in Strata}
        mapped = cast(StrataMap[InputT], value)
        for key, item in mapped.items():
            normalized[_strata_key(key)] = coerce(item)
        return normalized

    item = cast(InputT, value)
    return {strata: coerce(item) for strata in Strata}


def _coerce_num_workers(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("num_workers must be an integer or None")
    if value < 0:
        raise ValueError("num_workers must be >= 0")
    return value


def _coerce_bool(name: str) -> Callable[[bool], bool]:
    def coerce(value: bool) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value

    return coerce


def _coerce_sharding(value: ShardingStrategy | str) -> ShardingStrategy:
    if isinstance(value, ShardingStrategy):
        return value
    if isinstance(value, str):
        return ShardingStrategy(value.strip().lower())
    raise TypeError("sharding must be a ShardingStrategy or string")


def _coerce_chunk_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("chunk_batch_size must be an integer")
    if value < 1:
        raise ValueError("chunk_batch_size must be >= 1")
    return value


def _coerce_buffer_size(name: str) -> Callable[[int], int]:
    def coerce(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
        return value

    return coerce


def _coerce_sample_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("sample_rate must be a number")
    if value <= 0.0 or value > 1.0:
        raise ValueError("sample_rate must be > 0 and <= 1")
    return float(value)


@beartype
def sha256(string: str, bits: int = 64) -> int:
    if not (1 <= bits <= 256):
        raise ValueError("bits must be between 1 and 256")

    # Hash the string using SHA-256
    h: bytes = hashlib.sha256(string.encode("utf-8")).digest()

    # Convert hash to integer and truncate to desired number of bits
    return int.from_bytes(h, "big") >> (256 - bits)


def _worker_identity() -> tuple[int, int]:
    worker_info = get_worker_info()
    if worker_info is None:
        return 0, 1

    return worker_info.id, worker_info.num_workers


def _is_assigned_to_worker(shard_key: str, worker_id: int, num_workers: int) -> bool:
    if num_workers <= 1:
        return True

    owner: int = sha256(shard_key) % num_workers
    return owner == worker_id


@beartype
def fetch(dataset: Dataset, strata: Strata, sharding: ShardingStrategy) -> Iterator[str]:
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

    worker_id, num_workers = _worker_identity()

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
) -> Iterator[RawObservation]:
    if dataset.root is None:
        # Processor-driven mode: seed a single synthetic observation for one worker.
        worker_id, num_workers = _worker_identity()
        if _is_assigned_to_worker(
            shard_key="synthetic:seed",
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            yield {}
        return

    paths: Iterator[str] = fetch(dataset=dataset, strata=strata, sharding=sharding)
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
    )


@beartype
def read(
    pipe: Iterable[str],
    dataset: Dataset,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
) -> Iterator[RawObservation]:
    worker_id, num_workers = _worker_identity()

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
    state: dict[Address, Any],
) -> Iterator[ProcessedObservation]:

    if dataset.processor is None:
        for item in pipe:
            yield [item]

    else:
        processor: Processor = PROCESSORS[dataset.processor]

        for item in pipe:
            yield from processor.outputs(item, **dataset.kwargs, strata=strata, state=state)


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


_jmespath_counter = Counter()


def spotcheck(result, address: Address, every: int = 1000):
    _jmespath_counter[address] += 1
    count = _jmespath_counter[address]

    if count % every != 0:
        return  # skip check

    # Fast non-recursive emptiness check
    stack = [result]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif item not in (None, "", [], {}):
            return

    raise ValueError(f"JMESPath query returned empty result for address: {address}")


def encode(
    batch: EncodedBatch,
    hyperparameters: Hyperparameters,
    strata: Strata,
    state: dict[Address, Any],
) -> EncodedInput:

    out: dict[Address, TensorFieldBase] = {}

    for address, request in hyperparameters.requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))

        if (strata == Strata.predict) & (address in hyperparameters.target):
            # basically, if we are in inference mode, we should create empty values

            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                hyperparameters=hyperparameters,
            )

            continue

        result: list = query(request.query).search(batch)

        spotcheck(result=result, address=address)

        out[address] = TensorField.new(
            values=result,
            address=address,
            hyperparameters=hyperparameters,
            strata=strata,
            state=state.get(address),
        )

        # otherwise, we should "target" them entirely during model training
        # but we still need to instantiate them for backpropagation
        if address in hyperparameters.target:
            out[address].target(p_target=1.0)

    inputs = cast(EncodedInput, TensorDict(source=cast(Any, out)))

    if strata == Strata.predict:
        inputs["metadata"] = batch

    return inputs


@beartype
def transform(
    pipe: Iterable[EncodedBatch],
    hyperparameters: Hyperparameters,
    strata: Strata,
    state: dict[Address, Any],
) -> Iterator[EncodedInput]:
    for batch in pipe:

        yield encode(batch=batch, hyperparameters=hyperparameters, strata=strata, state=state)


@beartype
def mask(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    if not hyperparameters.p_mask > 0.0:
        yield from pipe

    else:
        for batch in pipe:
            for address in hyperparameters.requests.keys():
                field: TensorFieldBase = batch[address]
                field.mask(p_mask=hyperparameters.p_mask)

            yield batch


@beartype
def target(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    for batch in pipe:
        for address in hyperparameters.requests.keys():
            field: TensorFieldBase = batch[address]
            field.target(p_target=hyperparameters.p_target)

        yield batch


def identity(data: Any) -> Any:
    return data


class BatchDataset(IterableDataset):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataset: Dataset,
        state: dict[Address, Any],
        batch_size: int,
        strata: Strata,
        sharding: ShardingStrategy,
        chunk_batch_size: int,
        file_buffer_size: int,
        observation_buffer_size: int,
        sample_rate: float,
    ):
        super().__init__()

        self.hyperparameters: Hyperparameters = hyperparameters
        self.dataset: Dataset = dataset
        self.state: dict[Address, Any] = state
        self.batch_size: int = batch_size
        self.strata: Strata = strata
        self.sharding: ShardingStrategy = sharding
        self.chunk_batch_size: int = chunk_batch_size
        self.file_buffer_size: int = file_buffer_size
        self.observation_buffer_size: int = observation_buffer_size
        self.sample_rate: float = sample_rate
        logger.bind(
            component="data",
            strata=self.strata,
            batch_size=self.batch_size,
            request_fields=len(self.hyperparameters.requests),
            stateful_fields=len(self.state),
            sharding=self.sharding,
            chunk_batch_size=self.chunk_batch_size,
            file_buffer_size=self.file_buffer_size,
            observation_buffer_size=self.observation_buffer_size,
            sample_rate=self.sample_rate,
        ).info("initialized batch dataset")

    def __iter__(self):
        yield from (
            Pipeline(
                hyperparameters=self.hyperparameters,
                dataset=self.dataset,
                strata=self.strata,
                state=self.state,
                sharding=self.sharding,
                chunk_batch_size=self.chunk_batch_size,
                file_buffer_size=self.file_buffer_size,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
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


def dataloader(
    hyperparameters: Hyperparameters,
    dataset: Dataset,
    state: dict[Address, Any],
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
) -> DataLoader:
    workers: int = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers: bool = persistent_workers and workers > 0
    active_pin_memory: bool = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    logger.bind(
        component="data",
        strata=strata,
        batch_size=batch_size,
        workers=workers,
        persistent_workers=active_persistent_workers,
        pin_memory=active_pin_memory,
        sharding=sharding,
        chunk_batch_size=chunk_batch_size,
        file_buffer_size=file_buffer_size,
        observation_buffer_size=observation_buffer_size,
        sample_rate=sample_rate,
    ).info("building dataloader")

    return DataLoader(
        dataset=BatchDataset(
            hyperparameters=hyperparameters,
            dataset=dataset,
            state=state,
            batch_size=batch_size,
            strata=strata,
            sharding=sharding,
            chunk_batch_size=chunk_batch_size,
            file_buffer_size=file_buffer_size,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
        ),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=workers,
        persistent_workers=active_persistent_workers,
        pin_memory=active_pin_memory,
    )


class StreamingDataModule(lit.LightningDataModule):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataset: Dataset,
        state: dict[Address, Any],
        batch_size: int,
        num_workers: int | None | StrataMap[int | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.chunk,
        chunk_batch_size: int | StrataMap[int] = 4096,
        file_buffer_size: int | StrataMap[int] = 1,
        observation_buffer_size: int | StrataMap[int] = 1,
        sample_rate: float | StrataMap[float] = 1.0,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.state = state
        self.batch_size = batch_size
        self.num_workers = _by_strata(num_workers, default=None, coerce=_coerce_num_workers)
        self.persistent_workers = _by_strata(
            persistent_workers,
            default=True,
            coerce=_coerce_bool("persistent_workers"),
        )
        self.pin_memory = _by_strata(pin_memory, default=True, coerce=_coerce_bool("pin_memory"))
        self.sharding = _by_strata(sharding, default=ShardingStrategy.chunk, coerce=_coerce_sharding)
        self.chunk_batch_size = _by_strata(
            chunk_batch_size,
            default=4096,
            coerce=_coerce_chunk_batch_size,
        )
        self.file_buffer_size = _by_strata(
            file_buffer_size,
            default=1,
            coerce=_coerce_buffer_size("file_buffer_size"),
        )
        self.observation_buffer_size = _by_strata(
            observation_buffer_size,
            default=1,
            coerce=_coerce_buffer_size("observation_buffer_size"),
        )
        self.sample_rate = _by_strata(sample_rate, default=1.0, coerce=_coerce_sample_rate)

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
            state=model.state,
            batch_size=model.batch_size,
            **kwargs,
        )

    def dataloader(self, strata: Strata) -> DataLoader:
        return dataloader(
            hyperparameters=self.hyperparameters,
            dataset=self.dataset,
            state=self.state,
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
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.train)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.validate)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.test)

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.predict)


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
