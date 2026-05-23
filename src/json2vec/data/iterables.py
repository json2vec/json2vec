"""Composable iterable stages for fetching, preprocessing, and encoding data."""

from __future__ import annotations

import inspect
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import cache
from typing import Annotated, Any, TypeVar, cast
from urllib.parse import urlparse

import jmespath
import polars as pl
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pydantic
from beartype import beartype
from tensordict import TensorDict

from json2vec.data.datasets.base import (
    Dataset,
    EncodedBatch,
    EncodedInput,
    InterprocessEncodingContext,
    ProcessedObservation,
    RawObservation,
    _is_assigned_to_worker,
    _worker_identity,
)
from json2vec.preprocessors.base import PREPROCESSORS, Preprocessor, PreprocessorMode
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS, TensorFieldBase

T = TypeVar("T")


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
            uri_path = f"{uri_prefix}{info.path}" if uri_prefix else info.path
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
        worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)
        if _is_assigned_to_worker(
            shard_key="synthetic:seed",
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            yield {}
        return

    paths = fetch(
        dataset=dataset,
        strata=strata,
        sharding=sharding,
        global_rank=global_rank,
        world_size=world_size,
    )
    shuffled_paths = shuffle(paths, size=file_buffer_size, strata=strata)
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
                record_index = 0

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
                key = parsed.path.lstrip("/")

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

                            yield from cast(list[RawObservation], batch.to_pylist())
                            continue

                        rows = cast(list[RawObservation], batch.to_pylist())

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
    if dataset.preprocessor is None:
        for item in pipe:
            yield [item]
        return

    if isinstance(dataset.preprocessor, str):
        preprocessor = PREPROCESSORS[dataset.preprocessor]
    elif isinstance(dataset.preprocessor, Preprocessor):
        preprocessor = dataset.preprocessor
    else:
        preprocessor = Preprocessor(
            name=getattr(dataset.preprocessor, "__name__", type(dataset.preprocessor).__name__),
            func=dataset.preprocessor,
            mode=PreprocessorMode.transformation,
        )

    for item in pipe:
        yield from preprocessor.outputs(
            item,
            **dataset.kwargs,
            strata=strata,
            interprocess_encoding_context=interprocess_encoding_context,
        )


@beartype
def batch(pipe: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    items: list[T] = []

    for item in pipe:
        items.append(item)
        if len(items) == batch_size:
            yield items
            items = []

    if items:
        yield items


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
    exhausted = False

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
    """Compile a request-level JMESPath query for an encoded batch.

    Request queries are written relative to one processed observation, not the
    whole batch. This helper prepends the outer batch selector, so a request
    query like `[*].amount` is searched as `[*][*].amount` at encode time.
    Do not include both leading selectors in request definitions.
    """
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

    for address, request in hyperparameters.active_requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))

        if (strata == Strata.predict) & (address in target_addresses):
            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                hyperparameters=hyperparameters,
            )
            continue

        expression = request.query
        if expression is None:
            raise ValueError(f"request '{address}' must define query")

        # `request.query` is relative to a processed observation. `query(...)`
        # adds the outer batch selector before JMESPath searches `batch`.
        result = query(expression).search(batch)
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
    for item in pipe:
        yield encode(
            batch=item,
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
    for item in pipe:
        for address, request in hyperparameters.active_requests.items():
            p_mask = float(request.p_mask or 0.0)
            if p_mask <= 0.0:
                continue

            item[address].mask(p_mask=p_mask)

        yield item


@beartype
def target(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    for item in pipe:
        for address, request in hyperparameters.active_requests.items():
            p_prune = float(request.p_prune or 0.0)
            if p_prune <= 0.0:
                continue

            item[address].target(p_prune=p_prune)

        yield item


def mock(hyperparameters: Hyperparameters, batch_size: int) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}

    for address, request in hyperparameters.active_requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))
        out[address] = TensorField.empty(
            batch_size=batch_size,
            address=address,
            hyperparameters=hyperparameters,
        )

    return cast(EncodedInput, TensorDict(source=cast(Any, out), batch_size=batch_size))
