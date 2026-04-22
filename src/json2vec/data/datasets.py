from __future__ import annotations

import hashlib
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from functools import cache, partial
from itertools import batched
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import jmespath
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from json2vec.data.processing import Pipeline
from json2vec.processors.base import PROCESSORS, Processor
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix
from json2vec.structs.environment import DataLoaderEnvironment
from json2vec.structs.experiment import Session, Structure

# import pyarrow.fs as fs
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec


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
def fetch(session: Session, strata: Strata, sharding: ShardingStrategy) -> Iterator[str]:
    if session.dataset.root is None:
        return

    regex: re.Pattern = re.compile(session.dataset.patterns[strata])

    parsed = urlparse(session.dataset.root)

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
    session: Session,
    strata: Strata,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
) -> Iterator[dict]:
    if session.dataset.root is None:
        # Processor-driven mode: seed a single synthetic observation for one worker.
        worker_id, num_workers = _worker_identity()
        if _is_assigned_to_worker(
            shard_key="synthetic:seed",
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            yield {}
        return

    paths: Iterator[str] = fetch(session=session, strata=strata, sharding=sharding)
    shuffled_paths: Iterator[str] = shuffle(
        paths,
        size=session.dataset.file_buffer_size,
        strata=strata,
    )
    yield from read(
        shuffled_paths,
        session=session,
        sharding=sharding,
        chunk_batch_size=chunk_batch_size,
    )


@beartype
def read(
    pipe: Iterable[str],
    session: Session,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
) -> Iterator[dict]:
    worker_id, num_workers = _worker_identity()

    match session.dataset.suffix:
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
                    dataset = ds.dataset(
                        f"{bucket}/{key}",
                        format=session.dataset.suffix.value,
                        filesystem=fs,
                    )

                    for chunk_index, batch in enumerate(dataset.to_batches(batch_size=chunk_batch_size)):
                        if sharding == ShardingStrategy.chunk:
                            if not _is_assigned_to_worker(
                                shard_key=f"chunk:{uri_path}:{chunk_index}",
                                worker_id=worker_id,
                                num_workers=num_workers,
                            ):
                                continue

                            rows: list[dict] = batch.to_pylist()
                            yield from rows
                            continue

                        rows: list[dict] = batch.to_pylist()

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
            raise ValueError(f"Unsupported suffix: {session.dataset.suffix}")


@beartype
def process(
    pipe: Iterable[dict],
    session: Session,
    strata: Strata,
    state: dict[Address, Any],
) -> Iterator[Any]:

    if session.dataset.processor is None:
        for item in pipe:
            yield [item]

    else:
        processor: Processor = PROCESSORS[session.dataset.processor]

        for item in pipe:
            yield from processor.outputs(item, **session.dataset.kwargs, strata=strata, state=state)


@beartype
def batch(pipe: Iterable[Any], session: Session) -> Iterator[list[Any]]:

    batch: list[Any] = []

    for item in pipe:
        batch.append(item)
        if len(batch) == session.structure.batch_size:
            yield batch
            batch = []

    if batch:
        yield batch



@beartype
def shuffle(pipe: Iterable[dict], size: int, strata: Strata) -> Iterator[dict]:

    if strata == Strata.predict:
        yield from pipe
        return

    iterable = iter(pipe)
    buffer: list[Any] = []
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
    batch: dict[str, Any],
    session: Session,
    strata: Strata,
    state: dict[Address, Any],
) -> TensorDict[Address, TensorFieldBase]:

    out: dict[Address, TensorFieldBase] = {}

    for address, request in session.structure.requests.items():
        TensorField: type[TensorFieldBase] = TENSORFIELDS[request.type].TensorField

        if (strata == Strata.predict) & (address in session.pruned):
            # basically, if we are in inference mode, we should create empty values

            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                structure=session.structure,
            )

            continue

        result: list = query(request.query).search(batch)

        spotcheck(result=result, address=address)

        out[address] = TensorField.new(
            values=result,
            address=address,
            session=session,
            strata=strata,
            state=state.get(address),
        )

        # otherwise, we should "prune" them entirely during model training
        # but we still need to instantiate them for backpropagation
        if address in session.pruned:
            out[address].prune(p_prune=1.0)

    inputs: TensorDict[Address, TensorFieldBase] = TensorDict(source=out)

    if strata == Strata.predict:
        inputs["metadata"] = batch

    
    return inputs


@beartype
def transform(
    pipe: Iterable[dict[str, Any]],
    session: Session,
    strata: Strata,
    state: dict[Address, Any],
) -> Iterator[TensorDict[Address, TensorFieldBase]]:
    for batch in pipe:

        yield encode(batch=batch, session=session, strata=strata, state=state)


@beartype
def mask(
    pipe: Iterable[TensorDict[Address, TensorFieldBase]],
    session: Session,
) -> Iterator[TensorDict[Address, TensorFieldBase]]:
    if not session.p_mask > 0.0:
        yield from pipe

    else:
        for batch in pipe:
            for address in session.structure.requests.keys():
                field: TensorFieldBase = batch[address]
                field.mask(p_mask=session.p_mask)

            yield batch


@beartype
def prune(
    pipe: Iterable[TensorDict[Address, TensorFieldBase]],
    session: Session,
) -> Iterator[TensorDict[Address, TensorFieldBase]]:
    for batch in pipe:
        for address in session.structure.requests.keys():
            field: TensorFieldBase = batch[address]
            field.prune(p_prune=session.p_prune)

        yield batch


def identity(data: Any) -> Any:
    return data


class BatchDataset(IterableDataset):
    def __init__(self, model: JSON2Vec, strata: Strata, environment: DataLoaderEnvironment):
        super().__init__()

        self.session: Session = model.session
        self.environment: DataLoaderEnvironment = environment

        self.state: dict[Address, Any] = model.state

        self.strata: Strata = strata
        logger.bind(
            component="data",
            session=self.session.name,
            strata=self.strata,
            request_fields=len(self.session.structure.requests),
            stateful_fields=len(self.state),
            sharding=self.environment.sharding,
            chunk_batch_size=self.environment.chunk_batch_size,
        ).info("initialized batch dataset")

    def __iter__(self):
        yield from (
            Pipeline(
                session=self.session,
                strata=self.strata,
                state=self.state,
                sharding=self.environment.sharding,
                chunk_batch_size=self.environment.chunk_batch_size,
            )
            | observe
            | process
            | partial(shuffle, size=self.session.dataset.observation_buffer_size)
            | batch
            | transform
            | mask
            | prune
        )


def dataloader(module: JSON2Vec, strata: Strata) -> DataLoader:
    environment: DataLoaderEnvironment = DataLoaderEnvironment.from_env()
    workers: int = environment.num_workers if environment.num_workers is not None else (os.cpu_count() or 0)
    persistent_workers: bool = environment.persistent_workers and workers > 0
    pin_memory: bool = environment.pin_memory and strata != Strata.predict and torch.cuda.is_available()
    logger.bind(
        component="data",
        session=module.session.name,
        strata=strata,
        batch_size=module.session.structure.batch_size,
        workers=workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sharding=environment.sharding,
        chunk_batch_size=environment.chunk_batch_size,
    ).info("building dataloader")

    return DataLoader(
        dataset=BatchDataset(model=module, strata=strata, environment=environment),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )


def mock(structure: Structure) -> TensorDict[Address, TensorFieldBase]:

    out: dict[Address, TensorFieldBase] = {}

    for address, request in structure.requests.items():

        TensorField: TensorFieldBase = TENSORFIELDS[request.type].TensorField

        out[address] = TensorField.empty(
            batch_size=structure.batch_size, 
            address=address, 
            structure=structure
        )

    return TensorDict(source=out, batch_size=structure.batch_size)
