"""File-backed streaming datasets and Lightning data modules."""

from __future__ import annotations

import os
import random
import re
import weakref
from collections.abc import Callable, Iterable, Iterator
from functools import partial, partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import lightning.pytorch as lit
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import torch
from beartype import beartype
from torch.utils.data import DataLoader, IterableDataset

from json2vec.data.datasets.base import (
    InterprocessEncodingContext,
    NonNegativeInt,
    PositiveInt,
    PreprocessorConfig,
    RawObservation,
    SampleRate,
    StrataMap,
    _is_assigned_to_worker,
    _worker_identity,
    identity,
)
from json2vec.data.iterables import (
    JMESPathResolutionMonitor,
    batch,
    mask,
    process,
    sample,
    shuffle,
    target,
    transform,
)
from json2vec.data.processing import Pipeline
from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.preprocessors.base import Preprocessor
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix
from json2vec.structs.experiment import Hyperparameters

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
else:
    Model = "json2vec.architecture.root.Model"


@beartype
def fetch(
    root: str | Path,
    pattern: re.Pattern[str],
    sharding: ShardingStrategy,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[str]:
    parsed = urlparse(str(root))

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
            if pattern.search(uri_path):
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
    root: str | Path,
    suffix: Suffix,
    pattern: re.Pattern[str],
    strata: Strata,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    file_buffer_size: int,
    replacement: bool = False,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[RawObservation]:
    fetch_sharding = ShardingStrategy.chunk if replacement else sharding
    paths = fetch(
        root=root,
        pattern=pattern,
        sharding=fetch_sharding,
        global_rank=global_rank,
        world_size=world_size,
    )
    if replacement:
        sampled_paths = list(paths)
        if not sampled_paths:
            raise ValueError(
                "no matching files available for replacement sampling; check the streaming root and split pattern"
            )

        def choices() -> Iterator[str]:
            while True:
                yield random.choice(sampled_paths)

        paths = choices()

    shuffled_paths = shuffle(paths, size=file_buffer_size, strata=strata)
    yield from read(
        shuffled_paths,
        suffix=suffix,
        sharding=sharding,
        chunk_batch_size=chunk_batch_size,
        global_rank=global_rank,
        world_size=world_size,
    )


@beartype
def read(
    pipe: Iterable[str],
    suffix: Suffix,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[RawObservation]:
    worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)

    match suffix:
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
                        format=suffix.value,
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
            raise ValueError(f"Unsupported suffix: {suffix}")


class BatchDataset(IterableDataset):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        root: str | Path,
        suffix: Suffix,
        pattern: re.Pattern[str],
        preprocessor: PreprocessorConfig.Value,
        preprocessor_kwargs: dict[str, Any],
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        sharding: ShardingStrategy,
        chunk_batch_size: int,
        file_buffer_size: int,
        observation_buffer_size: int,
        sample_rate: float,
        replacement: bool = False,
        global_rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.root = root
        self.suffix = suffix
        self.pattern = pattern
        self.preprocessor = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs
        self.interprocess_encoding_context = interprocess_encoding_context
        self.global_rank = distributed_rank() if global_rank is None else global_rank
        self.world_size = distributed_world_size() if world_size is None else world_size
        self.batch_size = batch_size
        self.strata = strata
        self.sharding = sharding
        self.chunk_batch_size = chunk_batch_size
        self.file_buffer_size = file_buffer_size
        self.observation_buffer_size = observation_buffer_size
        self.sample_rate = sample_rate
        self.replacement = replacement

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        yield from (
            Pipeline(
                hyperparameters=self.hyperparameters,
                root=self.root,
                suffix=self.suffix,
                pattern=self.pattern,
                preprocessor=self.preprocessor,
                preprocessor_kwargs=self.preprocessor_kwargs,
                strata=self.strata,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=JMESPathResolutionMonitor(),
                sharding=self.sharding,
                chunk_batch_size=self.chunk_batch_size,
                file_buffer_size=self.file_buffer_size,
                sample_rate=self.sample_rate,
                replacement=self.replacement,
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


def dataloader(
    hyperparameters: Hyperparameters,
    root: str | Path,
    suffix: Suffix,
    pattern: re.Pattern[str],
    preprocessor: PreprocessorConfig.Value,
    preprocessor_kwargs: dict[str, Any],
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
    replacement: bool = False,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    workers = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers = persistent_workers and workers > 0
    active_pin_memory = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    global_rank = distributed_rank() if global_rank is None else global_rank
    world_size = distributed_world_size() if world_size is None else world_size

    return DataLoader(
        dataset=BatchDataset(
            hyperparameters=hyperparameters,
            root=root,
            suffix=suffix,
            pattern=pattern,
            preprocessor=preprocessor,
            preprocessor_kwargs=preprocessor_kwargs,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=batch_size,
            strata=strata,
            sharding=sharding,
            chunk_batch_size=chunk_batch_size,
            file_buffer_size=file_buffer_size,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
            replacement=replacement,
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
    """Lightning data module for streaming records from files.

    Reads file-backed records, applies an optional preprocessor, batches
    observations, and encodes them with model hyperparameters.
    """

    @beartype
    def __init__(
        self,
        model: Model,
        root: str | Path,
        suffix: Suffix | str,
        train: re.Pattern[str] | None = None,
        validate: re.Pattern[str] | None = None,
        test: re.Pattern[str] | None = None,
        predict: re.Pattern[str] | None = None,
        preprocessor: str | Callable[..., Any] | Preprocessor | None = None,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.file,
        chunk_batch_size: PositiveInt | StrataMap[PositiveInt] = 4096,
        file_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
        replacement: bool | StrataMap[bool] | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.hyperparameters = model.hyperparameters
        self.root = root
        self.suffix = Suffix(suffix)
        self.train = train
        self.validate = validate
        self.test = test
        self.predict = predict
        self.preprocessor = PreprocessorConfig.normalize(preprocessor)
        self.preprocessor_kwargs = dict(kwargs)
        try:
            self._model_ref = weakref.ref(model)
        except TypeError:
            self._model_ref = None
        self._interprocess_encoding_context = model.interprocess_encoding_context
        self.batch_size = model.batch_size
        self.num_workers = Strata.expand(num_workers, default=None)
        self.persistent_workers = Strata.expand(persistent_workers, default=True)
        self.pin_memory = Strata.expand(pin_memory, default=True)
        self.sharding = ShardingStrategy.expand(sharding, default=ShardingStrategy.file)
        self.chunk_batch_size = Strata.expand(chunk_batch_size, default=4096)
        self.file_buffer_size = Strata.expand(file_buffer_size, default=1)
        self.observation_buffer_size = Strata.expand(observation_buffer_size, default=1)
        self.sample_rate = {strata: float(rate) for strata, rate in Strata.expand(sample_rate, default=1.0).items()}
        self.replacement = (
            {strata: strata == Strata.train for strata in Strata}
            if replacement is None
            else Strata.expand(replacement, default=False)
        )

    @property
    def interprocess_encoding_context(self) -> InterprocessEncodingContext:
        if self._model_ref is not None:
            model = self._model_ref()
            if model is not None:
                return model.interprocess_encoding_context

        return self._interprocess_encoding_context

    @interprocess_encoding_context.setter
    def interprocess_encoding_context(self, context: InterprocessEncodingContext) -> None:
        self._model_ref = None
        self._interprocess_encoding_context = context

    def dataloader(self, strata: Strata, required: bool = True) -> DataLoader | None:
        strata = Strata.normalize(strata)
        pattern = getattr(self, strata.value)
        if pattern is None:
            if not required:
                return None
            raise ValueError(f"no file pattern configured for strata: {strata}")

        trainer = getattr(self, "trainer", None)
        global_rank = getattr(trainer, "global_rank", None)
        world_size = getattr(trainer, "world_size", None)

        return dataloader(
            hyperparameters=self.hyperparameters,
            root=self.root,
            suffix=self.suffix,
            pattern=pattern,
            preprocessor=self.preprocessor,
            preprocessor_kwargs=self.preprocessor_kwargs,
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
            replacement=self.replacement[strata],
            global_rank=global_rank,
            world_size=world_size,
        )

    train_dataloader = partialmethod(dataloader, strata=Strata.train, required=False)
    val_dataloader = partialmethod(dataloader, strata=Strata.validate, required=False)
    test_dataloader = partialmethod(dataloader, strata=Strata.test, required=False)
    predict_dataloader = partialmethod(dataloader, strata=Strata.predict, required=False)
