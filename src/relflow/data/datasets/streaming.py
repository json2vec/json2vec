"""File-backed streaming datasets and Lightning data modules."""

from __future__ import annotations

import os
import random
import re
import sys
import weakref
from collections.abc import Iterable, Iterator
from contextlib import suppress
from functools import partial, partialmethod
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import lightning.pytorch as lit
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import torch
from beartype import beartype
from torch.utils.data import DataLoader, IterableDataset

from relflow.data.datasets.base import (
    DataModuleDisplay,
    InterprocessEncodingContext,
    NonNegativeInt,
    Pipeline,
    PositiveInt,
    PreprocessorConfig,
    RawObservation,
    SampleRate,
    StrataMap,
    _is_assigned_to_worker,
    _worker_buffer_size,
    _worker_identity,
    compact_strata,
    display_label,
    identity,
    share_interprocess_encoding_context,
)
from relflow.data.iterables import (
    JMESPathResolutionMonitor,
    batch,
    mask,
    process,
    sample,
    shuffle,
    transform,
)
from relflow.data.processors import Preprocessor
from relflow.distributed import rank as distributed_rank
from relflow.distributed import world_size as distributed_world_size
from relflow.rich import Incident, IncidentSummary, console, log_incident_summaries
from relflow.structs.enums import ShardingStrategy, Strata, Suffix
from relflow.structs.experiment import Schema

if TYPE_CHECKING:
    from relflow.architecture.root import Model
else:
    Model = "relflow.architecture.root.Model"

PatternInput = str | re.Pattern[str]


def _compile_pattern(pattern: PatternInput) -> re.Pattern[str]:
    return re.compile(pattern) if isinstance(pattern, str) else pattern


def diagnostic_path(uri_path: str | Path, *, limit: int = 240) -> str:
    """Return a useful path label without URI credentials, queries, or fragments."""

    parsed = urlparse(str(uri_path))
    if not parsed.scheme:
        return parsed.path[:limit]

    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return parsed._replace(netloc=hostname, query="", fragment="").geturl()[:limit]


class ReadDiagnostics:
    """Bound streaming read diagnostics across one data-module split lifecycle."""

    MAX_KEYS = 8
    MAX_COUNT = sys.maxsize
    OVERFLOW_KEY = ("streaming-read", "<additional-incidents>")

    def __init__(self) -> None:
        self.manager: SyncManager | None = None
        self.counts: Any = {}
        # overflow count, overflow notice emitted, summary emitted
        self.state: Any = [0, False, False]
        self.lock: Any = Lock()
        self.shared = False
        self.enabled = True

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["manager"] = None
        if not self.shared:
            state["lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.lock is None:
            self.lock = Lock()

    def share(self) -> None:
        """Move counters into process-safe storage before workers start."""

        if self.shared:
            return

        manager: SyncManager | None = None
        try:
            manager = Manager()
            counts = manager.dict(dict(self.counts))
            state = manager.list(list(self.state))
            lock = manager.Lock()
        except Exception:
            if manager is not None:
                with suppress(Exception):
                    manager.shutdown()
            raise

        self.counts = counts
        self.state = state
        self.lock = lock
        self.manager = manager
        self.shared = True

    def disable(self) -> None:
        """Suppress worker diagnostics when their shared budget is unavailable."""

        self.enabled = False

    def record(self, suffix: str, reason: str) -> Incident:
        """Record one failed source and return whether its detail should emit."""

        key = (suffix[:160], reason[:160])
        if not self.enabled:
            return Incident(
                key=("streaming-read", *key),
                count=1,
                suppressed=0,
                emit=False,
            )

        with self.lock:
            previous = self.counts.get(key)
            if previous is not None:
                count = min(int(previous) + 1, self.MAX_COUNT)
                self.counts[key] = count
                return Incident(
                    key=("streaming-read", *key),
                    count=count,
                    suppressed=count - 1,
                    emit=False,
                )

            if len(self.counts) < self.MAX_KEYS:
                self.counts[key] = 1
                return Incident(
                    key=("streaming-read", *key),
                    count=1,
                    suppressed=0,
                    emit=True,
                )

            count = min(int(self.state[0]) + 1, self.MAX_COUNT)
            emit = not bool(self.state[1])
            self.state[0] = count
            self.state[1] = True
            return Incident(
                key=self.OVERFLOW_KEY,
                count=count,
                suppressed=max(count - 1, 0),
                emit=emit,
                overflow=True,
            )

    def summary_for_log(self) -> tuple[IncidentSummary, ...]:
        """Return the first useful suppression summary, then stay quiet."""

        if not self.enabled:
            return ()

        with self.lock:
            counts = [int(count) for count in self.counts.values()]
            overflowed = int(self.state[0])
            overflow_notice = int(bool(self.state[1]))
            suppressed = sum(max(count - 1, 0) for count in counts) + max(
                overflowed - overflow_notice,
                0,
            )
            if not suppressed or bool(self.state[2]):
                return ()

            self.state[2] = True
            return (
                IncidentSummary(
                    kind="streaming-read",
                    occurrences=sum(counts) + overflowed,
                    emitted=len(counts) + overflow_notice,
                    suppressed=suppressed,
                    unique=len(counts),
                    overflowed=overflowed,
                ),
            )


@beartype
def fetch(
    root: str | Path,
    pattern: PatternInput,
    sharding: ShardingStrategy,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[str]:
    regex = _compile_pattern(pattern)
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
            if regex.search(uri_path):
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
    pattern: PatternInput,
    strata: Strata,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    file_buffer_size: int,
    replacement: bool = False,
    global_rank: int | None = None,
    world_size: int | None = None,
    read_diagnostics: ReadDiagnostics | None = None,
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
        max_consecutive_failures=32 if replacement else None,
        read_diagnostics=read_diagnostics,
    )


@beartype
def read(
    pipe: Iterable[str],
    suffix: Suffix,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    global_rank: int | None = None,
    world_size: int | None = None,
    max_consecutive_failures: int | None = None,
    read_diagnostics: ReadDiagnostics | None = None,
) -> Iterator[RawObservation]:
    if max_consecutive_failures is not None and max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be positive")

    worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)
    context = {"worker": worker_id, "workers": num_workers} if num_workers > 1 else {}
    diagnostics = read_diagnostics if read_diagnostics is not None else ReadDiagnostics()
    attempts_since_yield = 0
    total_attempts = 0
    failed_attempts = 0
    yielded_any = False

    def record_failure(uri_path: str, error: BaseException) -> None:
        nonlocal failed_attempts
        failed_attempts += 1
        safe_path = diagnostic_path(uri_path)
        reason = type(error).__qualname__
        try:
            incident = diagnostics.record(suffix.value, reason)
        except Exception:
            return
        if not incident.emit:
            return
        if incident.overflow:
            with suppress(Exception):
                console.log(
                    "[relflow.warning]additional unreadable streaming files are suppressed[/]",
                    context,
                )
            return
        with suppress(Exception):
            console.log(
                "[relflow.warning]skipping an unreadable streaming dataset file[/]",
                {**context, "path": safe_path, "suffix": suffix.value, "reason": reason},
            )

    def check_progress() -> None:
        if max_consecutive_failures is not None and attempts_since_yield >= max_consecutive_failures:
            raise RuntimeError(
                f"streaming reader made no progress after {max_consecutive_failures} consecutive file attempts"
            )

    try:
        match suffix:
            case Suffix.ndjson:
                import json

                for uri_path in pipe:
                    attempts_since_yield += 1
                    total_attempts += 1
                    record_index = 0
                    try:
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

                                elif sharding == ShardingStrategy.record and not _is_assigned_to_worker(
                                    shard_key=f"record:{uri_path}:{record_index}",
                                    worker_id=worker_id,
                                    num_workers=num_workers,
                                ):
                                    record_index += 1
                                    continue

                                record_index += 1
                                row = json.loads(line)
                                attempts_since_yield = 0
                                yielded_any = True
                                yield row
                    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
                        record_failure(uri_path, error)
                    check_progress()

            case Suffix.feather | Suffix.parquet | Suffix.avro | Suffix.csv | Suffix.orc | Suffix.json:
                for uri_path in pipe:
                    attempts_since_yield += 1
                    total_attempts += 1
                    parsed = urlparse(uri_path)

                    if parsed.scheme == "s3":
                        fs = pafs.S3FileSystem()  # type: ignore[attr-defined]
                    elif parsed.scheme in ("", "file"):
                        fs = pafs.LocalFileSystem()
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

                                for row in cast(list[RawObservation], batch.to_pylist()):
                                    attempts_since_yield = 0
                                    yielded_any = True
                                    yield row
                                continue

                            rows = cast(list[RawObservation], batch.to_pylist())

                            if sharding == ShardingStrategy.record:
                                for row_index, row in enumerate(rows):
                                    if _is_assigned_to_worker(
                                        shard_key=f"record:{uri_path}:{chunk_index}:{row_index}",
                                        worker_id=worker_id,
                                        num_workers=num_workers,
                                    ):
                                        attempts_since_yield = 0
                                        yielded_any = True
                                        yield row
                                continue

                            for row in rows:
                                attempts_since_yield = 0
                                yielded_any = True
                                yield row
                    except (FileNotFoundError, UnicodeDecodeError, pa.ArrowInvalid) as error:
                        record_failure(uri_path, error)
                    check_progress()

            case _:
                raise ValueError(f"Unsupported suffix: {suffix}")

        if (
            max_consecutive_failures is None
            and total_attempts > 0
            and failed_attempts == total_attempts
            and not yielded_any
        ):
            raise RuntimeError(f"streaming reader exhausted {total_attempts} file attempts without yielding a row")
    finally:
        with suppress(Exception):
            log_incident_summaries(
                diagnostics.summary_for_log(),
                message="suppressed repeated streaming read diagnostics",
                context=context,
            )


class BatchDataset(IterableDataset):
    def __init__(
        self,
        schema: Schema,
        root: str | Path,
        suffix: Suffix,
        pattern: PatternInput,
        preprocessor: PreprocessorConfig.Value,
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
        read_diagnostics: ReadDiagnostics | None = None,
    ):
        super().__init__()

        self.schema = schema
        self.root = root
        self.suffix = suffix
        self.pattern = pattern
        self.preprocessor = preprocessor
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
        self.read_diagnostics = read_diagnostics if read_diagnostics is not None else ReadDiagnostics()

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        observation_buffer_size = _worker_buffer_size(self.observation_buffer_size)
        yield from (
            Pipeline(
                schema=self.schema,
                root=self.root,
                suffix=self.suffix,
                pattern=self.pattern,
                preprocessor=self.preprocessor,
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
                read_diagnostics=self.read_diagnostics,
            )
            | observe
            | process
            | sample
            | partial(shuffle, size=observation_buffer_size)
            | batch
            | transform
            | mask
        )


def dataloader(
    schema: Schema,
    root: str | Path,
    suffix: Suffix,
    pattern: PatternInput,
    preprocessor: PreprocessorConfig.Value,
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
    read_diagnostics: ReadDiagnostics | None = None,
) -> DataLoader:
    workers = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers = persistent_workers and workers > 0
    active_pin_memory = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    global_rank = distributed_rank() if global_rank is None else global_rank
    world_size = distributed_world_size() if world_size is None else world_size
    diagnostics = read_diagnostics if read_diagnostics is not None else ReadDiagnostics()
    if workers > 0:
        try:
            diagnostics.share()
        except Exception:
            diagnostics.disable()

    return DataLoader(
        dataset=BatchDataset(
            schema=schema,
            root=root,
            suffix=suffix,
            pattern=pattern,
            preprocessor=preprocessor,
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
            read_diagnostics=diagnostics,
        ),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=workers,
        persistent_workers=active_persistent_workers,
        pin_memory=active_pin_memory,
    )


class StreamingDataModule(DataModuleDisplay, lit.LightningDataModule):
    """Lightning data module for streaming records from files.

    Reads file-backed records, applies an optional preprocessor, batches
    observations, and encodes them with model schema.
    """

    @beartype
    def __init__(
        self,
        model: Model,
        root: str | Path,
        suffix: Suffix | str,
        train: PatternInput | None = None,
        validate: PatternInput | None = None,
        test: PatternInput | None = None,
        predict: PatternInput | None = None,
        preprocessor: Preprocessor | None = None,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.file,
        chunk_batch_size: PositiveInt | StrataMap[PositiveInt] = 4096,
        file_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
        replacement: bool | StrataMap[bool] | None = None,
    ):
        super().__init__()

        self.root = root
        self.suffix = Suffix(suffix)
        self.train = _compile_pattern(train) if train is not None else None
        self.validate = _compile_pattern(validate) if validate is not None else None
        self.test = _compile_pattern(test) if test is not None else None
        self.predict = _compile_pattern(predict) if predict is not None else None
        self.preprocessor = PreprocessorConfig.normalize(preprocessor)
        try:
            self._model_ref = weakref.ref(model)
        except TypeError:
            self._model_ref = None
        self._schema = model.schema
        self._interprocess_encoding_context = model.interprocess_encoding_context
        self._batch_size = model.batch_size
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
        self._read_diagnostics = {strata: ReadDiagnostics() for strata in Strata}

    def __rich_repr__(self):
        splits = {strata: None for strata in Strata if getattr(self, strata.value) is not None}
        yield "root", display_label(diagnostic_path(self.root))
        yield "suffix", self.suffix.value
        yield from self.data_module_rich_repr(splits)
        strata = tuple(splits)
        if not strata:
            return
        yield "sharding", compact_strata(self.sharding, strata), ShardingStrategy.file
        yield "chunk_batch_size", compact_strata(self.chunk_batch_size, strata), 4096
        yield "file_buffer_size", compact_strata(self.file_buffer_size, strata), 1
        yield "replacement", compact_strata(self.replacement, strata), False

    def _model(self) -> Model | None:
        if self._model_ref is None:
            return None

        return self._model_ref()

    @property
    def schema(self) -> Schema:
        model = self._model()
        if model is not None:
            return model.schema

        return self._schema

    @schema.setter
    def schema(self, schema: Schema) -> None:
        self._model_ref = None
        self._schema = schema

    @property
    def batch_size(self) -> int:
        model = self._model()
        if model is not None:
            return model.batch_size

        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._model_ref = None
        self._batch_size = batch_size

    @property
    def interprocess_encoding_context(self) -> InterprocessEncodingContext:
        model = self._model()
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

        workers = self.num_workers[strata]
        if workers is None:
            workers = os.cpu_count() or 0

        interprocess_encoding_context = self.interprocess_encoding_context
        if strata == Strata.train and workers > 0:
            share_interprocess_encoding_context(interprocess_encoding_context)

        return dataloader(
            schema=self.schema,
            root=self.root,
            suffix=self.suffix,
            pattern=pattern,
            preprocessor=self.preprocessor,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=self.batch_size,
            strata=strata,
            num_workers=workers,
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
            read_diagnostics=self._read_diagnostics[strata],
        )

    train_dataloader = partialmethod(dataloader, strata=Strata.train, required=False)
    val_dataloader = partialmethod(dataloader, strata=Strata.validate, required=False)
    test_dataloader = partialmethod(dataloader, strata=Strata.test, required=False)
    predict_dataloader = partialmethod(dataloader, strata=Strata.predict, required=False)
