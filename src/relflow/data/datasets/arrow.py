"""The canonical Arrow-backed Lightning data module."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, IterableDataset

import relflow
from relflow.data.arrow import IDENTITY, Batch, matrix, mix
from relflow.data.datasets.base import InterprocessEncodingContext
from relflow.data.iterables import encode
from relflow.data.processors import Preprocessor
from relflow.distributed import world_size
from relflow.structs.enums import Strata

ArrowUnit: TypeAlias = Batch | pa.Table | pa.RecordBatch
ArrowStream: TypeAlias = pa.RecordBatchReader | Iterable[ArrowUnit]
ArrowSource: TypeAlias = ArrowUnit | ds.Dataset | Callable[[], ArrowStream]
Retain: TypeAlias = tuple[str, ...] | Literal["*"]


@dataclass(slots=True)
class Schemas:
    """Persist exact source and processed schemas for one configured split."""

    source: pa.Schema | None = None
    processed: pa.Schema | None = None


def lock(schemas: Schemas, stage: Literal["source", "processed"], actual: pa.Schema, *, context: str) -> None:
    """Record one exact Arrow schema or reject drift from an earlier pass."""

    expected = getattr(schemas, stage)
    if expected is None:
        setattr(schemas, stage, actual)
        return
    if not expected.equals(actual, check_metadata=True):
        raise TypeError(f"{context}: expected {expected}, got {actual}")


def passthrough(value: Any) -> Any:
    """Keep batches intact when Lightning's DataLoader has batching disabled."""

    return value


def splits(
    *,
    train: Any = None,
    validate: Any = None,
    test: Any = None,
    predict: Any = None,
) -> dict[Strata, Any]:
    """Collect explicitly named, non-null data splits."""

    values = {
        Strata.train: train,
        Strata.validate: validate,
        Strata.test: test,
        Strata.predict: predict,
    }
    configured = {strata: value for strata, value in values.items() if value is not None}
    if not configured:
        raise ValueError("at least one named data split is required")
    return configured


def expand(value: Any, *, default: Any) -> dict[Strata, Any]:
    """Expand a scalar or named stratum overrides to all strata."""

    return Strata.expand(value, default=default)


def accept(source: Any, *, strata: Strata) -> ArrowSource:
    """Validate the restartable Arrow source boundary."""

    if isinstance(source, (Batch, pa.Table, pa.RecordBatch, ds.Dataset)) or callable(source):
        return source
    if isinstance(source, ds.Scanner):
        raise TypeError(f"{strata} source is a configured Scanner; pass its Dataset so RelFlow can plan each scan")
    if isinstance(source, pa.RecordBatchReader):
        raise TypeError(f"{strata} source is a one-shot RecordBatchReader; pass a callable that creates a fresh reader")
    if isinstance(source, Mapping):
        raise TypeError(f"{strata} source is a mapping; use CustomDataModule or SyntheticDataModule")
    if isinstance(source, (str, os.PathLike)):
        raise TypeError(f"{strata} source is a path; build a pyarrow Dataset before using ArrowDataModule")
    raise TypeError(
        f"{strata} source must be an rf.Batch, pyarrow Table, pyarrow RecordBatch, Dataset, "
        f"or restartable Arrow factory; got {type(source).__name__}"
    )


def identity(size: int, *, namespace: str, offset: int = 0) -> pa.StructArray:
    """Build stable source-position identities without constructing Python rows."""

    if size < 0 or offset < 0:
        raise ValueError("identity size and offset must be non-negative")

    prefix = np.frombuffer(hashlib.sha256(namespace.encode()).digest()[:24], dtype=np.uint8)
    values = np.empty((size, 32), dtype=np.uint8)
    values[:, :24] = prefix
    positions = np.arange(offset, offset + size, dtype=">u8")
    values[:, 24:] = positions.view(np.uint8).reshape(size, 8)
    logical = pa.FixedSizeBinaryArray.from_buffers(pa.binary(32), size, [None, pa.py_buffer(values)])

    offsets = np.arange(size + 1, dtype=np.int64) * 8
    order = pa.LargeBinaryArray.from_buffers(
        pa.large_binary(),
        size,
        [None, pa.py_buffer(offsets), pa.py_buffer(positions)],
    )
    return pa.StructArray.from_arrays([logical, logical, order], fields=list(IDENTITY))


def convert(unit: ArrowUnit, *, namespace: str, offset: int) -> Batch:
    """Normalize one Arrow unit to the shared carrier."""

    if isinstance(unit, Batch):
        return unit
    if isinstance(unit, pa.RecordBatch):
        table = pa.Table.from_batches([unit])
    elif isinstance(unit, pa.Table):
        table = unit
    else:
        raise TypeError(f"Arrow factories must yield rf.Batch, Table, or RecordBatch; got {type(unit).__name__}")
    return Batch(data=table, identity=identity(table.num_rows, namespace=namespace, offset=offset))


def scan(source: ArrowSource, *, namespace: str, schemas: Schemas | None = None) -> Iterator[Batch]:
    """Read an in-memory unit or restartable Arrow factory without row conversion."""

    schemas = Schemas() if schemas is None else schemas
    if isinstance(source, (Batch, pa.Table, pa.RecordBatch)):
        current = convert(source, namespace=namespace, offset=0)
        lock(schemas, "source", current.data.schema, context="Arrow source schema changed")
        yield current
        return

    stream = source.scanner().to_reader() if isinstance(source, ds.Dataset) else source()
    if isinstance(stream, (Batch, pa.Table, pa.RecordBatch)):
        raise TypeError("an Arrow source factory must return a reader or iterable, not one Arrow unit")
    if not isinstance(stream, (pa.RecordBatchReader, Iterable)):
        raise TypeError(
            "an Arrow source factory must return a RecordBatchReader or iterable of Arrow units; "
            f"got {type(stream).__name__}"
        )

    declared = stream.schema if isinstance(stream, pa.RecordBatchReader) else None
    if declared is not None:
        lock(schemas, "source", declared, context="Arrow source schema changed")

    offset = 0
    emitted = False
    for unit in stream:
        current = convert(unit, namespace=namespace, offset=offset)
        lock(schemas, "source", current.data.schema, context="Arrow source schema changed")
        emitted = True
        offset += len(current)
        yield current
    if emitted:
        return
    if declared is None:
        raise ValueError("an empty Arrow factory must yield an empty Arrow unit carrying its schema")

    empty = pa.Table.from_batches([], schema=declared)
    yield convert(empty, namespace=namespace, offset=0)


def merge(batches: Iterable[Batch]) -> Batch | None:
    """Concatenate aligned Arrow batches with one exact schema."""

    items = list(batches)
    if not items:
        return None

    schema = items[0].data.schema
    for item in items[1:]:
        if not schema.equals(item.data.schema, check_metadata=True):
            raise TypeError(f"Arrow batch schema changed: expected {schema}, got {item.data.schema}")

    data = pa.concat_tables([item.data for item in items]) if len(schema) else pa.Table.from_arrays([], schema=schema)
    chunks: list[pa.Array] = []
    for item in items:
        if isinstance(item.identity, pa.ChunkedArray):
            chunks.extend(item.identity.chunks)
        else:
            chunks.append(item.identity)
    return Batch(data=data, identity=pa.chunked_array(chunks, type=IDENTITY))


def process(
    batches: Iterable[Batch],
    *,
    preprocessor: Preprocessor | None,
    strata: Strata,
    schema: Any,
    encoding_context: InterprocessEncodingContext,
    schemas: Schemas | None = None,
) -> Iterator[Batch]:
    """Apply the configured Arrow preprocessor and enforce one output schema."""

    schemas = Schemas() if schemas is None else schemas
    for item in batches:
        if preprocessor is not None and preprocessor.requires is not None:
            missing = [name for name in preprocessor.requires if name not in item.data.column_names]
            if missing:
                names = ", ".join(repr(name) for name in missing)
                raise KeyError(f"preprocessor '{preprocessor.name}' requires absent column(s): {names}")
        outputs = (
            (item,)
            if preprocessor is None
            else preprocessor.run(
                item,
                strata=strata,
                schema=schema,
                encoding_context=encoding_context,
            )
        )
        for output in outputs:
            if preprocessor is not None:
                missing = [name for name in preprocessor.produces if name not in output.data.column_names]
                if missing:
                    names = ", ".join(repr(name) for name in missing)
                    raise KeyError(f"preprocessor '{preprocessor.name}' did not produce declared column(s): {names}")
            lock(
                schemas,
                "processed",
                output.data.schema,
                context=f"preprocessor schema changed in {strata}",
            )
            yield output


def randomizer(seed: int, *, strata: Strata, epoch: int, operation: str) -> np.uint64:
    """Create one operation-isolated deterministic random salt."""

    payload = f"{seed}:{strata}:{epoch}:{operation}".encode()
    return np.uint64(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def scores(batch: Batch, *, salt: np.uint64, field: str) -> np.ndarray:
    """Derive stable pseudorandom keys from Arrow identity."""

    words = matrix(pc.struct_field(batch.identity, field))
    result = np.full(len(batch), salt, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for column in range(words.shape[1]):
            lane = np.uint64((0x9E3779B97F4A7C15 * (column + 1)) & ((1 << 64) - 1))
            result = mix(result ^ mix(words[:, column] + lane))
    return result


def arrange(batch: Batch, *, salt: np.uint64) -> Batch:
    """Order one bounded batch by its stable random identity key."""

    identities = matrix(pc.struct_field(batch.identity, "instance"))
    random = scores(batch, salt=salt, field="instance")
    order = np.lexsort((*reversed(identities.T), random))
    return batch.take(pa.array(order, type=pa.int64()))


def sample(batches: Iterable[Batch], *, rate: float, salt: np.uint64) -> Iterator[Batch]:
    """Select observations with an identity-stable Arrow boolean mask."""

    if rate >= 1.0:
        yield from batches
        return
    threshold = np.uint64(int(rate * np.iinfo(np.uint64).max))
    for item in batches:
        selected = item.filter(pa.array(scores(item, salt=salt, field="logical") <= threshold))
        if len(selected):
            yield selected


def limit(batches: Iterable[Batch], *, size: int | None) -> Iterator[Batch]:
    """Stop after a fixed number of logical observations."""

    if size is None:
        yield from batches
        return

    remaining = size
    iterator = iter(batches)
    while remaining:
        try:
            item = next(iterator)
        except StopIteration:
            return
        selected = item.slice(0, min(remaining, len(item)))
        remaining -= len(selected)
        if len(selected):
            yield selected


def shuffle(batches: Iterable[Batch], *, rows: int, salt: np.uint64) -> Iterator[Batch]:
    """Mix Arrow batches in a chunk-invariant bounded row buffer."""

    if rows == 1:
        yield from batches
        return

    held: Batch | None = None
    capacity = rows * 2
    for item in batches:
        offset = 0
        while offset < len(item):
            available = capacity - (len(held) if held is not None else 0)
            selected = item.slice(offset, min(available, len(item) - offset))
            held = merge((held, selected)) if held is not None else selected
            offset += len(selected)
            if held is not None and len(held) == capacity:
                ordered = arrange(held, salt=salt)
                yield ordered.slice(0, rows)
                held = ordered.slice(rows)

    if held is not None and len(held):
        yield arrange(held, salt=salt)


def rebatch(batches: Iterable[Batch], *, size: int, drop_last: bool) -> Iterator[Batch]:
    """Form exact model batches by slicing and concatenating Arrow buffers."""

    held: Batch | None = None
    for item in batches:
        held = merge((held, item)) if held is not None else item
        while held is not None and len(held) >= size:
            yield held.slice(0, size)
            held = held.slice(size)

    if held is not None and len(held) and not drop_last:
        yield held


class ArrowDataset(IterableDataset):
    """Feed one Arrow source through the shared model-input pipeline."""

    def __init__(
        self,
        *,
        source: ArrowSource,
        schema: Any,
        preprocessor: Preprocessor | None,
        encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        seed: int,
        shuffle_data: bool,
        sample_rate: float,
        epoch_size: int | None,
        shuffle_rows: int,
        drop_last: bool,
        retain: Retain,
        schemas: Schemas,
        epochs: dict[Strata, int],
    ):
        super().__init__()
        self.source = source
        self.schema = schema
        self.preprocessor = preprocessor
        self.encoding_context = encoding_context
        self.batch_size = batch_size
        self.strata = strata
        self.seed = seed
        self.shuffle_data = shuffle_data
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        self.shuffle_rows = shuffle_rows
        self.drop_last = drop_last
        self.retain = retain
        self.schemas = schemas
        self.epochs = epochs

    def __iter__(self):
        epoch = self.epochs[self.strata] if self.strata == Strata.train else 0
        if self.strata == Strata.train:
            self.epochs[self.strata] += 1
        for field_context in self.encoding_context.values():
            configure = getattr(field_context, "configure_distributed", None)
            if callable(configure):
                configure(global_rank=0, world_size=1)

        scanned: Iterable[Batch] = scan(
            self.source,
            namespace=f"{self.strata}:source",
            schemas=self.schemas,
        )
        if self.preprocessor is not None and self.preprocessor.scope == "dataset":
            materialized = merge(scanned)
            scanned = () if materialized is None else (materialized,)

        batches: Iterable[Batch] = process(
            scanned,
            preprocessor=self.preprocessor,
            strata=self.strata,
            schema=self.schema,
            encoding_context=self.encoding_context,
            schemas=self.schemas,
        )
        batches = sample(
            batches,
            rate=self.sample_rate,
            salt=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="sample"),
        )

        if isinstance(self.source, (Batch, pa.Table, pa.RecordBatch)):
            materialized = merge(batches)
            batches = () if materialized is None else (materialized,)
            if self.shuffle_data and materialized is not None:
                materialized = arrange(
                    materialized,
                    salt=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="shuffle"),
                )
                batches = (materialized,)
            batches = limit(batches, size=self.epoch_size)
        else:
            if callable(self.source):
                batches = limit(batches, size=self.epoch_size)
            if self.shuffle_data:
                batches = shuffle(
                    batches,
                    rows=self.shuffle_rows,
                    salt=randomizer(self.seed, strata=self.strata, epoch=epoch, operation="shuffle"),
                )
            batches = limit(batches, size=self.epoch_size)

        for item in rebatch(batches, size=self.batch_size, drop_last=self.drop_last):
            yield encode(
                batch=item,
                schema=self.schema,
                strata=self.strata,
                interprocess_encoding_context=self.encoding_context,
                seed=self.seed,
                epoch=epoch,
                retain=self.retain,
            )


def loader(
    *,
    source: ArrowSource,
    schema: Any,
    preprocessor: Preprocessor | None,
    encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    seed: int,
    shuffle_data: bool,
    sample_rate: float,
    epoch_size: int | None,
    shuffle_rows: int,
    drop_last: bool,
    retain: Retain,
    pin_memory: bool,
    schemas: Schemas,
    epochs: dict[Strata, int],
) -> DataLoader:
    """Build the sole Lightning DataLoader used by all four data modules."""

    return DataLoader(
        dataset=ArrowDataset(
            source=source,
            schema=schema,
            preprocessor=preprocessor,
            encoding_context=encoding_context,
            batch_size=batch_size,
            strata=strata,
            seed=seed,
            shuffle_data=shuffle_data,
            sample_rate=sample_rate,
            epoch_size=epoch_size,
            shuffle_rows=shuffle_rows,
            drop_last=drop_last,
            retain=retain,
            schemas=schemas,
            epochs=epochs,
        ),
        batch_size=None,
        collate_fn=passthrough,
        num_workers=0,
        persistent_workers=False,
        pin_memory=pin_memory and strata != Strata.predict and torch.cuda.is_available(),
    )


class ArrowDataModule(lit.LightningDataModule):
    """Lightning data module for native in-memory or restartable Arrow sources.

    Tables and record batches stay columnar through preprocessing, selection,
    shuffling, and model rebatching. A callable source must create a fresh
    ``RecordBatchReader`` or Arrow-unit iterable for every iteration.

    Arrow Datasets are scanned afresh for each iteration. Projection pushdown,
    coordinated caches, DataLoader workers, and distributed scheduling are
    intentionally deferred until they can preserve the same identity and
    ordering guarantees as this single-reader implementation.
    """

    def __init__(
        self,
        model: relflow.Model,
        *,
        train: ArrowSource | None = None,
        validate: ArrowSource | None = None,
        test: ArrowSource | None = None,
        predict: ArrowSource | None = None,
        preprocessor: Preprocessor | None | Mapping[Strata | str, Preprocessor | None] = None,
        seed: int = 0,
        shuffle: bool | None | Mapping[Strata | str, bool | None] = None,
        sample: float | Mapping[Strata | str, float] = 1.0,
        replacement: bool | Mapping[Strata | str, bool] = False,
        epoch_size: int | None | Mapping[Strata | str, int | None] = None,
        shuffle_rows: int | None | Mapping[Strata | str, int | None] = None,
        drop_last: bool | Mapping[Strata | str, bool] = False,
        num_workers: int | Mapping[Strata | str, int] = 0,
        persistent_workers: bool | Mapping[Strata | str, bool] = False,
        pin_memory: bool | Mapping[Strata | str, bool] = False,
        retain: Retain | Mapping[Strata | str, Retain] = (),
    ):
        super().__init__()
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        self.model = model
        self.sources = {
            strata: accept(source, strata=strata)
            for strata, source in splits(train=train, validate=validate, test=test, predict=predict).items()
        }
        self.schemas = {strata: Schemas() for strata in self.sources}
        self.epochs = {strata: 0 for strata in self.sources}
        if isinstance(retain, Mapping):
            normalized = {Strata.normalize(key): value for key, value in retain.items()}
            if set(normalized) != set(self.sources):
                missing = set(self.sources) - set(normalized)
                extra = set(normalized) - set(self.sources)
                details = []
                if missing:
                    details.append("missing " + ", ".join(sorted(str(item) for item in missing)))
                if extra:
                    details.append("extra " + ", ".join(sorted(str(item) for item in extra)))
                raise ValueError("retain mapping must match configured splits exactly: " + "; ".join(details))
            self.retain = normalized
        else:
            self.retain = {strata: retain for strata in self.sources}
        for strata, names in self.retain.items():
            if names != "*" and (
                not isinstance(names, tuple)
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"retain for {strata} must be '*', or a tuple of unique non-empty column names")
        self.preprocessor = expand(preprocessor, default=None)
        for strata, processor in self.preprocessor.items():
            self.preprocessor[strata] = Preprocessor.normalize(processor)
            if (
                self.preprocessor[strata] is not None
                and self.preprocessor[strata].scope == "dataset"
                and callable(self.sources.get(strata))
            ):
                raise NotImplementedError(
                    "dataset-scoped preprocessing of an Arrow factory requires the deferred coordinated cache"
                )

        configured_shuffle = expand(shuffle, default=None)
        self.shuffle = {
            strata: strata == Strata.train if enabled is None else enabled
            for strata, enabled in configured_shuffle.items()
        }
        if any(not isinstance(value, bool) for value in self.shuffle.values()):
            raise TypeError("shuffle must contain booleans or None")

        configured_sample = expand(sample, default=1.0)
        if any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in configured_sample.values()):
            raise TypeError("sample must contain numeric probabilities")
        self.sample = {strata: float(rate) for strata, rate in configured_sample.items()}
        if any(not 0.0 < rate <= 1.0 for rate in self.sample.values()):
            raise ValueError("sample must be greater than zero and at most one")

        self.replacement = expand(replacement, default=False)
        if any(not isinstance(value, bool) for value in self.replacement.values()):
            raise TypeError("replacement must contain booleans")
        if any(self.replacement.values()):
            raise NotImplementedError("replacement sampling is deferred until the Arrow identity index is implemented")

        self.epoch_size = expand(epoch_size, default=None)
        if any(
            size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1)
            for size in self.epoch_size.values()
        ):
            raise ValueError("epoch_size must contain positive integers or None")

        configured_rows = expand(shuffle_rows, default=None)
        self.shuffle_rows = {
            strata: max(model.batch_size * 32, 4096) if rows is None else rows
            for strata, rows in configured_rows.items()
        }
        if any(not isinstance(rows, int) or isinstance(rows, bool) or rows < 1 for rows in self.shuffle_rows.values()):
            raise ValueError("shuffle_rows must contain positive integers or None")

        self.drop_last = expand(drop_last, default=False)
        if any(not isinstance(value, bool) for value in self.drop_last.values()):
            raise TypeError("drop_last must contain booleans")

        self.num_workers = expand(num_workers, default=0)
        if any(
            not isinstance(workers, int) or isinstance(workers, bool) or workers < 0
            for workers in self.num_workers.values()
        ):
            raise ValueError("num_workers must contain non-negative integers")
        if any(self.num_workers.values()):
            raise NotImplementedError(
                "Arrow DataLoader workers are deferred until deterministic worker ownership and merge are implemented"
            )

        self.persistent_workers = expand(persistent_workers, default=False)
        self.pin_memory = expand(pin_memory, default=False)
        if any(not isinstance(value, bool) for value in (*self.persistent_workers.values(), *self.pin_memory.values())):
            raise TypeError("persistent_workers and pin_memory must contain booleans")
        if any(self.persistent_workers.values()):
            raise ValueError("persistent_workers requires num_workers > 0; Arrow workers are not implemented yet")
        self.seed = seed

    @property
    def schema(self):
        return self.model.schema

    @property
    def batch_size(self) -> int:
        return self.model.batch_size

    @property
    def encoding_context(self) -> InterprocessEncodingContext:
        return self.model.interprocess_encoding_context

    def dataloader(self, strata: Strata | str, required: bool = True) -> DataLoader | None:
        """Create the shared loader for one configured split."""

        normalized = Strata.normalize(strata)
        if normalized not in self.sources:
            if not required:
                return None
            raise ValueError(f"no source configured for strata: {normalized}")
        if world_size() > 1:
            raise NotImplementedError(
                "distributed Arrow scheduling is deferred until stable rank ownership and equal-step tails are implemented"
            )

        return loader(
            source=self.sources[normalized],
            schema=self.schema,
            preprocessor=self.preprocessor[normalized],
            encoding_context=self.encoding_context,
            batch_size=self.batch_size,
            strata=normalized,
            seed=self.seed,
            shuffle_data=self.shuffle[normalized],
            sample_rate=self.sample[normalized],
            epoch_size=self.epoch_size[normalized],
            shuffle_rows=self.shuffle_rows[normalized],
            drop_last=self.drop_last[normalized],
            retain=self.retain[normalized],
            pin_memory=self.pin_memory[normalized],
            schemas=self.schemas[normalized],
            epochs=self.epochs,
        )

    def train_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.train, required=False)

    def val_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.validate, required=False)

    def test_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.test, required=False)

    def predict_dataloader(self) -> DataLoader | None:
        return self.dataloader(Strata.predict, required=False)


__all__ = ["ArrowDataModule"]
