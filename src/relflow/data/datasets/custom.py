"""Bounded Python-mapping ingress for the canonical Arrow data module."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pyarrow as pa
from torch.utils.data import IterableDataset

import relflow
from relflow.data.arrow import mappings
from relflow.data.datasets.arrow import ArrowDataModule, ArrowStream, Retain
from relflow.data.processors import Preprocessor
from relflow.structs.enums import Strata


def schemas(
    value: pa.Schema | Mapping[Strata | str, pa.Schema] | None,
    *,
    configured: set[Strata],
) -> dict[Strata, pa.Schema | None]:
    """Normalize optional explicit schemas for configured mapping sources."""

    if value is None or isinstance(value, pa.Schema):
        return {strata: value for strata in configured}
    if not isinstance(value, Mapping):
        raise TypeError("arrow_schema must be a pyarrow.Schema, named schema mapping, or None")

    normalized = {Strata.normalize(key): schema for key, schema in value.items()}
    if set(normalized) != configured:
        missing = sorted(str(item) for item in configured - set(normalized))
        extra = sorted(str(item) for item in set(normalized) - configured)
        raise ValueError(f"arrow_schema keys must exactly match configured splits; missing={missing}, extra={extra}")
    if any(not isinstance(schema, pa.Schema) for schema in normalized.values()):
        raise TypeError("every arrow_schema mapping value must be a pyarrow.Schema")
    return normalized


def adapt(
    source: Callable[[], Iterator[Mapping[str, Any]]],
    *,
    schema: pa.Schema | None,
    rows: int,
    name: str,
) -> Callable[[], ArrowStream]:
    """Convert bounded mapping groups into restartable Arrow record batches."""

    inferred = schema
    previous: object | None = None

    def factory() -> Iterator[pa.RecordBatch]:
        nonlocal inferred, previous
        stream = source()
        if stream is previous:
            raise TypeError(f"{name} source must return a fresh iterator for every pass")
        previous = stream
        iterator = iter(stream)

        emitted = False
        while True:
            values: list[Mapping[str, Any]] = []
            for _ in range(rows):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if not isinstance(item, Mapping):
                    raise TypeError(f"{name} source must yield mappings, got {type(item).__name__}")
                if any(not isinstance(key, str) for key in item):
                    raise TypeError(f"{name} source mapping keys must be strings")
                values.append(item)

            if not values:
                if not emitted and inferred is None:
                    raise ValueError(f"{name} source is empty; provide arrow_schema")
                if not emitted and inferred is not None:
                    yield pa.RecordBatch.from_pylist([], schema=inferred)
                return

            emitted = True
            if inferred is None:
                record_batch = pa.RecordBatch.from_pylist(mappings(values, context=f"{name} source"))
                ambiguous = [field.name for field in record_batch.schema if pa.types.is_null(field.type)]
                if ambiguous:
                    names = ", ".join(repr(field) for field in ambiguous)
                    raise TypeError(f"{name} source has all-null field(s) {names}; provide arrow_schema")
                inferred = record_batch.schema
            else:
                try:
                    record_batch = pa.RecordBatch.from_pylist(
                        mappings(values, schema=inferred, context=f"{name} source"),
                        schema=inferred,
                    )
                except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as error:
                    raise TypeError(f"{name} source does not match arrow_schema: {error}") from error

            yield record_batch
            if len(values) < rows:
                return

    return factory


class CustomDataModule(ArrowDataModule):
    """Adapt restartable mapping ``IterableDataset`` splits to Arrow once per chunk."""

    def __init__(
        self,
        model: relflow.Model,
        *,
        train: IterableDataset | None = None,
        validate: IterableDataset | None = None,
        test: IterableDataset | None = None,
        predict: IterableDataset | None = None,
        arrow_schema: pa.Schema | Mapping[Strata | str, pa.Schema] | None = None,
        ingress_rows: int = 4096,
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
        if not isinstance(ingress_rows, int) or isinstance(ingress_rows, bool) or ingress_rows < 1:
            raise ValueError("ingress_rows must be a positive integer")

        datasets = {
            Strata.train: train,
            Strata.validate: validate,
            Strata.test: test,
            Strata.predict: predict,
        }
        configured = {strata for strata, dataset in datasets.items() if dataset is not None}
        if not configured:
            raise ValueError("at least one named data split is required")
        resolved_schemas = schemas(arrow_schema, configured=configured)

        sources: dict[Strata, Callable[[], ArrowStream]] = {}
        for strata, dataset in datasets.items():
            if dataset is None:
                continue
            if not isinstance(dataset, IterableDataset):
                raise TypeError(f"{strata} must be a torch IterableDataset, got {type(dataset).__name__}")
            sources[strata] = adapt(
                dataset.__iter__,
                schema=resolved_schemas[strata],
                rows=ingress_rows,
                name=str(strata),
            )

        super().__init__(
            model=model,
            train=sources.get(Strata.train),
            validate=sources.get(Strata.validate),
            test=sources.get(Strata.test),
            predict=sources.get(Strata.predict),
            preprocessor=preprocessor,
            seed=seed,
            shuffle=shuffle,
            sample=sample,
            replacement=replacement,
            epoch_size=epoch_size,
            shuffle_rows=shuffle_rows,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            retain=retain,
        )


__all__ = ["CustomDataModule"]
