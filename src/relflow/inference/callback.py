from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from lightning.pytorch import callbacks
from tensordict import TensorDict

from relflow.data.processors import Postprocessor
from relflow.structs.enums import TensorKey
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import TensorFieldBase

if TYPE_CHECKING:
    from relflow.architecture.root import Model


def _list_shape(data_type: pa.DataType) -> tuple[str, int | None] | None:
    if pa.types.is_list(data_type):
        return ("list", None)
    if pa.types.is_large_list(data_type):
        return ("large_list", None)
    if pa.types.is_fixed_size_list(data_type):
        return ("fixed_size_list", data_type.list_size)
    return None


def _types_have_compatible_shape(actual: pa.DataType, expected: pa.DataType) -> bool:
    if pa.types.is_null(actual):
        return True
    if pa.types.is_null(expected):
        return False

    if pa.types.is_struct(actual) or pa.types.is_struct(expected):
        if not pa.types.is_struct(actual) or not pa.types.is_struct(expected):
            return False
        if [field.name for field in actual] != [field.name for field in expected]:
            return False
        return all(
            _types_have_compatible_shape(actual_field.type, expected_field.type)
            for actual_field, expected_field in zip(actual, expected, strict=True)
        )

    actual_list = _list_shape(actual)
    expected_list = _list_shape(expected)
    if actual_list is not None or expected_list is not None:
        if actual_list != expected_list:
            return False
        return _types_have_compatible_shape(actual.value_type, expected.value_type)

    if pa.types.is_map(actual) or pa.types.is_map(expected):
        if not pa.types.is_map(actual) or not pa.types.is_map(expected):
            return False
        return _types_have_compatible_shape(actual.key_type, expected.key_type) and _types_have_compatible_shape(
            actual.item_type, expected.item_type
        )

    if pa.types.is_nested(actual) or pa.types.is_nested(expected):
        return actual == expected

    return True


def _schemas_have_compatible_shape(actual: pa.Schema, expected: pa.Schema) -> bool:
    if actual.names != expected.names:
        return False
    return all(
        _types_have_compatible_shape(actual_field.type, expected_field.type)
        for actual_field, expected_field in zip(actual, expected, strict=True)
    )


class Writer(callbacks.BasePredictionWriter):
    def __init__(
        self,
        path: os.PathLike | str,
        flush_every_n_batches: int | None = None,
        postprocessor: Postprocessor | None = None,
    ):
        super().__init__(write_interval="batch")

        self.path = Path(path)
        self.flush_every_n_batches: int | None = flush_every_n_batches
        self.postprocessor: Postprocessor | None = Postprocessor.normalize(postprocessor)
        self.schema: pa.Schema | None = None
        self.writer: pq.ParquetWriter | None = None

    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: Model,
        output: dict[str, list[Prediction]],
        batch_indices: list[int] | None,
        batch: TensorDict[Address, TensorFieldBase],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:  # ty:ignore[invalid-method-override]
        num_rows = len(batch[TensorKey.metadata])

        predictions: dict[Address, dict[str, Any]] = pl_module.write(predictions=output["predictions"])
        postprocessor = self.postprocessor

        if postprocessor is not None:
            processed = postprocessor.run(
                predictions,
                available={
                    "input": batch,
                    "batch": batch,
                    "metadata": batch[TensorKey.metadata],
                    "batch_indices": batch_indices,
                    "batch_idx": batch_idx,
                    "dataloader_idx": dataloader_idx,
                },
            )

            if processed is not None:
                predictions = dict(processed)

        if len(predictions) == 0:
            predictions_frame = pl.DataFrame({"predictions": [None] * num_rows})
        else:
            columns: list[pl.DataFrame] = []
            for address, values in predictions.items():
                field_frame = pl.DataFrame(data=values)
                if field_frame.height != num_rows:
                    raise ValueError(
                        f"prediction output at address {str(address)!r} has {field_frame.height} rows; "
                        f"expected {num_rows}"
                    )
                columns.append(field_frame.select(pl.struct(pl.all()).alias(name=address)))

            nested: pl.DataFrame = pl.concat(items=columns, how="horizontal")
            predictions_frame = nested.select(pl.struct(pl.all()).alias(name="predictions"))

        items = [
            pl.DataFrame({"inputs": batch[TensorKey.metadata]}),
            predictions_frame,
        ]

        table: pa.Table = pl.concat(items=items, how="horizontal").to_arrow()

        if self.writer is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema

            self.writer = pq.ParquetWriter(
                where=self.path / f"rank-{trainer.global_rank}.parquet",
                schema=self.schema,
            )

        if table.schema != self.schema:
            if not _schemas_have_compatible_shape(table.schema, self.schema):
                raise ValueError(
                    "batch schema is structurally incompatible with the first batch; "
                    "input and prediction fields must remain stable"
                )
            table = table.cast(self.schema)

        self.writer.write_table(table)

        flush = getattr(self.writer, "flush", None)
        if self.flush_every_n_batches and (batch_idx + 1) % self.flush_every_n_batches == 0 and callable(flush):
            flush()

    def _close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self._close()

    def on_exception(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        exception: BaseException,
    ) -> None:
        self._close()
