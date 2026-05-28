from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import lightning.pytorch as lit
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from lightning.pytorch import callbacks
from tensordict import TensorDict

from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TensorFieldBase

if TYPE_CHECKING:
    from json2vec.architecture.root import Model

Postprocessor: TypeAlias = Callable[
    [dict[str, Any], dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]],
    tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]] | None,
]


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
        self.postprocessor: Postprocessor | None = postprocessor
        self.schema: pa.Schema | None = None
        self.writer: pq.ParquetWriter | None = None

    @staticmethod
    def _as_struct_frame(values_by_address: dict[Address, dict[str, Any]], alias: str, num_rows: int) -> pl.DataFrame:
        if len(values_by_address) == 0:
            return pl.DataFrame({alias: [None] * num_rows})

        columns: list[pl.DataFrame] = []
        for address, values in values_by_address.items():
            field_frame = pl.DataFrame(data=values)
            columns.append(field_frame.select(pl.struct(pl.all()).alias(name=address)))

        nested: pl.DataFrame = pl.concat(items=columns, how="horizontal")
        return nested.select(pl.struct(pl.all()).alias(name=alias))

    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: Model,
        output: dict[str, list[Prediction]],
        batch_indices: list[int] | None,
        batch: TensorDict[Address, TensorFieldBase],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        num_rows = len(batch["metadata"])

        supervised: dict[Address, dict[str, Any]]
        embeddings: dict[Address, dict[str, Any]]

        supervised, embeddings = pl_module.write(predictions=output["predictions"])
        postprocessor = self.postprocessor

        if postprocessor is not None:
            context = {
                "input": batch,
                "batch": batch,
                "metadata": batch["metadata"],
                "batch_indices": batch_indices,
                "batch_idx": batch_idx,
                "dataloader_idx": dataloader_idx,
            }
            processed = postprocessor(context, supervised, embeddings)

            if processed is not None:
                supervised, embeddings = processed

        items = [
            pl.from_records(data=batch["metadata"], schema=["inputs"], orient="row"),
            self._as_struct_frame(values_by_address=supervised, alias="predictions", num_rows=num_rows),
        ]

        if embeddings:
            items.append(self._as_struct_frame(values_by_address=embeddings, alias="embeddings", num_rows=num_rows))

        table: pa.Table = pl.concat(items=items, how="horizontal").to_arrow()

        if self.writer is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema

            self.writer = pq.ParquetWriter(
                where=self.path / f"rank-{trainer.local_rank}.parquet",
                schema=self.schema,
            )

        if table.schema != self.schema:
            table = table.cast(self.schema)

        self.writer.write_table(table)

        flush = getattr(self.writer, "flush", None)
        if self.flush_every_n_batches and (batch_idx + 1) % self.flush_every_n_batches == 0 and callable(flush):
            flush()

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None
