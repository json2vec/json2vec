"""Arrow-native prediction persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightning.pytorch as lit
import pyarrow as pa
import pyarrow.parquet as pq
from lightning.pytorch import callbacks

from relflow.data.arrow import Batch
from relflow.data.processors import Postprocessor, PostprocessorInput, apply


class Writer(callbacks.BasePredictionWriter):
    """Persist each written prediction batch as one Arrow Parquet shard.

    ``Model.predict_step`` has already converted tensors into an
    identity-aligned :class:`~relflow.Batch`. The callback therefore does no
    model writing, shape inference, or Python-row assembly. Configured Arrow
    postprocessors run in order before the protected ``identity`` column is
    added. The first result locks the exact Parquet schema for the rank.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        postprocessor: PostprocessorInput = (),
    ) -> None:
        super().__init__(write_interval="batch")
        self.path = Path(path)
        self.postprocessors = Postprocessor.normalize(postprocessor)
        self.schema: pa.Schema | None = None
        self.writer: pq.ParquetWriter | None = None

    def write_on_batch_end(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        output: Batch,
        batch_indices: list[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:  # ty:ignore[invalid-method-override]
        """Write the Arrow result returned by one prediction step."""

        try:
            if not isinstance(output, Batch):
                raise TypeError(f"Writer requires predict_step to return an rf.Batch, got {type(output).__name__}")

            result = apply(output, self.postprocessors)
            if "identity" in result.data.column_names:
                raise ValueError("prediction output cannot use the reserved column name 'identity'")

            table = pa.table(
                [result.identity, *result.data.columns],
                names=["identity", *result.data.column_names],
            )

            if self.writer is None:
                self.path.mkdir(parents=True, exist_ok=True)
                self.schema = table.schema
                self.writer = pq.ParquetWriter(
                    where=self.path / f"rank-{trainer.global_rank}.parquet",
                    schema=table.schema,
                )
            elif self.schema is None or not table.schema.equals(self.schema, check_metadata=True):
                raise ValueError(
                    f"prediction batch schema differs from the first batch; expected {self.schema}, got {table.schema}"
                )

            self.writer.write_table(table)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the active Parquet shard, if any."""

        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def on_predict_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        self.close()

    def on_exception(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        exception: BaseException,
    ) -> None:
        self.close()


__all__ = ["Writer"]
