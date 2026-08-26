from __future__ import annotations

from types import SimpleNamespace

import lightning.pytorch as lit
import polars as pl
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.inference.callback import Writer


class _DummyModule:
    def write(self, predictions):
        return {"root/label": {"value": ["ok"]}}


class _RowsModule:
    def __init__(self, num_rows: int):
        self.num_rows = num_rows

    def write(self, predictions):
        return {"root/label": {"value": ["ok"] * self.num_rows}}


def test_writer_postprocess_receives_batch_context(tmp_path):
    seen = {}

    @rf.postprocess
    def processor(
        predictions,
        *,
        input,
        batch,
        metadata,
        batch_indices,
        batch_idx,
        dataloader_idx,
    ):
        seen["input"] = input
        seen["batch"] = batch
        seen["metadata"] = metadata
        seen["batch_indices"] = batch_indices
        seen["batch_idx"] = batch_idx
        seen["dataloader_idx"] = dataloader_idx
        seen["predictions"] = predictions

    batch = TensorDict(
        {
            "metadata": [{"color": "r"}],
            "dummy": torch.tensor([1]),
        },
        batch_size=[1],
    )
    writer = Writer(path=tmp_path, postprocessor=processor)

    writer.write_on_batch_end(
        trainer=SimpleNamespace(local_rank=0, global_rank=7),
        pl_module=_DummyModule(),
        output={"predictions": []},
        batch_indices=[12],
        batch=batch,
        batch_idx=3,
        dataloader_idx=4,
    )
    writer.on_predict_end(SimpleNamespace(), SimpleNamespace())

    assert seen["input"] is batch
    assert seen["batch"] is batch
    assert list(seen["metadata"]) == [{"color": "r"}]
    assert seen["batch_indices"] == [12]
    assert seen["batch_idx"] == 3
    assert seen["dataloader_idx"] == 4
    assert seen["predictions"]["root/label"]["value"] == ["ok"]
    assert (tmp_path / "rank-7.parquet").exists()
    assert not (tmp_path / "rank-0.parquet").exists()
    assert pl.read_parquet(tmp_path / "rank-7.parquet").to_dicts() == [
        {"inputs": {"color": "r"}, "predictions": {"root/label": {"value": "ok"}}}
    ]


@pytest.mark.parametrize("prediction_rows", [1, 3])
def test_writer_rejects_prediction_rows_misaligned_with_batch(tmp_path, prediction_rows):
    batch = TensorDict(
        {
            "metadata": [{"color": "r"}, {"color": "b"}],
            "dummy": torch.tensor([1, 2]),
        },
        batch_size=[2],
    )
    writer = Writer(path=tmp_path)

    with pytest.raises(ValueError, match=rf"root/label.*has {prediction_rows} rows; expected 2"):
        writer.write_on_batch_end(
            trainer=SimpleNamespace(local_rank=0, global_rank=0),
            pl_module=_RowsModule(prediction_rows),
            output={"predictions": []},
            batch_indices=[0, 1],
            batch=batch,
            batch_idx=0,
            dataloader_idx=0,
        )

    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.parametrize("prediction_rows", [1, 3])
def test_writer_rejects_postprocessor_row_mismatch_and_closes_on_exception(tmp_path, prediction_rows):
    @rf.postprocess
    def processor(predictions, *, batch_idx):
        if batch_idx == 0:
            return predictions
        return {"root/label": {"value": ["ok"] * prediction_rows}}

    batch = TensorDict(
        {
            "metadata": [{"color": "r"}, {"color": "b"}],
            "dummy": torch.tensor([1, 2]),
        },
        batch_size=[2],
    )
    trainer = SimpleNamespace(local_rank=0, global_rank=0)
    module = _RowsModule(num_rows=2)
    writer = Writer(path=tmp_path, postprocessor=processor)

    writer.write_on_batch_end(
        trainer=trainer,
        pl_module=module,
        output={"predictions": []},
        batch_indices=[0, 1],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    with pytest.raises(ValueError, match=rf"root/label.*has {prediction_rows} rows; expected 2") as error:
        writer.write_on_batch_end(
            trainer=trainer,
            pl_module=module,
            output={"predictions": []},
            batch_indices=[2, 3],
            batch=batch,
            batch_idx=1,
            dataloader_idx=0,
        )

    assert writer.writer is not None
    writer.on_exception(trainer, SimpleNamespace(), error.value)
    assert writer.writer is None
    assert len(pl.read_parquet(tmp_path / "rank-0.parquet")) == 2


def test_writer_rejects_later_batch_schema_drift(tmp_path):
    trainer = SimpleNamespace(local_rank=0, global_rank=0)
    module = _RowsModule(num_rows=1)
    writer = Writer(path=tmp_path)
    first = TensorDict(
        {"metadata": [{"id": 1}], "dummy": torch.tensor([1])},
        batch_size=[1],
    )
    drifted = TensorDict(
        {"metadata": [{"id": 2, "extra": "lost"}], "dummy": torch.tensor([2])},
        batch_size=[1],
    )

    writer.write_on_batch_end(
        trainer=trainer,
        pl_module=module,
        output={"predictions": []},
        batch_indices=[0],
        batch=first,
        batch_idx=0,
        dataloader_idx=0,
    )

    with pytest.raises(ValueError, match="schema.*incompatible.*first batch") as error:
        writer.write_on_batch_end(
            trainer=trainer,
            pl_module=module,
            output={"predictions": []},
            batch_indices=[1],
            batch=drifted,
            batch_idx=1,
            dataloader_idx=0,
        )

    writer.on_exception(trainer, SimpleNamespace(), error.value)
    assert writer.writer is None
    assert pl.read_parquet(tmp_path / "rank-0.parquet").to_dicts() == [
        {"inputs": {"id": 1}, "predictions": {"root/label": {"value": "ok"}}}
    ]


def test_writer_completes_real_prediction_loop_with_throughput_callback(tmp_path):
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        amount=rf.Number,
        label=rf.Category(target=True, size=2, p_unavailable=0.0),
    )
    model.encode(
        [
            {"amount": 1.0, "label": "no"},
            {"amount": 2.0, "label": "yes"},
        ],
        strata="train",
    )

    datamodule = rf.PolarsDataModule(
        model=model,
        predict=pl.DataFrame({"amount": [1.5, 2.5]}),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    output = tmp_path / "predictions"
    trainer = lit.Trainer(
        accelerator="cpu",
        callbacks=[rf.Writer(output)],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)

    frame = pl.read_parquet(output / "rank-0.parquet")
    assert len(frame) == 2
    assert frame.columns == ["inputs", "predictions"]
