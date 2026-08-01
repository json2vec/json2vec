from __future__ import annotations

from types import SimpleNamespace

import lightning.pytorch as lit
import polars as pl
import torch
from tensordict import TensorDict

import json2vec as jv
from json2vec.inference.callback import Writer


class _DummyModule:
    def write(self, predictions):
        return {"root/label": {"value": ["ok"]}}


def test_writer_postprocess_receives_batch_context(tmp_path):
    seen = {}

    @jv.postprocess
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
        trainer=SimpleNamespace(local_rank=0),
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


def test_writer_completes_real_prediction_loop_with_throughput_callback(tmp_path):
    model = jv.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        amount=jv.Number,
        label=jv.Category(target=True, size=2, p_unavailable=0.0),
    )
    model.encode(
        [
            {"amount": 1.0, "label": "no"},
            {"amount": 2.0, "label": "yes"},
        ],
        strata="train",
    )

    datamodule = jv.PolarsDataModule(
        model=model,
        predict=pl.DataFrame({"amount": [1.5, 2.5]}),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    output = tmp_path / "predictions"
    trainer = lit.Trainer(
        accelerator="cpu",
        callbacks=[jv.Writer(output)],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)

    frame = pl.read_parquet(output / "rank-0.parquet")
    assert len(frame) == 2
    assert frame.columns == ["inputs", "predictions"]
