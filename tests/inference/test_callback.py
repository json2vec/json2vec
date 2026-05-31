from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from json2vec.inference.callback import Writer


class _DummyModule:
    def write(self, predictions):
        return {"root/label": {"value": ["ok"]}}


def test_writer_postprocess_receives_batch_context(tmp_path):
    seen = {}

    def processor(context, predictions):
        seen["context"] = context
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

    assert seen["context"]["input"] is batch
    assert seen["context"]["batch"] is batch
    assert list(seen["context"]["metadata"]) == [{"color": "r"}]
    assert seen["context"]["batch_indices"] == [12]
    assert seen["context"]["batch_idx"] == 3
    assert seen["context"]["dataloader_idx"] == 4
    assert seen["predictions"]["root/label"]["value"] == ["ok"]
