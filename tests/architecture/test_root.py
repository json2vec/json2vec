from __future__ import annotations

import json
from pathlib import Path

import torch

from json2vec.architecture.root import JSON2Vec
from json2vec.structs.enums import Stage, Strata, Suffix
from json2vec.structs.experiment import Dataset, Session
from json2vec.structs.structure import Structure


def _structure() -> Structure:
    return Structure.model_validate(
        {
            "name": "checkpoint-root",
            "type": "structure",
            "batch_size": 2,
            "dropout": 0.1,
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "context",
                "context_size": 1,
                "n_outputs": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "query": "[*].label",
                        "max_vocab_size": 32,
                    }
                ],
            },
        }
    )


def _session(dataset_root: Path) -> Session:
    dataset = Dataset.model_validate(
        {
            "root": str(dataset_root),
            "sample_rate": 1.0,
            "file_buffer_size": 4,
            "observation_buffer_size": 4,
            "processor": "default",
            "kwargs": {},
            "suffix": Suffix.ndjson,
            "patterns": {strata: r".*\.ndjson$" for strata in Strata},
        }
    )

    return Session.model_validate(
        {
            "name": "checkpoint-root",
            "dataset": dataset,
            "structure": _structure(),
            "task": Stage.predict,
            "output": ["root/label"],
        }
    )


def _build_checkpoint(tmp_path: Path) -> tuple[Path, Session]:
    dataset_path = tmp_path / "fake_records.ndjson"
    dataset_path.write_text(json.dumps({"label": "alpha"}), encoding="utf-8")
    session = _session(dataset_root=dataset_path)
    model = JSON2Vec.get_or_create(session=session)
    checkpoint_path = tmp_path / "model.ckpt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "session": session.model_dump(mode="python"),
        },
        checkpoint_path,
    )

    return checkpoint_path, session


class FakeS3FileSystem:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.opened_paths: list[str] = []

    def open_input_file(self, path: str):
        self.opened_paths.append(path)
        return self.checkpoint_path.open("rb")


def test_get_or_create_loads_checkpoint_from_s3_uri(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path, session = _build_checkpoint(tmp_path)
    filesystem = FakeS3FileSystem(checkpoint_path=checkpoint_path)
    monkeypatch.setattr("json2vec.architecture.root.pafs.S3FileSystem", lambda: filesystem)

    model = JSON2Vec.get_or_create(checkpoint="s3://bucket/models/model.ckpt")

    assert filesystem.opened_paths == ["bucket/models/model.ckpt"]
    assert model.session.name == session.name
