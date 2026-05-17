from __future__ import annotations

from pathlib import Path

import torch

from json2vec.architecture.root import JSON2Vec
from json2vec.data.datasets import encode
from json2vec.structs.enums import Strata
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.tree import Address


def _hyperparameters() -> Hyperparameters:
    return Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "array",
                "dropout": 0.1,
                "max_length": 1,
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


def _prediction_hyperparameters() -> Hyperparameters:
    return Hyperparameters(
        d_model=8,
        target=Address("root", "label"),
        embed=Address("root"),
        fields={
            "name": "root",
            "type": "array",
            "max_length": 1,
            "n_outputs": 1,
            "attention": "none",
            "fields": [
                {
                    "name": "color",
                    "type": "category",
                    "query": "[*].color",
                    "max_vocab_size": 16,
                },
                {
                    "name": "label",
                    "type": "category",
                    "query": "[*].label",
                    "max_vocab_size": 16,
                    "topk": [2],
                },
            ],
        },
    )


def _primed_prediction_model() -> JSON2Vec:
    hyperparameters = _prediction_hyperparameters()
    model = JSON2Vec(hyperparameters=hyperparameters, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        hyperparameters=hyperparameters,
        strata=Strata.train,
        state=model.state,
    )

    model(inputs)
    return model


def _build_checkpoint(tmp_path: Path) -> tuple[Path, Hyperparameters]:
    hyperparameters = _hyperparameters()
    model = JSON2Vec.get_or_create(hyperparameters=hyperparameters, batch_size=2)
    checkpoint_path = tmp_path / "model.ckpt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparameters": hyperparameters.model_dump(mode="python"),
        },
        checkpoint_path,
    )

    return checkpoint_path, hyperparameters


class FakeS3FileSystem:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.opened_paths: list[str] = []

    def open_input_file(self, path: str):
        self.opened_paths.append(path)
        return self.checkpoint_path.open("rb")


def test_get_or_create_loads_checkpoint_from_s3_uri(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path, hyperparameters = _build_checkpoint(tmp_path)
    filesystem = FakeS3FileSystem(checkpoint_path=checkpoint_path)
    monkeypatch.setattr("json2vec.architecture.root.pafs.S3FileSystem", lambda: filesystem)

    model = JSON2Vec.get_or_create(checkpoint="s3://bucket/models/model.ckpt")

    assert filesystem.opened_paths == ["bucket/models/model.ckpt"]
    assert model.hyperparameters.model_dump(mode="python") == hyperparameters.model_dump(mode="python")


def test_configure_optimizers_uses_user_supplied_optimizer(tmp_path: Path) -> None:
    _, hyperparameters = _build_checkpoint(tmp_path)
    model = JSON2Vec.get_or_create(
        hyperparameters=hyperparameters,
        batch_size=2,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
    )
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)


def test_configure_optimizers_uses_user_supplied_scheduler(tmp_path: Path) -> None:
    _, hyperparameters = _build_checkpoint(tmp_path)
    model = JSON2Vec.get_or_create(
        hyperparameters=hyperparameters,
        batch_size=2,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
        scheduler=lambda _module, optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    )

    configured = model.configure_optimizers()

    assert isinstance(configured["optimizer"], torch.optim.AdamW)
    assert isinstance(configured["lr_scheduler"], torch.optim.lr_scheduler.StepLR)


def test_predict_encodes_batch_and_returns_supervised_outputs() -> None:
    model = _primed_prediction_model()
    model.train()

    supervised = model.predict(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ]
    )

    assert model.training
    assert Address("root", "label") in supervised
    content = supervised[Address("root", "label")]["content"]
    state = supervised[Address("root", "label")]["state"]

    assert len(content["value"]) == 2
    assert all(not isinstance(value, list) for value in content["value"])
    assert all(not isinstance(probability, list) for probability in content["probability"])
    assert len(content["topk"]) == 2
    assert all(row and isinstance(row[0], dict) for row in content["topk"])
    assert all(
        len(probabilities) == 2 and all(not isinstance(probability, list) for probability in probabilities)
        for probabilities in state.values()
    )


def test_embed_encodes_batch_and_returns_embedding_outputs() -> None:
    model = _primed_prediction_model()

    embeddings = model.embed(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ]
    )

    assert Address("root") in embeddings
    embedding = embeddings[Address("root")]["embedding"]
    assert len(embedding) == 2
    assert all(not isinstance(row[0], list) for row in embedding)
