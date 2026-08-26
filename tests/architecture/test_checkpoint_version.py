"""Checkpoint package-version metadata contracts."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest
import torch

import relflow as rf


def _model() -> rf.Model:
    return rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        attention="none",
        amount=rf.Number,
    )


def test_public_version_matches_installed_distribution() -> None:
    assert rf.__version__ == version("relflow")


def _checkpoint(model: rf.Model, **metadata: object) -> dict:
    return {
        "state_dict": model.state_dict(),
        "schema": model.schema.model_dump(mode="python"),
        "batch_size": model.batch_size,
        **metadata,
    }


def test_new_model_version_is_current_and_immutable() -> None:
    model = _model()

    assert model.version == rf.__version__
    with pytest.raises(AttributeError):
        model.version = "1.2.3"
    assert model.version == rf.__version__


def test_checkpoint_dump_records_version() -> None:
    checkpoint: dict = {}
    model = _model()

    model.on_save_checkpoint(checkpoint)

    assert checkpoint["version"] == model.version
    assert "relflow_version" not in checkpoint


def test_model_save_persists_version(tmp_path: Path) -> None:
    pathname = tmp_path / "model.ckpt"
    model = _model()

    model.save(pathname)

    checkpoint = torch.load(pathname, weights_only=False, map_location="cpu")
    assert checkpoint["version"] == model.version
    assert "relflow_version" not in checkpoint


def test_model_load_accepts_legacy_checkpoint_without_version(tmp_path: Path) -> None:
    pathname = tmp_path / "legacy.ckpt"
    model = _model()
    torch.save(_checkpoint(model), pathname)

    restored = rf.Model.load(pathname)

    assert restored.schema.model_dump(mode="python") == model.schema.model_dump(mode="python")
    assert restored.batch_size == model.batch_size
    assert restored.version == "0+unknown"


def test_model_load_and_resave_preserve_version_without_gating(tmp_path: Path) -> None:
    pathname = tmp_path / "future.ckpt"
    model = _model()
    torch.save(_checkpoint(model, version="999.0.0"), pathname)

    restored = rf.Model.load(pathname)

    assert restored.schema.model_dump(mode="python") == model.schema.model_dump(mode="python")
    assert restored.version == "999.0.0"

    resaved = tmp_path / "resaved.ckpt"
    restored.save(resaved)
    state = torch.load(resaved, weights_only=False, map_location="cpu")
    assert state["version"] == "999.0.0"


def test_in_place_and_lightning_restore_update_version() -> None:
    model = _model()
    checkpoint = _checkpoint(model, version="1.2.3")

    model.restore_checkpoint_state(checkpoint)
    assert model.version == "1.2.3"

    model.on_load_checkpoint({"version": "2.0.0"})
    assert model.version == "2.0.0"
