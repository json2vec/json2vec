from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import json2vec.helpers as helpers


def test_tuning_dependencies_are_optional_extra() -> None:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())

    dependencies = pyproject["project"]["dependencies"]
    tuning = pyproject["project"]["optional-dependencies"]["tuning"]

    assert not any(dependency.startswith("optuna") for dependency in dependencies)
    assert not any(dependency.startswith("pydantic-optuna-bridge") for dependency in dependencies)
    assert "optuna>=4.9.0" in tuning
    assert "pydantic-optuna-bridge>=0.1.1" in tuning


def test_tune_import_warns_when_tuning_extra_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in helpers._TUNING_EXPORTS:
        helpers.__dict__.pop(name, None)

    original_import_module = helpers.importlib.import_module

    def import_module(name: str):
        if name == "json2vec.helpers.tuning":
            raise ModuleNotFoundError("No module named 'optuna'", name="optuna")
        return original_import_module(name)

    monkeypatch.setattr(helpers.importlib, "import_module", import_module)

    with pytest.warns(RuntimeWarning, match=r"json2vec\.helpers\.tune requires the tuning extra"):
        with pytest.raises(ModuleNotFoundError, match=r"pip install json2vec\[tuning\]"):
            helpers.tune
