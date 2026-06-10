from __future__ import annotations

from typing import Any

import pydantic
import torch
from pydantic_optuna_bridge import get_search_space

import json2vec as j2v
from json2vec.helpers.optimizers import adamw


class DummyTrial:
    def __init__(self, choices: str = "first") -> None:
        self.choices = choices
        self.params: dict[str, Any] = {}

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        value = low if self.choices == "first" else high
        self.params[name] = value
        return value

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        value = low if self.choices == "first" else high
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = choices[0] if self.choices == "first" else choices[-1]
        self.params[name] = value
        return value


def _hyperparameters() -> j2v.Hyperparameters:
    return j2v.Hyperparameters.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            name="transactions",
            max_length=4,
        ),
        j2v.Category("label", target=True, max_vocab_size=2),
        name="record",
        d_model=16,
        n_layers=1,
        n_heads=4,
    )


def test_tuning_models_are_pydantic_and_bridge_configured() -> None:
    objects = [
        j2v.helpers.Root(),
        j2v.helpers.Array(),
        j2v.helpers.Leaf(),
        j2v.helpers.Optimizer(),
    ]

    assert all(isinstance(item, pydantic.BaseModel) for item in objects)
    assert get_search_space(j2v.helpers.Root)["d_model"]["distribution"] == "categorical"
    assert get_search_space(j2v.helpers.Optimizer)["learning_rate"]["log"] is True


def test_compound_spaces_materialize_concrete_values_from_trial() -> None:
    trial = DummyTrial()

    root = j2v.helpers.Root.from_trial(trial)

    assert root.as_dict()["d_model"] == 32
    assert root.as_dict()["n_heads"] == 2
    assert "root.d_model" in trial.params
    assert "root.n_heads" in trial.params


def test_tune_returns_model_without_mutating_input_hyperparameters() -> None:
    hyperparameters = _hyperparameters()
    original = hyperparameters.model_dump(mode="python", round_trip=True)
    trial = DummyTrial()

    model = j2v.helpers.tune(hyperparameters, trial=trial, optimizer=adamw, batch_size=8)

    assert isinstance(model, j2v.Model)
    assert model.batch_size == 8
    assert model.hyperparameters.d_model == 32
    assert model.hyperparameters.fields.n_heads == 2
    assert model.hyperparameters.arrays["record/transactions"].n_heads == 2
    assert model.hyperparameters.requests["record/transactions/amount"].pooling == "query"
    assert model.hyperparameters.requests["record/label"].p_prune == 1.0
    assert hyperparameters.model_dump(mode="python", round_trip=True) == original

    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-5


def test_tune_samples_prefixed_optimizer_parameters() -> None:
    trial = DummyTrial()

    model = j2v.helpers.tune(_hyperparameters(), trial=trial)

    assert model.optimizer is not None
    assert "optimizer.learning_rate" in trial.params
    assert "optimizer.weight_decay" in trial.params
    assert "optimizer.beta1" in trial.params
    assert "optimizer.beta2" in trial.params
