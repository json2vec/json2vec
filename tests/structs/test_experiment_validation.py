import pytest

from json2vec.processors.base import PROCESSORS, register
from json2vec.structs.enums import Stage, Strata
from json2vec.structs.experiment import Session


def _processor_name() -> str:
    if PROCESSORS:
        return next(iter(PROCESSORS))

    def _session_test_processor(observation: dict):
        return observation

    _session_test_processor.__name__ = "__session_test_processor"
    register.transformation(_session_test_processor)
    return _session_test_processor.__name__


def _structure_payload() -> dict:
    return {
        "name": "demo",
        "type": "structure",
        "batch_size": 2,
        "dropout": 0.1,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "context",
            "context_size": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "identifier",
                    "type": "entity",
                    "query": "[*].id",
                }
            ],
        },
    }


def _session_payload(task: Stage = Stage.fit) -> dict:
    payload = {
        "name": "session",
        "task": task.value,
        "dataset": {
            "root": "/tmp/dataset",
            "sample_rate": 1.0,
            "file_buffer_size": 32,
            "observation_buffer_size": 64,
            "processor": _processor_name(),
            "kwargs": {},
            "suffix": "ndjson",
            "patterns": {strata.value: ".*" for strata in Strata},
        },
        "structure": _structure_payload(),
        "trainer": {"min_epochs": 1, "max_epochs": 3},
    }

    if task == Stage.fit:
        payload["learning_rate"] = 1e-3

    return payload


def test_session_requires_learning_rate_for_fit():
    payload = _session_payload(Stage.fit)
    payload.pop("learning_rate")

    with pytest.raises(ValueError, match="learning_rate must be defined"):
        Session.model_validate(payload)


def test_session_rejects_learning_rate_when_not_fit():
    payload = _session_payload(Stage.predict)
    payload["learning_rate"] = 1e-3

    with pytest.raises(ValueError, match="must not be defined when task is not 'fit'"):
        Session.model_validate(payload)


def test_session_rejects_patience_when_not_fit():
    payload = _session_payload(Stage.test)
    payload["patience"] = 5

    with pytest.raises(ValueError, match="patience must not be defined when task is not 'fit'"):
        Session.model_validate(payload)


def test_session_validates_trainer_epoch_bounds():
    payload = _session_payload(Stage.fit)
    payload["trainer"] = {"min_epochs": 5, "max_epochs": 2}

    with pytest.raises(ValueError, match="min_epochs must be <= trainer.max_epochs"):
        Session.model_validate(payload)


def test_session_rejects_boolean_epoch_values():
    payload = _session_payload(Stage.fit)
    payload["trainer"] = {"min_epochs": True}

    with pytest.raises(ValueError, match="trainer.min_epochs must be an integer"):
        Session.model_validate(payload)


def test_session_rejects_negative_epoch_values():
    payload = _session_payload(Stage.fit)
    payload["trainer"] = {"max_epochs": -1}

    with pytest.raises(ValueError, match="trainer.max_epochs must be >= 0"):
        Session.model_validate(payload)


def test_session_rejects_unknown_pruned_field():
    payload = _session_payload(Stage.fit)
    payload["pruned"] = ["root/missing"]

    with pytest.raises(ValueError, match="pruned field 'root/missing' not found"):
        Session.model_validate(payload)


def test_session_rejects_unknown_output_context():
    payload = _session_payload(Stage.fit)
    payload["output"] = ["root/not_a_context"]

    with pytest.raises(ValueError, match="output context 'root/not_a_context' not found"):
        Session.model_validate(payload)
