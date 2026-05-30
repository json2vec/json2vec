import json2vec.structs.experiment as experiment_module
from json2vec.structs.experiment import Hyperparameters


def _structure_payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "dropout": 0.1,
            "max_length": 1,
            "fields": [
                {
                    "name": "identifier",
                    "type": "category",
                    "max_vocab_size": 1024,
                    "query": "[*].id",
                }
            ],
        },
    }


def test_hyperparameters_supports_programmatic_instantiation():
    hyperparameters = Hyperparameters.model_validate(_structure_payload())

    assert hyperparameters.d_model == 16
    assert "root/identifier" in hyperparameters.requests


def test_experiment_model_is_removed():
    assert not hasattr(experiment_module, "Experiment")
