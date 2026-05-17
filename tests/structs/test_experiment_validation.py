import pytest

from json2vec.structs.experiment import Hyperparameters


def _structure_payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "dropout": 0.1,
            "max_length": 2,
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


def _hyperparameters_payload() -> dict:
    return _structure_payload()


def test_hyperparameters_rejects_unknown_target_field():
    payload = _hyperparameters_payload()
    payload["target"] = ["root/missing"]

    with pytest.raises(ValueError, match="target field 'root/missing' not found"):
        Hyperparameters.model_validate(payload)


def test_hyperparameters_rejects_unknown_embed_array():
    payload = _hyperparameters_payload()
    payload["embed"] = ["root/not_a_array"]

    with pytest.raises(ValueError, match="embed target 'root/not_a_array' not found"):
        Hyperparameters.model_validate(payload)


def test_hyperparameters_rejects_dataset_configuration():
    payload = _hyperparameters_payload()
    payload["dataset"] = {"root": "/tmp/dataset"}

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Hyperparameters.model_validate(payload)
