import pytest

from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.tree import Address


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


def test_hyperparameters_accepts_single_string_target():
    payload = _hyperparameters_payload()
    payload["target"] = "root/identifier"

    hyperparameters = Hyperparameters.model_validate(payload)

    assert hyperparameters.target == [Address("root", "identifier")]


def test_hyperparameters_rejects_unknown_embed_array():
    payload = _hyperparameters_payload()
    payload["embed"] = ["root/not_a_array"]

    with pytest.raises(ValueError, match="embed target 'root/not_a_array' not found"):
        Hyperparameters.model_validate(payload)


def test_hyperparameters_accepts_single_address_embed():
    payload = _hyperparameters_payload()
    payload["embed"] = Address("root")

    hyperparameters = Hyperparameters.model_validate(payload)

    assert hyperparameters.embed == [Address("root")]


def test_hyperparameters_rejects_dataset_configuration():
    payload = _hyperparameters_payload()
    payload["dataset"] = {"root": "/tmp/dataset"}

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Hyperparameters.model_validate(payload)


@pytest.mark.parametrize("rate", ["dropout", "p_mask", "p_target"])
def test_hyperparameters_rejects_root_rate_configuration(rate: str):
    payload = _hyperparameters_payload()
    payload[rate] = 0.1

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Hyperparameters.model_validate(payload)


def test_hyperparameters_rejects_invalid_node_mask_rate():
    payload = _hyperparameters_payload()
    payload["fields"]["p_mask"] = 1.0

    with pytest.raises(ValueError):
        Hyperparameters.model_validate(payload)


def test_hyperparameters_rejects_invalid_leaf_target_rate():
    payload = _hyperparameters_payload()
    payload["fields"]["fields"][0]["p_target"] = -0.1

    with pytest.raises(ValueError):
        Hyperparameters.model_validate(payload)
