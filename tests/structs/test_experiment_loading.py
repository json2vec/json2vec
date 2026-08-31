import relflow.structs.experiment as experiment_module
from relflow.structs.experiment import Schema


def _structure_payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "length": 1,
            "fields": [
                {
                    "name": "identifier",
                    "type": "category",
                    "size": 1024,
                }
            ],
        },
    }


def test_schema_supports_programmatic_instantiation():
    schema = Schema.model_validate(_structure_payload())

    assert schema.d_model == 16
    assert "root/identifier" in schema.requests


def test_experiment_model_is_removed():
    assert not hasattr(experiment_module, "Experiment")
