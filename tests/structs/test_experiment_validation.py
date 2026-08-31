import pytest

from relflow.structs.experiment import Schema


def _structure_payload() -> dict:
    field: dict = {
        "name": "identifier",
        "type": "hash",
    }
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def _schema_payload() -> dict:
    return _structure_payload()


def test_schema_rejects_target_constructor_list():
    payload = _schema_payload()
    payload["target"] = ["root/missing"]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


def test_schema_derives_target_from_node_prune_rate():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["p_prune"] = 1.0
    payload["fields"]["fields"][0]["fields"][0]["embed"] = False

    schema = Schema.model_validate(payload)

    assert schema.target == ["root/items/identifier"]


def test_schema_rejects_embed_constructor_list():
    payload = _schema_payload()
    payload["embed"] = ["root/not_a_branch"]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


def test_schema_derives_embed_from_node_attribute():
    payload = _schema_payload()
    payload["fields"]["embed"] = True

    schema = Schema.model_validate(payload)

    assert schema.embed == ["root"]


def test_schema_rejects_dataset_configuration():
    payload = _schema_payload()
    payload["dataset"] = {"root": "/tmp/dataset"}

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


@pytest.mark.parametrize("rate", ["dropout", "p_mask", "p_prune"])
def test_schema_rejects_root_rate_configuration(rate: str):
    payload = _schema_payload()
    payload[rate] = 0.1

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


def test_schema_rejects_invalid_node_mask_rate():
    payload = _schema_payload()
    payload["fields"]["p_mask"] = 1.0

    with pytest.raises(TypeError, match="tree field 'p_mask'"):
        Schema.model_validate(payload)


def test_schema_rejects_invalid_leaf_target_rate():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["p_prune"] = -0.1

    with pytest.raises(ValueError):
        Schema.model_validate(payload)
