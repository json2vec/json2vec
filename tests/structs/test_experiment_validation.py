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


def test_schema_rejects_reconstruct_constructor_list():
    payload = _schema_payload()
    payload["reconstruct"] = ["root/missing"]

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


def test_schema_derives_effective_reconstruct_roles_from_masks():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["mask"] = {"reconstruct": True, "dropout": False}

    schema = Schema.model_validate(payload)

    assert schema.reconstruct == ["root/items/identifier"]
    assert schema.objectives == ["root/items/identifier"]
    assert schema.decodes == ["root/items/identifier"]
    assert schema.forward_for("train") == ["root/items/identifier"]
    assert schema.forward_for("predict") == ["root/items/identifier"]


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


def test_schema_rejects_removed_branch_mask_rate():
    payload = _schema_payload()
    payload["fields"]["p_mask"] = 1.0

    with pytest.raises(ValueError, match="removed node field"):
        Schema.model_validate(payload)


def test_schema_rejects_removed_leaf_target_rate():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["p_prune"] = -0.1

    with pytest.raises(ValueError):
        Schema.model_validate(payload)


def test_schema_excludes_dynamic_reconstruct_from_prediction_decodes():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["mask"] = {
        "rate": 0.5,
        "reconstruct": True,
        "dropout": False,
    }

    schema = Schema.model_validate(payload)

    assert schema.objectives == ["root/items/identifier"]
    assert schema.decodes == []
    assert schema.forward_for("train") == ["root/items/identifier"]
    assert schema.forward_for("predict") == []


def test_schema_mask_round_trip_preserves_skip_policy():
    payload = _schema_payload()
    payload["fields"]["mask"] = {"skip": True, "reconstruct": True, "dropout": False}
    schema = Schema.model_validate(payload)

    restored = Schema.model_validate(schema.model_dump(mode="python", round_trip=True))

    assert restored.fields.mask == schema.fields.mask
    assert restored.reconstruct == ["root/items/identifier"]

    restored_json = Schema.model_validate_json(schema.model_dump_json())

    assert restored_json.fields.mask == schema.fields.mask
    assert restored_json.reconstruct == ["root/items/identifier"]
