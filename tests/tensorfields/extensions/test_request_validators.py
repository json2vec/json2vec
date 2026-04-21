import pytest

from json2vec.structs.structure import Structure


def _structure_with_field(field: dict) -> dict:
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
            "fields": [field],
        },
    }


def test_category_topk_rejects_non_positive():
    payload = _structure_with_field(
        {
            "name": "cat",
            "type": "category",
            "query": "[*].code",
            "max_vocab_size": 64,
            "topk": [0],
        }
    )
    with pytest.raises(ValueError, match="topk values must be positive"):
        Structure.model_validate(payload)


def test_category_topk_rejects_values_at_or_above_vocab():
    payload = _structure_with_field(
        {
            "name": "cat",
            "type": "category",
            "query": "[*].code",
            "max_vocab_size": 8,
            "topk": [8],
        }
    )
    with pytest.raises(ValueError, match="topk values must be less than max_vocab_size"):
        Structure.model_validate(payload)


def test_dateparts_dateparts_reject_duplicates():
    payload = _structure_with_field(
        {
            "name": "ts",
            "type": "dateparts",
            "query": "[*].created_at",
            "dateparts": ["day_of_week", "day_of_week"],
        }
    )
    with pytest.raises(ValueError, match="dateparts must be unique"):
        Structure.model_validate(payload)


def test_dateparts_pattern_rejects_invalid_tokens():
    payload = _structure_with_field(
        {
            "name": "ts",
            "type": "dateparts",
            "query": "[*].created_at",
            "dateparts": ["day_of_week"],
            "pattern": "%Q-%m-%d",
        }
    )
    with pytest.raises(ValueError, match="is not a valid format pattern"):
        Structure.model_validate(payload)


def test_dateparts_pattern_accepts_valid_format():
    payload = _structure_with_field(
        {
            "name": "ts",
            "type": "dateparts",
            "query": "[*].created_at",
            "dateparts": ["day_of_week", "month_of_year"],
            "pattern": "%Y-%m-%d",
        }
    )
    structure = Structure.model_validate(payload)
    assert "root/ts" in structure.requests
