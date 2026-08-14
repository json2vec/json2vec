import pytest

from relflow.structs.experiment import Schema


def _structure_with_field(field: dict) -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "length": 2,
            "fields": [field],
        },
    }


def test_category_topk_rejects_non_positive():
    payload = _structure_with_field(
        {
            "name": "cat",
            "type": "category",
            "query": "[*].code",
            "size": 64,
            "topk": [0],
        }
    )
    with pytest.raises(ValueError, match="topk values must be positive"):
        Schema.model_validate(payload)


def test_category_topk_rejects_values_at_or_above_vocab():
    payload = _structure_with_field(
        {
            "name": "cat",
            "type": "category",
            "query": "[*].code",
            "size": 8,
            "topk": [8],
        }
    )
    with pytest.raises(ValueError, match="topk values must be less than size"):
        Schema.model_validate(payload)


def test_category_allows_extra_n_bands_option():
    payload = _structure_with_field(
        {
            "name": "cat",
            "type": "category",
            "query": "[*].code",
            "size": 64,
            "n_bands": 8,
        }
    )
    schema = Schema.model_validate(payload)

    assert schema.requests["root/cat"].n_bands == 8


def test_set_threshold_rejects_values_above_one():
    payload = _structure_with_field(
        {
            "name": "tags",
            "type": "set",
            "query": "[*].tags",
            "threshold": 1.1,
        }
    )
    with pytest.raises(ValueError, match="less than or equal to 1"):
        Schema.model_validate(payload)


def test_boolean_threshold_rejects_values_above_one():
    payload = _structure_with_field(
        {
            "name": "enabled",
            "type": "boolean",
            "query": "[*].enabled",
            "threshold": 1.1,
        }
    )
    with pytest.raises(ValueError, match="less than or equal to 1"):
        Schema.model_validate(payload)


@pytest.mark.parametrize(
    ("threshold", "match"),
    [
        ([], "at least 1 item"),
        ([-0.1, 0.5], "greater than or equal to 0"),
        ([0.5, 1.1], "less than or equal to 1"),
    ],
)
def test_boolean_threshold_rejects_invalid_lists(threshold: list[float], match: str):
    payload = _structure_with_field(
        {
            "name": "enabled",
            "type": "boolean",
            "query": "[*].enabled",
            "threshold": threshold,
        }
    )

    with pytest.raises(ValueError, match=match):
        Schema.model_validate(payload)


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
        Schema.model_validate(payload)


def test_dateparts_normalizes_friendly_datepart_names():
    payload = _structure_with_field(
        {
            "name": "ts",
            "type": "dateparts",
            "query": "[*].created_at",
            "dateparts": ["Day Of Week", "month-of-year", "HourOfDay"],
        }
    )
    structure = Schema.model_validate(payload)
    request = structure.requests["root/ts"]

    assert [datepart.value for datepart in request.dateparts] == [
        "day_of_week",
        "month_of_year",
        "hour_of_day",
    ]


def test_dateparts_unknown_name_suggests_canonical_value():
    payload = _structure_with_field(
        {
            "name": "ts",
            "type": "dateparts",
            "query": "[*].created_at",
            "dateparts": ["day of wek"],
        }
    )
    with pytest.raises(ValueError, match="did you mean 'day_of_week'"):
        Schema.model_validate(payload)


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
        Schema.model_validate(payload)


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
    structure = Schema.model_validate(payload)
    assert "root/ts" in structure.requests
