"""Tests for schema inference (`json2vec.helpers.infer_schema`)."""

import warnings

import pytest

import json2vec as jv
from json2vec.helpers import InferenceConfig, infer_schema


def _by_name(fields):
    return {field.name: field for field in fields}


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #


def test_inference_helpers_exported_from_helpers_namespace():
    assert jv.helpers.infer_schema is infer_schema
    assert jv.helpers.InferenceConfig is InferenceConfig
    assert not hasattr(jv, "infer_schema")
    assert not hasattr(jv, "create_model_from_records")
    assert not hasattr(jv, "InferenceConfig")
    assert not hasattr(jv.Model, "from_records")
    assert not hasattr(jv.helpers, "create_model_from_records")


def test_empty_input_raises():
    with pytest.raises(ValueError):
        infer_schema([])


# --------------------------------------------------------------------------- #
# Scalar typing
# --------------------------------------------------------------------------- #


def test_string_column_becomes_category():
    fields = _by_name(infer_schema([{"tier": "gold"}, {"tier": "silver"}, {"tier": "gold"}]))
    assert fields["tier"].type == "category"
    assert fields["tier"].query == "[*].tier"


def test_float_column_becomes_number():
    records = [{"amount": float(i) + 0.5} for i in range(100)]
    fields = _by_name(infer_schema(records))
    assert fields["amount"].type == "number"


def test_high_cardinality_int_becomes_number():
    records = [{"reading": i * 7} for i in range(100)]
    fields = _by_name(infer_schema(records))
    assert fields["reading"].type == "number"


def test_low_cardinality_int_becomes_category():
    records = [{"rating": i % 5} for i in range(100)]
    fields = _by_name(infer_schema(records))
    assert fields["rating"].type == "category"


def test_id_named_int_becomes_category():
    # Unique-per-row ints would normally be Number, but the name marks identity.
    records = [{"zip_code": 10000 + i} for i in range(100)]
    fields = _by_name(infer_schema(records))
    assert fields["zip_code"].type == "category"


def test_boolean_column_becomes_boolean():
    fields = _by_name(infer_schema([{"flag": True}, {"flag": False}, {"flag": True}]))
    assert fields["flag"].type == "boolean"


def test_iso_date_column_becomes_dateparts():
    records = [{"d": "2023-01-05"}, {"d": "2024-06-30"}, {"d": "2022-11-02"}]
    fields = _by_name(infer_schema(records))
    assert fields["d"].type == "dateparts"
    assert "hour_of_day" not in [str(p) for p in fields["d"].dateparts]


def test_native_date_objects_become_dateparts():
    import datetime

    records = [{"d": datetime.date(2023, 1, 5)}, {"d": datetime.date(2024, 6, 30)}]
    fields = _by_name(infer_schema(records))
    assert fields["d"].type == "dateparts"


def test_native_datetime_objects_with_time_add_hour_part():
    import datetime

    records = [
        {"ts": datetime.datetime(2023, 1, 5, 8, 30)},
        {"ts": datetime.datetime(2024, 6, 30, 17, 45)},
    ]
    fields = _by_name(infer_schema(records))
    assert fields["ts"].type == "dateparts"
    assert any("hour_of_day" in str(p) for p in fields["ts"].dateparts)


def test_numpy_scalar_types_are_recognized():
    np = pytest.importorskip("numpy")
    # numpy.int64 is NOT a Python int subclass; numpy.bool_ is not a Python bool.
    records = [{"i": np.int64(i % 4), "f": np.float64(i) + 0.5, "b": np.bool_(i % 2)} for i in range(100)]
    fields = _by_name(infer_schema(records))
    assert fields["i"].type == "category"  # low-cardinality integer
    assert fields["f"].type == "number"
    assert fields["b"].type == "boolean"


def test_iso_datetime_with_time_adds_hour_part():
    records = [{"ts": "2023-01-05T08:30:00"}, {"ts": "2024-06-30T17:45:00"}]
    fields = _by_name(infer_schema(records))
    assert fields["ts"].type == "dateparts"
    assert any("hour_of_day" in str(p) for p in fields["ts"].dateparts)


def test_vocab_size_scales_with_cardinality():
    records = [{"cat": f"c{i % 7}"} for i in range(100)]
    fields = _by_name(infer_schema(records))
    assert fields["cat"].type == "category"
    assert fields["cat"].size >= 7


# --------------------------------------------------------------------------- #
# List columns
# --------------------------------------------------------------------------- #


def test_string_list_becomes_set():
    records = [{"tags": ["a", "b"]}, {"tags": ["b"]}, {"tags": ["a", "c"]}]
    fields = _by_name(infer_schema(records))
    assert fields["tags"].type == "set"
    assert fields["tags"].query == "[*].tags"


def test_fixed_width_numeric_list_becomes_vector():
    records = [{"emb": [0.1, 0.2, 0.3]}, {"emb": [0.4, 0.5, 0.6]}]
    fields = _by_name(infer_schema(records))
    assert fields["emb"].type == "vector"
    assert fields["emb"].n_dim == 3


def test_variable_width_numeric_list_is_skipped_with_warning():
    records = [{"ok": 1.0, "vals": [1.0, 2.0]}, {"ok": 2.0, "vals": [1.0, 2.0, 3.0]}]
    with pytest.warns(UserWarning, match="vals"):
        fields = infer_schema(records)
    assert all(field.name != "vals" for field in fields)
    assert any(field.name == "ok" for field in fields)


def test_fixed_length_branch_is_not_padded_to_a_power_of_two():
    # Every record has exactly 5 items; length should be 5, not rounded to 8.
    records = [{"m": [{"v": float(j)} for j in range(5)]} for _ in range(20)]
    fields = _by_name(infer_schema(records))
    assert fields["m"].length == 5


def test_branch_length_uses_the_configured_quantile():
    # Lengths 1..10 uniformly; the p50 length is 5-6, well under the max of 10.
    records = [{"m": [{"v": 1.0}] * n} for n in range(1, 11)]
    fields = _by_name(infer_schema(records, array_length_quantile=0.5))
    assert fields["m"].length <= 6


def test_list_of_objects_becomes_branch_with_inferred_queries():
    records = [
        {"items": [{"sku": "A", "price": 1.0}, {"sku": "B", "price": 2.0}]},
        {"items": [{"sku": "C", "price": 3.0}]},
    ]
    fields = _by_name(infer_schema(records))
    branch = fields["items"]
    assert branch.type == "branch"
    assert branch.length >= 2
    children = _by_name(branch.fields)
    assert children["sku"].query == "[*].items[*].sku"
    assert children["price"].query == "[*].items[*].price"
    assert children["price"].type == "number"


# --------------------------------------------------------------------------- #
# Nested objects (flattened into dotted queries)
# --------------------------------------------------------------------------- #


def test_nested_dict_is_flattened_with_dotted_queries():
    records = [
        {"profile": {"country": "US", "score": 1.5}},
        {"profile": {"country": "CA", "score": 9.5}},
    ]
    fields = _by_name(infer_schema(records))
    assert "profile_country" in fields
    assert fields["profile_country"].query == "[*].profile.country"
    assert fields["profile_score"].query == "[*].profile.score"


def test_object_inside_branch_keeps_one_array_selector():
    records = [
        {"orders": [{"meta": {"id": 1}, "total": 9.99}]},
        {"orders": [{"meta": {"id": 2}, "total": 4.50}]},
    ]
    fields = _by_name(infer_schema(records))
    children = _by_name(fields["orders"].fields)
    assert children["meta_id"].query == "[*].orders[*].meta.id"
    # Exactly one array selector for one Branch ancestor.
    assert children["meta_id"].query.count("[*]") == 2  # outer observation + orders


# --------------------------------------------------------------------------- #
# Config / overrides / targets
# --------------------------------------------------------------------------- #


def test_category_threshold_override_switches_to_number():
    records = [{"n": i % 5} for i in range(100)]
    fields = _by_name(infer_schema(records, category_max_cardinality=2))
    assert fields["n"].type == "number"


def test_ordinal_integer_stays_number_despite_low_distinct_over_many_rows():
    # 24 distinct values over 5000 rows: a distinct/rows ratio test would call
    # this categorical, but an ordinal integer should stay a Number.
    records = [{"finishing_position": i % 24} for i in range(5000)]
    fields = _by_name(infer_schema(records))
    assert fields["finishing_position"].type == "number"


def test_id_named_integer_is_category_regardless_of_cardinality():
    records = [{"raceId": i} for i in range(5000)]
    fields = _by_name(infer_schema(records))
    assert fields["raceId"].type == "category"


def test_target_marks_field_as_supervised():
    records = [{"x": 1.0, "label": "yes"}, {"x": 2.0, "label": "no"}]
    fields = _by_name(infer_schema(records, target="label"))
    assert fields["label"].target is True
    assert fields["x"].target is False


def test_config_and_overrides_are_mutually_exclusive():
    with pytest.raises(TypeError):
        infer_schema([{"x": 1.0}], config=InferenceConfig(), category_max_cardinality=3)


def test_inconsistent_column_is_skipped_with_warning():
    records = [{"ok": "a", "x": 1}, {"ok": "b", "x": [1, 2]}, {"ok": "a", "x": "text"}]
    with pytest.warns(UserWarning, match="x"):
        fields = infer_schema(records)
    assert all(field.name != "x" for field in fields)
    assert any(field.name == "ok" for field in fields)


def test_explain_prints_table(capsys):
    infer_schema([{"tier": "gold"}, {"tier": "silver"}], explain=True)
    out = capsys.readouterr().out
    assert "tier" in out


# --------------------------------------------------------------------------- #
# Frame input
# --------------------------------------------------------------------------- #


def test_polars_dataframe_input():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"tier": ["gold", "silver", "gold"], "amount": [1.0, 2.0, 3.0]})
    fields = _by_name(infer_schema(frame))
    assert fields["tier"].type == "category"
    assert fields["amount"].type == "number"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_inferred_schema_builds_a_working_model():
    records = [
        {
            "tier": "gold",
            "amount": 12.5,
            "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 1}],
            "returned": "false",
        },
        {
            "tier": "silver",
            "amount": 8.0,
            "items": [{"sku": "C", "qty": 5}],
            "returned": "true",
        },
        {
            "tier": "gold",
            "amount": 40.0,
            "items": [{"sku": "A", "qty": 1}],
            "returned": "false",
        },
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = jv.helpers.infer_schema(records, target="returned")
        model = jv.Model(
            *fields,
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    requests = model.schema.requests

    # The `returned` field is a supervised target.
    returned = requests[jv.Address("record", "returned")]
    assert returned.target is True

    # Invariant: a leaf's [*] selectors equal its Branch-ancestor count + 1 (root).
    for address, request in requests.items():
        ancestors = len(model.schema.shapes[address])
        assert request.query.count("[*]") == ancestors

    # The model predicts the withheld target end to end.
    predictions = model.predict([{k: v for k, v in records[0].items() if k != "returned"}])
    decoded = predictions[jv.Address("record", "returned")]
    assert decoded["content"]["value"] is not None
