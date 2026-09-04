from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pytest

import relflow as rf
from relflow.data.datasets.arrow import convert
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, Tokens
from tests.arrow import batch as arrow_batch
from tests.arrow import table


def build(*fields):
    return rf.Model(*fields, d_model=8, n_layers=1, n_heads=2)


def request(field_type: str, *, query: str | None = None):
    if field_type == "hash":
        return rf.Hash("identity", query=query, n_hashes=2)
    if field_type == "category":
        return rf.Category("identity", query=query, size=8, p_unavailable=0.0)
    if field_type == "cluster":
        return rf.Cluster("identity", query=query, capacity=8, n_clusters=2, p_unavailable=0.0)
    if field_type == "set":
        return rf.Set("identity", query=query, size=8, p_unavailable=0.0)
    raise AssertionError(f"unsupported test field type: {field_type}")


def test_coalesce_requires_arrow_batch():
    model = build(rf.Number("value"))

    with pytest.raises(TypeError, match="must be an Arrow Batch"):
        coalesce(pa.table({"value": [1.0]}), schema=model.schema, strata=Strata.predict)


def test_ragged_field_distinguishes_value_null_and_missing():
    model = build(rf.Number("value"))
    source = arrow_batch(
        [{"value": 1.5}, {"value": None}, {}],
        schema=pa.schema([pa.field("value", pa.float64())]),
    )

    field = coalesce(source, schema=model.schema, strata=Strata.predict)["record/value"].pristine

    assert field.shape == (3, 1)
    assert field.batch_size == 3
    assert field.state.type == pa.int8()
    assert field.dense.tolist() == [
        [Tokens.valued.value],
        [Tokens.null.value],
        [Tokens.null.value],
    ]
    assert field.values.to_pylist() == [1.5]
    assert field.placement.to_pylist() == [0]
    assert field.place(np.asarray([9.0]), fill=-1.0).tolist() == [[9.0], [-1.0], [-1.0]]


def test_sibling_fields_share_branch_geometry_without_sharing_leaf_state():
    model = build(
        rf.Branch(
            rf.Number("left"),
            rf.Number("right"),
            name="items",
            length=3,
        )
    )
    fields = coalesce(
        arrow_batch(
            [
                {
                    "items": [
                        {"left": 1.0, "right": None},
                        None,
                        {"left": None, "right": 2.0},
                    ]
                }
            ]
        ),
        schema=model.schema,
        strata=Strata.train,
    )

    left = fields["record/items/left"].pristine
    right = fields["record/items/right"].pristine
    assert left.dense.tolist() == [[[Tokens.valued.value, Tokens.padded.value, Tokens.null.value]]]
    assert right.dense.tolist() == [[[Tokens.null.value, Tokens.padded.value, Tokens.valued.value]]]
    assert left.values.to_pylist() == [1.0]
    assert right.values.to_pylist() == [2.0]
    assert left.placement.to_pylist() == [0]
    assert right.placement.to_pylist() == [2]


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            pa.UnionArray.from_dense(
                pa.array([0, 1, 0], type=pa.int8()),
                pa.array([0, 0, 1], type=pa.int32()),
                [pa.array([1, None], type=pa.int64()), pa.array([2], type=pa.int32())],
            ),
            id="dense",
        ),
        pytest.param(
            pa.UnionArray.from_sparse(
                pa.array([0, 1, 0], type=pa.int8()),
                [
                    pa.array([1, None, None], type=pa.int64()),
                    pa.array([None, 2, None], type=pa.int32()),
                ],
            ),
            id="sparse",
        ),
        pytest.param(
            pa.UnionArray.from_dense(
                pa.array([0, 0, 1, 0], type=pa.int8()),
                pa.array([0, 1, 0, 2], type=pa.int32()),
                [pa.array([99, 1, None], type=pa.int64()), pa.array([2], type=pa.int32())],
            ).slice(1),
            id="sliced-dense",
        ),
        pytest.param(
            pa.UnionArray.from_sparse(
                pa.array([1, 0, 1, 0], type=pa.int8()),
                [
                    pa.array([None, 1, None, None], type=pa.int64()),
                    pa.array([99, None, 2, None], type=pa.int32()),
                ],
            ).slice(1),
            id="sliced-sparse",
        ),
    ],
)
def test_union_leaf_uses_selected_child_validity(values):
    model = build(rf.Number("value"))
    source = convert(pa.table({"value": values}), namespace="union", offset=0)

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/value"].pristine

    assert field.dense.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
        [Tokens.null.value],
    ]
    assert field.values.to_pylist() == [1, 2]
    assert field.placement.to_pylist() == [0, 1]

    encoded = model.encode(source.data, strata=Strata.predict)["record/value"]
    assert encoded.content.tolist() == [[1.0], [2.0], [0.0]]
    assert encoded.state.tolist() == field.dense.tolist()


def test_coalesce_rejects_modeled_field_missing_from_arrow_schema():
    model = build(rf.Number("value"))

    with pytest.raises(ValueError, match="field 'value' is absent"):
        coalesce(
            arrow_batch([{"metadata": 1}, {"metadata": 2}]),
            schema=model.schema,
            strata=Strata.train,
        )


def test_mask_spelling_is_ordinary_string_content():
    model = build(rf.Category("label", size=8, p_unavailable=0.0))
    field = model.encode(table([{"label": "<MASK>"}]), strata=Strata.train)["record/label"]

    assert field.state.tolist() == [[Tokens.valued.value]]
    assert rf.Category.vocabulary(model, "record/label") == ("<MASK>",)


def test_structured_leaf_mask_spelling_is_ordinary_codec_input():
    model = build(rf.Set("labels", size=8, p_unavailable=0.0))
    field = coalesce(
        arrow_batch([{"labels": ["<MASK>", "A"]}]),
        schema=model.schema,
        strata=Strata.predict,
    )["record/labels"].pristine

    assert field.dense.tolist() == [[Tokens.valued.value]]
    assert field.values.to_pylist() == [["<MASK>", "A"]]


def test_tail_overflow_finishes_before_leaf_codec_observes_values():
    model = build(
        rf.Branch(
            rf.Vector("value", n_dim=2),
            name="items",
            length=2,
            overflow="tail",
        )
    )
    field = coalesce(
        arrow_batch([{"items": [{"value": [0, 1, 2]}, {"value": [2, 3]}, {"value": [4, 5]}]}]),
        schema=model.schema,
        strata=Strata.train,
    )["record/items/value"].pristine

    assert field.dense.tolist() == [[[Tokens.valued.value, Tokens.valued.value]]]
    assert field.values.to_pylist() == [[2, 3], [4, 5]]
    assert field.placement.to_pylist() == [0, 1]


def test_branch_overflow_precedes_queries_on_discarded_children():
    model = build(
        rf.Branch(
            rf.Number("value", query='attributes["x"]'),
            name="items",
            length=1,
            overflow="head",
        )
    )
    item = pa.struct([pa.field("attributes", pa.map_(pa.string(), pa.float64()))])
    source = pa.table(
        {
            "items": pa.array(
                [
                    [
                        {"attributes": [("x", 1.0)]},
                        {"attributes": [("x", 2.0), ("x", 3.0)]},
                    ]
                ],
                type=pa.list_(item),
            )
        }
    )

    field = coalesce(
        convert(source, namespace="overflow", offset=0),
        schema=model.schema,
        strata=Strata.train,
    )["record/items/value"].pristine

    assert field.values.to_pylist() == [1.0]


def test_error_overflow_names_address_and_axis():
    model = build(
        rf.Branch(
            rf.Number("value"),
            name="items",
            length=1,
            overflow="error",
        )
    )
    with pytest.raises(ValueError, match="branch overflow at dimension 2 for record/items/value"):
        coalesce(
            arrow_batch([{"items": [{"value": 1}, {"value": 2}]}]),
            schema=model.schema,
            strata=Strata.train,
        )


def test_all_empty_deep_branches_materialize_declared_geometry():
    model = build(
        rf.Branch(
            rf.Branch(
                rf.Branch(rf.Number("value"), name="deep", length=2),
                name="inner",
                length=2,
            ),
            name="outer",
            length=2,
        )
    )
    deep = pa.struct([pa.field("value", pa.float64())])
    inner = pa.struct([pa.field("deep", pa.list_(deep))])
    outer = pa.struct([pa.field("inner", pa.list_(inner))])
    field = coalesce(
        arrow_batch(
            [{"outer": []}, {"outer": None}],
            schema=pa.schema([pa.field("outer", pa.list_(outer))]),
        ),
        schema=model.schema,
        strata=Strata.train,
    )["record/outer/inner/deep/value"].pristine

    assert field.shape == (2, 1, 2, 2, 2)
    assert np.all(field.dense == Tokens.padded.value)
    assert field.values.to_pylist() == []
    assert field.placement.to_pylist() == []


def test_typed_empty_nested_batch_preserves_declared_geometry():
    model = build(rf.Branch(rf.Number("value"), name="items", length=3))
    item = pa.struct([pa.field("value", pa.float64())])
    field = coalesce(
        arrow_batch([], schema=pa.schema([pa.field("items", pa.list_(item))])),
        schema=model.schema,
        strata=Strata.train,
    )["record/items/value"].pristine

    assert field.shape == (0, 1, 3)
    assert field.dense.shape == field.shape
    assert field.values.to_pylist() == []
    assert field.placement.to_pylist() == []


def test_singleton_branch_accepts_one_item_list():
    model = build(rf.Branch(rf.Number("value"), name="details", length=1))
    field = coalesce(
        arrow_batch([{"details": [{"value": 4}]}]),
        schema=model.schema,
        strata=Strata.train,
    )["record/details/value"].pristine

    assert field.dense.tolist() == [[[Tokens.valued.value]]]
    assert field.values.to_pylist() == [4]


def test_singleton_branch_lists_nest_inside_repeated_branch():
    model = build(
        rf.Branch(
            rf.Branch(rf.Number("value"), name="details", length=1),
            name="items",
            length=3,
        )
    )
    source = arrow_batch(
        [
            {
                "items": [
                    {"details": [{"value": 4}]},
                    {"details": [{"value": 5}]},
                ]
            }
        ]
    )
    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/items/details/value"].pristine

    assert field.dense.tolist() == [
        [
            [
                [Tokens.valued.value],
                [Tokens.valued.value],
                [Tokens.padded.value],
            ]
        ]
    ]
    assert field.values.to_pylist() == [4, 5]
    assert field.placement.to_pylist() == [0, 1]


def test_branch_rejects_struct_where_list_axis_is_required():
    model = build(rf.Branch(rf.Number("value"), name="items", length=2))

    with pytest.raises(ValueError, match="expected a list"):
        coalesce(
            arrow_batch([{"items": {"value": 4}}]),
            schema=model.schema,
            strata=Strata.train,
        )


def test_coalesce_ignores_unmodeled_arrow_columns():
    model = build(rf.Hash("identifier"))
    source = arrow_batch([{"identifier": "A", "metadata": {"tags": [1, 2]}}])

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/identifier"].pristine

    assert source.data["metadata"].to_pylist() == [{"tags": [1, 2]}]
    assert field.values.to_pylist() == ["A"]


def test_coalesce_does_not_ingest_inactive_field_values():
    model = build(rf.Number("value"), rf.Hash("unused", active=False))
    source = arrow_batch([{"value": 1.0, "unused": {"opaque": True}}])

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/value"].pristine

    assert source.data["unused"].to_pylist() == [{"opaque": True}]
    assert field.values.to_pylist() == [1.0]


def test_predict_reconstruction_values_are_prepared_when_present():
    model = build(rf.Number("value"), rf.Hash("label", mask=True))
    source = table([{"value": 1.0, "label": {"not": "hashable"}}])

    with pytest.raises(TypeError, match="extension 'hash'.*does not accept Arrow type struct"):
        model.encode(source, strata=Strata.predict)


def test_query_only_branch_ignores_same_named_direct_source_value():
    model = build(
        rf.Branch(
            rf.Number("value"),
            name="synthetic",
            query="payload.values",
            length=2,
        )
    )

    encoded = model.encode(
        table(
            [
                {
                    "payload": {"values": [{"value": 1.0}, {"value": 2.0}]},
                    "synthetic": 99,
                }
            ]
        ),
        strata=Strata.predict,
    )

    field = encoded[rf.Address("record/synthetic/value")]
    assert field.state.tolist() == [[[Tokens.valued.value, Tokens.valued.value]]]
    assert field.content.tolist() == [[[1.0, 2.0]]]


def test_inactive_only_branch_does_not_ingest_same_named_source_value():
    model = build(
        rf.Number("value"),
        rf.Branch(rf.Hash("unused", active=False), name="synthetic", length=2),
    )
    source = arrow_batch([{"value": 1.0, "synthetic": {"opaque": True}}])

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/value"].pristine

    assert source.data["synthetic"].to_pylist() == [{"opaque": True}]
    assert field.values.to_pylist() == [1.0]


def test_datetime_leaf_round_trips_through_awkward():
    model = build(rf.DateParts("created", dateparts=["day_of_year"]))
    value = datetime.datetime(2025, 2, 3, 4, 5, 6)
    field = coalesce(
        arrow_batch([{"created": value}]),
        schema=model.schema,
        strata=Strata.train,
    )["record/created"].pristine

    assert field.values.to_pylist() == [value]


def test_dateparts_tensorfield_encodes_arrow_timestamp_end_to_end():
    model = build(rf.DateParts("created", dateparts=["day_of_year", "hour_of_day"]))
    encoded = model.encode(
        table([{"created": datetime.datetime(2025, 2, 3, 4, 5, 6)}]),
        strata=Strata.train,
    )[rf.Address("record/created")]

    assert encoded.state.tolist() == [[Tokens.valued.value]]
    assert encoded.content["day_of_year"].shape == (1, 1, 2)
    assert encoded.content["hour_of_day"].shape == (1, 1, 2)
    assert encoded.content.isfinite().all()


@pytest.mark.parametrize(
    ("value", "expected_vocabulary"),
    [
        pytest.param(["A", "B"], ("A", "B"), id="list"),
        pytest.param("AB", ("AB",), id="scalar-string"),
        pytest.param(b"AB", (b"AB",), id="scalar-bytes"),
    ],
)
def test_set_accepts_arrow_lists_and_scalar_labels(value, expected_vocabulary):
    model = build(rf.Set("labels", size=8, p_unavailable=0.0))
    field = model.encode(
        table([{"labels": value}]),
        strata=Strata.train,
    )[rf.Address("record/labels")]

    assert field.state.tolist() == [[Tokens.valued.value]]
    assert rf.Set.vocabulary(model, "record/labels") == expected_vocabulary
    assert field.content.sum(dim=-1).tolist() == [[float(len(expected_vocabulary))]]


@pytest.mark.parametrize("query", [None, "source"], ids=["direct", "query"])
def test_set_treats_scalar_bytes_as_one_label(query):
    model = build(request("set", query=query))
    key = "source" if query is not None else "identity"
    field = model.encode(
        table([{key: b"AB"}, {key: b"AB"}]),
        strata=Strata.train,
    )[rf.Address("record/identity")]

    assert rf.Set.vocabulary(model, "record/identity") == (b"AB",)
    assert field.content.sum(dim=-1).tolist() == [[1.0], [1.0]]
    assert field.content[0].tolist() == field.content[1].tolist()


def test_place_validates_encoded_count_and_value_shape():
    model = build(rf.Number("value"))
    field = coalesce(
        arrow_batch([{"value": 1}, {"value": 2}]),
        schema=model.schema,
        strata=Strata.train,
    )["record/value"].pristine

    with pytest.raises(ValueError, match=r"must have shape \(2,\), got \(1,\)"):
        field.place(np.asarray([1]), fill=0)

    encoded = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    placed = field.place(encoded, fill=0.0, value_shape=(2,))
    assert placed.shape == (2, 1, 2)
    assert placed.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]]]
