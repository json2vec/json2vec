import datetime

import awkward as ak
import numpy as np
import pytest

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, Tokens


def _model(*fields):
    return rf.Model(*fields, d_model=8, n_layers=1, n_heads=2)


def _identity_request(field_type: str, *, query: str | None = None):
    if field_type == "hash":
        return rf.Hash("identity", query=query, n_hashes=2)
    if field_type == "category":
        return rf.Category("identity", query=query, size=8, p_unavailable=0.0)
    if field_type == "cluster":
        return rf.Cluster("identity", query=query, capacity=8, n_clusters=2, p_unavailable=0.0)
    if field_type == "set":
        return rf.Set("identity", query=query, size=8, p_unavailable=0.0)
    raise AssertionError(f"unsupported test field type: {field_type}")


def test_ragged_field_distinguishes_value_null_missing_and_prediction_literal():
    model = _model(rf.Number("value"))
    batch = [
        [{"value": 1.5}],
        [{"value": None}],
        [{}],
        [{"value": "<MASK>"}],
    ]

    field = coalesce(batch, schema=model.schema, strata=Strata.predict)["record/value"]

    assert field.shape == (4, 1)
    assert field.batch_size == 4
    assert field.state.dtype == np.int64
    assert field.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.null.value],
        [Tokens.padded.value],
        [Tokens.masked.value],
    ]
    assert ak.to_list(field.values) == [1.5]
    assert field.placement.tolist() == [0]
    assert field.place(np.asarray([9.0]), fill=-1.0).tolist() == [[9.0], [-1.0], [-1.0], [-1.0]]


def test_ragged_field_materializes_field_missing_from_every_record():
    model = _model(rf.Number("value"))
    field = coalesce(
        [[{"metadata": 1}], [{"metadata": 2}]],
        schema=model.schema,
        strata=Strata.train,
    )["record/value"]

    assert field.state.tolist() == [[Tokens.padded.value], [Tokens.padded.value]]
    assert ak.to_list(field.values) == []


def test_ragged_field_rejects_prediction_literal_outside_predict():
    model = _model(rf.Number("value"))
    with pytest.raises(ValueError, match="only valid during predict"):
        coalesce([[{"value": "<MASK>"}]], schema=model.schema, strata=Strata.train)


def test_structured_leaf_mask_string_is_ordinary_codec_input():
    model = _model(rf.Set("labels", size=8))
    field = coalesce(
        [[{"labels": ["<MASK>", "A"]}]],
        schema=model.schema,
        strata=Strata.predict,
    )["record/labels"]

    assert field.state.tolist() == [[Tokens.valued.value]]
    assert ak.to_list(field.values) == [["<MASK>", "A"]]


def test_tail_overflow_finishes_before_leaf_codec_observes_values():
    model = _model(
        rf.Branch(
            rf.Vector("value", n_dim=2),
            name="items",
            length=2,
            overflow="tail",
        )
    )
    field = coalesce(
        [[{"items": [{"value": [0, 1, 2]}, {"value": [2, 3]}, {"value": [4, 5]}]}]],
        schema=model.schema,
        strata=Strata.train,
    )["record/items/value"]

    assert field.state.tolist() == [[[Tokens.valued.value, Tokens.valued.value]]]
    assert ak.to_list(field.values) == [[2, 3], [4, 5]]
    assert field.placement.tolist() == [0, 1]


def test_error_overflow_names_address_and_axis():
    model = _model(
        rf.Branch(
            rf.Number("value"),
            name="items",
            length=1,
            overflow="error",
        )
    )
    with pytest.raises(ValueError, match="branch overflow at dimension 2 for record/items/value"):
        coalesce(
            [[{"items": [{"value": 1}, {"value": 2}]}]],
            schema=model.schema,
            strata=Strata.train,
        )


def test_all_empty_deep_branches_materialize_declared_geometry():
    model = _model(
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
    field = coalesce(
        [[{"outer": []}], [{}]],
        schema=model.schema,
        strata=Strata.train,
    )["record/outer/inner/deep/value"]

    assert field.shape == (2, 1, 2, 2, 2)
    assert np.all(field.state == Tokens.padded.value)
    assert ak.to_list(field.values) == []
    assert field.placement.tolist() == []


def test_singleton_branch_accepts_mapping_shorthand():
    model = _model(rf.Branch(rf.Number("value"), name="details", length=1))
    field = coalesce(
        [[{"details": {"value": 4}}]],
        schema=model.schema,
        strata=Strata.train,
    )["record/details/value"]

    assert field.state.tolist() == [[[Tokens.valued.value]]]
    assert ak.to_list(field.values) == [4]


def test_singleton_branch_mapping_broadcasts_inside_repeated_branch():
    model = _model(
        rf.Branch(
            rf.Branch(rf.Number("value"), name="details", length=1),
            name="items",
            length=3,
        )
    )
    batch = [[{"items": [{"details": {"value": 4}}, {"details": {"value": 5}}]}]]
    field = coalesce(batch, schema=model.schema, strata=Strata.train)["record/items/details/value"]

    assert field.state.tolist() == [
        [
            [
                [Tokens.valued.value],
                [Tokens.valued.value],
                [Tokens.padded.value],
            ]
        ]
    ]
    assert ak.to_list(field.values) == [4, 5]
    assert field.placement.tolist() == [0, 1]


def test_repeated_branch_rejects_mapping_shorthand():
    model = _model(rf.Branch(rf.Number("value"), name="items", length=2))

    with pytest.raises(TypeError, match="mapping shorthand is only valid for length=1"):
        coalesce([[{"items": {"value": 4}}]], schema=model.schema, strata=Strata.train)


def test_coalesce_ignores_unmodeled_opaque_metadata():
    model = _model(rf.Hash("identifier"))
    metadata = object()
    source = [[{"identifier": "A", "metadata": metadata}]]

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/identifier"]

    assert source[0][0]["metadata"] is metadata
    assert ak.to_list(field.values) == ["A"]


def test_coalesce_does_not_ingest_inactive_field_values():
    inactive = object()
    model = _model(rf.Number("value"), rf.Hash("unused", active=False))
    source = [[{"value": 1.0, "unused": inactive}]]

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/value"]

    assert source[0][0]["unused"] is inactive
    assert ak.to_list(field.values) == [1.0]


def test_predict_target_values_are_not_coalesced_or_consumed():
    model = _model(rf.Number("value"), rf.Hash("label", target=True))
    iterator = (item for item in (1, 2))

    encoded = model.encode(
        [{"value": 1.0, "label": iterator}],
        strata=Strata.predict,
        mask=False,
    )

    assert list(iterator) == [1, 2]
    assert encoded[rf.Address("record/label")].state.tolist() == [[Tokens.masked.value]]


def test_query_only_branch_ignores_same_named_direct_source_value():
    model = _model(
        rf.Branch(
            rf.Number("value", query="[*].payload.values"),
            name="synthetic",
            length=2,
        )
    )

    encoded = model.encode(
        [{"payload": {"values": [1.0, 2.0]}, "synthetic": object()}],
        strata=Strata.predict,
        mask=False,
    )

    field = encoded[rf.Address("record/synthetic/value")]
    assert field.state.tolist() == [[[Tokens.valued.value, Tokens.valued.value]]]
    assert field.content.tolist() == [[[1.0, 2.0]]]


def test_inactive_only_branch_does_not_ingest_same_named_source_value():
    opaque = object()
    model = _model(
        rf.Number("value"),
        rf.Branch(rf.Hash("unused", active=False), name="synthetic", length=2),
    )
    source = [[{"value": 1.0, "synthetic": opaque}]]

    field = coalesce(source, schema=model.schema, strata=Strata.train)["record/value"]

    assert source[0][0]["synthetic"] is opaque
    assert ak.to_list(field.values) == [1.0]


def test_coalesce_rejects_unsupported_modeled_values():
    model = _model(rf.Hash("identifier"))

    with pytest.raises(TypeError, match="record/identifier.*normalize it in a preprocessor"):
        coalesce([[{"identifier": object()}]], schema=model.schema, strata=Strata.train)

    with pytest.raises(TypeError, match="record/identifier.*unsupported int"):
        coalesce([[{"identifier": 2**64}]], schema=model.schema, strata=Strata.train)


def test_datetime_leaf_round_trips_through_awkward():
    model = _model(rf.DateParts("created", dateparts=["day_of_year"]))
    value = datetime.datetime(2025, 2, 3, 4, 5, 6)
    field = coalesce([[{"created": value}]], schema=model.schema, strata=Strata.train)["record/created"]

    assert ak.to_list(field.values) == [value]


def test_numpy_datetime_scalar_is_canonicalized_without_losing_nanosecond_scale():
    model = _model(rf.DateParts("created", dateparts=["day_of_year", "hour_of_day"]))
    encoded = model.encode(
        [{"created": np.datetime64("2025-02-03T04:05:06.123456789", "ns")}],
        strata=Strata.train,
        mask=False,
    )[rf.Address("record/created")]

    assert encoded.state.tolist() == [[Tokens.valued.value]]
    assert encoded.content.isfinite().all()


def test_dateparts_tensorfield_encodes_ragged_datetime_end_to_end():
    model = _model(rf.DateParts("created", dateparts=["day_of_year", "hour_of_day"]))
    encoded = model.encode(
        [{"created": datetime.datetime(2025, 2, 3, 4, 5, 6)}],
        strata=Strata.train,
        mask=False,
    )[rf.Address("record/created")]

    assert encoded.state.tolist() == [[Tokens.valued.value]]
    assert encoded.content["day_of_year"].shape == (1, 1, 2)
    assert encoded.content["hour_of_day"].shape == (1, 1, 2)
    assert encoded.content.isfinite().all()


@pytest.mark.parametrize(
    "value_factory",
    [
        pytest.param(lambda: {"A", "B"}, id="set"),
        pytest.param(lambda: range(2), id="range"),
        pytest.param(lambda: {"A": True}, id="mapping"),
    ],
)
def test_set_rejects_unregistered_container_types(value_factory):
    model = _model(rf.Set("labels", size=8, p_unavailable=0.0))

    with pytest.raises(
        TypeError,
        match=r"record/labels.*unsupported .*normalize it in a preprocessor",
    ):
        coalesce([[{"labels": value_factory()}]], schema=model.schema, strata=Strata.train)


@pytest.mark.parametrize(
    "value",
    [{"A", "B"}, range(2), {"A": True}],
    ids=["set", "range", "mapping"],
)
def test_queried_set_rejects_unregistered_container_types(value):
    model = _model(rf.Set("labels", query="[*].source", size=8, p_unavailable=0.0))

    with pytest.raises(TypeError, match=r"record/labels.*unsupported .*normalize it in a preprocessor"):
        model.encode([{"source": value}], strata=Strata.predict, mask=False)


def test_coalesce_preserves_unmodeled_one_shot_iterator_without_consuming_it():
    model = _model(rf.Number("value"))
    iterator = (item for item in (1, 2))
    source = [[{"value": 1, "metadata": iterator}]]

    coalesce(source, schema=model.schema, strata=Strata.train)

    assert source[0][0]["metadata"] is iterator
    assert list(iterator) == [1, 2]


def test_coalesce_rejects_directly_modeled_iterator_without_consuming_it():
    model = _model(rf.Number("value"))
    iterator = (item for item in (1, 2))

    with pytest.raises(TypeError, match=r"one-shot iterator.*materialize it as a list"):
        coalesce([[{"value": iterator}]], schema=model.schema, strata=Strata.train)

    assert list(iterator) == [1, 2]


def test_coalesce_rejects_iterator_nested_in_sequence_subclass_without_consuming_it():
    class Labels(list):
        pass

    model = _model(rf.Set("labels", size=8))
    iterator = (item for item in (1, 2))

    with pytest.raises(TypeError, match=r"one-shot iterator.*materialize it as a list"):
        coalesce(
            [[{"labels": Labels(["A", iterator])}]],
            schema=model.schema,
            strata=Strata.train,
        )

    assert list(iterator) == [1, 2]


def test_explicit_query_rejects_returned_iterator_without_consuming_it():
    model = _model(rf.Number("value", query="[*].metadata"))
    iterator = (item for item in (1, 2))

    with pytest.raises(TypeError, match=r"one-shot iterator.*materialize it as a list"):
        coalesce([[{"metadata": iterator}]], schema=model.schema, strata=Strata.predict)

    assert list(iterator) == [1, 2]


@pytest.mark.parametrize(
    ("value", "expected_vocabulary"),
    [
        pytest.param(["A", "B"], ("A", "B"), id="list"),
        pytest.param(("A", "B"), ("A", "B"), id="tuple"),
        pytest.param(np.asarray(["A", "B"]), ("A", "B"), id="numpy-array"),
        pytest.param("AB", ("AB",), id="scalar-str"),
        pytest.param(b"AB", (b"AB",), id="scalar-bytes"),
    ],
)
def test_set_accepts_canonical_containers_and_scalar_text(value, expected_vocabulary):
    model = _model(rf.Set("labels", size=8, p_unavailable=0.0))
    field = model.encode(
        [{"labels": value}],
        strata=Strata.train,
        mask=False,
    )[rf.Address("record/labels")]

    assert field.state.tolist() == [[Tokens.valued.value]]
    assert rf.Set.vocabulary(model, "record/labels") == expected_vocabulary
    assert field.content.sum(dim=-1).tolist() == [[float(len(expected_vocabulary))]]


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
@pytest.mark.parametrize(
    ("field_type", "values"),
    [
        pytest.param("hash", (1, 1.0), id="hash-int-float"),
        pytest.param("category", ("A", b"A"), id="category-str-bytes"),
        pytest.param("cluster", ("A", b"A"), id="cluster-str-bytes"),
        pytest.param("set", ("A", b"A"), id="set-str-bytes"),
    ],
)
def test_identity_fields_reject_mixed_exact_python_types(field_type, values, query):
    model = _model(_identity_request(field_type, query=query))
    key = "source" if query is not None else "identity"

    with pytest.raises(TypeError, match="record/identity"):
        model.encode([{key: value} for value in values], strata=Strata.predict, mask=False)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_string_mask_literal_is_captured_before_awkward_text_coercion(query):
    model = _model(rf.Category("identity", query=query, size=8, p_unavailable=0.0))
    key = "source" if query is not None else "identity"
    batch = [[{key: b"x"}], [{key: "<MASK>"}]]
    field = coalesce(batch, schema=model.schema, strata=Strata.predict)["record/identity"]

    assert field.state.tolist() == [[Tokens.valued.value], [Tokens.masked.value]]
    assert ak.to_list(field.values) == [b"x"]


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_bytes_spelling_of_mask_literal_is_an_ordinary_identity_value(query):
    model = _model(rf.Category("identity", query=query, size=8, p_unavailable=0.0))
    key = "source" if query is not None else "identity"
    field = coalesce(
        [[{key: b"<MASK>"}]],
        schema=model.schema,
        strata=Strata.predict,
    )["record/identity"]

    assert field.state.tolist() == [[Tokens.valued.value]]
    assert ak.to_list(field.values) == [b"<MASK>"]

    with pytest.raises(TypeError, match="record/identity"):
        coalesce(
            [[{key: "ordinary-label"}], [{key: b"<MASK>"}]],
            schema=model.schema,
            strata=Strata.predict,
        )


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
@pytest.mark.parametrize("field_type", ["category", "set"])
def test_mixed_str_and_bytes_cannot_turn_bytes_into_mask_literal(field_type, query):
    model = _model(_identity_request(field_type, query=query))
    key = "source" if query is not None else "identity"

    with pytest.raises(TypeError, match="record/identity"):
        model.encode(
            [{key: "ordinary-label"}, {key: b"<MASK>"}],
            strata=Strata.predict,
            mask=False,
        )


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
@pytest.mark.parametrize(
    ("field_type", "numpy_value", "python_value"),
    [
        pytest.param("hash", np.int64(7), 7, id="hash-int"),
        pytest.param("category", np.str_("A"), "A", id="category-str"),
        pytest.param("set", np.str_("A"), "A", id="set-str"),
    ],
)
def test_numpy_scalar_and_equivalent_python_scalar_share_one_identity(
    field_type,
    numpy_value,
    python_value,
    query,
):
    model = _model(_identity_request(field_type, query=query))
    key = "source" if query is not None else "identity"
    field = model.encode(
        [{key: numpy_value}, {key: python_value}],
        strata=Strata.train,
        mask=False,
    )[rf.Address("record/identity")]

    assert field.state.tolist() == [[Tokens.valued.value], [Tokens.valued.value]]
    assert field.content[0].tolist() == field.content[1].tolist()
    if field_type == "category":
        assert rf.Category.vocabulary(model, "record/identity") == (python_value,)
    elif field_type == "set":
        assert rf.Set.vocabulary(model, "record/identity") == (python_value,)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_set_treats_scalar_bytes_as_one_label(query):
    model = _model(_identity_request("set", query=query))
    key = "source" if query is not None else "identity"
    field = model.encode(
        [{key: b"AB"}, {key: b"AB"}],
        strata=Strata.train,
        mask=False,
    )[rf.Address("record/identity")]

    assert rf.Set.vocabulary(model, "record/identity") == (b"AB",)
    assert field.content.sum(dim=-1).tolist() == [[1.0], [1.0]]
    assert field.content[0].tolist() == field.content[1].tolist()


def test_place_validates_encoded_count_and_value_shape():
    model = _model(rf.Number("value"))
    field = coalesce(
        [[{"value": 1}], [{"value": 2}]],
        schema=model.schema,
        strata=Strata.train,
    )["record/value"]

    with pytest.raises(ValueError, match=r"must have shape \(2,\), got \(1,\)"):
        field.place(np.asarray([1]), fill=0)

    encoded = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    placed = field.place(encoded, fill=0.0, value_shape=(2,))
    assert placed.shape == (2, 1, 2)
    assert placed.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]]]


@pytest.mark.parametrize("roots", [[], [{}, {}]])
def test_coalesce_requires_singleton_generated_root(roots):
    model = _model(rf.Number("value"))

    with pytest.raises(ValueError, match="exactly one generated-root record"):
        coalesce([roots], schema=model.schema, strata=Strata.train)
