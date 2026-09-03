from __future__ import annotations

import pyarrow as pa
import pytest

import relflow as rf
from relflow.data.arrow import IDENTITY, Batch
from relflow.data.query import Index, Literal, Member, Slice, Traverse, bind, compile, query
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, Tokens


def identities(size: int) -> pa.Array:
    return pa.array(
        [
            {
                "logical": index.to_bytes(32),
                "instance": index.to_bytes(32),
                "order": index.to_bytes(8),
            }
            for index in range(size)
        ],
        type=IDENTITY,
    )


def test_compile_structural_path():
    parsed = compile('customers[*].profile["display name"]')

    assert parsed.steps == (
        Member(name="customers", segment="customers"),
        Traverse(),
        Member(name="profile", segment="profile"),
        Literal(value="display name", segment='["display name"]'),
    )


def test_compile_index_and_slice():
    parsed = compile("events[-10:][*].legs[0]")

    assert parsed.steps == (
        Member(name="events", segment="events"),
        Slice(start=-10, stop=None, segment="[-10:]"),
        Traverse(),
        Member(name="legs", segment="legs"),
        Literal(value=0, segment="[0]"),
    )


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("[*].value", "must not begin"),
        ("events[?kind]", "filters are not supported"),
        ("sort(events)", "functions are not supported"),
        ("events|sort", "pipes are not supported"),
        ("parent..child", "recursive descent is not supported"),
        ("record.*", "object wildcards are not supported"),
        ("events[]", "flattening is not supported"),
        ("value==1", "expressions are not supported"),
    ],
)
def test_compile_rejects_expression_language_features(expression: str, reason: str):
    with pytest.raises(ValueError, match=reason):
        compile(expression)


def test_query_selects_struct_and_quoted_members_with_presence():
    profile = pa.struct([pa.field("display name", pa.string())])
    table = pa.table(
        {
            "customer": pa.array(
                [
                    {"profile": {"display name": "Ada"}},
                    {"profile": {"display name": None}},
                    {"profile": None},
                ],
                type=pa.struct([pa.field("profile", profile)]),
            )
        }
    )

    selected = query(table, 'customer.profile["display name"]')

    assert selected.values.to_pylist() == ["Ada", None, None]
    assert selected.present.to_pylist() == [True, True, False]


def test_query_traverses_lists_without_compacting_coordinates():
    table = pa.table(
        {
            "items": pa.array(
                [
                    [{"sku": "A"}, {"sku": None}, None],
                    None,
                    [],
                ],
                type=pa.list_(pa.struct([pa.field("sku", pa.string())])),
            )
        }
    )

    selected = query(table, "items[*].sku")

    assert selected.values.to_pylist() == [["A", None, None], None, []]
    assert selected.present.to_pylist() == [[True, True, False], None, []]


def test_query_indexes_each_list_with_negative_and_missing_positions():
    table = pa.table(
        {
            "legs": pa.array(
                [
                    [{"origin": "A"}, {"origin": "B"}],
                    [],
                    None,
                    [{"origin": None}],
                ],
                type=pa.list_(pa.struct([pa.field("origin", pa.string())])),
            )
        }
    )

    selected = query(table, "legs[-1].origin")

    assert selected.values.to_pylist() == ["B", None, None, None]
    assert selected.present.to_pylist() == [True, False, False, True]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("events[-2:]", [[2, 3], [4], None, []]),
        ("events[:-1]", [[0, 1, 2], [], None, []]),
        ("events[9:20]", [[], [], None, []]),
        ("events[3:1]", [[], [], None, []]),
    ],
)
def test_query_slices_each_list_with_python_bounds(expression: str, expected: list):
    table = pa.table({"events": pa.array([[0, 1, 2, 3], [4], None, []])})

    assert query(table, expression).values.to_pylist() == expected


def test_query_preserves_fixed_size_lists_through_slice_and_traversal():
    table = pa.table(
        {
            "items": pa.array(
                [[{"value": 1}, {"value": 2}, {"value": 3}], None],
                type=pa.list_(pa.struct([pa.field("value", pa.int64())]), 3),
            )
        }
    )

    selected = query(table, "items[-2:][*].value")

    assert selected.values.type == pa.list_(pa.int64(), 2)
    assert selected.values.to_pylist() == [[2, 3], None]
    assert selected.present.to_pylist() == [[True, True], None]


def test_query_preserves_nested_field_metadata_for_empty_fixed_size_input():
    child = pa.field("element", pa.struct([pa.field("value", pa.int64())]), metadata={b"unit": b"row"})
    table = pa.Table.from_arrays(
        [pa.array([], type=pa.list_(child, 3))],
        schema=pa.schema([pa.field("items", pa.list_(child, 3))]),
    )

    sliced_values = query(table, "items[-2:]").values
    traversed = query(table, "items[*].value").values

    assert sliced_values.type == pa.list_(child, 2)
    assert traversed.type.value_field.name == "element"
    assert traversed.type.value_field.metadata == {b"unit": b"row"}


def test_query_looks_up_maps_and_distinguishes_absent_from_null():
    table = pa.table(
        {
            "attributes": pa.array(
                [
                    [("country", "US")],
                    [("country", None)],
                    [("region", "EU")],
                    None,
                ],
                type=pa.map_(pa.string(), pa.string()),
            )
        }
    )

    selected = query(table, 'attributes["country"]')

    assert selected.values.to_pylist() == ["US", None, None, None]
    assert selected.present.to_pylist() == [True, True, False, False]


def test_query_rejects_duplicate_map_keys():
    table = pa.table(
        {
            "attributes": pa.array(
                [[("country", "US"), ("country", "CA")]],
                type=pa.map_(pa.string(), pa.string()),
            )
        }
    )

    with pytest.raises(ValueError, match="duplicate key 'country'"):
        query(table, 'attributes["country"]', address="record/country")


def test_bind_resolves_integer_brackets_from_parent_type():
    schema = pa.schema([pa.field("values", pa.list_(pa.int64()))])

    plan = bind("values[0]", schema)

    assert isinstance(plan.steps[-1], Index)
    assert plan.output == pa.int64()


def test_bind_reuses_the_plan_for_one_exact_arrow_schema():
    schema = pa.schema([pa.field("values", pa.list_(pa.int64()))])

    assert bind("values[*]", schema, address="record/values") is bind("values[*]", schema, address="record/values")


def test_bind_rejects_wrong_parent_type_with_address_context():
    schema = pa.schema([pa.field("value", pa.int64())])

    with pytest.raises(ValueError, match="address 'record/value'.*expected a list"):
        bind("value[*]", schema, address="record/value")


def test_map_key_binding_is_exact():
    schema = pa.schema([pa.field("flags", pa.map_(pa.bool_(), pa.int64()))])

    assert query(
        pa.table({"flags": pa.array([[(True, 1)]], type=schema.field("flags").type)}), "flags[true]"
    ).values.to_pylist() == [1]
    with pytest.raises(ValueError, match="not an exact bool map key"):
        bind("flags[1]", schema)


def test_root_and_branch_queries_feed_coalesce_from_arrow():
    model = rf.Model(
        query="payload",
        events=rf.Branch(
            query="events[-2:]",
            length=2,
            sku=rf.Category(query="product.sku", size=8, p_unavailable=0.0),
            risk=rf.Number(query='metrics["risk score"]'),
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    event = pa.struct(
        [
            pa.field("product", pa.struct([pa.field("sku", pa.string())])),
            pa.field("metrics", pa.struct([pa.field("risk score", pa.float64())])),
        ]
    )
    payload = pa.struct([pa.field("events", pa.list_(event))])
    table = pa.table(
        {
            "payload": pa.array(
                [
                    {
                        "events": [
                            {"product": {"sku": "A"}, "metrics": {"risk score": 1.0}},
                            {"product": {"sku": "B"}, "metrics": {"risk score": None}},
                            {"product": {"sku": "C"}, "metrics": {"risk score": 3.0}},
                        ]
                    },
                    None,
                ],
                type=payload,
            )
        }
    )

    fields = coalesce(Batch(data=table, identity=identities(2)), model.schema, Strata.predict)
    sku = fields[rf.Address("record/events/sku")]
    risk = fields[rf.Address("record/events/risk")]

    assert model.schema.fields.query == "payload"
    assert model.schema.branches[rf.Address("record/events")].query == "events[-2:]"
    assert sku.dense.tolist() == [
        [[Tokens.valued.value, Tokens.valued.value]],
        [[Tokens.padded.value, Tokens.padded.value]],
    ]
    assert sku.values.to_pylist() == ["B", "C"]
    assert risk.dense.tolist() == [
        [[Tokens.null.value, Tokens.valued.value]],
        [[Tokens.padded.value, Tokens.padded.value]],
    ]
    assert risk.values.to_pylist() == [3.0]


def test_arrow_query_treats_mask_spelling_as_ordinary_content():
    model = rf.Model(
        amount=rf.Number(query="payload.amount"),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    table = pa.table({"payload": pa.array([{"amount": "<MASK>"}, {"amount": None}])})

    field = coalesce(Batch(data=table, identity=identities(2)), model.schema, Strata.predict)["record/amount"]

    assert field.dense.tolist() == [[Tokens.valued.value], [Tokens.null.value]]
    assert field.values.to_pylist() == ["<MASK>"]
