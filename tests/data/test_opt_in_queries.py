import pytest

import relflow as rf
from relflow.structs.enums import Strata, Tokens
from tests.arrow import table


def build(*fields):
    return rf.Model(*fields, d_model=8, n_layers=1, n_heads=2)


def test_direct_schema_binding_and_explicit_nested_query_can_share_a_batch():
    model = build(
        rf.Branch(
            rf.Number("value"),
            name="items",
            length=2,
        ),
        rf.Number("gross_amount", query='payload.metrics["gross amount"]'),
    )

    encoded = model.encode(
        table(
            [
                {
                    "items": [{"value": 1.0}, {"value": 2.0}],
                    "payload": {"metrics": {"gross amount": 9.5}},
                },
                {
                    "items": [{"value": 3.0}],
                    "payload": {"metrics": {"gross amount": 10.5}},
                },
            ]
        ),
        strata=Strata.predict,
        mask=False,
    )

    direct = encoded[rf.Address("record/items/value")]
    queried = encoded[rf.Address("record/gross_amount")]

    assert direct.state.tolist() == [
        [[Tokens.valued.value, Tokens.valued.value]],
        [[Tokens.valued.value, Tokens.padded.value]],
    ]
    assert direct.content.tolist() == [[[1.0, 2.0]], [[3.0, 0.0]]]
    assert queried.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
    ]
    assert queried.content.tolist() == [[9.5], [10.5]]


def test_filters_are_rejected_with_a_preprocessor_remedy():
    with pytest.raises(ValueError, match="filters are not supported; use a preprocessor"):
        rf.Branch(
            rf.Number("device_id"),
            name="login_events",
            query="events[?kind]",
            length=2,
        )


def test_query_backed_leaf_ignores_same_named_direct_source_values():
    model = build(
        rf.Category(
            "label",
            query="payload.label",
            size=8,
            p_unavailable=0.0,
        )
    )

    encoded = model.encode(
        table(
            [
                {"label": 1, "payload": {"label": "A"}},
                {"label": 2, "payload": {"label": "B"}},
            ]
        ),
        strata=Strata.train,
        mask=False,
    )

    assert encoded[rf.Address("record/label")].state.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
    ]
    assert rf.Category.vocabulary(model, "record/label") == ("A", "B")


def test_scalar_plugin_rejects_list_valued_query_with_field_context():
    model = build(
        rf.Branch(
            rf.Number("value", query="values[*]"),
            name="items",
            length=2,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"number field at 'record/items/value' expects scalar Arrow values",
    ):
        model.encode(
            table([{"items": [{"values": [1.0, 2.0]}]}]),
            strata=Strata.predict,
            mask=False,
        )


def test_set_owns_the_list_produced_by_a_traversal_query():
    model = build(
        rf.Set(
            "aliases",
            query="contacts[*].alias",
            size=8,
            p_unavailable=0.0,
        )
    )

    encoded = model.encode(
        table(
            [
                {"contacts": [{"alias": "Ada"}, {"alias": "A"}]},
                {"contacts": []},
                {"contacts": None},
            ]
        ),
        strata=Strata.train,
        mask=False,
    )[rf.Address("record/aliases")]

    assert encoded.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
        [Tokens.padded.value],
    ]
    assert rf.Set.vocabulary(model, "record/aliases") == ("Ada", "A")
    assert encoded.content[0, 0].sum().item() == 2
    assert encoded.content[1, 0].sum().item() == 0


def test_vector_owns_the_list_produced_by_a_traversal_query():
    model = build(
        rf.Vector(
            "coordinates",
            query="measurements[*].value",
            n_dim=2,
        )
    )

    encoded = model.encode(
        table(
            [
                {"measurements": [{"value": 1.0}, {"value": 2.0}]},
                {"measurements": [{"value": 3.0}, {"value": 4.0}]},
            ]
        ),
        strata=Strata.predict,
        mask=False,
    )[rf.Address("record/coordinates")]

    assert encoded.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
    ]
    assert encoded.content.tolist() == [[[1.0, 2.0]], [[3.0, 4.0]]]
