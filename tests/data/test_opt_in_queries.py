import pytest

import relflow as rf
from relflow.structs.enums import Strata, Tokens


def _model(*fields):
    return rf.Model(*fields, d_model=8, n_layers=1, n_heads=2)


def test_direct_schema_binding_and_explicit_nested_query_can_share_a_batch():
    model = _model(
        rf.Branch(
            rf.Number("value"),
            name="items",
            length=2,
        ),
        rf.Number("gross_amount", query='[*].payload.metrics."gross amount"'),
    )

    encoded = model.encode(
        [
            {
                "items": [{"value": 1.0}, {"value": 2.0}],
                "payload": {"metrics": {"gross amount": 9.5}},
            },
            {
                "items": [{"value": 3.0}],
                "payload": {"metrics": {"gross amount": 10.5}},
            },
        ],
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


def test_explicit_filter_and_map_query_preserves_coordinates_before_overflow():
    model = _model(
        rf.Branch(
            rf.Number("device_id", query="[*].map(&device_id, events[?kind == 'login'])"),
            rf.Number("risk_score", query="[*].map(&risk_score, events[?kind == 'login'])"),
            name="login_events",
            length=2,
            overflow="head",
        )
    )

    encoded = model.encode(
        [
            {
                "events": [
                    {"kind": "login", "device_id": 1, "risk_score": None},
                    {"kind": "purchase", "device_id": 99, "risk_score": 99.0},
                    {"kind": "login", "device_id": 2, "risk_score": 2.0},
                    {"kind": "login", "device_id": 3, "risk_score": 3.0},
                ]
            }
        ],
        strata=Strata.predict,
        mask=False,
    )

    device_id = encoded[rf.Address("record/login_events/device_id")]
    risk_score = encoded[rf.Address("record/login_events/risk_score")]

    # `map` retains the null risk at the first selected event. The third login
    # is then removed by the branch's normal head-overflow policy.
    assert device_id.state.tolist() == [[[Tokens.valued.value, Tokens.valued.value]]]
    assert device_id.content.tolist() == [[[1.0, 2.0]]]
    assert risk_score.state.tolist() == [[[Tokens.null.value, Tokens.valued.value]]]
    assert risk_score.content.tolist() == [[[0.0, 2.0]]]


def test_explicit_query_result_uses_ragged_mask_literal_semantics():
    model = _model(rf.Number("amount", query="[*].payload.amount"))
    record = {"payload": {"amount": rf.MASK_LITERAL}}

    encoded = model.encode([record], strata=Strata.predict, mask=False)
    amount = encoded[rf.Address("record/amount")]

    assert amount.state.tolist() == [[Tokens.masked.value]]
    assert amount.content.tolist() == [[0.0]]
    assert not amount.trainable.any()

    with pytest.raises(ValueError, match="only valid during predict"):
        model.encode([record], strata=Strata.train, mask=False)


def test_query_backed_leaf_ignores_same_named_direct_source_values():
    model = _model(
        rf.Category(
            "label",
            query="[*].payload.label",
            size=8,
            p_unavailable=0.0,
        )
    )

    encoded = model.encode(
        [
            {"label": 1, "payload": {"label": "A"}},
            {"label": "ignored", "payload": {"label": "B"}},
        ],
        strata=Strata.train,
        mask=False,
    )

    assert encoded[rf.Address("record/label")].state.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
    ]
    assert rf.Category.vocabulary(model, "record/label") == ("A", "B")


def test_explicit_query_reports_schema_rank_mismatch_with_field_context():
    model = _model(
        rf.Branch(
            rf.Number("value", query="[*].payload.value"),
            name="items",
            length=2,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"JMESPath query for address 'record/items/value'.*must produce 3 list axes",
    ):
        model.encode([{"payload": {"value": 1.0}}], strata=Strata.predict, mask=False)
