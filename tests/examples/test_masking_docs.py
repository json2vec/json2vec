from __future__ import annotations

from pathlib import Path

import awkward as ak
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import relflow as rf


def model(mask: rf.Mask) -> rf.Model:
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        value=rf.Number(mask=mask),
    )


def selected(field) -> list[bool]:
    state = field.state.reshape(-1)
    return state.ne(rf.Tokens.valued).tolist()


@pytest.mark.parametrize(
    ("query", "rate", "selection"),
    [
        (None, None, "all"),
        ("eligible", None, "query"),
        (None, 0.5, "rate"),
        ("eligible", 0.5, "query_rate"),
    ],
)
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize(
    ("dropout", "reconstruct", "purpose"),
    [
        (True, False, "dropout"),
        (False, True, "reconstruction"),
        (False, False, "ablation"),
    ],
)
@pytest.mark.parametrize("strata", list(rf.Strata))
def test_documented_mask_configuration_matrix(
    query: str | None,
    rate: float | None,
    selection: str,
    skip: bool,
    dropout: bool,
    reconstruct: bool,
    purpose: str,
    strata: rf.Strata,
):
    size = 256 if rate is not None else 8
    eligible = [index % 2 == 0 for index in range(size)]
    source = pa.table(
        {
            "value": [float(index) for index in range(size)],
            "eligible": eligible,
        }
    )
    policy = rf.Mask(
        query=query,
        rate=rate,
        skip=skip,
        dropout=dropout,
        reconstruct=reconstruct,
    )

    field = model(policy).encode(
        source,
        strata=strata,
        seed=19,
        epoch=3,
    )[rf.Address("record/value")]

    active = strata == rf.Strata.train or (not dropout and (strata != rf.Strata.predict or rate is None))
    chosen = selected(field)

    if not active:
        assert not any(chosen)
    elif selection == "all":
        assert all(chosen)
    elif selection == "query":
        assert chosen == eligible
    elif selection == "rate":
        assert any(chosen)
        assert not all(chosen)
    else:
        assert any(chosen)
        assert not any(value for value, allowed in zip(chosen, eligible, strict=True) if not allowed)

    expected_target = [active and reconstruct and value for value in chosen]
    if strata == rf.Strata.predict:
        assert field.trainable.reshape(-1).tolist() == [False] * size
        assert field.inferred.reshape(-1).tolist() == expected_target
    else:
        assert field.trainable.reshape(-1).tolist() == expected_target
        assert field.inferred.reshape(-1).tolist() == [False] * size

    if skip:
        assert field.present.reshape(-1).tolist() == [not value for value in chosen]
    else:
        assert field.present.all()

    assert purpose in {"dropout", "reconstruction", "ablation"}


@rf.preprocess(
    requires=("amount",),
    produces=("mask_amount",),
)
def threshold(batch: rf.Batch, *, cutoff: float) -> rf.Batch:
    values = pc.fill_null(pc.greater(batch.data["amount"], cutoff), False)
    return batch.replace(batch.data.append_column("mask_amount", values))


def test_documented_arrow_preprocessor_selector_and_partial_binding():
    prepare = threshold.partial(cutoff=1_000.0)
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        amount=rf.Number(
            mask=rf.Mask(
                query="mask_amount",
                skip=True,
                dropout=False,
            )
        ),
    )
    source = pa.table({"amount": [100.0, 2_000.0, None]})

    field = configured.encode(
        source,
        preprocess=prepare,
        strata="predict",
    )[rf.Address("record/amount")]

    assert field.present[:, 0].tolist() == [True, False, True]
    assert field.state[:, 0].tolist() == [
        rf.Tokens.valued,
        rf.Tokens.padded,
        rf.Tokens.null,
    ]


def test_documented_arrow_preprocessor_attaches_to_data_module():
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=3,
        amount=rf.Number(
            mask=rf.Mask(
                query="mask_amount",
                skip=True,
                dropout=False,
            )
        ),
    )
    data = rf.ArrowDataModule(
        model=configured,
        validate=pa.table({"amount": [100.0, 2_000.0, None]}),
        preprocessor=threshold.partial(cutoff=1_000.0),
        shuffle=False,
    )

    encoded = next(iter(data.val_dataloader()))
    field = encoded.tensors[rf.Address("record/amount")]

    assert field.present[:, 0].tolist() == [True, False, True]


@rf.preprocess(
    requires=("deny_amount",),
    produces=("mask_amount",),
)
def prediction_policy(batch: rf.Batch, *, strata: rf.Strata) -> rf.Batch:
    values = (
        pc.fill_null(batch.data["deny_amount"], False)
        if strata == rf.Strata.predict
        else pa.repeat(pa.scalar(False, type=pa.bool_()), len(batch))
    )
    return batch.replace(batch.data.append_column("mask_amount", values))


def test_documented_stratum_aware_selector():
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        amount=rf.Number(
            mask=rf.Mask(
                query="mask_amount",
                skip=True,
                dropout=False,
            )
        ),
    )
    source = pa.table({"amount": [1.0, 2.0], "deny_amount": [True, False]})

    training = configured.encode(source, preprocess=prediction_policy, strata="train")
    prediction = configured.encode(source, preprocess=prediction_policy, strata="predict")

    assert training[rf.Address("record/amount")].present[:, 0].tolist() == [True, True]
    assert prediction[rf.Address("record/amount")].present[:, 0].tolist() == [False, True]


EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("kind", pa.large_string()),
            pa.field("amount", pa.float64()),
        ]
    )
)
MASKED_EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("kind", pa.large_string()),
            pa.field("amount", pa.float64()),
            pa.field("mask_event", pa.bool_(), nullable=False),
        ]
    )
)


@rf.preprocess(
    requires=("events",),
    produces=("events",),
)
def refunds(batch: rf.Batch) -> rf.Batch:
    index = batch.data.schema.get_field_index("events")
    source = batch.data["events"]
    if source.null_count == len(source):
        values = pa.nulls(len(source), type=MASKED_EVENTS)
    else:
        events = ak.from_arrow(source)
        mask = ak.fill_none(events["amount"] < 0, False)
        events = ak.with_field(events, mask, "mask_event")
        values = pc.cast(ak.to_arrow(events, extensionarray=False), MASKED_EVENTS)
    return batch.replace(batch.data.set_column(index, "events", values))


def events() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "events": [
                    {"kind": "sale", "amount": 5.0},
                    {"kind": "refund", "amount": -2.0},
                    {"kind": "unknown", "amount": None},
                ]
            }
        ],
        schema=pa.schema([pa.field("events", EVENTS)]),
    )


def test_documented_awkward_selector_is_atomic_for_a_branch():
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            mask=rf.Mask(
                query="mask_event",
                skip=True,
                dropout=False,
            ),
            kind=rf.Category(size=8),
            amount=rf.Number,
        ),
    )

    encoded = configured.encode(events(), preprocess=refunds, strata="predict")
    amount = encoded[rf.Address("record/events/amount")]
    kind = encoded[rf.Address("record/events/kind")]

    assert amount.present.tolist() == kind.present.tolist() == [[[True, False, True]]]


def test_documented_shared_selector_can_drive_different_leaf_effects():
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            kind=rf.Category(
                size=8,
                mask=rf.Mask(query="mask_event", dropout=False),
            ),
            amount=rf.Number(
                mask=rf.Mask(
                    query="mask_event",
                    skip=True,
                    dropout=False,
                ),
            ),
        ),
    )

    encoded = configured.encode(events(), preprocess=refunds, strata="predict")
    amount = encoded[rf.Address("record/events/amount")]
    kind = encoded[rf.Address("record/events/kind")]

    assert amount.present.tolist() == [[[True, False, True]]]
    assert kind.present.tolist() == [[[True, True, True]]]
    assert amount.state.tolist() == [[[rf.Tokens.valued, rf.Tokens.padded, rf.Tokens.null]]]
    assert kind.state.tolist() == [[[rf.Tokens.valued, rf.Tokens.masked, rf.Tokens.valued]]]


@rf.preprocess(
    requires=("events",),
    produces=("events",),
)
def recent(batch: rf.Batch) -> rf.Batch:
    index = batch.data.schema.get_field_index("events")
    source = batch.data["events"]
    if source.null_count == len(source):
        converted = pa.nulls(len(source), type=MASKED_EVENTS)
    else:
        values = ak.from_arrow(source)
        position = ak.local_index(values, axis=1)
        mask = position >= ak.num(values, axis=1) - 2
        values = ak.with_field(values, mask, "mask_event")
        converted = pc.cast(ak.to_arrow(values, extensionarray=False), MASKED_EVENTS)
    return batch.replace(batch.data.set_column(index, "events", converted))


def test_documented_last_n_selector():
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            amount=rf.Number(
                mask=rf.Mask(
                    query="mask_event",
                    skip=True,
                    dropout=False,
                )
            ),
        ),
    )

    field = configured.encode(
        events(),
        preprocess=recent,
        strata="predict",
    )[rf.Address("record/events/amount")]

    assert field.present.tolist() == [[[True, False, False]]]


RISK_EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("amount", pa.float64()),
            pa.field("risk", pa.float64()),
        ]
    )
)
MASKED_RISK_EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("amount", pa.float64()),
            pa.field("risk", pa.float64()),
            pa.field("mask_event", pa.bool_(), nullable=False),
        ]
    )
)


@rf.preprocess(
    requires=("events",),
    produces=("events",),
)
def top_risk(batch: rf.Batch, *, count: int = 2) -> rf.Batch:
    index = batch.data.schema.get_field_index("events")
    source = batch.data["events"]
    if source.null_count == len(source):
        converted = pa.nulls(len(source), type=MASKED_RISK_EVENTS)
    else:
        values = ak.from_arrow(source)
        risk = ak.fill_none(values["risk"], float("-inf"))
        order = ak.argsort(risk, axis=1, ascending=False, stable=True)
        rank = ak.argsort(order, axis=1, ascending=True, stable=True)
        values = ak.with_field(values, rank < count, "mask_event")
        converted = pc.cast(ak.to_arrow(values, extensionarray=False), MASKED_RISK_EVENTS)
    return batch.replace(batch.data.set_column(index, "events", converted))


def test_documented_exact_k_selector():
    source = pa.Table.from_pylist(
        [
            {
                "events": [
                    {"amount": 10.0, "risk": 0.1},
                    {"amount": 20.0, "risk": 0.9},
                    {"amount": 30.0, "risk": 0.4},
                ]
            }
        ],
        schema=pa.schema([pa.field("events", RISK_EVENTS)]),
    )
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            amount=rf.Number(
                mask=rf.Mask(
                    query="mask_event",
                    skip=True,
                    dropout=False,
                )
            ),
        ),
    )

    field = configured.encode(
        source,
        preprocess=top_risk,
        strata="predict",
    )[rf.Address("record/events/amount")]

    assert field.present.tolist() == [[[True, False, False]]]


@pytest.mark.parametrize(
    ("preprocess", "datatype"),
    [
        (refunds, EVENTS),
        (recent, EVENTS),
        (top_risk, RISK_EVENTS),
    ],
)
def test_documented_nested_selectors_preserve_all_null_schema(preprocess, datatype):
    source = pa.Table.from_arrays(
        [pa.nulls(2, type=datatype)],
        names=["events"],
    )
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            amount=rf.Number(
                mask=rf.Mask(
                    query="mask_event",
                    skip=True,
                    dropout=False,
                )
            ),
        ),
    )

    field = configured.encode(
        source,
        preprocess=preprocess,
        strata="predict",
    )[rf.Address("record/events/amount")]

    assert not field.present.any()


@rf.preprocess(
    requires=("events", "deny_events"),
    produces=("events",),
)
def broadcast(batch: rf.Batch) -> rf.Batch:
    index = batch.data.schema.get_field_index("events")
    source = batch.data["events"]
    if source.null_count == len(source):
        converted = pa.nulls(len(source), type=MASKED_EVENTS)
    else:
        values = ak.from_arrow(source)
        denied = ak.from_arrow(pc.fill_null(batch.data["deny_events"], False))
        mask = ak.broadcast_arrays(values["amount"], denied)[1]
        values = ak.with_field(values, mask, "mask_event")
        converted = pc.cast(ak.to_arrow(values, extensionarray=False), MASKED_EVENTS)
    return batch.replace(batch.data.set_column(index, "events", converted))


def test_documented_observation_selector_broadcasts_into_nested_records():
    source = pa.Table.from_pylist(
        [
            {
                "events": [
                    {"kind": "sale", "amount": 1.0},
                    {"kind": "sale", "amount": 2.0},
                ],
                "deny_events": True,
            },
            {
                "events": [{"kind": "sale", "amount": 3.0}],
                "deny_events": False,
            },
        ],
        schema=pa.schema(
            [
                pa.field("events", EVENTS),
                pa.field("deny_events", pa.bool_()),
            ]
        ),
    )
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=2,
            mask=rf.Mask(
                query="mask_event",
                skip=True,
                dropout=False,
            ),
            amount=rf.Number,
        ),
    )

    field = configured.encode(
        source,
        preprocess=broadcast,
        strata="predict",
    )[rf.Address("record/events/amount")]

    assert field.present.tolist() == [
        [[False, False]],
        [[True, False]],
    ]


def test_train_only_query_is_not_required_during_prediction():
    configured = model(rf.Mask(query="eligible"))

    field = configured.encode(
        pa.table({"value": [1.0, 2.0]}),
        strata="predict",
    )[rf.Address("record/value")]

    assert field.state[:, 0].tolist() == [rf.Tokens.valued, rf.Tokens.valued]


def test_query_backed_target_is_not_documented_as_source_less():
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        feature=rf.Number,
        label=rf.Boolean(query="labels.value", mask=True),
    )

    with pytest.raises(ValueError, match="labels"):
        configured.predict(pa.table({"feature": [1.0, 2.0]}))


def test_masking_pages_document_the_complete_public_wiring():
    root = Path(__file__).resolve().parents[2]
    reference = (root / "docs/core-concepts/dynamic-masking.qmd").read_text()
    recipes = (root / "docs/guides/dynamic-mask-preprocessors.qmd").read_text()

    assert "4 × 2 × 3 = 24" in reference
    assert "owner-relative" in reference.lower()
    assert "Query-backed reconstruction" in reference
    assert "preprocessor=prepare" in recipes
    assert "preprocess=prepare" in recipes
    assert ".preprocess(prepare)" in recipes
    assert "skip=True` is an encoder-routing boundary, not secure deletion" in reference
