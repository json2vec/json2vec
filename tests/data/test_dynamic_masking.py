import numpy as np
import pyarrow as pa
import pytest

import relflow as rf
from relflow.data.arrow import Batch
from relflow.data.iterables import encode
from relflow.data.ragged import boolean, coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.tensorfields.base import TENSORFIELDS
from tests.arrow import batch as arrow_batch


def model(*fields):
    return rf.Model(*fields, d_model=8, n_layers=1, n_heads=2)


def bits(values: pa.Array) -> list[bool]:
    return boolean(values).tolist()


def test_branch_query_skip_is_atomic_across_descendants():
    schema = model(
        rf.Branch(
            rf.Number("amount"),
            rf.Category("code", size=8),
            name="items",
            length=3,
            mask=rf.Mask(query="selected", skip=True, dropout=False, reconstruct=True),
        )
    ).schema
    source = arrow_batch(
        [
            {
                "items": [
                    {"amount": 1.0, "code": "A", "selected": False},
                    {"amount": 2.0, "code": "B", "selected": True},
                ]
            }
        ]
    )

    projections = coalesce(source, schema, Strata.train)
    amount = projections["record/items/amount"]
    code = projections["record/items/code"]

    assert bits(amount.present) == bits(code.present) == [True, False, False]
    assert bits(amount.trainable) == bits(code.trainable) == [False, True, False]
    assert bits(amount.visible) == bits(code.visible) == [True, False, False]

    inputs, targets = amount.split(amount.pristine.values)
    assert inputs.values.to_pylist() == [1.0]
    assert inputs.dense.tolist() == [[[Tokens.valued.value, Tokens.padded.value, Tokens.padded.value]]]
    assert targets.values.to_pylist() == [2.0]
    assert targets.dense.tolist() == [[[Tokens.padded.value, Tokens.valued.value, Tokens.padded.value]]]


def test_rate_skip_is_stable_across_rebatching_and_changes_by_epoch():
    schema = model(
        rf.Number(
            "value",
            mask=rf.Mask(rate=0.5, skip=True, dropout=False),
        )
    ).schema
    source = arrow_batch([{"value": float(index)} for index in range(512)])

    whole = coalesce(source, schema, Strata.train, seed=17, epoch=3)["record/value"]
    first = coalesce(source.slice(0, 173), schema, Strata.train, seed=17, epoch=3)["record/value"]
    second = coalesce(source.slice(173), schema, Strata.train, seed=17, epoch=3)["record/value"]
    rebatching = np.concatenate([boolean(first.present), boolean(second.present)])
    next_epoch = coalesce(source, schema, Strata.train, seed=17, epoch=4)["record/value"]

    assert np.array_equal(boolean(whole.present), rebatching)
    assert not np.array_equal(boolean(whole.present), boolean(next_epoch.present))


def test_query_and_rate_sample_only_eligible_owner_coordinates():
    schema = model(
        rf.Number(
            "value",
            mask=rf.Mask(query="eligible", rate=0.5, skip=True, dropout=False),
        )
    ).schema
    source = arrow_batch([{"value": float(index), "eligible": index % 2 == 0} for index in range(256)])

    projection = coalesce(source, schema, Strata.train, seed=5, epoch=0)["record/value"]
    skipped = ~boolean(projection.present)

    assert skipped.any()
    assert not skipped[1::2].any()


@pytest.mark.parametrize(
    "mask",
    [
        rf.Mask(query="selected"),
        rf.Mask(query="selected", skip=True, dropout=False),
        rf.Mask(query="selected", reconstruct=True),
    ],
)
def test_category_observes_pristine_values_before_mask_projection(mask):
    configured = model(rf.Category("value", size=8, mask=mask))
    source = arrow_batch(
        [
            {"value": "hidden", "selected": True},
            {"value": "visible", "selected": False},
        ]
    )

    encoded = encode(
        source,
        configured.schema,
        Strata.train,
        configured.interprocess_encoding_context,
    )

    assert rf.Category.vocabulary(configured, "record/value") == ("hidden", "visible")
    assert encoded.observations["record/value"][TensorKey.content].sum() == 2


def test_fully_skipped_source_is_prepared_and_observed(monkeypatch):
    configured = model(
        rf.Category(
            "value",
            size=8,
            mask=rf.Mask(skip=True, dropout=False),
        )
    )
    source = arrow_batch([{"value": "A"}, {"value": "B"}])
    extension = TENSORFIELDS["category"]
    prepare = extension.prepare
    seen = []

    def spy(values, *, address):
        seen.append(values.to_pylist())
        return prepare(values, address=address)

    monkeypatch.setattr(extension, "prepare", spy)
    encoded = encode(
        source,
        configured.schema,
        Strata.train,
        configured.interprocess_encoding_context,
    )

    assert seen == [["A", "B"]]
    assert rf.Category.vocabulary(configured, "record/value") == ("A", "B")
    assert encoded.tensors["record/value"].state.reshape(-1).tolist() == [Tokens.padded.value] * 2


def test_number_learns_pristine_moments_once_before_forward():
    configured = model(rf.Number("value", mask=rf.Mask(query="selected")))
    source = arrow_batch(
        [
            {"value": 1.0, "selected": True},
            {"value": 100.0, "selected": False},
        ]
    )
    encoded = encode(
        source,
        configured.schema,
        Strata.train,
        configured.interprocess_encoding_context,
    )
    extension = TENSORFIELDS["number"]

    extension.learn(
        module=configured,
        observation=encoded.observations["record/value"],
        address="record/value",
        strata=Strata.train,
    )
    before = rf.Number.normalization(configured, "record/value")
    configured(encoded.tensors, strata=Strata.train)
    after = rf.Number.normalization(configured, "record/value")

    assert before["count"] == 2
    assert before["mean"] == pytest.approx(50.5)
    assert before["variance"] == pytest.approx(2450.25)
    assert after == before


def test_observation_learning_is_frozen_outside_training():
    configured = model(rf.Category("value", size=8))
    source = arrow_batch([{"value": "A"}, {"value": "B"}])

    encoded = encode(
        source,
        configured.schema,
        Strata.validate,
        configured.interprocess_encoding_context,
    )

    assert encoded.observations == {}
    assert rf.Category.vocabulary(configured, "record/value") == ()


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ([True, None], "contains 1 null"),
        ([1, 0], "must return Boolean"),
    ],
)
def test_query_selector_requires_non_null_scalar_booleans(selector, message):
    schema = model(
        rf.Number(
            "value",
            mask=rf.Mask(query="selected", skip=True, dropout=False),
        )
    ).schema
    source = arrow_batch([{"value": 1.0, "selected": selector[0]}, {"value": 2.0, "selected": selector[1]}])

    with pytest.raises((TypeError, ValueError), match=message):
        coalesce(source, schema, Strata.train)


def test_source_less_prediction_uses_vacancy_and_fixed_routing():
    schema = model(rf.Category("label", size=8, mask=True)).schema
    identity = arrow_batch([{"unused": 1}, {"unused": 2}]).identity
    source = Batch(data=pa.table({}), identity=identity)

    projection = coalesce(source, schema, Strata.predict)["record/label"]
    inputs, targets = projection.split(projection.pristine.values)

    assert projection.vacant is True
    assert pa.types.is_null(projection.pristine.values.type)
    assert projection.pristine.shape == (2, 1)
    assert bits(projection.present) == [False, False]
    assert bits(projection.inferred) == [True, True]
    assert inputs.dense.tolist() == [[Tokens.padded.value], [Tokens.padded.value]]
    assert targets.dense.tolist() == [[Tokens.padded.value], [Tokens.padded.value]]


@pytest.mark.parametrize("strata", [Strata.train, Strata.validate, Strata.test])
def test_source_less_reconstruction_fails_outside_prediction(strata):
    schema = model(rf.Number("value"), rf.Category("label", size=8, mask=True)).schema
    source = arrow_batch([{"value": 1.0}, {"value": 2.0}])

    with pytest.raises(ValueError, match="field 'label' is absent"):
        coalesce(source, schema, strata)


def test_source_less_learned_mask_reconstruction_remains_present():
    schema = model(
        rf.Category(
            "label",
            size=8,
            mask=rf.Mask(dropout=False, reconstruct=True),
        )
    ).schema
    identity = arrow_batch([{"unused": 1}, {"unused": 2}]).identity
    source = Batch(data=pa.table({}), identity=identity)

    projection = coalesce(source, schema, Strata.predict)["record/label"]
    inputs, _ = projection.split(projection.pristine.values)

    assert projection.vacant is True
    assert bits(projection.present) == [True, True]
    assert bits(projection.inferred) == [True, True]
    assert inputs.dense.tolist() == [[Tokens.masked.value], [Tokens.masked.value]]
