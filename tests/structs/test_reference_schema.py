import inspect

import pydantic
import pytest

import relflow as rf


def test_pooling_configs_are_frozen_strict_tagged_values():
    assert rf.Mean().model_dump() == {"type": "mean", "width": None}
    assert rf.Attention(width=2, n_heads=4, n_layers=3, dropout=0.1).type == "attention"
    assert rf.Convolution(width=2, kernel_size=5, n_layers=2).type == "convolution"

    with pytest.raises(pydantic.ValidationError, match="frozen"):
        rf.Attention().width = 2
    with pytest.raises(pydantic.ValidationError):
        rf.Attention(width=True)
    with pytest.raises(pydantic.ValidationError, match="odd"):
        rf.Convolution(kernel_size=2)
    assert not hasattr(rf, "CrossAttention")


def test_reduce_normalizes_named_axes_and_builtin_reducers():
    reduced = rf.Reduce("mean", b1=True, ignored=False)
    assert reduced.reducer == rf.Mean()
    assert reduced.axes == (rf.AxisResize(rf.AxisName("b1"), 1),)

    exact = rf.Reduce("sum", axes={rf.Address("record", "b1"): 2})
    assert exact.reducer == "sum"
    assert exact.axes[0].address == "record/b1"
    assert isinstance(exact.axes[0].address, rf.Address)
    assert exact.axes[0].size == 2

    with pytest.raises(ValueError, match="positive"):
        rf.Reduce(b1=0)
    with pytest.raises(TypeError, match="booleans or positive integers"):
        rf.Reduce(b1=1.5)
    with pytest.raises(ValueError, match="width"):
        rf.Reduce(rf.Attention(width=2), b1=1)
    with pytest.raises(ValueError, match="exactly one"):
        rf.Reduce(rf.Convolution(), b1=1, b2=1)


def test_reference_accepts_one_exact_address_and_normalizes_empty_reduce():
    string = rf.Reference("record/a", graft=True, reduce=rf.Reduce(a=False))
    address = rf.Reference(rf.Address("record", "a"), graft=True)
    assert string == address
    assert isinstance(string.address, rf.Address)
    assert string.reduce is None
    assert string.model_dump(mode="json") == {
        "address": "record/a",
        "graft": True,
        "reduce": None,
    }

    assert rf.Reference.model_validate({"address": "record/a"}) == rf.Reference("record/a")
    with pytest.raises((TypeError, pydantic.ValidationError)):
        rf.Reference(rf.where(address="record/a"))
    with pytest.raises((TypeError, pydantic.ValidationError)):
        rf.Reference(["record/a"])
    with pytest.raises(pydantic.ValidationError):
        rf.Reference("record/a", graft=1)
    with pytest.raises(pydantic.ValidationError, match="detach"):
        rf.Reference("record/a", detach=True)


def test_branch_reference_cardinality_round_trips_canonically():
    first = rf.Reference("record/a")
    second = rf.Reference("record/c", graft=True)

    empty = rf.Branch(name="b", value=rf.Number)
    scalar = rf.Branch(name="b", reference=(first,), value=rf.Number)
    multiple = rf.Branch(name="b", reference=(first, second), value=rf.Number)

    assert empty.reference == ()
    assert scalar.reference == first
    assert multiple.reference == (first, second)
    assert empty.model_dump(mode="json")["reference"] == []
    assert scalar.model_dump(mode="json")["reference"] == first.model_dump(mode="json")
    assert multiple.model_dump(mode="json")["reference"] == [
        first.model_dump(mode="json"),
        second.model_dump(mode="json"),
    ]

    for branch in (empty, scalar, multiple):
        payload = branch.model_dump(mode="json", round_trip=True)
        assert rf.Branch.model_validate(payload).model_dump(mode="json") == branch.model_dump(mode="json")

    singleton_payload = scalar.model_dump(mode="json", round_trip=True)
    singleton_payload["reference"] = [singleton_payload["reference"]]
    assert rf.Branch.model_validate(singleton_payload).reference == first

    with pytest.raises(pydantic.ValidationError):
        rf.Branch(name="b", reference="record/a", value=rf.Number)
    with pytest.raises(ValueError, match="singular reference"):
        rf.Branch(name="b", references=(first,), value=rf.Number)


def test_root_pooling_reference_and_removed_options_are_public_schema_state():
    schema = rf.Schema.from_tree(
        rf.Branch(name="a", length=2, value=rf.Number),
        rf.Branch(name="b", length=2, value=rf.Number),
        d_model=16,
        n_layers=1,
        n_heads=4,
        pooling=rf.Convolution(width=2),
        reference=rf.Reference("record/a"),
    )

    assert schema.fields.pooling == rf.Convolution(width=2)
    assert schema.fields.reference == rf.Reference("record/a")
    assert "n_linear" not in rf.Branch.model_fields
    assert "n_linear" not in rf.Leaf.model_fields
    assert "n_linear" not in inspect.signature(rf.Model).parameters
    assert "n_linear" not in inspect.signature(rf.Model.from_tree).parameters
    assert "n_linear" not in inspect.signature(rf.Schema.from_tree).parameters

    with pytest.raises(ValueError, match="n_linear was removed"):
        rf.Schema.from_tree(
            rf.Number("value"),
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_linear=2,
        )
    with pytest.raises(ValueError, match="unsupported tensorfield option"):
        rf.Number("value", reference=rf.Reference("record/a"))


def test_pool_width_and_attention_head_geometry_validate_after_binding():
    with pytest.raises(ValueError, match="flattened schema size 2"):
        rf.Schema.from_tree(
            rf.Branch(
                rf.Number("value", pooling=rf.Attention(width=1)),
                name="items",
                length=2,
            ),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    with pytest.raises(ValueError, match="Mean pooling width must be 1"):
        rf.Schema.from_tree(
            rf.Branch(rf.Number("value"), name="items", pooling=rf.Mean(width=2)),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    with pytest.raises(ValueError, match="Attention pooling requires"):
        rf.Schema.from_tree(
            rf.Number("value", pooling=rf.Attention(n_heads=6)),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_exact_address_selector_sugar_shares_where_cache_identity():
    schema = rf.Schema.from_tree(
        rf.Branch(name="a", length=2, value=rf.Number),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    direct = schema.select("record/a/value")
    typed = schema.select(rf.Address("record", "a", "value"))
    keyword = schema.select(rf.where(address="record/a/value"))
    expanded = schema.select(rf.where("address") == "record/a/value")

    assert direct == typed == keyword == expanded
    assert rf.where(address="record/a/value").key == (rf.where("address") == "record/a/value").key

    schema.update("record/a/value", weight=2.0)
    assert schema.requests["record/a/value"].weight == 2.0
    with pytest.raises(ValueError, match="derived"):
        schema.update("record/a/value", address="record/renamed")
    with pytest.raises(TypeError, match="either"):
        rf.where("address", address="record/a")
    with pytest.raises(TypeError, match="requires"):
        rf.where()
