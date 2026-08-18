import relflow as rf
from relflow.data.datasets.base import EncodedInput


def test_encoded_input_display_summarizes_fields_without_raw_metadata() -> None:
    secret = "SENSITIVE_OBSERVATION_VALUE"
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
        items=rf.Branch(
            length=3,
            sku=rf.Category(size=8),
        ),
    )

    encoded = model.encode(
        [{"amount": 1.0, "items": [{"sku": "A"}], "credential": secret}],
        strata=rf.Strata.predict,
        mask=False,
    )

    assert isinstance(encoded, EncodedInput)
    for rendered in (str(encoded), repr(encoded)):
        assert "EncodedInput [encoded] batch_size=(1,) fields=2 metadata=hidden" in rendered
        assert "record/amount [number] state=(1, 1) axes=(batch, /record)" in rendered
        assert "record/items/sku [category] state=(1, 1, 3)" in rendered
        assert secret not in rendered
        assert "credential" not in rendered

    bundle = encoded._repr_mimebundle_()
    assert set(bundle) == {"text/html", "text/plain"}
    assert secret not in bundle["text/html"]
    assert secret not in bundle["text/plain"]


def test_sliced_encoded_input_keeps_the_compact_display() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
    )
    encoded = model.encode([{"amount": 1.0}, {"amount": 2.0}], mask=False)

    first = encoded[0]

    assert isinstance(first, EncodedInput)
    assert "EncodedInput [encoded] batch_size=() fields=1 metadata=hidden" in str(first)
    assert "record/amount [number] state=(1,) axes=(/record)" in str(first)


def test_encoded_input_display_bounds_wide_schemas() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        **{f"field_{index:02d}": rf.Number for index in range(30)},
    )
    encoded = model.encode(
        [{f"field_{index:02d}": float(index) for index in range(30)}],
        mask=False,
    )

    rendered = str(encoded)

    assert "fields=30" in rendered
    assert "… +6 fields" in rendered
    assert "record/field_29" not in rendered
    assert len(rendered) < 4_000


def test_encoded_input_display_keeps_deep_sibling_addresses_distinct() -> None:
    shared = f"shared_{'x' * 130}"
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        **{
            shared: rf.Branch(
                length=1,
                alpha=rf.Number,
                beta=rf.Number,
            )
        },
    )
    encoded = model.encode(
        [{shared: [{"alpha": 1.0, "beta": 2.0}]}],
        mask=False,
    )

    field_lines = [line for line in str(encoded).splitlines() if "[number]" in line]

    assert len(field_lines) == 2
    assert any("/alpha [number]" in line for line in field_lines)
    assert any("/beta [number]" in line for line in field_lines)
    assert len(set(field_lines)) == 2
    assert all(len(line) <= 120 for line in field_lines)
