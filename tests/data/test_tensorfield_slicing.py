import torch

import relflow as rf
from relflow.data.datasets.base import EncodedInput
from relflow.data.iterables import mock
from relflow.structs.enums import Strata, TensorKey, Tokens


def nested_model() -> rf.Model:
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        events=rf.Branch(
            length=3,
            items=rf.Branch(
                length=4,
                amount=rf.Number,
                embedding=rf.Vector(n_dim=3),
                tags=rf.Set(size=5, p_unavailable=0.0),
                occurred_at=rf.DateParts(dateparts=["day_of_week"]),
            ),
        ),
    )


def nested_records() -> list[dict]:
    return [
        {
            "events": [
                {
                    "items": [
                        {
                            "amount": 1.0,
                            "embedding": [1.0, 2.0, 3.0],
                            "tags": ["new", "sale"],
                            "occurred_at": "2026-08-18",
                        },
                        {
                            "amount": None,
                            "embedding": [3.0, 2.0, 1.0],
                            "tags": ["sale"],
                            "occurred_at": "2026-08-17",
                        },
                    ]
                }
            ]
        },
        {"events": []},
    ]


def test_encoded_tensorfields_use_named_structural_batch_dimensions() -> None:
    model = nested_model()
    encoded = model.encode(nested_records(), strata=Strata.train, mask=False)
    address = "record/events/items/amount"
    field = encoded[address]

    assert isinstance(encoded, EncodedInput)
    assert encoded.batch_size == torch.Size([2])
    assert encoded.names == ["batch"]
    assert field.batch_size == field.state.shape == torch.Size([2, 1, 3, 4])
    assert field.names == [
        "batch",
        "/record",
        "/record/events",
        "/record/events/items",
    ]
    assert field.targets.batch_size == field.batch_size

    selected = torch.zeros_like(field.state, dtype=torch.bool)
    selected[0, 0, 0, 1] = True
    field.hide(selected)

    sliced = field[0, 0, :2, :3]
    assert sliced.batch_size == torch.Size([2, 3])
    assert sliced.names == ["/record/events", "/record/events/items"]
    assert sliced.state.shape == torch.Size([2, 3])
    assert sliced.content.shape == torch.Size([2, 3])
    assert sliced.trainable.shape == torch.Size([2, 3])
    assert sliced.targets.batch_size == torch.Size([2, 3])
    assert sliced.targets[TensorKey.state].shape == torch.Size([2, 3])
    assert sliced.targets[TensorKey.content].shape == torch.Size([2, 3])


def test_structural_slices_preserve_field_specific_payload_dimensions() -> None:
    model = nested_model()
    encoded = model.encode(nested_records(), strata=Strata.train, mask=False)
    selection = (0, 0, slice(None, 2), slice(None, 3))

    vector = encoded["record/events/items/embedding"][selection]
    assert vector.state.shape == torch.Size([2, 3])
    assert vector.content.shape == torch.Size([2, 3, 3])

    tags = encoded["record/events/items/tags"][selection]
    assert tags.state.shape == torch.Size([2, 3])
    assert tags.content.shape == torch.Size([2, 3, 5])

    dateparts = encoded["record/events/items/occurred_at"][selection]
    assert dateparts.state.shape == torch.Size([2, 3])
    assert dateparts.content.batch_size == torch.Size([2, 3])
    assert dateparts.content.names == ["/record/events", "/record/events/items"]
    assert {tuple(value.shape) for value in dateparts.content.values()} == {(2, 3, 2)}


def test_outer_encoded_input_slices_every_field_by_observation() -> None:
    model = nested_model()
    encoded = model.encode(nested_records(), strata=Strata.predict, mask=False)

    first = encoded[0]
    field = first["record/events/items/amount"]

    assert isinstance(first, EncodedInput)
    assert first.batch_size == torch.Size([])
    assert first.names == []
    assert field.batch_size == torch.Size([1, 3, 4])
    assert field.names == ["/record", "/record/events", "/record/events/items"]
    assert field.state.shape == torch.Size([1, 3, 4])
    assert TensorKey.metadata in first.keys()

    retained = encoded[:1]
    retained_field = retained["record/events/items/embedding"]
    assert retained.batch_size == torch.Size([1])
    assert retained.names == ["batch"]
    assert retained_field.batch_size == torch.Size([1, 1, 3, 4])
    assert retained_field.content.shape == torch.Size([1, 1, 3, 4, 3])


def test_predict_target_empty_uses_the_full_structural_layout() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        groups=rf.Branch(
            length=3,
            label=rf.Number(target=True),
        ),
    )
    encoded = model.encode([{"groups": [{}, {}]}, {"groups": []}])
    field = encoded["record/groups/label"]

    assert field.batch_size == field.state.shape == torch.Size([2, 1, 3])
    assert field.names == ["batch", "/record", "/record/groups"]
    assert field.targets.batch_size == field.batch_size
    assert field.state.eq(Tokens.masked.value).all()

    sliced = field[0, 0, :2]
    assert sliced.batch_size == torch.Size([2])
    assert sliced.names == ["/record/groups"]
    assert sliced.content.shape == torch.Size([2])


def test_empty_encoded_batches_keep_a_sliceable_zero_length_batch() -> None:
    model = nested_model()
    encoded = model.encode([], mask=False)

    assert encoded.batch_size == torch.Size([0])
    assert encoded.names == ["batch"]

    retained = encoded[:0]
    field = retained["record/events/items/amount"]
    assert retained.batch_size == torch.Size([0])
    assert field.batch_size == field.state.shape == torch.Size([0, 1, 3, 4])
    assert field.names == [
        "batch",
        "/record",
        "/record/events",
        "/record/events/items",
    ]


def test_mock_uses_the_same_structural_layout_as_real_encodes() -> None:
    model = nested_model()
    encoded = mock(model.schema, batch_size=3)

    assert isinstance(encoded, EncodedInput)
    assert encoded.batch_size == torch.Size([3])
    assert encoded.names == ["batch"]

    for address, field in encoded.items():
        expected_shape = torch.Size((3, *model.schema.shapes[address]))
        assert field.batch_size == field.state.shape == expected_shape
        assert field.names == [
            "batch",
            "/record",
            "/record/events",
            "/record/events/items",
        ]

    vector = encoded["record/events/items/embedding"][0, 0, :2, :3]
    assert vector.content.shape == torch.Size([2, 3, 3])


def test_full_address_axis_names_cannot_collide_with_batch() -> None:
    model = rf.Model(
        rf.Branch(rf.Number("value"), name="batch", length=2),
        name="batch",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"batch": [{"value": 1.0}]}], mask=False)["batch/batch/value"]

    assert field.names == ["batch", "/batch", "/batch/batch"]
    assert len(field.names) == len(set(field.names))
