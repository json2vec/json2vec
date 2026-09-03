from __future__ import annotations

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.arrow import IDENTITY, Batch, Encoded, variants


def identities(size: int) -> pa.Array:
    return pa.array(
        [
            {
                "logical": index.to_bytes(32),
                "instance": (index + 100).to_bytes(32),
                "order": f"row/{index:04d}".encode(),
            }
            for index in range(size)
        ],
        type=IDENTITY,
    )


def test_batch_is_public():
    assert rf.Batch is Batch


def test_variants_respects_sliced_union_offsets():
    values = pa.UnionArray.from_dense(
        pa.array([0, 1, 0], type=pa.int8()),
        pa.array([0, 0, 1], type=pa.int32()),
        [pa.array([1, 3]), pa.array([2.0])],
    ).slice(1)

    codes, offsets = variants(values)

    assert codes.tolist() == [1, 0]
    assert offsets is not None
    assert offsets.tolist() == [0, 1]


def test_batch_validates_data_alignment():
    with pytest.raises(ValueError, match="data has 2 rows but identity has 3 rows"):
        Batch(data=pa.table({"value": [10, 20]}), identity=identities(3))


def test_batch_validates_identity_type_and_nullability():
    data = pa.table({"value": [10]})

    with pytest.raises(TypeError, match="identity must have type"):
        Batch(data=data, identity=pa.array([1]))

    null_row = pa.array([None], type=IDENTITY)
    with pytest.raises(ValueError, match="identity cannot contain null rows"):
        Batch(data=data, identity=null_row)

    null_field = pa.array(
        [{"logical": None, "instance": b"i" * 32, "order": b"row/0000"}],
        type=IDENTITY,
    )
    with pytest.raises(ValueError, match="field 'logical' cannot contain nulls"):
        Batch(data=data, identity=null_field)


def test_slice_keeps_data_and_chunked_identity_aligned():
    identity = identities(4)
    chunked = pa.chunked_array([identity.slice(0, 1), identity.slice(1)])
    batch = Batch(data=pa.table({"value": [10, 20, 30, 40]}), identity=chunked)

    sliced = batch.slice(1, 2)

    assert sliced.data.equals(pa.table({"value": [20, 30]}))
    assert sliced.identity.equals(chunked.slice(1, 2))
    assert len(sliced) == 2


def test_take_selects_and_reorders_data_and_identity():
    identity = identities(4)
    batch = Batch(data=pa.table({"value": [10, 20, 30, 40]}), identity=identity)
    indices = pa.array([3, 1, 1], type=pa.int64())

    taken = batch.take(indices)

    assert taken.data.equals(pa.table({"value": [40, 20, 20]}))
    assert taken.identity.equals(identity.take(indices))


def test_take_rejects_non_integer_and_null_indices():
    batch = Batch(data=pa.table({"value": [10, 20]}), identity=identities(2))

    with pytest.raises(TypeError, match="integer Arrow type"):
        batch.take(pa.array([True, False]))
    with pytest.raises(ValueError, match="indices cannot contain nulls"):
        batch.take(pa.array([0, None], type=pa.int64()))


def test_filter_uses_one_mask_for_data_and_identity():
    identity = identities(4)
    batch = Batch(data=pa.table({"value": [10, 20, 30, 40]}), identity=identity)

    selected = batch.filter(pa.array([True, None, False, True]))

    assert selected.data.equals(pa.table({"value": [10, 40]}))
    assert selected.identity.equals(identity.take(pa.array([0, 3])))


def test_replace_preserves_identity_and_validates_length():
    identity = identities(3)
    batch = Batch(data=pa.table({"value": [10, 20, 30]}), identity=identity)

    replaced = batch.replace(pa.table({"label": ["a", "b", "c"]}))

    assert replaced.data.equals(pa.table({"label": ["a", "b", "c"]}))
    assert replaced.identity is identity

    with pytest.raises(ValueError, match="data has 2 rows but identity has 3 rows"):
        batch.replace(pa.table({"label": ["a", "b"]}))


def test_zero_column_batch_uses_identity_as_logical_row_count():
    identity = identities(4)
    batch = Batch(data=pa.table({}), identity=identity)

    assert batch.data.num_rows == 0
    assert len(batch) == 4

    sliced = batch.slice(1, 2)
    assert sliced.data.num_columns == 0
    assert len(sliced) == 2
    assert sliced.identity.equals(identity.slice(1, 2))

    taken = batch.take(pa.array([3, 0]))
    assert taken.data.num_columns == 0
    assert len(taken) == 2
    assert taken.identity.equals(identity.take(pa.array([3, 0])))

    restored = batch.replace(pa.table({"value": [10, 20, 30, 40]}))
    assert len(restored) == 4

    with pytest.raises(ValueError, match="data has 3 rows but identity has 4 rows"):
        batch.replace(pa.table({"value": [10, 20, 30]}))


def test_encoded_keeps_tensors_and_arrow_source_separate():
    source = Batch(data=pa.table({"value": [10, 20]}), identity=identities(2))
    tensors = TensorDict({"value": torch.tensor([10, 20])}, batch_size=[2])

    encoded = Encoded(tensors=tensors, source=source, retain=("value",))

    assert encoded.tensors is tensors
    assert encoded.source is source
    assert encoded.retain == ("value",)


def test_encoded_is_internal_and_validates_retain():
    source = Batch(data=pa.table({"value": [10]}), identity=identities(1))
    tensors = TensorDict({}, batch_size=[])

    assert "Encoded" not in __import__("relflow.data.arrow", fromlist=["__all__"]).__all__
    with pytest.raises(ValueError, match="tuple of unique"):
        Encoded(tensors=tensors, source=source, retain=("value", "value"))


def test_order_preserves_observation_identity_and_replaces_sort_order():
    source = identities(3)
    batch = Batch(data=pa.table({"value": [10, 20, 30]}), identity=source)

    ordered = batch.order(pa.array([2, 0, 1]))

    assert ordered.data["value"].to_pylist() == [30, 10, 20]
    assert ordered.identity.field("logical").equals(source.take(pa.array([2, 0, 1])).field("logical"))
    assert ordered.identity.field("instance").equals(source.take(pa.array([2, 0, 1])).field("instance"))
    assert ordered.identity.field("order").to_pylist() == sorted(ordered.identity.field("order").to_pylist())
    with pytest.raises(ValueError, match="permutation of every row"):
        batch.order(pa.array([2, 0]))
    with pytest.raises(ValueError, match="permutation of every row"):
        batch.order(pa.array([2, 2, 0]))


def test_expand_derives_stable_identity_from_parent_and_ordinal():
    batch = Batch(data=pa.table({"value": [10, 20]}), identity=identities(2))
    parents = pa.array([1, 0, 1])
    ordinals = pa.array([0, 7, 1])
    data = pa.table({"child": ["a", "b", "c"]})

    expanded = batch.expand(data, parents, ordinals)
    repeated = batch.expand(data, parents, ordinals)

    assert expanded.identity.equals(repeated.identity)
    assert len(set(expanded.identity.field("logical").to_pylist())) == 3
    assert expanded.identity.field("order").to_pylist()[0] < expanded.identity.field("order").to_pylist()[2]
    with pytest.raises(ValueError, match="unique within each parent"):
        batch.expand(pa.table({"child": ["a", "b"]}), pa.array([0, 0]), pa.array([1, 1]))


@pytest.mark.parametrize(
    "datatype",
    [
        pa.list_(pa.int64()),
        pa.large_list(pa.int64()),
        pa.list_(pa.int64(), 2),
    ],
)
def test_explode_repeats_other_columns_and_skips_absent_lists(datatype: pa.DataType):
    lists = [[1, 2], None, [3, 4]] if pa.types.is_fixed_size_list(datatype) else [[1, 2], None, []]
    batch = Batch(
        data=pa.table(
            {
                "label": ["a", "b", "c"],
                "items": pa.array(lists, type=datatype),
            }
        ),
        identity=identities(3),
    )

    exploded = batch.explode("items", name="item")

    expected_items = [1, 2, 3, 4] if pa.types.is_fixed_size_list(datatype) else [1, 2]
    expected_labels = ["a", "a", "c", "c"] if pa.types.is_fixed_size_list(datatype) else ["a", "a"]
    assert exploded.data.column_names == ["label", "item"]
    assert exploded.data["item"].to_pylist() == expected_items
    assert exploded.data["label"].to_pylist() == expected_labels
    assert len(exploded) == len(expected_items)


def test_group_derives_identity_from_ordered_parent_sequences():
    batch = Batch(data=pa.table({"value": [10, 20, 30]}), identity=identities(3))
    data = pa.table({"total": [30, 30]})

    grouped = batch.group(data, pa.array([[0, 1], [2]], type=pa.list_(pa.int64())))
    reversed_group = batch.group(
        pa.table({"total": [30]}),
        pa.array([[1, 0]], type=pa.list_(pa.int64())),
    )

    assert len(grouped) == 2
    assert len(set(grouped.identity.field("logical").to_pylist())) == 2
    assert grouped.identity.field("logical")[0] != reversed_group.identity.field("logical")[0]
    with pytest.raises(ValueError, match="repeat a parent"):
        batch.group(pa.table({"total": [20]}), pa.array([[0, 0]], type=pa.list_(pa.int64())))


def test_cardinality_operations_keep_zero_column_row_counts():
    batch = Batch(data=pa.table({"value": [10, 20]}), identity=identities(2))

    expanded = batch.expand(pa.table({}), pa.array([0, 0, 1]), pa.array([0, 1, 0]))
    grouped = batch.group(pa.table({}), pa.array([[0, 1]], type=pa.list_(pa.int64())))

    assert expanded.data.num_columns == 0
    assert len(expanded) == 3
    assert grouped.data.num_columns == 0
    assert len(grouped) == 1
