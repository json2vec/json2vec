from collections.abc import Iterable, Iterator

import numpy as np
import pytest

from json2vec.data.processing import Pipeline, apply, contains_mask_literal, extract_mask_literals, pad
from json2vec.structs.enums import Overflow, Strata, Tokens
from json2vec.structs.tree import Address


def test_pipeline():
    def source() -> Iterator[int]:
        yield from range(5)

    def step1(pipe: Iterable[int]) -> Iterator[int]:
        yield from (x + 1 for x in pipe)

    def step2(pipe: Iterable[int], multiplier: int) -> Iterator[int]:
        yield from (x * multiplier for x in pipe)

    pipe = Pipeline(multiplier=2) | source | step1 | step2
    assert list(pipe) == [2, 4, 6, 8, 10]


def test_pad_tracks_null_and_padding():
    values, flags = pad(
        nested=[[1, None], [2]],
        shape=(2, 2),
        dtype=object,
        pad_value="PAD",
    )

    assert values.tolist() == [[1, "PAD"], [2, "PAD"]]
    assert flags[0, 0] == Tokens.valued
    assert flags[0, 1] == Tokens.null
    assert flags[1, 0] == Tokens.valued
    assert flags[1, 1] == Tokens.padded


def test_pad_truncates_shape_and_skips_incomplete_scalars():
    values, flags = pad(
        nested=[[1, 2, 3], 9, [None, 4]],
        shape=(3, 2),
        dtype=object,
        pad_value="PAD",
    )

    assert values.tolist() == [[1, 2], ["PAD", "PAD"], ["PAD", 4]]
    assert flags[0, 0] == Tokens.valued
    assert flags[0, 1] == Tokens.valued
    assert flags[1, 0] == Tokens.padded
    assert flags[1, 1] == Tokens.padded
    assert flags[2, 0] == Tokens.null
    assert flags[2, 1] == Tokens.valued


def test_pad_tail_overflow_keeps_last_items_and_compacts_slots():
    values, flags = pad(
        nested=[[1, 2, 3]],
        shape=(1, 2),
        dtype=object,
        pad_value="PAD",
        overflows=(Overflow.error, Overflow.tail),
    )

    assert values.tolist() == [[2, 3]]
    assert flags[0, 0] == Tokens.valued
    assert flags[0, 1] == Tokens.valued


def test_pad_error_overflow_raises():
    with pytest.raises(ValueError, match="branch overflow at root node dimension 1"):
        pad(
            nested=[[1, 2, 3]],
            shape=(1, 2),
            overflows=(Overflow.error, Overflow.error),
        )


def test_pad_error_overflow_includes_address_when_provided():
    with pytest.raises(ValueError, match="branch overflow at root node dimension 1 for record/events/amount"):
        pad(
            nested=[[1, 2, 3]],
            shape=(1, 2),
            overflows=(Overflow.error, Overflow.error),
            address="record/events/amount",
        )


def test_pad_batch_overflow_raises():
    with pytest.raises(ValueError, match="branch overflow at batch"):
        pad(
            nested=[[1], [2]],
            shape=(1, 1),
            overflows=(Overflow.error, Overflow.head),
        )


def test_pad_root_overflow_raises():
    with pytest.raises(ValueError, match="branch overflow at root node dimension 1"):
        pad(
            nested=[[[1], [2]]],
            shape=(1, 1, 1),
            overflows=(Overflow.error, Overflow.error, Overflow.head),
        )


def test_pad_nested_overflow_policies_are_per_depth():
    values, _ = pad(
        nested=[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
        shape=(1, 2, 2),
        dtype=object,
        pad_value="PAD",
        overflows=(Overflow.error, Overflow.head, Overflow.tail),
    )

    assert values.tolist() == [[[2, 3], [5, 6]]]


def test_pad_overflow_does_not_slice_leaf_ndarray():
    leaf = np.array([1, 2, 3])
    values, flags = pad(
        nested=[[leaf]],
        shape=(1, 1),
        dtype=object,
        pad_value=None,
        overflows=(Overflow.error, Overflow.error),
    )

    assert np.array_equal(values[0, 0], leaf)
    assert flags[0, 0] == Tokens.valued


def test_pad_encodes_leaf_values_into_trailing_value_shape():
    vocab = {"ALPHA": 0, "BETA": 1, "GAMMA": 2}

    def encode(items: list[str]) -> np.ndarray:
        encoded = np.zeros(3, dtype=np.float32)
        for item in items:
            encoded[vocab[item]] = 1.0
        return encoded

    values, flags = pad(
        nested=[[["ALPHA", "BETA"], None]],
        shape=(1, 2),
        dtype=np.float32,
        pad_value=0.0,
        value_shape=(3,),
        encode=encode,
    )

    assert values.shape == (1, 2, 3)
    assert values.tolist() == [[[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]
    assert flags[0, 0] == Tokens.valued
    assert flags[0, 1] == Tokens.null


def test_pad_numeric_dtype_tracks_null_without_object_array():
    values, flags = pad(
        nested=[[1, None], [3, 4]],
        shape=(2, 2),
        dtype=np.int64,
        pad_value=0,
    )

    assert values.tolist() == [[1, 0], [3, 4]]
    assert flags[0, 0] == Tokens.valued
    assert flags[0, 1] == Tokens.null
    assert flags[1, 0] == Tokens.valued
    assert flags[1, 1] == Tokens.valued


def test_apply_recursively_maps_nested_scalars_and_preserves_none():
    values = [[1, None], [2, 3]]
    output = apply(values, lambda value: value + 1)

    assert output == [[2, None], [3, 4]]


def test_apply_leaf_depth_maps_at_target_depth_only():
    values = [[[1, 2], [3, 4]], [[5, 6], None], 99]
    output = apply(values, tuple, leaf_depth=2)

    assert output == [[(1, 2), (3, 4)], [(5, 6), None], 99]


def test_contains_mask_literal_checks_mapping_values():
    assert contains_mask_literal({"items": [{"code": "<MASK>"}]})
    assert not contains_mask_literal({"items": [{"code": "MASK"}]})


def test_extract_mask_literals_is_noop_outside_predict():
    values = [[["A", "<MASK>"]]]
    cleaned, literal_masks = extract_mask_literals(
        values,
        strata=Strata.train,
        address=Address("record/items/code"),
        leaf_depth=2,
    )

    assert cleaned is values
    assert literal_masks is False
