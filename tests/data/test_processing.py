from collections.abc import Iterable, Iterator

import numpy as np

from json2vec.data.processing import Pipeline, apply, pad
from json2vec.structs.enums import Tokens


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
