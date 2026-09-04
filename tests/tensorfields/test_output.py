import pyarrow as pa
import pytest
import torch

from relflow.structs.enums import Tokens
from relflow.tensorfields.base import TENSORFIELDS
from relflow.tensorfields.output import (
    STATE,
    array,
    embedding,
    fixed,
    inferred,
    offsets,
    shape,
    state,
    struct,
    variable,
)
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel


def test_array_flattens_contiguous_primitive_tensor():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).T

    result = array(values, pa.float32())

    assert result.type == pa.float32()
    assert result.to_pylist() == [1.0, 3.0, 2.0, 4.0]


def test_fixed_and_shape_preserve_singleton_axes():
    values = array(torch.arange(6), pa.int64())

    once = fixed(values, 3)
    nested = shape(values, (1, 2, 3))

    assert once.type == pa.list_(pa.int64(), 3)
    assert nested.type == pa.list_(pa.list_(pa.list_(pa.int64(), 3), 2), 1)
    assert len(nested) == 1


def test_fixed_rejects_incompatible_length():
    with pytest.raises(ValueError, match="cannot wrap 5 values"):
        fixed(array(torch.arange(5), pa.int64()), 2)


def test_struct_enforces_declared_order_and_types():
    dtype = pa.struct(
        [
            pa.field("value", pa.int32(), nullable=False),
            pa.field("probability", pa.float32(), nullable=False),
        ]
    )
    result = struct(
        {
            "value": array(torch.tensor([1, 2]), pa.int32()),
            "probability": array(torch.tensor([0.2, 0.8]), pa.float32()),
        },
        dtype,
    )

    assert result.type == dtype

    with pytest.raises(ValueError, match="struct fields"):
        struct(
            {
                "probability": array(torch.tensor([0.2]), pa.float32()),
                "value": array(torch.tensor([1]), pa.int32()),
            },
            dtype,
        )


def test_offsets_and_variable_build_lists_without_rows():
    counts = torch.tensor([2, 0, 1])
    values = array(torch.tensor([10, 11, 12]), pa.int64())

    assert offsets(counts).to_pylist() == [0, 2, 2, 3]
    assert variable(values, counts).to_pylist() == [[10, 11], [], [12]]


def test_variable_rejects_counts_that_do_not_consume_values():
    with pytest.raises(ValueError, match="do not consume"):
        variable(array(torch.tensor([1, 2]), pa.int64()), torch.tensor([1]))


def test_state_emits_one_probability_field_per_token():
    logits = torch.zeros(2, 1, len(Tokens))
    logits[0, 0, Tokens.valued.value] = 10.0
    logits[1, 0, Tokens.null.value] = 10.0

    result = state(logits)

    assert result.type == STATE
    assert len(result) == 2
    assert result.field(Tokens.valued.name)[0].as_py() > 0.99
    assert result.field(Tokens.null.name)[1].as_py() > 0.99


def test_inferred_flattens_boolean_mask():
    result = inferred(torch.tensor([[True, False], [False, True]]))

    assert result.type == pa.bool_()
    assert result.to_pylist() == [True, False, False, True]


def test_embedding_normalizes_and_preserves_width_one():
    result = embedding(torch.tensor([[[3.0]], [[0.0]]]))

    assert result.type == pa.list_(pa.float32(), 1)
    assert result.to_pylist() == [[1.0], [0.0]]


def test_vocabulary_labels_are_large_strings_cached_by_revision():
    vocabulary = OnlineVocabularyModel(size=4)
    stateful = vocabulary.state
    stateful.reserve([1, b"two"], learn=True)

    first = vocabulary.labels()
    second = vocabulary.labels()
    stateful.reserve("three", learn=True)
    third = vocabulary.labels()

    assert first is second
    assert first.type == pa.large_string()
    assert first.to_pylist() == ["1", "b'two'"]
    assert third is not first
    assert third.to_pylist() == ["1", "b'two'", "three"]


@pytest.mark.parametrize("name", ["dateparts", "hash", "text"])
def test_embedding_only_plugins_declare_no_decoded_output(name: str):
    plugin = TENSORFIELDS[name]

    assert plugin.output(module=object(), address=object()) is None
    assert plugin.write(module=object(), prediction=object(), datatype=None) is None
