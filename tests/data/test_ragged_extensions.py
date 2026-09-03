from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import uuid
from datetime import date, datetime

import pyarrow as pa
import pytest

import relflow as rf
import relflow.data.ragged as ragged_module
from relflow.data.iterables import encode
from relflow.data.ragged import RaggedField, coalesce
from relflow.structs.enums import Strata, Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, Plugin
from relflow.tensorfields.extensions.number import number
from tests.arrow import batch


def plugin_name() -> str:
    return f"ragged_{uuid.uuid4().hex[:8]}"


@pytest.mark.parametrize(
    ("family", "datatype"),
    [
        ((str,), pa.large_string()),
        ((bytes,), pa.binary()),
        ((bool,), pa.bool_()),
        ((int,), pa.int64()),
        ((float,), pa.float32()),
        ((int | float,), pa.list_(pa.float64())),
        ((dict,), pa.struct([("key", pa.string())])),
        ((object,), pa.map_(pa.string(), pa.int64())),
    ],
)
def test_plugin_accepts_registered_arrow_terminal_families(family, datatype):
    plugin = Plugin(plugin_name(), types=family)
    try:
        assert plugin.accepts(datatype)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize(
    ("family", "datatype"),
    [
        ((str,), pa.int64()),
        ((bool,), pa.int8()),
        ((int,), pa.bool_()),
        ((float,), pa.decimal128(10, 2)),
        ((dict,), pa.list_(pa.string())),
    ],
)
def test_plugin_rejects_unregistered_arrow_terminal_families(family, datatype):
    plugin = Plugin(plugin_name(), types=family)
    try:
        assert not plugin.accepts(datatype)
        with pytest.raises(TypeError, match="does not accept Arrow type.*normalize it in a preprocessor"):
            plugin.prepare(pa.array([], type=datatype), address=Address("record/value"))
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_prepare_validates_and_preserves_one_whole_arrow_array():
    plugin = Plugin(plugin_name(), types=(int | float,))
    values = pa.array([[1, 2], [3]], type=pa.large_list(pa.int64()))
    try:
        prepared = plugin.prepare(values, address=Address("record/value"))

        assert prepared is values
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize(
    "field",
    [
        rf.Category(size=8, p_unavailable=0.0),
        rf.Hash(),
        rf.Set(size=8, p_unavailable=0.0),
        rf.Cluster(capacity=8, n_clusters=2, p_unavailable=0.0),
    ],
)
def test_multifamily_plugins_accept_all_null_arrow_columns(field):
    model = rf.Model(value=field, d_model=8, n_layers=1, n_heads=2)

    encoded = model.encode(pa.table({"value": pa.nulls(2)}), mask=False)["record/value"]

    assert encoded.state.tolist() == [
        [Tokens.null.value],
        [Tokens.null.value],
    ]


def test_encode_calls_plugin_prepare_once_with_the_whole_leaf_column(monkeypatch: pytest.MonkeyPatch):
    model = rf.Model(value=rf.Number, d_model=8, n_layers=1, n_heads=2)
    calls: list[pa.DataType] = []
    original = number.prepare

    def prepare(values, *, address):
        calls.append(values.type)
        return original(values, address=address)

    monkeypatch.setattr(number, "prepare", prepare)

    encode(
        batch=batch([{"value": 1}, {"value": 2}, {"value": None}]),
        schema=model.schema,
        strata=Strata.predict,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    assert calls == [pa.int64()]


def test_builtin_plugin_type_contract_fails_before_codec_coercion():
    model = rf.Model(value=rf.Number, d_model=8, n_layers=1, n_heads=2)

    with pytest.raises(TypeError, match="plugin 'number'.*does not accept Arrow type string"):
        model.encode(pa.table({"value": ["1.5"]}))


def test_mask_spelling_is_ordinary_typed_string_content():
    model = rf.Model(value=rf.Category(size=8, p_unavailable=0), d_model=8, n_layers=1, n_heads=2)

    fields = model.encode(pa.table({"value": ["<MASK>"]}), mask=False)

    assert fields["record/value"].state.tolist() == [[Tokens.valued.value]]


def test_nested_dictionary_chunks_decode_without_combining_indices():
    chunks = []
    expected = []
    for prefix in ("A", "B"):
        labels = [f"{prefix}{index}" for index in range(100)]
        dictionary = pa.DictionaryArray.from_arrays(
            pa.array(range(100), type=pa.int8()),
            pa.array(labels),
        )
        records = pa.StructArray.from_arrays([dictionary], names=["value"])
        chunks.append(
            pa.ListArray.from_arrays(
                pa.array([0, 100], type=pa.int32()),
                records,
            )
        )
        expected.extend(labels)
    source = pa.table({"items": pa.chunked_array(chunks)})
    model = rf.Model(
        items=rf.Branch(
            length=100,
            value=rf.Category(size=256, p_unavailable=0.0),
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    encoded = model.encode(source, strata=Strata.train, mask=False)["record/items/value"]

    assert encoded.state.eq(Tokens.valued.value).all()
    assert set(rf.Category.vocabulary(model, "record/items/value")) == set(expected)


def test_dictionary_null_values_remain_null_after_decoding():
    values = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, None], type=pa.int8()),
        pa.array(["value", None]),
    )
    model = rf.Model(
        value=rf.Category(size=8, p_unavailable=0.0),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    encoded = model.encode(pa.table({"value": values}), strata=Strata.train, mask=False)["record/value"]

    assert encoded.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.null.value],
        [Tokens.null.value],
    ]
    assert rf.Category.vocabulary(model, "record/value") == ("value",)


def test_extension_storage_uses_dictionary_value_validity():
    class EncodedType(pa.ExtensionType):
        def __init__(self):
            super().__init__(
                pa.dictionary(pa.int8(), pa.string()),
                f"relflow.test.dictionary.{uuid.uuid4().hex}",
            )

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    storage = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, None], type=pa.int8()),
        pa.array(["value", None]),
    )
    values = pa.ExtensionArray.from_storage(EncodedType(), storage)
    model = rf.Model(
        value=rf.Category(size=8, p_unavailable=0.0),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    encoded = model.encode(pa.table({"value": values}), strata=Strata.train, mask=False)["record/value"]

    assert encoded.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.null.value],
        [Tokens.null.value],
    ]


def test_dateparts_encodes_heterogeneous_date_union():
    values = pa.UnionArray.from_dense(
        pa.array([0, 1, 2], type=pa.int8()),
        pa.array([0, 0, 0], type=pa.int32()),
        [
            pa.array(["2024-01-02"]),
            pa.array([date(2024, 2, 3)], type=pa.date32()),
            pa.array([datetime(2024, 3, 4, 5, 6)], type=pa.timestamp("us")),
        ],
    )
    model = rf.Model(
        value=rf.DateParts(dateparts=["day_of_month"]),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    encoded = model.encode(pa.table({"value": values}), strata=Strata.test, mask=False)["record/value"]

    assert encoded.state.tolist() == [
        [Tokens.valued.value],
        [Tokens.valued.value],
        [Tokens.valued.value],
    ]


def test_typed_empty_root_map_does_not_abort_arrow():
    program = """
import pyarrow as pa
import relflow as rf
from relflow.data.datasets.arrow import convert
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata

source = pa.table({"payload": pa.chunked_array([], type=pa.map_(pa.string(), pa.float64()))})
model = rf.Model(
    query="payload",
    value=rf.Number(query='["value"]'),
    d_model=8,
    n_layers=1,
    n_heads=2,
)
field = coalesce(convert(source, namespace="empty", offset=0), model.schema, Strata.predict)["record/value"]
assert field.shape == (0, 1)
assert field.values.type == pa.float64()
"""
    process = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr


def test_typed_empty_branch_preserves_union_storage():
    union = pa.dense_union(
        [
            pa.field("integer", pa.int64()),
            pa.field("floating", pa.float64()),
        ]
    )
    item = pa.struct([pa.field("value", union)])
    source = pa.table({"items": pa.chunked_array([], type=pa.list_(item))})
    model = rf.Model(
        items=rf.Branch(length=3, value=rf.Number),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    encoded = model.encode(source, strata=Strata.test, mask=False)["record/items/value"]

    assert encoded.state.shape == (0, 1, 3)
    assert encoded.content.shape == (0, 1, 3)


def test_coalesce_returns_arrow_backed_ragged_fields():
    model = rf.Model(
        items=rf.Branch(length=3, value=rf.Number),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    source = batch([{"items": [{"value": 1.0}, {"value": None}]}])

    field = coalesce(source, schema=model.schema, strata=Strata.predict)["record/items/value"]

    assert isinstance(field, RaggedField)
    assert isinstance(field.values, pa.Array)
    assert field.values.to_pylist() == [1.0]
    assert field.state.type == pa.int8()
    assert field.state.to_pylist() == [Tokens.valued.value, Tokens.null.value, Tokens.padded.value]
    assert field.placement.type == pa.int64()
    assert field.placement.to_pylist() == [0]
    assert field.shape == (1, 1, 3)


def test_ragged_field_rejects_null_retained_values():
    with pytest.raises(ValueError, match="values cannot contain nulls"):
        RaggedField(
            values=pa.array([None], type=pa.int64()),
            state=pa.array([Tokens.valued.value], type=pa.int8()),
            placement=pa.array([0], type=pa.int64()),
            shape=(1,),
        )


def test_ragged_field_rejects_nulls_hidden_in_union_children():
    values = pa.UnionArray.from_dense(
        pa.array([0], type=pa.int8()),
        pa.array([0], type=pa.int32()),
        [pa.array([None], type=pa.int64()), pa.array([], type=pa.float64())],
    )

    with pytest.raises(ValueError, match="values cannot contain nulls"):
        RaggedField(
            values=values,
            state=pa.array([Tokens.valued.value], type=pa.int8()),
            placement=pa.array([0], type=pa.int64()),
            shape=(1,),
        )


def test_ragged_field_rejects_unknown_state_tokens():
    with pytest.raises(ValueError, match="unknown token"):
        RaggedField(
            values=pa.array([], type=pa.int64()),
            state=pa.array([99], type=pa.int8()),
            placement=pa.array([], type=pa.int64()),
            shape=(1,),
        )


def test_ragged_field_requires_exact_valued_placement():
    with pytest.raises(ValueError, match="every valued state position"):
        RaggedField(
            values=pa.array([1], type=pa.int64()),
            state=pa.array([Tokens.valued.value, Tokens.padded.value], type=pa.int8()),
            placement=pa.array([1], type=pa.int64()),
            shape=(2,),
        )


def test_ragged_core_does_not_name_or_import_registered_tensorfield_types():
    tree = ast.parse(inspect.getsource(ragged_module))
    string_literals = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert string_literals.isdisjoint(TENSORFIELDS)
    assert not any(module.startswith("relflow.tensorfields.extensions") for module in imported_modules)
