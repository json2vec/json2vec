from __future__ import annotations

import ast
import inspect
import uuid

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

    with pytest.raises(TypeError, match="plugin 'number'.*does not accept Arrow type large_string"):
        model.encode(pa.table({"value": ["1.5"]}))


def test_mask_spelling_is_ordinary_typed_string_content():
    model = rf.Model(value=rf.Category(size=8, p_unavailable=0), d_model=8, n_layers=1, n_heads=2)

    fields = model.encode(pa.table({"value": ["<MASK>"]}), mask=False)

    assert fields["record/value"].state.tolist() == [[Tokens.valued.value]]


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
