from __future__ import annotations

import ast
import inspect
import uuid
from types import UnionType
from typing import Any, Literal

import awkward as ak
import numpy as np
import pytest

import relflow.data.ragged as ragged_module
from relflow.data.ragged import RaggedBatch, RaggedField
from relflow.structs.enums import Strata, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, Plugin, RequestBase


def _plugin_name() -> str:
    return f"ragged_{uuid.uuid4().hex[:8]}"


def _extension(
    value_types: tuple[type[Any] | UnionType, ...],
    *,
    query: str | None = None,
) -> tuple[Plugin, Schema]:
    plugin_name = _plugin_name()
    plugin = Plugin(plugin_name, types=value_types)

    @plugin.register
    class Request(RequestBase):
        type: Literal[plugin_name] = plugin_name

    schema = Schema.from_tree(
        Request("identity", query=query),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    return plugin, schema


def _project(
    schema: Schema,
    values: list[Any],
    *,
    query: str | None,
) -> RaggedField:
    source_name = "source" if query is not None else "identity"
    batch = RaggedBatch.new(
        [[{source_name: value}] for value in values],
        schema=schema,
    )
    return RaggedField.new(batch, address="record/identity", strata=Strata.predict)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("alpha", "alpha", id="str"),
        pytest.param(b"alpha", b"alpha", id="bytes"),
    ],
)
def test_custom_plugin_accepts_each_registered_type_family(query, value, expected):
    plugin, schema = _extension((str, bytes), query=query)
    try:
        field = _project(schema, [value], query=query)

        assert ak.to_list(field.values) == [expected]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_custom_plugin_rejects_an_unregistered_value_type(query):
    plugin, schema = _extension((str,), query=query)
    try:
        with pytest.raises(TypeError, match="record/identity"):
            _project(schema, [1], query=query)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_direct_type_validation_runs_when_ragged_batch_is_created():
    plugin, schema = _extension((str,))
    try:
        with pytest.raises(TypeError, match="record/identity"):
            RaggedBatch.new([[{"identity": 1}]], schema=schema)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_query_type_validation_is_deferred_until_field_projection():
    query = "[*].source"
    plugin, schema = _extension((str,), query=query)
    try:
        batch = RaggedBatch.new([[{"source": 1}]], schema=schema)

        with pytest.raises(TypeError, match="record/identity"):
            RaggedField.new(batch, address="record/identity", strata=Strata.predict)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_union_entry_allows_safe_mixed_python_types(query):
    plugin, schema = _extension((int | float,), query=query)
    try:
        field = _project(schema, [1, 2.5], query=query)

        assert ak.to_list(field.values) == [1.0, 2.5]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_numeric_family_does_not_accept_boolean_subclass(query):
    plugin, schema = _extension((int | float,), query=query)
    try:
        with pytest.raises(TypeError, match="unsupported bool"):
            _project(schema, [True], query=query)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize(
    ("value", "family", "expected"),
    [
        pytest.param(np.longdouble("1.25"), (float,), 1.25, id="long-double"),
        pytest.param(np.clongdouble("1.25+0.5j"), (complex,), 1.25 + 0.5j, id="complex-long-double"),
    ],
)
def test_extended_numpy_scalars_canonicalize_without_recursion(value, family, expected):
    plugin, schema = _extension(family)
    try:
        field = _project(schema, [value], query=None)

        assert ak.to_list(field.values) == [expected]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_homogeneous_numpy_array_stays_columnar_during_plugin_preparation():
    plugin = Plugin(_plugin_name(), types=(int | float,))
    value = np.arange(200_000, dtype=np.float32)
    try:
        prepared, observed = plugin.prepare_value(value, address=Address("record/vector"))

        assert prepared is value
        assert observed is None
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_separate_entries_reject_mixed_python_type_families(query):
    plugin, schema = _extension((int, float), query=query)
    try:
        with pytest.raises(TypeError, match="record/identity"):
            _project(schema, [1, 2.5], query=query)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(["alpha", "beta"], id="list"),
        pytest.param(("alpha", "beta"), id="tuple"),
        pytest.param(np.asarray(["alpha", "beta"]), id="numpy-array"),
    ],
)
def test_custom_plugin_recursively_validates_sequence_atoms(query, value):
    plugin, schema = _extension((str,), query=query)
    try:
        field = _project(schema, [value], query=query)

        assert ak.to_list(field.values) == [["alpha", "beta"]]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_custom_plugin_rejects_mixed_families_inside_one_structured_value(query):
    plugin, schema = _extension((str, bytes), query=query)
    try:
        with pytest.raises(TypeError, match="record/identity"):
            _project(schema, [["alpha", b"beta"]], query=query)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_numpy_and_python_scalars_resolve_to_the_same_family(query):
    plugin, schema = _extension((int,), query=query)
    try:
        field = _project(schema, [np.int64(1), 2], query=query)

        assert ak.to_list(field.values) == [1, 2]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_exact_matching_keeps_bool_and_int_in_separate_families(query):
    plugin, schema = _extension((int, bool), query=query)
    try:
        with pytest.raises(TypeError, match="record/identity"):
            _project(schema, [True, 1], query=query)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize(
    ("query", "null_state"),
    [
        pytest.param(None, Tokens.null.value, id="direct"),
        pytest.param("[*].source", Tokens.padded.value, id="query"),
    ],
)
def test_null_and_mask_routing_bypass_plugin_value_types(query, null_state):
    plugin, schema = _extension((int,), query=query)
    try:
        field = _project(schema, [None, "<MASK>", 1], query=query)

        assert field.state.tolist() == [
            [null_state],
            [Tokens.masked.value],
            [Tokens.valued.value],
        ]
        assert ak.to_list(field.values) == [1]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


@pytest.mark.parametrize("query", [None, "[*].source"], ids=["direct", "query"])
def test_bytes_mask_spelling_remains_an_ordinary_checked_value(query):
    plugin, schema = _extension((int,), query=query)
    try:
        with pytest.raises(TypeError, match="record/identity"):
            _project(schema, [b"<MASK>"], query=query)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_request_base_has_no_datatype_value_type_hooks():
    assert "ragged_value_types" not in RequestBase.__dict__
    assert "preserve_python_identity" not in RequestBase.__dict__


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
