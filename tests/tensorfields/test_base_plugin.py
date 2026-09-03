import uuid
from decimal import Decimal
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from lightning.pytorch import Callback

from relflow.structs.enums import Component, Strata
from relflow.structs.tree import Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)


def _plugin_name(prefix: str = "plug") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _build_plugin() -> Plugin:
    plugin = Plugin(name=_plugin_name(), types=(object,))

    class Request(RequestBase):
        type: str = plugin.name

    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, field, address, schema, strata):
            return object()

        def mask(self, p_mask: float):
            return None

        def target(self, p_prune: float):
            return None

    class Embedder(EmbedderBase):
        def __init__(self, schema: object, address: object):
            super().__init__(schema=schema, address=address)

    class Decoder(DecoderBase):
        def __init__(self, schema: object, address: object):
            super().__init__(schema=schema, address=address)

    plugin.register(Request)
    plugin.register(TensorField)
    plugin.register(Embedder)
    plugin.register(Decoder)

    def loss(module: object, prediction: object, batch: object, strata: Strata):
        return 3.14

    def output(module: object, address: object) -> pa.StructType:
        return pa.struct([])

    def write(module: object, prediction: object, datatype: pa.StructType | None) -> None:
        return None

    plugin.register(loss)
    plugin.register(output)
    plugin.register(write)

    return plugin


def test_plugin_rejects_invalid_name():
    with pytest.raises(ValueError, match="lowercase letters"):
        Plugin(name="Bad-Name", types=(str,))


def test_plugin_requires_value_types_without_polluting_registry():
    name = _plugin_name("missingtypes")

    with pytest.raises(TypeError):
        Plugin(name=name)  # ty: ignore[missing-argument]

    assert name not in TENSORFIELDS


@pytest.mark.parametrize(
    "value_types",
    [
        pytest.param([], id="list"),
        pytest.param((), id="empty"),
        pytest.param(("str",), id="non-type"),
        pytest.param((Any,), id="typing-any"),
        pytest.param((list[int],), id="parameterized-generic"),
        pytest.param((int, int | float), id="overlapping-families"),
        pytest.param((str, str), id="duplicate-type"),
        pytest.param((list,), id="list-container"),
        pytest.param((tuple,), id="tuple-container"),
        pytest.param((np.ndarray,), id="numpy-array-container"),
        pytest.param((np.int64,), id="numpy-scalar"),
    ],
)
def test_plugin_rejects_invalid_value_type_families_without_polluting_registry(value_types):
    name = _plugin_name("badtypes")

    with pytest.raises((TypeError, ValueError)):
        Plugin(name=name, types=value_types)

    assert name not in TENSORFIELDS


def test_plugin_stores_separate_and_union_value_type_families():
    name = _plugin_name("families")
    plugin = Plugin(name=name, types=(str, bytes, int | float))
    try:
        assert plugin.types == (str, bytes, int | float)
        with pytest.raises(AttributeError):
            plugin.types = (object,)  # ty: ignore[invalid-assignment]
    finally:
        TENSORFIELDS.pop(name, None)


def test_custom_value_type_requires_and_uses_an_arrow_matcher():
    missing = _plugin_name("decimalmissing")
    with pytest.raises(TypeError, match="custom value type.*require arrow matchers.*Decimal"):
        Plugin(name=missing, types=(Decimal,))
    assert missing not in TENSORFIELDS

    name = _plugin_name("decimal")
    plugin = Plugin(name=name, types=(Decimal,), arrow={Decimal: pa.types.is_decimal})
    try:
        assert plugin.accepts(pa.decimal128(12, 2))
        assert plugin.accepts(pa.decimal256(50, 8))
        assert not plugin.accepts(pa.float64())
    finally:
        TENSORFIELDS.pop(name, None)


def test_custom_arrow_matcher_sees_extension_type_before_storage():
    class Identifier:
        pass

    class IdentifierType(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.binary(16), f"relflow.test.identifier.{uuid.uuid4().hex}")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    datatype = IdentifierType()
    name = _plugin_name("extension")
    plugin = Plugin(
        name=name,
        types=(Identifier,),
        arrow={Identifier: lambda candidate: isinstance(candidate, IdentifierType)},
    )
    bytes_name = _plugin_name("extensionstorage")
    bytes_plugin = Plugin(name=bytes_name, types=(bytes,))
    try:
        assert plugin.accepts(datatype)
        assert bytes_plugin.accepts(datatype)
    finally:
        TENSORFIELDS.pop(name, None)
        TENSORFIELDS.pop(bytes_name, None)


def test_custom_matchers_preserve_declared_union_family_boundaries():
    datatype = pa.dense_union(
        [
            pa.field("text", pa.string()),
            pa.field("decimal", pa.decimal128(12, 2)),
        ]
    )
    union_name = _plugin_name("unionfamily")
    union = Plugin(
        name=union_name,
        types=(str | Decimal,),
        arrow={Decimal: pa.types.is_decimal},
    )
    separate_name = _plugin_name("separatefamilies")
    separate = Plugin(
        name=separate_name,
        types=(str, Decimal),
        arrow={Decimal: pa.types.is_decimal},
    )
    try:
        assert union.accepts(datatype)
        assert not separate.accepts(datatype)
    finally:
        TENSORFIELDS.pop(union_name, None)
        TENSORFIELDS.pop(separate_name, None)


def test_plugin_rejects_invalid_arrow_matcher_declarations():
    with pytest.raises(TypeError, match="arrow must be a mapping"):
        Plugin(name=_plugin_name("arrowmapping"), types=(int,), arrow=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absent from types"):
        Plugin(name=_plugin_name("arrowkey"), types=(int,), arrow={float: pa.types.is_floating})
    with pytest.raises(TypeError, match="must be callable"):
        Plugin(name=_plugin_name("arrowcallable"), types=(int,), arrow={int: object()})  # type: ignore[dict-item]


def test_plugin_judges_python_ingress_after_arrow_canonicalization():
    values = pa.array(["text", b"bytes"])
    name = _plugin_name("canonical")
    plugin = Plugin(name=name, types=(str, bytes))
    try:
        assert pa.types.is_binary(values.type)
        assert plugin.accepts(values.type)
    finally:
        TENSORFIELDS.pop(name, None)


def test_plugin_warns_and_overwrites_duplicate_name():
    name = _plugin_name("duplicate")
    first = Plugin(name=name, types=(str,))
    try:
        with pytest.warns(UserWarning, match="overriding existing tensorfield plugin"):
            second = Plugin(name=name, types=(str,))

        assert TENSORFIELDS[name] is second
        assert TENSORFIELDS[name] is not first
    finally:
        TENSORFIELDS.pop(name, None)


def test_plugin_registers_components_and_wraps_loss():
    plugin = _build_plugin()
    try:
        assert Component.Request in plugin.components
        assert Component.TensorField in plugin.components
        assert Component.Embedder in plugin.components
        assert Component.Decoder in plugin.components
        assert Component.loss in plugin.components
        assert Component.output in plugin.components
        assert Component.write in plugin.components

        class DummyModule:
            def __init__(self):
                self.calls: list[tuple] = []

            def track(self, key: tuple, value: float):
                self.calls.append((key, value))
                return value

        class DummyPrediction:
            address = "root/field"

        module = DummyModule()
        value = plugin.loss(module, prediction=DummyPrediction(), batch=object(), strata=Strata.train)
        assert value == 3.14
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_write_defaults_to_no_output_when_unregistered():
    plugin = Plugin(name=_plugin_name("defaultwrite"), types=(object,))
    try:
        assert plugin.write(module=object(), prediction=object(), datatype=None) is None
        assert Component.write not in plugin.components
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_output_defaults_to_no_output_when_unregistered():
    plugin = Plugin(name=_plugin_name("defaultoutput"), types=(object,))
    try:
        assert plugin.output(module=object(), address=object()) is None
        assert Component.output not in plugin.components
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_accepts_explicit_none_write():
    plugin = Plugin(name=_plugin_name("nonehooks"), types=(object,))
    try:
        plugin.register(None, component=Component.write)

        assert plugin.write(module=object(), prediction=object(), datatype=None) is None
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_accepts_explicit_none_output():
    plugin = Plugin(name=_plugin_name("noneoutput"), types=(object,))
    try:
        plugin.register(None, component=Component.output)

        assert plugin.output(module=object(), address=object()) is None
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_output_requires_module_and_address_parameters():
    plugin = Plugin(name=_plugin_name("badoutput"), types=(object,))

    def output(module: object) -> pa.StructType:
        return pa.struct([])

    try:
        with pytest.raises(TypeError, match="Output function must accept"):
            plugin.register(output)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_write_accepts_a_return_annotation():
    plugin = Plugin(name=_plugin_name("typedwrite"), types=(object,))

    def write(module: object, prediction: object, datatype: pa.StructType | None) -> None:
        return None

    try:
        assert plugin.register(write) is write
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_write_requires_declared_datatype_parameter():
    plugin = Plugin(name=_plugin_name("legacywrite"), types=(object,))

    def write(module: object, prediction: object) -> None:
        return None

    try:
        with pytest.raises(TypeError, match="Write function must accept"):
            plugin.register(write)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_registers_multiple_callback_factories():
    class FirstCallback(Callback):
        pass

    class SecondCallback(Callback):
        pass

    plugin = Plugin(name=_plugin_name("callbacks"), types=(object,))
    try:
        result = plugin.callback(FirstCallback, SecondCallback)

        assert result == (FirstCallback, SecondCallback)
        assert plugin.callback_factories == [FirstCallback, SecondCallback]
        assert [type(callback) for callback in plugin.callbacks] == [FirstCallback, SecondCallback]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_invalid_callback_factory_without_registering():
    class InvalidCallback:
        pass

    plugin = Plugin(name=_plugin_name("badcallbacks"), types=(object,))
    try:
        with pytest.raises(TypeError, match="must produce a Lightning Callback"):
            plugin.callback(InvalidCallback)

        assert plugin.callback_factories == []
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_request_must_implement_leaf_contract():
    class Request(Node):
        type: str = "invalid_request"

    plugin = Plugin(name=_plugin_name("badrequest"), types=(object,))
    try:
        with pytest.raises(TypeError, match="subclass of RequestBase"):
            plugin.register(Request)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_legacy_tensorfield_new_signature():
    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, values, address, schema, strata):
            return object()

        def mask(self, p_mask: float):
            return None

        def target(self, p_prune: float):
            return None

    plugin = Plugin(name=_plugin_name("legacynew"), types=(object,))
    try:
        with pytest.raises(TypeError, match="TensorField.new must accept 'field'"):
            plugin.register(TensorField)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_embedder_with_missing_address_param():
    class Embedder(EmbedderBase):
        def __init__(self, schema: object):
            super().__init__(schema=schema, address=None)

    plugin = Plugin(name=_plugin_name("badembedder"), types=(object,))
    try:
        with pytest.raises(TypeError, match="must accept 'schema' and 'address'"):
            plugin.register(Embedder)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_getattr_rejects_invalid_component_key():
    plugin = Plugin(name=_plugin_name("badattr"), types=(object,))
    try:
        with pytest.raises(ValueError, match="is not a valid Component enum value"):
            plugin.__getattr__("not_real")
    finally:
        TENSORFIELDS.pop(plugin.name, None)
