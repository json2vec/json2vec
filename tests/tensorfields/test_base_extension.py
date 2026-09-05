import sys
import uuid
from decimal import Decimal
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from lightning.pytorch import Callback

from relflow.structs.enums import Component, Strata
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
)


def extension_name(prefix: str = "extension") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def build_extension() -> Extension:
    extension = Extension(name=extension_name(), types=(object,))

    class Request(RequestBase):
        type: str = extension.name

    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, input, target, present, trainable, inferred, address, schema, strata, context):
            return object()

    class Embedder(EmbedderBase):
        def __init__(self, schema: object, address: object):
            super().__init__(schema=schema, address=address)

        def forward(self, inputs):
            return inputs

    class Decoder(DecoderBase):
        def __init__(self, schema: object, address: object):
            super().__init__(schema=schema, address=address)

    extension.register(Request)
    extension.register(TensorField)
    extension.register(Embedder)
    extension.register(Decoder)

    def loss(module: object, prediction: object, batch: object, strata: Strata):
        return 3.14

    def output(module: object, address: object) -> pa.StructType:
        return pa.struct([])

    def write(module: object, prediction: object, datatype: pa.StructType | None) -> None:
        return None

    extension.register(loss)
    extension.register(output)
    extension.register(write)

    return extension


def test_extension_rejects_invalid_name():
    with pytest.raises(ValueError, match="lowercase letters"):
        Extension(name="Bad-Name", types=(str,))


def test_extension_requires_value_types_without_polluting_registry():
    name = extension_name("missingtypes")

    with pytest.raises(TypeError):
        Extension(name=name)  # ty: ignore[missing-argument]

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
def test_extension_rejects_invalid_value_type_families_without_polluting_registry(value_types):
    name = extension_name("badtypes")

    with pytest.raises((TypeError, ValueError)):
        Extension(name=name, types=value_types)

    assert name not in TENSORFIELDS


def test_extension_stores_separate_and_union_value_type_families():
    name = extension_name("families")
    extension = Extension(name=name, types=(str, bytes, int | float))
    try:
        assert extension.types == (str, bytes, int | float)
        with pytest.raises(AttributeError):
            extension.types = (object,)  # ty: ignore[invalid-assignment]
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_stores_immutable_runtime_requirements():
    name = extension_name("requires")
    extension = Extension(
        name=name,
        types=(str,),
        requires={"example.codec": "example-codec[fast] >= 2"},
    )
    try:
        assert extension.requires == {"example.codec": "example-codec[fast] >= 2"}
        with pytest.raises(TypeError):
            extension.requires["other"] = "other"  # type: ignore[index]
    finally:
        TENSORFIELDS.pop(name, None)


@pytest.mark.parametrize(
    ("requires", "error"),
    [
        pytest.param([], TypeError, id="non-mapping"),
        pytest.param({"bad-name": "example"}, ValueError, id="invalid-import"),
        pytest.param({1: "example"}, ValueError, id="non-string-import"),
        pytest.param({"example": ""}, ValueError, id="empty-requirement"),
        pytest.param({"example": None}, ValueError, id="non-string-requirement"),
    ],
)
def test_extension_rejects_invalid_runtime_requirements_without_polluting_registry(requires, error):
    name = extension_name("badrequires")

    with pytest.raises(error):
        Extension(name=name, types=(str,), requires=requires)

    assert name not in TENSORFIELDS


def test_extension_requires_reports_all_missing_imports_and_install_targets():
    name = extension_name("missingpackages")
    first = f"missing_{uuid.uuid4().hex}"
    second = f"missing_{uuid.uuid4().hex}"
    extension = Extension(
        name=name,
        types=(str,),
        requires={first: "example[first]", second: "example[second]"},
    )
    try:
        with pytest.raises(ModuleNotFoundError) as raised:
            extension.require(address=Address("record/value"))

        message = str(raised.value)
        assert raised.value.name == first
        assert "record/value" in message
        assert name in message
        assert first in message
        assert second in message
        assert "python -m pip install 'example[first]' 'example[second]'" in message
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_requires_accepts_imported_modules_without_specs(monkeypatch: pytest.MonkeyPatch):
    name = extension_name("loadedpackage")
    module = f"loaded_{uuid.uuid4().hex}"
    marker = object()
    extension = Extension(name=name, types=(str,), requires={module: "example-loaded"})
    monkeypatch.setitem(sys.modules, module, marker)
    try:
        extension.require(address=Address("record/value"))
    finally:
        TENSORFIELDS.pop(name, None)


def test_custom_value_type_requires_and_uses_an_arrow_matcher():
    missing = extension_name("decimalmissing")
    with pytest.raises(TypeError, match="custom value type.*require arrow matchers.*Decimal"):
        Extension(name=missing, types=(Decimal,))
    assert missing not in TENSORFIELDS

    name = extension_name("decimal")
    extension = Extension(name=name, types=(Decimal,), arrow={Decimal: pa.types.is_decimal})
    try:
        assert extension.accepts(pa.decimal128(12, 2))
        assert extension.accepts(pa.decimal256(50, 8))
        assert not extension.accepts(pa.float64())
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
    name = extension_name("extension")
    extension = Extension(
        name=name,
        types=(Identifier,),
        arrow={Identifier: lambda candidate: isinstance(candidate, IdentifierType)},
    )
    bytes_name = extension_name("extensionstorage")
    bytes_extension = Extension(name=bytes_name, types=(bytes,))
    try:
        assert extension.accepts(datatype)
        assert bytes_extension.accepts(datatype)

        values = pa.ExtensionArray.from_storage(
            datatype,
            pa.array([b"0123456789abcdef"], type=pa.binary(16)),
        )
        assert extension.prepare(values, address=Address("record/value")) is values
        prepared = bytes_extension.prepare(values, address=Address("record/value"))
        assert prepared.type == pa.binary(16)
        assert prepared.to_pylist() == [b"0123456789abcdef"]
    finally:
        TENSORFIELDS.pop(name, None)
        TENSORFIELDS.pop(bytes_name, None)


def test_extension_prepare_decodes_dictionary_wrappers_recursively():
    extension = Extension(name=extension_name("dictionary"), types=(str,))
    dictionary = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0], type=pa.int8()),
        pa.array(["A", "B"]),
    )
    values = pa.ListArray.from_arrays(pa.array([0, 2, 3]), dictionary)
    try:
        prepared = extension.prepare(values, address=Address("record/value"))

        assert prepared.type == pa.list_(pa.string())
        assert prepared.to_pylist() == [["A", "B"], ["A"]]
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_prepare_promotes_compatible_unions_recursively():
    extension = Extension(name=extension_name("union"), types=(int | float,))
    union = pa.UnionArray.from_dense(
        pa.array([0, 1, 0], type=pa.int8()),
        pa.array([0, 0, 1], type=pa.int32()),
        [pa.array([1, 3], type=pa.int64()), pa.array([2.5], type=pa.float64())],
    )
    values = pa.ListArray.from_arrays(pa.array([0, 2, 3]), union)
    try:
        prepared = extension.prepare(values, address=Address("record/value"))

        assert prepared.type == pa.list_(pa.float64())
        assert prepared.to_pylist() == [[1.0, 2.5], [3.0]]
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_prepare_reports_unsafe_union_promotion_with_context():
    extension = Extension(name=extension_name("unsafeunion"), types=(int,))
    values = pa.UnionArray.from_dense(
        pa.array([0, 1], type=pa.int8()),
        pa.array([0, 0], type=pa.int32()),
        [pa.array([1], type=pa.int64()), pa.array([2**63], type=pa.uint64())],
    )
    try:
        with pytest.raises(TypeError, match=f"extension '{extension.name}'.*record/value.*cannot safely normalize"):
            extension.prepare(values, address=Address("record/value"))
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_custom_matchers_preserve_declared_union_family_boundaries():
    datatype = pa.dense_union(
        [
            pa.field("text", pa.string()),
            pa.field("decimal", pa.decimal128(12, 2)),
        ]
    )
    union_name = extension_name("unionfamily")
    union = Extension(
        name=union_name,
        types=(str | Decimal,),
        arrow={Decimal: pa.types.is_decimal},
    )
    separate_name = extension_name("separatefamilies")
    separate = Extension(
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


def test_extension_rejects_invalid_arrow_matcher_declarations():
    with pytest.raises(TypeError, match="arrow must be a mapping"):
        Extension(name=extension_name("arrowmapping"), types=(int,), arrow=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absent from types"):
        Extension(name=extension_name("arrowkey"), types=(int,), arrow={float: pa.types.is_floating})
    with pytest.raises(TypeError, match="must be callable"):
        Extension(name=extension_name("arrowcallable"), types=(int,), arrow={int: object()})  # type: ignore[dict-item]


def test_extension_judges_python_ingress_after_arrow_canonicalization():
    values = pa.array(["text", b"bytes"])
    name = extension_name("canonical")
    extension = Extension(name=name, types=(str, bytes))
    try:
        assert pa.types.is_binary(values.type)
        assert extension.accepts(values.type)
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_warns_and_overwrites_duplicate_name():
    name = extension_name("duplicate")
    first = Extension(name=name, types=(str,))
    try:
        with pytest.warns(UserWarning, match="overriding existing tensorfield extension"):
            second = Extension(name=name, types=(str,))

        assert TENSORFIELDS[name] is second
        assert TENSORFIELDS[name] is not first
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_registers_components_and_wraps_loss():
    extension = build_extension()
    try:
        assert Component.Request in extension.components
        assert Component.TensorField in extension.components
        assert Component.Embedder in extension.components
        assert Component.Decoder in extension.components
        assert Component.loss in extension.components
        assert Component.output in extension.components
        assert Component.write in extension.components

        class DummyModule:
            def __init__(self):
                self.calls: list[tuple] = []

            def track(self, key: tuple, value: float):
                self.calls.append((key, value))
                return value

        class DummyPrediction:
            address = "root/field"

        module = DummyModule()
        value = extension.loss(module, prediction=DummyPrediction(), batch=object(), strata=Strata.train)
        assert value == 3.14
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_write_defaults_to_no_output_when_unregistered():
    extension = Extension(name=extension_name("defaultwrite"), types=(object,))
    try:
        assert extension.write(module=object(), prediction=object(), datatype=None) is None
        assert Component.write not in extension.components
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_output_defaults_to_no_output_when_unregistered():
    extension = Extension(name=extension_name("defaultoutput"), types=(object,))
    try:
        assert extension.output(module=object(), address=object()) is None
        assert Component.output not in extension.components
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_observation_components_default_to_no_op():
    extension = Extension(name=extension_name("defaultobserve"), types=(object,))
    try:
        assert (
            extension.observe(
                field=object(),
                address=Address("record/value"),
                schema=object(),
                state=None,
                learn=False,
            )
            is None
        )
        assert (
            extension.learn(
                module=object(),
                observation=object(),
                address=Address("record/value"),
                strata=Strata.train,
            )
            is None
        )
    finally:
        TENSORFIELDS.pop(extension.name, None)


@pytest.mark.parametrize("component", ["observe", "learn"])
def test_extension_observation_components_require_exact_signatures(component):
    extension = Extension(name=extension_name(f"bad{component}"), types=(object,))

    if component == "observe":

        def observe(field, address):
            return None

        function = observe
    else:

        def learn(module, observation):
            return None

        function = learn

    try:
        with pytest.raises(TypeError, match=f"{component.title()} function must accept"):
            extension.register(function)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_accepts_explicit_none_write():
    extension = Extension(name=extension_name("nonehooks"), types=(object,))
    try:
        extension.register(None, component=Component.write)

        assert extension.write(module=object(), prediction=object(), datatype=None) is None
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_accepts_explicit_none_output():
    extension = Extension(name=extension_name("noneoutput"), types=(object,))
    try:
        extension.register(None, component=Component.output)

        assert extension.output(module=object(), address=object()) is None
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_output_requires_module_and_address_parameters():
    extension = Extension(name=extension_name("badoutput"), types=(object,))

    def output(module: object) -> pa.StructType:
        return pa.struct([])

    try:
        with pytest.raises(TypeError, match="Output function must accept"):
            extension.register(output)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_write_accepts_a_return_annotation():
    extension = Extension(name=extension_name("typedwrite"), types=(object,))

    def write(module: object, prediction: object, datatype: pa.StructType | None) -> None:
        return None

    try:
        assert extension.register(write) is write
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_write_requires_declared_datatype_parameter():
    extension = Extension(name=extension_name("legacywrite"), types=(object,))

    def write(module: object, prediction: object) -> None:
        return None

    try:
        with pytest.raises(TypeError, match="Write function must accept"):
            extension.register(write)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_registers_multiple_callback_factories():
    class FirstCallback(Callback):
        pass

    class SecondCallback(Callback):
        pass

    extension = Extension(name=extension_name("callbacks"), types=(object,))
    try:
        result = extension.callback(FirstCallback, SecondCallback)

        assert result == (FirstCallback, SecondCallback)
        assert extension.callback_factories == [FirstCallback, SecondCallback]
        assert [type(callback) for callback in extension.callbacks] == [FirstCallback, SecondCallback]
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_rejects_invalid_callback_factory_without_registering():
    class InvalidCallback:
        pass

    extension = Extension(name=extension_name("badcallbacks"), types=(object,))
    try:
        with pytest.raises(TypeError, match="must produce a Lightning Callback"):
            extension.callback(InvalidCallback)

        assert extension.callback_factories == []
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_request_must_implement_leaf_contract():
    class Request(Node):
        type: str = "invalid_request"

    extension = Extension(name=extension_name("badrequest"), types=(object,))
    try:
        with pytest.raises(TypeError, match="subclass of RequestBase"):
            extension.register(Request)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_rejects_legacy_tensorfield_new_signature():
    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, values, address, schema, strata):
            return object()

    extension = Extension(name=extension_name("legacynew"), types=(object,))
    try:
        with pytest.raises(TypeError, match="TensorField.new must accept these parameters"):
            extension.register(TensorField)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_rejects_embedder_with_missing_address_param():
    class Embedder(EmbedderBase):
        def __init__(self, schema: object):
            super().__init__(schema=schema, address=None)

    extension = Extension(name=extension_name("badembedder"), types=(object,))
    try:
        with pytest.raises(TypeError, match="must accept 'schema' and 'address'"):
            extension.register(Embedder)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_getattr_rejects_invalid_component_key():
    extension = Extension(name=extension_name("badattr"), types=(object,))
    try:
        with pytest.raises(ValueError, match="is not a valid Component enum value"):
            extension.__getattr__("not_real")
    finally:
        TENSORFIELDS.pop(extension.name, None)
